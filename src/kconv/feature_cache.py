"""Content-addressed cache for fitted feature builders and fold matrices.

Feature engineering is often the most expensive part of a repeated run.  The
cache key covers the resolved config, raw data hash, kconv version, fold split,
feature family and target values.  Cache entries are verified before use; a
corrupt or partially written entry is discarded and rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from kconv.atomic import dump_joblib_atomic, write_parquet_atomic
from kconv.context import environment_versions, read_json, write_json
from kconv.features.pipeline import FeatureBuilder

CACHE_VERSION = 2


def _digest(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _array_digest(values: Any) -> str | None:
    if values is None:
        return None
    try:
        hashed = pd.util.hash_pandas_object(pd.Series(values), index=False).to_numpy()
        return hashlib.sha256(hashed.tobytes()).hexdigest()
    except Exception:
        return _digest(str(values))


def _index_digest(frame: pd.DataFrame) -> str:
    try:
        hashed = pd.util.hash_pandas_object(frame.index, index=False).to_numpy()
        return hashlib.sha256(hashed.tobytes()).hexdigest()
    except Exception:
        return _digest(f"{len(frame)}:{list(frame.index[:10])}:{list(frame.index[-10:])}")


def cache_key(
    ctx: Any,
    family: str,
    fold_index: int,
    X_train: pd.DataFrame,
    y_train: Any,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame | None,
    *,
    fold_train_idx: Any | None = None,
    fold_valid_idx: Any | None = None,
) -> str:
    config = getattr(ctx, "config", None)
    runtime = getattr(config, "runtime", None)
    payload = {
        "cache_version": CACHE_VERSION,
        "config_hash": _digest(config.model_dump(mode="json", exclude_none=True) if config else ""),
        "data_hash": getattr(ctx, "data_hash", None),
        "kconv_version": environment_versions().get("kconv_version"),
        "family": str(family),
        "fold": int(fold_index),
        "train_index": _index_digest(X_train),
        "valid_index": _index_digest(X_valid),
        "test_index": _index_digest(X_test) if X_test is not None else None,
        "fold_train": _array_digest(fold_train_idx),
        "fold_valid": _array_digest(fold_valid_idx),
        "train_shape": list(X_train.shape),
        "valid_shape": list(X_valid.shape),
        "test_shape": list(X_test.shape) if X_test is not None else None,
        "target": _digest(
            pd.util.hash_pandas_object(pd.Series(y_train), index=False).to_numpy().tobytes()
        ),
        "feature_cache": getattr(runtime, "feature_cache", True),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:32]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksums(cache_dir: Path) -> dict[str, str]:
    names = ("builder.joblib", "train.parquet", "valid.parquet", "test.parquet")
    return {
        name: _file_sha256(cache_dir / name)
        for name in names
        if (cache_dir / name).is_file()
    }


def _verify(cache_dir: Path) -> bool:
    meta = read_json(cache_dir / "meta.json", default={})
    checksums = read_json(cache_dir / "checksums.json", default={})
    if not isinstance(meta, dict) or not isinstance(checksums, dict) or not checksums:
        return False
    if meta.get("cache_version") != CACHE_VERSION:
        return False
    for name, expected in checksums.items():
        path = cache_dir / str(name)
        if not path.is_file() or _file_sha256(path) != expected:
            return False
    return (cache_dir / "builder.joblib").is_file() and (cache_dir / "train.parquet").is_file()


def _build(ctx: Any, spec: Any, family: str, X_train, y_train, X_valid, X_test):
    builder = FeatureBuilder(ctx.config, spec, family=family)
    Xt = builder.fit_transform(X_train, y_train)
    Xv = builder.transform(X_valid)
    X_test_out = builder.transform(X_test) if X_test is not None else None
    return builder, Xt, Xv, X_test_out


def load_or_build_features(
    ctx: Any,
    spec: Any,
    family: str,
    fold_index: int,
    X_train: pd.DataFrame,
    y_train: Any,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame | None,
    *,
    fold_train_idx: Any | None = None,
    fold_valid_idx: Any | None = None,
) -> tuple[FeatureBuilder, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, bool]:
    """Return ``(builder, train, valid, test, cache_hit)``.

    Cache failures are non-fatal: the function falls back to a normal fit and
    leaves the run deterministic.
    """
    runtime = getattr(getattr(ctx, "config", None), "runtime", None)
    if runtime is not None and not getattr(runtime, "feature_cache", True):
        builder, Xt, Xv, Xtst = _build(ctx, spec, family, X_train, y_train, X_valid, X_test)
        return builder, Xt, Xv, Xtst, False

    key = cache_key(
        ctx,
        family,
        fold_index,
        X_train,
        y_train,
        X_valid,
        X_test,
        fold_train_idx=fold_train_idx,
        fold_valid_idx=fold_valid_idx,
    )
    cache_root = Path(getattr(ctx, "project_dir", Path.cwd())) / ".cache" / "features"
    cache_dir = cache_root / key
    if _verify(cache_dir):
        try:
            builder = joblib.load(cache_dir / "builder.joblib")
            Xt = pd.read_parquet(cache_dir / "train.parquet")
            Xv = pd.read_parquet(cache_dir / "valid.parquet")
            Xtst = pd.read_parquet(cache_dir / "test.parquet") if X_test is not None else None
            if len(Xt) == len(X_train) and len(Xv) == len(X_valid):
                return builder, Xt, Xv, Xtst, True
        except Exception:
            pass
        shutil.rmtree(cache_dir, ignore_errors=True)

    builder, Xt, Xv, Xtst = _build(ctx, spec, family, X_train, y_train, X_valid, X_test)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        dump_joblib_atomic(builder, cache_dir / "builder.joblib", compress=3)
        write_parquet_atomic(Xt, cache_dir / "train.parquet")
        write_parquet_atomic(Xv, cache_dir / "valid.parquet")
        if Xtst is not None:
            write_parquet_atomic(Xtst, cache_dir / "test.parquet")
        meta = {
            "cache_version": CACHE_VERSION,
            "key": key,
            "family": family,
            "fold": int(fold_index),
            "train_rows": len(X_train),
            "valid_rows": len(X_valid),
            "test_rows": len(X_test) if X_test is not None else None,
            "created_at": datetime.now(UTC).isoformat(),
        }
        write_json(cache_dir / "meta.json", meta)
        write_json(cache_dir / "checksums.json", _checksums(cache_dir))
    except Exception as exc:  # cache is an optimization, never a failure
        shutil.rmtree(cache_dir, ignore_errors=True)
        try:
            ctx.warnings.append(f"Feature cache write skipped: {exc}")
        except Exception:
            pass
    return builder, Xt, Xv, Xtst, False
