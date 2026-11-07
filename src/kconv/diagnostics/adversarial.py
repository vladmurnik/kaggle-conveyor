"""Train-vs-test adversarial validation.

The target is never used. A classifier is trained to distinguish train rows
from test rows using only the feature columns; a high OOF AUC means that the
two samples are distinguishable (possible shift or leakage), so ordinary CV
may be optimistic relative to the leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_MAX_CATEGORIES = 50


def _encode_column(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Encode one feature without using the target or the test labels."""
    combined = pd.concat([train.reset_index(drop=True), test.reset_index(drop=True)], ignore_index=True)
    if pd.api.types.is_datetime64_any_dtype(combined):
        try:
            values = pd.to_datetime(combined, errors="coerce").astype("int64").to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError):
            values = np.zeros(len(combined), dtype=float)
    elif pd.api.types.is_numeric_dtype(combined):
        numeric = pd.to_numeric(combined, errors="coerce")
        median = numeric.median()
        values = numeric.fillna(median if pd.notna(median) else 0.0).to_numpy(dtype=float)
    else:
        text = combined.astype("string").fillna("__NA__")
        codes, uniques = pd.factorize(text, sort=True)
        values = codes.astype(float)
        # Rare categories add noise and can make a domain classifier memorize
        # identifiers. Keep the most frequent categories only.
        if len(uniques) > _MAX_CATEGORIES:
            valid = codes[codes >= 0]
            counts = np.bincount(valid, minlength=len(uniques))
            keep = set(np.argsort(counts)[-_MAX_CATEGORIES:].tolist())
            values = np.array([code if code in keep else -1.0 for code in codes], dtype=float)
    return values.reshape(-1)


def _frame_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common = [
        str(column)
        for column in feature_columns
        if column in train.columns and column in test.columns
    ]
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    kept: list[str] = []
    n_train = len(train)
    for column in common:
        values = _encode_column(train[column], test[column])
        if len(values) != n_train + len(test):
            continue
        train_values, test_values = values[:n_train], values[n_train:]
        if np.nanstd(train_values) <= 1e-12 and np.nanstd(test_values) <= 1e-12:
            continue
        train_parts.append(train_values)
        test_parts.append(test_values)
        kept.append(column)
    if not train_parts:
        return np.empty((n_train, 0)), np.empty((len(test), 0)), []
    return np.column_stack(train_parts), np.column_stack(test_parts), kept


def run_adversarial(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    folds: int = 3,
    threshold: float = 0.75,
    seed: int = 42,
) -> dict:
    """Return an adversarial-validation report for ``train`` and ``test``.

    The function is intentionally best-effort: missing test data, too few rows
    or no usable features produce ``status: skipped`` rather than failing the
    pipeline. The competition target must not be in ``feature_columns``.
    """
    report: dict = {
        "enabled": True,
        "status": "skipped",
        "folds": 0,
        "threshold": float(threshold),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "features": 0,
        "feature_names": [],
        "auc": None,
        "warning": False,
        "message": "",
    }
    if train.empty or test.empty:
        report["message"] = "Adversarial validation skipped: train or test table is empty."
        return report

    x_train, x_test, feature_names = _frame_matrix(train, test, feature_columns)
    if not feature_names:
        report["message"] = "Adversarial validation skipped: no usable common feature columns."
        return report
    domain = np.concatenate([np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)])
    n_splits = min(folds, len(train), len(test))
    if n_splits < 2 or len(domain) < 20:
        report["message"] = "Adversarial validation skipped: too few rows for stratified folds."
        return report

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(domain), dtype=float)
    x_all = np.vstack([x_train, x_test])
    for train_index, valid_index in splitter.split(x_all, domain):
        model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
            early_stopping=False,
        )
        model.fit(x_all[train_index], domain[train_index])
        oof[valid_index] = model.predict_proba(x_all[valid_index])[:, 1]

    auc = float(roc_auc_score(domain, oof))
    warning = auc >= threshold
    report.update(
        {
            "status": "ok",
            "folds": int(n_splits),
            "features": int(x_all.shape[1]),
            "feature_names": feature_names,
            "auc": auc,
            "warning": bool(warning),
            "message": (
                f"Adversarial validation AUC {auc:.3f} >= {threshold:.2f}: "
                "train/test distributions differ; CV may be optimistic."
                if warning
                else f"Adversarial validation AUC {auc:.3f} < {threshold:.2f}: "
                "no strong train/test shift detected."
            ),
        }
    )
    return report
