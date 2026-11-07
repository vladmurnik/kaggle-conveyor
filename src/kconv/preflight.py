"""Preflight checks used by ``kconv doctor``.

The checks are deliberately best-effort: a missing optional model runtime is a
warning, while a missing core package or an invalid config is an error.  The
same payload is printed by the CLI and returned as JSON for CI/web clients.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from kconv.config import Config
from kconv.errors import KconvError
from kconv.io import safe_read_sample
from kconv.models import create_model, model_available
from kconv.models.validation import validate_model_params
from kconv.registry import model_names


def _check(
    name: str,
    status: str,
    message: str,
    hint: str | None = None,
) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "hint": hint}


def _package_check(dist: str, *, required: bool) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return _check(
            f"package:{dist}",
            "error" if required else "warn",
            f"{dist} is not installed",
            "Run 'uv sync' (or install the optional extra)." if required else "Optional; install only if you use it.",
        )
    return _check(f"package:{dist}", "ok", f"{dist} {version}")


def _memory_gb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024**3), 1)
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1)
    except (AttributeError, ValueError, OSError):
        return None


def _gpu_check() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return _check("gpu", "warn", "torch is not installed; CUDA unavailable", "Install a CUDA torch build for GPU models.")
    try:
        if not torch.cuda.is_available():
            return _check("gpu", "warn", "torch installed, but CUDA is not available", "Use CPU models or install a CUDA torch build.")
        count = int(torch.cuda.device_count())
        names: list[str] = []
        for index in range(count):
            name = str(torch.cuda.get_device_name(index))
            try:
                vram = torch.cuda.get_device_properties(index).total_memory / (1024**3)
                names.append(f"{name} ({vram:.1f} GiB VRAM)")
            except Exception:
                names.append(name)
        return _check("gpu", "ok", f"CUDA ready: {count} device(s): {', '.join(names)}")
    except Exception as exc:  # pragma: no cover - driver-specific
        return _check("gpu", "warn", f"CUDA probe failed: {exc}", "Check the NVIDIA driver and torch build.")


def _config_checks(config: Config, project_dir: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    train_value = config.data.train
    if not train_value:
        checks.append(_check("schema", "error", "config.data.train is empty", "Set data.train to a CSV/parquet file."))
        return checks
    train_path = Path(train_value)
    if not train_path.is_absolute():
        train_path = project_dir / train_path
    if not train_path.exists():
        checks.append(_check("schema", "error", f"Training file not found: {train_path}", "Fix data.train or generate the project again."))
    else:
        try:
            sample = safe_read_sample(train_path, nrows=1000)
            missing = [column for column in (config.data.target, config.data.id) if column and column not in sample.columns]
            if missing:
                checks.append(_check("schema", "error", f"Missing columns in train: {', '.join(missing)}", "Check data.target/data.id."))
            else:
                checks.append(_check("schema", "ok", f"Train file readable: {len(sample)} sampled rows, {len(sample.columns)} columns"))
        except KconvError as exc:
            checks.append(_check("schema", "error", exc.message, exc.hint))
    for label, value in (("test", config.data.test), ("sample_submission", config.data.sample_submission)):
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = project_dir / path
        if not path.exists():
            checks.append(_check(f"data:{label}", "warn", f"{label} file not found: {path}", "Submit may be unavailable."))
        else:
            checks.append(_check(f"data:{label}", "ok", f"{label} file found"))
    for model in config.models:
        if not model.enabled:
            continue
        try:
            adapter = create_model(model.name)
            if config.task != "auto" and config.task not in adapter.task_types:
                checks.append(_check(f"model:{model.name}", "error", f"does not support task '{config.task}'", "Choose a compatible model."))
            elif not model_available(model.name):
                checks.append(_check(f"model:{model.name}", "error", "optional runtime is not installed", "Install the model extra or select another model."))
            else:
                validate_model_params(model.name, model.params)
                checks.append(_check(f"model:{model.name}", "ok", "available; params valid"))
        except (KconvError, Exception) as exc:  # noqa: BLE001 - report every model independently
            message = getattr(exc, "message", str(exc))
            hint = getattr(exc, "hint", None)
            checks.append(_check(f"model:{model.name}", "error", message, hint))
    return checks


def run_preflight(
    *,
    config: Config | None = None,
    project_dir: str | Path | None = None,
    check_models: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable environment and config report."""
    project = Path(project_dir or Path.cwd()).resolve()
    checks: list[dict[str, Any]] = []
    for dist, required in (
        ("kconv", True),
        ("pandas", True),
        ("numpy", True),
        ("scikit-learn", True),
        ("pyarrow", True),
        ("lightgbm", True),
        ("xgboost", False),
        ("catboost", False),
    ):
        checks.append(_package_check(dist, required=required))
    checks.append(_check("python", "ok", f"{platform.python_version()} ({platform.platform()})"))
    memory = _memory_gb()
    checks.append(
        _check(
            "memory",
            "warn" if memory is not None and memory < 4 else "ok",
            f"{memory} GiB RAM" if memory is not None else "RAM size unavailable",
            "Use a smaller runtime profile or fewer parallel models." if memory is not None and memory < 4 else None,
        )
    )
    checks.append(_gpu_check())
    try:
        usage = shutil.disk_usage(project)
        free_gb = usage.free / (1024**3)
        checks.append(
            _check(
                "disk",
                "warn" if free_gb < 1 else "ok",
                f"{free_gb:.1f} GiB free at {project}",
                "Free disk space before training." if free_gb < 1 else None,
            )
        )
    except OSError as exc:
        checks.append(_check("disk", "warn", f"Could not check disk space: {exc}"))
    if check_models:
        registered = model_names()
        checks.append(_check("models", "ok" if registered else "error", f"{len(registered)} registered model adapters"))
    if config is not None:
        checks.extend(_config_checks(config, project))
    counts = {"ok": 0, "warn": 0, "error": 0}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    return {
        "ok": counts.get("error", 0) == 0,
        "status": "ok" if counts.get("error", 0) == 0 else "error",
        "project_dir": str(project),
        "python": sys.version.split()[0],
        "checks": checks,
        "summary": counts,
    }
