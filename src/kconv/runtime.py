"""Simple 1-5 runtime profiles for CPU/GPU and memory budgets.

The profiles are deliberately coarse.  They choose an existing strategy, cap the
number of models, and set thread/device parameters for the built-in boosted
trees.  They do not pretend that every scikit-learn estimator has a GPU path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from kconv.errors import KconvError
from kconv.models import create_model, model_available
from kconv.registry import model_names

Device = Literal["auto", "cpu", "gpu", "cuda"]
Strategy = Literal["fast", "standard", "deep"]
_DEVICES = {"auto", "cpu", "gpu", "cuda"}


@dataclass(frozen=True)
class GpuCapabilities:
    """What the installed third-party libraries can actually use.

    Detection is best effort and never imports training data.  ``auto`` uses it
    to decide whether profiles 4-5 run on GPU or quietly fall back to CPU;
    an explicit ``--device cuda`` is rejected when no usable GPU is found so
    the user does not get a half-broken run.
    """

    gpu_available: bool
    lgbm_gpu: bool
    xgb_cuda: bool
    catboost_cuda: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeProfile:
    level: int
    name: str
    ram_gb: int
    vram_gb: int
    cpu_threads: int
    strategy: Strategy
    model_limit: int
    seeds: int
    hpo: bool
    device: Device
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILES: dict[int, RuntimeProfile] = {
    1: RuntimeProfile(
        level=1,
        name="micro",
        ram_gb=4,
        vram_gb=0,
        cpu_threads=1,
        strategy="fast",
        model_limit=2,
        seeds=1,
        hpo=False,
        device="cpu",
        description="One CPU thread, 2 models, no HPO. For tiny datasets or a laptop on battery.",
    ),
    2: RuntimeProfile(
        level=2,
        name="low",
        ram_gb=8,
        vram_gb=0,
        cpu_threads=2,
        strategy="fast",
        model_limit=3,
        seeds=1,
        hpo=False,
        device="cpu",
        description="Low-RAM CPU profile; useful for quick iteration on small/medium tables.",
    ),
    3: RuntimeProfile(
        level=3,
        name="balanced",
        ram_gb=16,
        vram_gb=0,
        cpu_threads=4,
        strategy="standard",
        model_limit=4,
        seeds=2,
        hpo=False,
        device="cpu",
        description="Default CPU profile: 4 models, 5 folds, 2 seeds, no HPO.",
    ),
    4: RuntimeProfile(
        level=4,
        name="gpu",
        ram_gb=32,
        vram_gb=12,
        cpu_threads=8,
        strategy="standard",
        model_limit=6,
        seeds=2,
        hpo=False,
        device="auto",
        description="GPU-capable profile for boosted trees; auto picks CUDA when available, 12 GB VRAM, 6 models.",
    ),
    5: RuntimeProfile(
        level=5,
        name="max",
        ram_gb=64,
        vram_gb=24,
        cpu_threads=0,
        strategy="deep",
        model_limit=10_000,
        seeds=3,
        hpo=True,
        device="auto",
        description="Maximum profile: deep strategy, HPO, all task-compatible models; auto device.",
    ),
}


def list_profiles() -> list[RuntimeProfile]:
    """Return the five built-in profiles in ascending cost order."""
    return [_PROFILES[level] for level in sorted(_PROFILES)]


def get_profile(level: int, *, device: str | None = None) -> RuntimeProfile:
    """Return a profile and optionally override its device.

    ``device='auto'`` keeps the profile default.  Profiles 1-3 are CPU-only;
    asking for GPU/CUDA on them is an explicit error instead of a silent
    fallback that would mislead the user about the actual hardware used.
    """
    if level not in _PROFILES:
        raise KconvError("Runtime profile must be an integer from 1 to 5.", "Use --profile 1..5.")
    if device is not None:
        requested = device.casefold()
        if requested not in _DEVICES:
            raise KconvError("Unknown device.", "Use one of: auto, cpu, gpu, cuda.")
        if requested in {"gpu", "cuda"} and level <= 3:
            raise KconvError(
                f"Profile {level} is CPU-only.",
                "Use profile 4 or 5 for GPU/CUDA, or choose --device cpu.",
            )
        if requested != "auto":
            return replace(_PROFILES[level], device=requested)  # type: ignore[arg-type]
    return _PROFILES[level]


@lru_cache(maxsize=1)
def _torch_cuda_available() -> bool:
    """Check torch's CUDA build without making torch a hard dependency."""
    try:
        from importlib.util import find_spec

        if find_spec("torch") is None:
            return False
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - broken/partial optional install
        return False


def select_models(profile: RuntimeProfile, task: str) -> list[str]:
    """Select at most ``profile.model_limit`` task-compatible built-ins.

    The order starts with the inexpensive boosted/linear workhorses and only
    adds more diverse estimators for higher profiles.  Profile 5 returns every
    compatible installed model; optional adapters without their runtime are
    skipped so a base install still works.
    """
    if task not in {"binary", "multiclass", "regression"}:
        raise KconvError(f"Unknown task '{task}'.", "Use binary, multiclass or regression.")
    available = {name for name in model_names() if model_available(name)}
    priority = [
        "lgbm",
        "histgb",
        "xgb",
        "catboost",
        "random_forest_classifier",
        "random_forest_regressor",
        "extra_trees_classifier",
        "extra_trees_regressor",
        "gradient_boosting_classifier",
        "gradient_boosting_regressor",
        "logreg",
        "ridge",
        "linear_regression",
        "elasticnet",
        "svc",
        "svr",
        "knn_classifier",
        "knn_regressor",
    ]
    selected = [
        name
        for name in priority
        if name in available and task in create_model(name).task_types
    ]
    if profile.level >= 5:
        selected = [
            name
            for name in model_names()
            if name in available and task in create_model(name).task_types
        ]
    else:
        selected = selected[: profile.model_limit]
    return selected


def _lightgbm_gpu_build() -> bool:
    """Best-effort check whether the installed LightGBM has a GPU learner.

    LightGBM ships separate CPU/GPU binaries.  We look for the known CPU-only
    error string in the shared library; absence of the marker plus a resolvable
    library is treated as a GPU-capable build.  This is detection, not proof,
    which is why ``auto`` also cross-checks with other libraries and why an
    explicit force can still fail.
    """
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return False
    candidates: list[str] = []
    try:
        from lightgbm.libpath import find_lib_path

        candidates = [str(path) for path in find_lib_path()]
    except Exception:
        candidates = []
    if not candidates:
        lib = getattr(getattr(lightgbm, "basic", None), "_LIB", None)
        name = getattr(lib, "_name", None)
        if name:
            candidates = [str(name)]
    seen = False
    for candidate in candidates:
        try:
            data = Path(candidate).read_bytes()
        except OSError:
            continue
        seen = True
        if b"GPU Tree Learner was not enabled" in data:
            return False
    return seen


@lru_cache(maxsize=1)
def gpu_capabilities() -> GpuCapabilities:
    """Detect CUDA/GPU support of the installed boosted-tree libraries.

    Cached because imports and binary reads are not free, and hardware does not
    change during a run.  Tests monkeypatch this function with a fake
    capability object to stay independent of the CI machine.
    """
    xgb_cuda = False
    catboost_cuda = False
    details: list[str] = []
    try:
        import xgboost as xgb

        info = xgb.build_info()
        xgb_cuda = bool(info.get("USE_CUDA", False))
        details.append(f"xgboost {xgb.__version__} USE_CUDA={xgb_cuda}")
    except Exception:
        details.append("xgboost unavailable")
    try:
        from catboost.utils import get_gpu_device_count

        catboost_cuda = int(get_gpu_device_count()) > 0
        details.append(f"catboost gpu_devices={int(get_gpu_device_count())}")
    except Exception:
        details.append("catboost GPU probe unavailable")
    nvidia_smi = _nvidia_smi_available()
    lgbm_gpu = _lightgbm_gpu_build()
    details.append(f"nvidia-smi={nvidia_smi}")
    details.append(f"lightgbm gpu_build={lgbm_gpu}")
    return GpuCapabilities(
        # A CUDA-capable build is not the same as a present device, so the
        # hardware check uses the actual device count / nvidia-smi.
        gpu_available=catboost_cuda or nvidia_smi,
        lgbm_gpu=lgbm_gpu,
        xgb_cuda=xgb_cuda,
        catboost_cuda=catboost_cuda,
        details="; ".join(details),
    )


def _nvidia_smi_available() -> bool:
    """Cheap hardware probe used by ``auto``; never raises."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def resolve_device(profile: RuntimeProfile) -> Literal["cpu", "cuda"]:
    """Resolve ``auto`` against the detected hardware.

    ``auto`` falls back to CPU when nothing usable is detected.  An explicit
    ``gpu``/``cuda`` request raises instead of silently training on CPU.
    """
    if profile.device == "cpu":
        return "cpu"
    caps = gpu_capabilities()
    if profile.device == "auto":
        return "cuda" if caps.gpu_available else "cpu"
    if not caps.gpu_available:
        raise KconvError(
            "No usable GPU/CUDA device detected for this profile.",
            "Use --device cpu or --profile 1..3, or install a CUDA/GPU-capable build.",
        )
    return "cuda"


def device_params(profile: RuntimeProfile) -> dict[str, dict[str, Any]]:
    """Return adapter params for CPU threads / GPU acceleration.

    Only the built-in boosted trees have an explicit device path.  We do not
    add fake GPU parameters to scikit-learn adapters.  CPU thread limits are
    applied to every boosted-tree model; GPU parameters are then layered on
    only for libraries whose installed build or hardware supports them, so one
    CPU-only wheel cannot break an otherwise valid run.

    TabPFN is optional and gets a ``device`` parameter.  Profiles 1-3 force CPU;
    GPU profiles use CUDA only when the installed torch build sees a device.
    An explicit ``--device cuda`` still reaches the adapter so it can raise an
    actionable error instead of silently falling back.
    """
    threads = profile.cpu_threads or (os.cpu_count() or 1)
    # CPU thread caps apply even when another model in the same run uses a GPU.
    params: dict[str, dict[str, Any]] = {
        "lgbm": {"num_threads": threads},
        "xgb": {"n_jobs": threads},
        "catboost": {"thread_count": threads},
    }
    tabpfn_device = "cpu"
    resolved = resolve_device(profile)
    if resolved == "cuda":
        # auto falls back to CPU when torch has no CUDA build; explicit
        # gpu/cuda reaches the adapter so it can raise an actionable error.
        if profile.device == "auto":
            tabpfn_device = "cuda" if _torch_cuda_available() else "cpu"
        else:
            tabpfn_device = "cuda"
    params["tabpfn"] = {"device": tabpfn_device}
    if resolved != "cuda":
        return params
    caps = gpu_capabilities()
    if caps.lgbm_gpu:
        params["lgbm"] = {"device": "cuda", "num_threads": threads}
    if caps.xgb_cuda:
        params["xgb"] = {"device": "cuda", "tree_method": "hist", "n_jobs": threads}
    if caps.catboost_cuda:
        params["catboost"] = {"task_type": "GPU", "devices": "0", "thread_count": threads}
    return params


__all__ = [
    "Device",
    "GpuCapabilities",
    "RuntimeProfile",
    "device_params",
    "get_profile",
    "gpu_capabilities",
    "list_profiles",
    "resolve_device",
    "select_models",
]
