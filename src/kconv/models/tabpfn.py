"""Optional TabPFN v2 adapter for small/medium tabular datasets.

TabPFN is a pretrained tabular foundation model.  It is intentionally an
optional dependency: the base install stays small, and importing ``kconv``
never imports torch.  The defaults target a 3.5 GB VRAM / 7 GB RAM laptop:
TabPFN v2 weights are ~28 MB, preprocessing uses ``fit_preprocessors`` (low
memory), and the adapter rejects datasets above the v2 in-context limits.

The v2 weights are distributed under the Prior Labs license (Apache-2.0 with
attribution).  See ``NOTICE.md``.  Newer TabPFN 3.x weights are gated and
non-commercial; pass ``model_version`` explicitly only if you accepted their
license and have the required credentials.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from typing import Any

import numpy as np
import pandas as pd

from kconv.errors import ModelError
from kconv.models.base import BaseAdapter

# v2 itself accepts 10k train rows, but 10k x 100 measured ~2.82 GB of reserved
# VRAM on a 4 GB laptop (plus ~0.8 GB desktop), which is too close to the
# 3.5 GB budget.  8k keeps the complete process under the budget with headroom.
SAFE_MAX_ROWS = 8_000
V2_MAX_ROWS = 10_000
V2_MAX_FEATURES = 500
V2_MAX_CLASSES = 10


class TabPFNAdapter(BaseAdapter):
    """Lazy optional adapter around ``tabpfn.TabPFNClassifier``/Regressor."""

    name = "tabpfn"
    task_types = {"binary", "multiclass", "regression"}
    family = "foundation"
    optional = True

    def __init__(self) -> None:
        super().__init__()
        self.device_: str | None = None
        self.model_version_: str | None = None
        self.peak_vram_bytes_ = 0
        self.last_train_rows_ = 0
        self.last_n_features_ = 0
        self.warnings: list[str] = []

    @classmethod
    def available(cls) -> bool:
        """Return whether the optional runtime is installed, without importing it."""
        return (
            importlib.util.find_spec("tabpfn") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def _release(self) -> None:
        # Clear the cached CUDA allocator even when a fit failed before
        # ``self.model`` was assigned (for example, a GPU OOM during fit).
        # Otherwise the next model inherits the reserved blocks and can OOM
        # for no reason.
        self.model = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - defensive cleanup only
                pass

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise ModelError(
                f"Model '{self.name}' expects a pandas DataFrame.",
                "Pass the numeric frame produced by FeatureBuilder.",
            )
        values = X.to_numpy(dtype=np.float32, copy=False)
        if values.ndim != 2:
            raise ModelError(f"Model '{self.name}' expects a 2-D matrix.", "Check the feature frame.")
        if not np.isfinite(values).all():
            raise ModelError(
                f"Model '{self.name}' received non-finite features.",
                "Impute NaN/inf before fitting; kconv's FeatureBuilder does this automatically.",
            )
        return values

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        requested = str(requested).casefold()
        if requested in {"gpu", "cuda"}:
            requested = "cuda"
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ModelError(
                "TabPFN was asked to use CUDA, but this torch build has no CUDA device.",
                "Install a CUDA torch wheel or pass params.device='cpu'.",
            )
        return requested

    @staticmethod
    def _cuda_index(torch: Any, device: str) -> int | None:
        if not device.startswith("cuda"):
            return None
        parsed = torch.device(device)
        return int(parsed.index) if parsed.index is not None else int(torch.cuda.current_device())

    def _reset_peak(self, torch: Any, device: str) -> None:
        index = self._cuda_index(torch, device)
        if index is not None:
            torch.cuda.reset_peak_memory_stats(index)

    def _record_peak(self, torch: Any, device: str) -> None:
        index = self._cuda_index(torch, device)
        if index is not None:
            value = int(torch.cuda.max_memory_allocated(index))
            self.peak_vram_bytes_ = max(self.peak_vram_bytes_, value)

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        del X_va, y_va, early_stopping_rounds  # TabPFN is in-context; no eval loop.
        if self.task not in self.task_types:
            raise ModelError(
                f"Model '{self.name}' does not support task '{self.task}'.",
                f"Use one of: {', '.join(sorted(self.task_types))}.",
            )
        self._release()
        supplied = dict(params or {})
        version_name = str(supplied.pop("model_version", "v2"))
        requested_device = str(supplied.pop("device", "auto"))
        max_train_rows = int(supplied.pop("max_train_rows", SAFE_MAX_ROWS))
        max_features = int(supplied.pop("max_features", V2_MAX_FEATURES))
        max_classes = int(supplied.pop("max_classes", V2_MAX_CLASSES))
        n_estimators = supplied.pop("n_estimators", 8)
        fit_mode = str(supplied.pop("fit_mode", "fit_preprocessors"))
        ignore_pretraining_limits = bool(supplied.pop("ignore_pretraining_limits", False))
        model_path = supplied.pop("model_path", None)

        # Check the cheap limits before converting a huge frame to a float32
        # matrix; otherwise a rejected 1M-row input could exhaust RAM first.
        if not isinstance(X_tr, pd.DataFrame):
            raise ModelError(
                f"Model '{self.name}' expects a pandas DataFrame.",
                "Pass the numeric frame produced by FeatureBuilder.",
            )
        if len(X_tr) > max_train_rows:
            raise ModelError(
                f"TabPFN input has {len(X_tr):,} rows; the configured limit is {max_train_rows:,} "
                f"(v2 hard limit {V2_MAX_ROWS:,}).",
                "Use lgbm/xgb/catboost for larger tables, or raise params.max_train_rows explicitly.",
            )
        if X_tr.shape[1] > max_features:
            raise ModelError(
                f"TabPFN input has {X_tr.shape[1]:,} features; the configured limit is {max_features:,}.",
                "Reduce features, use a GBDT model, or raise params.max_features explicitly.",
            )
        classes = np.unique(np.asarray(y_tr))
        if self.task != "regression" and len(classes) > max_classes:
            raise ModelError(
                f"TabPFN v2 supports at most {max_classes} classes; got {len(classes)}.",
                "Use lgbm/xgb/catboost for high-cardinality multiclass targets.",
            )
        values_tr = self._matrix(X_tr)
        self._remember_features(X_tr)

        tabpfn = self._require("tabpfn", "tabpfn")
        torch = self._require("torch", "torch")
        device = self._resolve_device(torch, requested_device)
        self.device_ = device
        self.model_version_ = version_name
        self._reset_peak(torch, device)
        self.last_train_rows_ = int(len(X_tr))
        self.last_n_features_ = int(X_tr.shape[1])
        self.warnings = []

        kwargs: dict[str, Any] = {
            "device": device,
            "n_estimators": n_estimators,
            "fit_mode": fit_mode,
            "ignore_pretraining_limits": ignore_pretraining_limits,
            "show_progress_bar": False,
            "random_state": seed,
            "n_preprocessing_jobs": 1,
            **supplied,
        }
        try:
            if model_path:
                estimator = (
                    tabpfn.TabPFNRegressor(model_path=model_path, **kwargs)
                    if self.task == "regression"
                    else tabpfn.TabPFNClassifier(model_path=model_path, **kwargs)
                )
            else:
                from tabpfn.constants import ModelVersion

                try:
                    version = ModelVersion(version_name)
                except ValueError as exc:
                    allowed = ", ".join(item.value for item in ModelVersion)
                    raise ModelError(
                        f"Unknown TabPFN model version '{version_name}'.",
                        f"Use one of: {allowed}.",
                    ) from exc
                factory = tabpfn.TabPFNRegressor if self.task == "regression" else tabpfn.TabPFNClassifier
                estimator = factory.create_default_for_version(version, **kwargs)
            estimator.fit(values_tr, np.asarray(y_tr))
        except ModelError:
            self._release()
            raise
        except Exception as exc:  # pragma: no cover - depends on weights/GPU
            self._release()
            hint = (
                "TabPFN v3.5 weights are gated and non-commercial; use model_version='v2' "
                "or accept the license and provide the required token."
                if version_name.startswith(("v3", "v3.5"))
                else "Check CUDA memory, torch/tabpfn versions, and input dtypes."
            )
            raise ModelError(f"TabPFN training failed: {exc}", hint) from exc
        self.model = estimator
        self._record_peak(torch, device)

    def predict(self, X):
        if self.model is None:
            raise ModelError(f"Model '{self.name}' is not fitted.", "Call fit() first.")
        try:
            values = self.model.predict(self._matrix(X))
        except Exception as exc:  # pragma: no cover - depends on weights/GPU
            self._release()
            raise ModelError(f"TabPFN prediction failed: {exc}", "Check input shape and CUDA memory.") from exc
        torch = sys.modules.get("torch")
        if torch is not None:
            self._record_peak(torch, self.device_ or "cpu")
        return np.asarray(values)

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("Regression models do not expose predict_proba")
        if self.model is None:
            raise ModelError(f"Model '{self.name}' is not fitted.", "Call fit() first.")
        try:
            proba = np.asarray(self.model.predict_proba(self._matrix(X)), dtype=float)
        except Exception as exc:  # pragma: no cover - depends on weights/GPU
            self._release()
            raise ModelError(f"TabPFN prediction failed: {exc}", "Check input shape and CUDA memory.") from exc
        torch = sys.modules.get("torch")
        if torch is not None:
            self._record_peak(torch, self.device_ or "cpu")
        if self.task == "binary":
            return proba[:, 1]
        return proba

    def importance(self) -> pd.Series | None:
        # The in-context transformer has no per-feature impurity importance.
        return None
