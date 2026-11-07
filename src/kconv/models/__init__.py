"""Model registry and adapter factory."""

from __future__ import annotations

from kconv.contracts import ModelAdapter
from kconv.models.catalog import SKLEARN_SPECS

# Importing the adapter modules registers their factories.
from kconv.models.catboost import CatBoostAdapter
from kconv.models.lgbm import LGBMAdapter
from kconv.models.sklearn_generic import SklearnAdapter
from kconv.models.sklearn_models import HistGBAdapter, LogRegAdapter, RidgeAdapter
from kconv.models.tabpfn import TabPFNAdapter
from kconv.models.xgb import XGBAdapter
from kconv.registry import get_model, register_model

for _name, _cls in {
    "lgbm": LGBMAdapter,
    "xgb": XGBAdapter,
    "catboost": CatBoostAdapter,
    "histgb": HistGBAdapter,
    "logreg": LogRegAdapter,
    "ridge": RidgeAdapter,
    "tabpfn": TabPFNAdapter,
}.items():
    register_model(_name)(lambda cls=_cls: cls())

for _name, _spec in SKLEARN_SPECS.items():
    register_model(_name)(lambda name=_name, spec=_spec: SklearnAdapter(name, spec))


def create_model(name: str) -> ModelAdapter:
    factory = get_model(name)
    return factory()


def model_available(name: str) -> bool:
    """Return whether a registered adapter's optional runtime is installed."""
    adapter = create_model(name)
    checker = getattr(adapter, "available", None)
    return bool(checker()) if callable(checker) else True


__all__ = [
    "CatBoostAdapter",
    "HistGBAdapter",
    "LGBMAdapter",
    "LogRegAdapter",
    "RidgeAdapter",
    "SklearnAdapter",
    "TabPFNAdapter",
    "XGBAdapter",
    "create_model",
    "model_available",
]
