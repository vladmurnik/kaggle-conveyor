"""Shared data contracts used by stages, features, models and tests.

The goal is to keep the boundary between modules explicit and boring:
plain numpy arrays for predictions, pandas for tables, dataclasses for
artifacts.  No stage imports another stage's private helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

TaskType = Literal["binary", "multiclass", "regression"]
ModelFamily = Literal["tree", "linear", "foundation"]


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold. Indices are positional row indices."""

    index: int
    train_idx: np.ndarray
    valid_idx: np.ndarray


@dataclass
class DatasetSpec:
    """Everything downstream needs to know about the input dataset."""

    train: Path
    test: Path | None
    sample_submission: Path | None
    target: str
    id_col: str | None
    task: TaskType
    metric: str
    feature_columns: list[str]
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    ignore_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _path(value: Path | None) -> str | None:
            return str(value).replace("\\", "/") if value is not None else None

        return {
            "train": _path(self.train),
            "test": _path(self.test),
            "sample_submission": _path(self.sample_submission),
            "target": self.target,
            "id_col": self.id_col,
            "task": self.task,
            "metric": self.metric,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "text_columns": self.text_columns,
            "ignore_columns": self.ignore_columns,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetSpec:
        return cls(
            train=Path(raw["train"]),
            test=Path(raw["test"]) if raw.get("test") else None,
            sample_submission=Path(raw["sample_submission"]) if raw.get("sample_submission") else None,
            target=raw["target"],
            id_col=raw.get("id_col"),
            task=raw["task"],
            metric=raw["metric"],
            feature_columns=list(raw["feature_columns"]),
            numeric_columns=list(raw.get("numeric_columns", [])),
            categorical_columns=list(raw.get("categorical_columns", [])),
            datetime_columns=list(raw.get("datetime_columns", [])),
            text_columns=list(raw.get("text_columns", [])),
            ignore_columns=list(raw.get("ignore_columns", [])),
            warnings=list(raw.get("warnings", [])),
        )


@dataclass
class ModelResult:
    """Artifacts produced by one model across all folds/seeds."""

    name: str
    task: TaskType
    metric: str
    oof: np.ndarray
    test_pred: np.ndarray
    fold_scores: list[float]
    oof_score: float
    classes: list[Any] | None = None
    importance: pd.Series | None = None
    params: dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class BlendResult:
    """Artifacts produced by the blending stage."""

    method: str
    models: list[str]
    weights: dict[str, float]
    oof: np.ndarray
    test_pred: np.ndarray
    oof_score: float
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class ModelAdapter(Protocol):
    """Every model adapter must implement this small surface."""

    name: str
    task_types: set[str]
    family: ModelFamily

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame | None = None,
        y_va: np.ndarray | None = None,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        early_stopping_rounds: int = 100,
    ) -> None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def importance(self) -> pd.Series | None: ...
