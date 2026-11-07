"""Curated groups of executable pipelines for the browser queue.

The recipe catalog contains many useful pipelines, but manually picking a
dozen checkboxes is slow and error-prone.  These presets are stable, named
starting points: tiered sets for "first run / fast / balanced / quality" and
specialty sets for common feature families.  They only reference existing
pipeline ids and never change a recipe or its defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PresetKind = Literal["tier", "specialty", "all"]


@dataclass(frozen=True)
class PipelinePreset:
    """A named group of executable ``kind: pipeline`` recipes."""

    id: str
    title: str
    description: str
    kind: PresetKind
    pipelines: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "kind": self.kind,
            "pipelines": list(self.pipelines),
        }


ALL_PIPELINES: tuple[str, ...] = (
    "pipeline/arnabbiswas-staged",
    "pipeline/cdeotte-datetime-cyclical",
    "pipeline/cdeotte-gbdt-blend",
    "pipeline/dptools-cluster-profile",
    "pipeline/dptools-group-profile",
    "pipeline/dptools-row-profile",
    "pipeline/ensemble-amazon-interaction-target",
    "pipeline/ensemble-amazon-numeric-cross",
    "pipeline/ensemble-amazon-stack",
    "pipeline/kaggle-ieee-cluster-anomaly",
    "pipeline/kaggle-ieee-frequency",
    "pipeline/kaggle-ieee-missingness",
    "pipeline/kaggle-ieee-time-profile",
    "pipeline/kaggler-target-frequency",
    "pipeline/kconv-linear-stack",
    "pipeline/kconv-sklearn-diverse",
    "pipeline/kconv-unsupervised-profile",
    "pipeline/knnfeat-class-distance",
    "pipeline/kozodoi-combination-frequency",
    "pipeline/kozodoi-high-cardinality",
    "pipeline/kozodoi-numeric-ratios",
    "pipeline/kozodoi-time-profile",
    "pipeline/minerva-oof-stack",
    "pipeline/nyaggle-cluster-stack",
    "pipeline/nyaggle-oof-ensemble",
    "pipeline/pesnik-clean",
    "pipeline/sergeyshilin-anomaly-contracts",
    "pipeline/sergeyshilin-contracts",
    "pipeline/statsmaster-honest-cv",
)

PIPELINE_PRESETS: tuple[PipelinePreset, ...] = (
    PipelinePreset(
        id="starter",
        title="Быстрый старт",
        description="Три простых pipeline для первого прогона и быстрой проверки данных.",
        kind="tier",
        pipelines=(
            "pipeline/pesnik-clean",
            "pipeline/kozodoi-numeric-ratios",
            "pipeline/kaggler-target-frequency",
        ),
    ),
    PipelinePreset(
        id="fast",
        title="Быстрые и простые",
        description="Лёгкие feature-engineering pipeline: числовые отношения, частоты и даты.",
        kind="tier",
        pipelines=(
            "pipeline/pesnik-clean",
            "pipeline/kozodoi-numeric-ratios",
            "pipeline/kozodoi-combination-frequency",
            "pipeline/kozodoi-high-cardinality",
            "pipeline/kaggler-target-frequency",
            "pipeline/cdeotte-datetime-cyclical",
            "pipeline/kaggle-ieee-missingness",
            "pipeline/dptools-row-profile",
        ),
    ),
    PipelinePreset(
        id="balanced",
        title="Сбалансированные",
        description="Средний набор: чистая база, категории, пропуски, время и профили строк.",
        kind="tier",
        pipelines=(
            "pipeline/pesnik-clean",
            "pipeline/kozodoi-numeric-ratios",
            "pipeline/kozodoi-combination-frequency",
            "pipeline/kozodoi-high-cardinality",
            "pipeline/kaggler-target-frequency",
            "pipeline/cdeotte-datetime-cyclical",
            "pipeline/kaggle-ieee-missingness",
            "pipeline/kaggle-ieee-frequency",
            "pipeline/kaggle-ieee-time-profile",
            "pipeline/dptools-row-profile",
            "pipeline/dptools-group-profile",
        ),
    ),
    PipelinePreset(
        id="quality",
        title="Тяжёлые / максимум качества",
        description="Стеки, ансамбли и дорогие преобразования для финального сравнения.",
        kind="tier",
        pipelines=(
            "pipeline/arnabbiswas-staged",
            "pipeline/cdeotte-gbdt-blend",
            "pipeline/ensemble-amazon-interaction-target",
            "pipeline/ensemble-amazon-numeric-cross",
            "pipeline/ensemble-amazon-stack",
            "pipeline/kconv-linear-stack",
            "pipeline/kconv-sklearn-diverse",
            "pipeline/knnfeat-class-distance",
            "pipeline/minerva-oof-stack",
            "pipeline/nyaggle-cluster-stack",
            "pipeline/nyaggle-oof-ensemble",
            "pipeline/statsmaster-honest-cv",
        ),
    ),
    PipelinePreset(
        id="categorical",
        title="Категориальные признаки",
        description="Target/frequency encoding и работа с высокой кардинальностью.",
        kind="specialty",
        pipelines=(
            "pipeline/arnabbiswas-staged",
            "pipeline/dptools-group-profile",
            "pipeline/kaggler-target-frequency",
            "pipeline/kozodoi-combination-frequency",
            "pipeline/kozodoi-high-cardinality",
        ),
    ),
    PipelinePreset(
        id="time",
        title="Время и даты",
        description="Календарные, циклические и временные профили.",
        kind="specialty",
        pipelines=(
            "pipeline/cdeotte-datetime-cyclical",
            "pipeline/kaggle-ieee-time-profile",
            "pipeline/kozodoi-time-profile",
        ),
    ),
    PipelinePreset(
        id="cluster_anomaly",
        title="Кластеры и аномалии",
        description="Кластерные профили, OOF-кластеры и признаки аномальности.",
        kind="specialty",
        pipelines=(
            "pipeline/dptools-cluster-profile",
            "pipeline/kaggle-ieee-cluster-anomaly",
            "pipeline/kconv-unsupervised-profile",
            "pipeline/knnfeat-class-distance",
            "pipeline/nyaggle-cluster-stack",
            "pipeline/sergeyshilin-anomaly-contracts",
            "pipeline/sergeyshilin-contracts",
        ),
    ),
    PipelinePreset(
        id="ensemble",
        title="Стеки и ансамбли",
        description="OOF-стеки, блендинги и разнообразные базовые модели.",
        kind="specialty",
        pipelines=(
            "pipeline/cdeotte-gbdt-blend",
            "pipeline/ensemble-amazon-interaction-target",
            "pipeline/ensemble-amazon-numeric-cross",
            "pipeline/ensemble-amazon-stack",
            "pipeline/kconv-linear-stack",
            "pipeline/kconv-sklearn-diverse",
            "pipeline/minerva-oof-stack",
            "pipeline/nyaggle-cluster-stack",
            "pipeline/nyaggle-oof-ensemble",
            "pipeline/statsmaster-honest-cv",
        ),
    ),
    PipelinePreset(
        id="all",
        title="Все pipelines",
        description="Все 29 исполняемых pipeline; удобно для полного ночного прогона.",
        kind="all",
        pipelines=ALL_PIPELINES,
    ),
)


def pipeline_presets() -> list[dict[str, object]]:
    """Return JSON-safe preset definitions for the web config endpoint."""
    return [preset.to_dict() for preset in PIPELINE_PRESETS]
