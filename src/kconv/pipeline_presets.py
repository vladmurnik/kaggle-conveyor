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
    "pipeline/arnabbiswas-modulo-profile",
    "pipeline/arnabbiswas-mutual-profile",
    "pipeline/arnabbiswas-staged",
    "pipeline/cdeotte-datetime-cyclical",
    "pipeline/cdeotte-datetime-numeric",
    "pipeline/cdeotte-gbdt-blend",
    "pipeline/dptools-cluster-profile",
    "pipeline/dptools-group-profile",
    "pipeline/dptools-row-profile",
    "pipeline/ensemble-amazon-interaction-group",
    "pipeline/ensemble-amazon-interaction-missing",
    "pipeline/ensemble-amazon-interaction-target",
    "pipeline/ensemble-amazon-numeric-cross",
    "pipeline/ensemble-amazon-stack",
    "pipeline/ensemble-amazon-time-interaction",
    "pipeline/kaggle-ieee-cluster-anomaly",
    "pipeline/kaggle-ieee-frequency",
    "pipeline/kaggle-ieee-missing-cluster",
    "pipeline/kaggle-ieee-missing-group",
    "pipeline/kaggle-ieee-missingness",
    "pipeline/kaggle-ieee-time-group",
    "pipeline/kaggle-ieee-time-profile",
    "pipeline/kaggle-ieee-time-row-profile",
    "pipeline/kaggler-target-frequency",
    "pipeline/kconv-datetime-cluster",
    "pipeline/kconv-datetime-row-profile",
    "pipeline/kconv-linear-stack",
    "pipeline/kconv-missing-row-profile",
    "pipeline/kconv-modulo-anomaly",
    "pipeline/kconv-sklearn-diverse",
    "pipeline/kconv-unsupervised-profile",
    "pipeline/knnfeat-class-distance",
    "pipeline/knnfeat-cluster-anomaly",
    "pipeline/knnfeat-hash-distance",
    "pipeline/kozodoi-combination-frequency",
    "pipeline/kozodoi-hash-datetime",
    "pipeline/kozodoi-hash-group-profile",
    "pipeline/kozodoi-hash-missing-numeric",
    "pipeline/kozodoi-high-cardinality",
    "pipeline/kozodoi-numeric-cluster",
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
        description="Лёгкие feature-engineering pipeline: числовые отношения, частоты, даты, пропуски, хеширование и профили строк.",
        kind="tier",
        pipelines=(
            "pipeline/pesnik-clean",
            "pipeline/kozodoi-numeric-ratios",
            "pipeline/kozodoi-combination-frequency",
            "pipeline/kozodoi-high-cardinality",
            "pipeline/kozodoi-hash-group-profile",
            "pipeline/kaggler-target-frequency",
            "pipeline/cdeotte-datetime-cyclical",
            "pipeline/cdeotte-datetime-numeric",
            "pipeline/kaggle-ieee-missingness",
            "pipeline/kconv-datetime-row-profile",
            "pipeline/kconv-missing-row-profile",
            "pipeline/arnabbiswas-modulo-profile",
            "pipeline/dptools-row-profile",
        ),
    ),
    PipelinePreset(
        id="balanced",
        title="Сбалансированные",
        description="Средний набор: чистая база, категории, пропуски, время, группы, хеши и профили строк.",
        kind="tier",
        pipelines=(
            "pipeline/pesnik-clean",
            "pipeline/kozodoi-numeric-ratios",
            "pipeline/kozodoi-combination-frequency",
            "pipeline/kozodoi-high-cardinality",
            "pipeline/kozodoi-hash-group-profile",
            "pipeline/kozodoi-hash-missing-numeric",
            "pipeline/kozodoi-hash-datetime",
            "pipeline/kozodoi-numeric-cluster",
            "pipeline/kaggler-target-frequency",
            "pipeline/cdeotte-datetime-cyclical",
            "pipeline/cdeotte-datetime-numeric",
            "pipeline/kaggle-ieee-missingness",
            "pipeline/kaggle-ieee-frequency",
            "pipeline/kaggle-ieee-time-profile",
            "pipeline/kaggle-ieee-time-group",
            "pipeline/kaggle-ieee-time-row-profile",
            "pipeline/kaggle-ieee-missing-cluster",
            "pipeline/kaggle-ieee-missing-group",
            "pipeline/kconv-datetime-row-profile",
            "pipeline/kconv-missing-row-profile",
            "pipeline/arnabbiswas-mutual-profile",
            "pipeline/arnabbiswas-modulo-profile",
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
            "pipeline/ensemble-amazon-interaction-group",
            "pipeline/ensemble-amazon-interaction-missing",
            "pipeline/ensemble-amazon-interaction-target",
            "pipeline/ensemble-amazon-numeric-cross",
            "pipeline/ensemble-amazon-stack",
            "pipeline/ensemble-amazon-time-interaction",
            "pipeline/kconv-datetime-cluster",
            "pipeline/kconv-linear-stack",
            "pipeline/kconv-modulo-anomaly",
            "pipeline/kconv-sklearn-diverse",
            "pipeline/knnfeat-class-distance",
            "pipeline/knnfeat-cluster-anomaly",
            "pipeline/knnfeat-hash-distance",
            "pipeline/kozodoi-numeric-cluster",
            "pipeline/minerva-oof-stack",
            "pipeline/nyaggle-cluster-stack",
            "pipeline/nyaggle-oof-ensemble",
            "pipeline/statsmaster-honest-cv",
        ),
    ),
    PipelinePreset(
        id="categorical",
        title="Категориальные признаки",
        description="Target/frequency encoding, хеширование, взаимодействия и работа с высокой кардинальностью.",
        kind="specialty",
        pipelines=(
            "pipeline/arnabbiswas-staged",
            "pipeline/ensemble-amazon-interaction-group",
            "pipeline/ensemble-amazon-interaction-missing",
            "pipeline/dptools-group-profile",
            "pipeline/kaggler-target-frequency",
            "pipeline/knnfeat-hash-distance",
            "pipeline/kozodoi-combination-frequency",
            "pipeline/kozodoi-hash-datetime",
            "pipeline/kozodoi-hash-group-profile",
            "pipeline/kozodoi-hash-missing-numeric",
            "pipeline/kozodoi-high-cardinality",
            "pipeline/kaggle-ieee-missing-group",
        ),
    ),
    PipelinePreset(
        id="time",
        title="Время и даты",
        description="Календарные, циклические и временные профили.",
        kind="specialty",
        pipelines=(
            "pipeline/cdeotte-datetime-cyclical",
            "pipeline/cdeotte-datetime-numeric",
            "pipeline/ensemble-amazon-time-interaction",
            "pipeline/kaggle-ieee-time-group",
            "pipeline/kaggle-ieee-time-profile",
            "pipeline/kaggle-ieee-time-row-profile",
            "pipeline/kconv-datetime-cluster",
            "pipeline/kconv-datetime-row-profile",
            "pipeline/kozodoi-hash-datetime",
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
            "pipeline/kaggle-ieee-missing-cluster",
            "pipeline/kconv-datetime-cluster",
            "pipeline/kconv-modulo-anomaly",
            "pipeline/kconv-unsupervised-profile",
            "pipeline/knnfeat-class-distance",
            "pipeline/knnfeat-cluster-anomaly",
            "pipeline/kozodoi-numeric-cluster",
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
            "pipeline/ensemble-amazon-interaction-group",
            "pipeline/ensemble-amazon-interaction-missing",
            "pipeline/ensemble-amazon-interaction-target",
            "pipeline/ensemble-amazon-numeric-cross",
            "pipeline/ensemble-amazon-stack",
            "pipeline/ensemble-amazon-time-interaction",
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
        description="Все 49 исполняемых pipeline; удобно для полного ночного прогона.",
        kind="all",
        pipelines=ALL_PIPELINES,
    ),
)


def pipeline_presets() -> list[dict[str, object]]:
    """Return JSON-safe preset definitions for the web config endpoint."""
    return [preset.to_dict() for preset in PIPELINE_PRESETS]
