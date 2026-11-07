"""Curated experiment catalog: unusual ideas, not automatic pipelines.

An experiment is a documented hypothesis with a source, status and risk.  Only
experiments marked ``implemented`` correspond to behavior already present in
the engine.  ``idea`` and ``prototype`` entries are deliberately not runnable
until a recipe/adapter and tests exist.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from kconv.errors import KconvError

_STATUSES = {"idea", "prototype", "implemented", "validated"}
_TASKS = {"binary", "multiclass", "regression"}
_EXPERIMENT_FIELDS = {
    "id",
    "title",
    "category",
    "status",
    "hypothesis",
    "source",
    "license",
    "task_types",
    "stages",
    "expected_gain",
    "cost",
    "risk",
    "recipe_id",
    "notes",
}
_CATALOG_SCHEMA_VERSION = 1
_CATALOG_RESOURCE = "experiments.json"


@dataclass(frozen=True)
class ExperimentInfo:
    id: str
    title: str
    category: str
    status: str
    hypothesis: str
    source: str
    license: str
    task_types: tuple[str, ...]
    stages: tuple[str, ...]
    expected_gain: str
    cost: str
    risk: str
    recipe_id: str | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_types"] = list(self.task_types)
        payload["stages"] = list(self.stages)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentInfo:
        """Validate one generated-catalog record and return a typed entry."""
        if not isinstance(payload, Mapping):
            raise KconvError("Experiment catalog entries must be JSON objects.")
        unknown = set(payload) - _EXPERIMENT_FIELDS
        missing = _EXPERIMENT_FIELDS - set(payload)
        if unknown:
            raise KconvError(f"Unknown experiment field(s): {', '.join(sorted(unknown))}.")
        if missing:
            raise KconvError(f"Missing experiment field(s): {', '.join(sorted(missing))}.")
        task_types = payload["task_types"]
        stages = payload["stages"]
        if (
            not isinstance(task_types, (list, tuple))
            or not task_types
            or not all(isinstance(item, str) and bool(item.strip()) for item in task_types)
        ):
            raise KconvError("Experiment task_types must be a non-empty list of non-empty strings.")
        if (
            not isinstance(stages, (list, tuple))
            or not stages
            or not all(isinstance(item, str) and bool(item.strip()) for item in stages)
        ):
            raise KconvError("Experiment stages must be a non-empty list of non-empty strings.")
        for field in (
            "id",
            "title",
            "category",
            "status",
            "hypothesis",
            "source",
            "license",
            "expected_gain",
            "cost",
            "risk",
            "notes",
        ):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise KconvError(f"Experiment field '{field}' must be a non-empty string.")
        if payload["status"] not in _STATUSES:
            raise KconvError(
                f"Unknown experiment status '{payload['status']}'.",
                f"Use one of: {', '.join(sorted(_STATUSES))}.",
            )
        unknown_tasks = set(task_types) - _TASKS
        if unknown_tasks:
            raise KconvError(f"Unknown experiment task(s): {', '.join(sorted(unknown_tasks))}.")
        recipe_id = payload["recipe_id"]
        if recipe_id is not None and (not isinstance(recipe_id, str) or not recipe_id.strip()):
            raise KconvError("Experiment recipe_id must be a non-empty string or null.")
        return cls(
            id=payload["id"],
            title=payload["title"],
            category=payload["category"],
            status=payload["status"],
            hypothesis=payload["hypothesis"],
            source=payload["source"],
            license=payload["license"],
            task_types=tuple(task_types),
            stages=tuple(stages),
            expected_gain=payload["expected_gain"],
            cost=payload["cost"],
            risk=payload["risk"],
            recipe_id=recipe_id,
            notes=payload["notes"],
        )


_EXPERIMENTS: tuple[ExperimentInfo, ...] = (
    ExperimentInfo(
        id="target-encoding-oof",
        title="Target encoding with inner OOF",
        category="experiment",
        status="implemented",
        hypothesis="High-cardinality categories can be encoded without leaking the validation target.",
        source="https://github.com/kozodoi/Kaggle_Home_Credit",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("features", "train"),
        expected_gain="medium",
        cost="low",
        risk="Leakage if fitted outside the fold; kconv uses inner OOF and smoothing.",
        recipe_id="kozodoi/tabular",
        notes="Implemented in FeatureBuilder; enable via features.target_encode or auto_encode.",
    ),
    ExperimentInfo(
        id="frequency-encoding",
        title="Frequency encoding instead of one-hot",
        category="experiment",
        status="implemented",
        hypothesis="Rare categories are better represented by their frequency than by thousands of dummy columns.",
        source="https://github.com/kozodoi/Kaggle_Home_Credit",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("features",),
        expected_gain="low-medium",
        cost="low",
        risk="Frequency can leak if computed on train+test; kconv fits it per fold.",
        recipe_id="kozodoi/tabular",
        notes="Implemented as <column>__freq in FeatureBuilder.",
    ),
    ExperimentInfo(
        id="rank-blend",
        title="Rank-average blending",
        category="experiment",
        status="implemented",
        hypothesis="Rank averaging is more robust than raw probability averaging when model scales differ.",
        source="https://github.com/cdeotte/KGMON-Playbook-2026",
        license="NO LICENSE — ideas only",
        task_types=("binary", "multiclass", "regression"),
        stages=("blend",),
        expected_gain="low",
        cost="low",
        risk="Can erase calibrated probabilities; only use for ranking metrics or as a blend candidate.",
        recipe_id="cdeotte/tabular",
        notes="Implemented as ensemble.method=rank_mean.",
    ),
    ExperimentInfo(
        id="oof-stacking",
        title="Out-of-fold stacking with a meta-model",
        category="experiment",
        status="implemented",
        hypothesis="A simple meta-model over OOF predictions can exploit complementary model errors.",
        source="https://github.com/minerva-ml/open-solution-home-credit",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("blend",),
        expected_gain="medium",
        cost="medium",
        risk="Meta-model must use OOF only; otherwise the stack is leaky.",
        recipe_id="minerva/stacking",
        notes="Implemented as ensemble.method=stack with logreg/ridge/lgbm meta-model.",
    ),
    ExperimentInfo(
        id="seed-averaging",
        title="Seed averaging",
        category="experiment",
        status="implemented",
        hypothesis="Averaging several seeds reduces variance of tree/MLP predictions.",
        source="https://github.com/cdeotte/KGMON-Playbook-2026",
        license="NO LICENSE — ideas only",
        task_types=("binary", "multiclass", "regression"),
        stages=("train",),
        expected_gain="low-medium",
        cost="linear in seeds",
        risk="Does not fix a bad validation split; multiplies runtime.",
        recipe_id="cdeotte/tabular",
        notes="Implemented as ModelConfig.seeds; profile 3-5 uses 2-3 seeds.",
    ),
    ExperimentInfo(
        id="group-aware-cv",
        title="Group-aware cross-validation",
        category="experiment",
        status="implemented",
        hypothesis="Rows sharing a group must stay in one fold, otherwise CV is optimistic.",
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("validate",),
        expected_gain="correctness",
        cost="low",
        risk="Wrong group column can make folds too small or unrepresentative.",
        recipe_id="congmnguyen/leakage-safe",
        notes="Implemented as validation.strategy=group_kfold with group_col.",
    ),
    ExperimentInfo(
        id="time-series-gap",
        title="Time-ordered validation with a gap",
        category="experiment",
        status="implemented",
        hypothesis="A gap between train and validation prevents leakage from lagged or near-duplicate rows.",
        source="https://github.com/arnabbiswas1/kaggle_pipeline_tps_aug_22",
        license="Apache-2.0",
        task_types=("binary", "multiclass", "regression"),
        stages=("validate",),
        expected_gain="correctness",
        cost="low",
        risk="A large gap reduces training data; test different gaps.",
        recipe_id="arnabbiswas1/staged",
        notes="Implemented as validation.strategy=time_series with gap.",
    ),
    ExperimentInfo(
        id="text-metadata",
        title="Text metadata features",
        category="experiment",
        status="implemented",
        hypothesis="Length and word count of text fields carry useful signal even without a language model.",
        source="https://github.com/pesnik/kaggle-starter-kit",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("features",),
        expected_gain="low",
        cost="low",
        risk="Can encode dataset artifacts; validate on a time/group split.",
        recipe_id="pesnik/baseline",
        notes="Implemented as <column>__len and <column>__words for text role columns.",
    ),
    ExperimentInfo(
        id="adversarial-validation",
        title="Adversarial train/test validation",
        category="experiment",
        status="implemented",
        hypothesis="If a classifier can separate train from test, the validation distribution is suspect.",
        source="https://github.com/StatsMaster/kaggle-grandmaster-templates",
        license="NO LICENSE — ideas only",
        task_types=("binary", "multiclass", "regression"),
        stages=("validate",),
        expected_gain="diagnostic",
        cost="low-medium",
        risk="High AUC can be caused by legitimate new categories; inspect importances.",
        recipe_id="statsmaster/tabular",
        notes="Implemented as 01_validate/adversarial.json; does not automatically change CV.",
    ),
    ExperimentInfo(
        id="pseudo-labeling",
        title="Pseudo-labeling confident test rows",
        category="experiment",
        status="idea",
        hypothesis="Adding high-confidence test predictions to training can improve a strong model.",
        source="Kaggle tabular write-ups; community pattern",
        license="ideas only",
        task_types=("binary", "multiclass", "regression"),
        stages=("train", "blend"),
        expected_gain="medium",
        cost="high",
        risk="Confirmation bias and leakage; must be evaluated with a clean holdout.",
        recipe_id=None,
        notes="Not implemented. Requires confidence thresholds, multiple rounds and ablation.",
    ),
    ExperimentInfo(
        id="feature-crosses",
        title="Targeted feature crosses",
        category="experiment",
        status="idea",
        hypothesis="A small number of domain-driven crosses can beat hundreds of automatic interactions.",
        source="https://github.com/WillKoehrsen/kaggle-automated-feature-engineering",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("features",),
        expected_gain="low-medium",
        cost="low",
        risk="Explosion of cardinality and overfitting; select by CV, not by train score.",
        recipe_id=None,
        notes="Not implemented. Candidate for a whitelist-based cross builder.",
    ),
    ExperimentInfo(
        id="row-statistics",
        title="Row-wise statistics",
        category="experiment",
        status="implemented",
        hypothesis="Mean/std/min/max across numeric columns can capture relative row profiles.",
        source="https://github.com/Quantyra/kaggle-icr-challenge",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("features",),
        expected_gain="low",
        cost="low",
        risk="Scales differ; normalize or use rank-based variants.",
        recipe_id="dptools/row-features",
        notes=(
            "Implemented in FeatureBuilder and the dptools/row-features recipe; "
            "enable via features.row_statistics. Benchmark: "
            "a local benchmark report (not committed)."
        ),
    ),
    ExperimentInfo(
        id="adversarial-sample-weighting",
        title="Adversarial sample weighting",
        category="experiment",
        status="idea",
        hypothesis="Down-weight training rows that look like test rows only if the validation split confirms the shift.",
        source="Kaggle adversarial-validation tutorials; community pattern",
        license="ideas only",
        task_types=("binary", "multiclass", "regression"),
        stages=("validate", "train"),
        expected_gain="medium",
        cost="medium",
        risk="Can amplify noise; never tune weights on the public leaderboard.",
        recipe_id=None,
        notes="Not implemented. Must be compared with an unweighted baseline on a fixed split.",
    ),
    ExperimentInfo(
        id="llm-feature-proposals",
        title="LLM-proposed additional features",
        category="experiment",
        status="idea",
        hypothesis="An LLM can propose a small, validated set of domain-aware feature expressions from schema and samples.",
        source="kconv experiment; no external code",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("features",),
        expected_gain="unknown",
        cost="high (tokens + review)",
        risk="Hallucinated columns, target leakage, privacy leakage, non-determinism.",
        recipe_id=None,
        notes="See docs/llm-features.md. LLM output is JSON proposals only; no eval/exec.",
    ),
    ExperimentInfo(
        id="multi-seed-snapshot",
        title="Multi-seed snapshot ensemble",
        category="experiment",
        status="idea",
        hypothesis="Checkpoints from different seeds/epochs can add diversity beyond final-model averaging.",
        source="Community ensembles; no specific repository",
        license="ideas only",
        task_types=("binary", "multiclass", "regression"),
        stages=("train", "blend"),
        expected_gain="low-medium",
        cost="high",
        risk="Large disk usage and correlated snapshots; validate diversity first.",
        recipe_id=None,
        notes="Not implemented. More relevant to neural models than GBDT.",
    ),
    ExperimentInfo(
        id="knn-class-distance-features",
        title="Class-conditional kNN distance features",
        category="experiment",
        status="implemented",
        hypothesis=(
            "Per-class distances to the nearest neighbours of that class encode local class "
            "geometry that axis-aligned trees cannot build."
        ),
        source="https://github.com/upura/knnFeat",
        license="MIT",
        task_types=("binary", "multiclass"),
        stages=("features", "train"),
        expected_gain="medium",
        cost="high",
        risk="Label-conditioned neighbour search must run inside each outer fold; full-matrix fitting leaks labels.",
        recipe_id="knnfeat/knn-features",
        notes=(
            "Implemented in kconv.features.knn with fit-in-fold OOF, bounded "
            "reference sets and a classification-only guard; enable via "
            "features.knn_features."
        ),
    ),
    ExperimentInfo(
        id="gaussian-rank-normalization",
        title="Gaussian rank normalization",
        category="experiment",
        status="prototype",
        hypothesis=(
            "Rank-to-normal mapping removes skew and outliers without assuming a parametric "
            "distribution, helping scale-sensitive and distance-based models."
        ),
        source="https://github.com/alno/kaggle-allstate-claims-severity",
        license="NO LICENSE — ideas only",
        task_types=("regression",),
        stages=("features",),
        expected_gain="low-medium",
        cost="low",
        risk="The source ranks train and test jointly; kconv fits the reference distribution on the train fold only.",
        recipe_id=None,
        notes=(
            "UP-03: prototype in kconv.features.unusual.GaussianRankTransformer; "
            "benchmark and fold-safe transform are in a local benchmark report (not committed)."
        ),
    ),
    ExperimentInfo(
        id="modulo-decomposition",
        title="Modulo and multi-resolution decomposition",
        category="experiment",
        status="implemented",
        hypothesis=(
            "An integer that mixes coarse buckets and a fine residual is hard for axis-aligned "
            "trees; x % m and x // step expose both views."
        ),
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary", "regression"),
        stages=("features",),
        expected_gain="low-medium",
        cost="low",
        risk="Modulo features can memorize individuals; accept only on a fixed split and never use as an identity feature.",
        recipe_id=None,
        notes=(
            "Implemented in FeatureBuilder behind features.modulo_features; "
            "opt-in because it should only be applied to identifier-like "
            "columns. Benchmark: a local benchmark report (not committed)."
        ),
    ),
    ExperimentInfo(
        id="dense-crossfit-target-encoding",
        title="Dense cross-fit target encoding",
        category="experiment",
        status="idea",
        hypothesis=(
            "More inner folds shrink the gap between train-time and inference-time target "
            "encodings, reducing encoder overfit on high-cardinality columns."
        ),
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary", "multiclass"),
        stages=("features",),
        expected_gain="low-medium",
        cost="medium",
        risk="The encoder must stay inside the outer fold; scikit-learn TargetEncoder semantics changed across versions.",
        recipe_id=None,
        notes="UP-15: candidate for an inner-CV parameter on FeatureBuilder's target encoder.",
    ),
    ExperimentInfo(
        id="risk-adjusted-target-encoding",
        title="Risk-adjusted cross-fitted effect encoding",
        category="experiment",
        status="idea",
        hypothesis=(
            "Estimating per-value effects around a fold-fitted mechanism margin captures "
            "conditional signal that a plain mean target encoder misses."
        ),
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary",),
        stages=("features",),
        expected_gain="medium",
        cost="low-medium",
        risk="Effects are fitted on labels; without inner cross-fitting this is a direct label encoder.",
        recipe_id=None,
        notes="UP-17: highest-ceiling feature family in the triage, but needs a dedicated cross-fit harness first.",
    ),
    ExperimentInfo(
        id="transductive-frequency-features",
        title="Label-free pooled frequency features",
        category="experiment",
        status="idea",
        hypothesis=(
            "Category frequencies estimated on pooled unlabelled train+test rows can be a "
            "better prior than train-only frequencies for distribution-shifted test data."
        ),
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("features",),
        expected_gain="medium",
        cost="low",
        risk="Transductive: the pipeline depends on the test set and some competitions forbid it; labels must never be pooled.",
        recipe_id=None,
        notes="UP-19: opt-in only; current FeatureBuilder frequency encoding is fit on the training fold.",
    ),
    ExperimentInfo(
        id="offset-boosting",
        title="Offset boosting with a fold-trained baseline",
        category="experiment",
        status="idea",
        hypothesis=(
            "Using a domain-mechanism prior as the booster's base margin lets trees model only "
            "the residual around a physically sensible baseline."
        ),
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary",),
        stages=("features", "train"),
        expected_gain="medium",
        cost="low",
        risk="The baseline must be refit per fold and the offset columns removed from the booster features.",
        recipe_id=None,
        notes="UP-21: top triage candidate, but needs adapter-level base_margin support for each booster.",
    ),
    ExperimentInfo(
        id="tabular-foundation-baseline",
        title="Tabular foundation model baseline",
        category="experiment",
        status="idea",
        hypothesis=(
            "A pretrained tabular foundation model on raw columns can beat a tuned GBDT on "
            "small/medium tables and is a diverse blend member."
        ),
        source="https://github.com/cdeotte/Tabular_Foundation_Models",
        license="NO LICENSE — ideas only; model terms must be checked",
        task_types=("binary", "multiclass", "regression"),
        stages=("train", "infer", "blend"),
        expected_gain="high on small/medium tables",
        cost="medium",
        risk="Model weights have non-commercial/version terms; context limits and non-reproducible releases are real blockers.",
        recipe_id=None,
        notes="UP-23: no code vendored; explicit model-license and GPU review required before integration.",
    ),
    ExperimentInfo(
        id="conditional-permutation-importance",
        title="Conditional permutation importance within strata",
        category="experiment",
        status="idea",
        hypothesis=(
            "Permuting a feature only within correlated strata isolates its contribution "
            "better than global permutation importance."
        ),
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary",),
        stages=("validate",),
        expected_gain="diagnostic",
        cost="medium",
        risk="Still an approximate diagnostic, not a causal effect or an independent holdout.",
        recipe_id=None,
        notes="UP-35: implement as a report/diagnostic, not as a model-selection signal.",
    ),
    ExperimentInfo(
        id="label-free-structure-audit",
        title="Label-free covariate-retention audit",
        category="experiment",
        status="idea",
        hypothesis=(
            "Train-to-test covariate copying can be detected without labels by comparing "
            "unique-row column matches against a within-bin permutation null."
        ),
        source="https://github.com/congmnguyen/tabular-ml-lab",
        license="MIT",
        task_types=("binary", "multiclass", "regression"),
        stages=("validate",),
        expected_gain="diagnostic / leak-audit",
        cost="medium",
        risk="A failed null does not prove absence of leakage; the procedure may read sensitive external data.",
        recipe_id=None,
        notes="UP-36: correctness gate candidate; report-only, never a feature or automatic decision.",
    ),
)


def catalog_payload() -> dict[str, Any]:
    """Return the deterministic JSON payload generated from ``_EXPERIMENTS``.

    ``scripts/generate_experiments_catalog.py`` writes this payload to the
    packaged asset.  Keep it free of timestamps so ``--check`` is stable.
    """
    return {
        "schema_version": _CATALOG_SCHEMA_VERSION,
        "generated_from": "src/kconv/experiments.py",
        "experiments": [experiment.to_dict() for experiment in _EXPERIMENTS],
    }


def validate_catalog(payload: Mapping[str, Any]) -> tuple[ExperimentInfo, ...]:
    """Validate a generated catalog and return typed experiments.

    The loader is intentionally strict: a stale wheel or a hand-edited asset
    should fail loudly instead of silently dropping experiments.
    """
    if not isinstance(payload, Mapping):
        raise KconvError("Experiment catalog root must be a JSON object.")
    version = payload.get("schema_version")
    if version != _CATALOG_SCHEMA_VERSION:
        raise KconvError(
            f"Unsupported experiment catalog schema_version {version!r}.",
            f"Regenerate it with scripts/generate_experiments_catalog.py (expected {_CATALOG_SCHEMA_VERSION}).",
        )
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise KconvError("Experiment catalog must contain a non-empty 'experiments' list.")
    experiments = tuple(ExperimentInfo.from_dict(item) for item in raw_experiments)
    seen: set[str] = set()
    for experiment in experiments:
        if experiment.id in seen:
            raise KconvError(f"Duplicate experiment id in catalog: {experiment.id}.")
        seen.add(experiment.id)
    return experiments


def load_catalog(path: str | Path | None = None) -> tuple[ExperimentInfo, ...]:
    """Load a generated experiment catalog from a path or the packaged asset."""
    if path is None:
        raw = files("kconv.assets").joinpath(_CATALOG_RESOURCE).read_text(encoding="utf-8")
        origin = f"packaged {_CATALOG_RESOURCE}"
    else:
        candidate = Path(path)
        if not candidate.is_file():
            raise KconvError(
                f"Experiment catalog not found: {candidate}",
                "Regenerate it with scripts/generate_experiments_catalog.py.",
            )
        raw = candidate.read_text(encoding="utf-8")
        origin = str(candidate)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KconvError(
            f"Experiment catalog is not valid JSON: {origin}",
            "Regenerate it with scripts/generate_experiments_catalog.py.",
        ) from exc
    return validate_catalog(payload)


@lru_cache(maxsize=1)
def _bundled_experiments() -> tuple[ExperimentInfo, ...]:
    return load_catalog()


def list_experiments(
    *,
    status: str | None = None,
    task: str | None = None,
) -> list[ExperimentInfo]:
    """List curated experiments, optionally filtered by status/task."""
    if status is not None and status not in _STATUSES:
        raise KconvError(
            f"Unknown experiment status '{status}'.",
            f"Use one of: {', '.join(sorted(_STATUSES))}.",
        )
    if task is not None and task not in _TASKS:
        raise KconvError(f"Unknown task '{task}'.", "Use binary, multiclass or regression.")
    result = []
    for experiment in _bundled_experiments():
        if status is not None and experiment.status != status:
            continue
        if task is not None and task not in experiment.task_types:
            continue
        result.append(experiment)
    return result


def get_experiment(experiment_id: str) -> ExperimentInfo:
    """Return one experiment by id; raises KconvError for unknown ids."""
    for experiment in _bundled_experiments():
        if experiment.id == experiment_id:
            return experiment
    raise KconvError(
        f"Unknown experiment '{experiment_id}'.",
        "Run 'kconv experiments' to list curated experiments.",
    )


__all__ = [
    "ExperimentInfo",
    "catalog_payload",
    "get_experiment",
    "list_experiments",
    "load_catalog",
    "validate_catalog",
]
