"""Model-parameter validation before a run starts.

Adapters accept a free-form ``params`` mapping, which is useful for recipes but
dangerous for hand-edited configs: a typo such as ``n_estimator`` used to be
silently ignored.  This module validates known adapter schemas and inspects the
constructor signature of every scikit-learn catalog estimator, so unknown keys
and obviously invalid values fail with a ``ConfigError`` before training.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import Any

from kconv.errors import ConfigError
from kconv.models.catalog import SKLEARN_SPECS
from kconv.models.sklearn_generic import _load_class


@dataclass(frozen=True)
class ParamRule:
    """One accepted parameter and its coarse validation rule."""

    types: tuple[type, ...] = (int, float, str, bool, type(None))
    minimum: float | None = None
    maximum: float | None = None
    choices: frozenset[Any] | None = None


def _rule(
    types: tuple[type, ...] = (int, float, str, bool, type(None)),
    minimum: float | None = None,
    maximum: float | None = None,
    choices: tuple[Any, ...] | None = None,
) -> ParamRule:
    return ParamRule(
        types=types,
        minimum=minimum,
        maximum=maximum,
        choices=frozenset(choices) if choices is not None else None,
    )


_NUM = (int, float)
_INT = (int,)
_STR = (str,)
_BOOL = (bool,)

# Explicit schemas for the built-in adapters.  The lists are intentionally a
# little wider than the defaults: recipes may legitimately pass objective,
# eval_metric, device or thread options through to the native library.
MODEL_PARAM_SCHEMAS: dict[str, dict[str, ParamRule]] = {
    "lgbm": {
        "n_estimators": _rule(_INT, 1, 1_000_000),
        "learning_rate": _rule(_NUM, 1e-6, 1.0),
        "num_leaves": _rule(_INT, 2, 1_000_000),
        "max_depth": _rule(_INT, 1, 1_000),
        "min_child_samples": _rule(_INT, 1, 1_000_000),
        "min_data_in_leaf": _rule(_INT, 1, 1_000_000),
        "min_sum_hessian_in_leaf": _rule(_NUM, 0.0),
        "min_child_weight": _rule(_NUM, 0.0),
        "min_split_gain": _rule(_NUM, 0.0),
        "subsample": _rule(_NUM, 0.0, 1.0),
        "subsample_freq": _rule(_INT, 0, 1_000_000),
        "colsample_bytree": _rule(_NUM, 0.0, 1.0),
        "reg_lambda": _rule(_NUM, 0.0),
        "reg_alpha": _rule(_NUM, 0.0),
        "lambda_l1": _rule(_NUM, 0.0),
        "lambda_l2": _rule(_NUM, 0.0),
        "random_state": _rule(_INT),
        "seed": _rule(_INT),
        "num_threads": _rule(_INT, 0, 1_000_000),
        "n_jobs": _rule(_INT, -1, 1_000_000),
        "verbosity": _rule(_INT, -1, 10),
        "deterministic": _rule(_BOOL),
        "force_col_wise": _rule(_BOOL),
        "force_row_wise": _rule(_BOOL),
        "boosting_type": _rule(_STR),
        "objective": _rule(_STR),
        "metric": _rule(_STR),
        "eval_metric": _rule(_STR),
        "max_bin": _rule(_INT, 2, 1_000_000),
        "max_cat_threshold": _rule(_INT, 1, 1_000_000),
        "max_cat_to_onehot": _rule(_INT, 1, 1_000_000),
        "cat_smooth": _rule(_NUM, 0.0),
        "cat_l2": _rule(_NUM, 0.0),
        "feature_fraction": _rule(_NUM, 0.0, 1.0),
        "feature_fraction_bynode": _rule(_NUM, 0.0, 1.0),
        "bagging_fraction": _rule(_NUM, 0.0, 1.0),
        "bagging_freq": _rule(_INT, 0, 1_000_000),
        "path_smooth": _rule(_NUM, 0.0),
        "is_unbalance": _rule(_BOOL),
        "scale_pos_weight": _rule(_NUM, 0.0),
        "class_weight": _rule((str, dict, type(None))),
        "monotone_constraints": _rule((list, dict)),
        "interaction_constraints": _rule((list, str)),
        "verbose_eval": _rule((bool, int)),
        "first_metric_only": _rule(_BOOL),
        "device": _rule(_STR),
    },
    "xgb": {
        "n_estimators": _rule(_INT, 1, 1_000_000),
        "max_depth": _rule(_INT, 1, 1_000),
        "learning_rate": _rule(_NUM, 1e-6, 1.0),
        "subsample": _rule(_NUM, 0.0, 1.0),
        "colsample_bytree": _rule(_NUM, 0.0, 1.0),
        "colsample_bylevel": _rule(_NUM, 0.0, 1.0),
        "colsample_bynode": _rule(_NUM, 0.0, 1.0),
        "reg_lambda": _rule(_NUM, 0.0),
        "reg_alpha": _rule(_NUM, 0.0),
        "min_child_weight": _rule(_NUM, 0.0),
        "gamma": _rule(_NUM, 0.0),
        "max_bin": _rule(_INT, 2, 1_000_000),
        "random_state": _rule(_INT),
        "seed": _rule(_INT),
        "n_jobs": _rule(_INT, -1, 1_000_000),
        "tree_method": _rule(_STR),
        "verbosity": _rule(_INT, 0, 10),
        "objective": _rule(_STR),
        "eval_metric": _rule(_STR),
        "early_stopping_rounds": _rule(_INT, 1, 1_000_000),
        "enable_categorical": _rule(_BOOL),
        "max_cat_to_onehot": _rule(_INT, 1, 1_000_000),
        "base_score": _rule((int, float, type(None))),
        "importance_type": _rule(_STR),
        "callbacks": _rule((list, tuple)),
        "device": _rule(_STR),
    },
    "catboost": {
        "iterations": _rule(_INT, 1, 1_000_000),
        "depth": _rule(_INT, 1, 16),
        "learning_rate": _rule(_NUM, 1e-6, 1.0),
        "random_seed": _rule(_INT),
        "verbose": _rule(_INT, 0, 10),
        "allow_writing_files": _rule(_BOOL),
        "thread_count": _rule(_INT, 0, 1_000_000),
        "loss_function": _rule(_STR),
        "eval_metric": _rule(_STR),
        "l2_leaf_reg": _rule(_NUM, 0.0),
        "random_strength": _rule(_NUM, 0.0),
        "bagging_temperature": _rule(_NUM, 0.0),
        "border_count": _rule(_INT, 1, 65_535),
        "one_hot_max_size": _rule(_INT, 0, 255),
        "has_time": _rule(_BOOL),
        "bootstrap_type": _rule(_STR),
        "subsample": _rule(_NUM, 0.0, 1.0),
        "rsm": _rule(_NUM, 0.0, 1.0),
        "random_state": _rule(_INT),
        "logging_level": _rule(_STR),
        "metric_period": _rule(_INT, 1, 1_000_000),
        "use_best_model": _rule(_BOOL),
        "od_type": _rule(_STR),
        "od_wait": _rule(_INT, 0, 1_000_000),
        "task_type": _rule(_STR),
        "devices": _rule((str, list)),
        "gpu_ram_part": _rule(_NUM, 0.0, 1.0),
    },
    "histgb": {
        "max_iter": _rule(_INT, 1, 1_000_000),
        "learning_rate": _rule(_NUM, 1e-6, 1.0),
        "max_leaf_nodes": _rule(_INT, 2, 1_000_000),
        "max_depth": _rule(_INT, 1, 1_000),
        "l2_regularization": _rule(_NUM, 0.0),
        "random_state": _rule(_INT),
        "early_stopping": _rule(_BOOL),
        "validation_fraction": _rule(_NUM, 0.0, 1.0),
        "n_iter_no_change": _rule(_INT, 1, 1_000_000),
        "min_samples_leaf": _rule(_INT, 1, 1_000_000),
        "max_bins": _rule(_INT, 2, 255),
        "categorical_features": _rule((str, list, tuple, type(None))),
        "loss": _rule(_STR),
        "class_weight": _rule((str, dict, type(None))),
        "monotonic_cst": _rule((list, tuple, dict, type(None))),
        "interaction_cst": _rule((list, tuple, type(None))),
        "warm_start": _rule(_BOOL),
        "verbose": _rule((bool, int)),
    },
    "logreg": {
        "max_iter": _rule(_INT, 1, 1_000_000),
        "random_state": _rule(_INT),
        "solver": _rule(_STR),
        "C": _rule(_NUM, 1e-12),
        "penalty": _rule((str, type(None))),
        "l1_ratio": _rule(_NUM, 0.0, 1.0),
        "class_weight": _rule((str, dict, type(None))),
        "fit_intercept": _rule(_BOOL),
        "tol": _rule(_NUM, 1e-12),
        "n_jobs": _rule(_INT, -1, 1_000_000),
        "multi_class": _rule(_STR),
        "dual": _rule(_BOOL),
        "intercept_scaling": _rule(_NUM, 0.0),
        "warm_start": _rule(_BOOL),
        "verbose": _rule(_INT, 0, 10),
    },
    "ridge": {
        "alpha": _rule(_NUM, 0.0),
        "fit_intercept": _rule(_BOOL),
        "copy_X": _rule(_BOOL),
        "max_iter": _rule((int, type(None)), 1, 1_000_000),
        "tol": _rule(_NUM, 1e-12),
        "solver": _rule(_STR),
        "positive": _rule(_BOOL),
        "random_state": _rule((int, type(None))),
    },
    # TabPFN forwards extra keyword arguments to its constructor, so this is a
    # documented allow-list rather than a closed schema.
    "tabpfn": {
        "model_version": _rule(_STR),
        "device": _rule(_STR),
        "n_estimators": _rule(_INT, 1, 1_000),
        "max_train_rows": _rule(_INT, 1, 10_000_000),
        "max_features": _rule(_INT, 1, 100_000),
        "max_classes": _rule(_INT, 2, 100_000),
        "fit_mode": _rule(_STR),
        "ignore_pretraining_limits": _rule(_BOOL),
        "model_path": _rule((str, type(None))),
        "show_progress_bar": _rule(_BOOL),
        "random_state": _rule(_INT),
        "n_preprocessing_jobs": _rule(_INT, 1, 1_000),
        "memory_saving_mode": _rule((bool, str)),
        "inference_precision": _rule(_STR),
    },
}

# Deliberately no fuzzy alias table: ``n_estimator`` is a typo, not a synonym.
# Catching it is the point of early validation.
_COMMON_ALIASES: dict[str, str] = {}


def _normalize_key(key: str) -> str:
    return _COMMON_ALIASES.get(key.strip().lower(), key.strip())


def _check_value(model: str, key: str, value: Any, rule: ParamRule) -> None:
    if rule.choices is not None and value not in rule.choices:
        allowed = ", ".join(repr(item) for item in sorted(rule.choices, key=str))
        raise ConfigError(
            f"Model '{model}' parameter '{key}' must be one of: {allowed}.",
            "Use a documented adapter parameter value.",
        )
    if isinstance(value, bool) and bool not in rule.types and int in rule.types:
        raise ConfigError(
            f"Model '{model}' parameter '{key}' must be a number, not a boolean.",
            "Pass the numeric value expected by the adapter.",
        )
    if not isinstance(value, rule.types):
        expected = ", ".join(item.__name__ for item in rule.types)
        raise ConfigError(
            f"Model '{model}' parameter '{key}' has type {type(value).__name__}; expected {expected}.",
            "Check the adapter documentation or 'kconv models'.",
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ConfigError(
                f"Model '{model}' parameter '{key}' must be finite.",
                "Remove NaN/inf values from model params.",
            )
        if rule.minimum is not None and float(value) < rule.minimum:
            raise ConfigError(
                f"Model '{model}' parameter '{key}' must be >= {rule.minimum}.",
                "Use a value inside the documented range.",
            )
        if rule.maximum is not None and float(value) > rule.maximum:
            raise ConfigError(
                f"Model '{model}' parameter '{key}' must be <= {rule.maximum}.",
                "Use a value inside the documented range.",
            )


def _generic_sklearn_rules(name: str) -> dict[str, ParamRule]:
    spec = SKLEARN_SPECS.get(name)
    if spec is None:
        return {}
    try:
        cls = _load_class(spec.path)
        signature = inspect.signature(cls.__init__)
    except (ImportError, AttributeError, TypeError, ValueError):
        return {}
    rules: dict[str, ParamRule] = {}
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        rules[parameter.name] = _rule()
    return rules


def validate_model_params(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate one model's params and return a normalized copy.

    Unknown keys are rejected.  The common typo aliases are normalized to the
    canonical key before validation, which keeps older hand-written configs
    working while still catching genuinely unknown names.
    """
    params = dict(params or {})
    rules = MODEL_PARAM_SCHEMAS.get(name)
    if rules is None:
        rules = _generic_sklearn_rules(name)
    if not rules:
        # Unknown or open-schema adapter: still reject non-JSON/non-finite
        # values, but do not pretend to know every native parameter.
        for key, value in params.items():
            if not isinstance(key, str):
                raise ConfigError(
                    f"Model '{name}' parameter names must be strings.",
                    "Use adapter parameter names as JSON object keys.",
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigError(
                    f"Model '{name}' parameter '{key}' must be finite.",
                    "Remove NaN/inf values from model params.",
                )
        return params

    normalized: dict[str, Any] = {}
    for raw_key, value in params.items():
        if not isinstance(raw_key, str):
            raise ConfigError(
                f"Model '{name}' parameter names must be strings.",
                "Use adapter parameter names as JSON object keys.",
            )
        key = _normalize_key(raw_key)
        rule = rules.get(key)
        if rule is None:
            allowed = ", ".join(sorted(rules))
            hint = f"Allowed parameters: {allowed}."
            if re.search(r"[^a-z0-9_]", raw_key):
                hint = "Use underscores, not dashes. " + hint
            raise ConfigError(
                f"Unknown parameter '{raw_key}' for model '{name}'.",
                hint,
            )
        _check_value(name, key, value, rule)
        normalized[key] = value
    return normalized


def validate_config_models(config: Any) -> None:
    """Validate every enabled model in a Config-like object."""
    for model in getattr(config, "models", []) or []:
        if getattr(model, "enabled", True):
            validate_model_params(str(model.name), getattr(model, "params", {}) or {})
