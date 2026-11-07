"""Iterative one-feature-at-a-time LLM generation and repair regressions."""

from __future__ import annotations

import json

import pytest

from kconv.errors import KconvError
from kconv.llm_features import FeatureProposal
from kconv.llm_iterative import (
    build_single_feature_prompt,
    generate_features_iteratively,
)
from kconv.llm_provider import LLMResponse, TokenUsage


def _response(text: str, *, total: int = 15, duration: float = 0.1) -> LLMResponse:
    return LLMResponse(
        text=text,
        provider="local",
        model="fake-local",
        usage=TokenUsage(10, 5, total),
        duration_sec=duration,
    )


def _feature(name: str, expression: str) -> str:
    return json.dumps(
        {
            "features": [
                {
                    "name": name,
                    "kind": "numeric",
                    "expression": expression,
                    "rationale": "test feature",
                }
            ]
        }
    )


def test_single_feature_prompt_contains_context_and_untrusted_hints() -> None:
    prompt = build_single_feature_prompt(
        ["a", "b"],
        [{"a": 1, "b": 2, "target": 1, "id": 7}],
        numeric_columns=["a", "b"],
        existing=[FeatureProposal("a_plus_b", "numeric", "a + b", "sum")],
        rejected=[{"answer": "bad", "error": "unknown column: nope"}],
        user_ideas=["try a ratio between a and b"],
        user_code=["def ratio(a, b):\n    return a / b"],
        feature_index=2,
        max_features=4,
    )

    assert len(prompt) <= 8000
    assert "exactly ONE" in prompt
    assert "Only use these columns" in prompt
    assert "Feature 2 of at most 4" in prompt
    assert '"target"' not in prompt
    assert '"id"' not in prompt
    assert "a_plus_b: a + b" in prompt
    assert "Previous invalid answer" in prompt
    assert "Validation error" in prompt
    assert "User ideas" in prompt
    assert "do NOT execute" in prompt
    assert "def ratio" in prompt


def test_generation_asks_for_one_feature_at_a_time_and_reuses_context() -> None:
    calls: list[str] = []

    def llm(prompt: str) -> LLMResponse:
        calls.append(prompt)
        index = len(calls)
        return _response(_feature(f"feat_{index}", f"a + {index}"))

    generation = generate_features_iteratively(
        ["a", "b"],
        [{"a": 1, "b": 2}],
        llm,
        max_features=3,
        numeric_columns=["a", "b"],
    )

    assert len(calls) == 3
    assert [proposal.name for proposal in generation.plan.proposals] == [
        "feat_1",
        "feat_2",
        "feat_3",
    ]
    assert "feat_1" in calls[1]
    assert "feat_1" in calls[2] and "feat_2" in calls[2]
    assert "Feature 1 of at most 3" in calls[0]
    assert "Feature 3 of at most 3" in calls[2]
    assert generation.usage.total_tokens == 45
    assert generation.duration_sec == pytest.approx(0.3)
    payload = generation.to_dict()
    assert payload["mode"] == "one-feature-per-call"
    assert payload["calls"] == 3
    assert payload["features"] == 3
    assert payload["repair_calls"] == 0


def test_invalid_answer_is_sent_back_as_repair_context() -> None:
    calls: list[str] = []
    answers = iter(
        [
            _feature("broken", "unknown_column + 1"),
            _feature("fixed", "a + 1"),
        ]
    )

    def llm(prompt: str) -> LLMResponse:
        calls.append(prompt)
        return _response(next(answers))

    generation = generate_features_iteratively(
        ["a", "b"],
        [{"a": 1, "b": 2}],
        llm,
        max_features=1,
        numeric_columns=["a", "b"],
        max_repairs=2,
    )

    assert len(calls) == 2
    assert "The previous answer was rejected" in calls[1]
    assert "Previous invalid answer" in calls[1]
    assert "unknown_column" in calls[1]
    assert [proposal.name for proposal in generation.plan.proposals] == ["fixed"]
    assert [attempt.valid for attempt in generation.attempts] == [False, True]
    assert generation.to_dict()["repair_calls"] == 1


def test_duplicate_expression_is_rejected_and_repaired() -> None:
    calls: list[str] = []
    answers = iter(
        [
            _feature("first", "a + 1"),
            _feature("duplicate", "a + 1"),
            _feature("second", "a + 2"),
        ]
    )

    def llm(prompt: str) -> LLMResponse:
        calls.append(prompt)
        return _response(next(answers))

    generation = generate_features_iteratively(
        ["a", "b"],
        [{"a": 1, "b": 2}],
        llm,
        max_features=2,
        numeric_columns=["a", "b"],
        max_repairs=2,
    )

    assert len(calls) == 3
    assert "Duplicate feature expression" in calls[2]
    assert [proposal.name for proposal in generation.plan.proposals] == [
        "first",
        "second",
    ]
    assert [proposal.expression for proposal in generation.plan.proposals] == [
        "a + 1",
        "a + 2",
    ]
    assert [attempt.valid for attempt in generation.attempts] == [True, False, True]


def test_no_valid_answer_raises_with_repair_hint() -> None:
    calls: list[str] = []

    def llm(prompt: str) -> LLMResponse:
        calls.append(prompt)
        return _response("not json at all")

    with pytest.raises(KconvError, match="no valid feature proposals"):
        generate_features_iteratively(
            ["a", "b"],
            [{"a": 1, "b": 2}],
            llm,
            max_features=1,
            numeric_columns=[],
            max_repairs=0,
            max_consecutive_failures=1,
        )
    assert len(calls) == 1


def test_single_argument_row_function_is_repaired_with_arity_guidance() -> None:
    calls: list[str] = []
    answers = iter(
        [
            _feature("bad_row_mean", "row_mean(a)"),
            _feature("good_sum", "a + b"),
        ]
    )

    def llm(prompt: str) -> LLMResponse:
        calls.append(prompt)
        return _response(next(answers))

    generation = generate_features_iteratively(
        ["a", "b"],
        [{"a": 1, "b": 2}],
        llm,
        max_features=1,
        numeric_columns=["a", "b"],
        max_repairs=2,
    )

    assert len(calls) == 2
    assert "row_mean(a, b, ...)" in calls[0]
    assert "at least two numeric columns" in calls[1]
    assert "row_mean requires at least two arguments" in calls[1]
    assert "Do not copy the rejected expression" in calls[1]
    assert [proposal.name for proposal in generation.plan.proposals] == ["good_sum"]
    assert [attempt.valid for attempt in generation.attempts] == [False, True]


def test_prompt_includes_novel_candidate_formulas() -> None:
    prompt = build_single_feature_prompt(
        ["a", "b", "c"],
        [{"a": 1, "b": 2, "c": 3}],
        numeric_columns=["a", "b", "c"],
        candidates=["a + b", "a * c"],
    )
    assert "Novel candidate formulas" in prompt
    assert "- a + b" in prompt
    assert "- a * c" in prompt


def test_duplicate_expression_uses_unique_deterministic_fallback() -> None:
    calls: list[str] = []

    def llm(prompt: str) -> LLMResponse:
        calls.append(prompt)
        return _response(_feature("same", "a + b"))

    generation = generate_features_iteratively(
        ["a", "b", "c", "d"],
        [{"a": 1, "b": 2, "c": 3, "d": 4}],
        llm,
        max_features=3,
        numeric_columns=["a", "b", "c", "d"],
        max_repairs=0,
        max_consecutive_failures=1,
    )

    expressions = [proposal.expression for proposal in generation.plan.proposals]
    assert len(expressions) == 3
    assert len(set(expressions)) == 3
    assert expressions[0] == "a + b"
    assert expressions[1:] == ["a - b", "a * b"]
    assert [proposal.name for proposal in generation.plan.proposals] == [
        "same",
        "llm_fallback_2",
        "llm_fallback_3",
    ]
    assert [attempt.valid for attempt in generation.attempts] == [
        True,
        False,
        True,
        False,
        True,
    ]
    warnings = generation.plan.warnings
    assert not any("giving up" in warning for warning in warnings)
    assert sum("deterministic fallback" in warning for warning in warnings) == 1
    assert any("2 feature(s) filled by deterministic fallback" in warning for warning in warnings)


def test_rename_warnings_are_aggregated() -> None:
    def llm(prompt: str) -> LLMResponse:
        return _response(_feature("Bad Name", "a + b"))

    generation = generate_features_iteratively(
        ["a", "b"],
        [{"a": 1, "b": 2}],
        llm,
        max_features=1,
        numeric_columns=["a", "b"],
        max_repairs=0,
    )

    assert [proposal.name for proposal in generation.plan.proposals] == ["bad_name"]
    warnings = generation.plan.warnings
    assert not any("renamed feature" in warning for warning in warnings)
    assert any("1 feature name(s) normalized to safe snake_case" in warning for warning in warnings)
