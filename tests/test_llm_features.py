"""LLM feature proposal parsing is strict and never executes expressions."""

from __future__ import annotations

import json

import pytest

from kconv.errors import KconvError
from kconv.llm_features import build_prompt, parse_plan, propose_features


def test_build_prompt_mentions_columns_and_json_contract() -> None:
    prompt = build_prompt({"columns": ["a", "b"]}, [{"a": 1, "b": 2}], max_features=3)
    assert "Return ONLY valid JSON" in prompt
    assert "a" in prompt and "b" in prompt
    assert "at most 3" in prompt


def test_build_prompt_filters_target_id_and_extra_keys() -> None:
    prompt = build_prompt(
        {"columns": ["a", "b"]},
        [{"a": 1, "b": 2, "target": 98765, "id": "row-secret", "email": "x@example.com"}],
    )
    assert '"a": 1' in prompt and '"b": 2' in prompt
    assert '"target":' not in prompt
    assert '"id":' not in prompt
    assert "98765" not in prompt
    assert "row-secret" not in prompt
    assert "x@example.com" not in prompt


def test_parse_plan_accepts_simple_arithmetic_and_reports_warnings() -> None:
    raw = json.dumps(
        {
            "features": [
                {
                    "name": "ratio_ab",
                    "kind": "numeric",
                    "expression": "a / b",
                    "rationale": "relative scale",
                },
                {
                    "name": "bad",
                    "kind": "numeric",
                    "expression": "target / b",
                    "rationale": "leaky",
                },
            ]
        }
    )
    plan = parse_plan(raw, columns={"a", "b"})
    assert [item.name for item in plan.proposals] == ["ratio_ab"]
    assert any("unknown columns" in warning for warning in plan.warnings)


def test_parse_plan_extracts_json_from_local_model_prose() -> None:
    raw = (
        "Sure, here is the plan:\n```json\n"
        '{"features":[{"name":"sum_ab","kind":"numeric","expression":"a + b","rationale":"add"}]}'
        "\n```\nDone."
    )
    plan = parse_plan(raw, columns={"a", "b"})
    assert plan.proposals[0].name == "sum_ab"
    assert plan.proposals[0].expression == "a + b"


def test_parse_plan_normalizes_spaced_and_punctuated_column_names() -> None:
    raw = json.dumps(
        {
            "features": [
                {
                    "name": "Gender Ratio",
                    "kind": "categorical",
                    "expression": "group_count(Gender)",
                    "rationale": "group size",
                },
                {
                    "name": "Loyalty Discount",
                    "kind": "numeric",
                    "expression": "group_mean(Cleanliness, Customer Type)",
                    "rationale": "loyalty signal",
                },
                {
                    "name": "Travel Mean",
                    "kind": "numeric",
                    "expression": "row_mean(Flight Distance, Departure/Arrival time convenient)",
                    "rationale": "row average",
                },
            ]
        }
    )
    plan = parse_plan(
        raw,
        columns=[
            "Gender",
            "Cleanliness",
            "Customer Type",
            "Flight Distance",
            "Departure/Arrival time convenient",
        ],
    )
    expressions = {proposal.name: proposal.expression for proposal in plan.proposals}
    assert expressions["gender_ratio"] == "group_count(Gender)"
    assert expressions["loyalty_discount"] == (
        'group_mean(Cleanliness, col("Customer Type"))'
    )
    assert expressions["travel_mean"] == (
        'row_mean(col("Flight Distance"), col("Departure/Arrival time convenient"))'
    )


def test_parse_plan_rejects_target_derived_proposals_and_reserves_names() -> None:
    raw = json.dumps(
        {
            "features": [
                {
                    "name": "Customer Satisfaction",
                    "kind": "numeric",
                    "expression": "group_mean(Satisfaction, Customer Type)",
                    "rationale": "leaky",
                },
                {
                    "name": "satisfaction",
                    "kind": "categorical",
                    "expression": "group_count(Gender)",
                    "rationale": "reserved name",
                },
            ]
        }
    )
    plan = parse_plan(
        raw,
        columns=["Gender", "Customer Type"],
        reserved=["satisfaction", "id"],
    )
    assert [proposal.name for proposal in plan.proposals] == ["satisfaction_2"]
    assert any("unknown columns" in warning for warning in plan.warnings)
    assert any("renamed feature" in warning for warning in plan.warnings)


def test_parse_plan_rejects_categorical_columns_in_numeric_positions() -> None:
    raw = json.dumps(
        {
            "features": [
                {
                    "name": "bad_row",
                    "kind": "numeric",
                    "expression": "row_mean(num, cat)",
                    "rationale": "categorical value",
                },
                {
                    "name": "bad_group",
                    "kind": "numeric",
                    "expression": "group_mean(cat, grp)",
                    "rationale": "categorical value",
                },
                {
                    "name": "bad_arithmetic",
                    "kind": "numeric",
                    "expression": "num + cat",
                    "rationale": "categorical operand",
                },
                {
                    "name": "good_group",
                    "kind": "numeric",
                    "expression": "group_mean(num, grp)",
                    "rationale": "numeric value, categorical key",
                },
            ]
        }
    )
    plan = parse_plan(
        raw,
        columns=["num", "cat", "grp"],
        numeric_columns=["num"],
    )
    assert [proposal.name for proposal in plan.proposals] == ["good_group"]
    assert sum("is not numeric" in warning for warning in plan.warnings) == 3


def test_parse_plan_rejects_invalid_payloads() -> None:
    with pytest.raises(KconvError):
        parse_plan("not json")
    with pytest.raises(KconvError):
        parse_plan(json.dumps({"features": []}))
    with pytest.raises(KconvError):
        parse_plan(
            json.dumps(
                {
                    "features": [
                        {"name": "x", "kind": "numeric", "expression": "missing + 1", "rationale": ""}
                    ]
                }
            ),
            columns={"a"},
        )


def test_propose_features_uses_injected_llm_without_network() -> None:
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(
            {
                "features": [
                    {
                        "name": "sum_ab",
                        "kind": "numeric",
                        "expression": "a + b",
                        "rationale": "combined signal",
                    }
                ]
            }
        )

    plan = propose_features({"columns": ["a", "b"]}, [{"a": 1, "b": 2}], fake_llm)
    assert len(calls) == 1
    assert plan.proposals[0].name == "sum_ab"
    assert plan.to_dict()["proposals"][0]["expression"] == "a + b"
