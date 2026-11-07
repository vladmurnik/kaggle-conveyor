"""CLI wiring for the opt-in real provider and the 05_report audit log."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from kconv.cli import app
from kconv.llm_provider import LLMResponse, OpenAICompatibleProvider, TokenUsage
from kconv.state import read_latest_pointer


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    n = 64
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": (rng.random(n) > 0.5).astype(int),
        }
    )
    frame.iloc[:48].to_csv(root / "train.csv", index=False)
    frame.iloc[48:].drop(columns=["target"]).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[48:]["id"], "target": 0}).to_csv(root / "sample_submission.csv", index=False)
    return root


def test_openai_provider_is_opt_in_and_writes_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    output = tmp_path / "plan.json"

    def fake_complete(self, prompt):  # noqa: ANN001
        assert "Only use these columns" in prompt
        assert '"target":' not in prompt
        assert '"id":' not in prompt
        assert '"a":' in prompt and '"b":' in prompt
        return LLMResponse(
            text=json.dumps(
                {
                    "features": [
                        {"name": "openai_sum", "kind": "numeric", "expression": "a + b", "rationale": "fake"}
                    ]
                }
            ),
            provider="openai-compatible",
            model="fake-model",
            usage=TokenUsage(prompt_tokens=21, completion_tokens=8, total_tokens=29),
            duration_sec=0.05,
            request_id="req-fake",
            finish_reason="stop",
        )

    monkeypatch.setenv("KCONV_LLM_API_KEY", "super-secret-key")
    monkeypatch.setattr("kconv.llm_provider.OpenAICompatibleProvider.complete", fake_complete)
    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "openai",
            "--llm-model",
            "fake-model",
            "--output",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["features"][0]["name"] == "openai_sum"
    sidecar = json.loads((tmp_path / "plan.llm.json").read_text(encoding="utf-8"))
    assert sidecar["provider"] == "openai-compatible"
    assert sidecar["usage"]["total_tokens"] == 29
    assert sidecar["usage_available"] is True
    assert sidecar["prompt_logged"] is False
    assert "super-secret-key" not in json.dumps(sidecar)
    assert "Only use these columns" not in json.dumps(sidecar)


def test_openai_provider_without_key_fails_before_network(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    monkeypatch.delenv("KCONV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "openai",
            "--llm-model",
            "fake-model",
        ],
    )
    assert result.exit_code != 0
    assert "credential" in result.output.lower() or "missing" in result.output.lower()


def test_openai_custom_endpoint_requires_an_explicit_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    monkeypatch.delenv("KCONV_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "openai",
            "--llm-model",
            "fake-model",
            "--llm-base-url",
            "https://third-party.example/v1",
        ],
    )
    assert result.exit_code != 0
    assert "custom base url" in result.output.lower()


def test_run_with_llm_plan_writes_05_report_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    plan = tmp_path / "plan.json"
    runner = CliRunner()
    proposed = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "mock",
            "--output",
            str(plan),
            "--json",
        ],
    )
    assert proposed.exit_code == 0, proposed.output
    assert (tmp_path / "plan.llm.json").is_file()

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--experiment",
            "llm-feature-proposals",
            "--approve-llm-features",
            "--llm-plan-file",
            str(plan),
            "--models",
            "lgbm",
            "--name",
            "llm-log-smoke",
            "--until",
            "ingest",
        ],
    )
    assert result.exit_code == 0, result.output
    project = tmp_path / "competitions" / "llm-log-smoke"
    run_id = read_latest_pointer(project)
    assert run_id is not None
    log_path = project / "runs" / run_id / "05_report" / "llm_features.json"
    assert log_path.is_file()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["provider"] == "mock"
    assert log["approved"] is True
    assert log["usage_available"] is False
    assert log["proposals"]


def test_run_rejects_metadata_when_plan_was_edited(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    plan = tmp_path / "plan.json"
    runner = CliRunner()
    proposed = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "mock",
            "--output",
            str(plan),
            "--json",
        ],
    )
    assert proposed.exit_code == 0, proposed.output
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["features"][0]["expression"] = "a - b"
    plan.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--experiment",
            "llm-feature-proposals",
            "--approve-llm-features",
            "--llm-plan-file",
            str(plan),
            "--models",
            "lgbm",
            "--name",
            "stale-metadata",
            "--until",
            "ingest",
        ],
    )
    assert result.exit_code != 0
    assert "does not match" in result.output.lower()


def test_propose_rejects_output_log_and_apply_collisions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    runner = CliRunner()
    same = tmp_path / "same.json"
    result = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "mock",
            "--output",
            str(same),
            "--llm-log",
            str(same),
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert "same path" in result.output.lower()

    result = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "mock",
            "--output",
            str(same),
            "--apply",
            str(same),
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert "same path" in result.output.lower()


def test_local_iterative_provider_uses_per_feature_prompts_and_user_hints(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    output = tmp_path / "plan.json"
    code_file = tmp_path / "idea.py"
    code_file.write_text("def ratio(a, b):\n    return a / b\n", encoding="utf-8")
    prompts: list[str] = []

    def fake_complete(self, prompt):  # noqa: ANN001
        prompts.append(prompt)
        index = len(prompts)
        return LLMResponse(
            text=json.dumps(
                {
                    "features": [
                        {
                            "name": f"local_{index}",
                            "kind": "numeric",
                            "expression": f"a + {index}",
                            "rationale": "local iterative test",
                        }
                    ]
                }
            ),
            provider="openai-compatible",
            model="fake-model",
            usage=TokenUsage(21, 8, 29),
            duration_sec=0.1,
        )

    monkeypatch.setattr(OpenAICompatibleProvider, "complete", fake_complete)
    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "local",
            "--llm-base-url",
            "http://127.0.0.1:11434/v1",
            "--llm-model",
            "fake-model",
            "--max-features",
            "2",
            "--llm-idea",
            "try a ratio between a and b",
            "--llm-code-file",
            str(code_file),
            "--llm-repairs",
            "1",
            "--output",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert len(plan["features"]) == 2
    assert len(prompts) == 2
    assert "Feature 1 of at most 2" in prompts[0]
    assert "Feature 2 of at most 2" in prompts[1]
    assert "try a ratio between a and b" in prompts[0]
    assert "do NOT execute" in prompts[0]
    assert "def ratio" in prompts[0]

    sidecar = json.loads((tmp_path / "plan.llm.json").read_text(encoding="utf-8"))
    generation = sidecar["generation"]
    assert generation["mode"] == "one-feature-per-call"
    assert generation["calls"] == 2
    assert generation["features"] == 2
    assert generation["repair_calls"] == 0
    assert sidecar["usage"]["total_tokens"] == 58
