"""Real opt-in LLM provider: request shape, token accounting and privacy."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kconv.errors import KconvError
from kconv.llm_features import FeaturePlan, FeatureProposal
from kconv.llm_provider import (
    LLMResponse,
    OpenAICompatibleProvider,
    TokenUsage,
    build_feature_log,
    normalize_local_base_url,
    resolve_api_key,
    write_feature_log,
)


def _answer(*, usage: dict | None = None) -> bytes:
    payload = {
        "id": "req-123",
        "model": "fake-model-2026",
        "choices": [
            {
                "message": {"content": '{"features":[{"name":"sum_ab","kind":"numeric","expression":"a + b","rationale":"add"}]}'},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload).encode("utf-8")


def test_provider_sends_exact_request_and_counts_tokens() -> None:
    captured: list[tuple[str, dict, float]] = []

    def transport(request: urllib.request.Request, timeout: float) -> bytes:
        captured.append((request.full_url, request.headers, timeout))
        assert request.get_method() == "POST"
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "fake-model"
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 64
        assert body["seed"] == 7
        assert body["response_format"] == {"type": "json_object"}
        assert body["messages"][0]["role"] == "system"
        assert "Only use these columns" in body["messages"][1]["content"]
        return _answer(usage={"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17})

    client = OpenAICompatibleProvider(
        api_key="secret-key",
        model="fake-model",
        base_url="https://example.test/v1/",
        max_tokens=64,
        seed=7,
        transport=transport,
    )
    response = client.complete("Only use these columns: ['a']")
    assert captured[0][0] == "https://example.test/v1/chat/completions"
    assert captured[0][1]["Authorization"] == "Bearer secret-key"
    assert captured[0][2] == 60.0
    assert response.usage == TokenUsage(12, 5, 17)
    assert response.usage.available
    assert response.request_id == "req-123"
    assert response.finish_reason == "stop"
    assert response.model == "fake-model-2026"
    assert response.to_dict()["usage"]["total_tokens"] == 17


def test_provider_keeps_missing_usage_as_none() -> None:
    client = OpenAICompatibleProvider(
        api_key="secret",
        model="fake",
        transport=lambda _request, _timeout: _answer(),
    )
    response = client.complete("prompt")
    assert response.usage == TokenUsage()
    assert not response.usage.available
    assert response.to_dict()["usage"]["total_tokens"] is None


def test_provider_rejects_invalid_json_and_empty_content() -> None:
    client = OpenAICompatibleProvider(api_key="secret", model="fake", transport=lambda *_: b"not json")
    with pytest.raises(KconvError, match="not valid JSON"):
        client.complete("prompt")
    empty = {"choices": [{"message": {"content": " "}}]}
    client = OpenAICompatibleProvider(
        api_key="secret",
        model="fake",
        transport=lambda *_: json.dumps(empty).encode(),
    )
    with pytest.raises(KconvError, match="empty message"):
        client.complete("prompt")


def test_http_error_is_wrapped_and_never_contains_the_key(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://example.test",
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"invalid token"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    client = OpenAICompatibleProvider(api_key="secret-key", model="fake")
    with pytest.raises(KconvError) as exc:
        client.complete("prompt")
    text = f"{exc.value.message} {exc.value.hint}"
    assert "HTTP 401" in text
    assert "secret-key" not in text


def test_resolve_api_key_supports_primary_and_openai_fallback() -> None:
    assert resolve_api_key("MY_KEY", environ={"MY_KEY": "abc"}) == "abc"
    assert resolve_api_key("KCONV_LLM_API_KEY", environ={"OPENAI_API_KEY": "fallback"}) == "fallback"
    with pytest.raises(KconvError, match="missing"):
        resolve_api_key("MY_KEY", environ={})
    with pytest.raises(KconvError, match="custom base URL"):
        resolve_api_key(
            "KCONV_LLM_API_KEY",
            environ={"OPENAI_API_KEY": "must-not-leak"},
            base_url="https://third-party.example/v1",
        )


def test_resolve_api_key_allows_explicit_local_loopback_only() -> None:
    assert (
        resolve_api_key(
            "KCONV_LLM_API_KEY",
            environ={},
            base_url="http://127.0.0.1:11434/v1",
            allow_local_no_key=True,
        )
        == "local-no-key"
    )
    assert (
        resolve_api_key(
            "KCONV_LLM_API_KEY",
            environ={},
            base_url="http://localhost:8000/v1",
            allow_local_no_key=True,
        )
        == "local-no-key"
    )
    with pytest.raises(KconvError, match="missing"):
        resolve_api_key(
            "KCONV_LLM_API_KEY",
            environ={},
            base_url="https://api.example.com/v1",
            allow_local_no_key=True,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080/v1"),
        ("http://127.0.0.1:8080/", "http://127.0.0.1:8080/v1"),
        ("http://127.0.0.1:8080/v1/", "http://127.0.0.1:8080/v1"),
        (
            "http://127.0.0.1:8080/v1/chat/completions",
            "http://127.0.0.1:8080/v1",
        ),
        ("http://[::1]:8000", "http://[::1]:8000/v1"),
    ],
)
def test_normalize_local_base_url_accepts_common_user_input(raw: str, expected: str) -> None:
    assert normalize_local_base_url(raw) == expected


def test_feature_log_has_usage_hashes_and_no_prompt_or_key(tmp_path: Path) -> None:
    plan = FeaturePlan(
        proposals=(FeatureProposal(name="sum_ab", kind="numeric", expression="a + b", rationale="add"),),
        warnings=("one warning",),
    )
    prompt = "Only use these columns: ['a', 'b']\nSample rows: ..."
    response = LLMResponse(
        text="{}",
        provider="openai-compatible",
        model="fake",
        usage=TokenUsage(10, 4, 14),
        duration_sec=0.25,
        request_id="req",
        finish_reason="stop",
    )
    metadata = build_feature_log(
        plan,
        provider=response.provider,
        model=response.model,
        prompt=prompt,
        usage=response.usage,
        duration_sec=response.duration_sec,
        request_id=response.request_id,
        finish_reason=response.finish_reason,
        sample_rows=5,
        max_features=5,
        approved=True,
        temperature=0.0,
        seed=3,
        max_tokens=64,
        json_mode=False,
    )
    encoded = json.dumps(metadata)
    assert metadata["usage"]["total_tokens"] == 14
    assert metadata["max_tokens"] == 64
    assert metadata["json_mode"] is False
    assert metadata["usage_available"] is True
    assert metadata["prompt_logged"] is False
    assert metadata["prompt_sha256"] and metadata["prompt_sha256"] != prompt
    assert prompt not in encoded
    assert "secret-key" not in encoded
    assert metadata["proposals"][0]["name"] == "sum_ab"
    path = write_feature_log(tmp_path, metadata)
    assert path == tmp_path / "05_report" / "llm_features.json"
    assert json.loads(path.read_text(encoding="utf-8"))["approved"] is True
