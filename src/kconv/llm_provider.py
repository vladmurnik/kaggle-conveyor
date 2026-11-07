"""Opt-in real LLM provider with explicit token accounting.

The provider talks to any OpenAI-compatible ``/chat/completions`` endpoint
using only the standard library.  It is never used unless the CLI receives
``--provider openai``/``local``; the default mock/plan paths remain offline.
Local loopback servers may explicitly run without an API key.

Security and privacy
--------------------
* The API key is read from an environment variable and never logged.
* Only the prompt built by :func:`kconv.llm_features.build_prompt` is sent:
  the schema plus at most five sample rows.
* The generated ``05_report/llm_features.json`` stores hashes and token
  counts, not the prompt text or the key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kconv.errors import KconvError
from kconv.llm_features import FeaturePlan

Transport = Callable[[urllib.request.Request, float], bytes | str]
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
# Never send loopback prompts through a corporate/system HTTP proxy.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_DEFAULT_KEY_ENV = "KCONV_LLM_API_KEY"


def _is_loopback(base_url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(base_url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }


_LOCAL_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("vLLM", "http://127.0.0.1:8000/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
)


@dataclass(frozen=True)
class LocalModelEndpoint:
    """A reachable loopback OpenAI-compatible endpoint."""

    runtime: str
    base_url: str
    models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "base_url": self.base_url,
            "models": list(self.models),
        }


def normalize_local_base_url(base_url: str) -> str:
    """Normalize a local base URL to the ``.../v1`` API root.

    llama.cpp and Ollama users naturally paste ``http://127.0.0.1:8080`` or
    ``http://127.0.0.1:11434``.  The OpenAI-compatible endpoint lives under
    ``/v1``; adding it here avoids a confusing 404.  Explicit paths are kept.
    """
    value = base_url.strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")].rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme in {"http", "https"} and parsed.hostname and (
        not parsed.path or parsed.path == "/"
    ):
        value = f"{value}/v1"
    return value


def discover_local_models(base_url: str, *, timeout: float = 1.5) -> tuple[str, ...]:
    """Return model ids exposed by a loopback OpenAI-compatible server.

    The function is read-only and never calls ``/chat/completions``.  A missing
    server, a non-JSON response, or an unsupported ``/models`` route simply
    yields an empty tuple; callers can still use an explicitly supplied model.
    """
    base = normalize_local_base_url(base_url)
    try:
        parsed = urllib.parse.urlsplit(base)
    except ValueError:
        return ()
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        return ()
    request = urllib.request.Request(
        f"{base}/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with _LOCAL_OPENER.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, Mapping):
        return ()
    items = payload.get("data")
    if not isinstance(items, list):
        items = payload.get("models")
    if not isinstance(items, list):
        return ()
    models: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = item.get("id") or item.get("name") or item.get("model")
        if isinstance(name, str) and name.strip() and name.strip() not in models:
            models.append(name.strip())
    return tuple(models)


def discover_local_endpoints(
    *,
    timeout: float = 1.5,
    candidates: Iterable[tuple[str, str]] | None = None,
) -> tuple[LocalModelEndpoint, ...]:
    """Probe the common local runtimes and return those with a model list."""
    candidates_list = [
        (runtime, normalize_local_base_url(base_url))
        for runtime, base_url in (candidates if candidates is not None else _LOCAL_ENDPOINTS)
    ]
    if not candidates_list:
        return ()
    found: list[LocalModelEndpoint] = []
    with ThreadPoolExecutor(max_workers=min(8, len(candidates_list))) as pool:
        pending = [
            (runtime, normalized, pool.submit(discover_local_models, normalized, timeout=timeout))
            for runtime, normalized in candidates_list
        ]
        for runtime, normalized, future in pending:
            try:
                models = future.result()
            except Exception:  # noqa: BLE001 - discovery must never break the page
                models = ()
            if models:
                found.append(LocalModelEndpoint(runtime=runtime, base_url=normalized, models=models))
    return tuple(found)


def resolve_local_target(
    base_url: str | None = None,
    model: str | None = None,
    *,
    timeout: float = 1.5,
) -> tuple[str, str]:
    """Resolve a local endpoint/model pair for llama.cpp, Ollama, or vLLM.

    An explicit base URL is honored.  When it is omitted, the common loopback
    ports are probed; when the model is omitted, ``GET /v1/models`` supplies the
    loaded model id.  This is what lets ``--provider local`` work with
    llama.cpp without passing Ollama-specific defaults.
    """
    selected_model = (model or "").strip()
    selected_base = normalize_local_base_url(base_url) if base_url else ""
    models: tuple[str, ...] = ()
    if not selected_base:
        endpoints = discover_local_endpoints(timeout=timeout)
        if not endpoints:
            raise KconvError(
                "Local LLM server was not found.",
                "Start llama.cpp on http://127.0.0.1:8080/v1, Ollama on :11434, "
                "or vLLM on :8000; alternatively pass --llm-base-url.",
            )
        selected_base = endpoints[0].base_url
        models = endpoints[0].models
    elif not selected_model:
        models = discover_local_models(selected_base, timeout=timeout)
    if not _is_loopback(selected_base):
        raise KconvError(
            "Local LLM base URL must be a loopback HTTP(S) endpoint.",
            "Use http://127.0.0.1:8080/v1, http://localhost:11434/v1 or http://[::1]:8000/v1.",
        )
    if not selected_model and models:
        selected_model = models[0]
    if not selected_model:
        raise KconvError(
            "Local LLM model was not detected.",
            "Pass --llm-model explicitly or check that GET <base-url>/models "
            "returns the loaded llama.cpp/Ollama/vLLM model.",
        )
    return selected_base, selected_model


def resolve_api_key(
    env_name: str = _DEFAULT_KEY_ENV,
    *,
    environ: Mapping[str, str] | None = None,
    base_url: str = _DEFAULT_BASE_URL,
    allow_local_no_key: bool = False,
) -> str:
    """Return a non-empty API key or raise a safe error.

    ``KCONV_LLM_API_KEY`` is the primary opt-in variable; ``OPENAI_API_KEY`` is
    accepted as a convenience only for the official OpenAI endpoint.  Any other
    base URL must use an explicit ``KCONV_LLM_API_KEY``/``--llm-api-key-env`` so
    an OpenAI credential is never forwarded to a third-party endpoint.  Local
    loopback servers (Ollama, llama.cpp, vLLM) may explicitly opt out of a key.
    """
    env = os.environ if environ is None else environ
    key = env.get(env_name, "").strip()
    source = env_name
    official = base_url.strip().rstrip("/").casefold() == _DEFAULT_BASE_URL.casefold()
    if allow_local_no_key and _is_loopback(base_url):
        return key or "local-no-key"
    if not key and env_name == _DEFAULT_KEY_ENV and official:
        key = env.get("OPENAI_API_KEY", "").strip()
        source = "OPENAI_API_KEY"
    if not key:
        hint = "Keys are never written to disk."
        if env_name == _DEFAULT_KEY_ENV and not official:
            hint = (
                "OPENAI_API_KEY is only accepted for the official OpenAI endpoint; "
                "set KCONV_LLM_API_KEY or --llm-api-key-env for a custom base URL. "
                "Keys are never written to disk."
            )
        raise KconvError(f"LLM credential is missing: {source} is not set.", hint)
    return key


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the provider; missing values stay ``None``."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def available(self) -> bool:
        return any(value is not None for value in (self.prompt_tokens, self.completion_tokens, self.total_tokens))

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMResponse:
    """One provider answer plus the metadata needed for accounting."""

    text: str
    provider: str
    model: str
    usage: TokenUsage
    duration_sec: float
    request_id: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "usage_available": self.usage.available,
            "duration_sec": round(self.duration_sec, 3),
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
        }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _usage_from_payload(raw: Any) -> TokenUsage:
    if not isinstance(raw, Mapping):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=_as_int(raw.get("prompt_tokens")),
        completion_tokens=_as_int(raw.get("completion_tokens")),
        total_tokens=_as_int(raw.get("total_tokens")),
    )


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat-completions client.

    Parameters
    ----------
    api_key:
        Secret; intentionally excluded from ``repr`` by not being a dataclass
        field and never included in errors.
    transport:
        Injectable callable used by tests.  It receives a prepared
        :class:`urllib.request.Request` and timeout and must return bytes/text.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        json_mode: bool = True,
        transport: Transport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not api_key.strip():
            raise KconvError("LLM API key must not be empty.")
        if not model.strip():
            raise KconvError("LLM model must not be empty.")
        if timeout <= 0:
            raise KconvError("LLM timeout must be positive.")
        if max_tokens is not None and max_tokens < 1:
            raise KconvError("LLM max_tokens must be at least 1.")
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = float(timeout)
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.seed = seed
        self.json_mode = bool(json_mode)
        self._transport = transport or self._post
        self._extra_headers = dict(extra_headers or {})

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

    def _payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only a JSON object that follows the user's schema. Do not include Markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, request: urllib.request.Request, timeout: float) -> bytes:
        opener = _LOCAL_OPENER.open if _is_loopback(self.base_url) else urllib.request.urlopen
        try:
            with opener(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # The response body is deliberately not included: a compatible
            # endpoint may echo the request, which contains sample rows.
            raise KconvError(
                f"LLM request failed (HTTP {exc.code}).",
                "Check the endpoint/model/key and the provider's JSON-mode support. "
                "The key, request body and provider response body are not logged.",
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise KconvError(
                f"Could not reach the LLM endpoint: {type(exc).__name__}.",
                "Check network access, the base URL and the timeout; no request body or key is logged.",
            ) from exc

    def complete(self, prompt: str) -> LLMResponse:
        """Send one prompt and return text plus usage metadata."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise KconvError("LLM prompt must be a non-empty string.")
        body = json.dumps(self._payload(prompt), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        started = time.perf_counter()
        raw = self._transport(request, self.timeout)
        duration = time.perf_counter() - started
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise KconvError("LLM response was not UTF-8.", "Check the provider endpoint.") from exc
        else:
            text = str(raw)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KconvError(
                "LLM response is not valid JSON.",
                "Ask the provider for JSON mode or use --no-llm-json-mode only when the model still returns JSON.",
            ) from exc
        if not isinstance(payload, Mapping):
            raise KconvError("LLM response JSON root must be an object.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise KconvError("LLM response has no choices[0].", "Check the provider compatibility.")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise KconvError("LLM response has empty message content.", "Check the provider/model response.")
        usage = _usage_from_payload(payload.get("usage"))
        return LLMResponse(
            text=content,
            provider=self.provider_name,
            model=str(payload.get("model") or self.model),
            usage=usage,
            duration_sec=duration,
            request_id=str(payload["id"]) if payload.get("id") else None,
            finish_reason=str(choices[0]["finish_reason"]) if choices[0].get("finish_reason") else None,
        )


def build_feature_log(
    plan: FeaturePlan,
    *,
    provider: str,
    model: str | None,
    prompt: str | None,
    usage: TokenUsage | None = None,
    duration_sec: float | None = None,
    request_id: str | None = None,
    finish_reason: str | None = None,
    sample_rows: int | None = None,
    max_features: int | None = None,
    plan_file: str | Path | None = None,
    approved: bool = True,
    temperature: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
    json_mode: bool | None = None,
) -> dict[str, Any]:
    """Build the privacy-safe ``05_report/llm_features.json`` payload."""
    plan_path = Path(plan_file) if plan_file is not None else None
    plan_hash = None
    if plan_path is not None and plan_path.is_file():
        plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt is not None else None
    token_usage = usage or TokenUsage()
    return {
        "schema_version": 1,
        "experiment": "llm-feature-proposals",
        "approved": bool(approved),
        "approval_required": True,
        "provider": provider,
        "model": model,
        "request_id": request_id,
        "finish_reason": finish_reason,
        "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
        "prompt_sha256": prompt_hash,
        "prompt_chars": len(prompt) if prompt is not None else None,
        "prompt_logged": False,
        "sample_rows": sample_rows,
        "max_features": max_features,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        "temperature": temperature,
        "seed": seed,
        "usage": token_usage.to_dict(),
        "usage_available": token_usage.available,
        "plan_file": str(plan_path) if plan_path is not None else None,
        "plan_sha256": plan_hash,
        "proposals": [proposal.to_dict() for proposal in plan.proposals],
        "warnings": list(plan.warnings),
        "generated_at": datetime.now(UTC).isoformat(),
    }


def write_feature_log(run_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write the LLM metadata inside ``05_report``."""
    from kconv.context import write_json

    path = Path(run_dir) / "05_report" / "llm_features.json"
    write_json(path, dict(payload))
    return path


__all__ = [
    "LLMResponse",
    "LocalModelEndpoint",
    "OpenAICompatibleProvider",
    "TokenUsage",
    "build_feature_log",
    "discover_local_endpoints",
    "discover_local_models",
    "normalize_local_base_url",
    "resolve_api_key",
    "resolve_local_target",
    "write_feature_log",
]
