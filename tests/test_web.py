"""Integration tests for the local browser UI.

These tests start the real standard-library server and, for the end-to-end
case, the real ``kconv run`` CLI subprocess.  They intentionally use a tiny
dataset so they stay fast while still exercising uploads, JSON endpoints and
the generated submission/report.
"""

from __future__ import annotations

import http.client
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest

from kconv.kaggle.auth import AuthStatus
from kconv.llm_provider import LocalModelEndpoint, discover_local_endpoints
from kconv.state import write_latest_pointer
from kconv.web import (
    MAX_UPLOAD_BYTES,
    KconvWebServer,
    ManagedRun,
    QueueItem,
    RunManager,
    WebApp,
    WebError,
    _llm_web_error,
    create_server,
)


@pytest.fixture(scope="module")
def web_server(tmp_path_factory: pytest.TempPathFactory) -> tuple[KconvWebServer, str, Path]:
    root = tmp_path_factory.mktemp("web-root")
    data = root / "data"
    server = create_server(
        root=root,
        data_dir=data,
        port=0,
        kaggle_token_path=root / "kaggle-access-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="kconv-web-test")
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server, base, data
    finally:
        server.shutdown()
        server.server_close()
        server.app.shutdown()
        thread.join(timeout=5)


def _get(base: str, path: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(base + path, timeout=30) as response:  # noqa: S310 - local test server
        return response.status, response.read()


def _json_get(base: str, path: str) -> tuple[int, dict]:
    status, body = _get(base, path)
    return status, json.loads(body.decode("utf-8"))


def _session_token(base: str) -> str:
    status, payload = _json_get(base, "/api/session")
    assert status == 200
    token = payload["token"]
    assert isinstance(token, str) and token
    return token


def _post(
    base: str,
    path: str,
    body: bytes,
    content_type: str = "application/octet-stream",
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    request_headers = {"Content-Type": content_type}
    if headers:
        request_headers.update(headers)
    if token is None:
        token = _session_token(base)
    if token:
        request_headers["X-Kconv-Token"] = token
    request = urllib.request.Request(
        base + path,
        data=body,
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - local test server
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(base: str, path: str, payload: dict, **kwargs: object) -> tuple[int, dict]:
    return _post(base, path, json.dumps(payload).encode("utf-8"), "application/json", **kwargs)


def _post_error(
    base: str,
    path: str,
    body: bytes = b"",
    content_type: str = "application/json",
    **kwargs: object,
) -> tuple[int, dict]:
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post(base, path, body, content_type, **kwargs)
    return exc_info.value.code, json.loads(exc_info.value.read().decode("utf-8"))


def _sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    rows = 90
    train = pd.DataFrame(
        {
            "id": np.arange(rows),
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "category": rng.choice(["a", "b", "c"], size=rows),
        }
    )
    train["target"] = (
        train["x1"] + (train["category"] == "a") * 0.8 + rng.normal(scale=0.25, size=rows) > 0.2
    ).astype(int)
    test = train.drop(columns=["target"]).head(12).copy()
    return train, test


def _upload_csv(base: str, kind: str, name: str, frame: pd.DataFrame) -> dict:
    status, payload = _post(base, f"/api/upload?kind={kind}&name={name}", frame.to_csv(index=False).encode("utf-8"))
    assert status == 200, payload
    return payload


class _FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
    """Tiny llama.cpp-like OpenAI endpoint used by the local-provider test."""

    calls: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "fake-llama", "object": "model"}]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        type(self).calls.append(body)
        if body.get("response_format"):
            # Simulate an older llama.cpp build that rejects JSON mode.
            self._json(400, {"error": "response_format is not supported"})
            return
        successful = sum(
            1 for call in type(self).calls if not call.get("response_format")
        )
        if successful == 1:
            name, expression = "llm_x1_x2_sum", "x1 + x2"
        else:
            name, expression = "llm_x1_x2_diff", "x1 - x2"
        content = (
            "Here is the plan:\n```json\n"
            + json.dumps(
                {
                    "features": [
                        {
                            "name": name,
                            "kind": "numeric",
                            "expression": expression,
                            "rationale": "fake llama.cpp",
                        }
                    ]
                }
            )
            + "\n```"
        )
        self._json(
            200,
            {
                "id": "fake-llama-request",
                "model": "fake-llama",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )


@pytest.fixture
def fake_llama() -> str:
    _FakeLlamaHandler.calls = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlamaHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="fake-llama")
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_session_token_is_returned_and_not_in_config(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, _data = web_server

    status, session = _json_get(base, "/api/session")
    assert status == 200
    assert session["token"] == server.app.csrf_token
    assert len(session["token"]) >= 32

    status, config = _json_get(base, "/api/config")
    assert status == 200
    assert session["token"] not in json.dumps(config, ensure_ascii=False)


def test_post_requires_valid_csrf_token(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    _server, base, _data = web_server

    status, payload = _post_error(base, "/api/run/stop", token="")
    assert status == 403
    assert "CSRF" in payload["message"]

    status, _payload = _post_error(base, "/api/run/stop", token="wrong-token")
    assert status == 403

    status, payload = _post(base, "/api/run/stop", b"", "application/json")
    assert status == 200
    assert payload["status"] == "idle"

    # Upload is also a mutation and must not be reachable without the token.
    status, payload = _post_error(
        base,
        "/api/upload?kind=train&name=train.csv",
        b"id,target\n1,0\n",
        "application/octet-stream",
        token="",
    )
    assert status == 403
    assert "CSRF" in payload["message"]


def test_post_rejects_cross_site_origin_and_fetch_metadata(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    _server, base, _data = web_server

    status, _payload = _post_error(
        base,
        "/api/run/stop",
        headers={"Origin": "http://evil.example"},
    )
    assert status == 403

    status, _payload = _post_error(
        base,
        "/api/run/stop",
        headers={"Origin": "https://127.0.0.1:1"},
    )
    assert status == 403

    status, _payload = _post_error(
        base,
        "/api/run/stop",
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert status == 403

    status, payload = _post(
        base,
        "/api/run/stop",
        b"{}",
        "application/json",
        headers={"Origin": base, "Sec-Fetch-Site": "same-origin"},
    )
    assert status == 200
    assert payload["status"] == "idle"


def test_post_rejects_unknown_host_header(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    _server, base, _data = web_server

    status, payload = _post_error(
        base,
        "/api/run/stop",
        headers={"Host": "evil.example"},
    )
    assert status == 403
    assert "Host" in payload["message"]

    status, payload = _post_error(
        base,
        "/api/run/stop",
        headers={"Host": "127.0.0.1:1"},
    )
    assert status == 403


def test_json_post_requires_application_json_content_type(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    _server, base, _data = web_server

    status, payload = _post_error(
        base,
        "/api/run",
        b"{}",
        "text/plain",
    )
    assert status == 415
    assert "application/json" in payload["message"]

    # A charset suffix is valid JSON content type.
    status, payload = _post_error(
        base,
        "/api/run",
        b"{}",
        "application/json; charset=utf-8",
    )
    assert status == 400
    assert payload["message"]


def test_upload_rejects_oversized_body(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, _data = web_server
    port = int(server.server_address[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest(
            "POST",
            "/api/upload?kind=train&name=train.csv",
            skip_host=True,
            skip_accept_encoding=True,
        )
        connection.putheader("Host", f"127.0.0.1:{port}")
        connection.putheader("X-Kconv-Token", _session_token(base))
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(MAX_UPLOAD_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        assert response.status == 413
        assert "256 MiB" in body["message"]
    finally:
        connection.close()


def test_create_server_requires_explicit_remote_opt_in(tmp_path: Path) -> None:
    root = tmp_path / "remote-root"
    root.mkdir()
    with pytest.raises(WebError) as exc_info:
        create_server(root=root, data_dir=tmp_path / "data", host="0.0.0.0", port=0)
    assert "--allow-remote" in (exc_info.value.hint or "")
    assert exc_info.value.status == 400

    server = create_server(
        root=root,
        data_dir=tmp_path / "data",
        host="0.0.0.0",
        port=0,
        allow_remote=True,
    )
    try:
        assert server.allow_remote is True
        assert server.server_address[0] == "0.0.0.0"
    finally:
        server.server_close()
        server.app.shutdown()


def test_static_config_upload_and_detection(web_server: tuple[KconvWebServer, str, Path]) -> None:
    _server, base, data = web_server

    status, index = _get(base, "/")
    assert status == 200
    html = index.decode("utf-8")
    assert "kconv studio" in html
    assert 'id="recipe-apply"' not in html
    assert 'id="recipe-kind"' in html
    assert 'id="recipe-blocks"' in html
    assert 'id="recipe-derived"' in html
    assert 'id="llm-local-help"' in html
    assert 'id="strategy-help"' in html
    assert 'id="kaggle-card"' in html
    assert 'id="kaggle-token"' in html
    assert 'id="kaggle-username"' in html
    assert 'id="kaggle-key"' in html
    assert 'id="kaggle-submit"' in html
    assert 'id="device"' in html
    assert 'id="cpu-threads"' in html
    assert 'id="ram-gb"' in html
    assert 'id="vram-gb"' in html
    assert 'id="model-params-rows"' in html

    status, app_js = _get(base, "/static/app.js")
    assert status == 200
    script = app_js.decode("utf-8")
    assert "recipe-apply" not in script
    assert 'applyRecipeModels({ announce: true })' in script
    assert '$("#recipe").addEventListener("change", () => {' in script
    assert "pipeline — executable blocks" in script
    assert 'item.kind === "pipeline"' in script
    assert 'id="recipe-batch"' in html
    assert 'id="pipeline-preset"' in html
    assert 'id="pipeline-select-all"' in html
    assert 'id="pipeline-select-none"' in html
    assert "populatePipelinePresets" in script
    assert "applyPipelinePreset" in script
    assert "addSelectedPipelines" in script
    assert 'id="comparison-dataset"' in html
    assert "renderComparisonDatasets" in script
    assert "renderModelParams" in script
    assert "variantResources" in script
    assert "parseModelParams" in script
    assert "resourceConfigError" in script
    assert 'postJson("/api/kaggle/credentials"' in script
    assert 'postJson("/api/kaggle/submit"' in script

    status, config = _json_get(base, "/api/config")
    assert status == 200
    assert len(config["models"]) == 51
    assert len(config["profiles"]) == 5
    assert config["recipes"]
    assert {item["kind"] for item in config["recipes"]} == {"preset", "pipeline"}
    all_pipelines = {
        item["id"] for item in config["recipes"] if item["kind"] == "pipeline"
    }
    presets = {item["id"]: item for item in config["pipeline_presets"]}
    assert {"starter", "fast", "balanced", "quality", "all"} <= set(presets)
    assert set(presets["all"]["pipelines"]) == all_pipelines
    assert all(set(item["pipelines"]) <= all_pipelines for item in presets.values())
    pipeline_entry = next(item for item in config["recipes"] if item["kind"] == "pipeline")
    assert pipeline_entry["pipeline"]["blocks"][0]["type"] == "base"
    assert pipeline_entry["derived_from"]
    assert all(isinstance(model["available"], bool) for model in config["models"])

    status, empty = _json_get(base, "/api/dataset")
    assert status == 200
    assert empty["files"]["train"] is None
    assert empty["ready"] is False

    train, test = _sample_frames()
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    _upload_csv(base, "train", "train.csv", train)
    _upload_csv(base, "test", "test.csv", test)
    _upload_csv(base, "sample", "sample_submission.csv", sample)

    assert (data / "train.csv").is_file()
    assert (data / "test.csv").is_file()
    assert (data / "sample_submission.csv").is_file()

    status, dataset = _json_get(base, "/api/dataset")
    assert status == 200
    assert dataset["files"]["train"]["name"] == "train.csv"
    assert dataset["previews"]["train"]["rows"] == 90
    assert dataset["detected"]["target"] == "target"
    assert dataset["detected"]["task"] == "binary"
    assert dataset["detected"]["metric"] == "roc_auc"
    assert dataset["target_summary"]["nunique"] == 2
    assert dataset["target_summary"]["suggested_task"] == "binary"

    status, body = _get(base, "/static/app.css")
    assert status == 200
    assert b"--mint" in body

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(
            base,
            "/api/run",
            {"project": "bad-task", "target": "target", "task": "multiclass", "models": ["lgbm"]},
        )
    assert exc_info.value.code == 400
    assert "2 класса" in json.loads(exc_info.value.read().decode("utf-8"))["message"]

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _upload_csv(base, "train", "train.zip", train)
    assert exc_info.value.code == 400

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(
            base,
            "/api/run",
            {"project": "bad-target", "target": "not-a-column", "models": ["lgbm"]},
        )
    assert exc_info.value.code == 400
    assert "not-a-column" in json.loads(exc_info.value.read().decode("utf-8"))["message"]


def test_run_command_accepts_resource_and_manual_model_overrides(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)

    common = {
        "project": "web-tuning",
        "target": "target",
        "id_col": "id",
        "task": "binary",
        "metric": "roc_auc",
        "recipe": "pesnik/baseline",
        "strategy": "fast",
        "profile": 3,
        "device": "cpu",
        "cpu_threads": 2,
        "ram_gb": 12,
        "vram_gb": 0,
        "models": ["lgbm", "xgb"],
        "until": "train",
        "sample_pct": 37,
        "force": True,
        "model_params": {"lgbm": {"n_estimators": 321}, "xgb": {"max_depth": 4}},
    }
    _project, command, normalized = server.app.build_run_command(common)
    assert command[command.index("--profile") + 1] == "3"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--cpu-threads") + 1] == "2"
    assert command[command.index("--ram-gb") + 1] == "12"
    assert command[command.index("--vram-gb") + 1] == "0"
    encoded = json.loads(command[command.index("--model-params") + 1])
    assert encoded == common["model_params"]
    assert normalized["model_params"] == common["model_params"]
    assert normalized["cpu_threads"] == 2
    assert normalized["ram_gb"] == 12
    assert normalized["vram_gb"] == 0
    assert command[command.index("--sample-pct") + 1] == "37"
    assert normalized["sample_pct"] == 37

    _, gpu_command, _ = server.app.build_run_command(
        {**common, "profile": 4, "device": "cuda", "cpu_threads": 0}
    )
    assert gpu_command[gpu_command.index("--profile") + 1] == "4"
    assert gpu_command[gpu_command.index("--device") + 1] == "cuda"
    assert "auto" in server.app._variant_label(
        {**common, "profile": 4, "device": "cuda", "cpu_threads": 0}
    )

    with pytest.raises(WebError):
        server.app.build_run_command({**common, "model_params": {"catboost": {"depth": 4}}})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "device": "cuda"})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "profile": None, "cpu_threads": 2})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "profile": 2.5})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "profile": True})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "model_params": []})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "model_params": {"lgbm": []}})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "model_params": {"lgbm": {"x": float("nan")}}})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "model_params": {"lgbm": {"x": "a" * 17_000}}})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "sample_pct": 0})
    with pytest.raises(WebError):
        server.app.build_run_command({**common, "sample_pct": 101})

    _, plain_command, plain_normalized = server.app.build_run_command(
        {
            **common,
            "profile": None,
            "device": None,
            "cpu_threads": None,
            "ram_gb": None,
            "vram_gb": None,
            "sample_pct": None,
            "model_params": {},
        }
    )
    for flag in ("--device", "--cpu-threads", "--ram-gb", "--vram-gb", "--model-params"):
        assert flag not in plain_command
    assert plain_command[plain_command.index("--sample-pct") + 1] == "100"
    assert not plain_normalized.get("model_params")


def test_config_keeps_metrics_and_profiles_when_recipe_catalog_fails(
    web_server: tuple[KconvWebServer, str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _server, base, _data = web_server

    def broken_catalog():
        raise RuntimeError("bad recipe catalog")

    monkeypatch.setattr("kconv.web.list_pipelines", broken_catalog)
    status, config = _json_get(base, "/api/config")
    assert status == 200
    assert config["recipes"] == []
    assert "bad recipe catalog" in config["recipe_error"]
    assert config["metrics"]
    assert len(config["profiles"]) == 5


def test_llm_propose_passes_long_cli_timeout(
    web_server: tuple[KconvWebServer, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(base, "sample", "sample_submission.csv", pd.DataFrame({"id": test["id"], "target": 0}))

    calls: list[list[str]] = []
    timeouts: list[float | None] = []
    original_run = subprocess.run

    def spy(*args, **kwargs):
        if args and isinstance(args[0], list):
            calls.append(list(args[0]))
        timeouts.append(kwargs.get("timeout"))
        return original_run(*args, **kwargs)

    monkeypatch.setattr("kconv.web.subprocess.run", spy)
    status, plan = _post_json(
        base,
        "/api/llm/propose",
        {"target": "target", "id_col": "id", "provider": "mock", "max_features": 2},
    )
    assert status == 200, plan
    command = next(item for item in calls if "--llm-timeout" in item)
    assert command[command.index("--llm-timeout") + 1] == "170"
    assert max(timeouts) == 270


def test_llm_feature_plan_is_visible_and_runs_after_approval(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(base, "sample", "sample_submission.csv", pd.DataFrame({"id": test["id"], "target": 0}))

    status, index = _get(base, "/")
    assert status == 200
    html = index.decode("utf-8")
    assert "LLM-признаки" in html
    assert 'id="llm-json-mode"' in html

    status, plan = _post_json(
        base,
        "/api/llm/propose",
        {"target": "target", "id_col": "id", "provider": "mock", "max_features": 4},
    )
    assert status == 200, plan
    assert plan["plan_id"]
    assert plan["target"] == "target"
    assert plan["id_col"] == "id"
    assert plan["provider"] == "mock"
    assert 1 <= len(plan["features"]) <= 4
    assert all(item["name"].startswith("llm_") for item in plan["features"])

    status, no_json_plan = _post_json(
        base,
        "/api/llm/propose",
        {
            "target": "target",
            "id_col": "id",
            "provider": "mock",
            "max_features": 3,
            "json_mode": False,
        },
    )
    assert status == 200, no_json_plan
    assert no_json_plan["plan_id"]

    run_payload = {
        "project": "web-llm-plan",
        "target": "target",
        "id_col": "id",
        "task": "binary",
        "metric": "roc_auc",
        "recipe": "pesnik/baseline",
        "strategy": "fast",
        "profile": None,
        "models": ["lgbm"],
        "until": "report",
        "force": True,
        "llm_plan_id": plan["plan_id"],
        "llm_approved": False,
    }
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(base, "/api/run", run_payload)
    assert exc_info.value.code == 400
    assert "подтвержд" in json.loads(exc_info.value.read().decode("utf-8"))["message"].lower()

    run_payload["llm_approved"] = True
    status, started = _post_json(base, "/api/run", run_payload)
    assert status == 202, started

    final: dict | None = None
    deadline = time.time() + 90
    while time.time() < deadline:
        _, final = _json_get(base, "/api/run/status?project=web-llm-plan")
        if final.get("status") in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.25)
    assert final is not None
    assert final["status"] == "completed", json.dumps(final, ensure_ascii=False)[:4000]
    assert final["has_report"] is True
    assert "LLM feature proposals" in final["report"]
    run_dir = server.app.root / "competitions" / "web-llm-plan" / "runs" / final["run_id"]
    llm_log = run_dir / "05_report" / "llm_features.json"
    assert llm_log.is_file()
    metadata = json.loads(llm_log.read_text(encoding="utf-8"))
    assert metadata["experiment"] == "llm-feature-proposals"
    assert metadata["approved"] is True
    assert metadata["plan_sha256"]
    assert metadata["proposals"] == plan["features"]


def test_llm_openai_without_key_reports_configuration_error(
    web_server: tuple[KconvWebServer, str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(base, "sample", "sample_submission.csv", pd.DataFrame({"id": test["id"], "target": 0}))

    monkeypatch.delenv("KCONV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(
            base,
            "/api/llm/propose",
            {
                "target": "target",
                "id_col": "id",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "max_features": 2,
            },
        )
    assert exc_info.value.code == 400
    payload = json.loads(exc_info.value.read().decode("utf-8"))
    assert "ключ" in payload["message"].lower()
    assert "mock" in payload["hint"].lower()


def test_local_llm_discovery_reports_llama_cpp(
    web_server: tuple[KconvWebServer, str, Path], fake_llama: str
) -> None:
    _server, base, _data = web_server
    found = discover_local_endpoints(candidates=[("llama.cpp", fake_llama)])
    assert len(found) == 1
    assert found[0].runtime == "llama.cpp"
    assert found[0].models == ("fake-llama",)

    status, payload = _json_get(
        base,
        "/api/llm/local/discover?base_url=" + quote(fake_llama, safe=""),
    )
    assert status == 200
    assert payload["found"] is True
    assert payload["endpoints"][0]["base_url"] == fake_llama
    assert payload["endpoints"][0]["models"] == ["fake-llama"]


def test_local_llama_cpp_plan_auto_detects_model_and_retries_without_json_mode(
    web_server: tuple[KconvWebServer, str, Path], fake_llama: str
) -> None:
    server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(base, "sample", "sample_submission.csv", pd.DataFrame({"id": test["id"], "target": 0}))

    status, plan = _post_json(
        base,
        "/api/llm/propose",
        {
            "target": "target",
            "id_col": "id",
            "provider": "local",
            "base_url": fake_llama,
            "max_features": 2,
            "max_tokens": 256,
        },
    )
    assert status == 200, plan
    assert plan["features"][0]["name"] == "llm_x1_x2_sum"
    assert plan["model"] == "fake-llama"
    assert len(_FakeLlamaHandler.calls) == 3
    first, second, third = _FakeLlamaHandler.calls
    assert first["model"] == "fake-llama"
    assert first["max_tokens"] == 256
    assert first["response_format"] == {"type": "json_object"}
    assert "response_format" not in second
    assert "response_format" not in third
    metadata_path = server.app.data_dir / "llm" / f"plan-{plan['plan_id']}.llm.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["max_tokens"] == 256
    assert metadata["json_mode"] is False


def test_local_llama_cpp_plan_auto_discovers_endpoint(
    web_server: tuple[KconvWebServer, str, Path],
    fake_llama: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(base, "sample", "sample_submission.csv", pd.DataFrame({"id": test["id"], "target": 0}))

    monkeypatch.setattr(
        "kconv.llm_provider.discover_local_endpoints",
        lambda **_: (LocalModelEndpoint("llama.cpp", fake_llama, ("fake-llama",)),),
    )
    status, plan = _post_json(
        base,
        "/api/llm/propose",
        {"target": "target", "id_col": "id", "provider": "local", "max_features": 2},
    )
    assert status == 200, plan
    assert plan["features"][0]["name"] == "llm_x1_x2_sum"
    assert plan["model"] == "fake-llama"
    assert len(_FakeLlamaHandler.calls) == 3


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("urlopen error [WinError 10061] connection refused", "подключиться"),
        ("LLM feature plan is not valid JSON.", "схем"),
        ("LLM request failed (HTTP 401).", "401"),
        ("LLM request failed (HTTP 404).", "404"),
        ("LLM output was truncated before a valid plan was completed.", "обрезан"),
        ("Local LLM server was not found.", "не найден"),
    ],
)
def test_llm_provider_errors_are_actionable(detail: str, expected: str) -> None:
    error = _llm_web_error(detail)
    assert expected in f"{error.message} {error.hint}".lower()


def test_run_status_metrics_submission_and_report(web_server: tuple[KconvWebServer, str, Path]) -> None:
    _server, base, data = web_server
    # Keep this test runnable on its own; the upload test normally runs first.
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(base, "sample", "sample_submission.csv", pd.DataFrame({"id": test["id"], "target": 0}))

    status, started = _post_json(
        base,
        "/api/run",
        {
            "project": "web-test",
            "target": "target",
            "id_col": "id",
            "task": "binary",
            "metric": "roc_auc",
            "recipe": "pesnik/baseline",
            "strategy": "fast",
            "profile": None,
            "models": ["lgbm"],
            "until": "report",
            "force": True,
        },
    )
    assert status == 202, started
    assert started["project"] == "web-test"
    assert started["status"] in {"starting", "running"}

    final: dict | None = None
    deadline = time.time() + 90
    while time.time() < deadline:
        _, final = _json_get(base, "/api/run/status?project=web-test")
        if final.get("status") in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.25)

    assert final is not None
    assert final["status"] == "completed", json.dumps(final, ensure_ascii=False)[:4000]
    assert [stage["name"] for stage in final["stages"]] == [
        "ingest",
        "validate",
        "hpo",
        "train",
        "blend",
        "submit",
        "report",
    ]
    assert all(stage["status"] == "done" for stage in final["stages"])
    assert final["metrics"]["blend"]["oof_score"] > 0.5
    assert final["metrics"]["models"][0]["name"] == "lgbm"
    assert final["has_submission"] is True
    assert final["submission_preview"]["columns"] == ["id", "target"]
    assert final["has_report"] is True
    assert "kconv" in final["report"].lower()
    assert "<h1>" in final["report_html"]
    assert "<table>" in final["report_html"]
    assert "<script" not in final["report_html"].lower()

    _, comparison = _json_get(base, "/api/comparison?project=web-test")
    assert comparison["project"] == "web-test"
    assert comparison["best_run_id"] == final["run_id"]
    assert len(comparison["runs"]) == 1
    assert comparison["runs"][0]["strategy"] == "fast"
    assert comparison["runs"][0]["models"] == ["lgbm"]
    assert comparison["runs"][0]["best"] is True

    status, submission = _get(base, "/api/submission?project=web-test")
    assert status == 200
    text = submission.decode("utf-8")
    assert text.startswith("id,target")
    assert len(text.strip().splitlines()) == 13

    status, report = _get(base, "/api/report?project=web-test")
    assert status == 200
    assert b"kconv" in report.lower()
    status, report_by_id = _get(base, f"/api/report?project=web-test&run_id={final['run_id']}")
    assert status == 200
    assert report_by_id == report
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(base, "/api/report?project=web-test&run_id=..%2F..%2Fsecret")
    assert exc_info.value.code == 404

    _, projects = _json_get(base, "/api/projects")
    assert any(item["project"] == "web-test" for item in projects)

    _, idle = _json_get(base, "/api/run/status?project=does-not-exist")
    assert idle["status"] == "idle"
    assert len(idle["stages"]) == 7

    _, stats = _json_get(base, "/api/stats")
    assert stats["models"]["total"] == 51
    assert stats["dataset"]["previews"]["train"]["rows"] == 90


def test_batch_queue_runs_sequentially_and_comparison(web_server: tuple[KconvWebServer, str, Path]) -> None:
    _server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)

    common = {
        "project": "web-batch",
        "target": "target",
        "id_col": "id",
        "task": "binary",
        "metric": "roc_auc",
        "recipe": "pesnik/baseline",
        "profile": None,
        "until": "report",
        "force": True,
    }
    _project, command, normalized = _server.app.build_run_command(
        {**common, "until": "train", "models": ["lgbm"]}
    )
    assert command[command.index("--until") + 1] == "blend"
    assert normalized["until"] == "blend"

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(
            base,
            "/api/run/batch",
            {
                "variants": [
                    {**common, "label": "broken", "strategy": "fast", "task": "multiclass", "models": ["lgbm"]},
                ]
            },
        )
    assert exc_info.value.code == 400
    _, empty_comparison = _json_get(base, "/api/comparison?project=web-batch")
    assert empty_comparison["runs"] == []

    status, started = _post_json(
        base,
        "/api/run/batch",
        {
            "variants": [
                {**common, "label": "fast lgbm", "strategy": "fast", "models": ["lgbm"]},
                {**common, "label": "standard lgbm", "strategy": "standard", "models": ["lgbm"]},
            ]
        },
    )
    assert status == 202, started
    assert len(started["items"]) == 2
    assert [item["label"] for item in started["items"]] == ["fast lgbm", "standard lgbm"]
    assert started["status"] in {"starting", "running"}

    final: dict | None = None
    deadline = time.time() + 150
    while time.time() < deadline:
        _, final = _json_get(base, "/api/run/status?project=web-batch")
        items = final.get("queue") or []
        terminal = {"completed", "failed", "stopped", "cancelled"}
        if (
            len(items) == 2
            and not final.get("batch_active")
            and all(item["status"] in terminal for item in items)
        ):
            break
        time.sleep(0.25)

    assert final is not None
    assert final["batch_active"] is False
    items = final["queue"]
    assert [item["status"] for item in items] == ["completed", "completed"], json.dumps(items, ensure_ascii=False)
    assert all(item["run_id"] for item in items)
    assert items[0]["run_id"] != items[1]["run_id"]
    assert final["status"] == "completed"

    _, comparison = _json_get(base, "/api/comparison?project=web-batch")
    runs = comparison["runs"]
    assert len(runs) == 2
    assert [run["strategy"] for run in runs] == ["fast", "standard"]
    assert all(run["models"] == ["lgbm"] for run in runs)
    assert all(run["status"] == "completed" for run in runs)
    assert all(run["oof_score"] is not None for run in runs)
    assert comparison["best_run_id"] in {run["run_id"] for run in runs}
    assert sum(1 for run in runs if run["best"]) == 1
    assert comparison["best_run_id"] == max(runs, key=lambda run: run["oof_score"])["run_id"]


def test_new_single_run_is_not_marked_completed_by_old_batch(tmp_path: Path) -> None:
    """A completed batch must not leak its terminal status into a new run."""

    (tmp_path / "root").mkdir()
    server = create_server(root=tmp_path / "root", data_dir=tmp_path / "data", port=0)
    manager = server.app.runs
    try:
        manager.enqueue(
            [("old batch", "generation-project", [sys.executable, "-c", "pass"], {})],
            project="generation-project",
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            snapshot = manager.queue_snapshot("generation-project")
            if not snapshot["active"] and snapshot["items"] and snapshot["items"][0]["status"] == "completed":
                break
            time.sleep(0.05)
        assert snapshot["items"][0]["status"] == "completed"

        manager.start(
            "generation-project",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            {"strategy": "fast", "models": ["lgbm"]},
        )
        status = server.app.run_status("generation-project")
        assert status["status"] in {"starting", "running"}, status
    finally:
        manager.stop()
        server.server_close()


def test_stale_disk_run_does_not_lock_the_ui(web_server: tuple[KconvWebServer, str, Path]) -> None:
    server, base, _data = web_server
    project_dir = server.app.root / "competitions" / "stale-project"
    run_dir = project_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "project": "stale-project",
                "status": "running",
                "strategy": "fast",
                "config": {"strategy": "fast", "metric": "roc_auc", "models": []},
            }
        ),
        encoding="utf-8",
    )
    (project_dir / "latest.json").write_text(
        json.dumps({"run_id": "run-1", "status": "running"}),
        encoding="utf-8",
    )

    _, status = _json_get(base, "/api/run/status?project=stale-project")
    assert status["status"] == "running"
    assert status["stale"] is True
    assert status["server_active"] is False
    assert status["server_busy"] is False


def test_comparison_groups_runs_by_dataset(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, _data = web_server
    project = "dataset-groups"
    project_dir = server.app.root / "competitions" / project
    runs_dir = project_dir / "runs"
    cards = {
        "run-1": {"source": {"train": "alpha.csv"}, "target": "target", "rows": {"train": 90}, "task": "multiclass", "class_balance": {"a": 1}, "roles": {"feature_columns": ["x"]}},
        "run-2": {"source": {"train": "alpha.csv"}, "target": "target", "rows": {"train": 90}, "task": "binary", "class_balance": {"b": 2}, "roles": {"feature_columns": ["x", "y"]}},
        "run-3": {"source": {"train": "beta.csv"}, "target": "target", "rows": {"train": 120}, "roles": {"feature_columns": ["p", "q"]}},
        "run-4": {"source": {"train": "beta.csv"}, "target": "target", "rows": {"train": 120}, "roles": {"feature_columns": ["p", "q"]}},
    }
    scores = {"run-1": 0.70, "run-2": 0.80, "run-3": 0.90, "run-4": 0.60}
    for run_id, card in cards.items():
        run_dir = runs_dir / run_id
        (run_dir / "00_ingest").mkdir(parents=True)
        (run_dir / "03_blend").mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "project": project,
                    "status": "completed",
                    "strategy": "fast",
                    "config": {
                        "strategy": "fast",
                        "metric": "roc_auc",
                        "models": [{"name": "lgbm"}],
                        "data": {"train": "../../data/train.csv", "target": "target"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "00_ingest" / "data_card.json").write_text(json.dumps(card), encoding="utf-8")
        (run_dir / "03_blend" / "blend.json").write_text(
            json.dumps({"oof_score": scores[run_id], "metric": "roc_auc"}),
            encoding="utf-8",
        )
    (project_dir / "latest.json").write_text(
        json.dumps({"run_id": "run-2", "status": "completed"}),
        encoding="utf-8",
    )

    payload = server.app.comparison_payload(project)
    assert len(payload["datasets"]) == 2
    assert len(payload["runs"]) == 4
    assert payload["best_run_id"] is None
    assert sum(1 for run in payload["runs"] if run["best"]) == 2
    by_id = {item["id"]: item for item in payload["datasets"]}
    alpha = next(item for item in payload["datasets"] if item["label"].startswith("alpha.csv"))
    beta = next(item for item in payload["datasets"] if item["label"].startswith("beta.csv"))
    assert alpha["best_run_id"] == "run-2"
    assert beta["best_run_id"] == "run-3"
    assert set(by_id) == {alpha["id"], beta["id"]}

    _, filtered = _json_get(base, f"/api/comparison?project={project}&dataset={beta['id']}")
    assert filtered["dataset"] == beta["id"]
    assert filtered["best_run_id"] == "run-3"
    assert len(filtered["runs"]) == 2
    assert {run["run_id"] for run in filtered["runs"]} == {"run-3", "run-4"}
    assert sum(1 for run in filtered["runs"] if run["best"]) == 1


def _kaggle_project(server: KconvWebServer, project: str = "kaggle-api") -> Path:
    project_dir = server.app.root / "competitions" / project
    submission_dir = project_dir / "runs" / "run-1" / "04_submit"
    submission_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.9]}).to_csv(
        submission_dir / "submission.csv",
        index=False,
    )
    write_latest_pointer(project_dir, "run-1", status="completed")
    return project_dir


def _ready_auth(**_kwargs: object) -> AuthStatus:
    return AuthStatus(
        cli_available=True,
        cli_path="kaggle",
        cli_version="Kaggle API 2.2.5",
        python_package=True,
        credentials_present=True,
        credential_source="file: test-kaggle.json",
        config_path="test-kaggle.json",
    )


def test_kaggle_credentials_and_one_click_submit(
    web_server: tuple[KconvWebServer, str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, base, _data = web_server
    _kaggle_project(server)
    config_dir = tmp_path / "kaggle-config"
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr("kconv.web.check_auth", _ready_auth)

    status, payload = _json_get(base, "/api/kaggle/status?project=kaggle-api")
    assert status == 200
    assert payload["has_submission"] is True
    assert payload["auth"]["ready"] is True
    assert payload["submission"]["name"] == "submission.csv"

    status, saved = _post_json(
        base,
        "/api/kaggle/credentials",
        {"project": "kaggle-api", "username": "alice", "key": "super-secret"},
    )
    assert status == 200, saved
    assert saved["saved"] is True
    assert saved["auth"]["ready"] is True
    assert "super-secret" not in json.dumps(saved)
    credential_file = config_dir / "kaggle.json"
    assert json.loads(credential_file.read_text(encoding="utf-8")) == {
        "username": "alice",
        "key": "super-secret",
    }

    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):  # noqa: ANN001, ANN003
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Success", stderr="")

    monkeypatch.setattr("kconv.kaggle.submit.subprocess.run", fake_run)
    status, submitted = _post_json(
        base,
        "/api/kaggle/submit",
        {
            "project": "kaggle-api",
            "run_id": "run-1",
            "competition": "titanic",
            "message": "web one click",
        },
    )
    assert status == 200, submitted
    assert submitted["executed"] is True
    assert submitted["returncode"] == 0
    assert submitted["competition"] == "titanic"
    assert submitted["rows"] == 2
    assert calls and calls[0][:5] == ["kaggle", "competitions", "submit", "-c", "titanic"]
    assert calls[0][calls[0].index("-m") + 1] == "web one click"


def test_kaggle_token_is_saved_without_echo(
    web_server: tuple[KconvWebServer, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, base, _data = web_server
    monkeypatch.setattr("kconv.web.check_auth", _ready_auth)
    token_path = server.app.kaggle_token_path
    assert token_path is not None
    token_path.unlink(missing_ok=True)

    status, saved = _post_json(
        base,
        "/api/kaggle/credentials",
        {"project": "kaggle-api", "token": "token-super-secret"},
    )
    assert status == 200, saved
    assert saved["saved"] is True
    assert saved["saved_path"] == str(token_path)
    assert "token-super-secret" not in json.dumps(saved)
    assert token_path.read_text(encoding="utf-8") == "token-super-secret\n"


def test_kaggle_submit_requires_ready_credentials(
    web_server: tuple[KconvWebServer, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, base, _data = web_server
    _kaggle_project(server, "kaggle-not-ready")
    not_ready = AuthStatus(
        cli_available=True,
        cli_path="kaggle",
        cli_version="Kaggle API 2.2.5",
        python_package=True,
        credentials_present=False,
        credential_source=None,
        config_path="test-kaggle.json",
    )
    monkeypatch.setattr("kconv.web.check_auth", lambda **_kwargs: not_ready)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(
            base,
            "/api/kaggle/submit",
            {"project": "kaggle-not-ready", "competition": "titanic"},
        )
    assert exc_info.value.code == 400
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "credentials" in f"{body['message']} {body['hint']}".lower()



def test_failure_runs_and_filtered_logs_endpoints(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, _data = web_server
    project = "web-failure"
    run_dir = server.app.root / "competitions" / project / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    from kconv.context import write_json

    write_json(
        run_dir / "run.json",
        {
            "run_id": "run-1",
            "project": project,
            "status": "failed",
            "config": {
                "project": project,
                "recipe": "pesnik/baseline",
                "strategy": "fast",
                "task": "binary",
                "metric": "roc_auc",
                "models": [{"name": "lgbm", "enabled": True}],
            },
            "failure_report": "failure_report.md",
            "failure_json": "failure.json",
        },
    )
    write_json(
        run_dir / "failure.json",
        {
            "run_id": "run-1",
            "stage": "train",
            "status": "failed",
            "exception": {"type": "RuntimeError", "message": "boom"},
            "last_log_lines": ["INFO started", "ERROR boom", "WARNING slow"],
        },
    )
    (run_dir / "failure_report.md").write_text("# Failure\n\nboom\n", encoding="utf-8")

    status, payload = _json_get(base, f"/api/failure?project={project}&run_id=run-1")
    assert status == 200
    assert payload["failure"]["exception"]["message"] == "boom"
    assert "boom" in payload["report_html"]

    status, payload = _json_get(base, f"/api/runs?project={project}")
    assert status == 200
    assert payload["runs"][0]["run_id"] == "run-1"

    status, payload = _json_get(
        base, f"/api/run/logs?project={project}&run_id=run-1&level=ERROR&q=boom"
    )
    assert status == 200
    assert payload["lines"] == ["ERROR boom"]


def test_llm_propose_passes_ideas_code_and_repairs(
    web_server: tuple[KconvWebServer, str, Path], monkeypatch
) -> None:
    _server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(
            base,
            "sample",
            "sample_submission.csv",
            pd.DataFrame({"id": test["id"], "target": 0}),
        )
    calls: list[list[str]] = []
    original_run = subprocess.run

    def spy(*args, **kwargs):
        if args and isinstance(args[0], list):
            calls.append(list(args[0]))
        return original_run(*args, **kwargs)

    monkeypatch.setattr("kconv.web.subprocess.run", spy)
    status, _plan = _post_json(
        base,
        "/api/llm/propose",
        {
            "target": "target",
            "id_col": "id",
            "provider": "mock",
            "max_features": 2,
            "ideas": "try a ratio\nadd a difference",
            "code": "def ratio(a, b):\n    return a / b\n",
            "repairs": 3,
        },
    )
    assert status == 200
    command = next(item for item in calls if "--llm-idea" in item)
    ideas = [
        command[index + 1]
        for index, item in enumerate(command)
        if item == "--llm-idea"
    ]
    assert ideas == ["try a ratio", "add a difference"]
    assert command[command.index("--llm-repairs") + 1] == "3"
    code_arg = command[command.index("--llm-code-file") + 1]
    assert Path(code_arg).name.startswith("code-")
    assert Path(code_arg).suffix == ".py"
    assert not list((data / "llm").glob("code-*.py"))


def test_run_dir_rejects_pointer_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    app = WebApp(root, tmp_path / "data")
    project_dir = root / "competitions" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "latest.json").write_text(json.dumps({"run_id": "../secret"}), encoding="utf-8")

    assert app._run_dir("demo") is None


def test_llm_plan_invalidated_when_train_content_changes(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(
            base,
            "sample",
            "sample_submission.csv",
            pd.DataFrame({"id": test["id"], "target": 0}),
        )

    status, plan = _post_json(
        base,
        "/api/llm/propose",
        {"target": "target", "id_col": "id", "provider": "mock", "max_features": 2},
    )
    assert status == 200, plan
    train_path = data / "train.csv"
    original = train_path.read_bytes()
    original_stat = train_path.stat()
    try:
        changed = bytearray(original)
        changed[0] = ord("x") if changed[0] != ord("x") else ord("y")
        train_path.write_bytes(bytes(changed))
        os.utime(train_path, (original_stat.st_atime, original_stat.st_mtime))

        status, session = _json_get(base, "/api/session")
        assert status == 200
        assert session.get("llm_plan") is None
    finally:
        train_path.write_bytes(original)
        os.utime(train_path, (original_stat.st_atime, original_stat.st_mtime))


def test_llm_plan_survives_server_memory_reset(
    web_server: tuple[KconvWebServer, str, Path],
) -> None:
    server, base, data = web_server
    if not (data / "train.csv").exists():
        train, test = _sample_frames()
        _upload_csv(base, "train", "train.csv", train)
        _upload_csv(base, "test", "test.csv", test)
        _upload_csv(
            base,
            "sample",
            "sample_submission.csv",
            pd.DataFrame({"id": test["id"], "target": 0}),
        )

    status, plan = _post_json(
        base,
        "/api/llm/propose",
        {"target": "target", "id_col": "id", "provider": "mock", "max_features": 2},
    )
    assert status == 200, plan
    with server.app._llm_lock:
        server.app._llm_plans.clear()

    status, session = _json_get(base, "/api/session")
    assert status == 200
    restored = session.get("llm_plan")
    assert restored is not None
    assert restored["plan_id"] == plan["plan_id"]
    assert restored["features"] == plan["features"]
    assert restored["target"] == "target"


def test_web_run_log_is_persisted_with_failure_tail(tmp_path: Path) -> None:
    root = tmp_path / "root"
    data = tmp_path / "data"
    app = WebApp(root, data)
    manager = RunManager(app)
    project = "log-project"
    project_dir = root / "competitions" / project
    run_dir = project_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    write_latest_pointer(project_dir, "run-1", status="running")
    (run_dir / "failure.json").write_text(
        json.dumps({"last_log_lines": []}),
        encoding="utf-8",
    )

    command = [sys.executable, "-c", "print('hello from web'); print('second line')"]
    item = QueueItem(
        id="logitem",
        label="log",
        project=project,
        command=command,
        payload={},
        previous_run_id="old",
    )
    run = ManagedRun(project=project, command=command)
    manager._execute(run, item)

    assert item.status == "completed"
    web_log = run_dir / "web.log"
    assert web_log.is_file()
    text = web_log.read_text(encoding="utf-8")
    assert "hello from web" in text
    assert "second line" in text
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["web_log"] == "web.log"
    assert any("hello from web" in line for line in failure["last_log_lines"])
    payload = app.logs_payload(project, "run-1")
    assert any("hello from web" in line for line in payload["lines"])


def test_recover_stale_runs_scans_projects(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "competitions" / "demo"
    run = project / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_recover(project_dir, run_id):
        calls.append((Path(project_dir).name, run_id))
        return {"recovered": True, "run_id": run_id, "status": "interrupted"}

    monkeypatch.setattr("kconv.watchdog.recover_stale_run", fake_recover)
    refreshed = []
    monkeypatch.setattr(
        "kconv.runs_index.refresh_index",
        lambda path: refreshed.append(Path(path).name) or {},
    )
    app = WebApp(tmp_path, tmp_path / "data")
    result = app.recover_stale_runs()
    assert calls == [("demo", "run-1")]
    assert refreshed == ["demo"]
    assert result == [{"project": "demo", "run_id": "run-1", "status": "interrupted"}]


def test_web_ux_scaffolding_is_present() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "kconv" / "web_static"
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    styles = (root / "app.css").read_text(encoding="utf-8")

    for anchor in (
        'id="section-nav"',
        'id="run-dock"',
        'id="dock-run"',
        'id="dock-stop"',
        'id="dock-results"',
        'id="reset-config"',
        'id="section-data"',
        'id="section-setup"',
        'id="section-results"',
        'id="section-compare"',
    ):
        assert anchor in html, anchor

    assert "function applyPrefsStatic()" in script
    assert "function applyPrefsDataset(" in script
    assert "function initShortcuts()" in script
    assert "function initSectionNav()" in script
    assert "function syncRunDock()" in script
    assert "localStorage.setItem(PREFS_KEY" in script

    assert ".section-nav" in styles
    assert ".run-dock" in styles
    assert ":focus-visible" in styles

