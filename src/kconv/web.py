"""Local browser UI for kconv.

The server deliberately uses only the Python standard library: no FastAPI or
Node build step is required.  It wraps the existing CLI (``kconv run``) rather
than duplicating pipeline logic, so the web UI and the command line produce the
same runs and artifacts.

Run it with::

    kconv web
    # or
    python -m kconv.cli web --data data

The first version covers the dataset-first workflow:

* upload ``train``/``test``/``sample_submission`` files;
* inspect columns and a preview;
* choose target/id/task/metric/recipe/strategy/profile/models;
* start, watch and stop a run;
* read stage status, OOF metrics, submission preview, report and download links.

All data stays on the local machine.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

from kconv.api import list_experiments, list_models, list_pipelines
from kconv.autodetect import detect_dataset
from kconv.context import file_sha256, validate_run_id
from kconv.errors import KconvError
from kconv.io import is_table, safe_read_sample
from kconv.kaggle.auth import check_auth, save_credentials
from kconv.kaggle.submit import submit_submission
from kconv.llm_provider import (
    LocalModelEndpoint,
    discover_local_endpoints,
    discover_local_models,
    normalize_local_base_url,
    resolve_local_target,
)
from kconv.metrics import get_metric, metric_names
from kconv.models import model_available
from kconv.pipeline_presets import pipeline_presets
from kconv.registry import model_names
from kconv.report_markdown import render_markdown
from kconv.runtime import list_profiles
from kconv.sources import catalog_stats
from kconv.stages.run import STAGE_ORDER

STATIC_DIR = Path(__file__).resolve().with_name("web_static")
MAX_UPLOAD_BYTES = 256 * 1024 * 1024  # 256 MiB; enough for local tabular datasets.
MAX_JSON_BYTES = 1024 * 1024
_ALLOWED_EXTENSIONS = {".csv", ".csv.gz", ".parquet", ".pq"}
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})
_KIND_PREFIX = {"train": "train", "test": "test", "sample": "sample_submission"}
_PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LOGGER = __import__("logging").getLogger("kconv.web")


class WebError(Exception):
    """An expected error that should be returned as JSON to the browser."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = int(status)
        self.hint = hint


def _jsonable(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _is_loopback_url(value: str) -> bool:
    """Accept only local HTTP(S) endpoints for the ``local`` LLM provider."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "::1", "localhost"}


def _llm_web_error(detail: str) -> WebError:
    """Translate a CLI/provider failure into an actionable UI message.

    The CLI deliberately never prints keys or prompts, so the detail stays
    safe to show.  We still avoid presenting every provider failure as the
    vague "invalid plan" error.
    """
    text = (detail or "").strip()
    lowered = text.lower()
    if "credential is missing" in lowered or "api key is not set" in lowered:
        return WebError(
            "Не настроен ключ OpenAI для LLM.",
            status=HTTPStatus.BAD_REQUEST,
            hint=(
                "Задайте KCONV_LLM_API_KEY (или OPENAI_API_KEY) в окружении сервера, "
                "перезапустите start.cmd и повторите. Либо выберите provider=mock/local."
            ),
        )
    if "http 401" in lowered or "http 403" in lowered:
        return WebError(
            "LLM endpoint отклонил ключ доступа (HTTP 401/403).",
            status=HTTPStatus.BAD_GATEWAY,
            hint="Проверьте ключ, права доступа и выбранную модель; затем перезапустите сервер.",
        )
    if "http 429" in lowered:
        return WebError(
            "LLM endpoint ограничил частоту запросов (HTTP 429).",
            status=HTTPStatus.TOO_MANY_REQUESTS,
            hint="Подождите немного или проверьте лимиты аккаунта/локального сервера.",
        )
    if "http 404" in lowered or "model not found" in lowered:
        return WebError(
            "LLM endpoint или модель не найдены (HTTP 404).",
            status=HTTPStatus.BAD_GATEWAY,
            hint="Проверьте имя модели и base URL; для local убедитесь, что модель загружена.",
        )
    if re.search(r"http 5\d\d", lowered):
        return WebError(
            "LLM endpoint вернул серверную ошибку.",
            status=HTTPStatus.BAD_GATEWAY,
            hint="Проверьте лог Ollama/vLLM или доступность провайдера и повторите.",
        )
    if (
        "could not reach" in lowered
        or "connection refused" in lowered
        or "urlopen error" in lowered
    ):
        return WebError(
            "Не удалось подключиться к LLM endpoint.",
            status=HTTPStatus.BAD_GATEWAY,
            hint=(
                "Для local запустите llama.cpp (:8080), Ollama (:11434) или vLLM (:8000) "
                "и нажмите «Найти локальный сервер»; для openai проверьте сеть и ключ."
            ),
        )
    if "timeout" in lowered or "timed out" in lowered:
        return WebError(
            "LLM endpoint не ответил вовремя.",
            status=HTTPStatus.GATEWAY_TIMEOUT,
            hint=(
                "Проверьте, что модель загружена и endpoint отвечает; "
                "можно выбрать mock для проверки UI."
            ),
        )
    if "http 400" in lowered:
        return WebError(
            "LLM endpoint отклонил запрос (HTTP 400).",
            status=HTTPStatus.BAD_GATEWAY,
            hint=(
                "Проверьте model/base URL и поддержку JSON mode. "
                "Для llama.cpp можно выключить JSON mode или обновить llama-server."
            ),
        )
    if "local llm server was not found" in lowered:
        return WebError(
            "Локальный LLM не найден.",
            status=HTTPStatus.BAD_REQUEST,
            hint=(
                "Запустите llama-server -m model.gguf --port 8080, Ollama :11434 "
                "или vLLM :8000; base URL и model можно задать вручную."
            ),
        )
    if "local llm model was not detected" in lowered:
        return WebError(
            "Не удалось определить модель локального LLM.",
            status=HTTPStatus.BAD_REQUEST,
            hint="Укажите model вручную; проверьте, что GET <base-url>/models возвращает загруженную модель.",
        )
    if "truncated" in lowered:
        return WebError(
            "Модель не успела закончить JSON: ответ обрезан.",
            status=HTTPStatus.BAD_GATEWAY,
            hint=(
                "Увеличьте Max tokens (например, 2048), контекст llama-server (-c) "
                "и уменьшите max_features; затем повторите."
            ),
        )
    if (
        "not valid json" in lowered
        or "not valid feature proposals" in lowered
        or "features' list" in lowered
    ):
        return WebError(
            "Модель вернула ответ, который не прошёл проверку схемы.",
            status=HTTPStatus.BAD_GATEWAY,
            hint=(
                "Попробуйте другую модель или provider=mock для проверки пути; "
                "подробность от CLI: " + (text[-1200:] or "нет деталей")
            ),
        )
    if not text:
        return WebError(
            "LLM не сгенерировал валидный план.",
            status=HTTPStatus.BAD_GATEWAY,
        )
    return WebError(
        "LLM не сгенерировал валидный план.",
        status=HTTPStatus.BAD_GATEWAY,
        hint=text[-1600:],
    )


def _basename(value: Any) -> str:
    """Return the final path component for Windows and POSIX paths."""
    if not value:
        return "dataset"
    parts = re.split(r"[\\/]", str(value))
    return parts[-1] or "dataset"


def _slug(value: str) -> str:
    value = str(value or "").strip()
    if not _PROJECT_RE.fullmatch(value):
        raise WebError(
            "Некорректное имя проекта.",
            hint="Используйте 1-64 символа: латиница, цифры, точка, дефис или подчёркивание.",
        )
    return value


def _table_row_count(path: Path) -> int | None:
    name = path.name.lower()
    try:
        if name.endswith((".parquet", ".pq")):
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        if name.endswith(".csv.gz"):
            import gzip

            with gzip.open(path, "rb") as handle:
                return max(sum(1 for _ in handle) - 1, 0)
        if name.endswith(".csv"):
            with path.open("rb") as handle:
                return max(sum(1 for _ in handle) - 1, 0)
    except (OSError, ValueError):
        return None
    return None


def _optional_payload_int(
    payload: dict[str, Any],
    key: str,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int | None:
    """Read an optional integer field without silently truncating floats."""
    raw = payload.get(key)
    if raw in (None, ""):
        return None
    if isinstance(raw, bool):
        raise WebError(f"{label} должен быть целым числом от {minimum} до {maximum}.")
    if isinstance(raw, float) and not raw.is_integer():
        raise WebError(f"{label} должен быть целым числом от {minimum} до {maximum}.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise WebError(f"{label} должен быть целым числом от {minimum} до {maximum}.") from exc
    if not minimum <= value <= maximum:
        raise WebError(f"{label} должен быть целым числом от {minimum} до {maximum}.")
    return value


def _validated_model_params(payload: dict[str, Any], models: list[str]) -> dict[str, dict[str, Any]]:
    """Validate manual per-model JSON before it is embedded in the CLI argv."""
    raw = payload.get("model_params")
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise WebError(
            "model_params должен быть JSON-объектом, где ключи — id моделей.",
            hint='Пример: {"lgbm": {"n_estimators": 300}, "xgb": {"max_depth": 4}}.',
        )
    selected = set(models)
    unknown = sorted(str(name) for name in raw if str(name) not in selected)
    if unknown:
        raise WebError(
            f"model_params содержит невыбранные модели: {', '.join(unknown)}.",
            hint="Добавьте их в models или удалите их параметры.",
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        key = str(name)
        if not isinstance(value, dict):
            raise WebError(
                f"model_params для '{key}' должен быть JSON-объектом.",
                hint='Пример: {"n_estimators": 300, "learning_rate": 0.05}.',
            )
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise WebError(
                f"model_params для '{key}' содержит non-finite или non-JSON значение.",
                hint="Используйте конечные числа, строки, booleans, списки или объекты.",
            ) from exc
        if len(encoded.encode("utf-8")) > 16_000:
            raise WebError(
                f"model_params для '{key}' слишком большой (лимит 16 KB).",
                hint="Оставьте только параметры, которые действительно нужно переопределить.",
            )
        normalized[key] = dict(value)
    return normalized


def _preview(path: Path) -> dict[str, Any]:
    frame = safe_read_sample(path, nrows=5)
    return {
        "rows": _table_row_count(path),
        "columns": [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()],
        "head": _jsonable(frame.head(5).to_dict(orient="records")),
    }


def _target_summary(path: Path, target: str | None) -> dict[str, Any] | None:
    """Describe target cardinality for the UI without loading the whole file."""
    if not target:
        return None
    frame = safe_read_sample(path, nrows=100_000)
    if target not in frame.columns:
        return None
    series = frame[target]
    present = series.dropna()
    if present.empty:
        return {
            "column": target,
            "sample_rows": int(len(frame)),
            "nunique": 0,
            "counts": {},
            "missing": int(series.isna().sum()),
            "dtype": str(series.dtype),
            "suggested_task": None,
        }
    nunique = int(present.nunique())
    kind = present.dtype.kind
    if nunique == 2:
        suggested_task: str | None = "binary"
    elif kind in {"i", "u"}:
        suggested_task = "multiclass" if nunique <= 20 else "regression"
    elif kind == "f":
        values = present.to_numpy(dtype=float)
        suggested_task = "multiclass" if nunique <= 20 and bool(np.all(values == np.floor(values))) else "regression"
    elif kind in {"M", "m"}:
        suggested_task = None
    else:
        suggested_task = "multiclass"
    counts = {str(value): int(count) for value, count in present.value_counts().head(8).items()}
    return {
        "column": target,
        "sample_rows": int(len(frame)),
        "nunique": nunique,
        "counts": counts,
        "missing": int(series.isna().sum()),
        "dtype": str(series.dtype),
        "suggested_task": suggested_task,
    }


def _file_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "name": path.name,
        "size": int(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


@dataclass
class ManagedRun:
    project: str
    command: list[str]
    status: str = "starting"
    process: subprocess.Popen[str] | None = None
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    stopped: bool = False
    # The live subprocess log is kept in memory for the UI.  ``log_path`` points
    # at the durable copy under ``data/logs``; it is copied into the run
    # directory once the CLI has published its run id.
    log_path: Path | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    # RLock: snapshot() is also called from the already-locked stop() path.
    lock: threading.RLock = field(default_factory=threading.RLock)

    def append(self, line: str) -> None:
        with self.lock:
            self.log.append(line.rstrip())

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            end = self.finished_at or time.time()
            return {
                "project": self.project,
                "status": self.status,
                "returncode": self.returncode,
                "started_at": datetime.fromtimestamp(self.started_at, UTC).isoformat(),
                "finished_at": datetime.fromtimestamp(self.finished_at, UTC).isoformat()
                if self.finished_at
                else None,
                "elapsed_sec": round(max(end - self.started_at, 0.0), 1),
                "error": self.error,
                "log": list(self.log)[-500:],
            }


@dataclass
class LLMPlan:
    """A validated LLM feature plan held in server memory by opaque id.

    Absolute paths stay on the server; the browser only receives the id and
    the reviewed feature rows.  This prevents the run API from accepting
    arbitrary filesystem paths.
    """

    id: str
    path: Path
    metadata_path: Path
    target: str
    id_col: str | None
    provider: str
    model: str | None
    proposals: list[dict[str, Any]]
    warnings: list[str]
    usage: dict[str, Any]
    train_modified: float
    train_size: int
    train_sha256: str
    created_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan_id": self.id,
            "target": self.target,
            "id_col": self.id_col,
            "provider": self.provider,
            "model": self.model,
            "features": [dict(item) for item in self.proposals],
            "warnings": list(self.warnings),
            "usage": dict(self.usage),
            "created_at": datetime.fromtimestamp(self.created_at, UTC).isoformat(),
            "dataset_modified": datetime.fromtimestamp(self.train_modified, UTC).isoformat(),
            "feature_count": len(self.proposals),
        }


@dataclass
class QueueItem:
    """One variant waiting for (or holding) the single run slot."""

    id: str
    label: str
    project: str
    command: list[str]
    payload: dict[str, Any]
    status: str = "queued"
    run_id: str | None = None
    previous_run_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    score: float | None = None
    # Starts from the same enqueue()/start() call share a generation.  The UI
    # must not treat completed items from an older batch as the terminal state
    # of a newly started run.
    generation: int = 0
    # A single (non-batch) run still occupies the worker, but it is not part of
    # the variants queue shown by the UI and /api/comparison.
    visible: bool = True
    lock: threading.RLock = field(default_factory=threading.RLock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "label": self.label,
                "project": self.project,
                "status": self.status,
                "run_id": self.run_id,
                "strategy": self.payload.get("strategy"),
                "profile": self.payload.get("profile"),
                "recipe": self.payload.get("recipe"),
                "until": self.payload.get("until"),
                "models": list(self.payload.get("models") or []),
                "sample_pct": self.payload.get("sample_pct"),
                "llm": bool(self.payload.get("llm_plan_id")),
                "started_at": datetime.fromtimestamp(self.started_at, UTC).isoformat()
                if self.started_at
                else None,
                "finished_at": datetime.fromtimestamp(self.finished_at, UTC).isoformat()
                if self.finished_at
                else None,
                "score": self.score,
                "generation": self.generation,
                "error": self.error,
            }


class RunManager:
    """Run one ``kconv run`` subprocess at a time with a FIFO queue."""

    def __init__(self, app: WebApp) -> None:
        self.app = app
        self._current: ManagedRun | None = None
        self._current_item: QueueItem | None = None
        self._items: list[QueueItem] = []
        self._pending: deque[QueueItem] = deque()
        self._generation = 0
        self._lock = threading.RLock()

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            current = self._current
        return current.snapshot() if current is not None else None

    def queue_snapshot(self, project: str | None = None) -> dict[str, Any]:
        with self._lock:
            items = [
                item.snapshot()
                for item in self._items
                if item.visible and (project is None or item.project == project)
            ]
            active = (
                self._current is not None
                or bool(self._pending)
                or any(item["status"] in {"queued", "starting", "running", "stopping"} for item in items)
            )
            pending = sum(1 for item in items if item["status"] == "queued")
            latest_generation = max(
                (item.generation for item in self._items if project is None or item.project == project),
                default=None,
            )
        return {
            "active": active,
            "pending": pending,
            "items": items,
            "latest_generation": latest_generation,
        }

    def enqueue(
        self,
        prepared: list[tuple[str, str, list[str], dict[str, Any]]],
        project: str | None = None,
    ) -> dict[str, Any]:
        """Queue ``(label, project, command, payload)`` variants without blocking.

        The queue is intentionally unbounded: variants run sequentially, so a
        large batch only costs memory for its metadata, not GPU memory.
        """
        if not prepared:
            raise WebError("Добавьте хотя бы один вариант запуска.")
        items = [
            QueueItem(
                id=uuid.uuid4().hex[:10],
                label=label,
                project=project,
                command=command,
                payload=payload,
            )
            for label, project, command, payload in prepared
        ]
        with self._lock:
            self._generation += 1
            for item in items:
                item.generation = self._generation
            self._items.extend(items)
            self._pending.extend(items)
            self._pump_locked()
            result = self.queue_snapshot(project)
            current = self._current
        if current is not None and (project is None or current.project == project):
            result["status"] = current.snapshot()["status"]
        else:
            result["status"] = "queued"
        return result

    def start(
        self, project: str, command: list[str], payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Start immediately; reject when the queue already owns the slot."""
        with self._lock:
            if self._current is not None or self._pending:
                raise WebError(
                    "Уже выполняется другой запуск.",
                    HTTPStatus.CONFLICT,
                    "Дождитесь завершения, поставьте вариант в очередь или нажмите «Стоп».",
                )
            self._generation += 1
            item = QueueItem(
                id=uuid.uuid4().hex[:10],
                label=project,
                project=project,
                command=command,
                payload=dict(payload or {}),
                generation=self._generation,
                visible=False,
            )
            self._items.append(item)
            self._pending.append(item)
            self._pump_locked()
            run = self._current
        assert run is not None
        return run.snapshot()

    def _pump_locked(self) -> None:
        """Start the next queued item while holding ``self._lock``."""
        if self._current is not None:
            return
        while self._pending:
            item = self._pending.popleft()
            item.previous_run_id = self.app.latest_run_id(item.project)
            with item.lock:
                item.status = "starting"
                item.started_at = time.time()
                item.error = None
            run = ManagedRun(project=item.project, command=item.command)
            self._current = run
            self._current_item = item
            threading.Thread(
                target=self._execute,
                args=(run, item),
                name="kconv-web-run",
                daemon=True,
            ).start()
            return

    def _execute(self, run: ManagedRun, item: QueueItem) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        returncode: int | None = None
        log_handle = None
        log_path = self.app.data_dir / "logs" / f"web-{item.id}.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8", errors="replace")
            log_handle.write(f"# {datetime.now(UTC).isoformat()} {run.project}\n")
            log_handle.write(f"# $ {' '.join(run.command)}\n")
            log_handle.flush()
            with run.lock:
                run.log_path = log_path
        except OSError as exc:
            _LOGGER.warning("Could not open web log %s: %s", log_path, exc)
            log_handle = None
            with run.lock:
                run.log_path = None
        try:
            process = subprocess.Popen(
                run.command,
                cwd=str(self.app.root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with run.lock:
                run.process = process
                run.status = "stopping" if run.stopped else "running"
                stopped_early = run.stopped
            with item.lock:
                item.status = "stopping" if stopped_early else "running"
            if stopped_early:
                try:
                    process.terminate()
                except OSError:
                    pass
            assert process.stdout is not None
            for line in process.stdout:
                run.append(line)
                if log_handle is not None:
                    log_handle.write(line if line.endswith("\n") else line + "\n")
                    log_handle.flush()
            returncode = process.wait()
        except Exception as exc:  # pragma: no cover - depends on local process setup
            if log_handle is not None:
                log_handle.write(f"# web exception: {type(exc).__name__}: {exc}\n")
                log_handle.flush()
            with run.lock:
                run.status = "failed"
                run.error = f"{type(exc).__name__}: {exc}"
                run.finished_at = time.time()
            with item.lock:
                item.status = "failed"
                item.error = run.error
                item.finished_at = time.time()
        else:
            with run.lock:
                run.returncode = returncode
                run.finished_at = time.time()
                if run.stopped:
                    run.status = "stopped"
                elif returncode == 0:
                    run.status = "completed"
                else:
                    run.status = "failed"
                    run.error = f"kconv завершился с кодом {returncode}."
            with item.lock:
                item.status = run.status
                item.error = run.error
                item.finished_at = run.finished_at
        finally:
            if log_handle is not None:
                log_handle.close()
        self._record_result(item)
        self._persist_web_log(run, item)
        with self._lock:
            if self._current is run:
                self._current = None
                self._current_item = None
            self._pump_locked()

    def _persist_web_log(self, run: ManagedRun, item: QueueItem) -> None:
        """Copy the durable web log into the run directory and failure report."""
        with run.lock:
            log_path = run.log_path
        with item.lock:
            run_id = item.run_id
        if log_path is None or run_id is None or not log_path.is_file():
            return
        run_dir = self.app.root / "competitions" / item.project / "runs" / run_id
        if not run_dir.is_dir():
            return
        target = run_dir / "web.log"
        try:
            shutil.copyfile(log_path, target)
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _LOGGER.warning("Could not persist web log for %s: %s", run_id, exc)
            return
        failure_path = run_dir / "failure.json"
        failure = _json_load(failure_path)
        if not isinstance(failure, dict):
            return
        failure["web_log"] = "web.log"
        failure["last_log_lines"] = text.splitlines()[-200:]
        try:
            from kconv.context import write_json

            write_json(failure_path, failure)
        except Exception as exc:  # noqa: BLE001 - logging must not fail the run
            _LOGGER.warning("Could not update failure.json with web log for %s: %s", run_id, exc)

    def _record_result(self, item: QueueItem) -> None:
        run_id = self.app.latest_run_id(item.project)
        if run_id is None or run_id == item.previous_run_id:
            return
        with item.lock:
            item.run_id = run_id
        score = self.app.run_score(item.project, run_id)
        with item.lock:
            item.score = score

    def retry(self, item_id: str, project: str | None = None) -> dict[str, Any]:
        """Requeue a failed/stopped item with the same command and payload."""
        with self._lock:
            item = next((entry for entry in self._items if entry.id == item_id), None)
            if item is None:
                raise WebError("Элемент очереди не найден.", HTTPStatus.NOT_FOUND)
            if item.status not in {"failed", "stopped", "cancelled"}:
                raise WebError(
                    "Повторить можно только failed/stopped элемент.",
                    HTTPStatus.CONFLICT,
                )
            if item is self._current_item:
                raise WebError("Элемент ещё выполняется.", HTTPStatus.CONFLICT)
            self._generation += 1
            with item.lock:
                item.status = "queued"
                item.error = None
                item.started_at = None
                item.finished_at = None
                item.score = None
                item.run_id = None
                item.generation = self._generation
                item.visible = True
            self._pending.append(item)
            self._pump_locked()
            result = self.queue_snapshot(project or item.project)
            current = self._current
        if current is not None and (project is None or current.project == project):
            result["status"] = current.snapshot()["status"]
        else:
            result["status"] = "queued"
        return result

    def stop(self) -> dict[str, Any]:
        with self._lock:
            cancelled = 0
            while self._pending:
                item = self._pending.popleft()
                with item.lock:
                    if item.status == "queued":
                        item.status = "cancelled"
                        item.error = "Очередь очищена."
                        item.finished_at = time.time()
                        cancelled += 1
            run = self._current
        if run is not None:
            with run.lock:
                process = run.process
                if process is None:
                    run.stopped = True
                    run.status = "stopping"
                    result = self.queue_snapshot()
                    result["status"] = "stopping"
                    result["cancelled"] = cancelled
                    return result
                if process.poll() is not None:
                    result = self.queue_snapshot()
                    result["status"] = run.snapshot()["status"]
                    result["cancelled"] = cancelled
                    return result
                run.stopped = True
                run.status = "stopping"
            try:
                process.terminate()
            except OSError:
                pass
            threading.Thread(target=self._kill_later, args=(run, process), daemon=True).start()
            result = self.queue_snapshot()
            result["status"] = "stopping"
            result["cancelled"] = cancelled
            return result
        result = self.queue_snapshot()
        result["status"] = "idle"
        result["cancelled"] = cancelled
        return result

    @staticmethod
    def _kill_later(run: ManagedRun, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass


class WebApp:
    """State and JSON payload builders for the local UI."""

    def __init__(
        self,
        root: Path,
        data_dir: Path,
        *,
        kaggle_token_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.kaggle_token_path = (
            Path(kaggle_token_path).expanduser() if kaggle_token_path is not None else None
        )
        self.runs = RunManager(self)
        # One random token per server process.  It is deliberately not part of
        # /api/config: the browser fetches it from /api/session at startup and
        # sends it as X-Kconv-Token on every mutation.
        self.csrf_token = secrets.token_urlsafe(32)
        self._llm_plans: dict[str, LLMPlan] = {}
        self._llm_lock = threading.RLock()

    def shutdown(self) -> None:
        self.runs.stop()

    def session_payload(self) -> dict[str, Any]:
        """Return the per-server CSRF token and the last plan for this dataset."""
        return {"token": self.csrf_token, "llm_plan": self._restore_llm_plan()}

    def recover_stale_runs(self) -> list[dict[str, Any]]:
        """Finalize runs whose process died and write their failure reports."""
        from kconv.runs_index import refresh_index
        from kconv.watchdog import recover_stale_run

        recovered: list[dict[str, Any]] = []
        competitions = self.root / "competitions"
        if not competitions.is_dir():
            return recovered
        for project_dir in sorted(competitions.iterdir()):
            if not project_dir.is_dir():
                continue
            runs_dir = project_dir / "runs"
            if not runs_dir.is_dir():
                continue
            project_recovered = False
            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir() or not (run_dir / "run.json").is_file():
                    continue
                try:
                    result = recover_stale_run(project_dir, run_dir.name)
                except Exception:  # noqa: BLE001 - startup recovery must not block serving
                    _LOGGER.warning(
                        "Failed to recover stale run %s in %s",
                        run_dir.name,
                        project_dir.name,
                        exc_info=True,
                    )
                    continue
                if not result.get("recovered"):
                    continue
                project_recovered = True
                recovered.append(
                    {
                        "project": project_dir.name,
                        "run_id": str(result.get("run_id") or run_dir.name),
                        "status": result.get("status"),
                    }
                )
            if project_recovered:
                try:
                    refresh_index(project_dir)
                except Exception:  # noqa: BLE001 - index refresh is best effort
                    _LOGGER.warning(
                        "Failed to refresh the run index for %s after stale-run recovery",
                        project_dir.name,
                        exc_info=True,
                    )
        return recovered

    def _llm_manifest_path(self) -> Path:
        return self.data_dir / "llm" / "latest.json"

    def _persist_llm_plan(self, plan: LLMPlan) -> None:
        from kconv.context import write_json

        write_json(
            self._llm_manifest_path(),
            {
                "plan_id": plan.id,
                "target": plan.target,
                "id_col": plan.id_col,
                "provider": plan.provider,
                "model": plan.model,
                "train_modified": plan.train_modified,
                "train_size": plan.train_size,
                "train_sha256": plan.train_sha256,
                "created_at": plan.created_at,
            },
        )

    def _restore_llm_plan(self) -> dict[str, Any] | None:
        """Rebuild the last validated plan from disk when the dataset is unchanged."""
        manifest = _json_load(self._llm_manifest_path())
        if not isinstance(manifest, dict):
            return None
        plan_id = str(manifest.get("plan_id") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{32}", plan_id):
            return None
        train_path = self._pick("train")
        if train_path is None:
            return None
        try:
            train_stat = train_path.stat()
        except OSError:
            return None
        expected_size = manifest.get("train_size")
        expected_modified = manifest.get("train_modified")
        expected_sha = str(manifest.get("train_sha256") or "").strip()
        if not expected_sha or expected_size != train_stat.st_size:
            with self._llm_lock:
                self._llm_plans.pop(plan_id, None)
            return None
        if expected_modified is None or abs(float(expected_modified) - train_stat.st_mtime) > 0.001:
            with self._llm_lock:
                self._llm_plans.pop(plan_id, None)
            return None
        actual_sha = file_sha256(train_path)
        if actual_sha != expected_sha:
            with self._llm_lock:
                self._llm_plans.pop(plan_id, None)
            return None
        with self._llm_lock:
            cached = self._llm_plans.get(plan_id)
        if cached is not None:
            return cached.snapshot()
        plan_dir = self.data_dir / "llm"
        plan_path = plan_dir / f"plan-{plan_id}.json"
        metadata_path = plan_dir / f"plan-{plan_id}.llm.json"
        if not plan_path.is_file() or not metadata_path.is_file():
            return None
        plan_payload = _json_load(plan_path)
        if not isinstance(plan_payload, dict) or not isinstance(plan_payload.get("features"), list):
            return None
        proposals = [dict(item) for item in plan_payload["features"] if isinstance(item, dict)]
        if not proposals:
            return None
        metadata = _json_load(metadata_path)
        metadata = metadata if isinstance(metadata, dict) else {}
        warnings = metadata.get("warnings")
        usage = metadata.get("usage")
        plan = LLMPlan(
            id=plan_id,
            path=plan_path,
            metadata_path=metadata_path,
            target=str(manifest.get("target") or ""),
            id_col=str(manifest.get("id_col") or "") or None,
            provider=str(metadata.get("provider") or manifest.get("provider") or ""),
            model=str(metadata.get("model") or manifest.get("model") or "") or None,
            proposals=proposals,
            warnings=[str(item) for item in warnings] if isinstance(warnings, list) else [],
            usage=dict(usage) if isinstance(usage, dict) else {},
            train_modified=train_stat.st_mtime,
            train_size=train_stat.st_size,
            train_sha256=actual_sha,
            created_at=float(manifest.get("created_at") or time.time()),
        )
        with self._llm_lock:
            self._llm_plans[plan_id] = plan
        return plan.snapshot()

    def _pick(self, prefix: str) -> Path | None:
        candidates: list[Path] = []
        for path in self.data_dir.iterdir():
            if path.is_file() and path.name.lower().startswith(prefix.lower() + ".") and is_table(path):
                candidates.append(path)
        preferred = [f"{prefix}.csv", f"{prefix}.parquet", f"{prefix}.pq"]
        candidates.sort(key=lambda item: (preferred.index(item.name.lower()) if item.name.lower() in preferred else 99, item.name))
        return candidates[0] if candidates else None

    def files(self) -> dict[str, Path | None]:
        return {
            "train": self._pick("train"),
            "test": self._pick("test"),
            "sample": self._pick("sample_submission"),
        }

    def _require_target(self, target: str, id_col: str = "") -> tuple[Path, set[str]]:
        """Validate target/id against the current train file."""
        if not target:
            raise WebError("Выберите target-колонку.", hint="Целевая колонка должна быть в train-файле.")
        train_path = self.files()["train"]
        if train_path is None:
            raise WebError("Нет train-файла.", hint="Загрузите train.csv или train.parquet в папку data.")
        try:
            columns = {str(column) for column in safe_read_sample(train_path, nrows=5).columns}
        except Exception as exc:
            raise WebError(f"Не удалось прочитать train: {exc}") from exc
        if target not in columns:
            raise WebError(f"Target-колонки '{target}' нет в train.")
        if id_col and id_col not in columns:
            raise WebError(f"ID-колонки '{id_col}' нет в train.")
        if id_col and id_col == target:
            raise WebError("ID и target не могут быть одной колонкой.")
        return train_path, columns

    def dataset_payload(self, query: dict[str, str] | None = None) -> dict[str, Any]:
        query = query or {}
        files = self.files()
        previews: dict[str, Any] = {}
        for key, path in files.items():
            if path is not None:
                try:
                    previews[key] = _preview(path)
                except Exception as exc:
                    previews[key] = {"error": f"{type(exc).__name__}: {exc}", "rows": None, "columns": [], "head": []}
        train = files["train"]
        if train is None:
            return {
                "data_dir": str(self.data_dir),
                "files": {key: _file_info(value) for key, value in files.items()},
                "previews": previews,
                "detected": None,
                "detected_error": "Положите train.csv (или parquet) в папку data.",
                "target_summary": None,
                "project_suggestion": "my-project",
                "ready": False,
            }

        target = query.get("target") or None
        id_col = query.get("id") or None
        task = query.get("task") or None
        metric = query.get("metric") or None
        if task == "auto":
            task = None
        if metric == "auto":
            metric = None
        detected: dict[str, Any] | None = None
        detected_error: str | None = None
        try:
            spec = detect_dataset(
                self.data_dir,
                target=target,
                id=id_col,
                task=task,
                metric=metric,
            )
            detected = spec.to_dict()
        except KconvError as exc:
            detected_error = exc.message
            if exc.hint:
                detected_error += f" {exc.hint}"
        except Exception as exc:
            detected_error = f"{type(exc).__name__}: {exc}"
        target_summary = None
        target_name = target or (detected or {}).get("target")
        if target_name:
            try:
                target_summary = _target_summary(train, str(target_name))
            except Exception as exc:
                target_summary = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "data_dir": str(self.data_dir),
            "files": {key: _file_info(value) for key, value in files.items()},
            "previews": previews,
            "detected": detected,
            "detected_error": detected_error,
            "target_summary": target_summary,
            "project_suggestion": "my-project",
            "ready": True,
        }

    def discover_local_llms(self, base_url: str | None = None) -> dict[str, Any]:
        """Probe loopback OpenAI-compatible servers for the local provider."""
        checked: list[str] = []
        if base_url:
            raw = base_url.strip()
            if not _is_loopback_url(raw):
                raise WebError(
                    "Для local разрешён только loopback base URL.",
                    hint="Используйте llama.cpp/Ollama/vLLM на 127.0.0.1, localhost или ::1.",
                )
            base = normalize_local_base_url(raw)
            checked.append(base)
            models = discover_local_models(base, timeout=1.5)
            endpoints = (
                [LocalModelEndpoint(runtime="custom", base_url=base, models=models).to_dict()]
                if models
                else []
            )
        else:
            checked = [
                "http://127.0.0.1:8080/v1",
                "http://127.0.0.1:11434/v1",
                "http://127.0.0.1:8000/v1",
                "http://127.0.0.1:1234/v1",
            ]
            endpoints = [item.to_dict() for item in discover_local_endpoints(timeout=1.2)]
        return {
            "endpoints": endpoints,
            "found": bool(endpoints),
            "checked": checked,
        }

    def config_payload(self) -> dict[str, Any]:
        models = []
        for info in list_models():
            item = info.to_dict()
            item["available"] = model_available(info.id)
            models.append(item)
        recipe_error: str | None = None
        try:
            recipes = [pipeline.to_dict() for pipeline in list_pipelines()]
        except Exception as exc:
            # One stale/invalid user recipe must not blank out metrics and
            # runtime profiles in the UI.  Surface the catalog error as a
            # visible warning while keeping the rest of the config usable.
            _LOGGER.exception("Failed to load recipe catalog")
            recipe_error = str(exc)
            if getattr(exc, "hint", None):
                recipe_error = f"{recipe_error} {exc.hint}"
            recipes = []
        return {
            "models": models,
            "recipes": recipes,
            "pipeline_presets": pipeline_presets(),
            "recipe_error": recipe_error,
            "profiles": [profile.to_dict() for profile in list_profiles()],
            "metrics": [
                {"name": name, "direction": get_metric(name).direction} for name in metric_names()
            ],
            "tasks": ["auto", "binary", "multiclass", "regression"],
            "data_dir": str(self.data_dir),
            "llm": {
                "openai_key_env": "KCONV_LLM_API_KEY",
                "openai_configured": bool(
                    os.environ.get("KCONV_LLM_API_KEY", "").strip()
                    or os.environ.get("OPENAI_API_KEY", "").strip()
                ),
                "local_default_url": "",
                "local_auto": True,
            },
        }

    def stats_payload(self) -> dict[str, Any]:
        models = list_models()
        by_source = Counter(item.source for item in models)
        by_family = Counter(item.family for item in models)
        by_task = {
            task: sum(task in item.tasks for item in models)
            for task in ("binary", "multiclass", "regression")
        }
        return {
            "models": {
                "total": len(models),
                "by_source": dict(by_source),
                "by_family": dict(by_family),
                "by_task": by_task,
            },
            "recipes": len(list_pipelines()),
            "profiles": len(list_profiles()),
            "experiments": len(list_experiments()),
            "sources": catalog_stats(),
            "dataset": self.dataset_payload(),
            "run": self.runs.snapshot(),
        }

    def projects_payload(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        competitions = self.root / "competitions"
        if not competitions.is_dir():
            return items
        for project_dir in competitions.iterdir():
            if not project_dir.is_dir():
                continue
            pointer = _json_load(project_dir / "latest.json")
            if not isinstance(pointer, dict) or not pointer.get("run_id"):
                continue
            items.append(
                {
                    "project": project_dir.name,
                    "run_id": str(pointer["run_id"]),
                    "status": pointer.get("status"),
                    "updated_at": pointer.get("updated_at"),
                }
            )
        return sorted(items, key=lambda item: item.get("updated_at") or "", reverse=True)

    def latest_run_id(self, project: str) -> str | None:
        pointer = _json_load(self.root / "competitions" / project / "latest.json")
        if isinstance(pointer, dict) and pointer.get("run_id"):
            return str(pointer["run_id"])
        return None

    def run_score(self, project: str, run_id: str | None) -> float | None:
        if not run_id:
            return None
        blend = _json_load(self.root / "competitions" / project / "runs" / run_id / "03_blend" / "blend.json")
        if not isinstance(blend, dict):
            return None
        value = blend.get("oof_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    @staticmethod
    def _dataset_identity(run_dir: Path, config: dict[str, Any]) -> dict[str, str]:
        """Return a stable dataset id and a human label for one run.

        New runs carry a data card written by the ingest stage.  Older runs
        without one fall back to the configured path/target.  The signature
        deliberately excludes run-specific fields (created_at, config hash,
        model list), so several pipelines on the same dataset share a group.
        """
        card = _json_load(run_dir / "00_ingest" / "data_card.json")
        if isinstance(card, dict):
            source = card.get("source") if isinstance(card.get("source"), dict) else {}
            rows = card.get("rows") if isinstance(card.get("rows"), dict) else {}
            train_source = source.get("train")
            signature = {
                # Dataset identity must survive different recipes, tasks and
                # feature plans.  Roles/feature columns, missing-cell counts,
                # task labels and class balance change when an LLM plan, an
                # ignore list or a different target task is applied, so they
                # cannot be part of the grouping key.
                "train": _basename(train_source),
                "test": _basename(source.get("test")),
                "target": card.get("target"),
                "id_col": card.get("id_col"),
                "rows_train": rows.get("train"),
                "rows_test": rows.get("test"),
            }
            digest = hashlib.sha256(
                json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()[:16]
            target = card.get("target") or (config.get("data") or {}).get("target") or "target"
            train_count = rows.get("train")
            label = f"{_basename(train_source)} · {target}"
            if isinstance(train_count, int):
                label += f" · {train_count:,} rows"
            return {"id": digest, "label": label}
        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        train = data.get("train") or "dataset"
        target = data.get("target") or "target"
        digest = hashlib.sha256(
            json.dumps({"train": str(train), "target": target}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return {"id": digest, "label": f"{_basename(train)} · {target}"}

    def comparison_payload(self, project: str, dataset: str | None = None) -> dict[str, Any]:
        """Summarise runs, grouped and ranked per dataset within a project."""
        project = _slug(project)
        project_dir = self.root / "competitions" / project
        latest_run_id = self.latest_run_id(project)
        rows: list[dict[str, Any]] = []
        runs_dir = project_dir / "runs"
        run_dirs = sorted((path for path in runs_dir.iterdir() if path.is_dir()), key=lambda path: path.name) if runs_dir.is_dir() else []
        for position, run_dir in enumerate(run_dirs, start=1):
            meta = _json_load(run_dir / "run.json")
            if not isinstance(meta, dict):
                continue
            config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
            validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
            hpo = config.get("hpo") if isinstance(config.get("hpo"), dict) else {}
            blend = _json_load(run_dir / "03_blend" / "blend.json")
            state = _json_load(run_dir / "state.json")
            blend = blend if isinstance(blend, dict) else {}
            state = state if isinstance(state, dict) else {}
            raw_stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
            first_error = next(
                (
                    stage.get("error")
                    for stage in raw_stages.values()
                    if isinstance(stage, dict) and stage.get("error")
                ),
                None,
            )
            models = [
                str(model.get("name"))
                for model in config.get("models", [])
                if isinstance(model, dict) and model.get("name")
            ]
            seeds = max(
                (
                    int(model.get("seeds") or 0)
                    for model in config.get("models", [])
                    if isinstance(model, dict)
                ),
                default=0,
            )
            oof_score = blend.get("oof_score")
            if not isinstance(oof_score, (int, float)) or not math.isfinite(float(oof_score)):
                oof_score = None
            identity = self._dataset_identity(run_dir, config)
            rows.append(
                {
                    "index": position,
                    "run_id": run_dir.name,
                    "dataset_id": identity["id"],
                    "dataset_label": identity["label"],
                    "status": meta.get("status") or "unknown",
                    "strategy": config.get("strategy") or meta.get("strategy"),
                    "profile": (config.get("runtime") or {}).get("profile")
                    if isinstance(config.get("runtime"), dict)
                    else None,
                    "recipe": config.get("recipe") or meta.get("recipe"),
                    "task": config.get("task"),
                    "metric": blend.get("metric") or config.get("metric"),
                    "models": models,
                    "n_splits": validation.get("n_splits"),
                    "seeds": seeds,
                    "hpo": bool(hpo.get("enabled")),
                    "oof_score": oof_score,
                    "best_single": blend.get("best_single"),
                    "blend_method": blend.get("method"),
                    "started_at": meta.get("started_at"),
                    "finished_at": meta.get("finished_at"),
                    "duration_sec": round(
                        sum(
                            float(value)
                            for value in (meta.get("durations_sec") or {}).values()
                            if isinstance(value, (int, float))
                        ),
                        1,
                    ),
                    "has_submission": (run_dir / "04_submit" / "submission.csv").is_file(),
                    "has_report": (run_dir / "05_report" / "report.md").is_file(),
                    "error": first_error,
                    "latest": run_dir.name == latest_run_id,
                }
            )

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row["dataset_id"]), []).append(row)

        datasets: list[dict[str, Any]] = []
        for dataset_id, group_rows in groups.items():
            latest_row = next(
                (
                    row
                    for row in reversed(group_rows)
                    if row["run_id"] == latest_run_id and row.get("metric")
                ),
                None,
            )
            metric = latest_row["metric"] if latest_row else next(
                (row["metric"] for row in group_rows if row.get("metric")),
                None,
            )
            try:
                direction = get_metric(str(metric)).direction if metric else "maximize"
            except Exception:
                direction = "maximize"
            scored = [
                row
                for row in group_rows
                if row["oof_score"] is not None and (metric is None or row["metric"] == metric)
            ]
            best = None
            if scored:
                pick = min if direction == "minimize" else max
                best = pick(scored, key=lambda row: float(row["oof_score"]))
            for row in group_rows:
                row["best"] = best is not None and row["run_id"] == best["run_id"]
            datasets.append(
                {
                    "id": dataset_id,
                    "label": group_rows[0]["dataset_label"],
                    "metric": metric,
                    "direction": direction,
                    "runs": len(group_rows),
                    "best_run_id": best["run_id"] if best else None,
                }
            )

        selected = dataset if dataset and dataset in groups else None
        visible_rows = groups[selected] if selected else rows
        selected_meta = next((item for item in datasets if item["id"] == selected), None)
        if selected_meta is not None:
            metric = selected_meta["metric"]
            direction = selected_meta["direction"]
            best_run_id = selected_meta["best_run_id"]
        elif len(datasets) == 1:
            metric = datasets[0]["metric"]
            direction = datasets[0]["direction"]
            best_run_id = datasets[0]["best_run_id"]
        else:
            metric = next((row["metric"] for row in rows if row.get("metric")), None)
            direction = "maximize"
            best_run_id = None
            try:
                direction = get_metric(str(metric)).direction if metric else "maximize"
            except Exception:
                direction = "maximize"
        return {
            "project": project,
            "metric": metric,
            "direction": direction,
            "latest_run_id": latest_run_id,
            "best_run_id": best_run_id,
            "dataset": selected,
            "datasets": datasets,
            "best_by_dataset": datasets,
            "runs": visible_rows,
        }

    def _disk_status(self, project: str) -> dict[str, Any]:
        project_dir = self.root / "competitions" / project
        pointer = _json_load(project_dir / "latest.json")
        run_id = str(pointer["run_id"]) if isinstance(pointer, dict) and pointer.get("run_id") else None
        if run_id is None:
            return {
                "project": project,
                "status": "idle",
                "run_id": None,
                "stages": [
                    {"name": name, "status": "pending", "duration_sec": None, "error": None}
                    for name in STAGE_ORDER
                ],
                "metrics": None,
                "report": None,
                "report_html": None,
                "submission_preview": None,
                "has_submission": False,
                "has_report": False,
                "failure_report": None,
                "failure_json": None,
                "failure_available": False,
                "stale_reason": None,
                "heartbeat_age_sec": None,
                "error": None,
            }
        run_dir = project_dir / "runs" / run_id
        state = _json_load(run_dir / "state.json") or {}
        run_meta = _json_load(run_dir / "run.json") or {}
        raw_stages = state.get("stages") if isinstance(state, dict) else {}
        stages: list[dict[str, Any]] = []
        for name in STAGE_ORDER:
            raw = raw_stages.get(name, {}) if isinstance(raw_stages, dict) else {}
            stages.append(
                {
                    "name": name,
                    "status": raw.get("status", "pending"),
                    "duration_sec": raw.get("duration_sec"),
                    "error": raw.get("error"),
                }
            )
        blend = _json_load(run_dir / "03_blend" / "blend.json")
        metrics_payload: dict[str, Any] | None = None
        if isinstance(blend, dict):
            metric_name = str(blend.get("metric") or "")
            try:
                direction = get_metric(metric_name).direction
            except Exception:
                direction = None
            model_rows = []
            for model_dir in sorted((run_dir / "02_train").glob("*")):
                model_metrics = _json_load(model_dir / "metrics.json")
                if not isinstance(model_metrics, dict):
                    continue
                model_rows.append(
                    {
                        "name": model_metrics.get("model", model_dir.name),
                        "oof_score": model_metrics.get("oof_score"),
                        "mean_fold_score": model_metrics.get("mean_fold_score"),
                        "duration_sec": model_metrics.get("duration_sec"),
                        "seeds": model_metrics.get("seeds"),
                    }
                )
            metrics_payload = {
                "blend": blend,
                "models": model_rows,
                "metric": metric_name or None,
                "direction": direction,
            }
        submission_check = _json_load(run_dir / "04_submit" / "submission_check.json")
        submission_preview = None
        submission_path = run_dir / "04_submit" / "submission.csv"
        if submission_path.is_file():
            try:
                frame = pd.read_csv(submission_path, nrows=5)
                submission_preview = {
                    "columns": [str(column) for column in frame.columns],
                    "rows": _jsonable(frame.to_dict(orient="records")),
                }
            except Exception:
                submission_preview = None
        report_path = run_dir / "05_report" / "report.md"
        report = None
        report_html = None
        if report_path.is_file():
            try:
                report = report_path.read_text(encoding="utf-8")[:200_000]
            except OSError:
                report = None
            if report:
                report_html = render_markdown(report)
        status = run_meta.get("status") or (pointer.get("status") if isinstance(pointer, dict) else None) or "unknown"
        failure_report = run_meta.get("failure_report")
        failure_json = run_meta.get("failure_json")
        stale_reason = None
        heartbeat_age = None
        try:
            from kconv.watchdog import inspect_run

            diagnosis = inspect_run(project_dir, run_id)
            stale_reason = diagnosis.get("reason") if diagnosis.get("stale") else None
            heartbeat_age = diagnosis.get("heartbeat_age_sec")
        except Exception:
            pass
        return {
            "project": project,
            "status": status,
            "run_id": run_id,
            "failure_report": failure_report,
            "failure_json": failure_json,
            "failure_available": bool((run_dir / str(failure_report)).is_file()) if failure_report else False,
            "stale_reason": stale_reason,
            "heartbeat_age_sec": heartbeat_age,
            "until": run_meta.get("until"),
            "started_at": run_meta.get("started_at"),
            "finished_at": run_meta.get("finished_at"),
            "elapsed_sec": round(sum(float(v) for v in (run_meta.get("durations_sec") or {}).values()), 1),
            "stages": stages,
            "metrics": metrics_payload,
            "submission_check": submission_check,
            "submission_preview": submission_preview,
            "report": report,
            "report_html": report_html,
            "has_submission": submission_path.is_file(),
            "has_report": report_path.is_file(),
            "error": next((stage["error"] for stage in stages if stage["error"]), None),
        }

    def run_status(self, project: str | None = None) -> dict[str, Any]:
        snapshot = self.runs.snapshot()
        requested = project or (snapshot.get("project") if snapshot else None) or "my-project"
        requested = _slug(requested)
        disk = self._disk_status(requested)
        queue = self.runs.queue_snapshot(requested)
        server_busy = snapshot is not None or self.runs.queue_snapshot()["active"]

        def queue_terminal_status() -> str | None:
            """Prefer the queue outcome when the batch has just drained.

            The CLI writes its own latest pointer asynchronously; a batch can
            become inactive a few milliseconds before that pointer says
            ``completed``.  Queue items are updated before the pump clears the
            active slot, so they are the reliable terminal signal here.
            """
            items = queue["items"]
            latest_generation = queue.get("latest_generation")
            if latest_generation is not None:
                items = [item for item in items if item.get("generation") == latest_generation]
            if not items or queue["active"] or queue["pending"]:
                return None
            statuses = {item["status"] for item in items}
            if statuses == {"completed"}:
                return "completed"
            if "failed" in statuses:
                return "failed"
            if statuses & {"stopped", "cancelled"}:
                return "stopped"
            return None

        if snapshot is None or snapshot.get("project") != requested:
            merged = dict(disk)
            if any(item["status"] in {"queued", "starting", "running"} for item in queue["items"]):
                merged["status"] = "queued" if queue["pending"] else "running"
            terminal = queue_terminal_status()
            if terminal is not None:
                if terminal == "completed" and disk.get("status") == "completed_with_warnings":
                    terminal = "completed_with_warnings"
                merged["status"] = terminal
            merged["queue"] = queue["items"]
            merged["batch_active"] = queue["active"]
            merged["queue_pending"] = queue["pending"]
            merged["server_active"] = queue["active"]
            merged["server_busy"] = server_busy
            merged["stale"] = bool(disk.get("stale_reason")) or (
                disk.get("status") == "running" and not queue["active"] and terminal is None
            )
            return merged
        merged = {**disk, **snapshot}
        if snapshot["status"] in {"starting", "running", "stopping"}:
            merged["status"] = "running" if snapshot["status"] == "starting" else snapshot["status"]
        elif snapshot["status"] in {"completed", "completed_with_warnings", "failed", "stopped"}:
            merged["status"] = snapshot["status"]
        if snapshot["status"] in {"completed", "completed_with_warnings", "failed", "stopped"} and queue["pending"]:
            merged["status"] = "queued"
        terminal = queue_terminal_status()
        if terminal is not None:
            if terminal == "completed" and disk.get("status") == "completed_with_warnings":
                terminal = "completed_with_warnings"
            merged["status"] = terminal
        if merged.get("status") == "completed" and disk.get("status") == "completed_with_warnings":
            merged["status"] = "completed_with_warnings"
        merged["disk_status"] = disk.get("status")
        merged["stages"] = disk.get("stages", merged.get("stages"))
        merged["metrics"] = disk.get("metrics")
        merged["report"] = disk.get("report")
        merged["report_html"] = disk.get("report_html")
        merged["submission_preview"] = disk.get("submission_preview")
        merged["has_submission"] = disk.get("has_submission", False)
        merged["has_report"] = disk.get("has_report", False)
        merged["failure_report"] = disk.get("failure_report")
        merged["failure_json"] = disk.get("failure_json")
        merged["failure_available"] = disk.get("failure_available", False)
        merged["stale_reason"] = disk.get("stale_reason")
        merged["heartbeat_age_sec"] = disk.get("heartbeat_age_sec")
        merged["queue"] = queue["items"]
        merged["batch_active"] = queue["active"]
        merged["queue_pending"] = queue["pending"]
        merged["server_active"] = True
        merged["server_busy"] = server_busy
        merged["stale"] = bool(disk.get("stale_reason"))
        return merged

    def _run_dir(self, project: str, run_id: str | None = None) -> Path | None:
        project_dir = self.root / "competitions" / _slug(project)
        if run_id is None:
            pointer = _json_load(project_dir / "latest.json")
            run_id = str(pointer["run_id"]) if isinstance(pointer, dict) and pointer.get("run_id") else None
        if run_id is None:
            return None
        try:
            run_id = validate_run_id(str(run_id))
        except KconvError:
            return None
        runs_dir = (project_dir / "runs").resolve()
        run_dir = (runs_dir / run_id).resolve()
        try:
            run_dir.relative_to(runs_dir)
        except ValueError:
            return None
        return run_dir if run_dir.is_dir() else None

    def submission_path(self, project: str, run_id: str | None = None) -> Path | None:
        run_dir = self._run_dir(project, run_id)
        if run_dir is None:
            return None
        path = run_dir / "04_submit" / "submission.csv"
        return path if path.is_file() else None

    def report_path(self, project: str, run_id: str | None = None) -> Path | None:
        run_dir = self._run_dir(project, run_id)
        if run_dir is None:
            return None
        path = run_dir / "05_report" / "report.md"
        return path if path.is_file() else None

    def kaggle_status_payload(self, project: str | None = None) -> dict[str, Any]:
        """Return safe Kaggle auth metadata and submission availability."""
        project = _slug(str(project or "my-project"))
        auth = check_auth(token_path=self.kaggle_token_path)
        submission = self.submission_path(project)
        run_id = self.latest_run_id(project)
        source = auth.credential_source or ""
        return {
            "project": project,
            "run_id": run_id,
            "has_submission": submission is not None,
            "submission": _file_info(submission),
            "auth": auth.to_dict(),
            "environment_override": source.startswith("environment:"),
        }

    def save_kaggle_credentials(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist a Kaggle token or legacy key file without echoing the secret."""
        project = _slug(str(payload.get("project") or "my-project"))
        token = str(payload.get("token") or "")
        username = str(payload.get("username") or "")
        key = str(payload.get("key") or "")
        try:
            path = save_credentials(
                username,
                key,
                token=token,
                token_path=self.kaggle_token_path,
            )
        except KconvError as exc:
            raise WebError(exc.message, hint=exc.hint) from exc
        status = self.kaggle_status_payload(project)
        status["saved"] = True
        status["saved_path"] = str(path)
        status["config_path"] = str(path)
        return status

    def submit_kaggle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Upload the selected submission after an explicit button press."""
        project = _slug(str(payload.get("project") or ""))
        competition = str(payload.get("competition") or "").strip()
        if not competition:
            raise WebError(
                "Укажите Kaggle competition slug.",
                hint="Например: titanic. Slug виден в URL соревнования.",
            )
        run_id = str(payload.get("run_id") or "").strip() or None
        submission = self.submission_path(project, run_id)
        if submission is None:
            raise WebError(
                "Submission ещё не создан.",
                status=HTTPStatus.NOT_FOUND,
                hint="Дождитесь полного прогона до стадии submit или выберите другой run.",
            )
        auth = check_auth(token_path=self.kaggle_token_path)
        if not auth.ready:
            raise WebError(
                "Kaggle CLI или credentials не готовы.",
                hint="Сохраните API token (или username/key) в панели Kaggle и убедитесь, что установлен Kaggle CLI.",
            )
        message = str(payload.get("message") or "kconv submission").strip()
        try:
            result = submit_submission(
                competition,
                submission,
                message=message,
                execute=True,
                auth=auth,
            )
        except KconvError as exc:
            status = HTTPStatus.BAD_GATEWAY if "Kaggle CLI" in exc.message else HTTPStatus.BAD_REQUEST
            raise WebError(exc.message, status=status, hint=exc.hint) from exc
        response = result.to_dict()
        response["project"] = project
        response["run_id"] = run_id
        return response

    def propose_llm_features(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate and validate a plan through the existing CLI experiment.

        The subprocess boundary keeps the web server and the CLI on the same
        validation path.  Only a reviewed, server-held plan id can reach a run.
        """
        target = str(payload.get("target") or "").strip()
        id_col = str(payload.get("id_col") or "").strip()
        train_path, _columns = self._require_target(target, id_col)
        provider = str(payload.get("provider") or "mock").strip().lower()
        if provider not in {"mock", "openai", "local"}:
            raise WebError(
                "Провайдер LLM должен быть mock, openai или local.",
                hint="mock работает offline; openai нужен ключ; local — Ollama/vLLM на loopback.",
            )
        try:
            max_features = int(payload.get("max_features") or 5)
        except (TypeError, ValueError) as exc:
            raise WebError("max_features должен быть числом 1-20.") from exc
        if not 1 <= max_features <= 20:
            raise WebError("max_features должен быть числом 1-20.")
        try:
            max_tokens = int(payload.get("max_tokens") or 1024)
        except (TypeError, ValueError) as exc:
            raise WebError("max_tokens должен быть числом 1-8192.") from exc
        if not 1 <= max_tokens <= 8192:
            raise WebError("max_tokens должен быть числом 1-8192.")
        json_mode = payload.get("json_mode", True)
        if not isinstance(json_mode, bool):
            raise WebError("json_mode должен быть true или false.")
        raw_ideas = payload.get("ideas") or []
        if isinstance(raw_ideas, str):
            raw_ideas = raw_ideas.splitlines()
        if not isinstance(raw_ideas, list):
            raise WebError("ideas должен быть списком строк.")
        ideas = [str(item).strip() for item in raw_ideas if str(item).strip()]
        if len(ideas) > 20:
            raise WebError("Можно передать не более 20 идей.")
        if any(len(item) > 500 for item in ideas):
            raise WebError("Каждая идея должна быть не длиннее 500 символов.")
        code_text = str(payload.get("code") or "").strip()
        if len(code_text) > 12000:
            raise WebError("Код/псевдокод должен быть не длиннее 12000 символов.")
        try:
            repairs = int(payload.get("repairs", 2))
        except (TypeError, ValueError) as exc:
            raise WebError("repairs должен быть числом 0-5.") from exc
        if not 0 <= repairs <= 5:
            raise WebError("repairs должен быть числом 0-5.")
        model = str(payload.get("model") or "").strip()
        base_url = str(payload.get("base_url") or "").strip()
        if model and (
            len(model) > 500
            or model.startswith("-")
            or "\n" in model
            or "\r" in model
            or (provider != "local" and any(ch.isspace() for ch in model))
        ):
            raise WebError("Некорректное имя модели LLM.")
        if base_url and (len(base_url) > 500 or any(ch.isspace() for ch in base_url)):
            raise WebError("Некорректный base URL LLM.")
        if provider == "openai" and not model:
            raise WebError(
                "Для provider=openai укажите модель.",
                hint="Например: gpt-4o-mini. Ключ читается из KCONV_LLM_API_KEY или OPENAI_API_KEY.",
            )
        if provider == "local":
            if base_url:
                base_url = normalize_local_base_url(base_url)
                if not _is_loopback_url(base_url):
                    raise WebError(
                        "Для provider=local разрешён только loopback base URL.",
                        hint="Используйте llama.cpp/Ollama/vLLM на 127.0.0.1, localhost или ::1.",
                    )
            try:
                base_url, model = resolve_local_target(base_url or None, model or None)
            except KconvError as exc:
                raise WebError(
                    "Не удалось определить локальный LLM.",
                    hint=(
                        "Запустите llama.cpp: llama-server -m model.gguf --port 8080, "
                        "Ollama :11434 или vLLM :8000; base URL и model можно задать вручную. "
                        f"Детали: {exc} {getattr(exc, 'hint', '') or ''}"
                    ).strip(),
                ) from exc
        plan_id = uuid.uuid4().hex
        plan_dir = self.data_dir / "llm"
        plan_path = plan_dir / f"plan-{plan_id}.json"
        metadata_path = plan_dir / f"plan-{plan_id}.llm.json"
        code_path = plan_dir / f"code-{plan_id}.py"
        try:
            plan_dir.mkdir(parents=True, exist_ok=True)
            if code_text:
                code_path.write_text(code_text, encoding="utf-8")
            train_before = train_path.stat()
        except OSError as exc:
            raise WebError(f"Не удалось подготовить папку LLM-плана: {exc}") from exc
        command = [
            sys.executable,
            "-m",
            "kconv.cli",
            "experiments",
            "propose",
            "--data",
            str(self.data_dir),
            "--target",
            target,
            "--provider",
            provider,
            "--max-features",
            str(max_features),
            "--llm-max-tokens",
            str(max_tokens),
            # The CLI default (60 s) is too short for a local 7B model on CPU;
            # keep a margin below the subprocess timeout below.
            "--llm-timeout",
            "170",
            "--output",
            str(plan_path),
            "--llm-log",
            str(metadata_path),
        ]
        if id_col:
            command += ["--id", id_col]
        if provider == "openai":
            command += ["--llm-model", model]
        elif provider == "local":
            command += ["--llm-model", model, "--llm-base-url", base_url]
        if not json_mode:
            command.append("--no-llm-json-mode")
        for idea in ideas:
            command += ["--llm-idea", idea]
        if code_text:
            command += ["--llm-code-file", str(code_path)]
        if repairs != 2:
            command += ["--llm-repairs", str(repairs)]
        overall_timeout = min(
            3600,
            max(180, max_features * (repairs + 1) * 25 + 120),
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(
                command,
                cwd=str(self.root),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=overall_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            plan_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            code_path.unlink(missing_ok=True)
            raise WebError(
                f"LLM не ответил за {overall_timeout} секунд.",
                hint="Проверьте base URL, модель и доступность провайдера; план не сохранён.",
                status=HTTPStatus.GATEWAY_TIMEOUT,
            ) from exc
        except Exception as exc:
            plan_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            code_path.unlink(missing_ok=True)
            raise WebError(f"Не удалось запустить генерацию LLM-плана: {exc}") from exc
        code_path.unlink(missing_ok=True)
        if result.returncode != 0:
            plan_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raw = (result.stderr or result.stdout or "").strip()
            raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
            detail = " ".join(line.strip() for line in raw.splitlines() if line.strip())
            if not detail:
                detail = f"CLI завершился с кодом {result.returncode}."
            raise _llm_web_error(detail)
        plan_payload = _json_load(plan_path)
        metadata = _json_load(metadata_path)
        if not isinstance(plan_payload, dict) or not isinstance(plan_payload.get("features"), list):
            plan_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise WebError("LLM вернул план без списка features.")
        try:
            train_after = train_path.stat()
        except OSError as exc:
            plan_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise WebError(f"Train исчез во время генерации плана: {exc}") from exc
        train_sha256 = file_sha256(train_path)
        if train_after.st_size != train_before.st_size or abs(train_after.st_mtime - train_before.st_mtime) > 0.001:
            plan_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise WebError(
                "Train изменился во время генерации плана.",
                hint="Сгенерируйте план ещё раз на актуальном файле.",
            )
        metadata = metadata if isinstance(metadata, dict) else {}
        warnings = metadata.get("warnings")
        usage = metadata.get("usage")
        proposals = [dict(item) for item in plan_payload["features"] if isinstance(item, dict)]
        if not proposals:
            plan_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise WebError("LLM вернул пустой список признаков.", hint="Попробуйте другую модель или mock.")
        plan = LLMPlan(
            id=plan_id,
            path=plan_path,
            metadata_path=metadata_path,
            target=target,
            id_col=id_col or None,
            provider=str(metadata.get("provider") or provider),
            model=str(metadata.get("model") or model or "") or None,
            proposals=proposals,
            warnings=[str(item) for item in warnings] if isinstance(warnings, list) else [],
            usage=dict(usage) if isinstance(usage, dict) else {},
            train_modified=train_after.st_mtime,
            train_size=train_after.st_size,
            train_sha256=train_sha256,
        )
        with self._llm_lock:
            self._llm_plans[plan_id] = plan
        self._persist_llm_plan(plan)
        return plan.snapshot()

    def build_run_command(
        self, payload: dict[str, Any]
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Validate one variant and turn it into a ``kconv run`` argv."""
        payload = dict(payload)
        project = _slug(str(payload.get("project") or "my-project"))
        target = str(payload.get("target") or "").strip()
        requested_id = str(payload.get("id_col") or "").strip()
        train_path, _columns = self._require_target(target, requested_id)
        raw_models = payload.get("models")
        if isinstance(raw_models, str):
            models = [item.strip() for item in raw_models.split(",") if item.strip()]
        elif isinstance(raw_models, list):
            models = [str(item).strip() for item in raw_models if str(item).strip()]
        else:
            models = []
        models = list(dict.fromkeys(models))
        if not models:
            raise WebError("Выберите хотя бы одну модель.")
        known = set(model_names())
        for name in models:
            if name not in known:
                raise WebError(f"Неизвестная модель: {name}.")
            if not model_available(name):
                raise WebError(
                    f"Модель {name} недоступна в этом окружении.",
                    hint="Установите optional extra или выберите другую модель.",
                )
        if "tabpfn" in models:
            train_rows = _table_row_count(train_path)
            if train_rows is not None and train_rows > 8000:
                raise WebError(
                    f"TabPFN не запускается на {train_rows:,} строках в web UI.",
                    hint="Безопасный лимит — 8 000 train rows. Снимите tabpfn или используйте CLI на своей ответственности.",
                )
        task = str(payload.get("task") or "auto")
        if task not in {"auto", "binary", "multiclass", "regression"}:
            raise WebError("Задача должна быть auto, binary, multiclass или regression.")
        if task != "auto":
            target_summary = _target_summary(train_path, target)
            nunique = target_summary.get("nunique") if target_summary else None
            if task == "binary" and nunique is not None and int(nunique) != 2:
                raise WebError(
                    f"Target '{target}' содержит {nunique} классов; binary не подходит.",
                    hint="Выберите multiclass или regression — либо проверьте target-колонку.",
                )
            if task == "multiclass" and nunique is not None and int(nunique) == 2:
                raise WebError(
                    f"Target '{target}' содержит 2 класса; для него нужна задача binary.",
                    hint="multiclass с двумя классами даёт другую семантику ROC AUC; выберите binary.",
                )
            for name in models:
                info = next((item for item in list_models() if item.id == name), None)
                if info is not None and task not in info.tasks:
                    raise WebError(f"Модель {name} не поддерживает задачу {task}.", hint="Снимите её или измените задачу.")
        metric = str(payload.get("metric") or "auto")
        if metric != "auto" and metric not in metric_names():
            raise WebError(f"Неизвестная метрика: {metric}.")
        raw_profile = payload.get("profile")
        if raw_profile is None or raw_profile == "":
            profile = None
        elif raw_profile == "0":
            profile = None
        elif isinstance(raw_profile, int) and not isinstance(raw_profile, bool) and raw_profile == 0:
            profile = None
        else:
            profile = _optional_payload_int(
                payload, "profile", label="Профиль", minimum=1, maximum=5
            )
        strategy = str(payload.get("strategy") or "standard")
        if profile is None and strategy not in {"fast", "standard", "deep"}:
            raise WebError("Стратегия должна быть fast, standard или deep.")
        device = str(payload.get("device") or "").strip().lower() or None
        if device is not None:
            if device not in {"auto", "cpu", "gpu", "cuda"}:
                raise WebError("Устройство должно быть auto, cpu, gpu или cuda.")
            if profile is None:
                raise WebError(
                    "Устройство можно переопределить только вместе с профилем.",
                    hint="Выберите профиль 1-5; профили 1-3 работают только на CPU.",
                )
            if device in {"gpu", "cuda"} and profile <= 3:
                raise WebError(
                    f"Профиль {profile} — CPU-only.",
                    hint="Выберите профиль 4 или 5 для GPU/CUDA.",
                )
        cpu_threads = _optional_payload_int(
            payload, "cpu_threads", label="CPU threads", minimum=0, maximum=256
        )
        ram_gb = _optional_payload_int(payload, "ram_gb", label="RAM GB", minimum=1, maximum=1024)
        vram_gb = _optional_payload_int(payload, "vram_gb", label="VRAM GB", minimum=0, maximum=256)
        sample_pct = _optional_payload_int(
            payload, "sample_pct", label="Sample %", minimum=1, maximum=100
        )
        if profile is None and any(value is not None for value in (cpu_threads, ram_gb, vram_gb)):
            raise WebError(
                "Лимиты CPU/RAM/VRAM можно задать только вместе с профилем.",
                hint="Выберите профиль 1-5, затем уточните его ресурсы.",
            )
        manual_params = _validated_model_params(payload, models)
        # "CV only" still needs the blend stage: it produces the OOF score and
        # the ensemble data used by the comparison table.
        until = "report" if payload.get("until") == "report" else "blend"
        if until == "report" and self.files()["test"] is None:
            raise WebError(
                "Для полного прогона нужен test-файл.",
                hint="Загрузите test.csv или выберите «Только CV».",
            )
        recipe = str(payload.get("recipe") or "pesnik/baseline")
        if recipe not in {item.id for item in list_pipelines()}:
            raise WebError(f"Неизвестный recipe: {recipe}.")
        command = [
            sys.executable,
            "-m",
            "kconv.cli",
            "run",
            "--data",
            str(self.data_dir),
            "--target",
            target,
            "--models",
            ",".join(models),
            "--name",
            project,
            "--until",
            until,
            "--recipe",
            recipe,
            "--sample-pct",
            str(100 if sample_pct is None else sample_pct),
        ]
        id_col = str(payload.get("id_col") or "").strip()
        if id_col:
            command += ["--id", id_col]
        if task != "auto":
            command += ["--task", task]
        if metric != "auto":
            command += ["--metric", metric]
        if profile is not None:
            command += ["--profile", str(profile)]
            if device is not None:
                command += ["--device", device]
            if cpu_threads is not None:
                command += ["--cpu-threads", str(cpu_threads)]
            if ram_gb is not None:
                command += ["--ram-gb", str(ram_gb)]
            if vram_gb is not None:
                command += ["--vram-gb", str(vram_gb)]
        else:
            command += ["--strategy", strategy]
        if manual_params:
            command += [
                "--model-params",
                json.dumps(manual_params, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            ]
        if payload.get("force"):
            command += ["--force"]
        llm_plan_id = str(payload.get("llm_plan_id") or "").strip()
        llm_approved = payload.get("llm_approved") is True
        if llm_plan_id:
            if not llm_approved:
                raise WebError(
                    "LLM-план не подтверждён.",
                    hint="Проверьте формулы и отметьте «Я проверил(а) формулы».",
                )
            with self._llm_lock:
                plan = self._llm_plans.get(llm_plan_id)
            if plan is None:
                raise WebError(
                    "LLM-план не найден или устарел.",
                    hint="Сгенерируйте план заново: сервер хранит планы только в текущей сессии.",
                )
            if plan.target != target or (plan.id_col or "") != requested_id:
                raise WebError(
                    "LLM-план относится к другому target или ID.",
                    hint="Сгенерируйте новый план после изменения target/ID.",
                )
            if not plan.path.is_file() or not plan.metadata_path.is_file():
                raise WebError(
                    "Файлы LLM-плана больше не доступны.",
                    hint="Сгенерируйте план заново — он не был использован.",
                )
            try:
                train_stat = train_path.stat()
            except OSError as exc:
                raise WebError(f"Не удалось проверить train перед запуском: {exc}") from exc
            if train_stat.st_size != plan.train_size or abs(train_stat.st_mtime - plan.train_modified) > 0.001:
                raise WebError(
                    "Train изменился после генерации LLM-плана.",
                    hint="Сгенерируйте план заново, чтобы формулы проверились на актуальных данных.",
                )
            if file_sha256(train_path) != plan.train_sha256:
                raise WebError(
                    "Train изменился после генерации LLM-плана.",
                    hint="Сгенерируйте план заново, чтобы формулы проверились на актуальных данных.",
                )
            command += [
                "--experiment",
                "llm-feature-proposals",
                "--approve-llm-features",
                "--llm-plan-file",
                str(plan.path),
                "--llm-log",
                str(plan.metadata_path),
            ]
        elif llm_approved:
            raise WebError("llm_approved передан без llm_plan_id.")
        payload["project"] = project
        payload["models"] = models
        payload["sample_pct"] = 100 if sample_pct is None else sample_pct
        payload["profile"] = profile
        payload["strategy"] = strategy
        payload["device"] = device
        payload["cpu_threads"] = cpu_threads
        payload["ram_gb"] = ram_gb
        payload["vram_gb"] = vram_gb
        payload["model_params"] = manual_params
        payload["until"] = until
        payload["llm_plan_id"] = llm_plan_id or None
        payload["llm_approved"] = bool(llm_plan_id and llm_approved)
        return project, command, payload

    @staticmethod
    def _variant_label(payload: dict[str, Any]) -> str:
        profile = payload.get("profile")
        strategy = f"profile {profile}" if profile not in (None, "", 0, "0") else payload.get("strategy", "standard")
        models = "+".join(payload.get("models") or []) or "recipe models"
        until = "full" if payload.get("until") == "report" else "CV"
        llm = " · LLM" if payload.get("llm_plan_id") else ""
        tuned = " · tuned" if payload.get("model_params") else ""
        resources: list[str] = []
        if payload.get("device"):
            resources.append(str(payload["device"]))
        if payload.get("cpu_threads") is not None:
            threads = payload["cpu_threads"]
            resources.append("auto" if threads == 0 else f"{threads}t")
        if payload.get("ram_gb") is not None:
            resources.append(f"{payload['ram_gb']}GB")
        if payload.get("vram_gb") is not None:
            resources.append(f"{payload['vram_gb']}GB VRAM")
        resource = f" · {' '.join(resources)}" if resources else ""
        return f"{strategy} · {models} · {until}{resource}{tuned}{llm}"

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        project, command, normalized = self.build_run_command(payload)
        return self.runs.start(project, command, normalized)

    def start_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate every variant, then queue all of them atomically."""
        raw = payload.get("variants")
        if not isinstance(raw, list) or not raw:
            raise WebError(
                "Передайте непустой массив variants.",
                hint="Каждый элемент — объект с target, models, strategy/profile и остальными полями запуска.",
            )
        base = {key: value for key, value in payload.items() if key != "variants"}
        prepared: list[tuple[str, str, list[str], dict[str, Any]]] = []
        for index, variant in enumerate(raw, start=1):
            if not isinstance(variant, dict):
                raise WebError(f"Вариант #{index} должен быть объектом.")
            merged = {**base, **variant}
            project, command, normalized = self.build_run_command(merged)
            label = str(variant.get("label") or self._variant_label(normalized)).strip()
            prepared.append((label or f"variant {index}", project, command, normalized))
        projects = {item[1] for item in prepared}
        return self.runs.enqueue(prepared, projects.pop() if len(projects) == 1 else None)

    def failure_payload(self, project: str, run_id: str | None = None) -> dict[str, Any]:
        run_dir = self._run_dir(project, run_id)
        if run_dir is None:
            raise WebError("Run не найден.", HTTPStatus.NOT_FOUND)
        failure = _json_load(run_dir / "failure.json")
        report_path = run_dir / "failure_report.md"
        report = None
        if report_path.is_file():
            try:
                report = report_path.read_text(encoding="utf-8")[:200_000]
            except OSError:
                report = None
        if not isinstance(failure, dict) and report is None:
            raise WebError("Для этого run нет postmortem.", HTTPStatus.NOT_FOUND)
        return {
            "project": _slug(project),
            "run_id": run_dir.name,
            "failure": failure if isinstance(failure, dict) else None,
            "report": report,
            "report_html": render_markdown(report) if report else None,
        }

    def runs_payload(self, project: str) -> dict[str, Any]:
        from kconv.runs_index import list_runs

        slug = _slug(project)
        project_dir = self.root / "competitions" / slug
        if not (project_dir / "runs").is_dir():
            raise WebError("Для проекта ещё нет runs.", HTTPStatus.NOT_FOUND)
        return {"project": slug, "runs": list_runs(project_dir)}

    def logs_payload(
        self,
        project: str,
        run_id: str | None = None,
        *,
        level: str | None = None,
        query: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        slug = _slug(project)
        lines: list[str] = []
        current = self.runs.snapshot()
        if current is not None and current.get("project") == slug:
            lines.extend(current.get("log") or [])
        run_dir = self._run_dir(slug, run_id)
        if run_dir is not None:
            web_log = run_dir / "web.log"
            if web_log.is_file():
                try:
                    lines.extend(web_log.read_text(encoding="utf-8", errors="replace").splitlines())
                except OSError:
                    pass
            else:
                failure = _json_load(run_dir / "failure.json")
                if isinstance(failure, dict):
                    lines.extend(str(item) for item in (failure.get("last_log_lines") or []))
        needle = (query or "").casefold()
        wanted = (level or "").upper()
        filtered: list[str] = []
        for line in lines:
            upper = line.upper()
            if wanted == "ERROR" and "ERROR" not in upper and "TRACEBACK" not in upper:
                continue
            if wanted == "WARNING" and "WARN" not in upper:
                continue
            if wanted == "INFO" and not any(token in upper for token in ("INFO", "STARTED", "DONE", "STAGE")):
                continue
            if needle and needle not in line.casefold():
                continue
            filtered.append(line)
        return {
            "project": slug,
            "run_id": run_dir.name if run_dir is not None else run_id,
            "lines": filtered[-max(1, min(int(limit), 2000)):],
            "total": len(filtered),
        }

    def continue_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = _slug(str(payload.get("project") or ""))
        run_id = str(payload.get("run_id") or "").strip()
        run_dir = self._run_dir(project, run_id)
        if run_dir is None:
            raise WebError("Run не найден.", HTTPStatus.NOT_FOUND)
        latest = self.latest_run_id(project)
        if latest and latest != run_dir.name:
            raise WebError(
                "Продолжить можно только последний run проекта.",
                hint="Сначала сделайте этот run последним через CLI или запустите новый.",
            )
        run_meta = _json_load(run_dir / "run.json") or {}
        until = str(payload.get("until") or run_meta.get("until") or "report")
        if until not in STAGE_ORDER:
            raise WebError(f"until должен быть одним из: {', '.join(STAGE_ORDER)}.")
        command = [sys.executable, "-m", "kconv.cli", "run"]
        config_path = str(run_meta.get("config_path") or "")
        if config_path and Path(config_path).is_file():
            command += ["--config", config_path]
        else:
            command += ["--comp", project]
        command += ["--resume", "--until", until]
        if payload.get("force"):
            command += ["--force"]
        label = f"continue {run_dir.name} · {until}"
        return self.runs.start(project, command, {"continue": True, "run_id": run_dir.name, "until": until, "label": label})

    def retry_queue_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(payload.get("item_id") or payload.get("id") or "").strip()
        if not item_id:
            raise WebError("Укажите item_id.")
        return self.runs.retry(item_id, str(payload.get("project") or "") or None)


class _Handler(BaseHTTPRequestHandler):
    server: KconvWebServer
    server_version = "kconv-web/0.1"

    @property
    def app(self) -> WebApp:
        return self.server.app

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        _LOGGER.debug("web: " + format, *args)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # A local UI is edited in place; never let the browser keep stale JS/CSS
        # after a server restart or an update to the static assets.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str, filename: str) -> None:
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_json({"message": f"Не удалось прочитать файл: {exc}"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: WebError) -> None:
        self._send_json({"message": exc.message, "hint": exc.hint}, exc.status)

    def _bound_port(self) -> int:
        return int(self.server.server_address[1])

    def _allowed_hostnames(self) -> set[str]:
        """Host names that may appear in Host/Origin for this server.

        Loopback is always accepted.  When the operator explicitly opts into
        remote access, the actual bound address and the local address of the
        current connection are accepted as well; arbitrary DNS names are not,
        which is what blocks DNS-rebinding attacks.
        """
        names = set(_LOOPBACK_HOSTS)
        if not getattr(self.server, "allow_remote", False):
            return names
        bound = str(self.server.server_address[0]).strip("[]").lower()
        if bound and bound not in _WILDCARD_HOSTS:
            names.add(bound)
        try:
            local = str(self.connection.getsockname()[0]).strip("[]").lower()
        except OSError:
            local = ""
        if local and local not in _WILDCARD_HOSTS:
            names.add(local)
        return names

    @staticmethod
    def _parse_host_header(value: str) -> tuple[str, int]:
        text = value.strip()
        if not text or "://" in text or "@" in text:
            raise ValueError("invalid host")
        parsed = urlsplit("//" + text)
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("invalid host")
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("invalid host")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid host") from exc
        return hostname.lower(), port if port is not None else 80

    def _check_host(self) -> None:
        raw = self.headers.get("Host")
        if raw is None:
            raise WebError("Не указан заголовок Host.", HTTPStatus.FORBIDDEN)
        try:
            hostname, port = self._parse_host_header(raw)
        except ValueError as exc:
            raise WebError("Некорректный заголовок Host.", HTTPStatus.FORBIDDEN) from exc
        if port != self._bound_port() or hostname not in self._allowed_hostnames():
            raise WebError(
                "Запрос с недопустимым Host отклонён.",
                HTTPStatus.FORBIDDEN,
                "Откройте kconv studio по адресу, на котором запущен сервер (loopback).",
            )

    def _check_origin(self) -> None:
        raw = self.headers.get("Origin")
        if raw is None:
            return
        try:
            parsed = urlsplit(raw)
            port = parsed.port if parsed.port is not None else 80
        except ValueError as exc:
            raise WebError("Некорректный заголовок Origin.", HTTPStatus.FORBIDDEN) from exc
        hostname = parsed.hostname
        if (
            parsed.scheme.lower() != "http"
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or not hostname
            or port != self._bound_port()
            or hostname.lower() not in self._allowed_hostnames()
        ):
            raise WebError(
                "Запрос с другого origin отклонён.",
                HTTPStatus.FORBIDDEN,
                "Откройте kconv studio напрямую по его локальному адресу.",
            )

    def _check_fetch_site(self) -> None:
        raw = self.headers.get("Sec-Fetch-Site")
        if raw is None:
            return
        value = raw.strip().lower()
        if value not in {"same-origin", "same-site", "none"}:
            raise WebError(
                "Cross-site запрос отклонён.",
                HTTPStatus.FORBIDDEN,
                "Откройте kconv studio напрямую по его локальному адресу.",
            )

    def _check_csrf(self) -> None:
        supplied = self.headers.get("X-Kconv-Token") or ""
        if not supplied or not hmac.compare_digest(supplied, self.app.csrf_token):
            raise WebError(
                "Неверный или отсутствующий CSRF-токен.",
                HTTPStatus.FORBIDDEN,
                "Обновите страницу kconv studio; если ошибка повторяется, перезапустите сервер.",
            )

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlsplit(self.path)
        path = parsed.path
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        try:
            self._check_host()
            if path == "/api/session":
                self._send_json(self.app.session_payload())
                return
            if path == "/":
                index = STATIC_DIR / "index.html"
                self._send_text(index.read_text(encoding="utf-8"), "text/html; charset=utf-8")
                return
            if path == "/static/app.css":
                self._send_text((STATIC_DIR / "app.css").read_text(encoding="utf-8"), "text/css; charset=utf-8")
                return
            if path == "/static/app.js":
                self._send_text((STATIC_DIR / "app.js").read_text(encoding="utf-8"), "application/javascript; charset=utf-8")
                return
            if path == "/api/config":
                self._send_json(self.app.config_payload())
                return
            if path == "/api/llm/local/discover":
                self._send_json(self.app.discover_local_llms(query.get("base_url")))
                return
            if path == "/api/dataset":
                self._send_json(self.app.dataset_payload(query))
                return
            if path == "/api/stats":
                self._send_json(self.app.stats_payload())
                return
            if path == "/api/projects":
                self._send_json(self.app.projects_payload())
                return
            if path == "/api/run/status":
                self._send_json(self.app.run_status(query.get("project")))
                return
            if path == "/api/comparison":
                project = query.get("project") or ""
                if not project:
                    raise WebError("Укажите project.", hint="Например: /api/comparison?project=my-project")
                self._send_json(self.app.comparison_payload(project, query.get("dataset")))
                return
            if path == "/api/kaggle/status":
                self._send_json(self.app.kaggle_status_payload(query.get("project")))
                return
            if path == "/api/submission":
                project = query.get("project") or ""
                submission = self.app.submission_path(project, query.get("run_id"))
                if submission is None:
                    raise WebError("Submission ещё не создан.", HTTPStatus.NOT_FOUND)
                self._send_file(submission, "text/csv; charset=utf-8", f"{_slug(project)}-submission.csv")
                return
            if path == "/api/report":
                project = query.get("project") or ""
                report = self.app.report_path(project, query.get("run_id"))
                if report is None:
                    raise WebError("Report ещё не создан.", HTTPStatus.NOT_FOUND)
                self._send_file(report, "text/markdown; charset=utf-8", f"{_slug(project)}-report.md")
                return
            if path == "/api/failure":
                project = query.get("project") or ""
                if not project:
                    raise WebError("Укажите project.")
                self._send_json(self.app.failure_payload(project, query.get("run_id")))
                return
            if path == "/api/runs":
                project = query.get("project") or ""
                if not project:
                    raise WebError("Укажите project.")
                self._send_json(self.app.runs_payload(project))
                return
            if path == "/api/run/logs":
                project = query.get("project") or ""
                if not project:
                    raise WebError("Укажите project.")
                try:
                    limit = int(query.get("limit") or "500")
                except ValueError:
                    limit = 500
                self._send_json(
                    self.app.logs_payload(
                        project,
                        query.get("run_id"),
                        level=query.get("level"),
                        query=query.get("q"),
                        limit=limit,
                    )
                )
                return
            raise WebError("Not found.", HTTPStatus.NOT_FOUND)
        except WebError as exc:
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            _LOGGER.exception("GET %s failed", path)
            self._send_json({"message": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlsplit(self.path)
        path = parsed.path
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        try:
            self._check_host()
            self._check_origin()
            self._check_fetch_site()
            self._check_csrf()
            if path == "/api/upload":
                self._handle_upload(query)
                return
            if path == "/api/llm/propose":
                self._send_json(self.app.propose_llm_features(self._read_json()))
                return
            if path == "/api/kaggle/credentials":
                self._send_json(self.app.save_kaggle_credentials(self._read_json()))
                return
            if path == "/api/kaggle/submit":
                self._send_json(self.app.submit_kaggle(self._read_json()))
                return
            if path == "/api/run":
                self._send_json(self.app.start_run(self._read_json()), HTTPStatus.ACCEPTED)
                return
            if path == "/api/run/batch":
                self._send_json(self.app.start_batch(self._read_json()), HTTPStatus.ACCEPTED)
                return
            if path == "/api/run/stop":
                result = self.app.runs.stop()
                self._send_json(result or {"status": "idle"})
                return
            if path == "/api/run/continue":
                self._send_json(self.app.continue_run(self._read_json()), HTTPStatus.ACCEPTED)
                return
            if path == "/api/queue/retry":
                self._send_json(self.app.retry_queue_item(self._read_json()), HTTPStatus.ACCEPTED)
                return
            raise WebError("Not found.", HTTPStatus.NOT_FOUND)
        except WebError as exc:
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            _LOGGER.exception("POST %s failed", path)
            self._send_json({"message": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise WebError(
                "Ожидается Content-Type: application/json.",
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Передайте JSON-тело с заголовком application/json; charset=utf-8.",
            )
        body = self._read_body(MAX_JSON_BYTES)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebError("Некорректный JSON.", hint=str(exc)) from exc
        if not isinstance(payload, dict):
            raise WebError("JSON должен быть объектом.")
        return payload

    def _read_body(self, limit: int) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise WebError("Не указан Content-Length.", HTTPStatus.LENGTH_REQUIRED)
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise WebError("Некорректный Content-Length.") from exc
        if length < 0 or length > limit:
            raise WebError("Слишком большой запрос.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        return self.rfile.read(length)

    def _handle_upload(self, query: dict[str, str]) -> None:
        kind = query.get("kind", "")
        prefix = _KIND_PREFIX.get(kind)
        if prefix is None:
            raise WebError("kind должен быть train, test или sample.")
        name = query.get("name") or f"{prefix}.csv"
        lowered = name.lower()
        suffix = ".csv.gz" if lowered.endswith(".csv.gz") else Path(lowered).suffix
        if suffix not in _ALLOWED_EXTENSIONS:
            raise WebError("Поддерживаются CSV, CSV.GZ и Parquet.", hint=f"Получено: {name}")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise WebError("Не указан Content-Length.", HTTPStatus.LENGTH_REQUIRED)
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise WebError("Некорректный Content-Length.") from exc
        if length <= 0:
            raise WebError("Файл пустой.")
        if length > MAX_UPLOAD_BYTES:
            raise WebError("Файл больше 256 MiB.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        destination = self.app.data_dir / f"{prefix}{suffix}"
        temporary = destination.with_name(destination.name + f".part-{os.getpid()}-{threading.get_ident()}")
        try:
            remaining = length
            with temporary.open("wb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise WebError("Загрузка оборвалась до завершения.")
                    handle.write(chunk)
                    remaining -= len(chunk)
            os.replace(temporary, destination)
            for old in self.app.data_dir.glob(f"{prefix}.*"):
                old_name = old.name.lower()
                old_suffix = ".csv.gz" if old_name.endswith(".csv.gz") else old.suffix.lower()
                if old.is_file() and old != destination and old_suffix in _ALLOWED_EXTENSIONS:
                    old.unlink()
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
        self._send_json({"ok": True, "file": _file_info(destination)})


class KconvWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        app: WebApp,
        *,
        allow_remote: bool = False,
    ) -> None:
        self.app = app
        self.allow_remote = bool(allow_remote)
        super().__init__(address, handler)


def create_server(
    *,
    root: str | Path | None = None,
    data_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    kaggle_token_path: str | Path | None = None,
    allow_remote: bool = False,
) -> KconvWebServer:
    """Create a local HTTP server; useful for tests and embedding.

    The web UI can control training runs and Kaggle credentials, so it binds
    loopback unless the caller explicitly opts into network access.
    """
    normalized_host = str(host).strip().lower()
    if normalized_host.startswith("[") and normalized_host.endswith("]"):
        normalized_host = normalized_host[1:-1]
    if normalized_host not in _LOOPBACK_HOSTS and not allow_remote:
        raise WebError(
            "Сервер web UI разрешено запускать только на loopback.",
            HTTPStatus.BAD_REQUEST,
            "Используйте --host 127.0.0.1 (или localhost/::1). "
            "Для доступа из сети добавьте --allow-remote и понимайте риск: "
            "любой, кто откроет порт, сможет запускать обучение и отправлять submission.",
        )
    root_path = Path(root or Path.cwd()).resolve()
    data_path = Path(data_dir or root_path / "data").resolve()
    app = WebApp(root_path, data_path, kaggle_token_path=kaggle_token_path)
    app.recover_stale_runs()
    return KconvWebServer((normalized_host, port), _Handler, app, allow_remote=allow_remote)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    data_dir: str | Path | None = None,
    open_browser: bool = True,
    allow_remote: bool = False,
) -> None:
    """Start the local site and block until Ctrl+C."""
    server = create_server(
        host=host,
        port=port,
        data_dir=data_dir,
        allow_remote=allow_remote,
    )
    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{actual_port}/"
    print(f"kconv studio: {url}")
    print("Данные: " + str(server.app.data_dir))
    print("Остановить: Ctrl+C")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nОстановка kconv studio.")
    finally:
        server.app.shutdown()
        server.server_close()


__all__ = ["KconvWebServer", "WebApp", "WebError", "create_server", "serve"]
