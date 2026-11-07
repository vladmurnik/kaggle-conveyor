"""Optional process isolation for stage and model timeouts.

The default path stays in-process: this keeps plugins, monkeypatching and the
existing test suite working.  When a timeout is configured, the work runs in a
spawned child so a hung native ``fit`` can be terminated without killing the
whole pipeline.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
import time
import traceback
from typing import Any

from kconv.errors import StageError


def _stage_worker(result_queue: Any, module_name: str, func_name: str, ctx: Any) -> None:
    try:
        module = importlib.import_module(module_name)
        runner = getattr(module, func_name)
        result_queue.put(("ok", runner(ctx)))
    except BaseException as exc:  # noqa: BLE001 - serialized across the process boundary
        result_queue.put(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "hint": getattr(exc, "hint", None),
                    "traceback": traceback.format_exc(),
                },
            )
        )


def run_stage_isolated(
    module_name: str,
    func_name: str,
    ctx: Any,
    timeout_sec: float,
) -> Any:
    """Run a stage runner in a child process and enforce a hard timeout."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_stage_worker,
        args=(result_queue, module_name, func_name, ctx),
        name=f"kconv-stage-{module_name.rsplit('.', 1)[-1]}",
    )
    process.start()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(2)
        raise TimeoutError(f"Stage exceeded {timeout_sec:.0f}s and was terminated.")
    try:
        status, payload = result_queue.get(timeout=1)
    except queue.Empty:
        raise RuntimeError(
            f"Stage process exited with code {process.exitcode} without a result."
        ) from None
    finally:
        result_queue.close()
    if status == "ok":
        return payload
    error = payload if isinstance(payload, dict) else {}
    message = f"{error.get('type', 'Error')}: {error.get('message', 'unknown error')}"
    if error.get("traceback"):
        message += f"\n{error['traceback']}"
    raise StageError(message, error.get("hint") or "Fix the stage error and rerun with --resume.")


def _fit_worker(result_queue: Any, adapter: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    try:
        adapter.fit(*args, **kwargs)
        result_queue.put(("ok", adapter))
    except BaseException as exc:  # noqa: BLE001 - serialized across the process boundary
        result_queue.put(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "hint": getattr(exc, "hint", None),
                },
            )
        )


def fit_with_timeout(
    adapter: Any,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    *,
    params: dict[str, Any],
    seed: int,
    early_stopping_rounds: int,
    timeout_sec: float,
) -> Any:
    """Fit ``adapter`` in-process or in a killable child process."""
    if timeout_sec <= 0:
        adapter.fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            params=params,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        return adapter

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_fit_worker,
        args=(
            result_queue,
            adapter,
            (X_train, y_train, X_valid, y_valid),
            {
                "params": params,
                "seed": seed,
                "early_stopping_rounds": early_stopping_rounds,
            },
        ),
        name=f"kconv-fit-{getattr(adapter, 'name', 'model')}-{seed}",
    )
    process.start()
    deadline = time.monotonic() + timeout_sec
    status: str
    payload: Any
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(2)
                raise TimeoutError(
                    f"Model '{getattr(adapter, 'name', 'model')}' fit exceeded {timeout_sec:.0f}s."
                )
            try:
                status, payload = result_queue.get(timeout=min(0.2, remaining))
                break
            except (queue.Empty, EOFError):
                if not process.is_alive():
                    try:
                        status, payload = result_queue.get(timeout=1)
                        break
                    except (queue.Empty, EOFError):
                        raise RuntimeError(
                            f"Model fit process exited with code {process.exitcode} without a result."
                        ) from None
    finally:
        result_queue.close()
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():
        process.kill()
        process.join(2)
    if status == "ok":
        return payload
    error = payload if isinstance(payload, dict) else {}
    raise RuntimeError(
        f"{error.get('type', 'Error')}: {error.get('message', 'model fit failed')}"
    )


def reduced_params(params: dict[str, Any]) -> dict[str, Any]:
    """Conservative retry parameters for a model that failed or timed out."""
    reduced = dict(params)
    for key in ("n_estimators", "iterations", "max_iter", "num_iterations"):
        value = reduced.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            reduced[key] = max(1, value // 2)
    for key in ("max_depth", "num_leaves", "max_leaf_nodes"):
        value = reduced.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            reduced[key] = max(1, value // 2)
    return reduced
