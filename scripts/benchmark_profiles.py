"""Measured CPU/GPU runtime-profile benchmark on a real UCI dataset.

Run (with optional RSS sampling)::

    uv run --with psutil python scripts/benchmark_profiles.py --reps 2

The dataset is ``sklearn.datasets.load_breast_cancer`` (UCI Breast Cancer
Wisconsin, 569 rows, 30 numeric features), split once into stratified
train/test files.  Every profile is invoked through ``python -m kconv run`` in a
fresh temporary working directory, so wall time includes the CLI/import
startup, not just the stages.

Profiles 1-3 use a fixed ``lgbm`` model set; profiles 4-5 use ``xgb,catboost``
so the GPU path is exercised.  Profile 5 keeps its real ``deep`` strategy and
HPO settings.  These overrides are recorded in the report; the benchmark does
not claim the full profile-5 25-model budget was run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports"
DEFAULT_CPU_MODELS = "lgbm"
DEFAULT_GPU_MODELS = "xgb,catboost"


def _rss_mib(pid: int) -> float | None:
    """Best-effort process-tree RSS; psutil is optional and never required."""
    try:
        import psutil  # type: ignore[import-not-found]

        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
        total = 0
        for process in processes:
            try:
                total += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return float(total / (1024 * 1024)) if total else None
    except Exception:
        return None


def _sample_rss(pid: int, stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        value = _rss_mib(pid)
        if value is not None:
            samples.append(value)
        stop.wait(0.1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dataset(root: Path, *, seed: int = 42) -> tuple[Path, dict[str, Any]]:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    frame.insert(0, "id", np.arange(len(frame)))
    frame["target"] = bunch.target.astype(int)
    train, test = train_test_split(
        frame,
        test_size=0.2,
        random_state=seed,
        stratify=frame["target"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(data / "train.csv", index=False)
    test.drop(columns=["target"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test["id"].to_numpy(), "target": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    metadata = {
        "name": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "source": "sklearn.datasets.load_breast_cancer",
        "seed": seed,
        "rows_total": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": int(frame.shape[1] - 2),
        "target_mean": float(frame["target"].mean()),
        "train_sha256": _sha(data / "train.csv"),
        "test_sha256": _sha(data / "test.csv"),
    }
    return data, metadata


def _probe_gpu() -> dict[str, Any]:
    """Actually train one tiny XGBoost and CatBoost model on CUDA."""
    result: dict[str, Any] = {"xgb_cuda_fit": False, "catboost_gpu_fit": False}
    from sklearn.datasets import load_breast_cancer as load

    x, y = load(return_X_y=True)
    x = x[:100]
    y = y[:100]
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=3,
            max_depth=2,
            device="cuda",
            verbosity=0,
            n_jobs=1,
        )
        model.fit(x, y)
        result["xgb_cuda_fit"] = True
    except Exception as exc:  # noqa: BLE001 - probe must report, not fail the script
        result["xgb_cuda_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=3,
            depth=2,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x, y)
        result["catboost_gpu_fit"] = True
    except Exception as exc:  # noqa: BLE001
        result["catboost_gpu_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return result


def _read_run(project_dir: Path) -> dict[str, Any]:
    from kconv.state import read_latest_pointer

    run_id = read_latest_pointer(project_dir)
    if not run_id:
        return {"run_id": None}
    run_dir = project_dir / "runs" / run_id
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    cv_path = run_dir / "01_validate" / "cv_summary.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8")) if cv_path.is_file() else {}
    submission = run_dir / "04_submit" / "submission.csv"
    rows = None
    if submission.is_file():
        rows = int(len(pd.read_csv(submission)))
    return {
        "run_id": run_id,
        "state": state,
        "oof": cv.get("models") or cv.get("oof") or cv,
        "submission_rows": rows,
        "run_dir": str(run_dir),
    }


def run_profile(
    data: Path,
    level: int,
    *,
    models: str,
    cwd: Path,
    timeout: float,
    rep: int,
) -> dict[str, Any]:
    name = f"bench-prof-{level}-r{rep}"
    command = [
        sys.executable,
        "-m",
        "kconv",
        "run",
        "--data",
        str(data),
        "--target",
        "target",
        "--profile",
        str(level),
        "--models",
        models,
        "--name",
        name,
        "--until",
        "report",
        "--force",
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stop = threading.Event()
    samples: list[float] = []
    sampler = threading.Thread(target=_sample_rss, args=(process.pid, stop, samples), daemon=True)
    sampler.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        stop.set()
        return {
            "profile": level,
            "rep": rep,
            "models": models,
            "command": command,
            "returncode": None,
            "error": f"timeout after {timeout:g}s",
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "wall_sec": round(time.perf_counter() - started, 3),
        }
    finally:
        stop.set()
    wall = time.perf_counter() - started
    project_dir = cwd / "competitions" / name
    run_info = _read_run(project_dir)
    state = run_info.get("state") or {}
    stages = state.get("stages") or {}
    stage_seconds = {
        key: entry.get("duration_sec")
        for key, entry in stages.items()
        if isinstance(entry, dict) and entry.get("duration_sec") is not None
    }
    return {
        "profile": level,
        "rep": rep,
        "models": models,
        "command": command,
        "returncode": process.returncode,
        "wall_sec": round(wall, 3),
        "stage_seconds": stage_seconds,
        "stage_sum_sec": round(sum(stage_seconds.values()), 3) if stage_seconds else None,
        "ram_peak_mib": round(max(samples), 1) if samples else None,
        "ram_samples": len(samples),
        "run": run_info,
        "stdout_tail": stdout[-3000:],
        "stderr_tail": stderr[-3000:],
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_profile.setdefault(str(run["profile"]), []).append(run)
    summary: dict[str, Any] = {}
    for level, items in sorted(by_profile.items()):
        good = [item for item in items if item.get("returncode") == 0]
        walls = sorted(item["wall_sec"] for item in good)
        rams = sorted(item["ram_peak_mib"] for item in good if item.get("ram_peak_mib") is not None)
        stages = sorted(
            item["stage_sum_sec"] for item in good if item.get("stage_sum_sec") is not None
        )
        summary[level] = {
            "runs": len(items),
            "successes": len(good),
            "wall_sec_median": walls[len(walls) // 2] if walls else None,
            "wall_sec_min": walls[0] if walls else None,
            "wall_sec_max": walls[-1] if walls else None,
            "ram_peak_mib_median": rams[len(rams) // 2] if rams else None,
            "stage_sum_sec_median": stages[len(stages) // 2] if stages else None,
            "models": sorted({item["models"] for item in items}),
        }
    return summary


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# kconv runtime-profile follow-up benchmark",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Dataset: {payload['dataset']['name']} ({payload['dataset']['rows_total']} rows, "
        f"{payload['dataset']['features']} features; train/test {payload['dataset']['train_rows']}/"
        f"{payload['dataset']['test_rows']})",
        "- Invocation: `python -m kconv run ... --until report --force` in a fresh temp cwd",
        f"- Repetitions: {payload['protocol']['reps']} per profile "
        f"(per-profile override: {payload['protocol'].get('profile_reps', 'none')})",
        f"- Timeout: {payload['protocol']['timeout_sec']} s",
        f"- Per-profile timeout: {payload['protocol'].get('profile_timeouts', 'same as above')}",
        f"- CPU profiles 1-3 model override: `{payload['protocol']['cpu_models']}`",
        f"- GPU profiles 4-5 model override: `{payload['protocol']['gpu_models']}`",
        "",
        "## GPU probe (actual tiny fits, not a build flag)",
        "",
        "```json",
        json.dumps(payload["gpu_probe"], indent=2),
        "```",
        "",
        "## Results",
        "",
        "| Profile | Models | Success | Wall s median (min-max) | Stage sum s | RAM peak MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for level, item in payload["summary"].items():
        wall = (
            f"{item['wall_sec_median']:.3f} ({item['wall_sec_min']:.3f}-{item['wall_sec_max']:.3f})"
            if item["wall_sec_median"] is not None
            else "-"
        )
        ram = f"{item['ram_peak_mib_median']:.1f}" if item["ram_peak_mib_median"] is not None else "-"
        stage_sum = (
            f"{item['stage_sum_sec_median']:.3f}"
            if item.get("stage_sum_sec_median") is not None
            else "-"
        )
        lines.append(
            f"| {level} | {', '.join(item['models'])} | {item['successes']}/{item['runs']} | "
            f"{wall} | {stage_sum} | {ram} |"
        )
    lines += [
        "",
        "## Per-run evidence",
        "",
    ]
    for run in payload["runs"]:
        lines.append(
            f"- profile {run['profile']} rep {run['rep']}: exit={run.get('returncode')}, "
            f"wall={run.get('wall_sec')} s, stages={run.get('stage_seconds')}, "
            f"run={run.get('run', {}).get('run_id')}, "
            f"submission_rows={run.get('run', {}).get('submission_rows')}, "
            f"stderr={len(run.get('stderr_tail') or '')} chars"
        )
        if run.get("error"):
            lines.append(f"  - error: {run['error']}")
    lines += [
        "",
        "## Honest limits",
        "",
        "- Profiles 4-5 were run only when the GPU probe reported successful tiny CUDA fits; detection alone is not evidence.",
        "- Profiles 1-3 use `lgbm` only and profiles 4-5 use `xgb,catboost`, so this measures the runtime presets and GPU path, not the full default model budget.",
        "- Profile 5 keeps HPO/deep/3-seed settings but uses the two-model override; it is not the 25-model maximum run.",
        "- RAM is present only when the script is run with `psutil` (`uv run --with psutil ...`); otherwise it is `-`.",
        "",
    ]
    return "\n".join(lines)


def benchmark(
    output_dir: Path,
    *,
    reps: int,
    profiles: list[int],
    timeout: float,
    cpu_models: str,
    gpu_models: str,
    profile_reps: dict[int, int] | None = None,
    profile_timeouts: dict[int, float] | None = None,
) -> dict[str, Any]:
    random.seed(42)
    np.random.seed(42)
    root = Path(tempfile.mkdtemp(prefix="kconv-profile-bench-"))
    data, dataset = make_dataset(root)
    gpu_probe = _probe_gpu()
    reps_by_profile = {level: (profile_reps or {}).get(level, reps) for level in profiles}
    timeouts_by_profile = {
        level: (profile_timeouts or {}).get(level, timeout) for level in profiles
    }
    runs: list[dict[str, Any]] = []
    for level in profiles:
        if level >= 4 and not (gpu_probe["xgb_cuda_fit"] or gpu_probe["catboost_gpu_fit"]):
            runs.append(
                {
                    "profile": level,
                    "rep": 0,
                    "models": gpu_models,
                    "returncode": None,
                    "error": "skipped: no successful tiny CUDA fit",
                    "wall_sec": None,
                    "stage_seconds": {},
                }
            )
            continue
        models = gpu_models if level >= 4 else cpu_models
        for rep in range(1, reps_by_profile[level] + 1):
            runs.append(
                run_profile(
                    data,
                    level,
                    models=models,
                    cwd=root,
                    timeout=timeouts_by_profile[level],
                    rep=rep,
                )
            )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "protocol": {
            "reps": reps,
            "profiles": profiles,
            "timeout_sec": timeout,
            "profile_reps": {str(level): count for level, count in reps_by_profile.items()},
            "profile_timeouts": {
                str(level): duration for level, duration in timeouts_by_profile.items()
            },
            "cpu_models": cpu_models,
            "gpu_models": gpu_models,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "temp_root": str(root),
        },
        "gpu_probe": gpu_probe,
        "summary": _summary(runs),
        "runs": runs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "runtime_benchmark_followup.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = output_dir / "runtime_benchmark_followup.md"
    md_path.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return payload


def _parse_int_overrides(raw: str, *, name: str) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            key, value = item.split("=", 1)
            level, count = int(key), int(value)
        except ValueError as exc:
            raise ValueError(f"{name} entries must look like 1=2,2=1") from exc
        if level < 1 or count < 1:
            raise ValueError(f"{name} levels and values must be positive")
        overrides[level] = count
    return overrides


def _parse_float_overrides(raw: str, *, name: str) -> dict[int, float]:
    overrides: dict[int, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            key, value = item.split("=", 1)
            level, duration = int(key), float(value)
        except ValueError as exc:
            raise ValueError(f"{name} entries must look like 1=600,5=900") from exc
        if level < 1 or duration <= 0:
            raise ValueError(f"{name} levels and values must be positive")
        overrides[level] = duration
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--profiles", default="1,2,3,4,5")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--profile-reps",
        default="",
        help="Per-profile repetition overrides, e.g. '1=2,2=2,5=1'.",
    )
    parser.add_argument(
        "--profile-timeouts",
        default="",
        help="Per-profile timeout overrides, e.g. '1=600,5=900'.",
    )
    parser.add_argument("--cpu-models", default=DEFAULT_CPU_MODELS)
    parser.add_argument("--gpu-models", default=DEFAULT_GPU_MODELS)
    args = parser.parse_args()
    levels = [int(part) for part in args.profiles.split(",") if part.strip()]
    if args.reps < 1 or not levels:
        parser.error("--reps and --profiles must be non-empty/positive")
    try:
        profile_reps = _parse_int_overrides(args.profile_reps, name="--profile-reps")
        profile_timeouts = _parse_float_overrides(
            args.profile_timeouts, name="--profile-timeouts"
        )
    except ValueError as exc:
        parser.error(str(exc))
    unknown = (set(profile_reps) | set(profile_timeouts)) - set(levels)
    if unknown:
        parser.error(f"per-profile overrides reference unselected profiles: {sorted(unknown)}")
    benchmark(
        args.output_dir,
        reps=args.reps,
        profiles=levels,
        timeout=args.timeout,
        cpu_models=args.cpu_models,
        gpu_models=args.gpu_models,
        profile_reps=profile_reps,
        profile_timeouts=profile_timeouts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
