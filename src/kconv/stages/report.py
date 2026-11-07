"""Report stage: human-readable Markdown summary of a completed run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kconv.atomic import write_text_atomic
from kconv.context import RunContext, read_json

STAGE = "05_report"


def _read_json(path: Path, default=None):
    return read_json(path, default=default)


def run_report(ctx: RunContext) -> dict:
    schema = _read_json(ctx.run_dir / "00_ingest" / "schema.json", default={})
    card = _read_json(ctx.run_dir / "00_ingest" / "data_card.json", default={})
    cv = _read_json(ctx.run_dir / "01_validate" / "cv_summary.json", default={})
    adversarial = _read_json(ctx.run_dir / "01_validate" / "adversarial.json", default={})
    hpo = _read_json(ctx.run_dir / "01_hpo" / "tuning.json", default={})
    blend = _read_json(ctx.run_dir / "03_blend" / "blend.json", default={})
    run_meta = _read_json(ctx.run_dir / "run.json", default={})
    environment = run_meta.get("environment") if isinstance(run_meta, dict) else {}
    if not isinstance(environment, dict) or not environment:
        environment = {
            "kconv_version": run_meta.get("kconv_version"),
            "python": run_meta.get("python"),
            "platform": run_meta.get("platform"),
            "packages": run_meta.get("packages"),
        }
    packages = environment.get("packages") if isinstance(environment.get("packages"), dict) else {}
    reported_status = run_meta.get("status", "n/a") if isinstance(run_meta, dict) else "n/a"
    if reported_status == "running":
        reported_status = "completed_with_warnings" if ctx.failed_models else "completed"
    lines: list[str] = [
        f"# kconv report — {ctx.config.project}",
        "",
        f"- Recipe: `{ctx.config.recipe}`",
        f"- Strategy: `{ctx.config.strategy}`",
        f"- Task: `{schema.get('task', ctx.config.task)}`",
        f"- Metric: `{schema.get('metric', ctx.config.metric)}`",
        f"- Target: `{schema.get('target', ctx.config.data.target)}`",
        f"- Id: `{schema.get('id_col')}`",
        f"- Run: `{ctx.run_id}`",
        f"- Status: `{reported_status}`",
        "",
        "## Data",
        "",
        f"- Train rows: {card.get('rows', {}).get('train', 'n/a')}",
        f"- Test rows: {card.get('rows', {}).get('test', 'n/a')}",
        f"- Missing cells (train): {card.get('missing_cells_train', 'n/a')}",
        f"- Duplicate rows (train): {card.get('duplicate_rows_train', 'n/a')}",
        "",
        "## Environment",
        "",
        f"- kconv: `{environment.get('kconv_version', 'n/a')}`",
        f"- Python: `{environment.get('python', 'n/a')}`",
        f"- Platform: `{environment.get('platform', 'n/a')}`",
    ]
    if packages:
        lines.append(
            "- Packages: "
            + ", ".join(f"`{name} {version or 'missing'}`" for name, version in packages.items())
        )
    lines += [
        "",
        "## Cross-validation",
        "",
        f"- Strategy: `{cv.get('strategy', 'n/a')}`",
        f"- Folds: {cv.get('n_splits', 'n/a')}",
        f"- Validation rows: {cv.get('validation_rows', 'n/a')}",
    ]
    if adversarial.get("enabled"):
        lines += [
            "",
            "## Adversarial validation",
            "",
            f"- Train/test AUC: {adversarial.get('auc', 'n/a')}",
            f"- Folds: {adversarial.get('folds', 'n/a')}",
            f"- Features: {adversarial.get('features', 'n/a')}",
            f"- Warning: {adversarial.get('warning', False)}",
        ]
        if adversarial.get("status") == "skipped":
            lines.append(f"- Reason: {adversarial.get('message', 'n/a')}")
        elif adversarial.get("warning"):
            lines.append(
                f"- {adversarial.get('message', 'Train/test shift detected; CV may be optimistic.')}"
            )
    lines += [
        "",
        "## Hyperparameter search",
        "",
    ]
    if hpo.get("enabled"):
        lines += [
            f"- Trials per fold/model: {hpo.get('trials', 'n/a')}",
            f"- Inner validation size: {hpo.get('inner_size', 'n/a')}",
            f"- Time budget: {hpo.get('timeout_sec', 'n/a')}s",
            f"- Budget exhausted: {hpo.get('timeout_hit', False)}",
            f"- Tuned fold/model pairs: {hpo.get('tuned_fold_models', 0)}",
            f"- Models tuned: {', '.join(hpo.get('models_tuned', []) or []) or '_none_'}",
        ]
        skipped = hpo.get("models_skipped") or []
        if skipped:
            lines.append("- Models skipped:")
            for item in skipped:
                if isinstance(item, dict):
                    reasons = ", ".join(str(value) for value in item.get("reasons", []) or [])
                    lines.append(f"  - `{item.get('model', '?')}`: {reasons or 'unknown'}")
        lines += [
            "",
            "_Tuning is nested: each outer fold has its own inner validation split._",
        ]
    else:
        lines.append("_Disabled; recipe parameters are used as-is._")
    lines += [
        "",
        "## Models",
        "",
        "| Model | OOF score | Fold mean | Seeds |",
        "| --- | ---: | ---: | ---: |",
    ]
    model_rows: list[tuple[str, float]] = []
    for metrics_path in sorted((ctx.run_dir / "02_train").glob("*/metrics.json")):
        data = _read_json(metrics_path, default={})
        name = str(data.get("model", metrics_path.parent.name))
        score = data.get("oof_score")
        mean = data.get("mean_fold_score")
        seeds = data.get("seeds", "?")
        if isinstance(score, (int, float)):
            model_rows.append((name, float(score)))
        lines.append(
            f"| {name} | {score if score is not None else 'n/a'} | "
            f"{mean if mean is not None else 'n/a'} | {seeds} |"
        )
    if not model_rows:
        lines.append("| _none_ | n/a | n/a | n/a |")
    failed_models = (
        run_meta.get("failed_models") if isinstance(run_meta, dict) else []
    ) or ctx.failed_models
    if isinstance(failed_models, list) and failed_models:
        lines += [
            "",
            "## Skipped models",
            "",
            "| Model | Error |",
            "| --- | --- |",
        ]
        for item in failed_models:
            if not isinstance(item, dict):
                continue
            error = str(item.get("error", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {item.get('model', '?')} | {error} |")

    lines += [
        "",
        "## Ensemble",
        "",
        f"- Method: `{blend.get('method', 'n/a')}`",
        f"- OOF score: {blend.get('oof_score', 'n/a')}",
        f"- Fallback used: {blend.get('fallback_used', False)}",
    ]
    stack = blend.get("stack") or {}
    model_scores = blend.get("model_scores", {}) or {}
    if blend.get("method") == "stack" and stack:
        lines += [
            f"- Meta model: `{stack.get('meta_model', 'n/a')}`",
            f"- Meta CV folds: {stack.get('meta_splits', 'n/a')}",
            f"- Meta features: {stack.get('meta_features', 'n/a')}",
            "",
            "| Model | OOF |",
            "| --- | ---: |",
        ]
        for name in blend.get("models", []) or []:
            lines.append(f"| {name} | {model_scores.get(name, 'n/a')} |")
    else:
        lines += [
            "",
            "| Model | Weight | OOF |",
            "| --- | ---: | ---: |",
        ]
        weights = blend.get("weights", {}) or {}
        for name in blend.get("models", []) or []:
            lines.append(f"| {name} | {weights.get(name, 0.0):.4f} | {model_scores.get(name, 'n/a')} |")
    if not blend.get("models"):
        lines.append("| _none_ | n/a | n/a |")

    lines += ["", "## Top features", ""]
    any_importance = False
    for importance_path in sorted((ctx.run_dir / "02_train").glob("*/importance.csv")):
        frame = pd.read_csv(importance_path)
        if frame.empty:
            continue
        any_importance = True
        lines.append(f"### {importance_path.parent.name}")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("| --- | ---: |")
        for _, row in frame.head(10).iterrows():
            feature = row.iloc[0] if len(row) > 1 else "feature"
            value = row.iloc[-1]
            lines.append(f"| {feature} | {value} |")
        lines.append("")
    if not any_importance:
        lines.append("_No feature importance artifact for these model families._")
        lines.append("")
    failure_report = ctx.run_dir / "failure_report.md"
    if failure_report.exists():
        lines += [
            "## Failure postmortem",
            "",
            "- See `failure_report.md` and `failure.json` in the run directory.",
            "",
        ]

    warnings = list(dict.fromkeys(list(ctx.warnings) + list(card.get("warnings", []) or [])))
    lines += ["## Warnings", ""]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("_None._")
    lines.append("")

    out_path = ctx.artifact(STAGE, "report.md")
    write_text_atomic(out_path, "\n".join(lines))
    return {
        "status": "ok",
        "artifacts": [str(out_path.relative_to(ctx.run_dir)).replace("\\", "/")],
        "report": str(out_path),
    }
