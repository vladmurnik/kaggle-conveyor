"""kconv command line interface."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from kconv.api import list_models
from kconv.autodetect import detect_dataset
from kconv.benchmark import load_benchmark_registry, run_benchmark
from kconv.config import Config, load_config, save_config
from kconv.context import RunContext, file_sha256, read_json, write_json
from kconv.contracts import DatasetSpec
from kconv.errors import KconvError
from kconv.experiments import get_experiment, list_experiments
from kconv.features.fold_plan import (
    expression_columns,
    parse_fold_expression,
    plan_requires_fold_transformer,
)
from kconv.features.llm_expr import materialize_plan
from kconv.inference import export_inference
from kconv.io import read_table, write_table
from kconv.kaggle.auth import check_auth
from kconv.kaggle.submit import submit_submission
from kconv.llm_features import FeaturePlan, build_prompt, parse_plan
from kconv.llm_iterative import generate_features_iteratively
from kconv.llm_provider import (
    OpenAICompatibleProvider,
    build_feature_log,
    resolve_api_key,
    resolve_local_target,
    write_feature_log,
)
from kconv.package import create_package
from kconv.plugins import list_plugins, load_plugins
from kconv.project import apply_fold_plan, build_config
from kconv.recipe_audit import audit_recipes, signature_digest
from kconv.recipes import list_recipes, load_recipe
from kconv.registry import metric_names, model_names
from kconv.runtime import list_profiles
from kconv.sources import catalog_stats, get_source, list_sources
from kconv.state import PipelineState, read_latest_pointer

app = typer.Typer(
    name="kconv",
    help="Dataset-first Kaggle conveyor: drop data, choose a recipe and models, get a submission.",
    no_args_is_help=True,
    add_completion=False,
)
recipes_app = typer.Typer(help="List or inspect professional recipes.")
config_app = typer.Typer(help="Inspect or validate configuration.")
kaggle_app = typer.Typer(help="Kaggle credentials and submission helpers.")
sources_app = typer.Typer(help="Browse the bundled kaggle-toolkit source catalog.")
experiments_app = typer.Typer(help="Browse curated experiments and unusual solution patterns.")
benchmark_app = typer.Typer(help="Run fixed-dataset benchmarks and write scorecards.")
app.add_typer(recipes_app, name="recipes")
app.add_typer(config_app, name="config")
app.add_typer(kaggle_app, name="kaggle")
app.add_typer(sources_app, name="sources")
app.add_typer(experiments_app, name="experiments")
app.add_typer(benchmark_app, name="benchmark")
console = Console()


def _abort(exc: KconvError) -> None:
    console.print(f"[bold red]Error:[/bold red] {exc.message}")
    if exc.hint:
        console.print(f"[yellow]Hint:[/yellow] {exc.hint}")
    raise typer.Exit(exc.exit_code)


def _emit_json(payload: object) -> None:
    """Write machine-readable JSON without depending on the console encoding.

    Rich's legacy Windows renderer cannot print non-ASCII source descriptions
    on a cp1251/cp866 console, even when the output is JSON.  Escaping
    non-ASCII characters keeps ``--json`` usable in pipelines on every locale.
    """
    typer.echo(json.dumps(payload, ensure_ascii=True))


def _project_dir(comp: str | None) -> Path:
    if comp:
        return (Path.cwd() / "competitions" / comp).resolve()
    # A config path may live directly in a project folder.
    raise typer.BadParameter("Pass --comp or --config.")


def _project_or_cwd(comp: str | None, config: Path | None = None) -> Path:
    """Resolve a project for read-only UX commands, defaulting to cwd."""
    if config is not None:
        return config.resolve().parent
    if comp:
        return _project_dir(comp)
    cwd = Path.cwd().resolve()
    if (cwd / "runs").is_dir() or (cwd / "config.yaml").exists():
        return cwd
    raise typer.BadParameter("Pass --comp, --config, or run from a kconv project directory.")


def _config_for(comp: str | None, config: Path | None) -> tuple[Path, Path, Config]:
    if config is not None:
        path = config.resolve()
        return path.parent, path, load_config(path)
    project_dir = _project_dir(comp)
    path = project_dir / "config.yaml"
    return project_dir, path, load_config(path)


def _latest_state(project_dir: Path) -> tuple[str | None, PipelineState | None, Path | None]:
    run_id = read_latest_pointer(project_dir)
    if not run_id:
        return None, None, None
    run_dir = project_dir / "runs" / run_id
    state_path = run_dir / "state.json"
    state = PipelineState.load(state_path) if state_path.exists() else None
    return run_id, state, run_dir


def _execute(
    project_dir: Path,
    config_path: Path,
    config: Config,
    until: str,
    resume: bool,
    force: bool,
    llm_metadata: dict | None = None,
) -> RunContext:
    from kconv.stages.run import run_pipeline

    try:
        ctx = run_pipeline(
            project_dir,
            config_path,
            config,
            until=until,
            resume=resume,
            force=force,
        )
    except KconvError as exc:
        try:
            from kconv.context import read_json
            from kconv.state import read_latest_pointer

            run_id = read_latest_pointer(project_dir)
            run_meta = read_json(project_dir / "runs" / str(run_id) / "run.json", default={}) if run_id else {}
            if isinstance(run_meta, dict) and run_meta.get("failure_report"):
                console.print(f"[red]Failure report:[/red] {project_dir / 'runs' / str(run_id) / str(run_meta['failure_report'])}")
                console.print("[dim]Inspect it, fix the issue, then resume with --resume.[/dim]")
        except Exception:
            pass
        _abort(exc)
    state = PipelineState.load(ctx.run_dir / "state.json")
    table = Table(title=f"Run {ctx.run_id}")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Seconds", justify="right")
    for name, entry in state.stages.items():
        table.add_row(name, entry.status, f"{entry.duration_sec:.2f}" if entry.duration_sec is not None else "")
    console.print(table)
    console.print(f"Artifacts: [cyan]{ctx.run_dir}[/cyan]")
    report = ctx.run_dir / "05_report" / "report.md"
    if report.exists():
        console.print(f"Report: [cyan]{report}[/cyan]")
    submission = ctx.run_dir / "04_submit" / "submission.csv"
    if submission.exists():
        console.print(f"Submission: [green]{submission}[/green]")
    if llm_metadata is not None:
        metadata = dict(llm_metadata)
        metadata["approved"] = True
        metadata["approval_required"] = True
        log_path = write_feature_log(ctx.run_dir, metadata)
        if report.exists():
            marker = "## LLM feature proposals"
            text = report.read_text(encoding="utf-8")
            if marker not in text:
                usage = metadata.get("usage") or {}
                tokens = usage.get("total_tokens")
                token_text = str(tokens) if tokens is not None else "not reported"
                section = (
                    "\n## LLM feature proposals\n\n"
                    f"- Provider: `{metadata.get('provider', 'unknown')}`\n"
                    f"- Model: `{metadata.get('model') or 'unknown'}`\n"
                    f"- Approved: `{metadata.get('approved')}`\n"
                    f"- Total tokens: {token_text}\n"
                    f"- Materialized proposals: {len(metadata.get('proposals') or [])}\n"
                    "- Privacy: prompt text is not logged; see `05_report/llm_features.json`.\n"
                )
                report.write_text(text.rstrip() + section, encoding="utf-8", newline="\n")
        console.print(f"LLM feature log: [cyan]{log_path}[/cyan]")
    return ctx


_LLM_EXPERIMENT = "llm-feature-proposals"
_SAFE_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _llm_allowed_columns(frame: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    """Feature columns the LLM may reference: no target, no id."""
    blocked = {spec.target}
    if spec.id_col:
        blocked.add(spec.id_col)
    return [str(column) for column in frame.columns if column not in blocked]


def _llm_reserved_columns(spec: DatasetSpec) -> list[str]:
    """Column names that must never be used for generated feature names."""
    reserved = [spec.target]
    if spec.id_col:
        reserved.append(spec.id_col)
    return reserved


def _llm_sample_rows(frame: pd.DataFrame, allowed: list[str], *, limit: int = 5) -> list[dict]:
    """Return at most ``limit`` rows containing only allowed feature columns."""
    allowed_set = set(allowed)
    rows: list[dict] = []
    for row in frame.head(200).to_dict(orient="records")[:limit]:
        rows.append({str(key): value for key, value in row.items() if str(key) in allowed_set})
    return rows


def _mock_llm(sample: pd.DataFrame, allowed: list[str]) -> Callable[[str], str]:
    """Deterministic offline provider for smoke tests and demos.

    It only proposes simple sums/ratios over numeric, identifier-safe columns;
    it exists so the whole propose -> validate -> materialize path is testable
    without a network call or credentials.
    """
    numeric = [
        column
        for column in allowed
        if column in sample.columns
        and _SAFE_COLUMN_RE.match(column)
        and pd.api.types.is_numeric_dtype(sample[column])
    ]
    features: list[dict[str, str]] = []
    index = 0
    for position in range(0, len(numeric) - 1, 2):
        index += 1
        left, right = numeric[position], numeric[position + 1]
        features.append(
            {
                "name": f"llm_sum_{index}",
                "kind": "numeric",
                "expression": f"{left} + {right}",
                "rationale": f"Mock additive interaction between {left} and {right}.",
            }
        )
        features.append(
            {
                "name": f"llm_ratio_{index}",
                "kind": "numeric",
                "expression": f"{left} / {right}",
                "rationale": f"Mock ratio between {left} and {right}.",
            }
        )
        if len(features) >= 20:
            break
    if not features and numeric:
        only = numeric[0]
        features.append(
            {
                "name": "llm_square_1",
                "kind": "numeric",
                "expression": f"{only} * {only}",
                "rationale": "Mock quadratic term for the first numeric column.",
            }
        )

    def _provider(_prompt: str) -> str:
        return json.dumps({"features": features})

    return _provider


def _llm_plan_for_cli(
    data: Path,
    *,
    target: str | None,
    id_col: str | None,
    provider: str,
    plan_file: Path | None,
    max_features: int,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "KCONV_LLM_API_KEY",
    llm_timeout: float = 60.0,
    llm_max_tokens: int | None = None,
    llm_temperature: float = 0.0,
    llm_seed: int | None = None,
    llm_json_mode: bool = True,
    llm_user_ideas: list[str] | None = None,
    llm_user_code: list[str] | None = None,
    llm_repairs: int = 2,
) -> tuple[DatasetSpec, pd.DataFrame, FeaturePlan, dict]:
    """Detect the dataset and produce a validated plan without executing code."""
    if plan_file is None and provider == "plan":
        raise KconvError("--provider plan needs --plan-file.", "Pass the JSON produced by a reviewed LLM.")
    spec = detect_dataset(data, target=target, id=id_col)
    frame = read_table(spec.train)
    allowed = _llm_allowed_columns(frame, spec)
    sample = frame.head(200)
    sample_rows = _llm_sample_rows(frame, allowed)
    prompt = build_prompt(allowed, sample_rows, max_features=max_features)
    if plan_file is not None:
        plan = parse_plan(
            plan_file.read_text(encoding="utf-8"),
            columns=allowed,
            max_features=max_features,
            reserved=_llm_reserved_columns(spec),
            numeric_columns=spec.numeric_columns,
        )
        metadata = build_feature_log(
            plan,
            provider="plan",
            model=None,
            prompt=None,
            sample_rows=len(sample_rows),
            max_features=max_features,
            plan_file=plan_file,
            approved=False,
        )
    elif provider == "mock":
        plan = parse_plan(
            _mock_llm(sample, allowed)(prompt),
            columns=allowed,
            max_features=max_features,
            reserved=_llm_reserved_columns(spec),
            numeric_columns=spec.numeric_columns,
        )
        metadata = build_feature_log(
            plan,
            provider="mock",
            model="deterministic-mock",
            prompt=prompt,
            sample_rows=len(sample_rows),
            max_features=max_features,
            approved=False,
        )
    elif provider in {"openai", "openai-compatible", "local"}:
        plan, metadata = _llm_plan_iterative_for_cli(
            spec,
            frame,
            allowed,
            sample_rows,
            provider=provider,
            max_features=max_features,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key_env=llm_api_key_env,
            llm_timeout=llm_timeout,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
            llm_json_mode=llm_json_mode,
            llm_user_ideas=list(llm_user_ideas or []),
            llm_user_code=list(llm_user_code or []),
            llm_repairs=llm_repairs,
        )
    else:
        raise KconvError(
            f"Unknown LLM provider '{provider}'.",
            "Use --provider mock (offline demo), plan (reviewed JSON), local (loopback server), "
            "or openai (opt-in API call).",
        )
    return spec, frame, plan, metadata


def _read_llm_code_files(paths: list[Path]) -> list[str]:
    snippets: list[str] = []
    for path in paths:
        try:
            snippets.append(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise KconvError(
                f"Could not read LLM code file: {path}",
                "Pass a readable UTF-8 file, or omit --llm-code-file.",
            ) from exc
    return snippets


def _llm_plan_iterative_for_cli(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    allowed: list[str],
    sample_rows: list[dict],
    *,
    provider: str,
    max_features: int,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key_env: str,
    llm_timeout: float,
    llm_max_tokens: int | None,
    llm_temperature: float,
    llm_seed: int | None,
    llm_json_mode: bool,
    llm_user_ideas: list[str],
    llm_user_code: list[str],
    llm_repairs: int,
) -> tuple[FeaturePlan, dict]:
    """Generate one validated feature per call, repairing bad answers inline."""
    local = provider == "local"
    if local:
        base_url, model = resolve_local_target(llm_base_url, llm_model)
    else:
        model = llm_model
        if not model:
            raise KconvError(
                "--llm-model is required for --provider openai.",
                "Example: --llm-model gpt-4o-mini --llm-api-key-env KCONV_LLM_API_KEY.",
            )
        base_url = llm_base_url or "https://api.openai.com/v1"
    api_key = resolve_api_key(
        llm_api_key_env,
        base_url=base_url,
        allow_local_no_key=local,
    )

    def make_client(json_mode: bool) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=llm_timeout,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
            seed=llm_seed,
            json_mode=json_mode,
        )

    effective_json_mode = llm_json_mode
    client = make_client(effective_json_mode)
    loopback = any(
        marker in base_url.casefold()
        for marker in ("127.0.0.1", "localhost", "[::1]")
    )
    # Preserve the original one-call behaviour for hosted providers.  Local
    # endpoints and explicit user hints use one-feature-per-call generation so
    # a 4096-token context is enough for a useful feature set.
    per_feature = (
        local
        or loopback
        or bool(llm_user_ideas)
        or bool(llm_user_code)
        or llm_repairs != 2
    )
    if not per_feature:
        prompt = build_prompt(
            {"columns": allowed},
            sample_rows,
            max_features=max_features,
        )
        try:
            response = client.complete(prompt)
        except KconvError:
            raise
        try:
            plan = parse_plan(
                response.text,
                columns=allowed,
                max_features=max_features,
                reserved=_llm_reserved_columns(spec),
                numeric_columns=spec.numeric_columns,
            )
        except KconvError as exc:
            finish_reason = (response.finish_reason or "").casefold()
            if finish_reason in {"length", "max_tokens"}:
                raise KconvError(
                    "The LLM response was truncated before valid JSON was complete.",
                    "Increase --llm-max-tokens or use --provider local for one-feature-per-call generation.",
                ) from exc
            raise
        metadata = build_feature_log(
            plan,
            provider=response.provider,
            model=response.model,
            prompt=prompt,
            usage=response.usage,
            duration_sec=response.duration_sec,
            request_id=response.request_id,
            finish_reason=response.finish_reason,
            sample_rows=len(sample_rows),
            max_features=max_features,
            plan_file=None,
            approved=False,
            temperature=llm_temperature,
            seed=llm_seed,
            max_tokens=llm_max_tokens,
            json_mode=effective_json_mode,
        )
        return plan, metadata

    def run_generation():
        return generate_features_iteratively(
            {"columns": allowed},
            sample_rows,
            client.complete,
            max_features=max_features,
            reserved=_llm_reserved_columns(spec),
            numeric_columns=spec.numeric_columns,
            user_ideas=llm_user_ideas,
            user_code=llm_user_code,
            max_repairs=llm_repairs,
        )

    try:
        generation = run_generation()
    except KconvError as exc:
        # Some llama.cpp builds reject response_format with HTTP 400.  Retry
        # once without it for local endpoints; the same compact prompt is sent.
        if local and effective_json_mode and "HTTP 400" in str(exc):
            effective_json_mode = False
            client = make_client(False)
            generation = run_generation()
        else:
            raise

    plan = generation.plan
    last = generation.responses[-1] if generation.responses else None
    prompt_text = "\n\n".join(attempt.prompt for attempt in generation.attempts)
    metadata = build_feature_log(
        plan,
        provider=last.provider if last is not None else provider,
        model=last.model if last is not None else model,
        prompt=prompt_text or None,
        usage=generation.usage,
        duration_sec=generation.duration_sec,
        request_id=last.request_id if last is not None else None,
        finish_reason=last.finish_reason if last is not None else None,
        sample_rows=len(sample_rows),
        max_features=max_features,
        plan_file=None,
        approved=False,
        temperature=llm_temperature,
        seed=llm_seed,
        max_tokens=llm_max_tokens,
        json_mode=effective_json_mode,
    )
    metadata["generation"] = generation.to_dict()
    return plan, metadata


def _llm_metadata_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.llm.json")


def _check_llm_output_paths(
    spec: DatasetSpec,
    *,
    output: Path | None,
    apply_output: Path | None,
    llm_log: Path | None,
) -> None:
    """Reject output/log collisions and writes over source dataset files."""
    protected = {
        path.resolve()
        for path in (spec.train, spec.test, spec.sample_submission)
        if path is not None
    }
    seen: dict[Path, str] = {}
    for label, path in (("--output", output), ("--apply", apply_output), ("--llm-log", llm_log)):
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in protected:
            raise KconvError(
                f"{label} would overwrite an input dataset file.",
                "Choose a new output path; source files are never modified by kconv.",
            )
        if resolved in seen:
            raise KconvError(
                f"{label} and {seen[resolved]} resolve to the same path.",
                "Use distinct paths for the plan, augmented table and metadata sidecar.",
            )
        seen[resolved] = label


def _refresh_plan_metadata(metadata: dict, plan_file: Path) -> dict:
    payload = dict(metadata)
    payload["plan_file"] = str(plan_file)
    payload["plan_sha256"] = file_sha256(plan_file)
    return payload


def _llm_metadata_for_run(plan_file: Path, llm_log: Path | None) -> dict:
    """Load proposal metadata from an explicit log or the propose sidecar.

    The metadata must belong to the exact plan being approved: a hash or
    proposal mismatch fails closed instead of logging stale proposals.
    """
    candidate = llm_log or _llm_metadata_path(plan_file)
    if not plan_file.is_file():
        raise KconvError(
            f"LLM plan file not found: {plan_file}",
            "Pass the plan produced by 'kconv experiments propose'.",
        )
    plan = parse_plan(plan_file.read_text(encoding="utf-8"), max_features=50)
    current_hash = file_sha256(plan_file)
    current_proposals = [proposal.to_dict() for proposal in plan.proposals]
    if candidate.is_file():
        payload = read_json(candidate, default=None)
        if isinstance(payload, dict):
            stored_hash = payload.get("plan_sha256")
            if stored_hash != current_hash:
                raise KconvError(
                    "LLM metadata does not match the current plan (plan_sha256 mismatch).",
                    "The plan was edited or the sidecar belongs to another file; re-run "
                    "'kconv experiments propose' and review the new <plan>.llm.json.",
                )
            if payload.get("proposals") != current_proposals:
                raise KconvError(
                    "LLM metadata proposals do not match the current plan.",
                    "Re-run 'kconv experiments propose' and review the regenerated metadata.",
                )
            payload["approved"] = True
            payload["approval_required"] = True
            payload["plan_file"] = str(plan_file)
            payload["plan_sha256"] = current_hash
            return payload
    return build_feature_log(
        plan,
        provider="plan",
        model=None,
        prompt=None,
        plan_file=plan_file,
        approved=True,
    )



def _augment_data_with_plan(
    data: Path,
    *,
    target: str | None,
    id_col: str | None,
    plan_file: Path,
    ignore: str | None = None,
) -> tuple[Path, dict | None]:
    """Stage a reviewed plan for a run.

    Every validated plan is applied as a fold-aware ``features.fold_plan``
    inside FeatureBuilder.  The original dataset path stays in config.yaml, so
    resume/project transfer never depends on an OS temp directory.
    """
    if data.suffix.lower() == ".zip":
        raise KconvError(
            "Zip archives cannot be augmented in place.",
            "Unpack the archive and pass the folder or train file as --data.",
        )
    spec = detect_dataset(data, target=target, id=id_col)
    ignored = {part.strip() for part in (ignore or "").split(",") if part.strip()}
    allowed = [column for column in spec.feature_columns if column not in ignored]
    if not allowed:
        raise KconvError(
            "No feature columns remain after applying --ignore.",
            "Keep at least one column available to the feature plan.",
        )
    # A reviewed plan may legitimately contain more than the default five
    # proposals; keep a generous cap instead of silently dropping approved
    # features.  parse_plan still rejects duplicates and unknown columns.
    plan = parse_plan(
        plan_file.read_text(encoding="utf-8"),
        columns=allowed,
        max_features=50,
        reserved=_llm_reserved_columns(spec),
        numeric_columns=spec.numeric_columns,
    )
    for warning in plan.warnings:
        console.print(f"[yellow]Plan warning:[/yellow] {warning}")
    # Parse every expression now, before any stage runs.  This catches malformed
    # calls and function names that parse_plan's identifier scan cannot check.
    for proposal in plan.proposals:
        parse_fold_expression(proposal.expression, allowed)
    # A plan may only reference columns that also exist in test; otherwise the
    # failure would surface after ingest/train, when the config already points
    # at a staged dataset.
    test_frame: pd.DataFrame | None = None
    if spec.test is not None and Path(spec.test).exists():
        test_frame = read_table(spec.test)
        test_columns = {str(column) for column in test_frame.columns}
        for proposal in plan.proposals:
            missing = expression_columns(proposal.expression) - test_columns
            if missing:
                raise KconvError(
                    f"Feature '{proposal.name}' references columns missing from test: "
                    f"{', '.join(sorted(missing))}.",
                    "Use only columns present in both train and test.",
                )
    payload = {"features": [proposal.to_dict() for proposal in plan.proposals]}
    return data, payload


def _parse_model_params_json(raw: str | None) -> dict[str, dict[str, Any]]:
    """Parse the optional CLI JSON object used for manual model overrides."""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KconvError(
            "--model-params is not valid JSON.",
            'Pass an object keyed by model id, e.g. \'{"lgbm": {"n_estimators": 300}}\'.',
        ) from exc
    if not isinstance(payload, dict):
        raise KconvError(
            "--model-params must be a JSON object keyed by model id.",
            'Example: {"lgbm": {"n_estimators": 300}, "xgb": {"max_depth": 4}}.',
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in payload.items():
        if not isinstance(value, dict):
            raise KconvError(
                f"--model-params for '{name}' must be a JSON object.",
                'Example: {"n_estimators": 300, "learning_rate": 0.05}.',
            )
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise KconvError(
                f"--model-params for '{name}' contains a non-finite or non-JSON value.",
                "Use finite JSON numbers, strings, booleans, lists or objects.",
            ) from exc
        normalized[str(name)] = dict(value)
    return normalized


@app.command()
def run(
    data: Path | None = typer.Option(None, "--data", help="Dataset folder, CSV/parquet, or zip."),
    target: str | None = typer.Option(None, "--target", help="Target column; auto-detected when omitted."),
    id_col: str | None = typer.Option(None, "--id", help="Id column; auto-detected when omitted."),
    ignore: str | None = typer.Option(None, "--ignore", help="Comma-separated columns to exclude from features."),
    sample_pct: int | None = typer.Option(
        None,
        "--sample-pct",
        min=1,
        max=100,
        help="Train on a deterministic sample of this percentage of rows.",
    ),
    recipe: str = typer.Option("pesnik/baseline", "--recipe", help="Methodology recipe id."),
    strategy: str = typer.Option("fast", "--strategy", help="fast|standard|deep."),
    profile: int | None = typer.Option(
        None,
        "--profile",
        min=1,
        max=5,
        help="Runtime profile 1-5: memory/CPU/GPU preset; overrides --strategy.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="auto|cpu|gpu|cuda; profiles 1-3 are CPU-only.",
    ),
    cpu_threads: int | None = typer.Option(
        None,
        "--cpu-threads",
        min=0,
        max=256,
        help="CPU threads for boosted trees; requires --profile; 0 = all cores.",
    ),
    ram_gb: int | None = typer.Option(
        None,
        "--ram-gb",
        min=1,
        max=1024,
        help="RAM budget in GB; requires --profile; recorded, not OS-enforced.",
    ),
    vram_gb: int | None = typer.Option(
        None,
        "--vram-gb",
        min=0,
        max=256,
        help="VRAM budget in GB; requires --profile; recorded, not OS-enforced.",
    ),
    model_params: str | None = typer.Option(
        None,
        "--model-params",
        help='JSON object of manual adapter params, e.g. \'{"lgbm":{"n_estimators":300}}\'.',
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        help="Comma-separated model ids, or 'all' for every task-compatible model; defaults to the recipe.",
    ),
    name: str | None = typer.Option(None, "--name", help="Project slug; defaults to the dataset name."),
    test: Path | None = typer.Option(None, "--test", help="Explicit test file."),
    sample_submission: Path | None = typer.Option(None, "--sample-submission", help="Explicit sample submission file."),
    task: str = typer.Option("auto", "--task", help="auto|binary|multiclass|regression."),
    metric: str = typer.Option("auto", "--metric", help="Metric name; auto-detected when omitted."),
    config: Path | None = typer.Option(None, "--config", help="Run an existing config.yaml."),
    comp: str | None = typer.Option(None, "--comp", help="Run competitions/<slug>/config.yaml."),
    until: str = typer.Option("report", "--until", help="Stop after ingest|validate|hpo|train|blend|submit|report."),
    resume: bool = typer.Option(False, "--resume", help="Resume the latest run if the config matches."),
    force: bool = typer.Option(False, "--force", help="Re-run stages even when marked done."),
    no_run: bool = typer.Option(False, "--no-run", help="Create config only; do not execute."),
    experiment: str | None = typer.Option(
        None,
        "--experiment",
        help="Apply a curated experiment before running; only llm-feature-proposals is wired.",
    ),
    approve_llm_features: bool = typer.Option(
        False,
        "--approve-llm-features",
        help="Confirm that LLM-proposed features were reviewed before materializing them.",
    ),
    llm_plan_file: Path | None = typer.Option(
        None,
        "--llm-plan-file",
        help="Validated JSON plan from 'kconv experiments propose'.",
    ),
    llm_log: Path | None = typer.Option(
        None,
        "--llm-log",
        help="Proposal metadata JSON; defaults to <plan>.llm.json next to --llm-plan-file.",
    ),
) -> None:
    """Run or create a project from a dataset."""
    if config is not None and data is not None:
        raise typer.BadParameter("Use either --config or --data, not both.")
    data_stem = data.stem if data is not None else None
    llm_metadata: dict | None = None
    fold_plan_payload: dict | None = None
    try:
        if (config is not None or comp is not None) and any(
            value is not None for value in (cpu_threads, ram_gb, vram_gb, model_params)
        ):
            _abort(
                KconvError(
                    "Resource/model overrides require --data.",
                    "Create a new project with --data, or edit models[].params in config.yaml. "
                    "runtime.* is a recorded profile snapshot; it is not re-applied at run time.",
                )
            )
        parsed_model_params = _parse_model_params_json(model_params)
        if experiment is None and (approve_llm_features or llm_plan_file is not None):
            _abort(
                KconvError(
                    "--approve-llm-features/--llm-plan-file require --experiment.",
                    "Use --experiment llm-feature-proposals with --data.",
                )
            )
        if experiment is not None:
            if data is None:
                _abort(KconvError("--experiment requires --data.", "Experiments augment a dataset before the run."))
            item = get_experiment(experiment)
            if item.id != _LLM_EXPERIMENT:
                _abort(
                    KconvError(
                        f"Experiment '{item.id}' is not wired into 'kconv run'.",
                        "Only llm-feature-proposals is executable here; use 'kconv experiments show <id>' to inspect ideas.",
                    )
                )
            if not approve_llm_features:
                _abort(
                    KconvError(
                        "LLM feature proposals require explicit approval.",
                        "Review the plan, then re-run with --approve-llm-features.",
                    )
                )
            if llm_plan_file is None:
                _abort(
                    KconvError(
                        "No LLM feature plan supplied.",
                        "Run 'kconv experiments propose --data ... --output plan.json', review it, then pass --llm-plan-file plan.json.",
                    )
                )
            data, fold_plan_payload = _augment_data_with_plan(
                data,
                target=target,
                id_col=id_col,
                plan_file=llm_plan_file,
                ignore=ignore,
            )
            llm_metadata = _llm_metadata_for_run(llm_plan_file, llm_log)
            console.print(f"[green]LLM features applied[/green] {llm_plan_file} -> {data}")
        if config is not None or comp is not None:
            project_dir, config_path, cfg = _config_for(comp, config)
            if not config_path.exists():
                _abort(KconvError(f"Config not found: {config_path}", "Create it with kconv run --data ..."))
            if sample_pct is not None and cfg.data.sample_pct != sample_pct:
                cfg = cfg.model_copy(
                    update={"data": cfg.data.model_copy(update={"sample_pct": sample_pct})}
                )
                save_config(cfg, config_path)
        elif data is not None:
            project_dir, config_path, cfg = build_config(
                data=data,
                target=target,
                recipe_id=recipe,
                strategy=strategy,
                models=models,
                name=name or (data_stem if experiment is not None else None),
                test=test,
                sample_submission=sample_submission,
                task=task,
                metric=metric,
                id_col=id_col,
                ignore=ignore,
                sample_pct=100 if sample_pct is None else sample_pct,
                profile=profile,
                device=device,
                cpu_threads=cpu_threads,
                ram_gb=ram_gb,
                vram_gb=vram_gb,
                model_params=parsed_model_params,
            )
        else:
            _abort(KconvError("No dataset supplied.", "Use --data PATH or --config PATH or --comp SLUG."))
            return
        console.print(
            f"[bold]Project[/bold] {cfg.project}  [dim]recipe={cfg.recipe} task={cfg.task} "
            f"metric={cfg.metric} models={','.join(m.name for m in cfg.models)}[/dim]"
        )
        if fold_plan_payload is not None:
            cfg = apply_fold_plan(cfg, fold_plan_payload)
            save_config(cfg, config_path)
            console.print(
                f"[green]Fold-aware plan[/green] {len(fold_plan_payload.get('features', []))} features "
                "will be fitted inside each training fold"
            )
        if cfg.runtime is not None:
            console.print(
                f"Runtime preset (recorded): profile {cfg.runtime.profile} ({cfg.runtime.device})  "
                f"RAM>={cfg.runtime.ram_gb or '?'} GB  "
                f"VRAM>={cfg.runtime.vram_gb if cfg.runtime.vram_gb is not None else '?'} GB  "
                f"threads={cfg.runtime.cpu_threads or 'auto'}; models[].params are effective"
            )
        console.print(f"Config: [cyan]{config_path}[/cyan]")
        if no_run:
            return
        _execute(project_dir, config_path, cfg, until, resume, force, llm_metadata=llm_metadata)
    except KconvError as exc:
        _abort(exc)


@app.command()
def new() -> None:
    """Interactive wizard: dataset -> target -> recipe -> strategy -> models."""
    try:
        data = Path(typer.prompt("Dataset path (folder/CSV/parquet/zip)"))
        from kconv.autodetect import detect_dataset

        spec = detect_dataset(data)
        target = typer.prompt("Target column", default=spec.target)
        recipes = list_recipes()
        table = Table(title="Choose a recipe")
        table.add_column("#", justify="right")
        table.add_column("id", style="cyan")
        table.add_column("best for")
        for index, item in enumerate(recipes, 1):
            table.add_row(str(index), item.id, item.best_for)
        console.print(table)
        default_recipe = "pesnik/baseline"
        recipe_input = typer.prompt("Recipe id or number", default=default_recipe)
        if recipe_input.isdigit() and 1 <= int(recipe_input) <= len(recipes):
            recipe_id = recipes[int(recipe_input) - 1].id
        else:
            recipe_id = recipe_input
        recipe = load_recipe(recipe_id)
        strategy = typer.prompt("Strategy (fast/standard/deep)", default="fast")
        default_models = ",".join(recipe.defaults.models)
        models = typer.prompt("Models (comma-separated)", default=default_models)
        name = typer.prompt("Project slug", default=Path(data).stem or "project")
        project_dir, config_path, cfg = build_config(
            data=data,
            target=target,
            recipe_id=recipe_id,
            strategy=strategy,
            models=models,
            name=name,
            task=spec.task,
            metric=spec.metric,
        )
        console.print(f"Created [cyan]{config_path}[/cyan]")
        if typer.confirm("Run now?", default=True):
            _execute(project_dir, config_path, cfg, "report", resume=False, force=False)
    except KconvError as exc:
        _abort(exc)


@recipes_app.callback(invoke_without_command=True)
def recipes_root(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json"),
    group: str | None = typer.Option(None, "--group", help="Filter by recipe group (baseline, ensemble, engineering, architecture)."),
    kind: str | None = typer.Option(None, "--kind", help="Filter by kind: preset|pipeline."),
) -> None:
    """List built-in and user recipes."""
    if ctx.invoked_subcommand is not None:
        return
    if kind is not None and kind not in {"preset", "pipeline"}:
        _abort(KconvError("Unknown recipe kind.", "Use --kind preset or --kind pipeline."))
        return
    try:
        recipes = list_recipes()
    except KconvError as exc:
        _abort(exc)
        return
    if group:
        recipes = [recipe for recipe in recipes if recipe.group == group]
    if kind:
        recipes = [recipe for recipe in recipes if recipe.kind == kind]
    if not recipes:
        console.print("[yellow]No recipes match the selected filters.[/yellow]")
        return
    if json_output:
        _emit_json([r.model_dump(mode="json") for r in recipes])
        return
    suffix = f" / {group}" if group else ""
    if kind:
        suffix += f" / {kind}"
    table = Table(title="kconv recipes" + suffix)
    table.add_column("id", style="cyan")
    table.add_column("kind")
    table.add_column("group")
    table.add_column("title")
    table.add_column("author")
    table.add_column("license")
    table.add_column("models")
    for recipe in recipes:
        table.add_row(
            recipe.id,
            recipe.kind,
            recipe.group,
            recipe.title,
            recipe.author,
            recipe.license,
            ",".join(recipe.defaults.models),
        )
    console.print(table)
    console.print("[dim]Use 'kconv recipes show <id>' for defaults, blocks and attribution.[/dim]")


@recipes_app.command("show")
def recipes_show(recipe_id: str) -> None:
    """Show one recipe's defaults and source."""
    try:
        recipe = load_recipe(recipe_id)
    except KconvError as exc:
        _abort(exc)
        return
    console.print(f"[bold cyan]{recipe.id}[/bold cyan] — {recipe.title}")
    console.print(f"Kind: {recipe.kind}")
    if recipe.derived_from:
        console.print(f"Derived from: {recipe.derived_from}")
    console.print(f"Group: {recipe.group}")
    console.print(f"Author: {recipe.author}")
    console.print(f"Source: {recipe.source}")
    console.print(f"License: {recipe.license}")
    console.print(f"Tasks: {', '.join(recipe.tasks)}")
    console.print(f"Best for: {recipe.best_for}")
    console.print(f"Models: {', '.join(recipe.defaults.models)}")
    if recipe.pipeline is not None:
        blocks = ", ".join(block.type for block in recipe.pipeline.blocks)
        console.print(f"Blocks: {blocks}")
    else:
        console.print("Blocks: - (preset defaults only)")
    console.print(f"Validation: {recipe.defaults.validation.model_dump(mode='json')}")
    console.print(f"Ensemble: {recipe.defaults.ensemble.model_dump(mode='json')}")
    console.print(f"Signature: {signature_digest(recipe)}")
    console.print(f"Attribution: {recipe.attribution}")


@recipes_app.command("audit")
def recipes_audit(
    strict: bool = typer.Option(False, "--strict", help="Fail on duplicate executable pipelines."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Audit normalized recipe signatures and duplicate variants."""
    try:
        result = audit_recipes(strict=strict)
    except KconvError as exc:
        _abort(exc)
        return
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(
        "[bold]Recipe audit[/bold] "
        + ", ".join(f"{kind}={count}" for kind, count in result.kind_counts.items())
    )
    if not result.duplicate_groups:
        console.print("[green]No duplicate signatures.[/green]")
        return
    table = Table(title="Duplicate signatures")
    table.add_column("recipes")
    table.add_column("kinds")
    table.add_column("pipeline?")
    for group in result.duplicate_groups:
        table.add_row(
            ", ".join(group.recipe_ids),
            ",".join(group.kinds),
            "yes" if group.has_pipeline else "no",
        )
    console.print(table)
    if result.has_pipeline_duplicates:
        console.print("[red]Duplicate executable pipeline signatures found.[/red]")
    else:
        console.print("[dim]Preset duplicates are historical variants and are allowed.[/dim]")


@benchmark_app.command("list")
def benchmark_list(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List fixed benchmark datasets without touching the data files."""
    try:
        registry = load_benchmark_registry()
    except KconvError as exc:
        _abort(exc)
        return
    if json_output:
        _emit_json([dataset.to_dict() for dataset in registry.datasets])
        return
    table = Table(title="kconv benchmark datasets")
    table.add_column("id", style="cyan")
    table.add_column("task")
    table.add_column("metric")
    table.add_column("kind")
    table.add_column("source")
    for dataset in registry.datasets:
        table.add_row(
            dataset.id,
            dataset.task,
            dataset.metric,
            dataset.kind,
            dataset.source,
        )
    console.print(table)
    console.print(f"[dim]Registry: {registry.path}[/dim]")


@benchmark_app.command("run")
def benchmark_run_command(
    dataset: str = typer.Option(..., "--dataset", help="Dataset id from 'kconv benchmark list'."),
    recipes: str = typer.Option("all", "--recipes", help="Comma-separated recipe ids or all."),
    strategy: str = typer.Option("fast", "--strategy", help="fast|standard|deep."),
    models: str | None = typer.Option(None, "--models", help="Optional comma-separated model override."),
    n_splits: int | None = typer.Option(None, "--n-splits", min=2, help="Override CV folds."),
    output: Path | None = typer.Option(None, "--output", help="Scorecard output folder."),
    work_dir: Path | None = typer.Option(None, "--work-dir", help="Keep the benchmark workspace here."),
    keep_artifacts: bool = typer.Option(
        False, "--keep-artifacts", help="Keep the temporary workspace under the output folder."
    ),
    max_minutes: float | None = typer.Option(
        None, "--max-minutes", min=0.1, help="Soft budget; remaining recipes are skipped."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run recipes on one fixed dataset and write scorecard JSON/Markdown."""
    try:
        run = run_benchmark(
            dataset,
            recipes=recipes,
            strategy=strategy,
            models=models,
            n_splits=n_splits,
            output_dir=output,
            work_dir=work_dir,
            keep_artifacts=keep_artifacts,
            max_minutes=max_minutes,
        )
    except KconvError as exc:
        _abort(exc)
        return
    if json_output:
        _emit_json(run.to_dict())
    else:
        table = Table(title=f"kconv benchmark - {run.dataset_id}")
        table.add_column("recipe", style="cyan")
        table.add_column("status")
        table.add_column("metric")
        table.add_column("mean", justify="right")
        table.add_column("folds", justify="right")
        table.add_column("seconds", justify="right")
        table.add_column("submission")
        for result in run.results:
            table.add_row(
                result.recipe_id,
                result.status,
                result.metric,
                "-" if result.mean is None else f"{result.mean:.6f}",
                str(len(result.fold_scores)),
                f"{result.duration_sec:.3f}",
                "yes" if result.submission_valid else "no",
            )
        console.print(table)
        console.print(f"[dim]Scorecard: {run.output_dir / 'scorecard.json'}[/dim]")
        for result in run.results:
            if result.error:
                console.print(f"[red]{result.recipe_id}:[/red] {result.error}")
    if not run.ok:
        raise typer.Exit(1)


@app.command("models")
def models_command(
    json_output: bool = typer.Option(False, "--json"),
    details: bool = typer.Option(False, "--details", help="Show family, tasks and source."),
) -> None:
    """List registered model adapters."""
    names = model_names()
    if json_output:
        _emit_json(names)
        return
    if details:
        table = Table(title=f"kconv models ({len(names)})")
        table.add_column("id", style="cyan")
        table.add_column("family")
        table.add_column("tasks")
        table.add_column("source")
        table.add_column("estimator")
        for info in list_models():
            table.add_row(
                info.id,
                info.family,
                ", ".join(info.tasks),
                info.source,
                info.estimator_path or "-",
            )
        console.print(table)
        return
    console.print("[bold]Models:[/bold] " + ", ".join(names))


@sources_app.callback(invoke_without_command=True)
def sources_command(
    ctx: typer.Context,
    all_sources: bool = typer.Option(False, "--all", help="Include repositories that are not downloaded."),
    category: str | None = typer.Option(None, "--category", help="Filter by toolkit category."),
    license_class: str | None = typer.Option(None, "--license-class", help="permissive|copyleft|unlicensed|restricted|unknown."),
    executable: bool = typer.Option(False, "--executable", help="Only sources with a kconv recipe."),
    pipeline: bool = typer.Option(False, "--pipeline", help="Only sources with pipeline coverage."),
    query: str | None = typer.Option(None, "--query", "-q", help="Search repo, description or recipe."),
    limit: int | None = typer.Option(None, "--limit", help="Limit rows."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List the bundled kaggle-toolkit source catalog."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        items = list_sources(
            downloaded=None if all_sources else True,
            category=category,
            license_class=license_class,
            executable=True if executable else None,
            has_pipeline=True if pipeline else None,
            query=query,
            limit=limit,
        )
    except KconvError as exc:
        _abort(exc)
        return
    if json_output:
        _emit_json([item.to_dict() for item in items])
        return
    table = Table(title=f"kconv sources ({len(items)})")
    table.add_column("id", style="cyan")
    table.add_column("downloaded")
    table.add_column("category")
    table.add_column("license")
    table.add_column("recipe")
    table.add_column("stages")
    for item in items:
        table.add_row(
            item.id,
            "yes" if item.downloaded else "no",
            item.category or "-",
            item.license_class,
            item.recipe_id or "-",
            ",".join(item.pipeline_stages) or "-",
        )
    console.print(table)
    console.print("[dim]Use 'kconv sources show <id>' for details; --all includes 1173 indexed sources.[/dim]")


@sources_app.command("show")
def sources_show(source_id: str) -> None:
    """Show one source and its license/runtime status."""
    try:
        item = get_source(source_id)
    except KconvError as exc:
        _abort(exc)
        return
    console.print(f"[bold cyan]{item.repo}[/bold cyan]")
    console.print(f"Id: {item.id}")
    console.print(f"URL: {item.url}")
    console.print(f"Downloaded: {item.downloaded}")
    console.print(f"Category: {item.category or '-'}")
    console.print(f"License: {item.license or '-'} ({item.license_class})")
    console.print(f"Files: {item.files if item.files is not None else '-'}")
    console.print(f"Executable: {item.executable}")
    console.print(f"Recipe: {item.recipe_id or '-'}")
    console.print(f"Stages: {', '.join(item.pipeline_stages) or '-'}")
    if item.partial_stages:
        console.print(f"Partial stages: {', '.join(item.partial_stages)}")
    console.print(f"Reason: {item.reason}")
    local_path = item.local_path()
    console.print(f"Local path: {local_path or '-'}")


@experiments_app.callback(invoke_without_command=True)
def experiments_root(
    ctx: typer.Context,
    status: str | None = typer.Option(None, "--status", help="idea|prototype|implemented|validated."),
    task: str | None = typer.Option(None, "--task", help="Filter by binary|multiclass|regression."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List curated experiments; only implemented entries are runnable behavior."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        items = list_experiments(status=status, task=task)
    except KconvError as exc:
        _abort(exc)
        return
    if json_output:
        _emit_json([item.to_dict() for item in items])
        return
    table = Table(title=f"kconv experiments ({len(items)})")
    table.add_column("id", style="cyan")
    table.add_column("status")
    table.add_column("tasks")
    table.add_column("gain")
    table.add_column("cost")
    table.add_column("risk")
    for item in items:
        table.add_row(
            item.id,
            item.status,
            ",".join(item.task_types),
            item.expected_gain,
            item.cost,
            item.risk,
        )
    console.print(table)
    console.print("[dim]Use 'kconv experiments show <id>' for hypothesis, source and license.[/dim]")


@experiments_app.command("show")
def experiments_show(experiment_id: str) -> None:
    """Show one experiment hypothesis, source, license and risk."""
    try:
        item = get_experiment(experiment_id)
    except KconvError as exc:
        _abort(exc)
        return
    console.print(f"[bold cyan]{item.id}[/bold cyan] — {item.title}")
    console.print(f"Category: {item.category}")
    console.print(f"Status: {item.status}")
    console.print(f"Hypothesis: {item.hypothesis}")
    console.print(f"Source: {item.source}")
    console.print(f"License: {item.license}")
    console.print(f"Tasks: {', '.join(item.task_types)}")
    console.print(f"Stages: {', '.join(item.stages)}")
    console.print(f"Expected gain: {item.expected_gain}")
    console.print(f"Cost: {item.cost}")
    console.print(f"Risk: {item.risk}")
    console.print(f"Recipe: {item.recipe_id or '-'}")
    console.print(f"Notes: {item.notes}")


@experiments_app.command("propose")
def experiments_propose(
    data: Path = typer.Option(..., "--data", help="Dataset folder, CSV/parquet, or zip."),
    target: str | None = typer.Option(None, "--target", help="Target column; auto-detected when omitted."),
    id_col: str | None = typer.Option(None, "--id", help="Id column; auto-detected when omitted."),
    provider: str = typer.Option(
        "mock",
        "--provider",
        help="mock (offline), plan (reviewed JSON), local (loopback server), or openai (opt-in call).",
    ),
    plan_file: Path | None = typer.Option(None, "--plan-file", help="Existing JSON plan; required for --provider plan."),
    output: Path | None = typer.Option(None, "--output", help="Write the validated plan JSON here."),
    apply_output: Path | None = typer.Option(None, "--apply", help="Write an augmented table with the validated features."),
    max_features: int = typer.Option(5, "--max-features", min=1, max=20),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Model id for openai/local; local auto-detects GET /v1/models when omitted.",
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        help="OpenAI-compatible base URL; local auto-detects llama.cpp :8080, Ollama :11434, vLLM :8000.",
    ),
    llm_api_key_env: str = typer.Option(
        "KCONV_LLM_API_KEY",
        "--llm-api-key-env",
        help="Environment variable holding the API key; OPENAI_API_KEY is a fallback for the default name.",
    ),
    llm_timeout: float = typer.Option(60.0, "--llm-timeout", min=1.0, help="Network timeout in seconds."),
    llm_max_tokens: int | None = typer.Option(None, "--llm-max-tokens", min=1, help="Optional completion cap."),
    llm_temperature: float = typer.Option(0.0, "--llm-temperature", min=0.0, max=2.0),
    llm_seed: int | None = typer.Option(None, "--llm-seed", help="Optional provider seed."),
    llm_json_mode: bool = typer.Option(
        True,
        "--llm-json-mode/--no-llm-json-mode",
        help="Ask the provider for JSON mode; disable only for compatible endpoints that reject it.",
    ),
    llm_idea: list[str] | None = typer.Option(
        None,
        "--llm-idea",
        help="Repeatable idea or hint for the LLM (never executed).",
    ),
    llm_code_file: list[Path] | None = typer.Option(
        None,
        "--llm-code-file",
        help="Repeatable UTF-8 file with user/third-party code as inspiration.",
    ),
    llm_repairs: int = typer.Option(
        2,
        "--llm-repairs",
        min=0,
        max=5,
        help="Validation repair attempts per generated feature.",
    ),
    llm_log: Path | None = typer.Option(
        None,
        "--llm-log",
        help="Write privacy-safe proposal metadata here; defaults to <output>.llm.json.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Propose and validate LLM-style arithmetic features without executing them.

    The model only returns JSON; this command parses it against a strict
    arithmetic whitelist and can materialize the validated plan locally.  Use
    ``--provider plan --plan-file`` for a reviewed answer from any LLM, or
    ``--provider openai`` for an explicit opt-in API call.
    """
    try:
        spec, frame, plan, metadata = _llm_plan_for_cli(
            data,
            target=target,
            id_col=id_col,
            provider=provider,
            plan_file=plan_file,
            max_features=max_features,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key_env=llm_api_key_env,
            llm_timeout=llm_timeout,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
            llm_json_mode=llm_json_mode,
            llm_user_ideas=list(llm_idea or []),
            llm_user_code=_read_llm_code_files(list(llm_code_file or [])),
            llm_repairs=llm_repairs,
        )
        payload = {"features": [proposal.to_dict() for proposal in plan.proposals]}
        log_path = llm_log or (_llm_metadata_path(output) if output is not None else None)
        _check_llm_output_paths(spec, output=output, apply_output=apply_output, llm_log=log_path)
        if json_output:
            _emit_json(payload)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            metadata = _refresh_plan_metadata(metadata, output)
            if not json_output:
                console.print(f"[green]Plan[/green] {output}")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(log_path, metadata)
            if not json_output:
                console.print(f"LLM metadata: [cyan]{log_path}[/cyan]")
        if apply_output is not None:
            if plan_requires_fold_transformer(plan):
                raise KconvError(
                    "This plan contains fold-aware row/group functions; --apply supports arithmetic only.",
                    "The plan was written; run 'kconv run --experiment llm-feature-proposals "
                    "--approve-llm-features --llm-plan-file <plan>' so features are fitted inside each fold.",
                )
            augmented = materialize_plan(plan, frame, columns=_llm_allowed_columns(frame, spec))
            write_table(augmented, apply_output)
            if not json_output:
                console.print(f"[green]Augmented table[/green] {apply_output} (+{len(plan.proposals)} features)")
        if not json_output and output is None and apply_output is None:
            table = Table(title=f"LLM feature proposals ({len(plan.proposals)})")
            table.add_column("name", style="cyan")
            table.add_column("kind")
            table.add_column("expression")
            table.add_column("rationale")
            for proposal in plan.proposals:
                table.add_row(proposal.name, proposal.kind, proposal.expression, proposal.rationale)
            console.print(table)
            console.print(
                "[dim]Review the expressions, then run "
                "'kconv run --data ... --experiment llm-feature-proposals "
                "--approve-llm-features --llm-plan-file plan.json'.[/dim]"
            )
        for warning in plan.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    except KconvError as exc:
        _abort(exc)
        return


@app.command("profiles")
def profiles_command(json_output: bool = typer.Option(False, "--json")) -> None:
    """List the five runtime profiles (RAM, threads, device, model budget)."""
    profiles = list_profiles()
    if json_output:
        _emit_json([profile.to_dict() for profile in profiles])
        return
    table = Table(title="kconv runtime profiles (1=cheapest, 5=max)")
    table.add_column("level", justify="right")
    table.add_column("name", style="cyan")
    table.add_column("RAM")
    table.add_column("VRAM")
    table.add_column("threads")
    table.add_column("device")
    table.add_column("strategy")
    table.add_column("models")
    table.add_column("HPO")
    for profile in profiles:
        table.add_row(
            str(profile.level),
            profile.name,
            f"{profile.ram_gb} GB",
            f"{profile.vram_gb} GB" if profile.vram_gb else "-",
            "auto" if profile.cpu_threads == 0 else str(profile.cpu_threads),
            profile.device,
            profile.strategy,
            "all" if profile.model_limit >= 1000 else str(profile.model_limit),
            "on" if profile.hpo else "off",
        )
    console.print(table)
    console.print("[dim]Use: kconv run --data ... --profile 1..5 [--device auto|cpu|gpu|cuda][/dim]")


@app.command("plugins")
def plugins_command(
    load: bool = typer.Option(False, "--load", help="Import/call installed plugins now."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List installed kconv.plugins entry points; loading is explicit."""
    items = load_plugins() if load else list_plugins()
    if json_output:
        _emit_json([item.to_dict() for item in items])
        return
    if not items:
        console.print("[yellow]No kconv.plugins entry points installed.[/yellow]")
        return
    table = Table(title="kconv plugins")
    table.add_column("name", style="cyan")
    table.add_column("value")
    table.add_column("loaded")
    table.add_column("error")
    for item in items:
        table.add_row(item.name, item.value, "yes" if item.loaded else "no", item.error or "")
    console.print(table)


@app.command("metrics")
def metrics_command(json_output: bool = typer.Option(False, "--json")) -> None:
    """List supported metrics."""
    names = metric_names()
    if json_output:
        _emit_json(names)
        return
    console.print("[bold]Metrics:[/bold] " + ", ".join(names))


@app.command()
def package(
    comp: str | None = typer.Option(None, "--comp", help="Package competitions/<slug>/ latest run."),
    config: Path | None = typer.Option(None, "--config", help="Package the project containing this config."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Zip file or output directory; defaults to packages/<project>-<run>.zip.",
    ),
    run_id: str | None = typer.Option(None, "--run", help="Run id; defaults to the latest run."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing package."),
) -> None:
    """Create an offline zip with submission, report, config and run metadata."""
    try:
        if comp is not None and config is not None:
            raise KconvError("Use either --comp or --config, not both.")
        if config is not None:
            config_path = config.resolve()
            project_dir = config_path.parent
        elif comp is not None:
            project_dir = _project_dir(comp)
            config_path = project_dir / "config.yaml"
        else:
            raise KconvError("No project supplied.", "Pass --comp SLUG or --config PATH.")
        result = create_package(
            project_dir,
            config_path,
            run_id=run_id,
            output=output,
            force=force,
        )
    except KconvError as exc:
        _abort(exc)
        return
    console.print(f"[green]Package[/green] {result['path']}")
    console.print(
        f"Run: [cyan]{result['run_id']}[/cyan]  "
        f"Files: {len(result['files'])}  "
        f"Size: {result['size_bytes']} bytes  "
        f"SHA256: {result['sha256']}"
    )


@app.command()
def export(
    comp: str | None = typer.Option(None, "--comp", help="Export competitions/<slug>/ latest run."),
    config: Path | None = typer.Option(None, "--config", help="Export the project containing this config."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Bundle directory; defaults to inference/<project>-<run>.",
    ),
    run_id: str | None = typer.Option(None, "--run", help="Run id; defaults to the latest run."),
    force: bool = typer.Option(False, "--force", help="Replace an existing bundle directory."),
) -> None:
    """Export a standalone model + features inference bundle."""
    try:
        if comp is not None and config is not None:
            raise KconvError("Use either --comp or --config, not both.")
        if config is not None:
            config_path = config.resolve()
            project_dir = config_path.parent
        elif comp is not None:
            project_dir = _project_dir(comp)
            config_path = project_dir / "config.yaml"
        else:
            raise KconvError("No project supplied.", "Pass --comp SLUG or --config PATH.")
        result = export_inference(
            project_dir,
            config_path,
            run_id=run_id,
            output=output,
            force=force,
        )
    except KconvError as exc:
        _abort(exc)
        return
    console.print(f"[green]Inference bundle[/green] {result['path']}")
    console.print(f"Run: [cyan]{result['run_id']}[/cyan]  Models: {', '.join(result['models'])}")
    console.print("Run: [bold]python submission.py --test test.csv[/bold]")


@app.command()
def status(
    comp: str | None = typer.Option(None, "--comp"),
    config: Path | None = typer.Option(None, "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable status."),
) -> None:
    """Show the latest (or selected) run, including stale/interrupted detection."""
    try:
        from kconv.watchdog import inspect_run

        project_dir = _project_or_cwd(comp, config)
        diagnosis = inspect_run(project_dir, run_id)
        selected = diagnosis.get("run_id")
        state_path = project_dir / "runs" / str(selected) / "state.json" if selected else None
        state = PipelineState.load(state_path) if state_path and state_path.exists() else None
    except KconvError as exc:
        _abort(exc)
        return
    if not selected or state is None:
        console.print("[yellow]No runs found.[/yellow]")
        return
    payload = {
        **diagnosis,
        "stages": {
            name: {
                "status": entry.status,
                "started_at": entry.started_at,
                "finished_at": entry.finished_at,
                "duration_sec": entry.duration_sec,
                "error": entry.error,
                "artifacts": list(entry.artifacts),
            }
            for name, entry in state.stages.items()
        },
        "durations_sec": state.durations(),
    }
    if as_json:
        _emit_json(payload)
        return
    table = Table(title=f"{comp or project_dir.name} / {selected}")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Seconds", justify="right")
    table.add_column("Error")
    for name, entry in state.stages.items():
        table.add_row(
            name,
            entry.status,
            f"{entry.duration_sec:.2f}" if entry.duration_sec is not None else "",
            entry.error or "",
        )
    console.print(table)
    console.print(f"Run directory: [cyan]{project_dir / 'runs' / selected}[/cyan]")
    if diagnosis.get("stale"):
        console.print(f"[yellow]Stale run:[/yellow] {diagnosis.get('reason')}")
        console.print("[yellow]Run `kconv recover` to mark it interrupted and write a postmortem.[/yellow]")
    if diagnosis.get("failure_report"):
        console.print(f"Failure report: [red]{project_dir / 'runs' / selected / str(diagnosis['failure_report'])}[/red]")


@app.command()
def recover(
    comp: str | None = typer.Option(None, "--comp"),
    config: Path | None = typer.Option(None, "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume: bool = typer.Option(False, "--resume", help="Resume after marking the run interrupted."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Recover a run whose process died without writing a final status."""
    from kconv.watchdog import recover_stale_run

    try:
        project_dir = _project_or_cwd(comp, config)
        result = recover_stale_run(project_dir, run_id)
    except (KconvError, ValueError) as exc:
        if as_json:
            _emit_json({"ok": False, "error": str(exc)})
            raise typer.Exit(1) from exc
        _abort(exc)
        return
    if as_json:
        _emit_json(result)
    elif result.get("recovered"):
        console.print(f"[yellow]Run {result['run_id']} marked interrupted.[/yellow]")
        console.print(f"Postmortem: [cyan]{project_dir / 'runs' / result['run_id'] / str(result['failure_report'])}[/cyan]")
    else:
        console.print(f"[green]Run is not stale ({result.get('status')}).[/green]")
    if resume and result.get("recovered"):
        selected = str(result["run_id"])
        run_meta_path = project_dir / "runs" / selected / "run.json"
        from kconv.context import read_json

        run_meta = read_json(run_meta_path, default={}) or {}
        config_path = Path(str(run_meta.get("config_path") or project_dir / "config.yaml"))
        if not config_path.exists():
            console.print(f"[red]Cannot resume: config not found at {config_path}.[/red]")
            raise typer.Exit(1)
        _, _, cfg = _config_for(comp, config_path)
        _execute(project_dir, config_path, cfg, "report", True, False)


@app.command()
def runs(
    comp: str | None = typer.Option(None, "--comp"),
    config: Path | None = typer.Option(None, "--config"),
    status_filter: str | None = typer.Option(None, "--status"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List indexed runs, newest first."""
    from kconv.runs_index import list_runs

    project_dir = _project_or_cwd(comp, config)
    items = list_runs(project_dir, status=status_filter, limit=limit)
    if as_json:
        _emit_json({"project": project_dir.name, "runs": items})
        return
    table = Table(title=f"Runs · {project_dir.name}")
    for column in ("run", "status", "metric", "score", "models", "time", "tags"):
        table.add_column(column)
    for item in items:
        score = item.get("score")
        table.add_row(
            str(item.get("run_id")),
            str(item.get("status")),
            str(item.get("metric") or "—"),
            f"{float(score):.5f}" if isinstance(score, (int, float)) else "—",
            "+".join(item.get("models") or []) or "—",
            f"{float(item['duration_sec']):.1f}s" if isinstance(item.get("duration_sec"), (int, float)) else "—",
            ", ".join(item.get("tags") or []) or "—",
        )
    console.print(table)


@app.command()
def compare(
    left: str = typer.Argument(..., help="Left run id."),
    right: str = typer.Argument(..., help="Right run id."),
    comp: str | None = typer.Option(None, "--comp"),
    config: Path | None = typer.Option(None, "--config"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Compare two runs: metrics, status and config diff."""
    from kconv.runs_index import compare_runs

    project_dir = _project_or_cwd(comp, config)
    try:
        result = compare_runs(project_dir, left, right)
    except ValueError as exc:
        _abort(KconvError(str(exc), "Check run ids with `kconv runs`."))
        return
    if as_json:
        _emit_json(result)
        return
    table = Table(title=f"{left} → {right}")
    table.add_column("Field")
    table.add_column("Left")
    table.add_column("Right")
    for key in ("status", "metric", "score", "duration_sec", "models", "warning_count"):
        table.add_row(key, str(result["left"].get(key)), str(result["right"].get(key)))
    console.print(table)
    if result.get("config_diff"):
        diff = Table(title="Config diff")
        diff.add_column("Path")
        diff.add_column("Left")
        diff.add_column("Right")
        for item in result["config_diff"][:50]:
            diff.add_row(str(item["path"]), str(item["left"]), str(item["right"]))
        console.print(diff)


@app.command()
def best(
    comp: str | None = typer.Option(None, "--comp"),
    config: Path | None = typer.Option(None, "--config"),
    metric: str | None = typer.Option(None, "--metric"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show the best completed run by OOF score."""
    from kconv.runs_index import best_run

    project_dir = _project_or_cwd(comp, config)
    result = best_run(project_dir, metric=metric)
    if as_json:
        _emit_json(result or {})
        return
    if not result:
        console.print("[yellow]No completed runs with a score.[/yellow]")
        return
    console.print(
        f"[green]{result['run_id']}[/green] · {result.get('metric')} = {result.get('score')} "
        f"· {', '.join(result.get('models') or [])}"
    )


@app.command()
def tag(
    run_id: str = typer.Argument(..., help="Run id."),
    tags: list[str] = typer.Argument(..., help="Tags to set on the run."),
    comp: str | None = typer.Option(None, "--comp"),
    config: Path | None = typer.Option(None, "--config"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Set searchable tags on a run."""
    from kconv.runs_index import tag_run

    project_dir = _project_or_cwd(comp, config)
    try:
        result = tag_run(project_dir, run_id, tags)
    except ValueError as exc:
        _abort(KconvError(str(exc), "Check run ids with `kconv runs`."))
        return
    if as_json:
        _emit_json(result)
    else:
        console.print(f"[green]Tagged {run_id}:[/green] {', '.join(result.get('tags') or [])}")


@app.command()
def report(
    comp: str | None = typer.Option(None, "--comp"),
    last: bool = typer.Option(False, "--last", help="Use the latest run (default behavior)."),
) -> None:
    """Print the Markdown report for the latest run."""
    try:
        project_dir = _project_dir(comp)
        run_id, state, run_dir = _latest_state(project_dir)
    except KconvError as exc:
        _abort(exc)
        return
    if run_dir is None:
        console.print("[yellow]No runs found.[/yellow]")
        return
    path = run_dir / "05_report" / "report.md"
    if not path.exists():
        console.print(f"[yellow]Report not found: {path}[/yellow]")
        return
    console.print(Markdown(path.read_text(encoding="utf-8")))


@kaggle_app.command("auth")
def kaggle_auth(json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON.")) -> None:
    """Check the Kaggle CLI and credential presence without printing secrets."""
    status = check_auth()
    if json_output:
        _emit_json(status.to_dict())
        return
    if status.ready:
        console.print("[green]Kaggle ready[/green]")
    else:
        console.print("[yellow]Kaggle not ready[/yellow]")
    console.print(f"  CLI: {'found' if status.cli_available else 'not found'}" + (f" ({status.cli_path})" if status.cli_path else ""))
    if status.cli_version:
        console.print(f"  Version: {status.cli_version}")
    console.print(f"  Python package: {'installed' if status.python_package else 'not installed'}")
    if status.credential_source:
        console.print(f"  Credentials: {status.credential_source}")
    else:
        console.print("  Credentials: not configured")
    console.print(f"  Config path: {status.config_path}")
    for warning in status.warnings:
        console.print(f"  [yellow]Warning:[/yellow] {warning}")


@kaggle_app.command("submit")
def kaggle_submit(
    comp: str | None = typer.Option(None, "--comp", help="Project under competitions/<slug>/ (also the default competition slug)."),
    competition: str | None = typer.Option(None, "--competition", help="Kaggle competition slug; defaults to --comp."),
    file: Path | None = typer.Option(None, "--file", help="Submission CSV; defaults to the latest run's 04_submit/submission.csv."),
    message: str = typer.Option("kconv submission", "--message", help="Kaggle submission message."),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Actually upload. Without this flag kconv only validates and prints the command.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate a submission and upload it only with explicit --execute."""
    try:
        slug = competition or comp
        if not slug:
            raise KconvError(
                "Competition slug is required.",
                "Pass --comp <project> or --competition <kaggle-slug>.",
            )
        submission = file
        if submission is None:
            if not comp:
                raise KconvError(
                    "No submission file supplied.",
                    "Pass --file PATH, or --comp <project> to use its latest run.",
                )
            project_dir = _project_dir(comp)
            _run_id, _state, run_dir = _latest_state(project_dir)
            if run_dir is None:
                raise KconvError(
                    f"No runs found for project '{comp}'.",
                    "Run 'kconv run --comp ...' first or pass --file.",
                )
            submission = run_dir / "04_submit" / "submission.csv"
        result = submit_submission(slug, submission, message=message, execute=execute)
        if json_output:
            _emit_json(result.to_dict())
            return
        if not result.executed:
            console.print("[yellow]Dry run[/yellow] — no upload was attempted.")
            console.print(f"Competition: [cyan]{result.plan.competition}[/cyan]")
            console.print(f"File: {result.plan.file}")
            console.print(f"Rows: {result.plan.rows}  Columns: {', '.join(result.plan.columns)}")
            console.print(f"SHA256: {result.plan.sha256}")
            console.print("Command: [bold]" + " ".join(result.plan.command) + "[/bold]")
            console.print("[dim]Re-run with --execute after verifying the file and message.[/dim]")
            return
        console.print(f"[green]Submitted[/green] {result.plan.competition} ({result.plan.file})")
    except KconvError as exc:
        _abort(exc)


@app.command("web")
def web_command(
    data: Path = typer.Option(
        Path("data"),
        "--data",
        help="Папка, куда сайт складывает train/test/sample_submission.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Адрес локального сервера."),
    port: int = typer.Option(8765, "--port", help="Порт локального сервера; 0 выберет свободный."),
    no_open: bool = typer.Option(False, "--no-open", help="Не открывать браузер автоматически."),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Разрешить привязку не к loopback. Небезопасно: API получит доступ из сети.",
    ),
) -> None:
    """Открыть локальный визуальный интерфейс kconv."""
    from kconv.web import WebError, serve

    if allow_remote:
        console.print(
            "[bold yellow]Внимание:[/bold yellow] --allow-remote открывает web UI из сети. "
            "Любой, кто получит доступ к порту, сможет запускать обучение и отправлять submission."
        )
    try:
        serve(
            host=host,
            port=port,
            data_dir=data,
            open_browser=not no_open,
            allow_remote=allow_remote,
        )
    except WebError as exc:
        _abort(KconvError(exc.message, exc.hint))
    except OSError as exc:
        _abort(
            KconvError(
                f"Не удалось запустить локальный сервер: {exc}",
                "Проверьте, свободен ли порт, или передайте --port 0.",
            )
        )


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", help="Validate a config file during preflight."),
    comp: str | None = typer.Option(None, "--comp", help="Validate competitions/<slug>/config.yaml."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON report for CI/web."),
) -> None:
    """Preflight the environment, optional models, resources and config."""
    from kconv.preflight import run_preflight

    project_dir = Path.cwd()
    cfg: Config | None = None
    config_error: KconvError | None = None
    try:
        if config is not None or comp is not None:
            project_dir, _, cfg = _config_for(comp, config)
    except KconvError as exc:
        config_error = exc
    if config_error is not None:
        payload = {
            "ok": False,
            "status": "error",
            "project_dir": str(project_dir),
            "checks": [
                {
                    "name": "config",
                    "status": "error",
                    "message": config_error.message,
                    "hint": config_error.hint,
                }
            ],
            "summary": {"ok": 0, "warn": 0, "error": 1},
        }
        if as_json:
            _emit_json(payload)
        else:
            console.print(f"[bold red]config:[/bold red] {config_error.message}")
            if config_error.hint:
                console.print(f"[yellow]Hint:[/yellow] {config_error.hint}")
        raise typer.Exit(1)
    report = run_preflight(config=cfg, project_dir=project_dir)
    if as_json:
        _emit_json(report)
        if not report["ok"]:
            raise typer.Exit(1)
        return
    console.print("[bold]kconv doctor[/bold]")
    colors = {"ok": "green", "warn": "yellow", "error": "red"}
    labels = {"ok": "OK", "warn": "WARN", "error": "ERROR"}
    for check in report["checks"]:
        status = check["status"]
        color = colors.get(status, "white")
        console.print(f"  [{color}]{labels.get(status, status)}[/{color}] {check['name']}: {check['message']}")
        if check.get("hint"):
            console.print(f"       [yellow]{check['hint']}[/yellow]")
    summary = report["summary"]
    console.print(
        f"  Summary: {summary.get('ok', 0)} ok, {summary.get('warn', 0)} warnings, "
        f"{summary.get('error', 0)} errors"
    )
    auth = check_auth()
    auth_label = "ready" if auth.ready else "not configured"
    console.print(f"  Kaggle auth: {auth_label}")
    stats = catalog_stats()
    console.print(f"  Recipes: {len(list_recipes())}; experiments: {len(list_experiments())}; "
                  f"runtime profiles: {len(list_profiles())}")
    console.print(f"  Sources: {stats['downloaded']} downloaded / {stats['total']} indexed; "
                  f"executable: {stats['executable']}")
    console.print(f"  Models: {', '.join(model_names())}")
    if not report["ok"]:
        raise typer.Exit(1)


@config_app.command("validate")
def config_validate(
    config: Path | None = typer.Option(None, "--config", help="Path to config.yaml."),
    comp: str | None = typer.Option(None, "--comp", help="Validate competitions/<slug>/config.yaml."),
) -> None:
    """Validate a config file and print a compact summary."""
    try:
        if config is None and comp is None:
            raise KconvError("No config supplied.", "Pass --config PATH or --comp SLUG.")
        path = config.resolve() if config is not None else _project_dir(comp) / "config.yaml"
        cfg = load_config(path)
    except KconvError as exc:
        _abort(exc)
        return
    console.print(f"[green]Valid[/green] {path}")
    console.print(f"project={cfg.project} recipe={cfg.recipe} task={cfg.task} metric={cfg.metric}")
    console.print(f"models={','.join(m.name for m in cfg.models)} strategy={cfg.strategy}")
    if cfg.runtime is not None:
        console.print(
            f"runtime_profile={cfg.runtime.profile} device={cfg.runtime.device} "
            "(recorded; models[].params are effective)"
        )
    console.print(f"hpo={'on' if cfg.hpo.enabled else 'off'} trials={cfg.hpo.trials}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
