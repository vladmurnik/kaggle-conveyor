"""Build a small offline zip for one completed run.

The archive intentionally contains metadata and the final submission only:
raw train/test tables, OOF predictions and the ``runs/`` tree are not copied.
This keeps packages safe to share and small enough to upload as a Kaggle
dataset or attach to an experiment.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

from kconv.context import file_sha256
from kconv.errors import KconvError
from kconv.state import PipelineState, read_latest_pointer

# destination name -> path relative to the run directory
_REQUIRED_ARTIFACTS = {
    "submission.csv": "04_submit/submission.csv",
    "report.md": "05_report/report.md",
    "blend.json": "03_blend/blend.json",
    "schema.json": "00_ingest/schema.json",
    "data_card.json": "00_ingest/data_card.json",
    "run.json": "run.json",
    "state.json": "state.json",
}

_OPTIONAL_ARTIFACTS = {
    "adversarial.json": "01_validate/adversarial.json",
    "cv_summary.json": "01_validate/cv_summary.json",
    "submission_check.json": "04_submit/submission_check.json",
}


def _resolve_run(project_dir: Path, run_id: str | None) -> tuple[str, Path]:
    selected = run_id or read_latest_pointer(project_dir)
    if not selected:
        raise KconvError(
            f"No runs found for project '{project_dir.name}'.",
            "Run the pipeline first with 'kconv run --comp ...'.",
        )
    if Path(selected).name != selected:
        raise KconvError(f"Invalid run id: {selected}", "Pass a run directory name from runs/.")
    run_dir = project_dir / "runs" / selected
    if not run_dir.is_dir():
        raise KconvError(
            f"Run not found: {run_dir}",
            "Use 'kconv status --comp ...' to see available runs.",
        )
    return selected, run_dir


def _resolve_output(project_dir: Path, run_id: str, output: str | Path | None) -> Path:
    if output is None:
        return project_dir / "packages" / f"{project_dir.name}-{run_id}.zip"
    candidate = Path(output).expanduser()
    if candidate.suffix.lower() != ".zip":
        return candidate / f"{project_dir.name}-{run_id}.zip"
    return candidate


def _check_state(run_dir: Path) -> PipelineState:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise KconvError(
            f"Run state is missing: {state_path}",
            "The run is incomplete; rerun it through report first.",
        )
    try:
        state = PipelineState.load(state_path)
    except ValueError as exc:
        raise KconvError(f"Invalid run state: {exc}", "Rerun the pipeline.") from exc
    if not state.is_done("submit"):
        raise KconvError(
            "The selected run has not completed the submit stage.",
            "Run the pipeline through submit/report before packaging.",
        )
    return state


def _read_entries(
    config_path: Path,
    run_dir: Path,
) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    for archive_name, relative_path in _REQUIRED_ARTIFACTS.items():
        source = run_dir / relative_path
        if not source.is_file():
            raise KconvError(
                f"Run artifact is missing: {relative_path}",
                "Run the pipeline through report before packaging.",
            )
        entries.append((archive_name, source.read_bytes()))
    if not config_path.is_file():
        raise KconvError(
            f"Config not found: {config_path}",
            "Pass the config used by this run with --config.",
        )
    entries.append(("config.yaml", config_path.read_bytes()))
    for archive_name, relative_path in _OPTIONAL_ARTIFACTS.items():
        source = run_dir / relative_path
        if source.is_file():
            entries.append((archive_name, source.read_bytes()))
    # Stable archive order makes repeated packages easier to compare.
    return sorted(entries, key=lambda item: item[0])


def _sha256_sums(entries: list[tuple[str, bytes]]) -> bytes:
    lines = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in entries]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_zip(output_path: Path, entries: list[tuple[str, bytes]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=output_path.name + ".",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in entries:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, data)
            sbom = _sha256_sums(entries)
            info = zipfile.ZipInfo("SHA256SUMS", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, sbom)
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def create_package(
    project_dir: str | Path,
    config_path: str | Path,
    *,
    run_id: str | None = None,
    output: str | Path | None = None,
    force: bool = False,
) -> dict:
    """Create the offline run archive and return its manifest.

    Required files are the final submission, report, blend metadata, ingest
    schema/data card, ``run.json``, ``state.json`` and the config. Optional
    validation/submission diagnostics are included when present.
    """
    project_dir = Path(project_dir).resolve()
    config_path = Path(config_path).resolve()
    selected, run_dir = _resolve_run(project_dir, run_id)
    _check_state(run_dir)
    if not config_path.is_file():
        raise KconvError(
            f"Config not found: {config_path}",
            "Pass the config used by this run with --config.",
        )

    output_path = _resolve_output(project_dir, selected, output).resolve()
    if output_path.exists() and not force:
        raise KconvError(
            f"Package already exists: {output_path}",
            "Pass --force to overwrite it.",
        )

    entries = _read_entries(config_path, run_dir)
    try:
        _write_zip(output_path, entries)
    except OSError as exc:
        raise KconvError(
            f"Could not write package: {exc}",
            "Check that the output directory is writable and try again.",
        ) from exc

    files = [name for name, _ in entries] + ["SHA256SUMS"]
    return {
        "path": str(output_path),
        "run_id": selected,
        "files": files,
        "size_bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
    }
