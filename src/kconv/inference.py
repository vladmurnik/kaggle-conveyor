"""Export a portable, model-only inference bundle for a completed run.

The bundle is intentionally independent from kconv: ``submission.py`` loads
plain joblib artifacts, reads a CSV/parquet test file and writes a submission.
Raw train/test tables, OOF predictions and the ``runs/`` tree are not copied.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib

from kconv.atomic import dump_joblib_atomic, write_bytes_atomic, write_text_atomic
from kconv.config import load_config
from kconv.context import read_json, write_json
from kconv.errors import KconvError
from kconv.features import runtime as feature_runtime_module
from kconv.inference_script import SUBMISSION_SCRIPT as _SUBMISSION_SCRIPT
from kconv.state import PipelineState, read_latest_pointer

_REQUIREMENTS = """\
joblib>=1.4
lightgbm>=4.3
numpy>=1.26
pandas>=2.2
pyarrow>=16.0
scikit-learn>=1.5
scipy>=1.13
xgboost>=2.0
catboost>=1.2
"""

_README = """\
kconv inference bundle
======================

This folder contains fitted fold models and a standalone submission script.
It does not depend on the kconv package.

    python submission.py --test path/to/test.csv --output submission.csv

Supported test formats: .csv, .csv.gz, .parquet, .pq.
Install dependencies with:

    pip install -r requirements.txt

Each model/fold/seed is stored under models/<name>/folds/. No raw training
data, OOF predictions or original run directory are included.

Open ``inference.ipynb`` for a notebook that installs dependencies, runs the
standalone submission script and previews the result.
"""


def _notebook(metadata: dict) -> str:
    project = metadata.get("project", "kconv")
    run_id = metadata.get("run_id", "run")
    cells = [
        {
            "cell_type": "markdown",
            "id": "intro",
            "metadata": {},
            "source": [
                f"# kconv inference — {project} / {run_id}\n",
                "\n",
                "This bundle is independent of kconv. It contains fitted fold models, ",
                "the fitted feature builders and `submission.py`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "install",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install once in a fresh environment (uncomment and run).\n",
                "# %pip install -r requirements.txt\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "paths",
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "\n",
                "TEST_PATH = Path(\"test.csv\")  # <-- change to your test file\n",
                "OUTPUT_PATH = Path(\"submission.csv\")\n",
                "assert TEST_PATH.exists(), f\"Missing test file: {TEST_PATH.resolve()}\"\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "run",
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import sys\n",
                "\n",
                "subprocess.run(\n",
                "    [sys.executable, \"submission.py\", \"--test\", str(TEST_PATH), \"--output\", str(OUTPUT_PATH)],\n",
                "    check=True,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "preview",
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "\n",
                "submission = pd.read_csv(OUTPUT_PATH)\n",
                "display(submission.head())\n",
                "print(submission.shape)\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"

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


def _output_dir(project_dir: Path, run_id: str, output: str | Path | None) -> Path:
    if output is None:
        return project_dir / "inference" / f"{project_dir.name}-{run_id}"
    return Path(output).expanduser().resolve()


def _resolve_sample(project_dir: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else project_dir / path


def export_inference(
    project_dir: str | Path,
    config_path: str | Path,
    *,
    run_id: str | None = None,
    output: str | Path | None = None,
    force: bool = False,
) -> dict:
    """Write a standalone inference bundle and return its manifest."""
    project_dir = Path(project_dir).resolve()
    config_path = Path(config_path).resolve()
    selected, run_dir = _resolve_run(project_dir, run_id)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise KconvError(
            f"Run state is missing: {state_path}",
            "The run is incomplete; rerun it through train first.",
        )
    state = PipelineState.load(state_path)
    if not state.is_done("train"):
        raise KconvError(
            "The selected run has not completed the train stage.",
            "Run through train/blend/report before exporting inference.",
        )
    schema = read_json(run_dir / "00_ingest" / "schema.json", default=None)
    blend = read_json(run_dir / "03_blend" / "blend.json", default=None)
    if not isinstance(schema, dict) or not isinstance(blend, dict):
        raise KconvError(
            "Inference needs completed ingest and blend artifacts.",
            "Run the pipeline through report before exporting inference.",
        )
    if not config_path.is_file():
        raise KconvError(f"Config not found: {config_path}", "Pass the config used by this run.")

    config = load_config(config_path)
    model_root = run_dir / "02_train"
    model_names = list(blend.get("models", []))
    available: dict[str, Path] = {}
    for name in model_names:
        folds = model_root / str(name) / "folds"
        if not folds.is_dir():
            raise KconvError(
                f"Model '{name}' has no saved fold artifacts.",
                "Rerun train after upgrading kconv so inference models are persisted.",
            )
        available[str(name)] = folds
    if not available:
        raise KconvError("No models are available for inference.", "Train at least one model.")

    output_dir = _output_dir(project_dir, selected, output)
    if output_dir in {project_dir, run_dir}:
        raise KconvError("Refusing to overwrite the project or run directory.", "Choose a separate --output directory.")
    if output_dir.exists():
        if not force:
            raise KconvError(
                f"Inference bundle already exists: {output_dir}",
                "Pass --force to replace it.",
            )
        if not output_dir.is_dir():
            raise KconvError(
                f"Output path is a file, not an inference directory: {output_dir}",
                "Choose a different --output path.",
            )
        if any(output_dir.iterdir()) and not (output_dir / "metadata.json").is_file():
            raise KconvError(
                f"Refusing to replace a directory that is not a kconv inference bundle: {output_dir}",
                "Choose an empty/new --output path, or remove the directory manually.",
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True)
    for name, folds in available.items():
        shutil.copytree(folds, models_dir / name / "folds")

    # Replace kconv objects with portable state/native estimators.  The bundle
    # keeps the same file names for compatibility with existing tooling.
    for name in available:
        fold_root = models_dir / name / "folds"
        for fold_dir in sorted(path for path in fold_root.iterdir() if path.is_dir()):
            feature_path = fold_dir / "features.joblib"
            if not feature_path.is_file():
                raise KconvError(
                    f"Missing feature state in {fold_dir}.",
                    "Rerun train after upgrading kconv so inference artifacts are persisted.",
                )
            builder = joblib.load(feature_path)
            if not hasattr(builder, "to_runtime_state"):
                raise KconvError(
                    f"{feature_path} was written by an older kconv version.",
                    "Rerun train and export inference again.",
                )
            unsupported = builder.runtime_unsupported_blocks()
            if unsupported:
                raise KconvError(
                    "This run uses feature blocks that the standalone bundle cannot reproduce: "
                    + ", ".join(unsupported)
                    + ".",
                    "Disable these feature blocks or use the full project environment for inference.",
                )
            dump_joblib_atomic(builder.to_runtime_state(), feature_path, compress=3)
            for seed_path in sorted(fold_dir.glob("seed_*.joblib")):
                adapter = joblib.load(seed_path)
                model = getattr(adapter, "model", None)
                if model is None:
                    raise KconvError(
                        f"Seed artifact has no fitted estimator: {seed_path}",
                        "Rerun train before exporting inference.",
                    )
                dump_joblib_atomic(
                    {
                        "model": model,
                        "task": getattr(adapter, "task", None),
                        "name": getattr(adapter, "name", None),
                        "family": getattr(adapter, "family", None),
                    },
                    seed_path,
                    compress=3,
                )
    write_bytes_atomic(
        output_dir / "feature_runtime.py",
        Path(feature_runtime_module.__file__).read_bytes(),
    )
    meta_source = run_dir / "03_blend" / "meta_model.joblib"
    if blend.get("method") == "stack":
        if not meta_source.is_file():
            raise KconvError(
                "The stacked run is missing its saved meta-model.",
                "Rerun blend after upgrading kconv, then export again.",
            )
        shutil.copy2(meta_source, models_dir / "meta_model.joblib")

    sample_name = None
    sample_source = _resolve_sample(project_dir, schema.get("sample_submission"))
    sample_columns: list[str] = []
    if sample_source is not None and sample_source.is_file():
        shutil.copy2(sample_source, output_dir / "sample_submission.csv")
        sample_name = "sample_submission.csv"
        try:
            import pandas as pd

            sample_columns = [str(column) for column in pd.read_csv(sample_source, nrows=0).columns]
        except Exception:
            sample_columns = []
    id_col = config.submit.id_col or schema.get("id_col")
    if id_col is None and sample_columns:
        id_col = sample_columns[0]
    target_col = config.submit.target_col or schema.get("target")
    if config.submit.target_col is None and sample_columns:
        non_id = [column for column in sample_columns if column != id_col]
        if non_id:
            target_col = non_id[0]

    metadata = {
        "schema_version": 1,
        "project": config.project,
        "run_id": selected,
        "task": schema.get("task"),
        "metric": schema.get("metric"),
        "target": schema.get("target"),
        "id_col": id_col,
        "target_col": target_col,
        "classes": blend.get("classes"),
        "method": blend.get("method"),
        "models": model_names,
        "weights": blend.get("weights", {}),
        "best_single": blend.get("best_single"),
        "stack": blend.get("stack"),
        "clip": list(config.submit.clip) if config.submit.clip is not None else None,
        "round": config.submit.round,
        "sample_submission": sample_name,
        "feature_runtime": "feature_runtime.py",
    }
    write_json(output_dir / "metadata.json", metadata)
    requirements = _REQUIREMENTS
    if "tabpfn" in model_names:
        requirements += "tabpfn>=9.2,<10\ntorch>=2.5\n"
    write_text_atomic(output_dir / "submission.py", _SUBMISSION_SCRIPT)
    write_text_atomic(output_dir / "requirements.txt", requirements)
    write_text_atomic(output_dir / "README.txt", _README)
    write_text_atomic(output_dir / "inference.ipynb", _notebook(metadata))

    files = sorted(str(path.relative_to(output_dir)).replace("\\", "/") for path in output_dir.rglob("*") if path.is_file())
    return {
        "path": str(output_dir),
        "run_id": selected,
        "models": model_names,
        "files": files,
    }
