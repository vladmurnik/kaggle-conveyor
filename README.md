# kconv

**A local, dataset-first Kaggle conveyor for tabular data.**

[Русский](README_RU.md) | English

[![CI](https://github.com/vladmurnik/kaggle-conveyor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vladmurnik/kaggle-conveyor/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

`kconv` takes a tabular dataset and a declarative recipe, runs an honest
cross-validation pipeline, and produces a validated `submission.csv` plus a
reproducible run directory. Everything runs on your machine: no hosted
service, no notebook glue, and no hidden state.

The project is intentionally narrow. It is built for tabular Kaggle-style
tasks (binary classification, multiclass classification, and regression), not
for NLP, computer vision, or online serving.

> **Status:** alpha (`0.1.0`). The CLI and Python API are usable, but minor
> breaking changes are still possible before `1.0`.

## Why kconv

| | |
| --- | --- |
| **Dataset-first** | Point it at a folder, CSV, parquet, or zip. Target, id, task, and metric can be detected automatically. |
| **Honest validation** | Fold-aware feature transforms, out-of-fold metrics, optional adversarial train/test diagnostics, and explicit CV contracts. |
| **Declarative recipes** | 65 built-in recipes: 16 presets and 49 executable pipelines. No third-party Python is executed. |
| **Many models** | LightGBM, XGBoost, CatBoost, HistGradientBoosting, linear models, 44 scikit-learn estimators, and optional TabPFN v2. |
| **Local web studio** | Upload data, queue runs, compare results, read reports, and submit to Kaggle. Standard library only; no Node or FastAPI. |
| **Reproducible artifacts** | Config, folds, metrics, OOF predictions, report, state, and an offline package with SHA256 checksums. |
| **Strict config** | Pydantic models reject unknown fields and invalid combinations before training starts. |
| **Optional LLM features** | Reviewed arithmetic feature plans with a strict whitelist and an explicit approval step. |

## Requirements

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- Windows, Linux, or macOS
- No GPU required. CUDA is optional for boosted trees and TabPFN.

## Installation

```bash
git clone https://github.com/vladmurnik/kaggle-conveyor.git
cd kaggle-conveyor
uv sync --extra dev
uv run kconv doctor
```

Without `uv`:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -e ".[dev]"
kconv doctor
```

Optional extras:

```bash
uv sync --extra dev --extra tabpfn   # pretrained TabPFN v2 adapter
```

## Quick start

### 1. Put the data in a folder

```text
data/
  train.csv
  test.csv
  sample_submission.csv   # optional
```

`train.csv` is required. `test.csv` and `sample_submission.csv` are optional
when the schema can be inferred from the training data. Use `--target` and
`--id` when automatic detection is ambiguous.

### 2. Open the web studio

```bash
uv run kconv web --data data
```

The server listens on `127.0.0.1:8765` by default and opens the browser.
On Windows you can also use the launcher:

```bat
start.cmd
```

The web UI supports drag-and-drop upload, a sequential run queue, comparison
of variants, Markdown/HTML reports, Kaggle credentials, and submission
validation. It binds to loopback unless you explicitly pass `--allow-remote`.

### 3. Or run from the CLI

```bash
uv run kconv run \
  --data data \
  --target target \
  --recipe pipeline/pesnik-clean \
  --strategy standard \
  --name first-run
```

Inspect the result:

```bash
uv run kconv status --comp first-run
uv run kconv report --comp first-run
```

The submission is written to
`competitions/first-run/runs/<run_id>/04_submit/submission.csv`.

### 4. Use a config file

Create a project skeleton without training:

```bash
uv run kconv run --data data --target target --no-run --name my-project
```

Edit `competitions/my-project/config.yaml`:

```yaml
project: my-project
seed: 42
recipe: pipeline/pesnik-clean
strategy: standard
task: auto
metric: auto

data:
  train: ../../data/train.csv
  test: ../../data/test.csv
  sample_submission: ../../data/sample_submission.csv
  target: target
  id: id

models:
  - name: lgbm
    seeds: 2
```

Validate and run:

```bash
uv run kconv config validate --comp my-project
uv run kconv run --comp my-project
```

Relative paths in a config are resolved from the project directory
(`competitions/<project>/`), not from the current shell directory.

## How it works

```text
ingest -> validate -> hpo -> train -> blend -> submit -> report
```

| Stage | What it does |
| --- | --- |
| `ingest` | Reads CSV/parquet/zip, normalizes columns, applies deterministic sampling, writes `00_ingest/train.parquet` and `test.parquet`. |
| `validate` | Checks schema, target, id, task, metric, folds, and optional adversarial train/test drift. |
| `hpo` | Runs nested HPO inside each outer fold when the strategy enables it. |
| `train` | Fits every model and seed per fold, builds OOF predictions, and caches fold-aware features. |
| `blend` | Produces `weighted`, `mean`, or `stack` predictions with an explicit fallback to the best single model. |
| `submit` | Validates ids, columns, row count, and NaN values, then writes `submission.csv`. |
| `report` | Writes Markdown/HTML reports, metrics, config snapshot, and run metadata. |

A completed project looks like this:

```text
competitions/my-project/
  config.yaml
  runs/<run_id>/
    00_ingest/
    01_validate/
    02_train/
    03_blend/
    04_submit/submission.csv
    05_report/report.md
    run.json
    state.json
```

`latest.json` points at the most recent run. `kconv status`, `kconv runs`, and
the web UI use the same run index.

## Recipes and pipelines

A **recipe** is a YAML description of validation, feature blocks, models, HPO,
and ensembling. Two kinds exist:

- **preset** — the original 16 recipes, using legacy feature flags.
- **pipeline** — an executable, ordered list of feature blocks. New recipes
  should normally be pipelines.

The catalog contains **65 recipes**: 16 presets and 49 pipelines. Examples:

```bash
uv run kconv recipes
uv run kconv recipes --kind pipeline
uv run kconv recipes show pipeline/pesnik-clean
uv run kconv recipes audit --strict
```

Built-in groups include `baseline`, `ensemble`, `engineering`, and
`architecture`. A user recipe can live in `recipes/` or
`~/.kconv/recipes/` and overrides a built-in recipe with the same id.

Pipeline blocks are declarative and fit inside the training fold. They do not
execute arbitrary Python. Available block families include target/frequency
encoding, hash encoding, group statistics, categorical interactions,
missingness indicators, datetime cyclical features, numeric interactions,
clustering, anomaly scores, time-series lags, row statistics, modulo views,
feature selection, and class-conditional kNN distances.

See the full guide: **[Adding a new conveyor](docs/adding-conveyors.md)**.

## Models

`kconv` registers **51 model ids**: 6 native/boosted adapters, 1 optional
foundation adapter, and 44 scikit-learn estimators.

| id | Family | Tasks |
| --- | --- | --- |
| `lgbm` | tree | binary, multiclass, regression |
| `xgb` | tree | binary, multiclass, regression |
| `catboost` | tree | binary, multiclass, regression |
| `histgb` | tree | binary, multiclass, regression |
| `logreg` | linear | binary, multiclass |
| `ridge` | linear | regression |
| `tabpfn` | foundation (optional) | binary, multiclass, regression |

The scikit-learn catalog covers linear models, regularized models, kNN, trees,
random forests, gradient boosting, SVMs, naive Bayes, discriminant analysis,
MLP, PLS, RANSAC, and dummy estimators. List them with:

```bash
uv run kconv models
uv run kconv models --details
uv run kconv models --json
```

`tabpfn` is never selected by `--models all` unless the optional extra is
installed. Its safe defaults are 8,000 training rows, 500 features, and 10
classes. See [docs/tabpfn.md](docs/tabpfn.md).

## Strategies and runtime profiles

| Strategy | Folds | Seeds | HPO |
| --- | ---: | ---: | --- |
| `fast` | 3 | 1 | off |
| `standard` | 5 | 2 | off |
| `deep` | 5 | 3 | nested, 8 trials per fold/model, 600 s budget |

Runtime profiles `1`–`5` are memory/CPU/GPU presets that override the strategy
where needed:

```bash
uv run kconv profiles
uv run kconv run --data data --target target --profile 3
```

Profiles record RAM, thread, device, and model-budget hints. They are not an
OS-enforced resource limit.

## Web studio

```bash
uv run kconv web --data data
uv run kconv web --data data --port 9000 --no-open
```

Features:

- upload `train`, `test`, and `sample_submission` files;
- choose a recipe, strategy, models, and sampling percentage;
- queue multiple variants and watch them run sequentially;
- compare OOF metrics and inspect the best run;
- read Markdown/HTML reports and failure postmortems;
- store Kaggle credentials locally and validate submissions;
- submit only after an explicit confirmation.

The server is implemented with the Python standard library. It is loopback-only
by default; `--allow-remote` disables that protection and should be used only
on a trusted network.

## Optional LLM feature plans

`kconv` can ask a local or OpenAI-compatible model for arithmetic feature
ideas, validate the answer against a strict whitelist, and materialize the
approved plan. The model returns JSON; it never executes code.

```bash
uv run kconv experiments propose \
  --data data \
  --target target \
  --provider local \
  --max-features 3 \
  --output plan.json

uv run kconv run --comp my-project \
  --experiment llm-feature-proposals \
  --llm-plan-file plan.json \
  --approve-llm-features
```

Providers:

- `mock` — deterministic offline output for tests;
- `plan` — a reviewed JSON plan supplied by the user;
- `local` — a loopback OpenAI-compatible server (llama.cpp, Ollama, vLLM);
- `openai` — an explicit opt-in API call.

Target and id columns are excluded from the prompt. Plans are persisted under
`data/llm/latest.json` and restored only when the training data hash, target,
and id still match. Approval is cleared on restore. See
[docs/llm-features.md](docs/llm-features.md).

## Kaggle submission

```bash
uv run kconv kaggle auth
uv run kconv kaggle submit --comp my-project --competition <slug>
```

Without `--execute`, the command only validates the submission and prints the
Kaggle CLI command. With `--execute`, it uploads. Credentials are read from the
standard Kaggle configuration and are never printed by `kconv`.

## Reproducibility and recovery

- The config is strict and hashed; a resumed run must match the stored hash.
- Folds are persisted, and feature-cache keys include fold membership.
- `kconv status` detects stale/interrupted runs from heartbeat files.
- `kconv recover --comp <project>` repairs a run whose process died.
- `kconv package --comp <project>` creates an offline zip with the submission,
  report, config, run metadata, and `SHA256SUMS`.
- `kconv export --comp <project>` writes a standalone model + feature
  inference bundle.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/config.md](docs/config.md) | Strict `config.yaml` schema, validation, data paths |
| [docs/cli.md](docs/cli.md) | CLI commands and examples |
| [docs/recipes.md](docs/recipes.md) | Recipe catalog, YAML schema, block reference |
| [docs/pipelines.md](docs/pipelines.md) | Executable pipelines and block semantics |
| [docs/adding-conveyors.md](docs/adding-conveyors.md) | Guide: add a new recipe/pipeline |
| [docs/models.md](docs/models.md) | Model catalog and adapter notes |
| [docs/validation.md](docs/validation.md) | CV, adversarial validation, smoke coverage |
| [docs/web.md](docs/web.md) | Local web studio and HTTP API |
| [docs/llm-features.md](docs/llm-features.md) | LLM feature proposals, privacy, approval |
| [docs/api.md](docs/api.md) | Python API and source catalog |
| [docs/inference.md](docs/inference.md) | Standalone inference bundle |
| [docs/testing.md](docs/testing.md) | Test layout and optional markers |

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests scripts examples
uv run python scripts/generate_experiments_catalog.py --check
uv build
```

The CI workflow runs the test suite and lint on Ubuntu and Windows with Python
3.12. Tests are synthetic by default; tests marked `realdata` are opt-in and
use a local dataset that is never committed.

Contributions are welcome. Please keep recipes declarative, add a regression
test for behavior changes, and do not commit datasets, credentials, run
directories, or personal benchmark reports.

## License and attribution

`kconv` is released under the [MIT License](LICENSE). Third-party recipe
methodologies, TabPFN licensing, and UI attribution are documented in
[NOTICE.md](NOTICE.md).

The web UI visual language is adapted from **unslop** by Damir Sokolovsky,
licensed under CC BY 4.0:
<https://github.com/heydamir/unslop>.

TabPFN v2 is optional and is not vendored. Its weights are distributed by
Prior Labs under the Prior Labs License 1.1 (Apache-2.0 with an attribution
requirement). See [docs/tabpfn.md](docs/tabpfn.md) and
[NOTICE.md](NOTICE.md).

Kaggle is a trademark of Kaggle Inc. This project is not affiliated with or
endorsed by Kaggle.
