# Adding a new conveyor

In kconv, a **conveyor** is a declarative recipe that selects validation,
feature blocks, models, HPO, and ensembling. A recipe is data, not code: kconv
never imports or executes Python from a recipe. This guide shows how to add a
new pipeline safely.

> A ready-to-copy example lives in
> [`src/kconv/recipes/pipeline__pesnik-clean.yaml`](../src/kconv/recipes/pipeline__pesnik-clean.yaml).
> The full YAML schema is in [recipes.md](recipes.md); block semantics are in
> [pipelines.md](pipelines.md).

## 1. Choose the recipe kind

| Kind | Use when |
| --- | --- |
| `pipeline` | You want an explicit, ordered feature graph. This is the recommended kind for new work. |
| `preset` | You only need legacy feature flags (`features.target_encode`, `features.row_statistics`, ...). Presets exist for compatibility and historical attribution. |

A pipeline has two parts:

- `defaults` — validation, models, model params, HPO, ensemble;
- `pipeline.blocks` — the ordered feature graph.

After `kconv run` builds a project, the blocks are copied into
`config.yaml` as `features.feature_blocks`, and the recipe itself is no longer
read during training. The config hash covers the resolved blocks, so `--resume`
cannot silently change the feature graph.

## 2. Pick a unique id and attribution

Use `pipeline/<author>-<name>` for new executable pipelines. Keep the id stable:
renaming a recipe breaks saved configs and documentation links. If behavior
changes, add a new id instead.

Required metadata:

```yaml
id: pipeline/my-team-clean
kind: pipeline
derived_from: pesnik/baseline       # optional lineage
title: My team clean baseline
author: My Team
source: https://github.com/my-team/my-repo
license: MIT
group: baseline                     # baseline|ensemble|engineering|architecture|general
tasks: [binary, multiclass, regression]
best_for: "A fast, honest baseline with target/frequency encoding and one LightGBM."
attribution: "Methodology adapted from ...; implementation is kconv's fit-in-fold code."
```

Only add `derived_from` when the recipe is genuinely derived from another
recipe. `source` and `license` must point to the methodology you adapted.
Never copy code from a repository without a compatible license; see
[Attribution and licensing](#8-attribution-and-licensing).

## 3. Start from a minimal pipeline

```yaml
id: pipeline/my-team-clean
kind: pipeline
derived_from: pesnik/baseline
title: My team clean baseline
author: My Team
source: https://github.com/my-team/my-repo
license: MIT
group: baseline
tasks: [binary, multiclass, regression]
best_for: "A fast, honest baseline with target/frequency encoding and one LightGBM."

defaults:
  validation:
    strategy: auto
    n_splits: 5
  models: [lgbm]
  hpo:
    enabled: false
    trials: 20
    timeout_sec: 600
  ensemble:
    method: weighted
    fallback: best_single

pipeline:
  blocks:
    - type: base
    - type: target_encoding
      params: {columns: auto, smoothing: 10}
    - type: frequency_encoding
      params: {columns: auto}

attribution: "Adapted from the public starter-kit structure; feature logic is kconv's fit-in-fold implementation."
```

This is intentionally small. Add one feature family at a time and measure the
OOF effect; a longer pipeline is not automatically better.

## 4. Block rules

`pipeline.blocks` must satisfy all of these rules before training starts:

- exactly one `base` block, first and enabled;
- every block type is a singleton in v1; duplicates are rejected;
- unknown types and unknown parameters are rejected;
- `feature_selection` must be the last enabled block;
- `group_statistics` and `fold_plan` are mutually exclusive;
- `knn_class_distance` is classification-only and needs numeric columns;
- `time_series` needs a datetime column (`time_col: auto` or explicit);
- `clustering` must enable `include_distances` or `include_labels`;
- explicit `columns` must exist in the dataset, or be `auto`;
- `hash_encoding.n_features` is an integer in `1..1024`;
- `time_series.lags` are in `1..100`, `windows` in `2..100`, and `stats` is a
  subset of `mean/std/min/max`.

### Block reference

| Block | Key params | What it does |
| --- | --- | --- |
| `base` | `scale`, `impute`, `max_categories`, `auto_encode`, `drop_constant` | Builds the numeric/categorical matrix. Required, first, exactly once. |
| `target_encoding` | `columns`, `smoothing` | Inner-OOF target encoding; fit only on the training fold. |
| `frequency_encoding` | `columns` | Category frequencies computed on the training fold. |
| `hash_encoding` | `columns`, `n_features` | Stable label-free hash buckets for high-cardinality categories. |
| `group_statistics` | `group_col`, `columns`, `stats`, `max_columns` | Fit-in-fold group mean/std/count; compiles to a reviewed `fold_plan`. |
| `categorical_interaction` | `columns`, `max_pairs`, `encoding` | Pairwise categorical combinations; `frequency` is label-free, `target` is inner-OOF. |
| `missingness` | `columns`, `indicators`, `row_count`, `zero_count` | Missing indicators, row-wise missing count, numeric zero count. |
| `datetime_cyclical` | `columns`, `periods` | Label-free sin/cos for month/dayofweek/dayofyear/hour. |
| `numeric_interactions` | `columns`, `max_pairs`, `operations`, `clip_value` | Products, ratios, differences, sums of numeric pairs; zero denominator gives 0. |
| `clustering` | `columns`, `n_clusters`, `max_iter`, `include_distances`, `include_labels`, `max_fit_rows` | Fit-in-fold MiniBatchKMeans distances and/or labels. |
| `anomaly` | `columns`, `n_estimators`, `contamination`, `include_flag` | Fit-in-fold IsolationForest score and optional flag. |
| `time_series` | `time_col`, `group_col`, `columns`, `lags`, `windows`, `stats`, `include_elapsed`, `include_diff`, `max_history_rows` | Past-only lags, trailing statistics, elapsed/diff time. |
| `row_statistics` | `stats` | Row-wise mean/std/min/max/median over numeric columns. |
| `modulo` | `moduli`, `steps` | Modulo/quotient views for identifier-like numeric columns. |
| `fold_plan` | `plan` | An approved fold-aware LLM plan; expressions are validated. |
| `feature_selection` | `method`, `max_features`, `min_variance` | Final selector: variance, mutual information, or model importance. |
| `knn_class_distance` | `columns`, `neighbors`, `folds`, `metric`, `max_reference_rows` | Class-conditional kNN distance; CPU-only, classification-only. |

`columns: auto` is resolved from a sample of train when the project is built,
and the resolved list is written back to `config.yaml`. This keeps the run
reproducible and independent of fit-time heuristics.

All fitted transforms are trained inside the outer training fold. Validation
and test rows never influence the fitted state. The one exception is
`feature_selection(method=model_importance)`, which fits a small ExtraTrees
model on the outer training fold only.

## 5. Where to put the file

Search order for a recipe id:

1. `recipes/<id with / replaced by __>.yaml`
2. `recipes/<last segment>.yaml`
3. `recipes/<id>/recipe.yaml`
4. `~/.kconv/recipes/...`
5. packaged recipes in `src/kconv/recipes/`

User recipes override built-ins with the same id. For a built-in contribution,
add the YAML under `src/kconv/recipes/` and keep the filename
`<author>__<name>.yaml` (for example `pipeline__my-team-clean.yaml`).

## 6. Validate before training

```bash
uv run kconv recipes show pipeline/my-team-clean
uv run kconv recipes audit --strict
uv run kconv recipes --json
```

`recipes audit --strict` normalizes every recipe signature from `defaults`,
`kind`, and `pipeline.blocks`. It fails if two executable pipelines have the
same signature. Historical duplicate presets are reported but are not an error.

Build a config without training:

```bash
uv run kconv run \
  --data data \
  --target target \
  --recipe pipeline/my-team-clean \
  --no-run \
  --name recipe-smoke
```

Inspect `competitions/recipe-smoke/config.yaml`: `features.feature_blocks`
should contain the resolved blocks, and `recipe_kind` should be `pipeline`.

## 7. Smoke-test and add tests

Run a small end-to-end job:

```bash
uv run kconv run \
  --data data \
  --target target \
  --recipe pipeline/my-team-clean \
  --strategy fast \
  --models lgbm \
  --name recipe-smoke
```

Check:

- `05_report/report.md` exists and contains OOF metrics;
- `04_submit/submission.csv` has the expected columns and row count;
- no NaN in the submission;
- the feature block names appear in the run metadata.

Targeted tests:

```bash
uv run pytest tests/test_recipes.py tests/test_recipe_audit.py tests/test_project.py -q
uv run pytest tests/test_pipeline_matrix.py -k "my-team-clean" -q
```

Add at least one assertion to `tests/test_recipes.py`:

```python
pipeline = load_recipe("pipeline/my-team-clean")
assert pipeline.kind == "pipeline"
assert pipeline.derived_from == "pesnik/baseline"
assert [block.type for block in pipeline.pipeline.blocks] == [
    "base",
    "target_encoding",
    "frequency_encoding",
]
```

If the recipe uses a block family that has no matrix coverage yet, add a case
to `tests/test_pipeline_matrix.py` so the block is exercised end to end.

## 8. Attribution and licensing

- Use only MIT, Apache-2.0, BSD, or similarly compatible methodology.
- For a source with **no license** or a GPL license, use ideas and links only;
  do not copy code, comments, or substantial parameter tables.
- Fill `source`, `license`, `author`, and `attribution`.
- Update the recipe tables in [recipes.md](recipes.md) and
  [pipelines.md](pipelines.md), and add the source to [NOTICE.md](../NOTICE.md).
- Do not commit the source repository, datasets, or benchmark outputs.

## 9. Adding a new block type

A new YAML block type is a code change, not only a recipe change. At minimum:

1. Add the type to `_ALLOWED_BLOCK_TYPES` in
   `src/kconv/recipes/__init__.py`.
2. Implement validation and fit/transform behavior in
   `src/kconv/features/blocks.py` and `src/kconv/features/pipeline.py`.
3. Make sure the transform fits only on the outer training fold and handles
   unseen categories/columns at validation and test time.
4. Add unit tests for shape, leakage, unseen values, and invalid params.
5. Add a pipeline recipe that exercises the block.
6. Document the block in [pipelines.md](pipelines.md) and
   [recipes.md](recipes.md).

If the block is not portable to the standalone inference bundle, update
`runtime_unsupported_blocks()` so the limitation is reported explicitly.

## 10. Checklist

- [ ] id is unique and stable;
- [ ] `kind: pipeline` and `pipeline.blocks` is non-empty;
- [ ] exactly one `base` block, first;
- [ ] no duplicate block types;
- [ ] explicit columns exist or use `auto`;
- [ ] `feature_selection` is last;
- [ ] task list matches the models and blocks;
- [ ] `kconv recipes audit --strict` passes;
- [ ] `--no-run` config contains resolved blocks;
- [ ] fast smoke run completes and writes a valid submission;
- [ ] targeted tests pass;
- [ ] docs, `NOTICE.md`, and attribution are updated;
- [ ] no data, credentials, runs, or personal reports are committed.

## Common errors

| Error | Likely cause |
| --- | --- |
| `pipeline.blocks must start with the 'base' block` | `base` is missing or not first. |
| `pipeline.blocks must contain 'base' exactly once` | Duplicate `base`. |
| `pipeline.blocks must contain each singleton block once` | Duplicate block type. |
| `unknown block type` | Typo or a block that is not implemented in this version. |
| `feature_selection must be the last enabled block` | Another block follows the selector. |
| `knn_class_distance` rejected | Regression task or no numeric columns. |
| `columns ... not found` | Explicit column list does not match the dataset. |
| duplicate signature in strict audit | Two pipelines resolve to the same defaults and blocks. |

## When to add a preset instead

Add a preset only when you need to preserve a historical feature-flag
configuration or when the recipe is intentionally not executable as a block
graph. For new methodology, prefer a pipeline: the feature order is explicit,
the audit can compare signatures, and the config records the resolved graph.
