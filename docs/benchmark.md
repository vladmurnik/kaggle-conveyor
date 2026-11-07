# Benchmark-контур kconv

Этот документ описывает **M0** benchmark-контура: фиксированные датасеты,
повторяемый запуск recipes через обычный `build_config`/`run_pipeline` и
scorecard в JSON/Markdown.

Главный принцип: benchmark не является вторым тренером. Он материализует
датасет во временном workspace, собирает проект штатным `build_config`,
запускает штатный `run_pipeline` до стадии `report` и извлекает метрики из
артефактов run. Поэтому benchmark проверяет тот же код, который использует
`kconv run`.

## 1. Реестр датасетов

Реестр лежит в `benchmarks/datasets.yaml`. Путь можно переопределить:

```powershell
$env:KCONV_BENCHMARK_REGISTRY = "C:\path\to\datasets.yaml"
```

Относительные пути внутри YAML разрешаются от корня репозитория для
встроенного реестра и от папки YAML для пользовательского реестра. Явный
`root` можно передать в Python API `load_benchmark_registry(path, root=...)`.

В M0 зафиксированы четыре датасета:

| id | task | metric | источник и split |
| --- | --- | --- | --- |
| `local-binary` | binary | `roc_auc` | локальные `data/train.csv`, `data/test.csv`, target `satisfaction`, id `id`; стратифицированные 10% train и 10% test, seed 42 |
| `kgmon-regression` | regression | `rmse` | локальный шаблон KGMON, target `Calories`, id `User_ID`; если пути нет — deterministic `sklearn.datasets.load_diabetes` fallback |
| `wine-multiclass` | multiclass | `log_loss` | `sklearn.datasets.load_wine`, target `target`, id `id`; 70/30 stratified split, seed 42 |
| `synthetic-contract` | binary | `roc_auc` | deterministic synthetic matrix только для проверки контрактов; не является evidence качества |

`local-binary` и `kgmon-regression` не копируются в репозиторий. Если
локального файла нет, `local-binary` завершается типизированной `KconvError`,
а `kgmon-regression` использует fallback. `synthetic-contract` доступен всегда.

### Формат записи

```yaml
datasets:
  - id: example
    title: Example dataset
    kind: local            # local | sklearn | synthetic
    task: binary           # binary | multiclass | regression
    metric: roc_auc
    target: target
    id_col: id
    train: data/train.csv  # для kind=local
    test: data/test.csv
    sample_submission: data/sample_submission.csv
    sample_fraction: 0.10  # опционально, (0, 1]
    seed: 42
    source: human-readable source
    license: human-readable license
    notes: optional
```

Для `kind: sklearn` нужен `sklearn_dataset: wine|diabetes`; для `kind: local`
можно указать `fallback_sklearn: diabetes|wine`. Дубликаты `id`, неизвестные
task/metric/kind и некорректные `sample_fraction` отклоняются при загрузке.

## 2. CLI

```powershell
# Список датасетов; JSON ASCII-safe и не читает train/test
uv run kconv benchmark list
uv run kconv benchmark list --json

# Один recipe на синтетическом контракте
uv run kconv benchmark run `
    --dataset synthetic-contract `
    --recipes pesnik/baseline `
    --strategy fast `
    --n-splits 2 `
    --output runs/benchmarks/smoke

# Все recipes на локальном 10% sample
uv run kconv benchmark run --dataset local-binary --recipes all --strategy fast

# JSON для автоматизации; exit code 1, если хотя бы один recipe failed
uv run kconv benchmark run --dataset wine-multiclass --recipes all --json
```

Опции `benchmark run`:

| Опция | Значение |
| --- | --- |
| `--dataset ID` | обязательный id из `benchmark list` |
| `--recipes all\|a,b` | список recipe id; по умолчанию `all` |
| `--strategy fast\|standard\|deep` | обычная стратегия kconv; по умолчанию `fast` |
| `--models a,b` | явный override моделей, как в `kconv run` |
| `--n-splits N` | переопределить число CV folds (N >= 2) |
| `--output PATH` | папка scorecard; по умолчанию `runs/benchmarks/<run_id>` |
| `--work-dir PATH` | сохранить workspace benchmark в указанной папке |
| `--keep-artifacts` | оставить workspace под `<output>/work` |
| `--max-minutes N` | soft budget: новые recipes после исчерпания помечаются `skipped`; текущий run не убивается |
| `--json` | вывести `BenchmarkRun` JSON |

`--json` использует `_emit_json` и `ensure_ascii=True`, поэтому вывод не
зависит от кодовой страницы Windows. Ошибки реестра/датасета/recipe
преобразуются в `KconvError` с подсказкой, без traceback.

## 3. Scorecard

`write_scorecard(run)` создаёт:

- `scorecard.json` — машинно-читаемый `BenchmarkRun`;
- `scorecard.md` — таблица для человека.

Для каждого recipe фиксируются:

`recipe_id`, `recipe_kind`, `dataset_id`, `task`, `metric`, `status`,
`mean`, `std`, `fold_scores`, `n_train`, `n_test`, `duration_sec`,
`peak_rss_mb`, `submission_valid`, `warnings`, `config_hash`, `seed`,
`git_sha`, `source`, `license`, `notes`, `error`.

Важные ограничения M0:

- `mean` — это `blend.oof_score` из run;
- `fold_scores` и `std` в M0 берутся у лучшей **single-model** записи
  `02_train/*/metrics.json`, потому что blend-артефакт пока не хранит
  fold-level OOF. Это честно помечено в документации и будет исправлено в
  M1/M2 при paired fold comparison;
- `peak_rss_mb` на Windows равен `null`: stdlib `resource` недоступен;
- `submission_valid` проверяет наличие `04_submit/submission.csv`, непустоту
  и отсутствие NaN; это smoke-проверка, а не Kaggle-валидация;
- `max_minutes` — мягкий бюджет, а не жёсткий timeout;
- 10% sample и synthetic data — ranking/smoke evidence, не leaderboard.

Scorecard не коммитится. Каталог `runs/` уже в `.gitignore`; исходные
`data/` и `kaggle-toolkit/` не изменяются.

## 4. Python API

```python
from kconv.benchmark import load_benchmark_registry, run_benchmark, write_scorecard

registry = load_benchmark_registry()
run = run_benchmark(
    "synthetic-contract",
    recipes=["pesnik/baseline"],
    strategy="fast",
    n_splits=2,
    output_dir="runs/benchmarks/smoke",
    keep_artifacts=True,
)
print(run.ok, run.results[0].mean)
write_scorecard(run)
```

`run_benchmark` не прерывает весь прогон из-за одного упавшего recipe: результат
получает `status="failed"` и `error`, остальные recipes продолжаются. Если
recipe не объявляет нужный task, он получает `status="skipped"` с причиной.

## 5. Проверка M0

```powershell
uv run kconv benchmark list --json
uv run kconv benchmark run --dataset synthetic-contract `
    --recipes pesnik/baseline --strategy fast --n-splits 2 --json
uv run pytest -q tests/test_benchmark.py
```

Критерии M0: четыре датасета в реестре, ASCII-safe JSON, smoke-scorecard с
metric/folds/runtime/submission_valid, типизированные ошибки для неизвестного
dataset/recipe и зелёные `tests/test_benchmark.py`.
