# Справочник функций kconv

Документ описывает **публичный контракт**: CLI-команды, функции `kconv.api`, каталоги
`sources`/`experiments`/`runtime`, плагины и LLM-предложения признаков. Внутренние
стадии (`kconv.stages.*`) не являются стабильным API; их сигнатуры могут меняться.

Импорт:

```python
from kconv.api import KConv
```

Функциональный API остаётся основным; `KConv` — тонкий фасад для приложений.

---

## 1. CLI

| Команда | Что делает |
|---|---|
| `kconv run` | Создаёт проект из датасета и запускает стадии до `--until`. |
| `kconv run --config PATH` | Запускает существующий `config.yaml`. |
| `kconv run --comp SLUG` | Запускает `competitions/<slug>/config.yaml`. |
| `kconv run --profile 1..5` | Выбирает профиль RAM/CPU/GPU и переопределяет `--strategy`. |
| `kconv run --device auto\|cpu\|gpu\|cuda` | Уточняет устройство; профили 1–3 — только CPU; `auto` у 4–5 падает на CPU без GPU. |
| `kconv run --experiment llm-feature-proposals --approve-llm-features --llm-plan-file plan.json` | Материализует проверенный LLM-план в временную папку и запускает конвейер. |
| `kconv run --models all` | Все модели, совместимые с задачей. |
| `kconv run --no-run` | Только создать `config.yaml`. |
| `kconv new` | Интерактивный мастер: данные → target → recipe → strategy → models. |
| `kconv web` | Локальный браузерный UI: загрузка train/test, настройка и запуск через тот же CLI. |
| `kconv recipes` | Список 45 recipes: 16 presets + 29 executable pipelines. |
| `kconv recipes --kind preset\|pipeline` | Фильтр по виду recipe. |
| `kconv recipes show ID` | Kind, lineage, blocks, defaults, источник, лицензия и attribution. |
| `kconv recipes audit [--strict] [--json]` | Нормализованные подписи и дубли; strict запрещает одинаковые executable pipelines. |
| `kconv benchmark list [--json]` | Четыре фиксированных датасета из `benchmarks/datasets.yaml`. |
| `kconv benchmark run --dataset ID [--recipes all\|a,b]` | Прогон recipes через штатный pipeline и JSON/Markdown scorecard. |
| `kconv models` | Список 51 model id (50 базовых + optional `tabpfn`). |
| `kconv models --details` | Семейство, задачи, native/sklearn/tabpfn и путь оценщика. |
| `kconv metrics` | Список поддерживаемых метрик. |
| `kconv sources` | 211 скачанных источников (по умолчанию). |
| `kconv sources --all` | Все 1173 проиндексированных источника. |
| `kconv sources --executable` | 14 источников с оригинальным recipe kconv. |
| `kconv sources --pipeline` | 20 источников со stage coverage. |
| `kconv sources --category`, `--license-class`, `--query`, `--limit` | Фильтры каталога. |
| `kconv sources show ID` | Лицензия, recipe, стадии, причина и локальный путь. |
| `kconv experiments` | 25 курируемых экспериментов из generated JSON-каталога. |
| `kconv experiments --status idea` | Только идеи, ещё не реализованные в движке. |
| `kconv experiments show ID` | Гипотеза, источник, лицензия, риск и cost. |
| `kconv experiments propose --data ... [--provider mock\|plan\|openai ...] [--output plan.json] [--apply augmented.csv]` | Валидирует JSON-план; `mock`/`plan` офлайн, `openai` — явный opt-in HTTP-вызов. |
| `kconv profiles` | Пять runtime-профилей 1–5. |
| `kconv plugins` | Установленные entry points без импорта. |
| `kconv plugins --load` | Явно импортировать/вызвать плагины. |
| `kconv config validate` | Проверить `config.yaml` и показать runtime. |
| `kconv status`, `kconv report` | Статус стадий и Markdown-отчёт последнего run. |
| `kconv package`, `kconv export` | Offline-архив и standalone inference bundle. |
| `kconv kaggle auth` | Проверка Kaggle CLI/credentials без печати секретов. |
| `kconv kaggle submit [--execute]` | Валидация submission; dry-run по умолчанию, загрузка только с `--execute`. |
| `kconv doctor` | Версии зависимостей, recipes, experiments, profiles, источник и auth. |

Примеры:

```powershell
uv run kconv run --data data --target target --recipe pesnik/baseline --profile 1
uv run kconv run --data data --target target --recipe kconv/linear-stack --profile 4 --models lgbm,xgb,histgb
uv run kconv experiments --status implemented
uv run kconv experiments show llm-feature-proposals
uv run kconv experiments propose --data data --target target --provider mock --output plan.json
uv run kconv profiles --json
```

---

## 2. `kconv.api`: модели

### `list_models(*, task=None, family=None, source=None) -> list[ModelInfo]`

Возвращает зарегистрированные адаптеры. Фильтры:

- `task`: `binary`, `multiclass`, `regression`;
- `family`: семейство адаптера (`tree`, `linear`, `foundation`);
- `source`: `native`, `sklearn` или `tabpfn`.

```python
from kconv.api import list_models

classifiers = list_models(task="binary")
print(len(classifiers))  # 26 with tabpfn installed, 25 otherwise
print(classifiers[0].to_dict())
```

### `get_model_info(name: str) -> ModelInfo`

Метаданные одной модели. Неизвестный id → `ModelError`.

```python
from kconv.api import get_model_info

info = get_model_info("random_forest_classifier")
print(info.estimator_path)  # sklearn.ensemble.RandomForestClassifier
```

### `create_model(name: str, *, task=None) -> ModelAdapter`

Создаёт адаптер. Если `task` передан, проверяет совместимость и выставляет
`adapter.task`; несовместимая задача → `ModelError`.

```python
from kconv.api import create_model

adapter = create_model("ridge", task="regression")
adapter.fit(X_train, y_train)
pred = adapter.predict(X_test)
```

`ModelInfo` содержит: `id`, `family`, `tasks`, `source`, `estimator_path`,
`description`; `to_dict()` превращает `tasks` в список.

---

## 3. `kconv.api`: recipes / pipelines

### `list_pipelines(*, task=None, group=None, kind=None) -> list[PipelineInfo]`

Список 45 entries: 16 presets + 29 pipelines. `kind` — `preset` или `pipeline`;
`group` — `baseline`, `ensemble`, `engineering`, `architecture`. Неизвестный
`kind`/`task` → `KconvError`.

### `get_pipeline(recipe_id: str) -> PipelineInfo`

Один entry по id. Ошибка загрузки → `KconvError`.

```python
from kconv.api import get_pipeline

preset = get_pipeline("kconv/linear-stack")
pipeline = get_pipeline("pipeline/knnfeat-class-distance")
print(pipeline.kind)                         # pipeline
print(pipeline.derived_from)                 # knnfeat/knn-features
print(pipeline.pipeline["blocks"])
print(pipeline.resource_hints)
```

`PipelineInfo` содержит `id`, `title`, `group`, `author`, `source`, `license`,
`tasks`, `models`, `best_for`, `attribution`, `defaults`, `kind`,
`derived_from`, `pipeline`, `resource_hints`.

---

## 4. `kconv.api`: runtime-профили 1–5

### `list_runtime_profiles() -> list[RuntimeProfile]`

Пять профилей в порядке роста стоимости.

### `get_runtime_profile(level: int, *, device=None) -> RuntimeProfile`

Профиль по уровню. `device` — `auto`, `cpu`, `gpu`, `cuda`. Профили 1–3
являются CPU-only: запрос `gpu`/`cuda` даёт `KconvError`, а не молчаливый
fallback. Профили 4–5 по умолчанию `auto`: при отсутствии GPU `device_params`
возвращает CPU-потоки, а явный `gpu`/`cuda` даёт `KconvError`.

### `gpu_capabilities() -> GpuCapabilities`

Best-effort проверка установленных LGBM/XGB/CatBoost: `gpu_available`,
`lgbm_gpu`, `xgb_cuda`, `catboost_cuda`, `details`. Кэшируется; тесты
подменяют её, чтобы не зависеть от железа.

### `resolve_device(profile) -> "cpu" | "cuda"`

Разрешает `auto` по результату `gpu_capabilities()`; явный `gpu`/`cuda` без
доступного GPU бросает `KconvError`.

### `runtime_models(level: int, task: str, *, device=None) -> list[str]`

Модели, которые профиль выберет автоматически. Это только предложение; явный
`models` всегда имеет приоритет.

```python
from kconv.api import get_runtime_profile, runtime_models

profile = get_runtime_profile(4)
print(profile.name, profile.ram_gb, profile.device)  # gpu 32 auto
print(runtime_models(1, "binary"))                    # ['lgbm', 'histgb']
```

`RuntimeProfile`:

| Поле | Значение |
|---|---|
| `level` | 1–5 |
| `name` | `micro`, `low`, `balanced`, `gpu`, `max` |
| `ram_gb` | минимальная RAM |
| `vram_gb` | минимальная VRAM (0 для CPU) |
| `cpu_threads` | потоков CPU; `0` = auto/all |
| `strategy` | `fast`, `standard`, `deep` |
| `model_limit` | максимум авто-моделей |
| `seeds` | число seed-повторов |
| `hpo` | включён ли HPO |
| `device` | `cpu` или `cuda` |
| `description` | человекочитаемое пояснение |

Профили — грубые пресеты, а не гарантия, что каждая модель умеет GPU. Реальный
GPU-путь есть у LightGBM, XGBoost и CatBoost; scikit-learn-адаптеры остаются на
CPU.

---

## 5. `kconv.api`: experiments

### `list_experiments(*, status=None, task=None) -> list[ExperimentInfo]`

Курируемый каталог необычных идей. Статусы:

- `idea` — описана, но не реализована;
- `prototype` — есть экспериментальный код;
- `implemented` — поведение уже есть в движке;
- `validated` — реализовано и подтверждено тестом/экспериментом.

### `get_experiment(experiment_id: str) -> ExperimentInfo`

Один эксперимент; неизвестный id → `KconvError`.

```python
from kconv.api import list_experiments, get_experiment

ideas = list_experiments(status="idea")
print([item.id for item in ideas])
print(get_experiment("llm-feature-proposals").risk)
```

`ExperimentInfo` содержит `id`, `title`, `category` (`experiment`), `status`,
`hypothesis`, `source`, `license`, `task_types`, `stages`, `expected_gain`,
`cost`, `risk`, `recipe_id`, `notes`.

### Generated catalog

```python
from kconv.experiments import catalog_payload, load_catalog, validate_catalog

payload = catalog_payload()          # source of truth: experiments.py
items = load_catalog()               # packaged src/kconv/assets/experiments.json
validate_catalog(payload)            # strict schema, unique ids
```

`load_catalog(path)` читает явный JSON-файл; без аргумента — packaged asset.
`catalog_payload()` детерминирован (без timestamp), поэтому
`uv run python scripts/generate_experiments_catalog.py --check` и тест
синхронизации ловят расхождение. Файл входит в wheel.

### `kconv.llm_provider`

- `OpenAICompatibleProvider(api_key=..., model=..., base_url=..., timeout=...,
  max_tokens=..., temperature=..., seed=..., json_mode=True)` — stdlib HTTP
  клиент; `complete(prompt)` возвращает `LLMResponse` с `TokenUsage`.
- `TokenUsage(prompt_tokens, completion_tokens, total_tokens)` — недостающие
  значения остаются `None`; `available` показывает, вернул ли их провайдер.
- `resolve_api_key(env_name, *, base_url=..., allow_local_no_key=False)` — ключ
  только из окружения, никогда из файла конфигурации; `OPENAI_API_KEY` fallback
  разрешён только для официального OpenAI endpoint; loopback-сервер может
  явно работать без ключа. Секрет не попадает в `repr`/лог.
- `build_feature_log(...)` / `write_feature_log(run_dir, payload)` — privacy-safe
  `05_report/llm_features.json` (prompt hash, usage, proposals, approval).

---

## 6. `kconv.api`: проекты и запуск

### `build_project(data, *, target=None, recipe="pesnik/baseline", strategy="fast", models=None, name=None, test=None, sample_submission=None, task="auto", metric="auto", id_col=None, ignore=None, profile=None, device=None, cpu_threads=None, ram_gb=None, vram_gb=None, model_params=None) -> tuple[Path, Path, Config]`

Создаёт `competitions/<slug>/config.yaml` и возвращает
`(project_dir, config_path, Config)`. `profile` переопределяет `strategy` и
автоматически выбирает модели только если `models is None`. `cpu_threads`,
`ram_gb`, `vram_gb` требуют профиля; `model_params` — ручной словарь
`{model_id: {param: value}}`, который применяется после recipe/runtime
defaults.

### `run_project(project_dir, config_path=None, *, until="report", resume=False, force=False) -> RunContext`

Запускает существующий проект. `until` — последняя стадия:
`ingest`, `validate`, `hpo`, `train`, `blend`, `submit`, `report`.

### `run_dataset(data, *, target=None, recipe="pesnik/baseline", strategy="fast", models=None, name=None, test=None, sample_submission=None, task="auto", metric="auto", id_col=None, ignore=None, profile=None, device=None, cpu_threads=None, ram_gb=None, vram_gb=None, model_params=None, until="report", resume=False, force=False) -> RunContext`

`build_project` + `run_project`.

```python
from kconv.api import run_dataset

ctx = run_dataset(
    "data",
    target="target",
    recipe="kconv/linear-stack",
    profile=3,
    until="report",
)
print(ctx.run_dir)
```

`Config` — pydantic-схема. При использовании профиля поле `runtime` содержит
`profile`, `device`, `ram_gb`, `vram_gb`, `cpu_threads`; оно пишется в YAML и
видно в `kconv config validate`.

---

## 7. `kconv.api`: источники и каталог

### `list_sources(*, downloaded=True, category=None, license_class=None, executable=None, has_pipeline=None, query=None, limit=None) -> list[SourceInfo]`

- `downloaded=True` — 211 локально скачанных;
- `downloaded=None` — все 1173;
- `executable=True` — 14 source-backed recipes;
- `has_pipeline=True` — 20 источников со stage coverage.

### `get_source(identifier: str) -> SourceInfo`

Ищет по catalog id, `owner/repo` или локальному имени каталога. Абсолютные пути
в bundled JSON не хранятся.

### `catalog_stats() -> dict[str, Any]`

Агрегаты: `total`, `downloaded`, `not_downloaded`, `executable`,
`recipe_backed_sources`, `pipeline_sources`, `by_license_class`,
`by_downloaded_category`.

`SourceInfo.local_path(toolkit_dir=None)` возвращает `Path` только если источник
скачан, `local_dir` известен и `kaggle-toolkit` найден. Поиск: аргумент →
`KCONV_TOOLKIT_DIR` → `kaggle-toolkit` в cwd и родителях.

`SourceInfo` содержит: `id`, `repo`, `url`, `downloaded`, `category`, `license`,
`license_class`, `files`, `bytes`, `local_dir`, `description`, `stars`,
`language`, `updated`, `recipe_id`, `executable`, `pipeline_stages`,
`partial_stages`, `reason`.

---

## 8. `KConv`: фасад

| Метод | Делегирует в |
|---|---|
| `models(...)` | `list_models` |
| `model(name)` | `get_model_info` |
| `create(name, task=...)` | `create_model` |
| `pipelines(...)` | `list_pipelines` |
| `pipeline(id)` | `get_pipeline` |
| `experiments(...)` | `list_experiments` |
| `experiment(id)` | `get_experiment` |
| `profiles()` | `list_runtime_profiles` |
| `profile(level, device=...)` | `get_runtime_profile` |
| `profile_models(level, task, device=...)` | `runtime_models` |
| `sources(...)` | `list_sources` |
| `source(identifier)` | `get_source` |
| `stats()` | `catalog_stats` |
| `build(...)` | `build_project` |
| `run(...)` | `run_dataset` |
| `run_existing(...)` | `run_project` |
| `plugins()` | `list_plugins` |
| `load_plugins()` | `load_plugins` |
| `register_model(name)` | `register_model` |

```python
from kconv.api import KConv

k = KConv()
print(len(k.models()), len(k.pipelines()), len(k.experiments()), len(k.profiles()))
print(k.experiment("llm-feature-proposals").status)
```

---

## 9. `kconv.sources`

| Функция | Назначение |
|---|---|
| `list_sources(...)` | фильтрация каталога |
| `get_source(identifier)` | одна запись |
| `catalog_stats()` | агрегаты |
| `SourceInfo.to_dict()` | JSON-совместимый dict |
| `SourceInfo.local_path(toolkit_dir=None)` | локальный путь или `None` |
| `_catalog`, `_sources` | внутренние кэши; не использовать в прикладном коде |

---

## 10. `kconv.plugins`

| Функция | Назначение |
|---|---|
| `list_plugins()` | entry points группы `kconv.plugins`, без импорта |
| `load_plugins()` | явно импортировать/вызвать плагины; ошибки изолируются |
| `PluginInfo.to_dict()` | `name`, `value`, `group`, `loaded`, `error` |

```python
from kconv.plugins import list_plugins, load_plugins

print(list_plugins())
print(load_plugins())  # только по явному запросу
```

Плагин регистрирует модель через `kconv.api.register_model`:

```python
from kconv.api import register_model
from my_package.adapter import MyAdapter

register_model("my_adapter")(MyAdapter)
```

---

## 11. `kconv.llm_features`

### `build_prompt(schema, sample_rows, *, max_features=5) -> str`

Собирает промпт: схема, до 5 sample rows, список колонок, JSON-контракт.
Секреты и target нужно вырезать до вызова.

### `parse_plan(raw, *, columns=None, max_features=5, reserved=(), numeric_columns=None) -> FeaturePlan`

Парсит JSON, убирает Markdown fence, проверяет:

- имя — безопасный identifier; небезопасные имена нормализуются в уникальный
  snake_case, `reserved` (target/id) не могут стать именем признака;
- `kind` — `numeric` или `categorical`;
- выражение — арифметика или разрешённая row/group-функция над известными
  колонками; имена с пробелами/пунктуацией можно писать как
  `col("exact column name")` (парсер также нормализует bare-имена);
- если передан `numeric_columns`, категориальные колонки запрещены в
  арифметике, `row_*` и value-аргументе `group_mean/group_std`; в качестве
  group-key они разрешены;
- дубликаты и лишние признаки отбрасываются с warning;
- выражения **не исполняются**.

### `propose_features(schema, sample_rows, llm, *, max_features=5, reserved=(), numeric_columns=None) -> FeaturePlan`

Вызывает `llm(prompt)`; `llm` — любой callable `str -> str`. Сетевой вызов
делает вызывающий код, а не kconv.

```python
from kconv.llm_features import propose_features

def fake_llm(prompt: str) -> str:
    return '{"features":[{"name":"sum_ab","kind":"numeric","expression":"a + b","rationale":"signal"}]}'

plan = propose_features({"columns": ["a", "b"]}, [{"a": 1, "b": 2}], fake_llm)
print(plan.proposals)
```

`FeaturePlan` содержит `proposals` и `warnings`; `to_dict()` даёт JSON.
`FeatureProposal` содержит `name`, `kind`, `expression`, `rationale`.

LLM-признаки — эксперимент, а не автоматическое улучшение. Материализация должна
быть fit-in-fold, с проверкой leakage, стоимости и воспроизводимости. Подробный
шаг описан в `docs/llm-features.md`.

### `kconv.features.llm_expr`: безопасная материализация

| Функция | Поведение |
|---|---|
| `parse_expression(expression, columns) -> ast.Expression` | Парсит `mode="eval"` и пропускает только `+ - * /`, unary, скобки, числа, известные имена и `col("exact name")`; остальные calls/attributes/index/lambda отклоняются. |
| `evaluate_expression(expression, frame, columns=None) -> pd.Series` | Рекурсивно вычисляет выражение; деление на ноль → `NaN`; без `eval`/`exec`. |
| `materialize_plan(plan, frame, *, columns=None) -> pd.DataFrame` | Копирует таблицу и добавляет валидированные `FeatureProposal`; `columns` — allow-list идентификаторов (target/id вырезаны); проверяет конфликты имён. |

```python
from kconv.features.llm_expr import materialize_plan
from kconv.llm_features import parse_plan

plan = parse_plan('{"features":[{"name":"ab","kind":"numeric","expression":"a / b","rationale":"ratio"}]}',
                  columns=["a", "b"])
augmented = materialize_plan(plan, frame, columns=["a", "b"])  # target/id исключены
```

CLI-путь: `kconv experiments propose ... --output plan.json [--apply out.csv]`,
затем `kconv run --experiment llm-feature-proposals --approve-llm-features
--llm-plan-file plan.json`. Target и id вырезаются до `parse_plan`.
`--provider openai` или `--provider local` — явный opt-in сетевой ветки; токены
пишутся в `05_report/llm_features.json`, prompt text — нет.

### `kconv.features.fold_plan`: fit-in-fold агрегаты

| Функция/класс | Поведение |
|---|---|
| `parse_fold_expression(expression, columns) -> ast.Expression` | Арифметика + `row_mean/row_std/row_min/row_max/row_sum` + `group_mean/group_std/group_count`; колонки можно писать как identifier или `col("exact name")`; attributes, subscripts, keywords, nested group-агрегаты и неизвестные функции отклоняются. |
| `FoldPlanTransformer(plan, *, columns=None)` | Парсит reviewed-план; `fit(train_fold)` учит group-mapping, `transform(valid/test)` переиспользует его; unseen group получает train-prior, `group_count` — 0; не-конечные результаты (`NaN`/`inf`) заменяются на `0.0`. |
| `expression_columns(expression) -> set[str]` | Имена колонок из AST без имён функций и числовых констант; `col("...")` даёт точное имя; используется при staging для проверки колонок в test. |
| `plan_requires_fold_transformer(plan) -> bool` | True, если в плане есть любая разрешённая функция; CLI направляет такой план в `FeatureConfig.fold_plan`, а не в предварительную материализацию. |

```python
from kconv.features.fold_plan import FoldPlanTransformer

transformer = FoldPlanTransformer(
    {"features": [{"name": "grp_mean_num", "expression": "group_mean(num, grp)"}]},
    columns=["num", "grp"],
).fit(train_fold)
valid_matrix = transformer.transform(valid_fold)
```

`FeatureBuilder` применяет `config.features.fold_plan` внутри каждого fold;
`--apply` поддерживает только арифметику. Target-derived статистики пока
запрещены.

### `kconv.features.unusual`: дешёвые прототипы

| Функция | Поведение |
|---|---|
| `GaussianRankTransformer(columns=None, suffix="__gauss_rank")` | sklearn-style `fit` на train fold, `transform` на validation/test; нормальные ранги, NaN → 0.5, без совместного ранжирования train+test. |
| `add_modulo_features(frame, columns=None, *, moduli=(10,100), steps=(5,))` | Добавляет `x % m` и `x // step`; проверяет положительные значения и колонки. |
| `add_row_statistics(frame, columns=None, *, stats=("mean","std","min","max"), prefix="row")` | Row-wise агрегаты; `std` с `ddof=0`; не меняет исходный frame. |

Row statistics и modulo подключены к `FeatureBuilder` через
`features.row_statistics` и `features.modulo_features`; `GaussianRankTransformer`
остаётся opt-in утилитой. Проверки — `tests/test_features.py` и
`tests/test_unusual_features.py`; benchmark-отчёты не коммитятся.

### `kconv.features.knn`: class-conditional kNN distance

| Класс | Поведение |
|---|---|
| `KNNClassDistanceTransformer(columns=None, *, neighbors=3, folds=5, metric="euclidean", max_reference_rows=50000, random_state=42)` | Строит `knn_c<idx>_k<k>` — кумулятивные расстояния до k ближайших соседей каждого класса. `fit`/`transform` используют только train-fold reference; `fit_transform_oof` делает inner OOF для train-строк. Только классификация; numeric-колонки. |

Метод адаптирован из `upura/knnFeat` (MIT, Copyright (c) 2018 Shotaro
Ishihara); код переписан, исходники не вендорятся. Включается через
`features.knn_features`; recipe — `knnfeat/knn-features`.

---

## 12. `kconv.kaggle`

| Функция | Назначение |
|---|---|
| `check_auth(*, environ=None, config_dir=None, token_path=None, which=None, version_runner=None) -> AuthStatus` | Безопасно проверяет CLI и env/`kaggle.json`/`access_token`/OAuth credentials; секреты не возвращает. |
| `save_credentials(username="", key="", *, token="", config_dir=None, token_path=None, environ=None) -> Path` | Атомарно пишет `kaggle.json` или `access_token` с правами `0600`; секрет не возвращается. |
| `validate_submission(path, *, expected_columns=None) -> (rows, columns)` | CSV/CSV.GZ: непустой, ≥2 колонки, уникальные заголовки, нет полностью пустых колонок. |
| `plan_submission(competition, submission, *, message, expected_columns=None) -> SubmissionPlan` | Валидирует и строит точный argv `kaggle competitions submit`; без сети. |
| `submit_submission(..., execute=False, auth=None, runner=None) -> SubmitResult` | Dry-run по умолчанию; при `execute=True` требует ready-auth и вызывает Kaggle CLI. |
| `SubmissionPlan.to_dict()` / `SubmitResult.to_dict()` | JSON-совместимое представление для CLI `--json`. |

`--execute` — единственный способ загрузки; dry-run не читает credentials и не
запускает subprocess. Ошибки CLI/таймаута возвращаются как `KconvError`.

---

## 13. `scripts/discover_experiments.py`

Read-only сканер локального `kaggle-toolkit`. Не импортирует чужой код.

```powershell
uv run python scripts/discover_experiments.py `
    --toolkit ..\kaggle-toolkit `
    --max-files 200 `
    --output experiments_candidates.json `
    --pretty
```

Ищет ключевые слова (`pseudo`, `adversarial`, `target encoding`, `stack`,
`rank blend`, `LLM`, ...), пишет `experiment`, `repo`, `path`, `line`,
`snippet`. Результат — кандидаты для ручной/LLM-триажки, а не готовые recipes.
Изображения/большие base64-строки пропускаются.

## 14. Ошибки

Публичные функции используют:

- `KconvError` — общая ошибка с `message`, `hint`, `exit_code`;
- `ConfigError` — неверный конфиг;
- `ModelError` — неизвестная/несовместимая модель;
- `DataError` — данные;
- `RecipeError` — recipe/task mismatch.

CLI перехватывает `KconvError` и печатает `Error` + `Hint`.

---

## 15. `kconv.web` (локальный UI)

`kconv web` — оболочка над CLI, а не второй движок. Публичные entry points:

| Функция | Назначение |
|---|---|
| `create_server(*, root=None, data_dir=None, host="127.0.0.1", port=8765, kaggle_token_path=None) -> KconvWebServer` | Создать `ThreadingHTTPServer` для тестов или встраивания; `port=0` выбирает свободный порт; `kaggle_token_path` подменяет путь к access token в тестах. |
| `serve(*, host="127.0.0.1", port=8765, data_dir=None, open_browser=True) -> None` | Запустить сайт и блокироваться до `Ctrl+C`. |
| `WebApp(root, data_dir, *, kaggle_token_path=None)` | Состояние: `dataset_payload`, `config_payload`, `stats_payload`, `start_run`, `run_status`, `submission_path`, `report_path`, `kaggle_status_payload`, `save_kaggle_credentials`, `submit_kaggle`. |
| `RunManager` | Один активный `kconv run` subprocess, лог в памяти, `stop()`. |
| `WebError` | Ожидаемая HTTP-ошибка с `status` и `hint`; handler отдаёт её как JSON. |

HTTP-контракт (все пути локальные, без auth):

```text
GET  /api/config
GET  /api/dataset[?target=&id=&task=&metric=]
GET  /api/stats
GET  /api/projects
GET  /api/run/status?project=NAME
POST /api/upload?kind=train|test|sample&name=FILE   # raw body
POST /api/run                                      # JSON
POST /api/run/stop
GET  /api/kaggle/status?project=NAME
POST /api/kaggle/credentials                       # JSON token or username/key
POST /api/kaggle/submit                            # JSON competition/run_id
GET  /api/submission?project=NAME
GET  /api/report?project=NAME
```

`/api/upload` принимает тело файла (не multipart), расширения — `.csv`,
`.csv.gz`, `.parquet`, `.pq`; лимит 1 GiB. Сервер по умолчанию слушает
loopback. Подробности и примеры — `docs/web.md`.



### LLM-предложения: ограничения

`experiments propose` поддерживает `mock`, `plan`, `local` и `openai`.
Модель возвращает JSON; kconv проверяет его по строгому whitelist и не
исполняет код. `--llm-idea`, `--llm-code-file` и `--llm-repairs` задают
подсказки, контекст и число repair-попыток. В web UI те же поля доступны
как **Идеи**, **Код идей / контекст** и **Repair attempts**. План
применяется только после approval.
