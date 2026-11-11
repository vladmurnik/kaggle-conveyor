# Публичный API и каталог источников

`kconv.api` — тонкий программируемый слой над теми же стадиями, которые
использует CLI. Через него можно:

- посмотреть все 51 модель и выбрать совместимые с задачей (50 базовых + optional `tabpfn`);
- получить 65 recipes: 16 presets и 49 executable pipelines;
- исследовать 1173 проиндексированных источника `kaggle-toolkit`;
- смотреть 25 экспериментов и 5 runtime-профилей;
- собрать проект и запустить полный конвейер из Python;
- явно подключить внешнюю модель через entry point.

Полный справочник сигнатур и CLI — `docs/functions.md`.

API не импортирует и не запускает код сторонних репозиториев. Каталог
источников — это только метаданные. `executable=True` означает, что для
источника есть **оригинальный** recipe kconv, а не разрешение копировать его
код.

## Импорт

```python
from kconv.api import (
    KConv,
    build_project,
    catalog_stats,
    create_model,
    get_experiment,
    get_model_info,
    get_pipeline,
    get_runtime_profile,
    get_source,
    list_experiments,
    list_models,
    list_pipelines,
    list_runtime_profiles,
    list_sources,
    load_plugins,
    run_dataset,
    run_project,
    runtime_models,
)
```

Можно использовать функции или фасад `KConv`:

```python
from kconv.api import KConv

kconv = KConv()
models = kconv.models(task="regression")
```

## Модели

```python
from kconv.api import create_model, get_model_info, list_models

all_models = list_models()                 # 50
native = list_models(source="native")      # 6
sklearn = list_models(source="sklearn")    # 44
foundation = list_models(source="tabpfn")  # 1 (optional)
regression = list_models(task="regression")  # 30
binary = list_models(task="binary")        # 26
trees = list_models(family="tree")

info = get_model_info("random_forest_classifier")
print(info.id, info.family, info.tasks, info.estimator_path)

adapter = create_model("ridge", task="regression")
adapter.fit(X_train, y_train)
prediction = adapter.predict(X_test)
```

`list_models` принимает фильтры:

| Параметр | Значения | Смысл |
| --- | --- | --- |
| `task` | `binary`, `multiclass`, `regression` | совместимость с задачей |
| `family` | `linear`, `tree`, `foundation` | семейство адаптера |
| `source` | `native`, `sklearn`, `tabpfn` | происхождение адаптера |

`get_model_info` и `create_model` бросают `ModelError`/`KconvError` для
неизвестного id или несовместимой задачи.

## Recipes: presets и pipelines

```python
from kconv.api import get_pipeline, list_pipelines

recipes = list_pipelines()                 # 65
presets = list_pipelines(kind="preset")    # 16
pipelines = list_pipelines(kind="pipeline")  # 49
ensembles = list_pipelines(group="ensemble")  # 19
regression = list_pipelines(task="regression")

pipeline = get_pipeline("pipeline/knnfeat-class-distance")
print(pipeline.kind)                         # "pipeline"
print(pipeline.derived_from)                 # "knnfeat/knn-features"
print(pipeline.pipeline["blocks"])           # declarative block list
print(pipeline.resource_hints)               # ("cpu", "ram", "cpu-heavy", "classification-only")
print(pipeline.models)
print(pipeline.defaults["ensemble"]["method"])
```

`PipelineInfo.defaults` — это сериализованный `RecipeDefaults`: те же значения,
которые recipe передаёт в `config.yaml`. Для `kind="pipeline"` поле
`PipelineInfo.pipeline` содержит нормализованный block graph, а
`resource_hints` — грубые CPU/RAM/classification-only подсказки для UI.
`list_pipelines(kind=...)` фильтрует по виду; неизвестный kind даёт
`KconvError`. Recipe не запускает чужой код.

## Runtime-профили

```python
from kconv.api import get_runtime_profile, list_runtime_profiles, runtime_models

profiles = list_runtime_profiles()          # 5
profile = get_runtime_profile(4)            # GPU preset
profile_models = runtime_models(1, "binary")  # ['lgbm', 'histgb']
```

`RuntimeProfile` описывает RAM/VRAM, CPU threads, strategy, лимит моделей,
seeds, HPO и device. `build_project(..., profile=..., cpu_threads=...,
ram_gb=..., vram_gb=..., device=..., model_params=...)` пишет `RuntimeConfig` и
ручные параметры в `config.yaml`; `model_params` — словарь вида
`{"lgbm": {"n_estimators": 300}}`. CPU threads/device применяются к
поддерживающим адаптерам, RAM/VRAM остаются записанными бюджетами. Профили
1–3 CPU-only: `get_runtime_profile(1, device="cuda")` выбрасывает
`KconvError`.

## Эксперименты

```python
from kconv.api import get_experiment, list_experiments

implemented = list_experiments(status="implemented")
ideas = list_experiments(status="idea", task="regression")
experiment = get_experiment("llm-feature-proposals")
print(experiment.risk)
```

`ExperimentInfo` — отдельная категория `experiment`; только статусы
`implemented`/`validated` соответствуют реальному поведению движка.

## Каталог источников

```python
from kconv.api import catalog_stats, get_source, list_sources

downloaded = list_sources()                # 211 локальных шаблонов
all_sources = list_sources(downloaded=None)  # 1173 проиндексированных
recipe_backed = list_sources(executable=True)  # 14
with_pipeline = list_sources(has_pipeline=True)  # 20

source = get_source("arnabbiswas1__kaggle_pipeline_tps_aug_22")
print(source.repo)
print(source.license, source.license_class)
print(source.recipe_id)
print(source.pipeline_stages)

stats = catalog_stats()
print(stats["downloaded"], stats["total"], stats["executable"])
```

`get_source` принимает:

- catalog id (`owner__repo`);
- GitHub-имя (`owner/repo`);
- имя локальной папки (`owner__repo`), если источник скачан.

Фильтры `list_sources`:

| Параметр | Пример | Смысл |
| --- | --- | --- |
| `downloaded` | `True`, `False`, `None` | `None` — все 1173 |
| `category` | `"pipeline"` | категория `download_manifest.json` |
| `license_class` | `"permissive"` | класс лицензии |
| `executable` | `True` | есть оригинальный recipe kconv |
| `has_pipeline` | `True` | есть stage coverage из каталога источников |
| `query` | `"home-credit"` | поиск по repo/id/description/recipe |
| `limit` | `20` | ограничение выдачи |

Классы лицензий: `permissive` (MIT/Apache/BSD), `copyleft` (GPL),
`unlicensed` (NO LICENSE), `restricted` (например, NC) и `unknown`.
Лицензия не скрывает источник из каталога, но влияет на то, можно ли
переиспользовать код. kconv не вендорит GPL/NO LICENSE код.

### Локальный путь

В bundled JSON нет абсолютных путей. `SourceInfo.local_path()` разрешает путь
только если каталог реально существует:

```python
source = get_source("arnabbiswas1__kaggle_pipeline_tps_aug_22")
path = source.local_path()          # Path или None

# Явно указать корень kaggle-toolkit:
path = source.local_path("path/to/kaggle-toolkit")
```

Порядок поиска: аргумент `toolkit_dir` → переменная окружения
`KCONV_TOOLKIT_DIR` → `kaggle-toolkit` в текущей папке или её родителях.
Скачанные шаблоны лежат в `<root>/templates/<local_dir>`.

## Сборка и запуск

```python
from pathlib import Path

from kconv.api import build_project, run_dataset, run_project

# 1. Создать проект и config.yaml без запуска.
project_dir, config_path, config = build_project(
    "data",
    target="target",
    recipe="pesnik/baseline",
    strategy="fast",
    models="lgbm,xgb,catboost",
    task="binary",
    metric="roc_auc",
    name="titanic-api",
)

# 2. Запустить существующий проект.
ctx = run_project(project_dir, config_path, until="report")

# 3. Или собрать и запустить одной функцией.
ctx = run_dataset(
    "data",
    target="target",
    recipe="kconv/linear-stack",
    strategy="standard",
    models="all",  # все модели, совместимые с задачей
    name="titanic-all",
)
```

`models="all"` разворачивается в список всех task-compatible id (25 для
классификации, 29 для регрессии). Это дороже, чем defaults recipe, но не
требует перечислять имена вручную.

`build_project` возвращает `(project_dir, config_path, Config)`, а
`run_dataset`/`run_project` — `RunContext` существующего движка. Артефакты
появляются в `competitions/<slug>/runs/<run_id>/`.

### Фасад `KConv`

```python
from kconv.api import KConv

kconv = KConv()
kconv.models(task="binary")
kconv.model("lgbm")
kconv.create("lgbm", task="binary")
kconv.pipelines(group="ensemble")
kconv.pipeline("pesnik/baseline")
kconv.sources(has_pipeline=True)
kconv.source("cdeotte/KGMON-Playbook-2026")
kconv.stats()
kconv.build("data", target="target", name="via-facade")
kconv.run_existing("competitions/via-facade")
kconv.plugins()
kconv.load_plugins()
```

## Плагины

Плагин — установленный Python-пакет с entry point в группе
`kconv.plugins`. Загрузка **явная**: импорт `kconv.api`, `list_models()` и
запуск конвейера не импортируют сторонний код.

`pyproject.toml` плагина:

```toml
[project.entry-points."kconv.plugins"]
my-models = "my_kconv_plugin"
```

`my_kconv_plugin.py`:

```python
from kconv.api import register_model
from my_kconv_plugin.adapter import MyAdapter

# entry point может указывать на модуль: импорт выполняет регистрацию
register_model("my_adapter")(MyAdapter)
```

После установки:

```python
from kconv.api import list_plugins, load_plugins

print(list_plugins())       # entry points, без импорта
results = load_plugins()    # импорт/вызов; ошибки изолированы в PluginInfo.error
```

Ошибка одного плагина не ломает остальные и не ломает импорт kconv.
`register_model` ожидает фабрику, совместимую с `ModelAdapter`; адаптер
должен реализовывать `fit`, `predict` и, для классификации,
`predict_proba`/`decision_function`.

## CLI

Те же каталоги доступны из терминала:

```powershell
uv run kconv models --details
uv run kconv models --json

uv run kconv sources
uv run kconv sources --all
uv run kconv sources --pipeline
uv run kconv sources --executable
uv run kconv sources --license-class permissive
uv run kconv sources show arnabbiswas1__kaggle_pipeline_tps_aug_22

uv run kconv plugins
uv run kconv plugins --load

uv run kconv run --data data --target target --models all --no-run
```

## Безопасность и лицензии

- 1173 источника проиндексированы как метаданные; 211 скачаны локально.
- 14 источников имеют оригинальный recipe kconv; это не значит, что их код
  скопирован.
- 20 источников имеют stage coverage из каталога источников; это описание
  методологии, а не автоматический запуск.
- GPL/NO LICENSE источники доступны для чтения и идей; их код не становится
  зависимостью kconv.
- Плагины выполняются только после явного `load_plugins()`.

## Troubleshooting

**`Bundled source catalog is unavailable`** — пакет установлен без
`kconv/assets/catalog.json`. Переустановите kconv или выполните
`uv run python scripts/build_catalog.py`.

**`local_path()` возвращает `None`** — источник не скачан, либо
`kaggle-toolkit` не найден. Укажите `KCONV_TOOLKIT_DIR` или передайте
`toolkit_dir` в `local_path()`.

**`Unknown source`** — проверьте id через `kconv sources --all`; для
нескачанных источников используйте GitHub-имя `owner/repo`.

**`Model '...' does not support task '...'`** — отфильтруйте модели через
`list_models(task=...)` или `kconv models --details`.
