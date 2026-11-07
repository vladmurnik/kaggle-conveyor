# kconv

**Локальный dataset-first конвейер для табличных Kaggle-задач.**

Русский | [English](README.md)

[![CI](https://github.com/vladmurnik/kaggle-conveyor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vladmurnik/kaggle-conveyor/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

`kconv` берёт табличный датасет и декларативный recipe, прогоняет честную
кросс-валидацию и выдаёт проверенный `submission.csv` вместе с
воспроизводимой папкой запуска. Всё работает локально: без облачного
сервиса, без склеек из ноутбуков и без скрытого состояния.

Проект намеренно узкий: binary/multiclass classification и regression для
табличных данных. Это не NLP, не computer vision и не online-serving.

> **Статус:** alpha (`0.1.0`). CLI и Python API уже пригодны для работы, но
> до `1.0` возможны небольшие breaking changes.

## Зачем нужен kconv

| | |
| --- | --- |
| **Dataset-first** | Достаточно указать папку, CSV, parquet или zip. Target, id, задача и метрика могут определиться автоматически. |
| **Честная валидация** | Fold-aware преобразования признаков, OOF-метрики, опциональная adversarial-диагностика train/test и явные CV-контракты. |
| **Декларативные recipes** | 45 встроенных recipes: 16 presets и 29 executable pipelines. Чужой Python-код не исполняется. |
| **Много моделей** | LightGBM, XGBoost, CatBoost, HistGradientBoosting, линейные модели, 44 оценки scikit-learn и опциональный TabPFN v2. |
| **Локальная web-студия** | Загрузка данных, очередь запусков, сравнение, отчёты и отправка в Kaggle. Только стандартная библиотека; без Node и FastAPI. |
| **Воспроизводимые артефакты** | Конфиг, фолды, метрики, OOF-предсказания, отчёт, state и offline-пакет с SHA256. |
| **Строгий конфиг** | Pydantic отвергает неизвестные поля и недопустимые комбинации до старта обучения. |
| **Опциональные LLM-признаки** | Проверяемые арифметические планы признаков со строгим whitelist и явным approval. |

## Требования

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) (рекомендуется) или `pip`
- Windows, Linux или macOS
- GPU не обязателен. CUDA опциональна для boosted trees и TabPFN.

## Установка

```bash
git clone https://github.com/vladmurnik/kaggle-conveyor.git
cd kaggle-conveyor
uv sync --extra dev
uv run kconv doctor
```

Без `uv`:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -e ".[dev]"
kconv doctor
```

Опциональные extras:

```bash
uv sync --extra dev --extra tabpfn   # pretrained TabPFN v2
```

## Быстрый старт

### 1. Положите данные в папку

```text
data/
  train.csv
  test.csv
  sample_submission.csv   # опционально
```

`train.csv` обязателен. `test.csv` и `sample_submission.csv` можно не
указывать, если схему удаётся вывести из train. Если автоопределение
неоднозначно, передайте `--target` и `--id`.

### 2. Откройте web-студию

```bash
uv run kconv web --data data
```

По умолчанию сервер слушает `127.0.0.1:8765` и открывает браузер. В Windows
можно использовать лаунчер:

```bat
start.cmd
```

Web UI умеет загружать файлы drag-and-drop, вести последовательную очередь,
сравнивать варианты, показывать Markdown/HTML-отчёты, хранить Kaggle
credentials и проверять submission. Сервер слушает только loopback, пока вы
явно не передадите `--allow-remote`.

### 3. Или запустите из CLI

```bash
uv run kconv run \
  --data data \
  --target target \
  --recipe pipeline/pesnik-clean \
  --strategy standard \
  --name first-run
```

Посмотреть результат:

```bash
uv run kconv status --comp first-run
uv run kconv report --comp first-run
```

Submission лежит в
`competitions/first-run/runs/<run_id>/04_submit/submission.csv`.

### 4. Запуск из конфига

Создайте каркас проекта без обучения:

```bash
uv run kconv run --data data --target target --no-run --name my-project
```

Отредактируйте `competitions/my-project/config.yaml`:

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

Проверьте и запустите:

```bash
uv run kconv config validate --comp my-project
uv run kconv run --comp my-project
```

Относительные пути в конфиге разрешаются от папки проекта
(`competitions/<project>/`), а не от текущего каталога shell.

## Как это работает

```text
ingest -> validate -> hpo -> train -> blend -> submit -> report
```

| Стадия | Что делает |
| --- | --- |
| `ingest` | Читает CSV/parquet/zip, нормализует колонки, применяет детерминированный сэмплинг, пишет `00_ingest/train.parquet` и `test.parquet`. |
| `validate` | Проверяет схему, target, id, задачу, метрику, фолды и опциональный adversarial drift train/test. |
| `hpo` | Запускает nested HPO внутри каждого outer fold, если стратегия это разрешает. |
| `train` | Обучает каждую модель и seed на каждом фолде, строит OOF и кэширует fold-aware признаки. |
| `blend` | Делает `weighted`, `mean` или `stack` с явным fallback на лучшую одиночную модель. |
| `submit` | Проверяет id, колонки, число строк и NaN, затем пишет `submission.csv`. |
| `report` | Пишет Markdown/HTML-отчёт, метрики, снимок конфига и метаданные запуска. |

Готовая папка проекта выглядит так:

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

`latest.json` указывает на последний запуск. `kconv status`, `kconv runs` и
web UI используют один и тот же индекс запусков.

## Recipes и pipelines

**Recipe** — это YAML-описание validation, feature-блоков, моделей, HPO и
ансамбля. Есть два вида:

- **preset** — 16 исходных recipes с legacy feature flags;
- **pipeline** — исполняемый упорядоченный список feature-блоков. Новые
  recipes обычно стоит делать именно pipeline.

В каталоге **45 recipes**: 16 presets и 29 pipelines. Примеры команд:

```bash
uv run kconv recipes
uv run kconv recipes --kind pipeline
uv run kconv recipes show pipeline/pesnik-clean
uv run kconv recipes audit --strict
```

Встроенные группы: `baseline`, `ensemble`, `engineering`, `architecture`.
Пользовательский recipe можно положить в `recipes/` или
`~/.kconv/recipes/`; он переопределяет встроенный recipe с тем же id.

Блоки pipeline декларативны и обучаются внутри training fold. Они не
исполняют произвольный Python. Доступны target/frequency encoding, hash
encoding, group statistics, categorical interactions, missingness,
datetime cyclical, numeric interactions, clustering, anomaly, time-series
lags, row statistics, modulo, feature selection и class-conditional kNN
distances.

Подробный гайд: **[Добавление нового конвейера](docs/adding-conveyors.ru.md)**.

## Модели

`kconv` регистрирует **51 model id**: 6 нативных/boosted-адаптеров,
1 опциональный foundation-адаптер и 44 оценки scikit-learn.

| id | Семейство | Задачи |
| --- | --- | --- |
| `lgbm` | tree | binary, multiclass, regression |
| `xgb` | tree | binary, multiclass, regression |
| `catboost` | tree | binary, multiclass, regression |
| `histgb` | tree | binary, multiclass, regression |
| `logreg` | linear | binary, multiclass |
| `ridge` | linear | regression |
| `tabpfn` | foundation (optional) | binary, multiclass, regression |

Каталог scikit-learn покрывает линейные и регуляризованные модели, kNN,
деревья, random forest, gradient boosting, SVM, naive Bayes, discriminant
analysis, MLP, PLS, RANSAC и dummy-оценки. Список:

```bash
uv run kconv models
uv run kconv models --details
uv run kconv models --json
```

`tabpfn` не попадает в `--models all`, пока не установлен optional extra.
Его safe defaults: 8 000 train rows, 500 features, 10 classes. Подробности —
[docs/tabpfn.md](docs/tabpfn.md).

## Стратегии и runtime-профили

| Стратегия | Фолды | Seeds | HPO |
| --- | ---: | ---: | --- |
| `fast` | 3 | 1 | выключен |
| `standard` | 5 | 2 | выключен |
| `deep` | 5 | 3 | nested, 8 trials на fold/model, бюджет 600 с |

Runtime-профили `1`–`5` — это пресеты памяти, CPU и GPU, которые при
необходимости переопределяют стратегию:

```bash
uv run kconv profiles
uv run kconv run --data data --target target --profile 3
```

Профили записывают RAM, threads, device и model budget как подсказки. Это не
жёсткое ограничение на уровне ОС.

## Web-студия

```bash
uv run kconv web --data data
uv run kconv web --data data --port 9000 --no-open
```

Возможности:

- загрузка `train`, `test` и `sample_submission`;
- выбор recipe, стратегии, моделей и процента сэмплинга;
- очередь из нескольких вариантов с последовательным выполнением;
- сравнение OOF-метрик и просмотр лучшего запуска;
- Markdown/HTML-отчёты и postmortem при падении;
- локальное хранение Kaggle credentials и проверка submission;
- отправка только после явного подтверждения.

Сервер реализован на стандартной библиотеке Python. По умолчанию он слушает
только loopback; `--allow-remote` снимает эту защиту и допустим только в
доверенной сети.

## Опциональные LLM-планы признаков

`kconv` может попросить локальную или OpenAI-совместимую модель предложить
арифметические признаки, проверить ответ по строгому whitelist и
материализовать только одобренный план. Модель возвращает JSON; код не
исполняется.

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

Провайдеры:

- `mock` — детерминированный offline-ответ для тестов;
- `plan` — готовый JSON-план, который вы проверили сами;
- `local` — loopback OpenAI-совместимый сервер (llama.cpp, Ollama, vLLM);
- `openai` — явный opt-in вызов API.

Target и id не попадают в prompt. План хранится в `data/llm/latest.json` и
восстанавливается только если hash train, target и id совпадают. Approval
при восстановлении сбрасывается. Подробности —
[docs/llm-features.md](docs/llm-features.md).

## Отправка в Kaggle

```bash
uv run kconv kaggle auth
uv run kconv kaggle submit --comp my-project --competition <slug>
```

Без `--execute` команда только проверяет submission и печатает команду
Kaggle CLI. С `--execute` она загружает файл. Credentials читаются из
стандартного конфига Kaggle и никогда не печатаются в `kconv`.

## Воспроизводимость и восстановление

- Конфиг строгий и хешируется; resume требует совпадения сохранённого hash.
- Фолды сохраняются, а ключ feature-cache включает состав fold.
- `kconv status` определяет stale/interrupted запуски по heartbeat.
- `kconv recover --comp <project>` восстанавливает запуск, процесс которого
  завершился аварийно.
- `kconv package --comp <project>` собирает offline zip с submission,
  отчётом, конфигом, метаданными и `SHA256SUMS`.
- `kconv export --comp <project>` пишет standalone bundle для inference.

## Документация

| Документ | Содержание |
| --- | --- |
| [docs/config.md](docs/config.md) | Строгая схема `config.yaml`, валидация, пути к данным |
| [docs/cli.md](docs/cli.md) | CLI-команды и примеры |
| [docs/recipes.md](docs/recipes.md) | Каталог recipes, YAML-схема, справочник блоков |
| [docs/pipelines.md](docs/pipelines.md) | Executable pipelines и семантика блоков |
| [docs/adding-conveyors.ru.md](docs/adding-conveyors.ru.md) | Гайд: добавить новый recipe/pipeline |
| [docs/models.md](docs/models.md) | Каталог моделей и заметки об адаптерах |
| [docs/validation.md](docs/validation.md) | CV, adversarial validation, smoke-покрытие |
| [docs/web.md](docs/web.md) | Локальная web-студия и HTTP API |
| [docs/llm-features.md](docs/llm-features.md) | LLM-планы, privacy, approval |
| [docs/api.md](docs/api.md) | Python API и каталог источников |
| [docs/inference.md](docs/inference.md) | Standalone inference bundle |
| [docs/testing.md](docs/testing.md) | Структура тестов и optional-маркеры |

## Разработка

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests scripts examples
uv run python scripts/generate_experiments_catalog.py --check
uv build
```

CI запускает тесты и lint на Ubuntu и Windows с Python 3.12. По умолчанию
тесты синтетические; тесты с маркером `realdata` включаются вручную и
используют локальный датасет, который не коммитится.

Мы рады вкладу. Пожалуйста, держите recipes декларативными, добавляйте
регрессионный тест на изменение поведения и не коммитьте датасеты,
credentials, папки запусков и личные benchmark-отчёты.

## Лицензия и атрибуция

`kconv` распространяется по [MIT License](LICENSE). Сторонние методики
recipes, лицензия TabPFN и атрибуция UI описаны в [NOTICE.md](NOTICE.md).

Визуальный язык web UI адаптирован из **unslop** Дамира Соколовского,
лицензия CC BY 4.0:
<https://github.com/heydamir/unslop>.

TabPFN v2 опционален и не вендорится. Веса распространяются Prior Labs по
Prior Labs License 1.1 (Apache-2.0 с требованием атрибуции). Подробности —
[docs/tabpfn.md](docs/tabpfn.md) и [NOTICE.md](NOTICE.md).

Kaggle — торговая марка Kaggle Inc. Проект не аффилирован с Kaggle и не
одобрен ею.
