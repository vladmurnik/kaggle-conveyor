# CLI `kconv`

Все команды запускаются из корня репозитория:

```powershell
uv run kconv <command> [options]
```

## `kconv run`

Создаёт проект из данных или продолжает существующий.

```powershell
# Данные + обязательные параметры
uv run kconv run --data data/train.csv --target target

# Явные test/sample submission
uv run kconv run --data data --target target `
    --test data/test.csv --sample-submission data/sample_submission.csv

# Только конфиг, без обучения
uv run kconv run --data data --target target --no-run

# Существующий проект
uv run kconv run --comp titanic
uv run kconv run --config path/to/config.yaml

# Остановиться после стадии
uv run kconv run --comp titanic --until validate
uv run kconv run --comp titanic --until hpo
uv run kconv run --comp titanic --until train

# Продолжить/перезапустить
uv run kconv run --comp titanic --resume
uv run kconv run --comp titanic --resume --force

# Runtime-профиль 1-5 и устройство
uv run kconv run --data data --target target --profile 1
uv run kconv run --data data --target target --profile 4 --device cuda
uv run kconv run --data data --target target --profile 3 --models lgbm,xgb,histgb

# Уточнить ресурсы профиля и вручную переопределить гиперпараметры
uv run kconv run --data data --target target --profile 3 --device cpu `
    --cpu-threads 4 --ram-gb 12 --vram-gb 0 `
    --model-params '{"lgbm":{"n_estimators":300,"learning_rate":0.03}}'

# LLM-эксперимент: сначала план, потом явное одобрение
uv run kconv experiments propose --data data --target target --provider mock --output plan.json
uv run kconv run --data data --target target --experiment llm-feature-proposals `
    --approve-llm-features --llm-plan-file plan.json
```

Основные опции:

| Опция | По умолчанию | Смысл |
| --- | --- | --- |
| `--data PATH` | — | Папка, CSV/parquet или zip |
| `--target NAME` | auto | Целевая колонка |
| `--id NAME` | auto | Колонка идентификатора |
| `--ignore A,B` | — | Исключить колонки из признаков |
| `--test PATH` | auto | Тестовый файл |
| `--sample-submission PATH` | auto | Файл схемы submission |
| `--recipe ID` | `pesnik/baseline` | Методология/defaults |
| `--strategy` | `fast` | `fast`, `standard`, `deep` |
| `--profile` | — | `1..5`; RAM/CPU/GPU preset, переопределяет `--strategy` |
| `--device` | profile | `auto`, `cpu`, `gpu`, `cuda`; профили 1-3 CPU-only |
| `--cpu-threads` | profile | Потоки LGBM/XGB/CatBoost; требует `--profile`; `0` = все ядра |
| `--ram-gb` | profile | RAM-бюджет в GB; требует `--profile`; не жёсткий OS-лимит |
| `--vram-gb` | profile | VRAM-бюджет в GB; требует `--profile`; не жёсткий лимит драйвера |
| `--model-params` | — | JSON-объект `{"lgbm":{"n_estimators":300}}`; ручной override params |
| `--models` | recipe | `lgbm,xgb,catboost,...` или `all` для всех task-compatible |
| `--name SLUG` | имя данных | Имя папки проекта |
| `--task` | `auto` | `binary`, `multiclass`, `regression` |
| `--metric` | auto | `roc_auc`, `log_loss`, `accuracy`, `f1`, `rmse`, `mae`, `r2` |
| `--config PATH` | — | Готовый `config.yaml` |
| `--comp SLUG` | — | `competitions/<slug>/config.yaml` |
| `--until STAGE` | `report` | Остановиться включительно |
| `--resume` | off | Продолжить последний совместимый run |
| `--force` | off | Перезапустить даже `done`-стадии |
| `--no-run` | off | Только создать конфиг |
| `--experiment ID` | — | Выполнить эксперимент до run; сейчас поддержан только `llm-feature-proposals` |
| `--approve-llm-features` | off | Явное подтверждение, что план LLM-признаков проверен человеком |
| `--llm-plan-file PATH` | — | JSON-план из `kconv experiments propose` |

Порядок стадий: `ingest → validate → hpo → train → blend → submit → report`.

## `kconv new`

Интерактивный мастер. Спрашивает путь, target, recipe, стратегию, модели и
slug, создаёт `competitions/<slug>/config.yaml`, затем предлагает запустить
конвейер. Неинтерактивный запуск лучше делать через `run --data ... --no-run`.

## `kconv web`

Поднимает локальный визуальный интерфейс в браузере. Сайт использует те же
данные и тот же CLI-запуск, что и `kconv run`.

```powershell
uv run kconv web
uv run kconv web --data data --port 9000
uv run kconv web --no-open
```

| Опция | По умолчанию | Смысл |
| --- | --- | --- |
| `--data PATH` | `data` | Папка для train/test/sample_submission |
| `--host HOST` | `127.0.0.1` | Адрес сервера; не публикуйте без защиты |
| `--port PORT` | `8765` | Порт; `0` выберет свободный |
| `--no-open` | off | Не открывать браузер |

На странице можно загрузить файлы, проверить preview, выбрать target/id,
задачу, метрику, recipe, стратегию/профиль и модели, запустить полный
pipeline или только CV, смотреть стадии и лог, скачать submission и отчёт.
Endpoint'ы, ограничения и тесты — `docs/web.md`.

## `kconv status` и `kconv report`

```powershell
uv run kconv status --comp titanic
uv run kconv report --comp titanic
```

`status` показывает последний run, статусы стадий, длительность и ошибку.
`report` печатает Markdown-отчёт из `05_report/report.md`.

## `kconv package`

Собирает offline-zip по завершённому run. По умолчанию берётся последний run
из `latest.json`; `--run` позволяет выбрать конкретный.

```powershell
uv run kconv package --comp titanic
uv run kconv package --config competitions/titanic/config.yaml
uv run kconv package --comp titanic --run <run_id> -o dist/titanic.zip
uv run kconv package --comp titanic --force
```

Содержимое архива (плоская структура):

| Файл | Источник |
| --- | --- |
| `submission.csv` | `04_submit/submission.csv` |
| `report.md` | `05_report/report.md` |
| `blend.json` | `03_blend/blend.json` |
| `schema.json`, `data_card.json` | `00_ingest/` |
| `adversarial.json`, `cv_summary.json` | `01_validate/` (если есть) |
| `submission_check.json` | `04_submit/` (если есть) |
| `run.json`, `state.json` | корень run |
| `config.yaml` | конфиг проекта |
| `SHA256SUMS` | контрольные суммы всех файлов выше |

Сырые таблицы, OOF-предсказания и каталог `runs/` намеренно не попадают в
архив. Если zip уже существует, нужен `--force`. Для packaging нужен
завершённый `report`; иначе команда завершится ошибкой с подсказкой.

## `kconv export`

Создаёт standalone inference bundle для завершённого run: fold-модели,
feature builders, `metadata.json`, `requirements.txt` и `submission.py`.

```powershell
uv run kconv export --comp titanic
uv run kconv export --config competitions/titanic/config.yaml -o dist/titanic-inference
uv run kconv export --comp titanic --run <run_id> --force
```

Запуск без установленного `kconv`:

```powershell
python submission.py --test test.csv --output submission.csv
```

Bundle не содержит train/test parquet, OOF и `runs/`; переносится вместе с
`sample_submission.csv`, если он был доступен. Для `ensemble.method: stack`
дополнительно сохраняется `models/meta_model.joblib`. Если run был создан
старой версией kconv без fold-артефактов, перезапустите `train`.

## `kconv recipes`

```powershell
uv run kconv recipes
uv run kconv recipes --kind pipeline
uv run kconv recipes --kind preset
uv run kconv recipes --group baseline
uv run kconv recipes --group ensemble
uv run kconv recipes --group engineering
uv run kconv recipes --group architecture
uv run kconv recipes --json
uv run kconv recipes show cdeotte/tabular
uv run kconv recipes show pipeline/knnfeat-class-distance
uv run kconv recipes audit
uv run kconv recipes audit --strict --json
```

`--kind` фильтрует `preset` (исторические defaults) и `pipeline`
(исполняемый block graph). `show` печатает kind, `derived_from`, blocks,
validation, ensemble, короткую audit-подпись и attribution. `audit --strict`
падает, если два executable pipeline имеют одинаковую нормализованную подпись;
дубли presets остаются историческими вариантами и не блокируют каталог.

Группы: `baseline`, `ensemble`, `engineering`, `architecture`. Пользовательские
recipe без поля `group` попадают в `general`.

## `kconv benchmark`

```powershell
uv run kconv benchmark list
uv run kconv benchmark list --json
uv run kconv benchmark run --dataset synthetic-contract `
    --recipes pesnik/baseline --strategy fast --n-splits 2 --json
uv run kconv benchmark run --dataset local-binary --recipes all --strategy fast
```

`list` показывает фиксированные датасеты из `benchmarks/datasets.yaml` и не
читает train/test. `run` материализует датасет во временном workspace, запускает
каждый recipe через штатный `run_pipeline` до `report` и пишет
`runs/benchmarks/<run_id>/scorecard.{json,md}`. `--recipes all` запускает все
45 recipes; `--max-minutes` — мягкий бюджет (текущий run не убивается).
Scorecard JSON ASCII-safe; при `status=failed` команда возвращает exit code 1,
но остальные recipes всё равно выполняются. Подробности, поля scorecard и
ограничения 10% sample — в `docs/benchmark.md`.

## `kconv models` и `kconv metrics`

```powershell
uv run kconv models
uv run kconv models --details
uv run kconv models --json
uv run kconv metrics
uv run kconv metrics --json
```

`--details` показывает семейство, задачи, источник (`native`/`sklearn`) и
полный путь scikit-learn-оценщика.

## `kconv sources`

Каталог `kaggle-toolkit` доступен без импорта чужих репозиториев:

```powershell
uv run kconv sources
uv run kconv sources --all
uv run kconv sources --pipeline
uv run kconv sources --executable
uv run kconv sources --category pipeline
uv run kconv sources --license-class permissive
uv run kconv sources --query home-credit
uv run kconv sources --limit 20
uv run kconv sources --json
uv run kconv sources show arnabbiswas1__kaggle_pipeline_tps_aug_22
```

`--json` во всех командах выводит ASCII-escaped JSON: это сохраняет
машиночитаемость на Windows-консолях с cp1251/cp866.

По умолчанию показываются 211 скачанных шаблонов; `--all` — все 1173
проиндексированных источника. `--pipeline` фильтрует 20 источников со stage
coverage из каталога источников, `--executable` — 14 источников с оригинальным
recipe kconv. `show` печатает лицензию, причину, стадии и локальный путь,
если `kaggle-toolkit` найден.

## `kconv experiments`

```powershell
uv run kconv experiments
uv run kconv experiments --status idea
uv run kconv experiments --status implemented --task regression
uv run kconv experiments show llm-feature-proposals
uv run kconv experiments --json
```

Каталог отделяет идеи (`idea`/`prototype`) от реально реализованного
(`implemented`/`validated`). Запись содержит гипотезу, источник, лицензию,
стадии, ожидаемый gain, cost и риск. Источник истины — `experiments.py`;
`src/kconv/assets/experiments.json` генерируется скриптом
`scripts/generate_experiments_catalog.py` и проверяется тестом синхронизации
(файл входит в wheel). В каталоге 25 записей: 10 новых идей/прототипов после
триажа необычных паттернов; см. `kconv experiments` и `docs/llm-features.md`.

### `kconv experiments propose`

```powershell
# офлайн-демо: детерминированные mock-предложения
uv run kconv experiments propose --data data --target target --provider mock `
    --output plan.json --apply augmented.csv --json

# проверенный JSON от любого LLM
uv run kconv experiments propose --data data --target target --provider plan `
    --plan-file reviewed_plan.json --output plan.json

# явный opt-in реального OpenAI-compatible провайдера
$env:KCONV_LLM_API_KEY = "..."
uv run kconv experiments propose --data data --target target `
    --provider openai --llm-model gpt-4o-mini --output plan.json

# локальный OpenAI-compatible сервер: llama.cpp :8080 / Ollama :11434 / vLLM :8000
# base URL и модель можно не указывать — kconv найдёт их через GET /v1/models
uv run kconv experiments propose --data data --target target `
    --provider local --llm-max-tokens 1024 --output plan.json

# явный endpoint/модель (например, нестандартный порт llama.cpp)
uv run kconv experiments propose --data data --target target `
    --provider local --llm-base-url http://127.0.0.1:8081/v1 `
    --llm-model my-model --output plan.json
```

Команда только парсит JSON и проверяет выражения по whitelist AST; `--provider
mock` и `--provider plan` не делают сетевых вызовов. `--provider openai` —
явная ветка HTTP-запроса с ключом из `KCONV_LLM_API_KEY` (fallback
`OPENAI_API_KEY` только для официального `https://api.openai.com/v1`; для
стороннего endpoint нужен явный ключ). `--provider local` сам проверяет
llama.cpp :8080, Ollama :11434, vLLM :8000 и LM Studio :1234, добавляет `/v1`
к base URL без пути и берёт модель из `/v1/models`; ключ не требуется только
для loopback и не печатается. Если сервер отвергает `response_format`, CLI
один раз повторяет запрос без JSON mode. `--output` пишет валидированный план и sidecar `<output>.llm.json`
с token usage и prompt hash, `--apply` материализует его в новую таблицу
(исходные данные не меняются), `--json` печатает план в stdout. Target и id
автоматически исключаются из допустимых колонок и из sample rows. `--llm-log
PATH` задаёт другой путь для metadata; `--output`, `--apply` и `--llm-log`
должны быть разными и не совпадать с входными файлами.
План с `row_*`/`group_*` функциями нельзя применять через `--apply`: `kconv run
--llm-plan-file` сохраняет его в `features.fold_plan`, и `FeatureBuilder`
считает group-mapping внутри каждого train fold.


## `kconv profiles`

```powershell
uv run kconv profiles
uv run kconv profiles --json
```

Профили: `1 micro` (4 GB RAM, 1 CPU thread), `2 low` (8 GB), `3 balanced`
(16 GB, standard), `4 gpu` (32 GB RAM + 12 GB VRAM, device `auto`), `5 max`
(64 GB RAM + 24 GB VRAM, deep + HPO + все модели, device `auto`). `auto`
проверяет установленные LGBM/XGB/CatBoost и молча падает на CPU, если GPU не
найден; явный `--device cuda` без GPU завершается ошибкой, а не тихим CPU-run.
GPU-параметры получают только LGBM/XGB/CatBoost и только если сборка библиотеки
их поддерживает; scikit-learn-модели остаются на CPU.

## `kconv plugins`

```powershell
uv run kconv plugins
uv run kconv plugins --load
uv run kconv plugins --json
```

Плагины — entry points группы `kconv.plugins`. Без `--load` команда только
показывает установленные entry points; с `--load` импортирует/вызывает их и
показывает ошибки по каждому плагину. Подробности — `docs/api.md`.

## `kconv kaggle auth`

```powershell
uv run kconv kaggle auth
uv run kconv kaggle auth --json
```

Проверяет наличие Kaggle CLI и credentials, не выполняя сетевых запросов.
Поддерживаются `KAGGLE_USERNAME`/`KAGGLE_KEY`, `KAGGLE_API_TOKEN`,
`~/.kaggle/kaggle.json` (или файл в `KAGGLE_CONFIG_DIR`), стандартный
`~/.kaggle/access_token` и OAuth-кэш `~/.kaggle/credentials.json`. Вывод и JSON
никогда не содержат значения ключей; показываются только источник credentials,
путь к конфигу и предупреждения. `ready` означает, что CLI найден и
credentials обнаружены. Это диагностика, а не загрузка submission. Web UI
(`kconv web`) сохраняет token в `access_token`, а legacy username/key — в
`kaggle.json`, отдельной кнопкой и не выводит секрет в ответах.

## `kconv kaggle submit`

```powershell
# безопасный dry-run: проверить CSV и напечатать команду, но не загружать
uv run kconv kaggle submit --comp titanic --competition titanic

# явная загрузка (нужны Kaggle CLI + credentials)
uv run kconv kaggle submit --comp titanic --competition titanic --execute

# явный файл вместо последнего run
uv run kconv kaggle submit --competition titanic --file submission.csv --json
```

Без `--execute` команда **никогда** не запускает `kaggle` и не читает
credentials: она проверяет CSV (непустой, ≥2 колонки, уникальные заголовки,
нет полностью пустых колонок), считает SHA256 и печатает точный argv. С
`--execute` требуется `ready`-статус Kaggle auth; ошибка CLI возвращается с
кодом и понятным hint. `--comp` также задаёт проект, из которого берётся
последний `04_submit/submission.csv`, если `--file` не указан.


## `kconv doctor`

Проверяет версии kconv, pandas, numpy, scikit-learn, LightGBM, XGBoost,
CatBoost и pyarrow, а также доступность recipes, моделей, каталога источников
(211 скачанных / 1173 проиндексированных) и строку состояния Kaggle auth.

## `kconv config validate`

```powershell
uv run kconv config validate --config competitions/titanic/config.yaml
uv run kconv config validate --comp titanic
```

Схема строгая: неизвестные поля и неверные типы считаются ошибкой. В выводе
есть project, recipe, task, metric, модели, стратегия и состояние HPO.

## Коды выхода

- `0` — успех.
- `1` — общая ожидаемая ошибка (нет данных, не найден run и т.п.).
- `2` — ошибка конфига/данных/recipe.
- `3` — ошибка модели, стадии или submission.


## Управление запусками

`kconv run` пишет heartbeat в `runs/<run_id>/heartbeat.json` и обновляет
`latest.json`. Если процесс завершился аварийно (SIGKILL, OOM, закрытие
терминала), `status`, `recover` и web-интерфейс определяют это по heartbeat
и PID. Такой run помечается как `interrupted`; для него сохраняются
`failure.json` и `failure_report.md`.

### `kconv recover`

```powershell
uv run kconv recover --comp titanic
uv run kconv recover --config competitions/titanic/config.yaml --run-id RUN_ID
uv run kconv recover --comp titanic --resume
```

Без `--run-id` восстанавливается последний run. `--resume` продолжает
прерванный run после пометки `interrupted`; остальные стадии перезапускаются
по state. `--json` печатает машиночитаемый результат.

### `kconv runs`, `compare`, `best`, `tag`

```powershell
uv run kconv runs --comp titanic
uv run kconv runs --comp titanic --status failed --limit 10
uv run kconv compare RUN_A RUN_B --comp titanic
uv run kconv best --comp titanic --metric roc_auc
uv run kconv tag RUN_ID baseline fast --comp titanic
```

`runs` читает `runs/index.json` и при необходимости сверяется с
`runs/<run_id>/run.json`. `compare` показывает diff конфигов и метрик;
`best` выбирает лучший завершённый run по OOF-метрике; `tag` добавляет
поисковые метки.

### `kconv status`

```powershell
uv run kconv status --comp titanic
uv run kconv status --comp titanic --run-id RUN_ID --json
```

Показывает стадию, heartbeat, stale/interrupted-состояние, модели и
артефакты. `--json` удобен для CI.

### Таймауты, повторы и fallback

```yaml
runtime:
  stage_timeout_sec: 3600   # 0 = без ограничения
  model_timeout_sec: 1800   # 0 = без ограничения на fit
  model_retries: 1
  fallback_models: [histgb]
  isolate_models: false
  feature_cache: true
```

`stage_timeout_sec` ограничивает стадию и пишет postmortem.
`model_timeout_sec`/`isolate_models` ограничивают fit модели;
`model_retries` повторяет fit при ошибке; `fallback_models` подставляет
запасную модель, если основная упала. Причины видны в `failed_models`,
`warnings` и `failure_report.md`.

### LLM-предложения признаков

```powershell
uv run kconv experiments propose --data data --target target `
    --provider local --max-features 3 `
    --llm-idea "добавить отношения числовых колонок" `
    --llm-code-file ideas.py `
    --llm-repairs 3 --output plan.json
```

`--provider` принимает `mock`, `plan`, `local` или `openai`. Модель
возвращает JSON; kconv проверяет его по строгому whitelist и не исполняет
код. `--llm-idea` добавляет подсказку, `--llm-code-file` — UTF-8 файл с
идеями, `--llm-repairs` задаёт число попыток исправления (0..5). Sidecar
`<output>.llm.json` хранит privacy-safe метаданные генерации. План
применяется только с `--approve-llm-features`.
