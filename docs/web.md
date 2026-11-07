# Локальный web UI (`kconv web`)

`kconv studio` — визуальная оболочка над тем же `kconv run`, который
используется из CLI. Сайт работает только на вашей машине, не требует
FastAPI/Node и не отправляет датасет в облако. На Kaggle уходит только
`submission.csv` и сообщение — после явного нажатия кнопки submit.

```powershell
cd kaggle-conveyor
uv run kconv web
```

На Windows быстрее всего — двойной клик по `start.cmd` в корне репозитория:
скрипт проверит `uv`, создаст `.venv` при первом запуске, создаст `data/`,
запустит `kconv web --data data` и передаст ему дополнительные аргументы.

```powershell
.\start.cmd
.\start.cmd --port 9000
.\start.cmd --no-open
.\start.cmd --help
```

Если порт занят, скрипт подскажет передать другой `--port`. Для запуска без
`.cmd` используйте `uv run kconv web`.

По умолчанию сервер слушает `127.0.0.1:8765` и автоматически открывает
браузер. Данные читаются из `./data`; проекты создаются в
`./competitions/<project>/`.

```text
http://127.0.0.1:8765/
```

Остановка — `Ctrl+C` в терминале.

## Быстрый сценарий

1. Положите `train.csv` и `test.csv` в `data/` **или** перетащите их в
   соответствующие зоны на странице.
2. При желании добавьте `sample_submission.csv` — тогда kconv проверит
   порядок id и схему колонок.
3. Нажмите «Обновить» (если файлы появились вне сайта). Сайт покажет
   колонки, первые строки, auto-target, задачу, метрику и распределение классов.
4. Проверьте `Target`, `ID`, `Задача`, `Метрика`, `Recipe`. Сайт сам
   переключает `binary` для target с двумя классами и блокирует несовместимую
   задачу; в строке под `Задача` видны число классов и баланс.
5. В карточке `Recipe` видны его defaults: модели, число folds, ensemble и HPO.
   При выборе recipe модели автоматически заменяются на task-compatible и
   доступные из этого recipe; после ручных изменений выбранные вами модели
   имеют приоритет, пока recipe не выберут заново.
6. Модели сгруппированы по источнику и семейству (`kconv native`,
   `scikit-learn`, `TabPFN`), а выбранные показаны отдельными chips.
   `lgbm` + `histgb` — безопасный старт; `tabpfn` — только если установлен
   optional extra и train не больше 8 000 строк. При необходимости раскройте
   «Ресурсы» и «Гиперпараметры моделей»: оба блока опциональны, пустые поля
   означают defaults профиля/recipe.
7. При желании включите карточку **«LLM-признаки»**, выберите провайдера
   (`mock`, `openai` или `local`), сгенерируйте план, проверьте формулы и
   отметьте подтверждение. Без подтверждения запуск блокируется.
8. Оставьте «Полный прогон» для submission или снимите галочку для CV-only
   (прогон до `blend`, чтобы получить OOF и сравнение без submission).
9. Нажмите «Запустить конвейер» для одного запуска **или** соберите варианты
   в блоке «Очередь запусков»: меняйте strategy/profile/recipe/модели и
   добавляйте текущую настройку. Селект «Готовый набор pipelines» выделяет
   сразу группу: быстрый старт, быстрые/сбалансированные/тяжёлые наборы,
   категориальные, временные, кластерные, ансамблевые или все 29 pipelines.
   В селекте «Пакет pipelines» можно вручную отметить несколько executable
   pipelines (Ctrl/Cmd); они получат текущие модели, strategy, profile и
   LLM-план. Количество вариантов не ограничено;
   сервер выполняет их по одному и сам переходит к следующему.
10. После завершения скачайте `submission.csv` и `report.md` из блока
    «Качество». Отчёт показывается как форматированный Markdown (заголовки,
    таблицы, списки, code); сырой HTML из `report.md` не исполняется.
11. Таблица «Варианты проекта» сравнивает runs по каждому датасету: strategy,
    profile, модели, OOF, best single, время и статус. Селектор «Датасет»
    фильтрует таблицу, а при «все датасеты» лучший run подсвечивается в
    каждой группе отдельно; общий междатасетный лидер не выбирается.
12. В карточке **«10 / kaggle»** вставьте API token из Kaggle Settings →
    API → **Generate New Token** (или раскройте legacy-блок и вставьте
    `username`/`key` из старого `kaggle.json`), укажите competition slug и
    нажмите «Сохранить credentials». После этого кнопка «Отправить в Kaggle»
    отправит текущий `04_submit/submission.csv` одним нажатием. Токен
    пишется только локально в `~/.kaggle/access_token`, legacy-ключ — в
    `$KAGGLE_CONFIG_DIR/kaggle.json` или `~/.kaggle/kaggle.json`; в браузер,
    ответы API и логи секрет не возвращается.

## Параметры команды

```powershell
uv run kconv web --help
```

| Опция | По умолчанию | Смысл |
| --- | --- | --- |
| `--data PATH` | `data` | Папка для `train.*`, `test.*`, `sample_submission.*` |
| `--host HOST` | `127.0.0.1` | Адрес сервера. Не-loopback адрес требует `--allow-remote` |
| `--port PORT` | `8765` | Порт; `0` выберет свободный |
| `--no-open` | off | Не открывать браузер автоматически |
| `--allow-remote` | off | Разрешить привязку не к loopback. Небезопасно: любой, кто откроет порт, сможет управлять запусками и Kaggle |

Пример для другого датасета и порта:

```powershell
uv run kconv web --data C:\datasets\house-prices --port 9000
```

Сайт не требует, чтобы `data/` была внутри репозитория. Проекты всё равно
создаются относительно текущей рабочей директории, из которой запущен
`kconv web`.

## Что можно выбрать на сайте

- **Target** — любая колонка train. Если target не выбран, полный запуск
  заблокирован. Под задачей показываются число классов, missing и баланс
  (по первым 100 000 строкам).
- **ID** — колонка идентификатора; `auto` оставляет автоопределение.
- **Задача** — `auto`, `binary`, `multiclass`, `regression`. Для target с
  двумя классами `multiclass` недоступен: сервер отклонит такую комбинацию.
- **Метрика** — `auto` или конкретная метрика из `kconv metrics`.
- **Recipe** — один из 45 entries: 16 presets и 29 executable pipelines.
  Селект сгруппирован по kind (`pipeline` сверху, `preset` ниже). Карточка
  показывает kind, lineage, blocks, модели, folds, ensemble и HPO defaults;
  при выборе recipe модели автоматически заменяются на task-compatible и
  доступные из recipe. Явно выбранные вручную модели имеют приоритет, пока
  recipe не изменят снова.
- **Готовый набор pipelines** — курируемые группы в блоке очереди:
  `starter` (3 простых pipeline для первого прогона), `fast`,
  `balanced`, `quality` (тяжёлые стеки/ансамбли), специализированные
  `categorical`, `time`, `cluster_anomaly`, `ensemble` и `all` (все 29).
  Выбор набора сразу отмечает его pipelines в multi-select; ручная правка
  переводит селект в «свой выбор». Наборы только ссылаются на существующие
  recipe id и не меняют их defaults. API отдаёт их в `/api/config` как
  `pipeline_presets`.
- **Пакет pipelines** — отдельный multi-select в блоке очереди, сгруппированный
  по `baseline`/`engineering`/`ensemble`/`architecture`. Выбранные
  pipelines добавляются как варианты с одинаковыми текущими настройками
  (target/ID/task/metric/strategy/profile/models/until и LLM-план). Это
  удобно для честного сравнения нескольких executable pipelines на одном
  датасете; сервер по-прежнему проверяет каждый вариант атомарно.
- **Стратегия** — `fast`, `standard`, `deep`. Точные различия:

  | Strategy | Folds | Seeds | HPO | Когда использовать |
  | --- | ---: | ---: | --- | --- |
  | `fast` | 3 | 1 | off | Первая проверка данных и быстрый baseline |
  | `standard` | 5 | 2 | off | Обычный прогон: стабильнее OOF, без поиска |
  | `deep` | 5 | 3 | on, 8 trials | Финальный прогон, когда время не критично |

  `deep` включает nested HPO: гиперпараметры выбираются на внутреннем split
  каждого внешнего fold, поэтому OOF остаётся честным. Больше folds/seeds и
  HPO означают больше времени и нагрузки на CPU/GPU.
- **Профиль** — 1–5. Если профиль выбран, он переопределяет стратегию:
  1–2 → `fast`, 3–4 → `standard`, 5 → `deep`; для профилей 1–3 доступен
  только CPU. Явно выбранные модели всё равно имеют приоритет.
- **Модели** — любые task-compatible модели из каталога. Недоступные
  optional-модели показываются как `off`.
- **Ресурсы (опционально)** — раскройте блок «Ресурсы: CPU / GPU / RAM» после
  выбора профиля. Можно уточнить `device` (`auto`/CPU/GPU-CUDA), число CPU
  threads, RAM и VRAM в GB. CPU threads и device применяются к поддерживающим
  моделям (LGBM/XGB/CatBoost; GPU — только профили 4–5 и подходящие сборки).
  RAM/VRAM записываются в конфиг и показываются в интерфейсе, но не ограничивают
  выполнение и не влияют на планирование: это не cgroup и не лимиты драйвера.
- **Гиперпараметры моделей (опционально)** — отдельный JSON-блок под списком
  моделей. Для каждой выбранной модели можно задать объект, например
  `{"n_estimators": 300, "learning_rate": 0.03}`. Пустое поле означает
  defaults recipe/profile; ручные значения применяются последними и имеют
  приоритет. При `deep` HPO может заново подобрать параметры из своего
  search space. Сервер проверяет JSON, имена выбранных моделей и finite-числа.
- **LLM-признаки** — optional-карточка между Recipe и моделями. Она не
  меняет обычный запуск, пока галочка выключена. После включения:
  1. выберите провайдера: `mock` (offline, без сети и ключа), `openai`
     (OpenAI-compatible API) или `local` (llama.cpp/Ollama/vLLM только на
     loopback). Для `local` кнопка **«Найти локальный сервер»** проверяет
     :8080 (llama.cpp), :11434 (Ollama), :8000 (vLLM) и :1234 (LM Studio);
     base URL и модель можно оставить пустыми — сервер найдёт их сам;
  2. при необходимости переключите **JSON mode** и задайте **Max tokens**
     (по умолчанию 1024): официальный OpenAI обычно требует JSON mode,
     некоторым local-моделям нужен `--no-llm-json-mode`; при HTTP 400 от
     старого llama.cpp CLI автоматически повторит запрос без JSON mode;
  3. нажмите «Сгенерировать план» — сервер вызовет тот же
     `kconv experiments propose`, что и CLI. Общий таймаут web-запроса
     динамический: `min(3600, max(180, max_features × (repairs + 1) × 25 + 120))`
     секунд (например, 10 признаков при 2 repair-попытках — 870 секунд), а
     каждый HTTP-запрос к модели по-прежнему получает `--llm-timeout 170`.
     Для более медленной модели используйте CLI с явным `--llm-timeout`;
  4. проверьте таблицу `имя / тип / выражение / зачем` и предупреждения;
  5. отметьте «Я проверил(а) формулы» — только после этого run получает
     `--experiment llm-feature-proposals --approve-llm-features`.

  В промпт уходят только имена/типы разрешённых колонок и до 5 sample rows;
  target и ID вырезаются. Сгенерированный код не исполняется: kconv
  валидирует JSON и whitelist-AST выражения. Колонки с пробелами или
  пунктуацией можно ссылаться как `col("точное имя")`; категориальные
  колонки отклоняются в арифметике и value-аргументах group-функций. Планы
  хранятся на сервере в
  `data/llm/` и привязаны к target/ID и mtime train; при изменении датасета,
  target, ID, модели, провайдера, `max_features`, `max_tokens` или JSON mode
  план нужно генерировать заново. Для `openai` ключ читается сервером из
  `KCONV_LLM_API_KEY` или `OPENAI_API_KEY` и никогда не вводится в браузере.
  Если ключа нет, генерация для `openai` заблокирована и в карточке показана
  причина; задайте переменную окружения и перезапустите `start.cmd`. Для
  `local` разрешены только `127.0.0.1`, `localhost` и `::1`; сервер сам
  добавляет `/v1`, если base URL введён без него. Ошибки провайдера
  различаются: нет ключа, endpoint/модель не найдены, HTTP 401/404/429,
  обрезанный ответ, невалидный JSON или неподходящая схема.
- **Полный прогон / только CV** — без test-файла полный запуск невозможен,
  сайт автоматически оставит `--until blend`: OOF и сравнение будут, а
  submission/report — нет.

`tabpfn` не выбирается автоматически в профилях 1–4. Если extra установлен,
его можно отметить вручную; для 4 GB GPU и 8 000+ строк используйте
`docs/tabpfn.md` и профиль без лишних моделей. Web UI не запускает TabPFN на
train больше 8 000 строк и возвращает подсказку — это защита 4 GB GPU/3.5 GB
VRAM safe cap; CLI остаётся доступен для осознанного override.

### Как запустить локальную LLM

kconv **не запускает** локальный LLM-сервер: llama.cpp, Ollama или vLLM должны
быть уже запущены на вашей машине. UI-режим `local` работает только с
OpenAI-compatible сервером на loopback.

Для llama.cpp:

```powershell
llama-server -m C:\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 8192
```

Для Ollama:

```powershell
ollama serve
ollama pull qwen2.5-coder:7b
```

Терминал с сервером держите открытым. При необходимости проверьте
`http://127.0.0.1:8080/v1/models`. В UI выберите провайдера `local`, при желании
нажмите «Найти локальный сервер», оставьте Base URL и модель пустыми для
автоопределения, сгенерируйте план, проверьте формулы и подтвердите запуск.
Для старых версий llama.cpp, отвергающих `response_format`, запрос
автоматически повторяется без JSON mode.

## Kaggle: токен и submit

Карточка `10 / kaggle` — единственное место, где web UI пишет секрет на диск.
Она показывает безопасный статус (`CLI`, источник credentials, путь к
credential-файлу) и никогда не возвращает значение API token/key.

1. Откройте `kaggle.com/settings` → **API** → **Generate New Token**.
2. Вставьте токен в поле **API token**. Для старого аккаунта можно раскрыть
   блок `username`/`key` и взять значения из `kaggle.json`.
3. Введите competition slug (часть URL после `/competitions/`) и нажмите
   **«Сохранить credentials»**. Токен атомарно пишется в
   `~/.kaggle/access_token`; legacy-ключ — в
   `$KAGGLE_CONFIG_DIR/kaggle.json` или `~/.kaggle/kaggle.json`. На
   Unix-подобных системах выставляются права `0600`.
4. После завершения полного run нажмите **«Отправить в Kaggle»**. Сервер
   повторно валидирует CSV (≥2 колонки, уникальные заголовки, нет пустых
   колонок), проверяет `ready`-статус Kaggle и вызывает
   `kaggle competitions submit`. Это единственный сетевой вызов web UI в
   сторону Kaggle; данные обучения не отправляются.

Если credentials уже заданы через `KAGGLE_API_TOKEN`,
`KAGGLE_USERNAME`/`KAGGLE_KEY` или `KAGGLE_CONFIG_DIR`, переменные окружения
имеют приоритет над сохранённым файлом — UI покажет это предупреждением.
`check_auth` также распознаёт стандартный `~/.kaggle/access_token` и OAuth-кэш
`~/.kaggle/credentials.json`, созданный `kaggle auth login`. Kaggle CLI
входит в базовые зависимости (`uv sync`); при прямом запуске через
`.venv\Scripts\python -m kconv.cli web` сервер также ищет `kaggle.exe` рядом
с интерпретатором.

## API сайта

Сайт — обычный local HTTP-сервер. Эти endpoint'ы полезны для своих скриптов,
дашбордов и будущей статистики:

| Метод | Endpoint | Назначение |
| --- | --- | --- |
| `GET` | `/` | HTML-интерфейс |
| `GET` | `/api/session` | CSRF-токен текущего сервера: `{"token": "..."}`. Нужен для всех POST |
| `GET` | `/api/config` | Модели, recipes, `pipeline_presets`, профили, метрики, доступность extra; при ошибке каталога — `recipe_error`, остальные селекты остаются рабочими |
| `GET` | `/api/dataset` | Файлы, preview, auto-detection, `target_summary`, warnings |
| `GET` | `/api/stats` | Счётчики каталога, источники, dataset, последний запуск |
| `GET` | `/api/projects` | Проекты с `latest.json` |
| `POST` | `/api/upload?kind=train\|test\|sample&name=train.csv` | Загрузка raw body файла |
| `GET` | `/api/llm/local/discover?base_url=...` | Проверить loopback llama.cpp/Ollama/vLLM/LM Studio и вернуть модели; без `base_url` сканирует стандартные порты |
| `POST` | `/api/llm/propose` | Сгенерировать и провалидировать LLM-план; вернуть `plan_id` и features |
| `POST` | `/api/run` | Валидация и запуск `kconv run`; `409`, если слот занят |
| `POST` | `/api/run/batch` | Валидировать и поставить в очередь массив `variants` |
| `GET` | `/api/run/status?project=NAME` | Статус, стадии, лог, очередь, метрики, `report`/`report_html`, submission preview |
| `POST` | `/api/run/stop` | Остановить активный процесс и очистить очередь |
| `GET` | `/api/comparison?project=NAME&dataset=ID` | Runs проекта с группировкой по датасету: `datasets`, `best_by_dataset`, strategy/models/OOF; `dataset` опционален |
| `GET` | `/api/kaggle/status?project=NAME` | Безопасный статус Kaggle CLI/credentials и наличие submission; секретов нет |
| `POST` | `/api/kaggle/credentials` | Сохранить `token` в `access_token` или `username`/`key` в `kaggle.json`; ответ не содержит секрет |
| `POST` | `/api/kaggle/submit` | Проверить submission и отправить его через `kaggle competitions submit` |
| `GET` | `/api/submission?project=NAME&run_id=ID` | Скачать `submission.csv` (последний или указанный run) |
| `GET` | `/api/report?project=NAME&run_id=ID` | Скачать `report.md` (последний или указанный run) |

`/api/upload` принимает не multipart, а тело файла с `Content-Type:
application/octet-stream`. Это упрощает локальный drag-and-drop и не тянет
зависимости. Имя файла передаётся в query, а расширение проверяется по
белому списку `.csv`, `.csv.gz`, `.parquet`, `.pq`; лимит тела — 256 MiB.
Как и остальные POST, загрузка требует заголовок `X-Kconv-Token`.

Все POST-запросы (`/api/upload`, `/api/llm/propose`,
`/api/kaggle/credentials`, `/api/kaggle/submit`, `/api/run`,
`/api/run/batch`, `/api/run/stop`) требуют `X-Kconv-Token`, полученный из
`GET /api/session`. Сервер также проверяет `Host`, `Origin` (если браузер
его прислал) и `Sec-Fetch-Site`: запросы с чужим Host/Origin и
`Sec-Fetch-Site: cross-site` отклоняются с `403`. JSON-эндпоинты требуют
`Content-Type: application/json` (допускается `; charset=...`) и иначе
возвращают `415`. CORS-заголовки сервер не добавляет.

`/api/run` дополнительно сверяет `task` с числом классов target: для двух
классов принимается `binary`, для трёх и более — `multiclass`; несовместимая
пара возвращает `400` с подсказкой. Это не даёт запустить 2-классовую задачу
как `multiclass` и получить неверную семантику ROC AUC.

LLM-план сначала создаётся отдельным запросом:

```python
payload = {
    "target": "Survived",
    "id_col": "PassengerId",
    "provider": "mock",          # mock | openai | local
    "base_url": None,           # local: None -> auto-detect llama.cpp/Ollama/vLLM/LM Studio
    "model": None,              # local: None -> GET /v1/models
    "max_features": 5,
    "max_tokens": 1024,
    "json_mode": True,           # false -> CLI --no-llm-json-mode
}
# POST /api/llm/propose -> {"plan_id": "...", "features": [...], "warnings": [...]}
```

Затем в `/api/run` (или в каждый вариант `/api/run/batch`) добавляются
`"llm_plan_id": "<plan_id>"` и `"llm_approved": true`. Сервер принимает
только план из своей памяти, проверяет target/ID и неизменность train.
Без `llm_approved=true` возвращается `400`; при изменении train/ target/ID —
подсказка сгенерировать план заново. Абсолютные пути к файлам плана через
API не принимаются.

Пример запуска через API из Python:

```python
import json
import urllib.request

base = "http://127.0.0.1:8765"
with urllib.request.urlopen(f"{base}/api/session") as response:
    token = json.loads(response.read())["token"]

payload = {
    "project": "titanic-web",
    "target": "Survived",
    "id_col": "PassengerId",
    "task": "binary",
    "metric": "roc_auc",
    "recipe": "pesnik/baseline",
    "strategy": "fast",
    "profile": None,
    "models": ["lgbm", "histgb"],
    "until": "report",
    "force": True,
}
request = urllib.request.Request(
    f"{base}/api/run",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "X-Kconv-Token": token,
    },
    method="POST",
)
with urllib.request.urlopen(request) as response:
    print(response.status, response.read().decode())
```

Очередь вариантов через API: общие поля можно вынести на верхний уровень,
а в `variants` переопределять strategy/recipe/models/profile и `label`.

```python
payload = {
    "project": "titanic-web",
    "target": "Survived",
    "id_col": "PassengerId",
    "task": "binary",
    "metric": "roc_auc",
    "until": "report",
    "force": True,
    "variants": [
        {"label": "fast lgbm", "strategy": "fast", "recipe": "pesnik/baseline", "models": ["lgbm"]},
        {"label": "standard lgbm+histgb", "strategy": "standard", "recipe": "pesnik/baseline", "models": ["lgbm", "histgb"]},
        {"label": "deep lgbm", "strategy": "deep", "recipe": "pesnik/baseline", "models": ["lgbm"]},
    ],
}
```

`POST /api/run/batch` возвращает `202` и снимок очереди. Варианты проверяются
атомарно: если хотя бы один некорректен (например, `multiclass` на target с
двумя классами), не ставится в очередь ни один. `GET /api/run/status`
содержит `queue`, `queue_pending` и `batch_active`.

## Как это устроено

`kconv.web` использует только стандартную библиотеку Python:

- `ThreadingHTTPServer` + `BaseHTTPRequestHandler` для HTTP;
- `subprocess.Popen` запускает `python -m kconv.cli run ...` в отдельном
  процессе;
- stdout/stderr читаются в фоновом потоке и отдаются браузеру как хвост лога;
- Kaggle submit запускает официальный CLI через `subprocess.run` только после
  явного нажатия кнопки и готового `ready`-статуса;
- `latest.json`, `run.json`, `state.json`, `blend.json` и артефакты
  читаются с диска. Поэтому статус переживает перезагрузку страницы, а
  сервер не дублирует логику стадий. На Windows `write_json` делает короткие
  повторы `os.replace`, если UI в этот момент читает `state.json`.

Одновременно выполняется один `kconv run`; остальные варианты ждут в FIFO
очереди без лимита на количество. Это защищает GPU/RAM и не создаёт
конкурирующие записи в `competitions/`. «Стоп» завершает текущий процесс и
помечает ожидающие варианты как `cancelled`.

## Ограничения и безопасность

- Сервер по умолчанию слушает только loopback (`127.0.0.1`, `localhost`
  или `::1`). Привязка к `0.0.0.0`, `::` или другому адресу возможна только
  с явным `--allow-remote`; CLI печатает предупреждение, потому что API
  управляет обучением и Kaggle-аккаунтом. Аутентификации нет — не
  публикуйте порт в интернет даже с `--allow-remote`.
- Все POST защищены случайным per-server CSRF-токеном
  (`secrets.token_urlsafe(32)`): браузер получает его из `GET /api/session`
  и отправляет в `X-Kconv-Token`. Без токена или с неверным токеном —
  `403`. Токен не попадает в `/api/config` и не пишется в логи.
- Проверяются `Host`, `Origin` (если есть) и `Sec-Fetch-Site` (если есть).
  Разрешены только loopback-имена и фактический порт сервера; чужой Host,
  cross-site Origin и `Sec-Fetch-Site: cross-site` получают `403`. Это
  защищает от CSRF и DNS rebinding. При `--allow-remote` дополнительно
  разрешается фактический адрес привязки/локальный IP соединения.
- JSON-эндпоинты принимают только `Content-Type: application/json`
  (с необязательным `charset`), иначе возвращают `415`. `/api/upload`
  остаётся бинарным и проверяет только токен, Host/Origin и расширение.
- Максимальный размер загрузки — 256 MiB; файлы пишутся атомарно через
  временный файл.
- UI не загружает датасет в Kaggle. Submit выполняется только по явному
  нажатию «Отправить в Kaggle» и только для готового
  `04_submit/submission.csv`; для скриптов остаётся
  `kconv kaggle submit --execute`.
- API token/key, введённый в web UI, не хранится в localStorage, не пишется
  в лог и не возвращается API. Сервер сохраняет его только в локальный
  `access_token` или `kaggle.json`; не открывайте порт web UI в интернет.
- Если после обновления кода селекты Metric/Recipe/Profile пустые, проверьте
  красное предупреждение в карточке «Запуск конвейера» и перезапустите
  `start.cmd`: уже запущенный старый процесс продолжает держать в памяти
  прежний `kconv`. Статические JS/CSS отдаются с `Cache-Control: no-store`,
  поэтому достаточно обновить страницу после перезапуска.
- LLM-вызовы выполняются только по явному нажатию «Сгенерировать план».
  `mock` полностью offline. `openai` отправляет схему и до 5 строк на
  выбранный API; ключ остаётся в окружении сервера. `local` разрешает только
  loopback endpoint, поэтому произвольный удалённый URL через UI не задать.
  Сгенерированные выражения не исполняются как Python; применяются только
  после подтверждения.
- Optional-модели вроде `tabpfn` остаются optional; кнопка запуска покажет
  ошибку, если extra не установлен.
- Статистика в UI — это каталожные счётчики и метрики последнего run.
  История вариантов одного проекта доступна в `/api/comparison` и в таблице
  «Варианты проекта»; она читается с диска и переживает перезапуск сервера.
  Группировка датасета строится по data card (имя train/test, target, ID,
  число строк), а не по recipe/task/моделям, поэтому разные pipeline и
  binary/multiclass-прогоны одного датасета попадают в одну группу.
  Очередь текущей сессии хранится в памяти: после рестарта сервера уже
  завершённые runs остаются в comparison, а невыполненные варианты нужно
  добавить заново.

## Проверка

```powershell
uv run pytest tests/test_web.py tests/test_kaggle_auth.py tests/test_kaggle_submit.py tests/test_report_markdown.py tests/test_models.py -q
uv run ruff check src/kconv/web.py src/kconv/kaggle tests/test_web.py tests/test_kaggle_auth.py
node --check src/kconv/web_static/app.js
```

Тест поднимает настоящий локальный сервер, загружает train/test/sample,
запускает маленький `lgbm`-pipeline через subprocess и проверяет
`submission.csv`, `report.md`, `report_html`, `/api/stats` и `/api/projects`.
Отдельный тест проверяет видимую карточку LLM, `POST /api/llm/propose` с
`mock`, запрет запуска без approval и полный run с планом; sidecar
`05_report/llm_features.json` должен содержать `approved=true` и hash плана.
Отдельный тест имитирует сломанный recipe-каталог и проверяет, что
`/api/config` всё равно отдаёт metrics/profiles и `recipe_error` вместо
пустого UI. Ещё один тест ставит в очередь два варианта (`fast` и
`standard`), ждёт последовательного завершения и проверяет
`/api/comparison` и best run. Тесты Kaggle сохраняют legacy credentials во
временную `KAGGLE_CONFIG_DIR` и API token во временный `access_token`,
проверяют отсутствие секрета в ответе и отправляют submission через
подменённый `subprocess.run`; отдельный тест проверяет отказ submit без
готовых credentials. Реальный upload требует ваших Kaggle credentials и не
выполняется в CI.
`start.cmd` можно проверить без запуска сервера командой `.\start.cmd --help`.



## Failure, recovery и управление run

Web UI показывает postmortem, если у run есть `failure.json` или
`failure_report.md`. Для interrupted run доступны `--resume`, просмотр
failure.json и кнопка Retry для `failed`, `stopped` и `cancelled`.

| Метод | Endpoint | Назначение |
| --- | --- | --- |
| `GET` | `/api/failure?project=NAME&run_id=ID` | failure, report, report_html |
| `GET` | `/api/runs?project=NAME` | список run'ов проекта |
| `GET` | `/api/run/logs?project=NAME&run_id=ID&level=ERROR&q=TEXT&limit=500` | фильтрованный лог |
| `POST` | `/api/run/continue` | `{project, run_id, until, force?}`; resume |
| `POST` | `/api/queue/retry` | `{item_id, project?}`; повтор failed/stopped/cancelled |

POST-запросы требуют `X-Kconv-Token` из `GET /api/session`; сервер также
проверяет `Host`, `Origin` и `Sec-Fetch-Site`. Всё работает на stdlib-only
сервере.

### LLM-предложения в web UI

В карточке **LLM-признаки** доступны:

- **Идеи** — textarea, до 20 строк и 500 символов;
- **Код идей / контекст** — textarea до 12000 символов; файл не исполняется;
- **Repair attempts** — 0..5.

Web UI вызывает `POST /api/llm/propose` с `ideas`, `code` и `repairs`.
Сервер сохраняет `code-<plan_id>.py` как текст, а не исполняет его. Для
`local`/loopback endpoint данные не уходят за пределы машины; `openai`
требует явного выбора провайдера.
