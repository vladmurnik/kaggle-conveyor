# LLM-признаки: отдельный экспериментальный шаг

Этот документ описывает **экспериментальную стадию** `llm-features`. Она не
включена в автоматический recipe и не должна запускаться на приватных данных
без явного согласия. LLM предлагает **только JSON-описания признаков**; код
kconv не исполняет сгенерированные выражения через `eval`/`exec`.

## Зачем отдельный шаг

LLM полезен не как «замена feature engineering», а как генератор гипотез:

- подсказать отношения между числовыми колонками;
- предложить агрегаты для текстовых/категориальных полей;
- найти domain-specific crosses;
- объяснить, какие признаки проверить первыми.

Он не должен:

- видеть target;
- получать полные строки или PII без необходимости;
- генерировать исполняемый Python;
- выбирать признаки по leaderboard;
- менять данные до честной CV-проверки.

## Пайплайн

### Шаг 0. Подготовка схемы и приватности

1. Взять только имена/типы колонок и 3–5 обезличенных sample rows.
2. Удалить target, id, тексты, email, URL и всё, что запрещено политикой.
3. Записать `prompt_version`, модель, temperature, seed, дату и хэш схемы.
4. Ограничить бюджет: `max_features=5`, максимум один вызов на fold/эксперимент.

### Шаг 1. Prompt

Использовать `kconv.llm_features.build_prompt`:

```python
from kconv.llm_features import build_prompt

prompt = build_prompt(
    {"columns": ["amount", "term", "rate"], "target": "REDACTED"},
    [{"amount": 1000, "term": 12, "rate": 0.2}],
    max_features=5,
)
```

Промпт требует только JSON, запрещает target/id/внешние данные и Python-функции.

### Шаг 2. Получить ответ

Есть три режима:

1. **Внешний вызов** остаётся через `propose_features`, если вы сами
   контролируете SDK/провайдера.
2. **Opt-in адаптер kconv** `kconv.llm_provider.OpenAICompatibleProvider`
   делает один HTTP-запрос к OpenAI-compatible `/chat/completions` и
   возвращает текст + token usage. Он никогда не включается по умолчанию:
   нужен явный `--provider openai --llm-model ...` или `--provider local`.
3. **Локальная модель** через `--provider local`: автоматически проверяет
   llama.cpp (`127.0.0.1:8080/v1`), Ollama (`:11434`), vLLM (`:8000`) и
   LM Studio (`:1234`), а модель берёт из `GET /v1/models`. Ключ не требуется
   только для loopback. Это тот же OpenAI-compatible клиент, поэтому token
   accounting и privacy-лог работают одинаково.

```python
from kconv.llm_features import propose_features

def llm(prompt: str) -> str:
    return call_my_provider(prompt, temperature=0.0, max_tokens=800)

plan = propose_features(
    {"columns": ["amount", "term", "rate"]},
    [{"amount": 1000, "term": 12, "rate": 0.2}],
    llm,
)
```

CLI-вариант с реальным провайдером:

```powershell
# Ключ читается только из окружения и никогда не пишется в отчёт.
$env:KCONV_LLM_API_KEY = "..."   # OPENAI_API_KEY — только для официального endpoint
uv run kconv experiments propose --data data --target target `
    --provider openai --llm-model gpt-4o-mini `
    --llm-base-url https://api.openai.com/v1 `
    --output plan.json
```

Поддерживается любой OpenAI-compatible endpoint: `--llm-base-url`,
`--llm-timeout`, `--llm-max-tokens`, `--llm-temperature`, `--llm-seed`,
`--llm-json-mode/--no-llm-json-mode`. `OPENAI_API_KEY` как fallback принимается
**только** для `https://api.openai.com/v1`; для стороннего endpoint задайте
`KCONV_LLM_API_KEY` или `--llm-api-key-env`, чтобы не отправить OpenAI-ключ
третьей стороне. Если ключа нет, команда падает **до** сетевого вызова с
понятной ошибкой; `mock`/`plan` остаются полностью офлайн-режимами. В текущем
окружении credentials отсутствуют, поэтому реальный вызов не выполнялся — это
зафиксировано в отчёте.

Локальный вариант без облака (llama.cpp):

```powershell
# llama-server сам отдаёт OpenAI-compatible /v1; модель можно не указывать в kconv.
llama-server -m C:\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf `
    --host 127.0.0.1 --port 8080 -c 8192

uv run kconv experiments propose --data data --target target `
    --provider local --llm-max-tokens 1024 --output plan.json
```

`--provider local` сам находит llama.cpp :8080, Ollama :11434, vLLM :8000 или
LM Studio :1234; `--llm-base-url` и `--llm-model` нужны только для
нестандартного порта/нескольких запущенных серверов. Можно передать base URL
без `/v1` — клиент добавит его сам. Отсутствие ключа разрешено только для
loopback; данные уходят только на локальный сервер.

Если локальный сервер не поддерживает `response_format`, CLI один раз
повторяет запрос без JSON mode. JSON в ответе может быть обёрнут в Markdown
или пояснение — парсер извлекает первый валидный объект. Если ответ обрезан
(`finish_reason=length`), ошибка прямо просит увеличить `--llm-max-tokens` и
контекст llama.cpp (`-c`); для веб-формы это поле **Max tokens**.

Ollama-вариант:

```powershell
ollama serve
ollama pull qwen2.5-coder:7b
uv run kconv experiments propose --data data --target target `
    --provider local --llm-model qwen2.5-coder:7b --output plan.json
```

### Web UI

В `kconv studio` (`kconv web` / `start.cmd`) есть отдельная карточка
**«LLM-признаки»** между Recipe и моделями. Она выключена по умолчанию и не
влияет на обычный запуск.

Для `local` kconv **не запускает** внешний LLM-сервер: llama.cpp, Ollama или
vLLM должны быть уже запущены на вашей машине.

```powershell
llama-server -m C:\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 8192

ollama serve
ollama pull qwen2.5-coder:7b
```

Терминал сервера держите открытым; при необходимости проверьте
`http://127.0.0.1:8080/v1/models`. Затем выберите провайдера `local`, при
желании нажмите «Найти локальный сервер», оставьте Base URL и модель пустыми
для автоопределения, сгенерируйте план, проверьте формулы и подтвердите запуск.

1. Включите галочку и выберите провайдера: `mock` (offline), `openai` или
   `local`. Для `local` кнопка **«Найти локальный сервер»** проверяет
   llama.cpp :8080, Ollama :11434, vLLM :8000 и LM Studio :1234 и подставляет
   base URL/модель. Поля можно оставить пустыми — сервер сам найдёт endpoint
   и модель.
2. При необходимости измените **JSON mode** и **Max tokens** (по умолчанию
   1024). Официальный OpenAI обычно требует `json_mode=true`; для некоторых
   local-моделей помогает выключить его — тогда CLI получает
   `--no-llm-json-mode`. Старый llama.cpp, отвергающий `response_format`,
   обрабатывается автоматическим повтором без JSON mode.
3. Нажмите «Сгенерировать план». Сайт вызывает тот же
   `kconv experiments propose`, сохраняет план в `data/llm/` и возвращает
   `plan_id`. Общий таймаут веб-запроса рассчитывается по числу признаков и
   repair-попыток: `min(3600, max(180, max_features × (repairs + 1) × 25 + 120))`
   секунд. Для 10 признаков и 2 repair это 870 секунд. Каждый отдельный
   HTTP-запрос к модели получает `--llm-timeout 170`; для более медленной
   модели используйте CLI с явным `--llm-timeout`.
4. Проверьте таблицу предложений и warnings.
5. Отметьте подтверждение. Только после этого run получает
   `--experiment llm-feature-proposals --approve-llm-features` и пути к
   плану/метаданным.

Сервер принимает только `plan_id` из своей памяти, а не путь из браузера;
план привязан к target, ID и mtime train. При изменении данных, target, ID,
провайдера, модели, `max_features`, `max_tokens` или JSON mode UI сбрасывает
план и требует сгенерировать новый. Запуск без approval блокируется и в UI,
и на сервере.
Для `openai` ключ остаётся в окружении (`KCONV_LLM_API_KEY` или
`OPENAI_API_KEY`) и никогда не вводится в браузере. Если ключа нет, кнопка
генерации блокируется с понятным сообщением; задайте переменную и
перезапустите `start.cmd`. Для `local` разрешён только loopback endpoint.
Ошибки провайдера показываются отдельно: недоступный endpoint, HTTP
401/404/429, невалидный JSON и несоответствие схемы.

Перед отправкой sample rows фильтруются по allow-list: target, id и посторонние
колонки вырезаются и в CLI, и защитно внутри `build_prompt`. Поэтому даже
случайно переданная полная строка не попадёт в HTTP-запрос.


### Шаг 3. Валидация без исполнения

`parse_plan`:

- убирает Markdown fence;
- проверяет JSON-схему;
- проверяет `name` и `kind`; небезопасные имена превращает в уникальный
  snake_case, а target/id (`reserved`) не разрешает как имя признака;
- проверяет, что все identifiers — известные колонки; имена с пробелами,
  слэшами и пунктуацией можно писать как `col("exact column name")`, а
  bare-варианты парсер нормализует сам;
- если передан `numeric_columns`, запрещает категориальные колонки в
  арифметике, `row_*` и value-аргументе `group_mean/group_std` (group-key
  может быть категориальным);
- отбрасывает дубликаты и лишние признаки;
- возвращает warnings;
- **не выполняет** выражение.

Пример допустимого ответа:

```json
{
  "features": [
    {
      "name": "amount_per_term",
      "kind": "numeric",
      "expression": "amount / term",
      "rationale": "relative monthly burden"
    }
  ]
}
```

### Шаг 4. Материализация

Реализовано в `kconv.features.llm_expr`:

- `parse_expression(expression, columns)` — строгий whitelist AST: `+ - * /`,
  unary `+/-`, скобки, числовые литералы, имена известных колонок и
  `col("exact column name")` для имён с пробелами/пунктуацией; остальные
  вызовы, атрибуты, индексы, lambda и comprehensions отклоняются с
  `KconvError`;
- `evaluate_expression(expression, frame)` — рекурсивный обход дерева без
  `eval`/`exec`; деление на ноль даёт `NaN`;
- `materialize_plan(plan, frame, *, columns=None)` — добавляет валидированные
  признаки в копию таблицы; `columns` задаёт allow-list идентификаторов
  (target/id исключены), проверяются конфликты имён.

Для планов с функциями используется `kconv.features.fold_plan`:

- `parse_fold_expression(expression, columns)` — арифметика плюс
  `row_mean/row_std/row_min/row_max/row_sum` и
  `group_mean/group_std/group_count`; колонки можно писать как identifier или
  `col("exact column name")`; attributes, subscripts, keywords, nested
  group-агрегаты и неизвестные функции отклоняются;
- `FoldPlanTransformer(plan, columns=...).fit(train_fold)` — парсит план и
  учит group-mapping только на outer train fold;
- `.transform(validation_or_test)` — использует сохранённые mapping; unseen
  group получает train-prior, `group_count` — 0; пересчёта по validation/test
  нет. Деление на ноль и любые не-конечные результаты (`NaN`/`inf`) заменяются
  на `0.0`, чтобы `feature_selection` и модели получали finite-матрицу;
- `FeatureBuilder` применяет `FeatureConfig.fold_plan` внутри каждого fold,
  поэтому признаки недоступны для утечки через предварительную материализацию.

Row-wise функции безопасны и без fit, но проходят через тот же transformer,
чтобы CLI не смешивал два пути. Group-агрегаты **label-free**; target-derived
статистики, nested group-агрегаты и произвольные Python-функции запрещены.
`--apply` поддерживает только арифметику; планы с функциями нужно передавать в
`kconv run --llm-plan-file`.


### Шаг 5. Оценка

Обязательный протокол:

1. Baseline без LLM-признаков.
2. Тот же split, те же seeds, те же модели.
3. Добавить признаки в train fold, не в validation/test.
4. Сравнить метрику и дисперсию по folds.
5. Оставить признак только если улучшение стабильно и объяснимо.
6. Для regression проверить и RMSE, и MAE; для classification — log loss и
   ROC AUC/PR AUC, где применимо.

### Шаг 6. Leakage-проверки

- target не должен встречаться в prompt;
- id/row index не должны попадать в признаки;
- выражение не должно использовать колонки, которых нет в fold;
- target encoding и агрегаты — только inner OOF;
- group/time split должен сохраняться;
- adversarial validation не должна ухудшаться;
- признаки не должны быть почти уникальными для каждой строки.

### Шаг 7. Логирование и воспроизводимость

`experiments propose` пишет sidecar `<output>.llm.json` (например,
`plan.llm.json`), а `kconv run --experiment ... --llm-plan-file plan.json`
копирует/обновляет его в `05_report/llm_features.json` после выполнения
конвейера. Можно передать явный файл: `--llm-log custom.json`.

Поля:

- `provider`, `model`, `request_id`, `finish_reason`, `duration_sec`;
- `prompt_sha256` и `prompt_chars`; **сам prompt не логируется**;
- `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`; если
  провайдер не вернул usage, значения `null` и `usage_available=false`
  (kconv не выдумывает оценки);
- `temperature`, `seed`, `sample_rows`, `max_features`, `max_tokens`,
  `json_mode` (фактический режим после возможного fallback);
- `plan_file`, `plan_sha256`, `proposals`, `warnings`;
- `approved` (true только после `--approve-llm-features`), `approval_required`;
- `generated_at`.

`mock`/`plan`-режимы пишут `usage_available=false`; это не ошибка, а честная
пометка, что токены не тратились/неизвестны. Метрики baseline vs experimental
остаются в `report.md` и `blend.json`; LLM-лог не заменяет CV-отчёт.


### Шаг 8. Human approval и CLI

LLM-признаки не включаются в `submission.csv` автоматически. Полный цикл:

```powershell
# 1. Предложить и валидировать план
#    mock — офлайн-демо без сети; plan — проверенный JSON от любого LLM;
#    openai — явный opt-in сетевой вызов с ключом из окружения
uv run kconv experiments propose --data data --target target --provider mock `
    --output plan.json --apply augmented.csv

# 1b. Реальный провайдер (только при наличии credentials)
uv run kconv experiments propose --data data --target target `
    --provider openai --llm-model gpt-4o-mini --output plan.json

# 2. Посмотреть/отредактировать plan.json (name, expression, rationale)
#    и plan.llm.json (provider, usage, prompt hash) до одобрения

# 3. Запуск с явным одобрением
uv run kconv run --data data --target target `
    --experiment llm-feature-proposals --approve-llm-features `
    --llm-plan-file plan.json
# 05_report/llm_features.json появится в каталоге запуска
```

`run --experiment llm-feature-proposals` материализует план в временную папку
(`train` и `test` с новыми колонками; sample_submission копируется без
изменений) и передаёт её конвейеру; исходный датасет не изменяется. Без
`--approve-llm-features` или без `--llm-plan-file` команда завершается ошибкой.
`parse_plan` получает только feature-колонки, поэтому target и id не могут
попасть в выражение.

`experiments propose` без `--provider openai`/`local` не вызывает сеть: ответ
LLM передаётся как JSON-файл (`--provider plan`) или генерируется
офлайн-провайдером `mock` для smoke-тестов. `--provider openai` и
`--provider local` — единственные ветки, которые делают HTTP-запрос; ключ
берётся из `KCONV_LLM_API_KEY` (fallback `OPENAI_API_KEY` только для
официального endpoint; loopback может работать без ключа) и не попадает ни в
stdout, ни в JSON-лог. `--output`, `--apply` и `--llm-log` не могут указывать
на один путь или на входной датасет: коллизия отклоняется до записи.


## Стоимость и риски

| Риск | Контроль |
|---|---|
| Галлюцинация колонки | whitelist identifiers |
| target leakage | target вырезан; проверка в AST |
| PII в prompt | redaction + только sample |
| Недетерминизм | temperature=0, seed, prompt hash |
| Дорогие токены | max_features, один вызов, кэш ответов |
| Переобучение на CV | фиксированный split, repeated seeds |
| Код из LLM | никакого eval/exec |

## Задачи для суб-агентов

| Агент | Задача | Acceptance | Статус |
|---|---|---|---|
| `researcher` | собрать 20 необычных feature-engineering паттернов из Kaggle/GitHub, с лицензиями | таблица pattern/source/license/risk | сделано |
| `ml-engineer` | проверить LLM-features на 2 табличных датасетах с leakage-контролем | baseline vs experiment, seeds, отчёт | частично |
| `implementer` | реализовать whitelist AST + `FeatureSpec` + materializer | unit-тесты на `a / b`, деление на 0, запрет `__import__` | сделано: `src/kconv/features/llm_expr.py`, `tests/test_llm_expr.py` |
| `reviewer` | проверить prompt/privacy/leakage | список блокеров или approve | self-review, P1 исправлен |

## Критерий готовности эксперимента

- [ ] Есть baseline и experimental run с одинаковыми folds/seeds.
- [ ] LLM не видит target и PII.
- [ ] Нет `eval`/`exec` над сырым ответом.
- [ ] Все принятые признаки воспроизводимы по `FeatureSpec`.
- [ ] Улучшение подтверждено на out-of-fold, а не на train.
- [ ] Отчёт содержит стоимость токенов и источник каждого признака.



## Ограничения и цикл исправлений

Модель с контекстом 4096 токенов генерирует один признак за вызов; за один
запуск можно получить 5–20 признаков. Ответ должен быть JSON, который
проходит строгую проверку AST и арифметического whitelist. Для `local` и
loopback-endpoint данные не покидают машину; `openai` — явный opt-in.

1. kconv формирует компактный prompt: имена колонок, до 3 sample rows,
   типы, статистики и ограничения. Target и ID не отправляются.
2. Ответ проходит `parse_plan`: проверяются имена, AST, допустимые
   операции и отсутствие запрещённых вызовов.
3. При ошибке kconv повторяет запрос с `Previous invalid answer` и
   `Validation error`. По умолчанию 2 попытки; `--llm-repairs 0..5`
   меняет число.
4. Успешный план сохраняется и требует approval перед материализацией.

Prompt ограничен 8000 символов, sample rows — 3, колонок — 40. Это
оставляет место для ответа в 4096-токенном контексте. Target и ID никогда
не попадают в prompt.

```powershell
uv run kconv experiments propose --data data --target target `
    --provider local --max-features 3 `
    --llm-idea "добавить отношения числовых колонок" `
    --llm-code-file ideas.py `
    --llm-repairs 3 --output plan.json
```

`--llm-idea` можно повторять. `--llm-code-file` читается как UTF-8 и
передаётся как **вдохновляющий контекст**: kconv не исполняет этот файл и
не доверяет ему. `--llm-repairs` ограничивает цикл исправлений.

Sidecar `<output>.llm.json` содержит `generation`: режим
`one-feature-per-call`, модели, repair-попытки, token usage и статус
attempts. План применяется только с `--approve-llm-features`; в web UI —
после явного approval.
