# Формат `config.yaml`

Конфиг строгий: лишнее поле, неверный тип или неизвестная модель — ошибка
валидации. Создать основу:

```powershell
uv run kconv run --data data --target target --no-run
```

Проверить:

```powershell
uv run kconv config validate --comp <slug>
```

## Минимальный пример

```yaml
project: titanic
seed: 42
recipe: pesnik/baseline
strategy: fast
task: binary
metric: roc_auc
data:
  train: ../../data/train.csv
  test: ../../data/test.csv
  sample_submission: ../../data/sample_submission.csv
  target: Survived
  id: PassengerId
models:
  - name: lgbm
    seeds: 1
    params:
      verbosity: -1
      num_threads: 1
```

## Верхний уровень

| Поле | Тип | По умолчанию | Смысл |
| --- | --- | --- | --- |
| `schema_version` | int | `1` | Версия схемы |
| `project` | str | обязательно | Slug проекта |
| `seed` | int | `42` | Глобальный seed |
| `recipe` | str | `pesnik/baseline` | Recipe для аудита |
| `recipe_kind` | str \| null | `null` | `preset` или `pipeline`; informational provenance |
| `strategy` | str | `fast` | `fast`, `standard`, `deep` |
| `task` | str | `auto` | `binary`, `multiclass`, `regression` |
| `metric` | str | `auto` | Метрика соревнования |
| `data` | mapping | обязательно | Источники и роли |
| `validation` | mapping | см. ниже | CV |
| `features` | mapping | см. ниже | Признаки |
| `models` | list | `[lgbm]` | Модели и параметры |
| `hpo` | mapping | выключен | Nested HPO |
| `ensemble` | mapping | `auto` | Блендинг |
| `submit` | mapping | пусто | Правила submission |

## `data`

```yaml
data:
  train: ../../data/train.csv
  test: ../../data/test.csv
  sample_submission: ../../data/sample_submission.csv
  target: target
  id: id
  ignore: [leak_column]
```

- `train` обязателен; может быть папкой, файлом или zip.
- `test`/`sample_submission` можно не указывать — включается автоопределение.
- `target`/`id` можно не указывать — включается автоопределение.
- `ignore` исключает колонки из признаков; target игнорировать нельзя.

Относительные пути разрешаются от папки проекта
(`competitions/<slug>/`), а не от текущего каталога. Это делает конфиг
переносимым, если данные лежат рядом с репозиторием. Пути на другом диске
остаются абсолютными.

## `validation`

```yaml
validation:
  strategy: auto
  n_splits: 5
  shuffle: true
  random_state: 42
  group_col: null
  time_col: null
  gap: 0
  holdout_size: 0.2
  adversarial: true
  adversarial_folds: 3
  adversarial_threshold: 0.75
```

`strategy`: `auto`, `kfold`, `stratified_kfold`, `group_kfold`, `time_series`,
`holdout`.

- `auto` выбирает group/time, если заданы `group_col`/`time_col`, иначе
  stratified KFold для классификации и KFold для регрессии.
- `group_kfold` требует `group_col`; группы не пересекаются между фолдами.
- `time_series` требует `time_col`; данные сортируются, будущее не попадает
  в train.
- `holdout` делает один split с `holdout_size`.
- `gap` используется для временного split.
- `adversarial` включает train-vs-test диагностику; target в ней не
  используется. `adversarial_folds` — число фолдов (2–10),
  `adversarial_threshold` — порог AUC для предупреждения (0.5–1.0).
  Результат пишется в `01_validate/adversarial.json`; при AUC выше порога
  warning попадает в `report.md`. Это диагностика, а не автоматическая смена
  CV-схемы.

## `features`

```yaml
features:
  target_encode: []
  frequency_encode: []
  auto_encode: true
  scale: auto
  impute: median
  drop_constant: true
  max_categories: 50
  # Label-free row aggregates over numeric features.
  row_statistics: false
  row_statistics_stats: [mean, std, min, max]
  # Opt-in modulo/quotient views for identifier-like numeric columns.
  modulo_features: false
  modulo_moduli: [10, 100]
  modulo_steps: [5]
  # Classification-only class-conditional kNN distance features.
  knn_features:
    enabled: false
    columns: []
    neighbors: 3
    folds: 5
    metric: euclidean
    max_reference_rows: 50000
  # Optional final selector; variance, mutual information or model importance.
  feature_selection:
    enabled: false
    method: variance
    max_features: 50
    min_variance: 0.0
  # Stable label-free hash buckets for high-cardinality categorical columns.
  hash_encode: []
  hash_features: 16
  # Pairwise categorical interactions; target mode is inner-OOF.
  interactions:
    - columns: [cat_a, cat_b]
      encoding: frequency
  # Label-free missingness / row-profile features.
  missingness:
    enabled: false
    columns: []
    indicators: true
    row_count: true
    zero_count: false
  # Label-free sin/cos views of parsed datetime columns.
  datetime_cyclical:
    enabled: false
    columns: []
    periods: [month, dayofweek, dayofyear, hour]
  # Label-free pairwise numeric products, ratios, differences and sums.
  numeric_interactions:
    enabled: false
    columns: []
    max_pairs: 6
    operations: [product, ratio]
    clip_value: 1.0e12
  # Fit-in-fold MiniBatchKMeans cluster distances and/or labels.
  clustering:
    enabled: false
    columns: []
    n_clusters: 8
    max_iter: 100
    include_distances: true
    include_labels: true
    max_fit_rows: 50000
  # Fit-in-fold IsolationForest anomaly score and optional flag.
  anomaly:
    enabled: false
    columns: []
    n_estimators: 100
    contamination: auto
    include_flag: true
  # Past-only grouped lags, rolling statistics and elapsed/diff time.
  time_series:
    enabled: false
    time_col: null
    group_col: null
    columns: []
    lags: [1]
    windows: [3]
    stats: [mean]
    include_elapsed: true
    include_diff: true
    max_history_rows: 10000
  # Optional reviewed plan; usually injected by:
  # kconv run --experiment llm-feature-proposals --llm-plan-file plan.json
  fold_plan:
    features:
      - name: grp_mean_num
        expression: group_mean(num, grp)
  # Executable plan; pipeline recipes compile into this. Absent/null keeps the
  # legacy flag-driven behavior and the old config hash.
  feature_blocks: null
  # Optional override for target-encoding smoothing; null means 10.0.
  target_smoothing: null
```

`feature_blocks` — ordered list of `{type, params, enabled}`. When present it
is authoritative: legacy flags below are kept only for introspection, and
`FeatureBuilder` resets undeclared feature families. Presets leave it `null`.
See `docs/pipelines.md` for the block catalog and execution rules.

`config_hash` excludes `recipe_kind` and default-valued additions
(`row_statistics: false`, default modulo lists, disabled `knn_features`,
disabled `feature_selection`, empty `hash_encode`/default `hash_features`,
empty `interactions`, disabled `missingness`, disabled `datetime_cyclical`,
disabled `numeric_interactions`/`clustering`/`anomaly`/`time_series`, etc.),
so a config created before these fields existed hashes exactly as before.
Non-default values remain part of the hash. `--resume` therefore accepts old
runs without a config migration.

- `target_encode` — список категориальных колонок; кодирование fit-in-fold с
  внутренней OOF-схемой и сглаживанием (по умолчанию `10.0`, переопределяется
  `target_smoothing` или параметром target-блока).
- `frequency_encode` — доля каждой категории в train-фолде.
- `auto_encode` — при создании проекта заполнить пустые списки для
  высококардинальных категорий. Итог записывается в конфиг; можно выключить.
- `scale`: `auto` (стандартизация для linear-моделей), `none`, `standard`.
- `impute`: `median`, `most_frequent`, `constant`.
- `drop_constant` — удалять константные признаки.
- `max_categories` — сколько категорий сохранять в категориальном словаре
  (для one-hot линейных моделей и ordinal для деревьев).
- `row_statistics` — добавить `row_mean`, `row_std`, `row_min`, `row_max`
  и/или `row_median` по всем numeric-колонкам. Агрегаты label-free,
  считаются построчно и не требуют fit; список статистик задаёт
  `row_statistics_stats`. NaN-строки заполняются нулём.
- `modulo_features` — добавить `<column>__mod_<m>` и `<column>__div_<step>`
  для identifier-like numeric-колонок. Это opt-in: применять к обычным
  непрерывным признакам без проверки не следует. Значения модулей и шагов
  задают `modulo_moduli` и `modulo_steps` (положительные целые).
- `knn_features` — class-conditional kNN distance features (`knn_c<idx>_k<k>`),
  только для `binary`/`multiclass`. Reference set — внешний train fold;
  training-строки получают inner-OOF значения, validation/test — отображение,
  fitted на полном train fold. `columns` по умолчанию означает все numeric
  колонки; `neighbors` — число соседей на класс; `folds` — максимум inner
  фолдов; `max_reference_rows` ограничивает reference set на класс и защищает
  память/время. Для регрессии включение даёт `DataError`.
- `feature_selection` — финальный selector после сборки матрицы.
  `method: variance` удаляет колонки с дисперсией не выше `min_variance` и
  оставляет не больше `max_features`; `method: mutual_info` ранжирует через
  mutual information на outer training fold (`min_variance` игнорируется);
  `method: model_importance` обучает небольшой ExtraTrees на outer training
  fold и оставляет признаки с наибольшей важностью (`min_variance`
  игнорируется). Selector не видит validation/test labels. В pipeline-блоке
  он должен быть последним enabled-блоком.
- `hash_encode` — список категориальных колонок для стабильного label-free
  hash encoding (`hash__0..N`); `hash_features` задаёт число общих bucket'ов
  (1..1024, по умолчанию 16). Fitted vocabulary не нужен, поэтому transform
  validation/test детерминирован; коллизии ожидаемы. В pipeline это делает
  блок `hash_encoding` с `columns: auto`/списком и `n_features`.
- `interactions` — список пар категориальных колонок. `encoding: frequency`
  считает частоту комбинации на outer training fold (label-free);
  `encoding: target` использует inner-OOF target map, как обычный target
  encoder. Имя признака: `<col_a>__x__<col_b>__frequency` или `__target`.
- `missingness` — label-free профиль пропусков: `row_count` даёт
  `missing__count`, `indicators` — `<column>__missing`, `zero_count` —
  `zero__count` по выбранным numeric-колонкам. `columns: []` означает все
  feature-колонки; явный список проверяется.
- `datetime_cyclical` — sin/cos для `month` (12), `dayofweek` (7),
  `dayofyear` (365.25) и `hour` (24). `columns: []` означает все datetime
  feature-колонки; на датасете без datetime блок не добавляет признаков.
- `numeric_interactions` — label-free пары numeric-колонок: произведения,
  отношения, разности и суммы (`<a>__x__<b>__product|ratio|diff|sum`).
  `columns: []` означает все numeric-колонки, `max_pairs` ограничивает число
  пар (1–50), `operations` — непустое подмножество
  `product/ratio/difference/sum`; деление на ноль даёт 0, выходы клипуются
  по `clip_value` (по умолчанию `1e12`).
- `clustering` — MiniBatchKMeans на outer training fold (не более
  `max_fit_rows` строк, `n_clusters` 2–50, `max_iter` 10–1000). Отдаёт
  `cluster__dist_<i>` при `include_distances` и/или `cluster__label` при
  `include_labels`; target не читается. Нужен хотя бы один из этих выходов,
  `columns: []` означает все numeric-колонки.
- `anomaly` — IsolationForest на outer training fold: `anomaly__score` и
  опциональный `anomaly__flag` при `include_flag`. `n_estimators` 10–500;
  `contamination` — `auto` или число в `(0, 0.5]`; `columns: []` означает
  все numeric-колонки.
- `time_series` — past-only признаки по времени: `ts__<col>__lag_<n>`,
  `ts__<col>__<stat>_<w>`, `ts__elapsed_seconds` и `ts__diff_seconds`.
  `time_col` обязателен (в pipeline допускается `auto` до компиляции),
  `group_col` опционален, `columns: []` означает все numeric-колонки.
  `lags` — 1–100, `windows` — 2–100, `stats` — подмножество
  `mean/std/min/max`; `max_history_rows` (100–1000000) ограничивает
  history-tail, который используется для validation/test. На датасете без
  datetime запуск отклоняется до обучения.
- `fold_plan` — опциональный reviewed-план. `FoldPlanTransformer` парсит
  арифметику и `row_mean/row_std/row_min/row_max/row_sum` плюс label-free
  `group_mean/group_std/group_count`; group-mapping учится только на train
  fold, validation/test используют сохранённые значения, а неизвестная группа
  получает train-prior; деление на ноль и не-конечные результаты заменяются на
  `0.0`. Target-derived статистики запрещены. Fold-aware план не
  копирует датасет в temp: `data.train`/`data.test` остаются путями к исходным
  данным, поэтому проект переживает очистку temp и resume. Pipeline-блок
  `group_statistics` компилируется в такой же plan автоматически; одновременно
  `group_statistics` и `fold_plan` использовать нельзя. Блок принимает только
  ASCII identifier-имена колонок; сгенерированные имена длиннее 40 символов
  хешируются в `group__<sha1[:32]>`.

## `models`

```yaml
models:
  - name: lgbm
    enabled: true
    seeds: 1
    params:
      n_estimators: 500
      learning_rate: 0.05
    early_stopping_rounds: 100
  - name: logreg
    enabled: false
```

- `name` — id из `kconv models`; доступно 51 id (6 нативных/boosted,
  1 optional `tabpfn` foundation и 44 scikit-learn), полный каталог —
  `docs/models.md`.
- `enabled` — выключенные модели не обучаются.
- `seeds` — число seed-усреднений (1–10).
- `params` — параметры адаптера; передаются в библиотеку модели. Их можно
  переопределить в web UI в опциональном блоке «Гиперпараметры моделей»:
  ручные значения применяются после recipe/profile defaults.
- `early_stopping_rounds` — используется, если адаптер поддерживает eval set.
- Имена включённых моделей должны быть уникальны; несовместимые с задачей
  id отбрасываются, явный несовместимый id даёт ошибку.

## `hpo`

```yaml
hpo:
  enabled: true
  trials: 8
  timeout_sec: 600
  inner_size: 0.2
```

HPO запускается стадией `hpo` до `train`. Для каждого внешнего CV-фолда
создаётся отдельный внутренний split; лучшие параметры хранятся по фолдам в
`01_hpo/tuning.json`. `trials` — попыток на fold/model (включая проверку
recipe-параметров), `inner_size` — доля внутренней валидации, `timeout_sec` —
общий бюджет стадии. `deep` включает HPO с 8 trials; `fast`/`standard` — нет.

Если бюджет закончился, оставшиеся фолды используют recipe-параметры, а
`timeout_hit` в отчёте будет `true`. Это ограниченный случайный поиск, не
Bayesian optimization.

## `ensemble`

```yaml
ensemble:
  method: auto
  fallback: best_single
  min_improvement: 0.0
  meta_model: auto
  meta_splits: 5
```

`method`: `auto`, `best_single`, `mean`, `rank_mean`, `weighted`, `stack`.

- `auto` — `weighted` при нескольких моделях, `best_single` при одной.
- `weighted` оптимизирует неотрицательные веса на OOF (SLSQP).
- `stack` обучает мета-модель на OOF-предсказаниях. Для регрессии по умолчанию
  используется Ridge, для классификации — LogisticRegression с вероятностями.
  `meta_model: auto` выбирает модель по задаче; `ridge` для классификации
  разрешается в LogReg, потому что нужны калиброванные вероятности.
  `meta_model: lgbm` включает нелинейную LightGBM-мета-модель для регрессии,
  binary и multiclass; её вероятности также проходят внутреннюю OOF-CV.
- `meta_splits` — число фолдов внутренней CV мета-модели (2–20). OOF-оценка
  стека считается на этих фолдах, а финальная мета-модель обучается на всех
  OOF-строках и предсказывает test.
- `stack` требует минимум две модели и достаточно строк для каждого класса;
  иначе используйте `weighted`.
- Если бленд не лучше лучшей одиночной модели на `min_improvement`, срабатывает
  `fallback` (`best_single` или `mean`). В `blend.json` поле `stack` сохраняет
  параметры мета-модели, а `fallback_used` показывает, что сработал fallback.

## `runtime`

Необязательный блок. Появляется, если проект создан через
`kconv run --profile 1..5`.

```yaml
runtime:
  profile: 4
  device: auto
  ram_gb: 32
  vram_gb: 12
  cpu_threads: 8
```

- `profile` — 1–5; профили 1–3 CPU-only.
- `device` — `auto`, `cpu`, `gpu`, `cuda`. Профили 4–5 по умолчанию `auto`:
  `auto` использует GPU, если найден рабочий LGBM/XGB/CatBoost, иначе CPU;
  явный `gpu`/`cuda` без GPU завершается `KconvError`.
- `ram_gb`/`vram_gb` — записанные бюджеты RAM/VRAM. Они сохраняются в конфиге и
  отображаются в интерфейсе/выводе, но не ограничивают выполнение: это не
  cgroup, не лимиты драйвера и не основа для автоматического планирования.
- `cpu_threads` — потоки для LGBM/XGB/CatBoost; `0` означает auto/all. В web UI
  и CLI можно переопределить `--cpu-threads`, `--ram-gb`, `--vram-gb`,
  `--device` при выбранном профиле. Эти значения попадают в `models[].params`
  при создании проекта; при запуске существующего `config.yaml` эффективны
  именно `models[].params`, а `runtime.*` остаётся снимком профиля.
- Профиль переопределяет `strategy`, seeds, HPO и лимит авто-моделей.
- Явный `--models` имеет приоритет над авто-выбором профиля.
- GPU-параметры получают только LGBM/XGB/CatBoost и только если установленная
  сборка библиотеки их поддерживает; scikit-learn остаётся на CPU.

- `stage_timeout_sec` — таймаут стадии в секундах; `0` отключает
  ограничение. При `>0` стадия прерывается и пишет postmortem.
- `model_timeout_sec` — таймаут на fit одной модели; `0` отключает
  ограничение.
- `model_retries` — число повторов fit при ошибке модели (0–3). Ошибки
  сохраняются в `warnings`.
- `fallback_models` — список запасных моделей из registry. Они используются,
  если основная модель не смогла обучиться; значения должны быть совместимы
  с задачей.
- `isolate_models` — запускать fit в отдельном процессе. Если
  `model_timeout_sec` не задан, используется 3600 секунд. Это защищает
  основной процесс от зависания модели.
- `feature_cache` — кэшировать fitted feature builder в `.cache/features/`.
  Ключ включает версию, seed, версию kconv, family, fold, split и target;
  кэш автоматически инвалидируется. По умолчанию `true`.

## `submit`

```yaml
submit:
  id_col: null
  target_col: null
  clip: [0.0, 1.0]
  round: null
```

- `id_col`/`target_col` — переопределить колонки sample submission.
- `clip` — обрезать предсказания после выбора labels/вероятностей.
- `round` — округлить числовой submission.

## Что проверяется автоматически

- Наличие target в train и совместимость задачи с моделями/recipe.
- Совместимость `metric` с задачей: classification-метрики для
  `binary`/`multiclass`, `rmse`/`mae`/`r2` для `regression`; неизвестная метрика
  отклоняется до запуска обучения.
- Наличие numeric-колонок для `numeric_interactions`/`clustering`/`anomaly`/
  `time_series` и datetime-колонки для `time_series`; `time_col`/`group_col`
  проверяются по схеме датасета до обучения.
- Уникальность включённых моделей.
- Диапазоны `n_splits`, `seeds`, `max_categories`, `holdout_size`, `clip`.
- Наличие group/time колонок на стадии validate.
- Совпадение числа строк и колонок submission с ожидаемой схемой.
- NaN/inf в предсказаниях и submission.

## Ошибки и подсказки

При ошибке kconv печатает сообщение и `Hint`. Для диагностики смотрите:

- `runs/<run_id>/state.json` — стадия и текст ошибки;
- `runs/<run_id>/00_ingest/data_card.json` — роли, пропуски, дисбаланс;
- `runs/<run_id>/01_validate/cv_summary.json` — выбранная CV-схема;
- `runs/<run_id>/05_report/report.md` — итоговая сводка.
