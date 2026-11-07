# Продвинутые feature-трансформеры

Четыре декларативных блока `numeric_interactions`, `clustering`, `anomaly` и
`time_series` живут в `src/kconv/features/`. Они не читают target, не выполняют
произвольный Python из recipe и fit-ятся только на outer training fold. Поэтому
их можно включать в pipeline без target leakage; статистики, выученные на
validation/test, не используются.

Общие правила:

- `FeatureBuilder` вызывает `fit()` на train-фолде, а `transform()` — на train,
  validation и test;
- при `oof=True` для train возвращаются признаки, построенные на этом же
  train-фолде; для test всегда используется `transform()`;
- пустой DataFrame с уже известными колонками возвращает пустой DataFrame с
  теми же именами признаков;
- `transform()` до `fit()` или с другим набором колонок даёт `DataError`;
- все выходные значения конечны: NaN/inf заполняются или клипуются.

## `numeric_interactions`

Параметры: `columns`, `max_pairs=6`, `operations=("product", "ratio")`,
`clip_value=1e12`.

Строит детерминированные pairwise-признаки по первым `max_pairs` парам
numeric-колонок. Порядок колонок и пар стабилен, поэтому одинаковый вход даёт
одинаковые имена и значения. Поддерживаются операции:

- `product` — произведение;
- `ratio` — деление; при нулевом знаменателе результат `0`;
- `difference` — разность;
- `sum` — сумма.

Имена: `<left>__x__<right>__product`, `<left>__x__<right>__ratio`,
`<left>__x__<right>__diff`, `<left>__x__<right>__sum`. Результат клипуется в
`[-clip_value, clip_value]`. Target не используется, fitted-состояния нет.

## `clustering`

Параметры: `columns`, `n_clusters=8`, `max_iter=100`,
`include_distances=True`, `include_labels=True`, `max_fit_rows=50000`.

На outer train fold:

1. выбранные numeric-колонки приводятся к `float64`;
2. NaN/inf заменяются медианой train-фолда (или `0`, если медиана не finite);
3. при необходимости детерминированно сэмплируются `max_fit_rows` строк;
4. считается mean/std и обучается `MiniBatchKMeans` с `random_state` из config;
5. `n_clusters` фактически ограничивается числом fit-строк.

Выходы: `cluster__dist_<i>` для расстояний до центров и/или `cluster__label`.
`transform()` использует уже выученные fill-значения, mean/scale и центры, а не
переобучается. Это unsupervised-признаки: target не читается.

## `anomaly`

Параметры: `columns`, `n_estimators=100`, `contamination="auto"`,
`include_flag=True`.

На outer train fold выбранные numeric-колонки заполняются медианой и
обучается `IsolationForest` (`n_jobs=1`, фиксированный `random_state`).
Выходы:

- `anomaly__score` — `-score_samples`; большее значение означает более
  аномальную строку;
- `anomaly__flag` — `1.0` для outlier и `0.0` иначе (если `include_flag`).

`contamination` — `"auto"` или число в `(0, 0.5]`. `transform()` использует
выученные fill-значения и модель; target не читается.

## `time_series`

Параметры: `time_col`, `group_col=None`, `columns`, `lags=(1,)`,
`windows=(3,)`, `stats=("mean",)`, `include_elapsed=True`,
`include_diff=True`, `max_history_rows=10000`.

`time_col` обязателен и должен быть datetime-колонкой. `group_col` делит ряд на
независимые группы; `columns` — numeric value-колонки (по умолчанию все numeric,
кроме time/group). Выходы:

- `ts__<col>__lag_<n>` — значение из прошлой строки группы;
- `ts__<col>__<stat>_<w>` — trailing-статистика по прошлым `w` строкам;
  поддерживаются `mean`, `std`, `min`, `max`;
- `ts__elapsed_seconds` — секунды от первого timestamp группы в train;
- `ts__diff_seconds` — секунды от предыдущего timestamp группы.

Ключевая гарантия: rolling считается после `shift(1)`, поэтому текущая строка
никогда не попадает в собственное окно. Для test используются сохранённые
`max_history_rows` прошлых строк train плюс уже обработанные test-строки.
`fit()` сохраняет ограниченный хвост истории и первые timestamp по группам;
`transform()` не переобучается. Время парсится как UTC и нормализуется до
наносекунд, а elapsed/diff считаются в секундах.

## Проверка

```powershell
uv run pytest -q tests/test_feature_advanced.py
uv run pytest -q tests/test_feature_blocks.py tests/test_project.py
uv run pytest -q tests/test_pipeline_matrix.py
uv run ruff check src/kconv/features tests/test_feature_advanced.py
```

Матрица `tests/test_pipeline_matrix.py` прогоняет все executable pipelines на
binary, multiclass и regression, включая новые time-series/cluster/anomaly
рецепты. Это проверяет исполнение и контракты, но не доказывает рост score на
реальном соревновании.
