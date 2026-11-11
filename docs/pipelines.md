# Executable pipelines

`kind: pipeline` — это декларативный feature graph, который kconv компилирует
в `config.yaml` и исполняет в `FeatureBuilder`. Чужой Python-код не
запускается: блоки реализованы в kconv, а recipe задаёт только их порядок и
параметры.

## Жизненный цикл

```text
YAML recipe
  -> build_config()
       -> pipeline.blocks копируются в features.feature_blocks
       -> columns: auto разрешаются по sample train
       -> legacy-флаги синхронизируются для интроспекции
       -> group_col/time_col проверяются по колонкам
  -> config.yaml (feature_blocks + recipe_kind: pipeline)
  -> FeatureBuilder.fit_transform()
       -> validate_block_plan()
       -> блоки исполняются по порядку внутри outer fold
  -> OOF/test/submission/report
```

После сборки проекта recipe больше не влияет на запуск: всё, включая
`feature_blocks`, зафиксировано в `config.yaml` и входит в `config_hash` для
`--resume`.

## Блоки

| Блок | Параметры | Семантика |
| --- | --- | --- |
| `base` | `scale`, `impute`, `max_categories`, `auto_encode`, `drop_constant` | Numeric/categorical matrix; обязателен, первый и ровно один. |
| `target_encoding` | `columns`, `smoothing` | Inner-OOF target encoding; fit только на training fold, leave-one-out при <8 строках. |
| `frequency_encoding` | `columns` | Частоты категорий, посчитанные на training fold. |
| `hash_encoding` | `columns`, `n_features` | Стабильные label-free hash-бакеты для высококардинальных категорий; общий bucket space, без fitted vocabulary. |
| `group_statistics` | `group_col`, `columns`, `stats`, `max_columns` | Fit-in-fold group mean/std/count; компилируется в reviewed `fold_plan`, unseen-группы получают 0/fallback. |
| `categorical_interaction` | `columns`, `max_pairs`, `encoding` | До `max_pairs` попарных категориальных комбинаций; `frequency` label-free, `target` — inner-OOF. |
| `missingness` | `columns`, `indicators`, `row_count`, `zero_count` | Label-free missing indicators, row-wise missing count и numeric zero count. |
| `datetime_cyclical` | `columns`, `periods` | Label-free sin/cos для month/dayofweek/dayofyear/hour; auto без datetime — no-op. |
| `numeric_interactions` | `columns`, `max_pairs`, `operations`, `clip_value` | Label-free произведения/отношения/разности/суммы до `max_pairs` пар numeric-колонок; деление на ноль даёт 0, выходы клипуются. |
| `clustering` | `columns`, `n_clusters`, `max_iter`, `include_distances`, `include_labels`, `max_fit_rows` | Fit-in-fold MiniBatchKMeans на outer training fold: `cluster__dist_<i>` и/или `cluster__label`; target не читается. |
| `anomaly` | `columns`, `n_estimators`, `contamination`, `include_flag` | Fit-in-fold IsolationForest: `anomaly__score` и опциональный `anomaly__flag`; `contamination` — `auto` или в `(0, 0.5]`. |
| `time_series` | `time_col`, `group_col`, `columns`, `lags`, `windows`, `stats`, `include_elapsed`, `include_diff`, `max_history_rows` | Past-only `ts__<col>__lag_<n>`, `ts__<col>__<stat>_<w>`, `ts__elapsed_seconds` и `ts__diff_seconds`; history ограничена. |
| `row_statistics` | `stats` | Mean/std/min/max/median по numeric-колонкам строки; label-free. |
| `modulo` | `moduli`, `steps` | Modulo/quotient views для identifier-like numeric. |
| `fold_plan` | `plan` | Утверждённый fold-aware LLM-план; выражения проверяются. |
| `feature_selection` | `method`, `max_features`, `min_variance` | Финальный selector: variance (label-free), mutual information или model importance (fit на outer training fold). Должен быть последним enabled-блоком. |
| `knn_class_distance` | `columns`, `neighbors`, `folds`, `metric`, `max_reference_rows` | Class-conditional kNN distance; CPU-only, binary/multiclass. |

Семантика, имена выходов и leakage-гарантии четырёх новых блоков подробно
описаны в `docs/feature-transformers.md`.

Правила:

- `base` обязателен, должен быть первым и включённым;
- каждый блок singleton, дубликаты отклоняются;
- явные `columns` у `target_encoding`/`frequency_encoding`/`hash_encoding`/
  `categorical_interaction`/`missingness`/`datetime_cyclical` должны быть
  колонками датасета (или `auto`), иначе `RecipeError` до запуска;
- `categorical_interaction` требует минимум две categorical-колонки при явном
  списке; `datetime_cyclical` принимает только datetime-колонки;
- `group_statistics` и `fold_plan` взаимоисключающи: оба определяют
  fold-агрегаты; `group_statistics` компилируется в reviewed fold plan;
- `group_statistics` принимает только ASCII identifier-имена колонок; если
  сгенерированное имя признака длиннее 40 символов, оно заменяется на
  `group__<sha1[:32]>`, чтобы не нарушать лимит fold plan;
- неизвестный тип, неизвестный параметр или отсутствующий обязательный
  параметр — `RecipeError` до запуска;
- `hash_encoding.n_features` — целое в `1..1024`; bucket space общий для всех
  выбранных колонок;
- `knn_class_distance` отклоняется для regression и без numeric-колонок;
- `feature_selection` должен быть последним enabled-блоком; `mutual_info` и
  `model_importance` используют target только внутри outer training fold;
- `target_encoding`/`frequency_encoding`/`hash_encoding`/`categorical_interaction`
  с `columns: auto` разрешаются в `build_config` и записываются обратно в
  конфиг; низкокардинальные колонки могут остаться без encoding, это ожидаемо;
- `datetime_cyclical` и `missingness` с `columns: auto` работают по всем
  подходящим колонкам; при отсутствии datetime cyclical-блок — no-op;
- `numeric_interactions`, `clustering`, `anomaly` и `time_series` требуют
  numeric-колонок; `time_series` дополнительно требует datetime-колонку
  (`time_col: auto` или явную), иначе запуск отклоняется до обучения;
- `clustering` должен включить `include_distances` или `include_labels`;
  `lags` лежат в `1..100`, `windows` — в `2..100`, `stats` — подмножество
  `mean/std/min/max`;
- `clustering`, `anomaly` и `time_series` fit-ятся только на outer training
  fold; `time_series` использует исключительно прошлые строки и сохраняет
  ограниченный history-tail для validation/test.

`feature_selection(method=variance)` удаляет колонки с дисперсией не выше
`min_variance` и оставляет не больше `max_features` самых вариативных.
`method=mutual_info` ранжирует признаки через `mutual_info_classif` или
`mutual_info_regression` на outer training fold и также оставляет не больше
`max_features`; `min_variance` в этом режиме не используется.
`method=model_importance` обучает небольшой `ExtraTreesClassifier`/`Regressor`
(64 дерева, фиксированный seed) на outer training fold и оставляет
`max_features` признаков с наибольшей `feature_importances_`; `min_variance`
игнорируется. Selector применяется после `drop_constant` и до обучения модели,
поэтому validation и test не влияют на выбранный набор.

`group_statistics` не пишет собственный transformer: `build_group_statistics_plan`
компилирует `group_mean`/`group_std`/`group_count` в тот же reviewed
`fold_plan`, который исполняет `FoldPlanTransformer`. Это сохраняет один
fit-in-fold механизм и исключает произвольные выражения из YAML.
`categorical_interaction` склеивает пары категориальных значений и считает
frequency на outer fold; `encoding: target` использует тот же inner-OOF
target map, что и `target_encoding`, поэтому строка не участвует в своём
собственном target-признаке. `missingness` и `datetime_cyclical` полностью
label-free и не требуют fitted state.

`numeric_interactions` считает детерминированные пары numeric-колонок:
`<a>__x__<b>__product`, `__ratio`, `__diff` и `__sum`. Это label-free блок:
он не читает target, а `max_pairs` ограничивает число пар, чтобы матрица не
росла квадратично. `clustering` обучает MiniBatchKMeans на outer training
fold (не более `max_fit_rows` строк) и отдаёт расстояния до центроидов и/или
метку кластера. `anomaly` обучает IsolationForest на том же fold и отдаёт
`anomaly__score` плюс опциональный `anomaly__flag`. `time_series` строит
лаги, trailing-статистики, elapsed и diff по времени внутри группы, используя
только строки не позже текущей; для validation/test он дополняет их
сохранённым history-tail.

## Каталог pipelines (39)

| id | derived_from | Блоки | Модели / ensemble |
| --- | --- | --- | --- |
| `pipeline/pesnik-clean` | `pesnik/baseline` | base → target → frequency | LGBM, weighted |
| `pipeline/cdeotte-gbdt-blend` | `cdeotte/tabular` | base → target → frequency | LGBM/XGB/CatBoost, weighted |
| `pipeline/dptools-row-profile` | `dptools/row-features` | base → target → frequency → row_statistics | LGBM/CatBoost, weighted |
| `pipeline/knnfeat-class-distance` | `knnfeat/knn-features` | base → target → frequency → knn_class_distance | LGBM/XGB, weighted; binary/multiclass only |
| `pipeline/kaggler-target-frequency` | `kaggler/target-encoding` | base → target → frequency | LGBM/XGB, weighted |
| `pipeline/kaggle-ieee-frequency` | `kaggle-ieee/fraud-pipeline` | base(max_categories=100) → frequency → target(smoothing=20) | LGBM/XGB, weighted |
| `pipeline/minerva-oof-stack` | `minerva/stacking` | base → target → frequency | LGBM/XGB/CatBoost/LogReg, stack |
| `pipeline/kconv-linear-stack` | `kconv/linear-stack` | base(scale=standard) → target → frequency | LogReg/Ridge/HistGB, stack |
| `pipeline/arnabbiswas-staged` | `arnabbiswas1/staged` | base → target → frequency → feature_selection(mutual_info, max_features=25) | LGBM/XGB, weighted |
| `pipeline/kozodoi-high-cardinality` | `kozodoi/tabular` | base(max_categories=100) → hash_encoding(32) → target → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/sergeyshilin-contracts` | `sergeyshilin/contracts` | base → target → frequency → feature_selection(model_importance, 30) | LGBM/CatBoost/LogReg, mean; adversarial threshold 0.8 |
| `pipeline/nyaggle-oof-ensemble` | `nyaggle/oof-ensemble` | base → target → frequency → feature_selection(model_importance, 35) | LGBM/XGB/CatBoost, stack |
| `pipeline/ensemble-amazon-stack` | `ensemble-amazon/stacking` | base(max_categories=100) → hash_encoding(24) → target → frequency → feature_selection(model_importance, 40) | LGBM/XGB/CatBoost/LogReg, stack |
| `pipeline/kconv-sklearn-diverse` | `kconv/sklearn-diverse` | base → target → frequency → feature_selection(model_importance, 30) | LGBM/HistGB/RandomForest, weighted |
| `pipeline/statsmaster-honest-cv` | `statsmaster/tabular` | base → target → frequency → feature_selection(variance, 50) | LGBM/XGB/CatBoost, weighted; adversarial threshold 0.8 |
| `pipeline/dptools-group-profile` | `dptools/row-features` | base → target → frequency → group_statistics(mean/std/count, max 8 numeric) → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/kozodoi-combination-frequency` | `kozodoi/tabular` | base(max_categories=100) → hash_encoding(32) → categorical_interaction(frequency, 4 pairs) → target → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/kaggle-ieee-missingness` | `kaggle-ieee/fraud-pipeline` | base(max_categories=100) → missingness(indicators/row/zero) → frequency → target(smoothing=20) → feature_selection(model_importance, 40) | LGBM/XGB, weighted |
| `pipeline/cdeotte-datetime-cyclical` | `cdeotte/tabular` | base → datetime_cyclical(month/dayofweek/dayofyear/hour) → target → frequency → feature_selection(model_importance, 40) | LGBM/XGB/CatBoost, weighted |
| `pipeline/ensemble-amazon-interaction-target` | `ensemble-amazon/stacking` | base(max_categories=100) → categorical_interaction(target, 4 pairs) → target → frequency → feature_selection(model_importance, 40) | LGBM/XGB/CatBoost/LogReg, stack |
| `pipeline/kozodoi-numeric-ratios` | `kozodoi/tabular` | base(max_categories=100) → numeric_interactions(6 pairs: product/ratio/difference) → target → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/kozodoi-time-profile` | `kozodoi/tabular` | base(max_categories=100) → time_series(lags 1/3, windows 3/7, mean/std/min/max) → numeric_interactions(4 pairs: ratio/difference) → target → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/kaggle-ieee-cluster-anomaly` | `kaggle-ieee/fraud-pipeline` | base(max_categories=100) → missingness(indicators/row/zero) → clustering(8 clusters, distances+labels) → anomaly(100 trees, auto contamination) → target(smoothing=20) → frequency → feature_selection(model_importance, 40) | LGBM/XGB, weighted |
| `pipeline/kaggle-ieee-time-profile` | `kaggle-ieee/fraud-pipeline` | base(max_categories=100) → time_series(lags 1/2, windows 3/7, mean/std) → datetime_cyclical(month/dayofweek/hour) → frequency → target(smoothing=20) → feature_selection(model_importance, 40) | LGBM/XGB, weighted |
| `pipeline/nyaggle-cluster-stack` | `nyaggle/oof-ensemble` | base(max_categories=100) → clustering(10 clusters, distances only) → target → frequency → feature_selection(model_importance, 40) | LGBM/XGB/CatBoost, stack |
| `pipeline/ensemble-amazon-numeric-cross` | `ensemble-amazon/stacking` | base(max_categories=100) → numeric_interactions(8 pairs: product/ratio) → categorical_interaction(frequency, 4 pairs) → target → frequency → feature_selection(model_importance, 40) | LGBM/XGB/CatBoost/LogReg, stack |
| `pipeline/sergeyshilin-anomaly-contracts` | `sergeyshilin/contracts` | base → anomaly(100 trees, contamination=0.05) → numeric_interactions(4 pairs: ratio/difference) → target → frequency → feature_selection(model_importance, 30) | LGBM/CatBoost/LogReg, mean; adversarial threshold 0.8 |
| `pipeline/dptools-cluster-profile` | `dptools/row-features` | base → clustering(6 clusters, distances+labels) → row_statistics(mean/std/min/max/median) → target → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/kconv-unsupervised-profile` | `kconv/sklearn-diverse` | base(scale=standard) → numeric_interactions(6 pairs: product/ratio/difference/sum) → clustering(8 clusters, distances only) → anomaly(80 trees) → feature_selection(variance, 50) | LGBM/HistGB/RandomForest, weighted |
| `pipeline/kconv-missing-row-profile` | `dptools/row-features` | base → missingness(indicators/row/zero) → row_statistics(mean/std/min/max/median) → numeric_interactions(6 pairs: product/ratio/difference) → target(smoothing=10) → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/kozodoi-hash-group-profile` | `kozodoi/tabular` | base(max_categories=100) → hash_encoding(32) → group_statistics(mean/std/count, max 8 numeric) → target(smoothing=10) → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/cdeotte-datetime-numeric` | `cdeotte/tabular` | base → datetime_cyclical(month/dayofweek/dayofyear) → numeric_interactions(6 pairs: product/ratio/difference/sum) → target(smoothing=10) → frequency → feature_selection(model_importance, 40) | LGBM/XGB/CatBoost, weighted |
| `pipeline/kaggle-ieee-time-group` | `kaggle-ieee/fraud-pipeline` | base(max_categories=100) → time_series(lags 1/2, windows 3/7, mean/std, elapsed+diff) → group_statistics(mean/std/count, max 8 numeric) → target(smoothing=20) → frequency → feature_selection(model_importance, 40) | LGBM/XGB, weighted |
| `pipeline/ensemble-amazon-interaction-group` | `ensemble-amazon/stacking` | base(max_categories=100) → categorical_interaction(target, 4 pairs) → group_statistics(mean/std/count, max 8 numeric) → target(smoothing=10) → frequency → feature_selection(model_importance, 40) | LGBM/XGB/CatBoost/LogReg, stack |
| `pipeline/knnfeat-cluster-anomaly` | `knnfeat/knn-features` | base → clustering(8 clusters, distances only) → anomaly(100 trees, auto contamination) → knn_class_distance(3 neighbors) → target(smoothing=10) → frequency → feature_selection(model_importance, 40) | LGBM/XGB, weighted; binary/multiclass only |
| `pipeline/kaggle-ieee-missing-cluster` | `kaggle-ieee/fraud-pipeline` | base(max_categories=100) → missingness(indicators/row/zero) → datetime_cyclical(month/dayofweek/hour) → clustering(8 clusters, distances only) → target(smoothing=20) → frequency → feature_selection(model_importance, 40) | LGBM/XGB, weighted |
| `pipeline/kozodoi-hash-missing-numeric` | `kozodoi/tabular` | base(max_categories=100) → hash_encoding(32) → missingness(indicators/row/zero) → numeric_interactions(6 pairs: product/ratio) → target(smoothing=10) → frequency → feature_selection(model_importance, 40) | LGBM/CatBoost, weighted |
| `pipeline/arnabbiswas-mutual-profile` | `arnabbiswas1/staged` | base → numeric_interactions(6 pairs: product/ratio/difference) → row_statistics(mean/std/min/max) → target(smoothing=10) → frequency → feature_selection(mutual_info, 30) | LGBM/XGB, weighted |
| `pipeline/kconv-modulo-anomaly` | `kconv/sklearn-diverse` | base(scale=standard) → modulo(moduli 10/100, steps 5) → anomaly(80 trees, auto contamination) → numeric_interactions(4 pairs: ratio/difference) → target(smoothing=10) → frequency → feature_selection(variance, 50) | LGBM/HistGB/RandomForest, weighted |

## Партия 2026-11-11: профили, хеши и кросс-блоки

Десять новых executable pipelines закрывают комбинации, которых не было в
каталоге: missingness + row profile, hash + group statistics, datetime +
numeric interactions, time series + group statistics, categorical interaction +
group statistics, kNN + cluster/anomaly, missingness + datetime + clustering,
hash + missingness + numeric, mutual-information profile и modulo + anomaly.

- `pipeline/kconv-missing-row-profile` — missingness, row statistics и numeric
  interactions перед target/frequency encoding.
- `pipeline/kozodoi-hash-group-profile` — компактный hash для высокой
  кардинальности плюс group mean/std/count.
- `pipeline/cdeotte-datetime-numeric` — циклический календарь и произведения/
  отношения числовых колонок.
- `pipeline/kaggle-ieee-time-group` — past-only time series и group aggregates
  в одном pipeline.
- `pipeline/ensemble-amazon-interaction-group` — target-encoded categorical
  interactions и group aggregates для stack.
- `pipeline/knnfeat-cluster-anomaly` — cluster distances, anomaly score и
  class-conditional kNN для classification.
- `pipeline/kaggle-ieee-missing-cluster` — missingness, datetime cyclicity и
  cluster distances для широких таблиц.
- `pipeline/kozodoi-hash-missing-numeric` — hash, missingness и numeric ratios
  перед target/frequency encoding.
- `pipeline/arnabbiswas-mutual-profile` — row profile и mutual-information
  selection для компактного набора признаков.
- `pipeline/kconv-modulo-anomaly` — modulo/divisibility, anomaly score и
  numeric ratios с variance selection.

## Новые техники партии 2026-10-26

- **Numeric interactions** — label-free произведения, отношения, разности и
  суммы пар numeric-колонок. Блок ограничен `max_pairs`, устойчив к нулевому
  знаменателю и клипует выбросы; применяется в
  `pipeline/kozodoi-numeric-ratios`, `pipeline/kozodoi-time-profile`,
  `pipeline/ensemble-amazon-numeric-cross`,
  `pipeline/sergeyshilin-anomaly-contracts` и
  `pipeline/kconv-unsupervised-profile`.
- **Clustering** — MiniBatchKMeans на outer training fold даёт расстояния до
  центроидов и/или метку кластера. Это unsupervised признаки, которые не
  читают target и не зависят от validation/test; применяется в
  `pipeline/kaggle-ieee-cluster-anomaly`, `pipeline/nyaggle-cluster-stack`,
  `pipeline/dptools-cluster-profile` и
  `pipeline/kconv-unsupervised-profile`.
- **Anomaly** — IsolationForest на outer training fold добавляет
  `anomaly__score` и опциональный `anomaly__flag`. `contamination: auto`
  оставляет порог на усмотрение модели, явное значение фиксирует долю
  выбросов; применяется в `pipeline/kaggle-ieee-cluster-anomaly`,
  `pipeline/sergeyshilin-anomaly-contracts` и
  `pipeline/kconv-unsupervised-profile`.
- **Time series** — past-only лаги, trailing mean/std/min/max, elapsed и diff
  по времени внутри группы. History-tail ограничен `max_history_rows`, а
  validation/test видят только прошлые строки; применяется в
  `pipeline/kozodoi-time-profile` и `pipeline/kaggle-ieee-time-profile`.

## Отличия от preset

Preset использует legacy feature flags (`features.target_encode`,
`features.row_statistics`, ...). Pipeline использует `pipeline.blocks` как
единственный источник истины: порядок и наличие блоков видны в конфиге,
CLI/API и UI. Preset и pipeline могут иметь одинаковые defaults — это разные
`kind`, и audit не считает это дубликатом executable pipeline.

## LLM fold plan

Обычный LLM-план материализует колонки до `build_config`. Fold-aware план
нельзя материализовать заранее, поэтому после approval:

- preset: заполняется `features.fold_plan`;
- pipeline: добавляется/обновляется блок `fold_plan` и синхронизируется
  `features.fold_plan`. Если в pipeline уже есть `group_statistics`, approval
  возвращает `RecipeError`: выберите один механизм fold-агрегатов.

Approval, хранение `plan_id` и `05_report/llm_features.json` не меняются.

## Contracts и honest validation

Контракты и «честная» валидация выражаются декларативно в `defaults`, без
Python-плагинов:

- `defaults.validation.adversarial: true` и
  `adversarial_threshold` включают проверку train/test drift до обучения;
- `defaults.validation.n_splits`, `shuffle`, `group_col`, `time_col` и `gap`
  задают CV-контракт;
- `defaults.ensemble.method: mean` фиксирует прозрачный blend без подгонки
  весов;
- `pipeline/sergeyshilin-contracts` и `pipeline/statsmaster-honest-cv`
  используют эти настройки вместе с `feature_selection`, поэтому рецепт
  остаётся полностью декларативным.

## Ресурсы и CUDA

Feature-блоки CPU-only. `resource_hints` в API/UI — грубая подсказка
(`cpu`, `ram`, `cpu-heavy`, `classification-only`), а не ограничение;
`feature_selection(model_importance)` добавляет `cpu-heavy`.
CUDA-профили применяются только к поддерживаемым model adapters; TabPFN
остаётся optional, tabular-only и не выбирается автоматически новыми
pipelines. UI-лимит 8 000 train rows для TabPFN сохраняется.

## Добавление pipeline

Подробный гайд: [adding-conveyors.ru.md](adding-conveyors.ru.md) (RU) и
[adding-conveyors.md](adding-conveyors.md) (EN).

Короткая версия:

1. Создайте `src/kconv/recipes/<author>__<name>.yaml` с `kind: pipeline` и
   `derived_from`.
2. Начните `pipeline.blocks` с `base`.
3. Запустите `uv run kconv recipes audit --strict`.
4. Соберите проект на маленьком датасете:
   `uv run kconv run --data data --target target --recipe <id> --no-run`.
5. Запустите targeted tests:

```powershell
uv run pytest tests/test_recipes.py tests/test_recipe_audit.py tests/test_project.py -q
uv run pytest tests/test_pipeline_matrix.py -k "<id>" -q
```

`docs/recipes.md` описывает каталог и YAML-схему; этот файл — исполнение и
ограничения. Гайд добавляет чек-лист, правила блоков, атрибуцию и разбор
частых ошибок.
