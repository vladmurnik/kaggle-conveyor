# Recipes: presets и pipelines

Recipe — декларативное описание запуска. Чужой код не исполняется. Есть два
вида:

- **preset** (`kind: preset`, по умолчанию) — исторический набор defaults:
  validation, features, models, ensemble. Feature flags применяются как раньше.
- **pipeline** (`kind: pipeline`) — исполняемый список feature-блоков
  (`pipeline.blocks`) плюс defaults. Порядок блоков фиксирован, блоки
  проверяются до запуска.

Все 16 исходных recipes сохранены как presets. Двадцать девять executable
pipelines выведены из них и указывают `derived_from`; всего в каталоге 45
recipes. Preset с тем же поведением, что и pipeline, — это осознанная
историческая вариация; два pipeline с одинаковой подписью запрещены в
strict-аудите.

## Список и аудит

```powershell
uv run kconv recipes
uv run kconv recipes --kind pipeline
uv run kconv recipes --kind preset
uv run kconv recipes --group ensemble
uv run kconv recipes --json
uv run kconv recipes show pipeline/knnfeat-class-distance
uv run kconv recipes audit
uv run kconv recipes audit --strict --json
```

`recipes audit` строит нормализованную подпись из `defaults`, `kind` и
`pipeline.blocks`. Он показывает исторические дубли presets и падает с
`RecipeError` при `--strict`, если два **executable** pipeline имеют
одинаковую подпись.

Встроенные группы presets:

| Группа | Recipes |
| --- | --- |
| `baseline` | `pesnik/baseline` |
| `ensemble` | `cdeotte/tabular`, `statsmaster/tabular`, `minerva/stacking`, `kconv/sklearn-diverse`, `kconv/linear-stack`, `nyaggle/oof-ensemble`, `ensemble-amazon/stacking` |
| `engineering` | `kozodoi/tabular`, `congmnguyen/leakage-safe`, `kaggler/target-encoding`, `dptools/row-features`, `knnfeat/knn-features`, `kaggle-ieee/fraud-pipeline` |
| `architecture` | `sergeyshilin/contracts`, `arnabbiswas1/staged` |

Пользовательский recipe без `group` получает `general`.

## Встроенные recipes и лицензии (16)

| id | Источник | Лицензия | Что задаёт |
| --- | --- | --- | --- |
| `pesnik/baseline` | pesnik/kaggle-starter-kit | MIT | Минимальный baseline, LGBM |
| `cdeotte/tabular` | cdeotte/KGMON-Playbook-2026 | NO LICENSE — ideas only | LGBM/XGB/CatBoost, weighted |
| `statsmaster/tabular` | StatsMaster/kaggle-grandmaster-templates | NO LICENSE — ideas only | Honest CV, три GBDT |
| `minerva/stacking` | minerva-ml/open-solution-home-credit | MIT | Разные семейства, OOF stacking |
| `kozodoi/tabular` | kozodoi/Kaggle_Home_Credit | MIT | High-cardinality + blend |
| `congmnguyen/leakage-safe` | congmnguyen/tabular-ml-lab | MIT | Аккуратный CV, LGBM/XGB |
| `sergeyshilin/contracts` | sergeyshilin/kaggle-competitions-framework | MIT | Контракты и mean-бленд |
| `arnabbiswas1/staged` | arnabbiswas1/kaggle_pipeline_tps_aug_22 | Apache-2.0 | Стадии, HPO-профиль |
| `nyaggle/oof-ensemble` | nyanp/nyaggle | MIT | OOF-эксперименты и stack из GBDT |
| `kaggler/target-encoding` | jeongyoonlee/Kaggler | MIT | Target/frequency encoding и weighted blend |
| `dptools/row-features` | kozodoi/dptools | MIT | Row-wise mean/std/min/max/median |
| `knnfeat/knn-features` | upura/knnFeat | MIT | Class-conditional kNN distance, fit-in-fold |
| `ensemble-amazon/stacking` | kaz-Anova/ensemble_amazon | Apache-2.0 | Meta-stacking над GBDT/linear |
| `kaggle-ieee/fraud-pipeline` | RobMulla/kaggle-ieee-fraud-detection | Apache-2.0 | Frequency encoding для широких таблиц |
| `kconv/sklearn-diverse` | local (kconv) | MIT | LGBM/HistGB + random forest, weighted |
| `kconv/linear-stack` | local (kconv) | MIT | LogReg/Ridge + HistGB, stack |

`knnfeat/knn-features` объявляет только `binary` и `multiclass`: это
class-conditional признаки, для регрессии они не определены. Остальные
recipes поддерживают все три задачи; модели, несовместимые с задачей,
отфильтровываются при сборке конфига.

`knnfeat/knn-features` — самый дорогой пресет по CPU/памяти: расстояния
считаются для каждой пары train/reference внутри каждого outer fold.
На больших таблицах уменьшайте `features.knn_features.folds` и
`max_reference_rows` или запускайте рецепт на стратифицированном срезе;
`max_reference_rows` ограничивает reference-set на класс и не даёт памяти
расти квадратично.

Для источников без LICENSE или с GPL код не копируется. Recipe — это
переписанная методология и ссылка на автора; см. поле `attribution`.

## Executable pipelines (29)

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

`columns: auto` в target/frequency/hash-блоках разрешается при создании проекта:
kconv читает sample train, исключает ID/ignore и слишком редкие/уникальные
колонки, а результат записывает обратно в `config.yaml`. Поэтому запуск
воспроизводим и не зависит от эвристики во время fit.

`knn_class_distance` остаётся CPU-only, classification-only и ограничивает
reference-set через `max_reference_rows`; CUDA-профили влияют только на
поддерживаемые model adapters, не на feature-блоки.

`pipeline/arnabbiswas-staged` использует `feature_selection(mutual_info)`.
`pipeline/kozodoi-high-cardinality` и `pipeline/ensemble-amazon-stack`
добавляют label-free `hash_encoding` для высококардинальных категорий, а
`pipeline/sergeyshilin-contracts`, `pipeline/nyaggle-oof-ensemble`,
`pipeline/kconv-sklearn-diverse` и `pipeline/statsmaster-honest-cv`
используют `feature_selection(model_importance)` или `variance` и явные
validation/ensemble defaults. `congmnguyen/leakage-safe` остаётся preset-only:
его поведение уже покрыто `kaggler-target-frequency` и
`kaggle-ieee-frequency`, поэтому отдельный pipeline не добавляет новой идеи.
`hash_encoding` label-free и детерминирован; `model_importance` fit-ится
внутри outer training fold. Оба блока не запускают чужой Python-код.

### Партия 2026-10-25: group, interaction, missingness и cyclical

Пять новых pipelines добавляют четыре декларативных семейства, которых не
было в первой волне:

- `group_statistics` компилируется в тот же reviewed `fold_plan`, который
  использует LLM-план: group mean/std/count считаются только на outer
  training fold, а unseen-группы получают fallback без утечки;
- `categorical_interaction` строит до `max_pairs` попарных комбинаций
  категориальных колонок и кодирует их frequency (label-free) или target
  (inner-OOF, как обычный target encoder);
- `missingness` добавляет row-wise missing count, missing indicators и
  numeric zero count; все признаки label-free;
- `datetime_cyclical` добавляет sin/cos для месяца, дня недели, дня года и
  часа; это label-free представление уже распознанных datetime-колонок.

`pipeline/dptools-group-profile` использует `group_statistics`;
`pipeline/kozodoi-combination-frequency` — frequency-комбинации;
`pipeline/kaggle-ieee-missingness` — missingness;
`pipeline/cdeotte-datetime-cyclical` — cyclical datetime;
`pipeline/ensemble-amazon-interaction-target` — target-комбинации и stack.
На датасете без datetime `datetime_cyclical(columns: auto)` не добавляет
колонок, но pipeline остаётся исполняемым. `group_statistics` и `fold_plan`
взаимоисключающи: оба определяют fold-агрегаты.

### Партия 2026-10-26: numeric, clustering, anomaly и time series

Девять pipelines добавляют четыре декларативных семейства:

- `numeric_interactions` — label-free произведения, отношения, разности и
  суммы до `max_pairs` пар numeric-колонок; деление на ноль даёт 0, выходы
  клипуются;
- `clustering` — MiniBatchKMeans на outer training fold: расстояния до
  центроидов и/или метка кластера, без чтения target;
- `anomaly` — IsolationForest на outer training fold: `anomaly__score` и
  опциональный `anomaly__flag`, `contamination: auto` или `(0, 0.5]`;
- `time_series` — past-only лаги, trailing mean/std/min/max, elapsed и diff
  внутри группы; history-tail ограничен `max_history_rows`.

`pipeline/kozodoi-numeric-ratios` и `pipeline/ensemble-amazon-numeric-cross`
строят numeric-пары; `pipeline/kozodoi-time-profile` и
`pipeline/kaggle-ieee-time-profile` — time-series профиль;
`pipeline/kaggle-ieee-cluster-anomaly`, `pipeline/nyaggle-cluster-stack`,
`pipeline/dptools-cluster-profile` и `pipeline/kconv-unsupervised-profile`
используют clustering; `pipeline/sergeyshilin-anomaly-contracts` и
`pipeline/kconv-unsupervised-profile` — anomaly. `clustering` и `anomaly`
требуют numeric-колонок, а `time_series` — datetime-колонку: без неё запуск
отклоняется до обучения, а не молча пропускается. Все четыре блока
fit-ятся внутри outer fold или label-free, поэтому не подглядывают в
validation/test.

### Партия 2026-10-23: новые permissive-источники

Добавлены шесть permissive-источников из локального `kaggle-toolkit` без
вендоринга их кода:

| Источник | Лицензия | Что адаптировано |
| --- | --- | --- |
| `nyanp/nyaggle` | MIT | OOF-стек и эксперимент-трекинг → `nyaggle/oof-ensemble` |
| `jeongyoonlee/Kaggler` | MIT | target/frequency encoding → `kaggler/target-encoding` |
| `kozodoi/dptools` | MIT | row-wise агрегаты → `dptools/row-features` |
| `upura/knnFeat` | MIT | class-conditional kNN distance → `knnfeat/knn-features` |
| `kaz-Anova/ensemble_amazon` | Apache-2.0 | weighted/rank/meta-stacking → `ensemble-amazon/stacking` |
| `RobMulla/kaggle-ieee-fraud-detection` | Apache-2.0 | frequency/wide-table workflow → `kaggle-ieee/fraud-pipeline` |

Исходники остаются в `kaggle-toolkit` как read-only справочный корпус;
`kconv` содержит только оригинальные рецепты и адаптеры.

## Схема YAML

```yaml
id: my-team/tabular          # уникальный id, обычно author/name
kind: preset                 # preset (default) | pipeline
# derived_from: other/recipe # optional lineage for adapted pipelines
title: My tabular recipe
author: My Team
source: https://example.com/repo
license: MIT
group: engineering           # baseline|ensemble|engineering|architecture|general
tasks: [binary, multiclass, regression]
best_for: "Short human description"
defaults:
  validation:
    strategy: auto
    n_splits: 5
    shuffle: true
    random_state: 42
  features:
    target_encode: []
    frequency_encode: []
    auto_encode: true
    scale: auto
    impute: median
    drop_constant: true
    max_categories: 50
    # opt-in feature families (see docs/config.md)
    row_statistics: false
    row_statistics_stats: [mean, std, min, max]
    modulo_features: false
    modulo_moduli: [10, 100]
    modulo_steps: [5]
    knn_features:
      enabled: false
      neighbors: 3
      folds: 5
      metric: euclidean
  models: [lgbm, xgb]
  model_params:
    lgbm:
      n_estimators: 800
      learning_rate: 0.03
  hpo:
    enabled: false
    trials: 8
    timeout_sec: 600
    inner_size: 0.2
  ensemble:
    method: weighted
    fallback: best_single
    min_improvement: 0.0
attribution: "Methodology only; no source code copied."
```

Executable pipeline добавляет блоки. `base` обязателен, должен быть первым и
встречаться ровно один раз; остальные блоки в v1 тоже singleton:

```yaml
id: my-team/pipeline
kind: pipeline
derived_from: my-team/tabular
title: My executable pipeline
author: My Team
source: local
license: MIT
group: engineering
tasks: [binary, multiclass, regression]
best_for: "Target/frequency encoding plus row statistics"
defaults:
  models: [lgbm]
  ensemble: {method: weighted, fallback: best_single}
pipeline:
  blocks:
    - type: base
      params: {scale: auto, max_categories: 50}
    - type: target_encoding
      params: {columns: auto, smoothing: 10}
    - type: frequency_encoding
      params: {columns: auto}
    - type: row_statistics
      params: {stats: [mean, std, min, max]}
    - type: modulo
      params: {moduli: [10, 100], steps: [5]}
    - type: fold_plan
      params: {plan: {features: []}}
    - type: knn_class_distance
      params: {neighbors: 3, folds: 5, max_reference_rows: 50000}
attribution: "Original work."
```

Доступные типы: `base`, `target_encoding`, `frequency_encoding`,
`hash_encoding`, `group_statistics`, `categorical_interaction`, `missingness`,
`datetime_cyclical`, `numeric_interactions`, `clustering`, `anomaly`,
`time_series`, `row_statistics`, `modulo`, `fold_plan`, `feature_selection`,
`knn_class_distance`. Неизвестный тип
или неверный порядок даёт `RecipeError` до запуска. `fold_plan` обычно
добавляется после утверждения LLM-плана, а не пишется вручную;
`group_statistics` — декларативная альтернатива ручному fold-плану.

Обязательные поля: `id`, `title`, `author`, `source`, `license`, `best_for`,
`attribution`. `defaults` необязателен, но должен быть валиден по схеме
`kconv.config`. `kind: pipeline` требует непустой `pipeline.blocks`;
`kind: preset` не может объявлять `pipeline`.

## Куда класть

Поиск идёт в таком порядке:

1. `recipes/<id с / заменённым на __>.yaml`;
2. `recipes/<последний сегмент id>.yaml`;
3. `recipes/<id>/recipe.yaml`;
4. `~/.kconv/recipes/...`;
5. встроенные recipes пакета.

Если имя файла не совпадает с id, kconv также просматривает YAML в
`recipes/` и `~/.kconv/recipes/` (включая вложенные `**/recipe.yaml`) и
сравнивает поле `id`.

Пользовательский recipe с тем же `id`, что и встроенный, побеждает в
`load_recipe()` и в списке `kconv recipes`. Приоритет: текущая `recipes/`,
затем `~/.kconv/recipes/`, затем пакет `kconv`. Если в YAML неизвестное или
пустое значение `tasks`, загрузка завершается понятной `RecipeError`.

Пример:

```text
recipes/
└── my-team__tabular.yaml
```

```yaml
id: my-team/tabular
title: My tuned LGBM
author: Me
source: local
license: MIT
group: engineering
best_for: "Fast LGBM with target encoding"
defaults:
  models: [lgbm]
attribution: "Original work."
```

```powershell
uv run kconv run --data data --target target --recipe my-team/tabular --no-run
```

## Проверка

```powershell
uv run kconv recipes show my-team/tabular
uv run kconv recipes audit --strict
uv run kconv recipes --json
uv run pytest tests/test_recipes.py tests/test_recipe_audit.py -q
```

`show` печатает kind, lineage, blocks, validation, ensemble, модели,
короткую подпись и attribution. Если YAML невалиден, kconv укажет поле и
ожидаемый тип. `--json` всегда ASCII-safe и пригоден для пайпов.

## Как recipe применяется к конфигу

При `kconv run --data ... --recipe ...`:

1. данные анализируются, определяются task/metric;
2. проверяется, что recipe поддерживает задачу;
3. модели recipe фильтруются по совместимости с задачей (явный `--models`
   с несовместимой моделью даёт ошибку); если после фильтрации не осталось ни
   одной совместимой установленной модели, kconv не подставляет другую модель
   молча, а возвращает `RecipeError`;
4. defaults копируются в `config.yaml`; `validation` и `hpo.inner_size`
   сохраняются, а стратегия переопределяет только `n_splits`, `seeds`,
   `hpo.enabled` и `hpo.trials`;
5. стратегия (`fast`/`standard`/`deep`) задаёт `n_splits`, `seeds` и
   включает/выключает HPO;
6. для preset auto-encoding может дополнить пустые
   `target_encode`/`frequency_encode`;
7. для pipeline `pipeline.blocks` копируются в
   `features.feature_blocks`, legacy-флаги синхронизируются только для
   интроспекции, а `columns: auto` разрешаются и записываются обратно;
8. `recipe_kind` сохраняется в конфиге, чтобы отчёт и UI различали preset и
   pipeline;
9. `group_col`/`time_col` проверяются по колонкам train до запуска.

После создания `config.yaml` recipe больше не влияет на запуск: все
фактические параметры, включая `feature_blocks`, зафиксированы в конфиге и
хэшируются для `--resume`. Проверенный LLM fold-aware план добавляется как
блок `fold_plan` в pipeline или как `features.fold_plan` в preset.
