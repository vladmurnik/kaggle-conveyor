# Добавление нового конвейера

В kconv **конвейер** — это декларативный recipe, который задаёт validation,
feature-блоки, модели, HPO и ансамблирование. Recipe — это данные, а не код:
kconv не импортирует и не исполняет Python из recipe. Ниже — практический
гайд по добавлению нового pipeline.

> Готовый пример для копирования:
> [`src/kconv/recipes/pipeline__pesnik-clean.yaml`](../src/kconv/recipes/pipeline__pesnik-clean.yaml).
> Полная YAML-схема — в [recipes.md](recipes.md), семантика блоков — в
> [pipelines.md](pipelines.md).

## 1. Выберите вид recipe

| Вид | Когда использовать |
| --- | --- |
| `pipeline` | Нужен явный упорядоченный feature graph. Рекомендуемый вид для новой работы. |
| `preset` | Нужны только legacy feature flags (`features.target_encode`, `features.row_statistics`, ...). Presets существуют для совместимости и исторической атрибуции. |

У pipeline две части:

- `defaults` — validation, модели, параметры моделей, HPO, ensemble;
- `pipeline.blocks` — упорядоченный feature graph.

После сборки проекта блоки копируются в `config.yaml` как
`features.feature_blocks`, и сам recipe больше не читается во время
обучения. Config hash включает разрешённые блоки, поэтому `--resume` не
может незаметно изменить feature graph.

## 2. Выберите уникальный id и атрибуцию

Для новых executable pipelines используйте
`pipeline/<author>-<name>`. Id должен быть стабильным: переименование ломает
сохранённые конфиги и ссылки в документации. Если поведение меняется,
добавьте новый id.

Обязательные метаданные:

```yaml
id: pipeline/my-team-clean
kind: pipeline
derived_from: pesnik/baseline       # опциональная линия происхождения
title: My team clean baseline
author: My Team
source: https://github.com/my-team/my-repo
license: MIT
group: baseline                     # baseline|ensemble|engineering|architecture|general
tasks: [binary, multiclass, regression]
best_for: "Быстрый честный baseline с target/frequency encoding и одним LightGBM."
attribution: "Методика адаптирована из ...; реализация — fit-in-fold код kconv."
```

`derived_from` добавляйте только если recipe действительно выведен из
другого recipe. `source` и `license` должны указывать на источник методики.
Не копируйте код из репозиториев с несовместимой лицензией; см.
[Атрибуция и лицензии](#8-атрибуция-и-лицензии).

## 3. Начните с минимального pipeline

```yaml
id: pipeline/my-team-clean
kind: pipeline
derived_from: pesnik/baseline
title: My team clean baseline
author: My Team
source: https://github.com/my-team/my-repo
license: MIT
group: baseline
tasks: [binary, multiclass, regression]
best_for: "Быстрый честный baseline с target/frequency encoding и одним LightGBM."

defaults:
  validation:
    strategy: auto
    n_splits: 5
  models: [lgbm]
  hpo:
    enabled: false
    trials: 20
    timeout_sec: 600
  ensemble:
    method: weighted
    fallback: best_single

pipeline:
  blocks:
    - type: base
    - type: target_encoding
      params: {columns: auto, smoothing: 10}
    - type: frequency_encoding
      params: {columns: auto}

attribution: "Структура адаптирована из публичного starter-kit; feature-логика — fit-in-fold реализация kconv."
```

Это намеренно маленький пример. Добавляйте одно семейство признаков за раз и
измеряйте эффект на OOF; длинный pipeline не становится лучше автоматически.

## 4. Правила блоков

`pipeline.blocks` должен удовлетворять всем правилам ещё до обучения:

- ровно один `base`, первым и включённым;
- каждый тип блока — singleton в v1; дубликаты отклоняются;
- неизвестные типы и неизвестные параметры отклоняются;
- `feature_selection` должен быть последним включённым блоком;
- `group_statistics` и `fold_plan` взаимоисключающи;
- `knn_class_distance` — только classification и только при наличии numeric
  колонок;
- `time_series` требует datetime-колонку (`time_col: auto` или явную);
- `clustering` должен включить `include_distances` или `include_labels`;
- явные `columns` должны существовать в датасете или быть `auto`;
- `hash_encoding.n_features` — целое в `1..1024`;
- `time_series.lags` — в `1..100`, `windows` — в `2..100`, `stats` —
  подмножество `mean/std/min/max`.

### Справочник блоков

| Блок | Ключевые параметры | Что делает |
| --- | --- | --- |
| `base` | `scale`, `impute`, `max_categories`, `auto_encode`, `drop_constant` | Строит numeric/categorical матрицу. Обязателен, первый, ровно один. |
| `target_encoding` | `columns`, `smoothing` | Inner-OOF target encoding; fit только на training fold. |
| `frequency_encoding` | `columns` | Частоты категорий, посчитанные на training fold. |
| `hash_encoding` | `columns`, `n_features` | Стабильные label-free hash-бакеты для высококардинальных категорий. |
| `group_statistics` | `group_col`, `columns`, `stats`, `max_columns` | Fit-in-fold group mean/std/count; компилируется в reviewed `fold_plan`. |
| `categorical_interaction` | `columns`, `max_pairs`, `encoding` | Попарные категориальные комбинации; `frequency` label-free, `target` — inner-OOF. |
| `missingness` | `columns`, `indicators`, `row_count`, `zero_count` | Индикаторы пропусков, row-wise missing count, numeric zero count. |
| `datetime_cyclical` | `columns`, `periods` | Label-free sin/cos для month/dayofweek/dayofyear/hour. |
| `numeric_interactions` | `columns`, `max_pairs`, `operations`, `clip_value` | Произведения, отношения, разности, суммы numeric-пар; деление на ноль даёт 0. |
| `clustering` | `columns`, `n_clusters`, `max_iter`, `include_distances`, `include_labels`, `max_fit_rows` | Fit-in-fold MiniBatchKMeans: расстояния и/или метки. |
| `anomaly` | `columns`, `n_estimators`, `contamination`, `include_flag` | Fit-in-fold IsolationForest score и опциональный flag. |
| `time_series` | `time_col`, `group_col`, `columns`, `lags`, `windows`, `stats`, `include_elapsed`, `include_diff`, `max_history_rows` | Past-only лаги, trailing-статистики, elapsed/diff времени. |
| `row_statistics` | `stats` | Row-wise mean/std/min/max/median по numeric-колонкам. |
| `modulo` | `moduli`, `steps` | Modulo/quotient views для identifier-like numeric. |
| `fold_plan` | `plan` | Утверждённый fold-aware LLM-план; выражения проверяются. |
| `feature_selection` | `method`, `max_features`, `min_variance` | Финальный selector: variance, mutual information или model importance. |
| `knn_class_distance` | `columns`, `neighbors`, `folds`, `metric`, `max_reference_rows` | Class-conditional kNN distance; CPU-only, classification-only. |

`columns: auto` разрешается по sample train при сборке проекта, а результат
записывается обратно в `config.yaml`. Это делает запуск воспроизводимым и не
зависит от эвристики во время fit.

Все fitted-преобразования обучаются внутри outer training fold. Validation и
test не влияют на fitted state. Единственное исключение —
`feature_selection(method=model_importance)`: небольшая ExtraTrees-модель
обучается только на outer training fold.

## 5. Куда положить файл

Порядок поиска recipe по id:

1. `recipes/<id с / заменённым на __>.yaml`
2. `recipes/<последний сегмент>.yaml`
3. `recipes/<id>/recipe.yaml`
4. `~/.kconv/recipes/...`
5. встроенные recipes в `src/kconv/recipes/`

Пользовательский recipe переопределяет встроенный с тем же id. Для вклада во
встроенный каталог добавьте YAML в `src/kconv/recipes/` с именем
`<author>__<name>.yaml` (например, `pipeline__my-team-clean.yaml`).

## 6. Проверка до обучения

```bash
uv run kconv recipes show pipeline/my-team-clean
uv run kconv recipes audit --strict
uv run kconv recipes --json
```

`recipes audit --strict` нормализует подпись каждого recipe из `defaults`,
`kind` и `pipeline.blocks`. Команда падает, если два executable pipeline
имеют одинаковую подпись. Исторические дубликаты presets показываются, но
ошибкой не считаются.

Соберите конфиг без обучения:

```bash
uv run kconv run \
  --data data \
  --target target \
  --recipe pipeline/my-team-clean \
  --no-run \
  --name recipe-smoke
```

Проверьте `competitions/recipe-smoke/config.yaml`: в
`features.feature_blocks` должны быть разрешённые блоки, а `recipe_kind` —
`pipeline`.

## 7. Smoke-тест и тесты

Запустите небольшой end-to-end прогон:

```bash
uv run kconv run \
  --data data \
  --target target \
  --recipe pipeline/my-team-clean \
  --strategy fast \
  --models lgbm \
  --name recipe-smoke
```

Проверьте:

- `05_report/report.md` существует и содержит OOF-метрики;
- `04_submit/submission.csv` имеет ожидаемые колонки и число строк;
- в submission нет NaN;
- имена feature-блоков видны в метаданных запуска.

Целевые тесты:

```bash
uv run pytest tests/test_recipes.py tests/test_recipe_audit.py tests/test_project.py -q
uv run pytest tests/test_pipeline_matrix.py -k "my-team-clean" -q
```

Добавьте хотя бы одну проверку в `tests/test_recipes.py`:

```python
pipeline = load_recipe("pipeline/my-team-clean")
assert pipeline.kind == "pipeline"
assert pipeline.derived_from == "pesnik/baseline"
assert [block.type for block in pipeline.pipeline.blocks] == [
    "base",
    "target_encoding",
    "frequency_encoding",
]
```

Если recipe использует семейство блоков, которого ещё нет в matrix-тестах,
добавьте кейс в `tests/test_pipeline_matrix.py`, чтобы блок прошёл end-to-end.

## 8. Атрибуция и лицензии

- Используйте только MIT, Apache-2.0, BSD или совместимую методику.
- Для источника **без лицензии** или с GPL используйте только идеи и ссылки;
  не копируйте код, комментарии или существенные таблицы параметров.
- Заполните `source`, `license`, `author`, `attribution`.
- Обновите таблицы recipes в [recipes.md](recipes.md) и
  [pipelines.md](pipelines.md), добавьте источник в [NOTICE.md](../NOTICE.md).
- Не добавляйте в коммит исходный репозиторий, датасеты и benchmark-выходы.

## 9. Добавление нового типа блока

Новый тип YAML-блока — это изменение кода, а не только recipe. Минимум:

1. Добавьте тип в `_ALLOWED_BLOCK_TYPES` в
   `src/kconv/recipes/__init__.py`.
2. Реализуйте валидацию и fit/transform в
   `src/kconv/features/blocks.py` и `src/kconv/features/pipeline.py`.
3. Убедитесь, что преобразование обучается только на outer training fold и
   корректно обрабатывает unseen-категории/колонки на validation и test.
4. Добавьте unit-тесты на shape, leakage, unseen values и неверные параметры.
5. Добавьте pipeline recipe, который использует блок.
6. Опишите блок в [pipelines.md](pipelines.md) и [recipes.md](recipes.md).

Если блок не переносится в standalone inference bundle, обновите
`runtime_unsupported_blocks()`, чтобы ограничение было явно видно.

## 10. Чек-лист

- [ ] id уникален и стабилен;
- [ ] `kind: pipeline` и непустой `pipeline.blocks`;
- [ ] ровно один `base`, первым;
- [ ] нет дубликатов типов блоков;
- [ ] явные колонки существуют или используется `auto`;
- [ ] `feature_selection` последний;
- [ ] список tasks соответствует моделям и блокам;
- [ ] `kconv recipes audit --strict` проходит;
- [ ] `--no-run` конфиг содержит разрешённые блоки;
- [ ] fast smoke run завершается и пишет валидный submission;
- [ ] целевые тесты проходят;
- [ ] docs, `NOTICE.md` и атрибуция обновлены;
- [ ] в репозиторий не попали данные, credentials, runs и личные отчёты.

## Частые ошибки

| Ошибка | Вероятная причина |
| --- | --- |
| `pipeline.blocks must start with the 'base' block` | `base` отсутствует или стоит не первым. |
| `pipeline.blocks must contain 'base' exactly once` | Дубликат `base`. |
| `pipeline.blocks must contain each singleton block once` | Дубликат типа блока. |
| `unknown block type` | Опечатка или блок ещё не реализован в этой версии. |
| `feature_selection must be the last enabled block` | После selector есть другой блок. |
| `knn_class_distance` отклонён | Regression или нет numeric-колонок. |
| `columns ... not found` | Явный список колонок не совпадает с датасетом. |
| duplicate signature in strict audit | Два pipeline дают одинаковые defaults и blocks. |

## Когда добавлять preset вместо pipeline

Preset стоит добавлять только для сохранения исторической конфигурации с
feature flags или если recipe намеренно не выражается графом блоков. Для
новой методики предпочитайте pipeline: порядок признаков явный, audit
сравнивает подписи, а конфиг сохраняет разрешённый граф.
