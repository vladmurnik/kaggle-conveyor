# Тесты на реальном датасете (10%)

`tests/test_real_data_10pct.py` — opt-in smoke-набор на локальном
`data/train.csv`, `data/test.csv` и `data/sample_submission.csv`.

Он не входит в обычный быстрый прогон: тесты пропускаются, пока не задана
переменная `KCONV_REAL_DATA_TESTS=1`. Так `uv run pytest -q` остаётся быстрым,
а проверка реальных адаптеров и pipeline запускается явно.

## Что проверяется

- Стратифицированные **10% train** (target сохраняет оба класса), 10% test и
  соответствующий sample_submission. Отдельный тест сравнивает доли классов в
  полном train и в 10% выборке.
- Все модели из каталога, совместимые с задачей `binary`, проходят
  `fit`/`predict_proba`/`predict` на 10% выборке.
  `tabpfn` пропускается: 10% датасета больше безопасного лимита 8 000 строк;
  для него остаются отдельные тесты `tests/test_tabpfn.py`. Regression-only
  адаптеры не выбираются для этого target; они покрыты обычными unit-тестами.
- Все 65 встроенных recipes (16 presets + 49 pipelines), совместимых с binary,
  запускаются через настоящий `run_pipeline` до `report` на 10% train/test:
  `ingest → validate → hpo → train → blend → submit → report`.
  Для скорости smoke-прогон использует 2 folds и `adversarial=False`;
  остальные стадии, submission и отчёт проверяются полностью.
  `knnfeat/knn-features` на 10% реального датасета заметно дороже остальных
  recipes; для точечной проверки используйте
  `uv run pytest -q tests/test_real_data_10pct.py -k knnfeat` и при
  необходимости уменьшайте `features.knn_features.max_reference_rows`.
- Для pipeline проверяются: все стадии `done`, наличие `submission.csv` и
  `report.md`, отсутствие NaN, совпадение id с sample_submission, конечный
  `blend.oof_score`.

## Запуск

Полный набор (на CPU может занять несколько минут):

```powershell
$env:KCONV_REAL_DATA_TESTS = "1"
uv run pytest -q tests/test_real_data_10pct.py
```

Только выбранные модели и/или pipeline:

```powershell
$env:KCONV_REAL_DATA_TESTS = "1"
$env:KCONV_REAL_DATA_MODELS = "lgbm,histgb,svc"
$env:KCONV_REAL_DATA_RECIPES = "pesnik/baseline,kconv/linear-stack"
uv run pytest -q tests/test_real_data_10pct.py
```

Переопределить target/id (если используется другой локальный датасет):

```powershell
$env:KCONV_REAL_DATA_TARGET = "target"
$env:KCONV_REAL_DATA_ID = "id"
```

`svc` и `nu_svc` в smoke-тесте получают `max_iter=1000`: это ограничивает
время квадратичных SVM, но всё равно проверяет реальный fit/predict на 10%
выборке. Остальные модели используют свои production defaults.

## Матрица встроенных recipes

`tests/test_pipeline_matrix.py` не требует реального датасета и проверяет все
65 встроенных recipes (16 presets + 49 pipelines) на всех объявленных ими
задачах; `knnfeat/knn-features` и `pipeline/knnfeat-class-distance`
поддерживают только `binary` и `multiclass`. Матрица проверяет:

- `build_config` сохраняет совместимые модели, ensemble и strategy defaults;
- pipeline-рецепт компилирует `pipeline.blocks`, разрешает `columns: auto` и
  сохраняет `recipe_kind: pipeline`;
- полный `run_pipeline` до `report` создаёт OOF/test/submission/report без NaN;
- для multiclass проверяется полная probability-матрица в `submission.csv` и
  сумма вероятностей по строке, равная 1;
- отдельные unit-тесты проверяют fit-in-fold `group_statistics`, inner-OOF
  target-комбинации, label-free missingness/zero-profile и datetime cyclical
  признаки, включая unseen-группы и отсутствие собственного target-вклада.

Матрица помечена `slow`, но запускается обычным `pytest`; для быстрого
прогона используйте `pytest -m "not slow"`.
