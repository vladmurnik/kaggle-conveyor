# Inference bundle

`kconv export` превращает завершённый run в переносимый bundle, который не
требует установленного `kconv`:

```powershell
uv run kconv export --comp <slug>
uv run kconv export --comp <slug> -o dist/<slug>-inference --force
```

Структура:

```text
<slug>-<run_id>/
  submission.py
  metadata.json
  requirements.txt
  sample_submission.csv       # если доступен
  models/
    <model>/folds/fold_<i>/
      features.joblib
      seed_<seed>.joblib
    meta_model.joblib          # только для реального stack
```

`submission.py` читает CSV/parquet test-файл, применяет сохранённые
feature builders, усредняет fold/seed-предсказания и повторяет blend/submit
логику: `best_single`, `mean`, `weighted`, `rank_mean` или `stack`.

```powershell
python submission.py --test data/test.csv --output submission.csv
```

Bundle не содержит сырые train/test, OOF и каталог `runs/`. Inference-модели
сохраняются стадией `train`; `stack`-мета-модель — стадией `blend`. Поэтому
run, созданный до появления этой функции, нужно перезапустить.

Проверка: `tests/test_inference.py` сравнивает `submission.csv`, созданный
standalone-скриптом, с submission исходного run (`allclose`).
