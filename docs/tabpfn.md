# TabPFN v2: pretrained tabular foundation model

`tabpfn` — optional kconv adapter для pretrained TabPFN v2. Модель не
дообучается на train: она делает in-context inference по train-выборке,
поэтому подходит для малых и средних табличных датасетов (до 8 000 train
rows в safe-профиле).

## Почему v2 по умолчанию

- **v2** — веса около 70 MB (classifier 29 MB + regressor 44 MB),
  скачиваются без токена. Лицензия Prior Labs 1.1 — Apache-2.0 с
  обязательной атрибуцией «Built with PriorLabs-TabPFN» при
  распространении.
- **v3.5** — качество выше, но веса gated, требуют принятия отдельной
  non-commercial лицензии и токена, а также заметно больше памяти. Поэтому
  v3.5 не является default.
- Adapter оставляет `model_version` параметром: при наличии доступа и
  лицензии можно явно попробовать `v3.5-fast`, но это не гарантируется
  данным профилем.

Лицензия и атрибуция: [NOTICE.md](../NOTICE.md),
<https://huggingface.co/Prior-Labs/TabPFN-v2-clf>.

## Установка

```bash
# Base project + optional adapter (Python 3.12)
uv sync --extra dev --extra tabpfn

# Optional CUDA wheel: выберите версию под свой драйвер.
uv pip install --reinstall "torch==<version>+cu<xxx>" \
  --index-url https://download.pytorch.org/whl/cu<xxx>
```

Проверка:

```bash
uv run python -c "import torch, tabpfn; print(torch.__version__, torch.cuda.is_available(), tabpfn.__version__)"
uv run kconv models --details | Select-String tabpfn
```

Без `--extra tabpfn` базовые команды и тесты работают как раньше: adapter
зарегистрирован, но `--models all` и профиль 5 его пропускают, а явный
`--models tabpfn` даёт подсказку про установку.

## CLI

```bash
uv run kconv run \
  --data data \
  --target target \
  --models tabpfn \
  --name tabpfn-run \
  --until report
```

Или в `config.yaml`:

```yaml
models:
  - name: tabpfn
    seeds: 1
    params:
      model_version: v2
      device: cuda        # auto | cpu | cuda
      n_estimators: 8
      fit_mode: fit_preprocessors
      max_train_rows: 8000
      max_features: 500
```

## Python API

```python
from kconv.api import create_model

adapter = create_model("tabpfn", task="binary")
adapter.fit(
    X_train,
    y_train,
    params={"device": "cuda", "n_estimators": 8},
    seed=42,
)
probabilities = adapter.predict_proba(X_valid)
```

`FeatureBuilder` для `family="foundation"` использует ordinal-категории и не
масштабирует числовые признаки — как для tree-моделей. `tabpfn` не
поддерживает `feature_importances_`, поэтому `importance()` возвращает
`None`; для интерпретации используйте permutation importance или GBDT.

## Ограничения и память

Safe defaults adapter:

| Параметр | Default | Смысл |
| --- | ---: | --- |
| `max_train_rows` | 8 000 | safe cap для GPU с небольшим объёмом памяти; v2 hard limit 10 000 |
| `max_features` | 500 | v2 hard limit |
| `max_classes` | 10 | v2 hard limit |
| `n_estimators` | 8 | ensemble size |
| `fit_mode` | `fit_preprocessors` | низкий расход VRAM |
| `device` | `auto` | CUDA при доступном torch CUDA, иначе CPU |

При превышении лимита adapter выбрасывает `ModelError` с подсказкой
использовать `lgbm`/`xgb`/`catboost` или явно поднять лимит. kconv не
обрезает данные без согласия.

## Проверка

```bash
uv run pytest tests/test_tabpfn.py -q
```

`tests/test_tabpfn.py` полностью пропускается без optional extra. Тесты
проверяют лимиты, обработку ошибок и базовый fit/predict на маленьком
синтетическом датасете. Benchmark-отчёты и результаты на конкретном
оборудовании в репозиторий не входят: они зависят от GPU, драйвера и
версии torch.
