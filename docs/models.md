# Каталог моделей

`kconv` регистрирует **51 model id**: 6 нативных/boosted-адаптеров,
1 optional foundation-адаптер (`tabpfn`) и 44 scikit-learn-оценщика.
Все они проходят через один контракт `fit`/`predict`/`predict_proba` и одну
7-стадийную схему.

```powershell
uv run kconv models
uv run kconv models --json
uv run kconv models --details
```

Задача выбирается автоматически из данных; несовместимые id отбрасываются
при построении списка моделей. Явный несовместимый id даёт ошибку с
подсказкой. `tabpfn` — optional: без extra он виден в каталоге, но не
включается в `--models all` и профиль 5.

## Нативные и boosted-адаптеры (6)

| id | Семейство | Задачи |
| --- | --- | --- |
| `lgbm` | tree | binary, multiclass, regression |
| `xgb` | tree | binary, multiclass, regression |
| `catboost` | tree | binary, multiclass, regression |
| `histgb` | tree | binary, multiclass, regression |
| `logreg` | linear | binary, multiclass |
| `ridge` | linear | regression |

## Foundation (1, optional)

| id | Семейство | Задачи | Установка |
| --- | --- | --- | --- |
| `tabpfn` | foundation | binary, multiclass, regression | `uv sync --extra tabpfn` |

`tabpfn` использует pretrained TabPFN v2, а не дообучение на ваших данных.
Веса скачиваются один раз в кэш Hugging Face. Подробности, лицензия и
замеры — `docs/tabpfn.md`.

## scikit-learn: регрессия (24)

`linear_regression`, `lasso`, `elasticnet`, `bayesian_ridge`, `huber`,
`sgd_regressor`, `passive_aggressive_regressor`, `knn_regressor`,
`decision_tree_regressor`, `extra_tree_regressor`,
`random_forest_regressor`, `extra_trees_regressor`,
`gradient_boosting_regressor`, `adaboost_regressor`, `bagging_regressor`,
`svr`, `linear_svr`, `nu_svr`, `kernel_ridge`, `pls_regression`, `ransac`,
`mlp_regressor`, `dummy_regressor`, `ard_regression`.

## scikit-learn: классификация (20)

`knn_classifier`, `decision_tree_classifier`, `extra_tree_classifier`,
`random_forest_classifier`, `extra_trees_classifier`,
`gradient_boosting_classifier`, `adaboost_classifier`, `bagging_classifier`,
`svc`, `linear_svc`, `nu_svc`, `ridge_classifier`, `perceptron`,
`passive_aggressive_classifier`, `sgd_classifier`, `gaussian_nb`,
`bernoulli_nb`, `lda`, `qda`, `mlp_classifier`.

## Что важно знать

- `family` задаёт препроцессинг: linear-модели получают one-hot и
  стандартизацию; tree- и foundation-модели — ordinal-категории без
  масштабирования.
- Модели без `predict_proba` (`linear_svc`, `ridge_classifier`,
  `perceptron`, `passive_aggressive_classifier`, `svc`, `nu_svc`) используют
  `decision_function`; для binary применяется sigmoid, для multiclass —
  softmax. Это даёт валидную матрицу вероятностей, но не калибровку.
- `passive_aggressive_*` в scikit-learn 1.8+ помечены deprecated; они
  оставлены для совместимости и могут быть удалены в scikit-learn 1.10.
- `svc`/`nu_svc` намеренно не используют `probability=True` (deprecated в
  scikit-learn 1.9); вероятности строятся из decision function.
- HPO-пространства заданы только для нативных/boosted-адаптеров. Модели без
  search space (включая `tabpfn`) не тратят бюджет на повторные одинаковые
  fit; параметры можно передать явно в `models[].params`.
- `tabpfn` defaults: v2, `n_estimators=8`, `fit_mode=fit_preprocessors`,
  safe cap 8 000 train rows, 500 features, 10 classes. Для таблиц больше
  лимита используйте GBDT или явно поднимите `params.max_train_rows`.
- Все 51 id проходят synthetic smoke (TabPFN — при установленном extra).
  Полные pipeline-прогоны выполняются пользователем на его данных; в
  репозитории нет пользовательских датасетов и benchmark-отчётов.
