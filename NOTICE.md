# NOTICE — источники и атрибуция

`kconv` — оригинальная реализация конвейера, распространяемая по MIT
(`LICENSE`). Код стадий, конфиг, адаптеры, блендинг и тесты написаны для
этого проекта. Встроенные recipes фиксируют методологические источники; код
из перечисленных репозиториев не вендорится.

## Recipes

| Recipe | Источник | Лицензия | Что взято |
| --- | --- | --- | --- |
| `pesnik/baseline` | https://github.com/pesnik/kaggle-starter-kit | MIT | структура стартового kit и config-first подход |
| `cdeotte/tabular` | https://github.com/cdeotte/KGMON-Playbook-2026 | NO LICENSE — ideas only | идеи GBDT-разнообразия, seed averaging, weighted blend |
| `statsmaster/tabular` | https://github.com/StatsMaster/kaggle-grandmaster-templates | NO LICENSE — ideas only | идеи honest CV, OOF-дисциплины и шаблонов |
| `minerva/stacking` | https://github.com/minerva-ml/open-solution-home-credit | MIT | идеи модульной организации и stacking/blending |
| `kozodoi/tabular` | https://github.com/kozodoi/Kaggle_Home_Credit | MIT | идеи работы с высококардинальными категориями и блендинга |
| `congmnguyen/leakage-safe` | https://github.com/congmnguyen/tabular-ml-lab | MIT | идеи leakage-safe CV/OOF |
| `sergeyshilin/contracts` | https://github.com/sergeyshilin/kaggle-competitions-framework | MIT | контракты DataLoader → Model → Ensemble |
| `arnabbiswas1/staged` | https://github.com/arnabbiswas1/kaggle_pipeline_tps_aug_22 | Apache-2.0 | идеи стадийности и HPO-профиля |
| `nyaggle/oof-ensemble` | https://github.com/nyanp/nyaggle | MIT | идеи OOF-экспериментов и stacking |
| `kaggler/target-encoding` | https://github.com/jeongyoonlee/Kaggler | MIT | идеи target/frequency encoding и blending |
| `dptools/row-features` | https://github.com/kozodoi/dptools | MIT | идеи row-wise агрегатов и feature engineering |
| `knnfeat/knn-features` | https://github.com/upura/knnFeat | MIT | метод class-conditional kNN distance; реализация переписана fit-in-fold |
| `ensemble-amazon/stacking` | https://github.com/kaz-Anova/ensemble_amazon | Apache-2.0 | идеи weighted/rank averaging и meta-stacking |
| `kaggle-ieee/fraud-pipeline` | https://github.com/RobMulla/kaggle-ieee-fraud-detection | Apache-2.0 | идеи frequency encoding и wide-table workflow |
| `kconv/sklearn-diverse` | local (kconv) | MIT | оригинальный recipe: LGBM/HistGB + random forest, weighted blend |
| `kconv/linear-stack` | local (kconv) | MIT | оригинальный recipe: LogReg/Ridge + HistGB, stacking |

Для источников без LICENSE и для GPL-проектов использованы только
методологические идеи, названия и ссылки. Исходный код не копировался.

## TabPFN

`kconv` не вендорит код или веса TabPFN. Optional adapter импортирует
установленный пакет `tabpfn` и скачивает pretrained v2 checkpoint в кэш
Hugging Face. Веса v2 распространяются по Prior Labs License 1.1 —
Apache-2.0 с дополнительным требованием атрибуции (Section 10):
https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt

При распространении bundle/моделей с этими весами нужно приложить копию
лицензии и prominently отобразить «Built with PriorLabs-TabPFN»; если веса
используются для обучения/дистилляции другой модели, её имя должно
начинаться с «TabPFN». Внутренний benchmark без внешней передачи не требует
этой атрибуции.

По умолчанию используется v2. Веса v3.5/v3.5-fast gated, требуют отдельного
принятия non-commercial лицензии и в проект не входят. Пользователь,
включающий `model_version: v3.5*`, делает это самостоятельно и обязан
соблюдать лицензию Prior Labs. Подробности — `docs/tabpfn.md`.

## Каталог источников

`src/kconv/assets/catalog.json` содержит метаданные 1173 репозиториев из
`kaggle-toolkit` (1169 найденных + 4 только в download-manifest): имя, URL,
категорию, лицензию, размер и stage coverage. 211 из них были доступны
локально при сборке каталога; 14 имеют оригинальный recipe kconv;
29 executable pipelines описаны в `docs/pipelines.md`. Каталог
генерируется `scripts/build_catalog.py` и не содержит кода репозиториев.
`executable=True` означает наличие оригинального recipe, а не разрешение
копировать исходники. GPL/NO LICENSE источники остаются справочными.

Перед распространением бинарной сборки проверьте лицензии установленных
зависимостей (LightGBM, XGBoost, CatBoost, scikit-learn, pandas, numpy,
pyarrow, Typer, Rich, pydantic, PyYAML, scipy, joblib, Kaggle CLI
(`kaggle`, Apache-2.0), а при установленном optional extra — tabpfn и torch).

## UI

Визуальный язык web UI адаптирован из **unslop** by Damir Sokolovsky,
лицензия CC BY 4.0: https://github.com/heydamir/unslop.

## Данные

Датасеты и результаты соревнований в репозиторий не коммитятся. Каталог
`data/`, `runs/`, `competitions/*/runs/` и `competitions/*/latest.json`
исключены через `.gitignore`.
