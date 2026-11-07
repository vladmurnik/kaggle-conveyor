# Проверка и smoke-покрытие

В репозитории нет пользовательских датасетов, соревновательных запусков и
личных benchmark-отчётов. Проверки ниже выполняются на синтетических данных
и на данных, которые пользователь добавляет локально. Это сделано намеренно:
публичный репозиторий не должен зависеть от приватных файлов или конкретного
соревнования.

## Что покрывают тесты

- полный CLI-прогон `ingest → validate → hpo → train → blend → submit →
  report` для binary, multiclass и regression;
- автоопределение target, id, задачи и метрики;
- `group_kfold` и `time_series` с сохранением train membership;
- adversarial validation train/test;
- OOF blending, stacking и fallback на лучшую одиночную модель;
- offline package и проверку `SHA256SUMS`;
- resume, heartbeat и recovery после аварийного завершения;
- схему submission, число строк и отсутствие NaN;
- feature-cache с ключом, включающим состав fold;
- отказ от частичного NaN в target;
- fold-aware LLM-планы и обязательный approval.

Основные тесты:

```bash
uv run pytest tests/test_pipeline.py tests/test_submit.py -q
uv run pytest tests/test_cv.py tests/test_blend.py -q
uv run pytest tests/test_package.py tests/test_watchdog_and_index.py -q
uv run pytest tests/test_web.py -q
```

## Синтетический end-to-end smoke

Тесты создают небольшие детерминированные таблицы и проверяют, что каждая
стадия завершается, submission содержит ожидаемые колонки и не содержит
NaN. Для classification проверяется сохранение исходных меток, для
regression — числовой формат предсказаний.

Такой smoke не доказывает качество на конкретном соревновании. Он проверяет
интеграцию стадий, схемы данных, сериализацию артефактов и отсутствие
очевидных поломок.

## Group и time-series CV

Для `group_kfold` тесты проверяют, что группы не пересекаются между train и
validation. Для `time_series` проверяется expanding-window split: будущие
строки не попадают в обучение, а `folds_train.json` сохраняет точный состав
каждого fold.

## Adversarial validation

При `validation.adversarial: true` стадия `validate` обучает классификатор
`train vs test` только на признаках. Target в этой диагностике не
используется. Результат пишется в `01_validate/adversarial.json`, а warning
при превышении `adversarial_threshold` попадает в отчёт.

Это диагностика сдвига, а не автоматическая смена CV-схемы. Высокий AUC
означает, что train и test различаются, и OOF-метрику нельзя напрямую
интерпретировать как оценку leaderboard.

## Реальные данные

Пользователь может запустить kconv на своём датасете:

```bash
uv run kconv run --data data --target target --recipe pipeline/pesnik-clean
```

Для тестов с реальными данными есть opt-in маркер `realdata`. Такие тесты не
входят в обычный CI и требуют локальный датасет, который не коммитится.

## Ограничения

- Синтетические тесты не заменяют аудит утечек на реальном train/test split.
- Adversarial validation показывает сдвиг, но не объясняет его причину.
- OOF-метрика зависит от схемы CV; для group/time-задач нужно выбирать
  схему, соответствующую лидерборду.
- HPO на маленьких фолдах может переобучаться; deep-стратегия не гарантирует
  улучшение.
- Submission проверяется по схеме, но не по правилам конкретного
  соревнования; перед отправкой проверьте sample submission и правила
  Kaggle.

Перед публикацией результата на Kaggle проверьте:

1. совпадает ли train/test распределение;
2. нет ли target leakage в признаках;
3. соответствует ли CV схеме лидерборда (group/time);
4. совпадает ли submission schema с sample submission;
5. не переобучается ли HPO на маленьких фолдах.
