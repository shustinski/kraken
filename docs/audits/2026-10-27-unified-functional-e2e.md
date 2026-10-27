# Единый функциональный E2E: 27 октября 2026

`tests/test_end_to_end_functional.py` выполнил все двадцать действий из раздела
14 задания в **одном локальном проекте**, с настоящими managed artifacts,
Agent subprocess, Contour, NeuralImage, Karakal, Krona и восстановлением backup.
Passing gate: `.codex_tmp_scratch/production-audit-20261027/unified-e2e-progress-fixed/`.
Итоговый `receipt.json` находится внутри `0001-test_end_to_end_functional-cache/pytest-temp/`.
Проект: `e227ff4e-c950-4850-a408-4a4e3b6d80aa`, время до итоговой сверки — **26.197 s**.

## Что именно проверено

Вход — три репрезентативных изображения 128×128 с контролируемым рисунком.
Физический стек M1/V1/M2 и наблюдения контакта заданы явно как synthetic fixture;
каталожная нумерация не используется как доказательство физических координат.
Качество обучения, реальная электрическая топология и корпусная accuracy не
оцениваются. Единственная эпоха CPU использует 2 train + 1 validation image;
для модельной фазы установлен лимит 90 s, для всего дочернего процесса — 300 s.
Длительное обучение не запускается.

| № | Действие | Проверяемый результат |
| --- | --- | --- |
| 1 | Проект | Новый UUID и единое хранилище для всех последующих действий |
| 2 | Слои | M1, V1, M2 с явными типами |
| 3 | План | Задача с исполнителем, критерием приёмки, переходом в running |
| 4 | Импорт | Три immutable PNG версии, исходные SHA сохранены |
| 5 | Подготовка | Закреплена точная версия каждого входа; dataset публикуется после получения CIF в шаге 6 |
| 6 | Векторизация | Настоящий Contour/Agent создаёт три CIF; они повторно читаются и служат labels для dataset |
| 7 | Модель | Dataset ZIP имеет ссылки на исходные PNG/CIF; одна CPU эпоха публикует managed checkpoint с measured execution |
| 8 | Сегментация | Опубликованная модель потребляется настоящим headless inference, получены три бинарные версии |
| 9 | Анализ | Контролируемый дефект записан новой дочерней версией маски; настоящий Karakal IoU/AnalysisRun создаёт QualityIssue |
| 10 | Назначение | Замечание назначено исполнителю, связанному с principal |
| 11 | Исправление | Новая parent-linked версия восстанавливает известную fixture-маску; новый AnalysisRun пересчитывает метрику |
| 12 | Приёмка | Самопроверка отклонена, отдельный reviewer выполняет stored-analysis recheck и accept |
| 13 | Сшивка | Agent создаёт mosaic из текущих масок, включая принятую правку; проверены canonical manifest и input lineage |
| 14 | Совмещение | SI calibration, affine fit, контакт с независимым V1 evidence и явным review |
| 15 | Связность | Из масок извлечены компоненты, явно подтверждены материал/контакт/терминалы; одна synthetic signal net |
| 16 | Экспорт | CIF, checkpoint и mosaic экспортированы; EDIF прочитан Krona, synthetic Y=A даёт [0,1] |
| 17 | Отчёт | Критерий плана удовлетворён versioned evidence другим reviewer; задача и план завершены; CSV отражает точную ревизию плана 8 и readiness=1 |
| 18 | Архив | Проект переведён в archived |
| 19 | Повторное открытие | Сохранены принятая правка, physical и electrical snapshots |
| 20 | Backup/restore | Новый каталог, integrity scan, сравнение 19 версий и SHA, план/quality/physical/electrical/оба AnalysisRun и metric rows совпали; исходные PNG неизменны |

Ручная правка не объявляется grid-preserving только из-за parent link. Для
сшивки явно задана geometry revision на текущих исправленных mask versions.
Существующий отказ на произвольный derived content без подтверждённой геометрии
сохранён. Геометрия и контакты этого примера не являются измерениями реальной ИС.

## Дефекты, найденные единым сценарием

После шагов анализа/приёмки Hub падал в `plugin_jobs()` при следующей синхронизации:
generic `StoredEvent` анализа не имеет атрибутов domain `EventEnvelope`.
Исправлен общий read-only history adapter; generic события остаются видимыми,
сохраняют UUID, payload, фактическое время и записанного actor, если он есть.
Отсутствующий actor отображается как неизвестный, не приписывается текущему
пользователю. Повреждённая generic запись вызывает явную ошибку, не скрывается.
ActivityRecord поддерживает отсутствие actor. `test_mixed_project_history.py`
проверяет history/as_of, jobs, matrix cells, analysis lookup, activity и неизменность журнала.

Неполная команда `record_progress` вызывала TypeError. Общий transport decoder
теперь проверяет обязательные поля всех planning commands и вложенных records.
`planning-required-before/`: 17 failures; `planning-required-fixed/` — passing.
`planning-required-http/` проверяет PostgreSQL + ASGI HTTP: 422 без изменения
плана и успешное исправление запроса с тем же Idempotency-Key.

Ранние failed E2E reports сохранены отдельно. Часть отказов была ошибками нового
сценария: неверная начальная stream revision, повторная definition version,
reference attachment вместо frame artifact и отсутствующая декларация grid.
Они не объявляются дефектами продукта и не засчитываются как passing gates.

## Воспроизведение и границы

```powershell
$env:KRAKEN_RUN_END_TO_END_FUNCTIONAL='1'
.venv/Scripts/python.exe tools/verify_repository.py tests/test_end_to_end_functional.py --jobs 1 --timeout 330 --report .codex_tmp_scratch/unified-e2e-new
```

Обычный общий набор пропускает этот opt-in сценарий, чтобы не запускать обучение
неявно. Его фактический passing результат учитывается отдельно от общего набора.
Числа overlapping scoped gates не складываются.

После исправлений общий `unified-e2e-integrated-root/summary.json` подтвердил:
**188 модулей root + CSliser, 1253 cases, 0 failures/errors, 14 skipped,
1239 выполнены**. PostgreSQL включён; новый HTTP/PG negative gate также выполнен
в этом наборе. Общий запуск не обучает модель; opt-in E2E в нём пропущен, его
passing 26.197 s приведён отдельно выше. Ruff изменённых modules, scoped diff
whitespace и offline lock check прошли.

Этот gate не выполняет двадцать действий через desktop widgets, PostgreSQL/HTTP
transport или установленный пакет. Он не подтверждает масштабирование, GPU,
обработку полной реальной топологии, power-loss или чистую ОС. Общая матрица
готовности остаётся частичной. Численную воспроизводимость обучения этот тест
также не утверждает (`deterministic_algorithms=false` в measured execution).
