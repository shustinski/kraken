# Функциональная цепочка исправления технического дефекта

26 октября 2026: **passed**, 1.267 s. Обучение и оценка качества векторизации
не выполнялись. Проверка ограничена ссылкой на изображение в заголовке CIF.

Команда из корня репозитория:

```powershell
.venv\Scripts\python.exe tools/verify_quality_functional.py --manifest .codex_tmp_scratch/production-audit-20261025/real-reference-inventory/pairs.jsonl --managed-profile .codex_tmp_scratch/production-audit-20261025/real-reference-project --output .codex_tmp_scratch/production-audit-20261026/quality-functional-01
```

Для повтора нужен новый пустой output. Сценарий читает одну пару с изолированным
`reference_image_name_mismatch`, проверяет её SHA-256 в существующем managed
хранилище и на исходном диске, копирует байты в отдельный профиль. Полный корпус
не пересчитывается и не изменяется. Для текущего manifest выбрана пара `1415`.

В production `EmbeddedProjectService` созданы три реально аутентифицированных
аккаунта, проект и ACL. Finding назначен исполнителю, связанному с аккаунтом
корректора. Исправлена только строка `( R ... );` на `( R 1415.jpg );`, создана
новая immutable версия с parent на исходную. Self-review корректора отклонён
даже при роли maintainer. Независимый reviewer проверил реальный parser Contour,
совпадение имени и размера JPEG, точное равенство остальных байтов CIF и принял
только это техническое исправление.

Принятый артефакт экспортирован. Повторный export в тот же путь отклонён с
сохранением существующего файла. Новый экземпляр сервиса повторно открыл проект;
issue и пять событий истории совпали, в том числе после удаления disposable
проекции. Старые и новые blobs проверены, исходники остались неизменными.

Receipt со всеми IDs, хешами, участниками и историей:
`.codex_tmp_scratch/production-audit-20261026/quality-functional-01/receipt.json`.
Исходный JPEG SHA-256:
`bbf4eab8c35e4266fffae8ccc195a7bd5d76b63f75f0ddd3f09d1a2c5ac66a20`.
Исходный CIF SHA-256:
`e5598a6858acb2401338ddfee8ef8db666770116e18ba5038a0673b36ebceaec`.

Исправление export сопровождают `tests/test_quality_functional_export.py`:
**4 passed**, 3.26 s. Покрыты существующие managed/external destinations,
сохранность external source, cleanup собственной частичной выгрузки с retry
и сохранение файла другого writer, заменившего частичную выгрузку до cleanup.

Эта приёмка не доказывает правильность геометрии, распознавания или электрической
схемы. Supplied CIF не является электрической ground truth. UI и PostgreSQL
в данном новом сценарии не запускались. Quality report как отдельный продуктовый
API остаётся GAP: receipt — результат проверочного tool; штатные issue/history
и экспорт артефактов проверены через сервис.
