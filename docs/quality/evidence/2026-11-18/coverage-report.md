# Quality coverage report

- Generated: 2026-11-18T13:49:35+00:00
- Registry: docs\quality\registry
- Source diff base: `HEAD~1`
- JUnit files: 10 (C:\code\kraken\docs\quality\evidence\2026-11-18\gate-e2e\junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-gui\junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-integration\junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-live\junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-registry\junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-regression\junit(1).xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-regression\junit(2).xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-smoke\junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-unit\junit(1).xml, C:\code\kraken\docs\quality\evidence\2026-11-18\gate-unit\junit(2).xml)

| Metric | Covered | Total | PASS | FAIL | PARTIAL | BLOCKED | NOT_TESTED | NOT_APPLICABLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ui_actions | 12 | 48 | 12 | 0 | 0 | 1 | 35 | 0 |
| api_operations | 3 | 13 | 3 | 0 | 0 | 0 | 10 | 0 |
| error_paths | 0 | 34 | 0 | 0 | 0 | 0 | 34 | 0 |
| state_transitions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tiers | 7 | 8 | 7 | 0 | 0 | 0 | 1 | 0 |
| defect_regressions | 17 | 22 | 17 | 0 | 0 | 0 | 5 | 0 |
| source | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

Rules: `Total` is the explicit denominator; a skipped test is NOT_TESTED, never PASS;
a unit is PASS only when all of its listed tests are present in the JUnit evidence and passed.

## NOT TESTED (no denominator source)

- `state_transitions`: NOT TESTED — no denominator source available.
- `source`: NOT TESTED — no denominator source available.

## Uncovered units

### ui_actions (36 uncovered of 48)

- `AGT-HEALTH`
- `AGT-OP-BINARY-SEGMENT`
- `AGT-OP-DATASET-PREPARE`
- `AGT-OP-TRAIN`
- `AGT-OP-VECTORIZE`
- `BLOB-PUT-GET-RESTART`
- `HUB-ADMIN-REFRESH`
- `HUB-ARCHIVE-PROJECT`
- `HUB-BIND-WORKSPACE`
- `HUB-BOOTSTRAP-ACCOUNT`
- `HUB-CHECK-UPDATES`
- `HUB-CONFIGURE-ROOTS`
- `HUB-CONNECT-SAVED-SERVER`
- `HUB-DELETE-PROJECT`
- `HUB-EXPORT-BACKUP`
- `HUB-IMPORT-BACKUP`
- `HUB-LOGIN`
- `HUB-LOGIN-TABBED`
- `HUB-MATRIX-CONTEXT`
- `HUB-MY-WORK`
- `HUB-OPEN-PROJECT`
- `HUB-PLUGINS`
- `HUB-PREFERENCES`
- `HUB-PROJECT-PROPERTIES`
- `HUB-REFRESH-CATALOG`
- `HUB-RENAME-PROJECT`
- `HUB-REORDER-LAYERS`
- `HUB-REPRESENTATION-LIFECYCLE`
- `HUB-RESTORE-PROJECT`
- `HUB-SCAN-INTEGRITY`
- `HUB-UPDATE-SOURCE`
- `HUB-WORK-PLAN`
- `MGR-MATRIX-VISUAL`
- `PLUGIN-CONTOUR-DATASET-INTERACTIVE`
- `PLUGIN-CONTOUR-HELP`
- `PLUGIN-NEURALIMAGE-VERSION`

### api_operations (10 uncovered of 13)

- `ADMIN-SERVER-BACKUP-RESTORE`
- `SRV-COMMANDS`
- `SRV-HISTORY-STATISTICS`
- `SRV-JOBS-AGENT-PROTOCOL`
- `SRV-LAYERS`
- `SRV-NOTES-CHECKS-PACKAGES`
- `SRV-PLANS`
- `SRV-PROJECTS`
- `SRV-REPRESENTATIONS`
- `SRV-WEBSOCKET`

### error_paths (34 uncovered of 34)

- `ADMIN-SERVER-BACKUP-RESTORE::Восстановление на занятой базе отклоняется`
- `AGT-HEALTH::Запрос без токена отклонён`
- `AGT-OP-BINARY-SEGMENT::Отмена задания, падение плагина — задание помечено failed, проект не испорчен`
- `AGT-OP-DATASET-PREPARE::Отмена задания, падение плагина — задание помечено failed, проект не испорчен`
- `AGT-OP-TRAIN::Отмена задания, падение плагина — задание помечено failed, проект не испорчен`
- `AGT-OP-VECTORIZE::Отмена задания, падение плагина — задание помечено failed, проект не испорчен`
- `BLOB-PUT-GET-RESTART::Нет билета, неверный HMAC, чужая операция, просрочка — 401`
- `HUB-ADD-LAYER::Смесь PNG и JPG блокирует скан`
- `HUB-ADD-REPRESENTATION::Импорт неподдерживаемого вида отклоняется`
- `HUB-BOOTSTRAP-ACCOUNT::Второй аккаунт отклоняется: A local account already exists`
- `HUB-CREATE-PROJECT::Пустое имя отклоняется: «Название проекта: значение обязательно.»`
- `HUB-DELETE-PROJECT::Неполное подтверждение именем отклоняется`
- `HUB-IMPORT-BACKUP::Испорченная копия отклоняется ошибкой проверки`
- `HUB-ISSUE-WORK::Отказ без каталога`
- `HUB-LAYER-CONTEXT::Удаление требует точного имени слоя`
- `HUB-LOGIN-TABBED::Неверные учётные данные — то же сообщение блокировки`
- `HUB-LOGIN::Неверный пароль или блокировка: «Неверные учётные данные или аккаунт временно заблокирован.»`
- `HUB-PROJECT-ROLES::Чужой If-Match — 409 revision.conflict`
- `HUB-REVIEW-RETURN::Отказ без каталога`
- `HUB-SCAN-INTEGRITY::«Обнаружены ошибки» при повреждённых данных`
- `PERSISTENT-GATEWAY-UPLOAD::HTTP500 до исправления D8`
- `PERSISTENT-GATEWAY-UPLOAD::Stale revision — 409`
- `PLUGIN-CONTOUR-DATASET-INTERACTIVE::Headless-ветка отвергает frames.dataset.prepare.v1 (ValueError в worker.py)`
- `SRV-ARTIFACTS::Stale revision — 409`
- `SRV-ARTIFACTS::Чужой билет — 401`
- `SRV-COMMANDS::Отказ без прав — 403`
- `SRV-HEALTH-SESSION::application/problem+json на ошибки`
- `SRV-HEALTH-SESSION::Неверный пароль — 401`
- `SRV-JOBS-AGENT-PROTOCOL::CAS-конфликт — 409`
- `SRV-JOBS-AGENT-PROTOCOL::Дубликат complete — 403`
- `SRV-JOBS-AGENT-PROTOCOL::Неверный токен — 401`
- `SRV-PLANS::Чужой If-Match — 409`
- `SRV-PRINCIPALS::Наблюдатель не переименовывает проект — 403`
- `SRV-PROJECTS::Чужой If-Match — 409 revision.conflict`

### tiers (1 uncovered of 8)

- `stateful`

### defect_regressions (5 uncovered of 22)

- `D12`
- `D13`
- `D15`
- `D16`
- `D17`
