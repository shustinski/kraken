# Quality coverage report

- Generated: 2026-11-20T09:24:55+00:00
- Registry: docs\quality\registry
- Source diff base: `HEAD~1`
- JUnit files: 10 (D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-e2e\junit.xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-gui\junit.xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-integration\junit.xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-live\junit.xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-registry\junit.xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-regression\junit(1).xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-regression\junit(2).xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-smoke\junit.xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-unit\junit(1).xml, D:\code\kraken\docs\quality\evidence\2026-11-20\full-run-1\gate-unit\junit(2).xml)

| Metric | Covered | Total | PASS | FAIL | PARTIAL | BLOCKED | NOT_TESTED | NOT_APPLICABLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ui_actions | 47 | 48 | 47 | 0 | 0 | 1 | 0 | 0 |
| api_operations | 12 | 13 | 12 | 0 | 0 | 0 | 1 | 0 |
| error_paths | 0 | 34 | 0 | 0 | 0 | 0 | 34 | 0 |
| state_transitions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tiers | 7 | 8 | 7 | 0 | 0 | 0 | 1 | 0 |
| defect_regressions | 26 | 29 | 26 | 0 | 0 | 0 | 3 | 0 |
| source | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

Rules: `Total` is the explicit denominator; a skipped test is NOT_TESTED, never PASS;
a unit is PASS only when all of its listed tests are present in the JUnit evidence and passed.

## NOT TESTED (no denominator source)

- `state_transitions`: NOT TESTED — no denominator source available.
- `source`: NOT TESTED — no denominator source available.

## Uncovered units

### ui_actions (1 uncovered of 48)

- `HUB-CHECK-UPDATES`

### api_operations (1 uncovered of 13)

- `ADMIN-SERVER-BACKUP-RESTORE`

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
- `PLUGIN-CONTOUR-DATASET-INTERACTIVE::Кадр без парного CIF отклоняется с ValueError; ZIP выборки не создаётся`
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

### defect_regressions (3 uncovered of 29)

- `D12`
- `D13`
- `D15`
