# Quality coverage report

- Generated: 2026-11-10T06:53:38+00:00
- Registry: docs\quality\registry
- Source diff base: `HEAD~1`
- JUnit files: 6 (C:\code\kraken\docs\quality\evidence\2026-11-01\contour-targeted-junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-01\live-contour-worker-junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-01\live-worker-contour-faults-junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-01\reproducer-d4-junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-01\target-lease-junit.xml, C:\code\kraken\docs\quality\evidence\2026-11-01\upload-preflight-junit.xml)

| Metric | Covered | Total | PASS | FAIL | PARTIAL | BLOCKED | NOT_TESTED | NOT_APPLICABLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ui_actions | 2 | 61 | 2 | 0 | 0 | 1 | 58 | 0 |
| api_operations | 1 | 17 | 1 | 0 | 0 | 0 | 16 | 0 |
| error_paths | 0 | 37 | 0 | 0 | 0 | 0 | 37 | 0 |
| state_transitions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tiers | 5 | 8 | 5 | 0 | 0 | 0 | 3 | 0 |
| defect_regressions | 2 | 14 | 2 | 0 | 0 | 0 | 12 | 0 |
| source | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

Rules: `Total` is the explicit denominator; a skipped test is NOT_TESTED, never PASS;
a unit is PASS only when all of its listed tests are present in the JUnit evidence and passed.

## NOT TESTED (no denominator source)

- `state_transitions`: NOT TESTED — no denominator source available.
- `source`: NOT TESTED — no denominator source available.

## Uncovered units

### ui_actions (59 uncovered of 61)

- `AGT-HEALTH`
- `BLOB-PUT-GET-RESTART`
- `HUB-ADD-LAYER`
- `HUB-ADD-PERFORMER`
- `HUB-ADD-REPRESENTATION`
- `HUB-ADMIN-REFRESH`
- `HUB-ANALYSIS`
- `HUB-ARCHIVE-PROJECT`
- `HUB-ATTACH-PROJECT`
- `HUB-BIND-WORKSPACE`
- `HUB-BOOTSTRAP-ACCOUNT`
- `HUB-CHECK-UPDATES`
- `HUB-COMPUTED-STITCH`
- `HUB-CONFIGURE-ROOTS`
- `HUB-CONNECT-SAVED-SERVER`
- `HUB-CREATE-PROJECT`
- `HUB-DELETE-PERFORMER`
- `HUB-DELETE-PROJECT`
- `HUB-DEV-SESSION`
- `HUB-EDIT-PERFORMER`
- `HUB-ELECTRICAL`
- `HUB-EXPORT-BACKUP`
- `HUB-IMPORT-BACKUP`
- `HUB-ISSUE-WORK`
- `HUB-LAYER-CONTEXT`
- `HUB-LAYER-WORKFLOW`
- `HUB-LOGIN`
- `HUB-MATRIX-CONTEXT`
- `HUB-MIGRATE-LEGACY-WORK`
- `HUB-MULTILAYER`
- `HUB-MY-WORK`
- `HUB-OPEN-PROJECT`
- `HUB-PIPELINE-STITCH-STEP`
- `HUB-PLUGINS`
- `HUB-PREFERENCES`
- `HUB-PROJECT-PROPERTIES`
- `HUB-PROJECT-ROLES`
- `HUB-QUALITY`
- `HUB-REFRESH-CATALOG`
- `HUB-RENAME-PROJECT`
- `HUB-REORDER-LAYERS`
- `HUB-REPRESENTATION-LIFECYCLE`
- `HUB-RESTORE-PROJECT`
- `HUB-REVIEW-RETURN`
- `HUB-SCAN-INTEGRITY`
- `HUB-STATISTICS`
- `HUB-STITCH-FILES`
- `HUB-UPDATE-SOURCE`
- `HUB-WORK-PLAN`
- `MGR-MATRIX-VISUAL`
- `PLUGIN-CONTOUR-DATASET-INTERACTIVE`
- `PLUGIN-CONTOUR-HELP`
- `PLUGIN-CSLISER-CAPABILITIES`
- `PLUGIN-CSLISER-COPY-CANCEL-RECOVERY`
- `PLUGIN-CSLISER-SELF-TEST`
- `PLUGIN-KARAKAL-COMPARE-MAP`
- `PLUGIN-KRONA-HELP`
- `PLUGIN-KRONA-LOGIC-VIEWER`
- `PLUGIN-NEURALIMAGE-VERSION`

### api_operations (16 uncovered of 17)

- `SRV-ANALYSIS`
- `SRV-COMMANDS`
- `SRV-ELECTRICAL`
- `SRV-HEALTH-SESSION`
- `SRV-HISTORY-STATISTICS`
- `SRV-JOBS-AGENT-PROTOCOL`
- `SRV-LAYER-WORKFLOW`
- `SRV-LAYERS`
- `SRV-NOTES-CHECKS-PACKAGES`
- `SRV-PHYSICS`
- `SRV-PLANS`
- `SRV-PRINCIPALS`
- `SRV-PROJECTS`
- `SRV-QUALITY`
- `SRV-REPRESENTATIONS`
- `SRV-WEBSOCKET`

### error_paths (37 uncovered of 37)

- `AGT-HEALTH::Запрос без токена отклонён`
- `AGT-REGISTRY-CONFIDENCE::interactive-режим переводил задание в waiting_for_user`
- `BLOB-PUT-GET-RESTART::Нет билета, неверный HMAC, чужая операция, просрочка — 401`
- `HUB-ADD-LAYER::Смесь PNG и JPG блокирует скан`
- `HUB-ADD-REPRESENTATION::Импорт неподдерживаемого вида отклоняется`
- `HUB-ANALYSIS::Отказ рантайма сохраняется без импорта замеры`
- `HUB-BOOTSTRAP-ACCOUNT::Второй аккаунт отклоняется: A local account already exists`
- `HUB-CREATE-PROJECT::Пустое имя отклоняется: «Название проекта: значение обязательно.»`
- `HUB-DELETE-PROJECT::Неполное подтверждение именем отклоняется`
- `HUB-IMPORT-BACKUP::Испорченная копия отклоняется ошибкой проверки`
- `HUB-ISSUE-WORK::Отказ без каталога`
- `HUB-LAYER-CONTEXT::Удаление требует точного имени слоя`
- `HUB-LOGIN::Неверный пароль или блокировка: «Неверные учётные данные или аккаунт временно заблокирован.»`
- `HUB-PIPELINE-STITCH-STEP::Нет геометрии слоя — шаг не добавляется`
- `HUB-PROJECT-ROLES::Чужой If-Match — 409 revision.conflict`
- `HUB-QUALITY::Неизвестная серьёзность отклоняется: DomainValidationError`
- `HUB-REVIEW-RETURN::Отказ без каталога`
- `HUB-SCAN-INTEGRITY::«Обнаружены ошибки» при повреждённых данных`
- `PERSISTENT-GATEWAY-UPLOAD::HTTP500 до исправления D8`
- `PERSISTENT-GATEWAY-UPLOAD::Stale revision — 409`
- `PLUGIN-CONTOUR-DATASET-INTERACTIVE::Headless-ветка отвергает frames.dataset.prepare.v1 (ValueError в worker.py)`
- `PLUGIN-CSLISER-COPY-CANCEL-RECOVERY::Перезапись отклоняется`
- `SRV-ANALYSIS::Отказ рантайма сохраняется без импорта замеров`
- `SRV-ARTIFACTS::Stale revision — 409`
- `SRV-ARTIFACTS::Чужой билет — 401`
- `SRV-COMMANDS::Отказ без прав — 403`
- `SRV-ELECTRICAL::Чужой If-Match — 409`
- `SRV-HEALTH-SESSION::application/problem+json на ошибки`
- `SRV-HEALTH-SESSION::Неверный пароль — 401`
- `SRV-JOBS-AGENT-PROTOCOL::CAS-конфликт — 409`
- `SRV-JOBS-AGENT-PROTOCOL::Дубликат complete — 403`
- `SRV-JOBS-AGENT-PROTOCOL::Неверный токен — 401`
- `SRV-PHYSICS::Чужой If-Match — 409`
- `SRV-PLANS::Чужой If-Match — 409`
- `SRV-PRINCIPALS::Наблюдатель не переименовывает проект — 403`
- `SRV-PROJECTS::Чужой If-Match — 409 revision.conflict`
- `SRV-QUALITY::Чужой If-Match — 409`

### tiers (3 uncovered of 8)

- `regression`
- `smoke`
- `stateful`

### defect_regressions (12 uncovered of 14)

- `D1`
- `D10`
- `D11`
- `D12`
- `D13`
- `D14`
- `D2`
- `D3`
- `D5`
- `D6`
- `D7`
- `D9`
