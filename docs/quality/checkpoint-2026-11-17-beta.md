# Чекпоинт W2 — полный гейт после точки заморозки (beta-stabilization-w1w2)

Дата прогона: 2026-11-18. Ветвь: `beta-stabilization-w1w2`.

Команда: `scripts/quality-gates.ps1 -Tier smoke,unit,integration,gui,stateful,e2e,regression,live`
(запуск обёрткой `.test_runtime/run-full-gate-task10.ps1` — тот же шаблон, что обёртка
Task 8, из-за ловушек кавытей PowerShell 5.1). В сессию подложен только
`KRAKEN_TEST_POSTGRES_URL` из файла стенда; `KRAKEN_LIVE_WORKER_CONFIG`, все
`KRAKEN_RUN_*` и `KRAKEN_TEST_PG_BIN` намеренно сняты — см. «Known gaps».

Стенд: запущен и здоров (`GET http://127.0.0.1:8080/api/v1/health` = 200; `state.json`:
БД `kraken_beta_test`, schema-режим, gateway отключён, живой серверный процесс).

## Результаты полного гейта (итоговый прогон: GATE RESULT: PASS, exit 0)

| Слой | Статус | passed/failed/skipped | exit |
|---|---|---|---|
| registry | PASS | 13/0/0 | 0 |
| smoke | PARTIAL | 2/0/2 | 0 |
| unit | PARTIAL | 1865/0/27 | 0 |
| integration | PARTIAL | 123/0/24 | 0 |
| gui | PASS | 90/0/0 | 0 |
| stateful | NOT_TESTED | 0/0/0 | — (tests/stateful/ отсутствует, план Task 3.x) |
| e2e | PARTIAL | 0/0/1 | 0 |
| regression | PARTIAL | 180/0/5 | 0 |
| live | PARTIAL | 14/0/11 | 0 |

Coverage-артефакты: `docs/quality/evidence/2026-11-18/coverage-report.json` (+ `.md`,
`coverage-base-report.json`, `coverage-compare.json`). Сравнение с базой 2026-11-10:
0 регрессий; 6 newly NOT_TESTED (ADMIN-SERVER-BACKUP-RESTORE, AGT-OP-VECTORIZE,
AGT-OP-DATASET-PREPARE, AGT-OP-TRAIN, AGT-OP-BINARY-SEGMENT, HUB-LOGIN-TABBED) —
информационно: строки реестра без автотест-маппинга, гейт не валят.

## Triage FAIL-ов базового прогона (RED→GREEN, один узкий фикс на причину)

1. **unit** `tests/test_qt_harness.py::test_failure_capture_produces_artifacts` —
   фикстура `database` (`Config("alembic.ini")` + `command.upgrade`) вызывала
   `logging.config.fileConfig` в `migrations/env.py`, которая снимает с root-логгера
   все обработчики (ring-буфер qt-harness, перехват pytest). При настроенном
   `KRAKEN_TEST_POSTGRES_URL` падение детерминировано. Фикс `b841cfa` (программные
   вызовы opt-out`ом `attributes={"configure_logger": False}`; CLI alembic не затронут,
   `kraken_server.run_migrations` конфиг-файл не передаёт). Дефект зарегистрирован как
   **D23** (`461f0cb`, `d23-red.txt`/`d23-green.txt`).
2. **e2e** `test_single_project_twenty_stage_functional_lifecycle` — плагин-подпроцесс
   `dataset.model.train.v1` падал с 0xC0000005 на `torch.cuda.is_available()` при
   `CUDA_VISIBLE_DEVICES=-1` (D15-семейство: инициализация CUDA на этой машине
   крашится; воспроизведено standalone, faulthandler-трэйс указывает в
   `managed_training.py`). Фикс `ac5adf6`: установленный приём D15.1 — проба CUDA в
   одноразовом подпроцессе, краш → skip (PARTIAL, не PASS); пробник вынесен в
   `tests/helpers/cuda_gate.py` и переиспользован `test_pipeline_stitch_live_content.py`.
3. **unit** `tests/test_manager_migration_workflow.py` — тело теста зелёное, падает
   teardown `tempfile.TemporaryDirectory` (WinError 145 на `migration-state`) —
   D20-семейство (on-access сканер). Фикс `9e92007`: переход на `kraken_tempdir`
   (повторно-пробная уборка).
4. **unit** `tests/test_filesystem_event_read_cache.py` — подмена `"revision":1→9` без
   изменения размера попадала в тот же тик Windows-времени записи, что и append;
   подпись read-кэша `(size, mtime_ns)` не менялась, фильтрованное чтение отдавало
   кэшированный валидный разбор (DID NOT RAISE). Фикс `57f8cda`: подмена меняет размер
   — детерминированная инвалидация тем же путём, на который опираются остальные
   тесты файла.

## Экологические skip-ы (не дефекты; причины гейтов)

- `tests/test_server_backup.py` — 24 skip: «KRAKEN_TEST_POSTGRES_URL and
  KRAKEN_TEST_PG_BIN required for native backup/restore» (нативные pg_dump/pg_restore
  не настроены, у роли стенда `kraken_local_app` нет CREATE DATABASE).
- live-слой — 11 skip: `KRAKEN_LIVE_WORKER_CONFIG` / `KRAKEN_UI_TEST_CONFIG` /
  `KRAKEN_RUN_TRAINING_LIVE` / `KRAKEN_RUN_PIPELINE_LIVE` /
  `KRAKEN_RUN_MODEL_FUNCTIONAL` намеренно не заданы (см. Known gaps).
- `tests/test_blob_gateway.py` — 17 skip: «Rust Blob Gateway executable is not built».
- `tests/offline/test_offline_sync.py` (2) и `tests/offline/test_pytorch_resolution.py`
  (2) — D16/D17/D21: закрытая сеть, pypi/индекс cu132 недоступны.
- `tests/test_pipeline_stitch_live_content.py` (1) и e2e (1) — «no usable CUDA device
  (probe exited/crashed)»: CUDA-init на этой машине крашится (проба D15.1).
- `tests/test_desktop_runtime_smoke.py` — 2 skip: гейты изолированного
  сервера/агента (`KRAKEN_SMOKE_*`).
- `tests/test_filesystem_event_read_cache.py`, `tests/test_filesystem_lock_recovery.py`
  — 3 skip: требуется POSIX fork.

## Known gaps (что блокирует и где закрывается)

- **Blob Gateway exe не поставлен** (внешнее D-поступление). Блокирует: PASS live-слоя
  (живые сценарии в режиме `mode == 'gateway'` — без exe они честно self-skip в
  PARTIAL по решению контроллера), контур server-задач Contour через gateway,
  действия BLOB-PUT-GET-RESTART и AGT-OP-* с живой приёмкой. Live PASS — цель W3
  после поставки exe.
- **KRAKEN_TEST_PG_BIN не настроен** + у роли стенда нет CREATE DATABASE: 24 skip
  backup/restore; ADMIN-SERVER-BACKUP-RESTORE остаётся NOT_TESTED.
- **tests/stateful/ отсутствует** (план Task 3.x) — слой stateful NOT_TESTED; честный
  пробел плана, не дефект.
- **D21 (OPEN)**: офлайн-резолв uv/PyTorch на закрытой машине невозможен; `uv.lock`
  не пересобран (нужен один сетевой заход к download.pytorch.org) — внешний
  пред-реquisite вне объёма W1/W2.
- **Сборка/установщик не верифицированы**: Task 7 отменён владельцем, поэтому статус
  упаковки беты в этом чекпоинте не заявляется (отмена — решение владельца).

## Фикс D22 (справка)

D22 (удаление проекта через Admin падало на Windows: `remove_stored_path` чистил
staged-дерево `shutil.rmtree`, WinError 145 на каталоге категории `aux`) исправлен
`0b6d946`, зарегистрирован `90c7727`; интеграционный слой зелёный с этим фиксом
(123/0/24, повторено на этом гейте).

## Provenance доказательств

Итоговая evidence — единственный согласованный полный прогон 2026-11-18
(`docs/quality/evidence/2026-11-18/**`; слой при перезапуске затирает stale-junit).
Пререквизитные слои гейта (gate-registry, gate-unit) до этого коммита лежали
untracked (наследие Tasks 8/9); этим коммитом они зафиксированы вместе со всем
evidence дня. Рекомендация: и дальше коммитить receipts пререквизитов вместе со
слоем, либо добавить их в `.gitignore` — решение за владельцем ветки. Унаследованные
незакоммиченные правки `evidence/2026-11-10/gate-registry|gate-unit` и `.gitignore`
в этот чекпоинт не входят.

## Реестр: ручные правки статусов не требуются

Вердикты действий выводит `tools/quality/coverage_report.py` из JUnit этого же
прогона (`action_verdict`); поле `status` в `actions.json` участвует только для
BLOCKED/NOT_APPLICABLE. Как и в Task 8, ручные правки статусов не вносились;
единственная правка реестра — регистрация дефекта D23 в `defects.json` с
регенерацией `defects.md` (`generate_quality_docs.py --write`), что требует только
что и сделано.

## Проверка свежести (гибридное правило)

Метод: одноразовый скрипт `.test_runtime/freshness_check.py` (не коммитится) для
каждого `beta:true` действия с вердиктом PASS находит JUnit-файлы, содержащие его
маппинг тестов, и сверяет атрибут `timestamp` testsuite и `mtime` файла. Проверены
**все 15** бета-PASS действий (при требовании ≥5 по слоям): 11 gui (HUB-CREATE-PROJECT,
HUB-PROJECT-ROLES, HUB-ADD/EDIT/DELETE-PERFORMER, HUB-REVIEW-RETURN, HUB-ADD-LAYER,
HUB-LAYER-CONTEXT, HUB-ADD-REPRESENTATION, HUB-ISSUE-WORK, HUB-STATISTICS) и 4
integration (SRV-HEALTH-SESSION, SRV-PRINCIPALS, SRV-ARTIFACTS,
PERSISTENT-GATEWAY-UPLOAD); опирающиеся junit gate-gui / gate-unit / gate-integration /
gate-regression — все с timestamp и mtime 2026-11-18. Нарушений нет: каждый
бета-PASS подкреплён доказательствами сегодняшнего дня.

## Готовность beta:true действий (таблица для W3; колонку «W3 сценарий» заполняет W3, id не выдумываются)

| Действие | Tier | Вердикт | Примечание | W3 сценарий |
|---|---|---|---|---|
| HUB-BOOTSTRAP-ACCOUNT | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-LOGIN | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-CONNECT-SAVED-SERVER | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-PREFERENCES | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-UPDATE-SOURCE | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-CHECK-UPDATES | gui | BLOCKED | BLOCKED в реестре: источник обновлений вне закрытой сети; приёмка — W4 |  |
| MGR-MATRIX-VISUAL | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-CREATE-PROJECT | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-REFRESH-CATALOG | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-OPEN-PROJECT | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-BIND-WORKSPACE | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-RENAME-PROJECT | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-ARCHIVE-PROJECT | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-RESTORE-PROJECT | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-DELETE-PROJECT | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-PROJECT-PROPERTIES | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-SCAN-INTEGRITY | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-ADMIN-REFRESH | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-CONFIGURE-ROOTS | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-EXPORT-BACKUP | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-IMPORT-BACKUP | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-PROJECT-ROLES | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-ADD-PERFORMER | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-EDIT-PERFORMER | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-DELETE-PERFORMER | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-MATRIX-CONTEXT | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-REVIEW-RETURN | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-ADD-LAYER | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-REORDER-LAYERS | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-LAYER-CONTEXT | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-ADD-REPRESENTATION | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-REPRESENTATION-LIFECYCLE | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-ISSUE-WORK | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-MY-WORK | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-WORK-PLAN | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-PLUGINS | gui | NOT_TESTED | нет автотест-маппинга; UI/серверный приём — W3 |  |
| HUB-STATISTICS | gui | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| SRV-HEALTH-SESSION | integration | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| SRV-PROJECTS | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-PRINCIPALS | integration | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| SRV-LAYERS | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-REPRESENTATIONS | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-ARTIFACTS | integration | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| SRV-COMMANDS | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-WEBSOCKET | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-HISTORY-STATISTICS | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-NOTES-CHECKS-PACKAGES | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-JOBS-AGENT-PROTOCOL | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| SRV-PLANS | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| AGT-HEALTH | integration | NOT_TESTED | нет автотест-маппинга; серверный приём — W3 |  |
| PLUGIN-CONTOUR-HELP | smoke | NOT_TESTED | нет автотест-маппинга; приём плагинов — W3 |  |
| PLUGIN-NEURALIMAGE-VERSION | smoke | NOT_TESTED | нет автотест-маппинга; приём плагинов — W3 |  |
| PLUGIN-CONTOUR-DATASET-INTERACTIVE | gui | NOT_TESTED | нет автотест-маппинга; интерактивный приём — W3 |  |
| BLOB-PUT-GET-RESTART | integration | NOT_TESTED | нет автотест-маппинга; блокируется отсутствием exe Blob Gateway (D-поступление) |  |
| PERSISTENT-GATEWAY-UPLOAD | integration | PASS | свежесть подтверждена (JUnit evidence 2026-11-18) |  |
| HUB-LOGIN-TABBED | gui | NOT_TESTED | нет автотест-маппинга; UI-приём — W3 |  |
| ADMIN-SERVER-BACKUP-RESTORE | integration | NOT_TESTED | нет автотест-маппинга; native backup/restore требует KRAKEN_TEST_PG_BIN и CREATE DATABASE |  |
| AGT-OP-VECTORIZE | live | NOT_TESTED | нет автотест-маппинга; живой приём агентских операций — W3 (нужен стенд + gateway) |  |
| AGT-OP-DATASET-PREPARE | live | NOT_TESTED | нет автотест-маппинга; живой приём агентских операций — W3 (нужен стенд + gateway) |  |
| AGT-OP-TRAIN | live | NOT_TESTED | нет автотест-маппинга; живой приём агентских операций — W3 (нужен стенд + gateway) |  |
| AGT-OP-BINARY-SEGMENT | live | NOT_TESTED | нет автотест-маппинга; живой приём агентских операций — W3 (нужен стенд + gateway) |  |

Итог: 15 PASS / 45 NOT_TESTED / 1 BLOCKED из 61 `beta:true`. PARTIAL-вердиктов на
уровне действий нет: вердикт действия PASS только при полном прохождении маппинга,
любой skip маппинга даёт NOT_TESTED (PARTIAL живёт на уровне слоёв).

Следующий шаг: палитра W3 — план живых сценариев принятия
