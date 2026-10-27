# Развитие Kraken: матрица требований и протокол проверки

Исходная ревизия: `b244b73`. Дата: 23 октября 2026. Аудит относится к заданию
«Полноценное развитие Kraken до производственной системы», разделы 1–19.
Исторические отчёты 13–19 октября не засчитываются как свежие проверки.

**Полная производственная готовность не подтверждена.** Наличие реализации
отделено от проверки. Реестр ниже сохраняет незавершённые требования, включая
те, для которых нет эталонов, стенда или полной реализации.

## Исходное состояние

В репозитории существуют шесть Python composition roots (`kraken_core`,
`kraken_manager`, `kraken_hub`, `kraken_server`, `kraken_admin`, `kraken_agent`),
Rust Blob Gateway и плагины Contour, NeuralImage, Karakal, Krona, CSliser, KateGB.
Локальные авторитетные данные — журнал событий и blobs; SQLite — перестраиваемая
проекция. На сервере события, проекции, ACL и очередь сохраняются в PostgreSQL.
Существуют immutable artifact versions, ReviewBatch, PluginJob, WorkOrder,
AnalysisRun, матрица кадров, история и отчёты. Рисуемый граф происхождения
результатов не является полноценным планировщиком исполняемого процесса.

Первоначальный сбор всех тестов: 88 ошибок импорта; корневой набор — шесть
ошибок сбора (отсутствовали numpy/cv2/shapely). В локальное `.venv` установлены
numpy 2.4.4, OpenCV 5.0.0, Shapely 2.1.2, SciPy 1.17.1. Lock-файл не менялся.
Монолитный корневой запуск прервался Windows access violation примерно на 62%;
пользователь предоставил снимок системного окна. Это не успешная проверка.

## Матрица требований

Статусы: **П** — реализовано и проверено указанным тестом; **Ч** — частично;
**НП** — реализовано, но целевой сценарий не проверен; **О** — отсутствует;
**Д** — некорректно; **В** — проверка ограничена внешней средой.
Статус относится к строке, а не ко всему разделу задания.

| ID / требование | Статус | Реализация / проверка | Оставшаяся работа |
|---|---|---|---|
| R03 инвентаризация и доказательства | Ч | Этот документ; `tools/verify_repository.py` сохраняет log/XML/JSON каждого модуля | Полный аудит всех платформ и внешних стендов |
| R04 планы, задачи, scope, исполнители, сроки, effort, приоритет | П | `domain/planning.py`, `infrastructure/planning_commands.py`; `test_work_planning.py`, `test_work_planning_integration.py` | Смежные требования отдельно ниже |
| R04 зависимости, иерархия, циклы и блокировки | П | DAG + combined parent/dependency deadlock validation; тот же набор | Масштабные измерения для больших планов |
| R04 переходы, повтор, отмена, ревизии, критерии и приёмка | П | `application/planning.py`; проверка ролей, scope и versioned evidence в codec | Независимая политика запрета самоприёмки, если она требуется проекту |
| R04 baseline, milestones, история планов | П | Immutable baseline snapshots в существующем event stream `work-plan:<uuid>` | Полноценная аналитика отклонения baseline |
| R04 библиотека технологических шаблонов и автоматическое создание задач | П | Версионированные шаблоны, provenance, атомарное применение по scope; local/HTTP/PostgreSQL/Qt проверки, `docs/work-planning-templates.md` | Независимая проверка библиотеки технологами |
| R04 разграничение план / поручение / запуск / artifact version | П | WorkPlan с проверенными ссылками на WorkOrder/PluginJob; существующие ArtifactVersion | Полная автоматическая синхронизация состояний |
| R05 происхождение результатов и ручной граф операций | НП | Hub `pipeline_snapshot.py`, `layer_management_parts`; существующие plugin manifests | Повторить все пользовательские сценарии графа |
| R05 исполняемый typed DAG, ready scheduling, политика retry/resources | Ч | Durable workflow, local/server scheduler, retry, approval, typed image/model ports, queue admission CPU/GPU/memory; настоящий Contour DAG и удалённый AnalysisRun dispatch/import | OS/per-host limits, длительные concurrent/fault испытания |
| R05 версии входов, алгоритма, параметров | П | Закрепление immutable versions/SHA/algorithm, transitive stale detection, управляемый пересчёт; unit/local/PG/live Contour проверки | Расширить реальную приёмку на все capability combinations |
| R05 FIFO claim и изоляция leased server job | П | `kraken_agent/jobs.py`, `runner.py`, `server_worker.py`; `test_agent_execution_safety.py` | Многопроцессная нагрузка длительного стенда |
| R05 отмена, поздние callbacks, retry публикации | П | Результат и состояние в одной SQLite-транзакции, callback dedup | Полный remote V2 import/end-to-end |
| R05 авария/остановка прямого subprocess | П | `recovery_required`; регрессии с настоящими subprocess | Дерево дочерних процессов, power-loss и серверная очередь |
| R06 Contour векторизация/CIF/сшивка/редактирование | Ч | 1245 full cases, 16 пропусков недоступных исторических fixtures; реальные 0516/0518 CIF, server UI и 3-job Agent DAG с импортом CIF | Глобальная сшивка и неоднозначности; качество на корпусе не входит в текущий функциональный этап |
| R06 NeuralImage dataset/training/inference/confidence | Ч | Подготовлен реальный dataset; ограниченный CPU train→публикация checkpoint→выбор model port→inference→import→reopen прошёл за 21.93 s, `2026-10-26-model-functional-acceptance.md` | GPU fault injection и воспроизводимость runtime; длительное обучение и оценка качества исключены из текущего этапа пользователем |
| R06 Karakal server versioned input/analysis | Ч | Реальные HTTP/PostgreSQL/Agent/Karakal: import, ошибка, retry, cancel, repeat, reopen/replay; local/remote PNG maps, worker receipt/pin/refusal/recheck, diagnostics, backup. Единый локальный representative 20-stage E2E с actual Karakal/quality/correction/review/backup прошёл 27 октября | Единый desktop/server/installed E2E; финальная упаковка |
| R06 Krona трассируемые гипотезы | П | SHA-256, provenance и метки unverified; `plugins/krona/tests/test_quality*` | Полная строгая электрическая модель: GAP-Q01 |
| R06 CSliser стандартные контракты и защита данных | Ч | Реальные Qt copy/cancel/close и файловые гонки; Linux source gate; 32 реальные пары → 64 Hub версии → архив/reopen/backup/restore, SHA исходников неизменны; `2026-10-27-csliser-functional.md` | Hub IPC actions ещё не подключены к окну; durable crash recovery, большие диапазоны и installed acceptance |
| R07 маски/геометрические метрики/analysis issues | Ч | Durable QualityIssue, correction/independent recheck/acceptance/exception, local/server/UI, lineage gates и AnalysisRun rules; `docs/quality-workflow.md` | Реальная corpus приёмка, межслойные эталоны и все quality dimensions |
| R07 неопределённость и отделение расчёта от электрической приёмки | Ч | Krona явно unverified; критерии плана требуют versioned evidence | Унифицированные независимые quality dimensions по всем плагинам |
| R08 многослойность, transforms, calibration, registration revisions | Ч | SI calibration, stack, SVD affine control points, stale anchors, uncertainty и independent contact review; local/HTTP/PG/Qt, `docs/physical-registration.md` | Проверенные реальные физические слои и контрольные точки |
| R09 проводимость/vias/elements/verified netlist | Ч | Krona анализирует существующие EDIF, не доказывает связь по микрофотографиям | Полная evidence-based цепочка геометрия→элементы→netlist |
| R10 планирование в настоящем Hub | Ч | WorkPlanningDialog, templates, workflow, quality, physical model; board/analytics проходят интеграционную проверку | Полная проверка массовых операций, всех рабочих представлений и многопользовательских сценариев |
| R10 отзывчивость планирования и восстановление UI после ошибок | П | Фоновые service calls, revision refresh, retry key; Qt tests | Реальная WAN/большие планы и все рабочие представления |
| R10 время жизни матрицы при удалении родителем | П | `background_resources.py`, weak owner, queue errors; `test_matrix_lifetime.py` | Совпадение с конкретным native crash не доказано |
| R10 RU/EN | Ч | Новая панель и Krona имеют обе локализации | Все существующие экраны ещё не сверены |
| R11 статистика и экспорт CSV/XLSX/PDF | НП | `infrastructure/reports`, `test_statistics_reports.py` | Измеренные cycle time/rework/utilization/forecast и baseline variance |
| R12 аудит/идемпотентность/CAS WorkOrder | П | `work_order_lifecycle.py`, UOW, `WorkOrderRecordedV1` reducer; security tests | Миграция legacy записей, созданных до event-sourced сохранения |
| R12 серверный RBAC отмены/завершения и запрет перезаписи record | П | `persistent_services_parts/issue_work.py`; `test_work_order_security.py` | Проверить на настоящем PostgreSQL + concurrent clients |
| R12 SHA-256/path/RBAC/secret/blob contracts | ? | Windows native junction regressions Python/Agent 20 ? Rust HTTP 13+8; generation claims; ???????? kernel-lock death/recovery. Linux fork ?????? ? ???????? ????? ? ????????? ????, before/after ????? ????????? | ?????? ??????????? ?????, ???????? ???? races, ???????? ? ??????? ???????? native plugins; path checks ?? ???????? OS sandbox |
| R12 согласованный PostgreSQL+BlobStore backup/restore | Ч | Реальный offline PostgreSQL dump/restore и 315 blobs проверены: 879 events, 835 projections, 253 idempotency records | Автоматизация эксплуатационного backup и online/concurrent disaster recovery |
| R13 small/medium/large измерения | Ч | Существуют performance tests и blob benchmark | Свежие базовые измерения всех указанных операций, memory/users/files |
| R14 многослойная стратегия проверок | Ч | Targeted + isolated XML/log/JSON, реальные Qt/subprocess; representative local 20-stage E2E и backup прошли. Последний root: 189 модулей, 1254 cases, 0 failures/errors, 14 skipped; 2630 source files unchanged | Единый desktop/remote/installed E2E, нагрузка нескольких пользователей, все внешние плагины |
| R14 дистрибутивы/установка/обновление/удаление | Ч | Актуальные 27 октября PyInstaller/production installers; audit install/acceptance/uninstall Desktop и Server прошли. Installed raster/vector/invalid-CIF workers, frozen Server/Admin/Gateway migrations/auth/HTTP/Range/SciPy/restart проверены; 453/368 project/Contour pure modules совпали с source | Production AppId install/update, SCM service, clean OS/reboot/Linux, standalone plugin packages и побитовая воспроизводимость; `2026-10-27-windows-installed-functional.md` |
| R15 реестр GAP | Ч | Реестр ниже | Полная трассировка каждого дефекта всех подсистем |
| R17–19 итоговая готовность/независимый аудит | Не достигнуто | Открытые MAJOR/CRITICAL и непроверенные целевые сценарии | Закрыть по воспроизводимой проверке, затем повторить весь E2E |

## Изменения архитектуры

Планы сохраняются snapshot-событиями в существующем EventStore. Отдельной
таблицы, базы, очереди или конкурирующего графа не добавлено. Для local применяется
LocalProjectUnitOfWork; сервер использует существующий PostgreSQL UOW. API:
`GET/POST /api/v1/projects/{project_id}/work-plans`,
`GET /api/v1/projects/{project_id}/work-plans/{plan_id}`,
`POST .../{plan_id}/commands` с `If-Match` и `Idempotency-Key`.

Файловая упаковка WorkOrder перенесена из `application/work_package.py` в
`infrastructure/work_package.py`: Path/ZIP/файловые операции нарушали оба
существующих архитектурных теста. Обновлены внутренние импорты; структура пакетов,
CLI и форматы файлов сохранены. Доменные объекты остаются без файловых путей API.

## Реестр дефектов и ограничений

| GAP | Важность / подсистема | Причина, воспроизведение и исправление | Проверка / статус |
|---|---|---|---|
| GAP-A01 | MAJOR Agent | DESC очередь и раздельные list/transition: starvation/двойной claim. Атомарный FIFO и targeted claim | `test_agent_execution_safety.py`: исправлено |
| GAP-A02 | CRITICAL Agent | Callback после cancel сохранял outputs; финальный retry конфликтовал. Atomic result/state и dedup | Тот же набор: исправлено |
| GAP-A03 | MAJOR Agent | Остановка могла оставлять RUNNING. Перевод в recovery_required | Реальные subprocess: исправлено |
| GAP-W01 | CRITICAL Server | Cancel/complete игнорировали actor, record мог переписать поручение. ACL внутри UOW, immutable metadata, references, CAS | `test_work_order_security.py`: исправлено |
| GAP-W02 | MAJOR storage | WorkOrder был только проекцией и терялся при rebuild. Событие snapshot + reducer + staged projection | Replay/rollback/reopen tests: исправлено для новых записей; legacy migration открыта |
| GAP-W03 | CRITICAL Server | Client destination воспринимался как путь сервера. Новая external registration не может назначить server placement; completion не читает client path | Security tests: исправлено; импорт внешнего результата требует отдельного workflow |
| GAP-W04 | MAJOR Server | due_at терялся для server work; invalid review due_at выходил сырым ValueError | Положительные/отрицательные due_at tests: исправлено |
| GAP-P01 | MAJOR planning | Не было планов, baseline и критериев отдельно от Job/Order | Domain/service/HTTP/local/Qt tests: исправлено в реализованном объёме; шаблоны/scheduler/board открыты |
| GAP-Q01 | CRITICAL Krona | Legacy aliases выводили электрическую связность из рисунка/номеров | Исправлено для EDIF: только explicit joined portRefs; ambiguous memberships дают conflicts/unknown; 44 fixture/CLI/Qt tests. Получение физически проверенного netlist из микрофотографий остаётся отдельным R09 |
| GAP-Q02 | CRITICAL Krona | Gate подменялся по соседнему номеру, неизвестный gate/конфликт питания могли выдавать определённый уровень | Регрессии: исправлено; физическая верификация этим не заменяется |
| GAP-Q03 | MAJOR Krona | EDIF cache зависел от пути, не байтов; stale results | Content hash и проверка неизменности во время анализа: исправлено |
| GAP-U01 | CRITICAL Qt | Родительское удаление обходило closeEvent, worker callbacks захватывали QWidget, scene cycle удерживал owner | Child-process lifetime tests воспроизвели и проверили исправление; причина конкретного снимка Windows не доказана |
| GAP-E01 | BLOCKER приёмка | Synthetic server UI не заменяет реальную engineering ground truth, чистые ОС и многопользовательскую нагрузку | Открыто. Предоставлен реальный корпус 5498 JPEG+CIF, выполнен полный inventory; импорт и реальная приёмка продолжаются. Электрическая/физическая ground truth пока не установлена |
| GAP-B01 | BLOCKER aggregate model backup | Импорт plugin attachment записывал строку `"None"` в optional IDs; после backup restore replay падал на UUID | Исправлены запись null и совместимое чтение старых attachment events; 3 регрессии и восстановление настоящей ранее обученной модели/принятой CIF-правки прошли. Подробности в протоколе 26 октября |
| GAP-D01 | CRITICAL remote download | Очистка после HTTP/FileExistsError удаляла существующий пользовательский файл | До исправления 6 failed/2 passed; после исправления 8 passed, соседние remote модули прошли. Удаляется только созданный этим запросом файл с совпавшей identity |
| GAP-A-CANCEL01 | MAJOR local analysis cancel | Отмена записывалась после сетевого вызова, а поздний result не проверял terminal state | Durable cancel до сети, repair после сбоя, сериализованные import/state updates и late-result refusal проверены; протокол локальной карты 26 октября |
| GAP-RUNTIME01 | MAJOR AnalysisRun reproducibility | Runtime объявлялся Hub/Server; signature не учитывал действительный worker; metadata version расходилась с пакетом | Исправлено и проверено в native local/server/PG/Qt: receipt, pin, mismatch/legacy refusal, Repeat constraints, actual recheck и diagnostics. Финальная упаковка ещё нужна; протокол worker runtime 26 октября |
| GAP-STITCH01 | MAJOR вычисляемая сшивка | Legacy workflow проверял attachment и отмечал статус без вычисления мозаики | Открыто частично: native local/server Agent V2, geometry, bounded raster/vector mosaic/coverage, atomic import, lease/cancel/recovery/authorization, backup, Qt и typed downstream Pipeline проверены. Native inference→mosaic и Contour→vector mosaic прошли локально и через PostgreSQL/HTTP; CAS/rollback, quality refusal/retry и desktop continuation проверены. Layer JSONL export/reader, coverage, remote Qt fault/retry и byte-identical Windows/Linux roundtrip реализованы. Полный real CIF parse: 4944/5498 прочитаны, 554 отказа — не passing gate. Нужны сложный CIF, downstream electrical/IR consumption, большая нагрузка, общий E2E и installed gate; протоколы vector stitching/exchange 26 октября. Legacy file workflow сохраняется |
| GAP-HISTORY01 | MAJOR Hub mixed history | После actual AnalysisRun generic StoredEvent не имел event_type attribute; следующая Agent synchronization падала в plugin_jobs(), также затрагивались matrix/history/activity | Исправлен read-only GenericHistoryEvent adapter без изменения журнала; unknown actor сохраняется неизвестным; 3 регрессии, statistics/layer tests и единый E2E 27 октября прошли |
| GAP-PLAN-ARGS01 | MAJOR planning transport | Неполные обязательные поля вызывали KeyError/TypeError вместо validation error | 17 negative cases падали до исправления; required fields decoder, rollback и PostgreSQL/ASGI HTTP 422 с исправлением того же Idempotency-Key проверены |
| GAP-PKG-CIF01 | MAJOR Windows native vector worker | Spec исключал CIF-кодек; его eager imports тянули Contour application/OpenCV в headless worker. Frozen и installed self-test воспроизвели отказ | Исправлены пакетный состав, lazy exports и helper imports; source guard без GUI/cv2/Torch, actual installed vector/hole/rotated-box и contextual invalid-CIF refusal прошли. RU/EN codec resources включены. 27 октября, Windows deployment protocol |

## Правила интерпретации проверок

Артефакты локального прогона находятся в `.codex_tmp_scratch/production-audit/`.
`tools/verify_repository.py tests --jobs 2 --report <каталог>` запускает модули
в отдельных процессах и сохраняет полный лог, JUnit XML и `results.json`.
Timeout, collection error, native crash и skipped остаются в результатах.
Windows ErrorMode меняется только в дочернем тестовом процессе, чтобы аварии
не открывали пользователю системные окна. Никакие проверки не отключаются.

Наборы пересекаются: количества нельзя складывать. API на файловом UOW не
подменяет реальную PostgreSQL интеграцию. Qt offscreen использует настоящие
виджеты, но не подтверждает физическое взаимодействие со всеми драйверами.
Наличие версионированного evidence не доказывает его инженерную корректность:
это основание для явного решения инженера с отдельным правом ACCEPT_REVIEW.

## Проверки 24 и 25 октября 2026

Шаблоны, исполняемый workflow, quality и physical registration реализованы и проверяются в production composition. Исторические прогоны не заменяют итоговую приёмку.

Подробный актуальный протокол, реальные данные, сборки, дефекты импорта и ограничения: [приёмка 25 октября](2026-10-25-production-acceptance.md).

Текущий функциональный этап без длительного обучения и оценки качества:
[приёмка 26 октября](2026-10-26-functional-readiness.md).

Последняя функциональная проверка CSliser, сохранности файлов и CIF import:
[приёмка 27 октября](2026-10-27-csliser-functional.md). Общий root + CSliser gate:
185 модулей, 1231 cases, 0 failures/errors, 13 skipped (1218 выполнены).
Этот результат не закрывает оставшиеся строки матрицы и installed acceptance.

Единый representative local service/Agent E2E из двадцати действий прошёл:
[протокол сквозного сценария](2026-10-27-unified-functional-e2e.md), 26.197 s,
19 версий, оба AnalysisRun и все проверяемые модели восстановлены из backup.
Настоящий desktop/remote/installed twenty-stage сценарий и нагрузка остаются открыты.

После исправлений смешанной истории и planning transport последний общий gate
`production-audit-20261027/unified-e2e-integrated-root/summary.json` прошёл:
188 модулей, 1253 cases, 0 failures/errors, 14 skipped (1239 выполнены),
PostgreSQL включён. Opt-in twenty-stage E2E выполнен отдельно и не суммируется.

Актуальные Windows audit installers прошли install/acceptance/uninstall:
[протокол сборки и установленного приложения](2026-10-27-windows-installed-functional.md).
Последний root — 189 модулей, 1254 cases, 0 failures/errors, 14 skipped
(1240 выполнены), с неизменностью 2630 закреплённых source files.
