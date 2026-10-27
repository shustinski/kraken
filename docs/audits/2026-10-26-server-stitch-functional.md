# Серверная вычисляемая сшивка: leased Agent и восстановление

Продолжение GAP-STITCH01. Проверяется функциональное исполнение контролируемых
геометрических fixtures. Длительное обучение и оценка качества векторизации
не запускаются. Физическое расположение реального корпуса не выдумывается.

## Исполнение

`request_layer_stitch`/`cancel_layer_stitch` используют прежние authenticated
multilayer command API, CAS/idempotency и canonical physical stream. После
commit сервер отправляет request в существующий PostgresAgentGateway;
background server cycle повторяет незавершённый handoff после перезапуска
или недоступности очереди. Нового queue store/исполнительного графа нет.

Queue envelope `kind=stitch` содержит неизменный native PluginJobManifestV2,
initiating actor и capability `layer.stitch.v1`. Worker регистрирует эту
capability и запускает `kraken_manager.infrastructure.stitch_worker` в
отдельном процессе. Для Python deployment нужны existing server/postgres/agent
dependencies и extra `stitching` (NumPy/Pillow) там, где вычисляется или
проверяется mosaic. Самостоятельный installed worker ещё не проверен.

Server Agent теперь скачивает V2 inputs по каждому `asset_id`, проверяет SHA
и использует staging конкретного lease claim. HTTP input/download-ticket
доступен только для объявленной версии на проекте leased manifest. Входы
другого job/project не становятся доступными по произвольному version ID.

Actual native output загружается через прежний Agent output protocol.
Complete route различает analysis/V1/stitch, проверяет текущий lease,
publication job identity/finality/scope и uploaded digest. Общий importer
проверяет полный mosaic container и атомарно фиксирует layer attachment,
lineage и physical receipt. Failed/cancelled native publication не импортирует
outputs. Серверный importer повторно проверяет состояние, исходные версии и
права инициатора.

В очереди задана reservation 1 CPU / 0 GPU / 512 MB. Это запрос к существующей
admission policy, **не** измеренный RSS и не hard process/OS limit. Проверка
больших нагрузок и реальных ограничений остаётся обязательной.

## Отмена и восстановление

Cancellation сохраняется до queue RPC. Дополнительная проверка после enqueue
закрывает race, когда отмена успела commit между чтением intent и enqueue.
Expired/reclaimed claim не может читать inputs, публиковать, complete/fail;
новый claim также теряет право на completion после cancellation.

Canonical imported receipt восстанавливает потерянный queue completion commit:
очередь сверяет тот же immutable manifest, снимает lease и завершает transport,
не запуская повторное вычисление. Canonical failed/cancelled request останавливает
незавершённую очередь. Единичная ошибка/невалидный actor одного request не
блокируют последующие requests того же проекта.

Если initiating identity/ACL больше не разрешает RUN_PLUGIN, scheduler только
отказывает в незавершённом запросе и останавливает очередь. Output не создаётся.
Решение вычисляется по текущему identity/ACL, причина сохраняется в canonical
history, automatic event имеет ProgramSnapshot «Kraken Server scheduler» и
original request actor snapshot. Это не обход проверки прав для импорта.

## Доказательства

Под `.codex_tmp_scratch/production-audit-20261026/`, gates пересекаются:

- `stitch-server-native-02/`: 3 модуля, 10 passed. Actual PostgreSQL + uvicorn
  HTTP + ServerAgentWorker + отдельный native worker; успешная mosaic и
  повреждённый source, reopen; соседний native AnalysisRun и handoff.
- `stitch-server-recovery/`: 3 модуля, 15 passed. Enqueue outage и queue ACK
  recovery, immutable source transport, соседние lease/model cases.
- `stitch-server-races/`: 3 модуля, 24 passed. Cancellation during enqueue,
  request isolation и соседние Agent generation/safety cases.
- `stitch-server-authorization/`: 3 модуля, 10 passed. Deactivated initiating
  identity -> terminal rejection, no artifact, cancelled queue и program audit;
  atomic PG и domain architecture cases. Этот gate предшествует дополнительному
  recovery canonical failure, который проверяется следующим root gate.

Первый `stitch-server-native/` сохранил red fixture: отрицательный тест вызывал
несуществующий `queue.claim`, тогда как настоящий API называется `lease`.
Два actual native success/failure cases уже проходили; fixture исправлена,
необходимая negative проверка выполнена повторно без skip.

Последний root gate `stitch-server-root-functional/`: 166 модулей,
1138 cases, 0 failures/errors, 13 skips (1125 выполнены). Он включает
7 server-stitch cases и соседние local/server/PG/Qt/Agent workflows после
исправления recovery canonical failure. Это source gate, не installed gate.
Focused изменённые модули прошли Ruff; для существующих scaffold
`create_app_sections/section_1.py`/`section_5.py` проверка выполнена с
`--ignore F401` из-за ранее оставленных общих unused imports.

## Открыто

Remote Qt управление/просмотр mosaic пока не подключены. Existing computed
dialog обслуживает локальный проект. Общий typed layer Pipeline, соседский
constraint graph/cycle consistency, vector geometry, большие workload/RSS/IO,
bounded import cancellation и final installed/clean OS остаются открыты.
Серверный cycle пока читает physical snapshots проектов; его scalability
на большом количестве requests ещё не измерена. Source/hash decode locality
и runtime provenance новых вычислений также требуют дальнейшей приёмки.
GAP-STITCH01 и общая производственная готовность этим протоколом не закрываются.

Remote UI и bounded block API позднее реализованы и проверены:
[remote Qt протокол](2026-10-26-remote-stitch-ui-functional.md). Общий typed
Pipeline и остальные production gates по-прежнему открыты.
