# Локальная вычисляемая сшивка: Agent, immutable result и Qt

Промежуточный протокол GAP-STITCH01. Это функциональная проверка управляемых
геометрических fixtures; не оценка качества векторизации или physical ground
truth корпуса `D:\ozi\NN`. Длительное обучение не запускалось.

## Реализовано

`layer.stitch.v1` использует существующий Agent V2, durable queue, authenticated
loopback API и отдельный subprocess `kraken_manager.infrastructure.stitch_worker`.
В Desktop capability регистрируется как встроенная при наличии NumPy/Pillow;
добавлен extra `stitching`. Локальная private frozen entry
`--kraken-stitch-worker` реализована в исходниках. Установленный frozen запуск
после этого изменения ещё не проверен. `uv lock --check --offline` проходит;
новый extra использует уже закреплённые NumPy/Pillow, версии пакетов не менялись.

Физический stream хранит `StitchRender`: request/job UUID, исходный неизменный
V2 manifest с timestamp, ссылку на конкретную stitch revision и terminal receipt.
Это ссылка на исполнение в прежнем Agent, не дополнительный execution graph.
RUN_PLUGIN/RBAC, project state, CAS и idempotency применяются к запросу/отмене.
Входы обязаны соответствовать version/SHA/frame и действительным catalog
coordinates; эти coordinates не используются как physical poses.

Worker сверяет состав inputs и observation evidence SHA, строит mosaic blocks
по закреплённой геометрии. Публикуется один **layer-scoped** immutable artifact
`application/vnd.kraken.mosaic+sqlite3`, без фиктивных output frame IDs.
SQLite служит индексированным контейнером результата, а не новым project store.
Это уточнение первоначального плана отдельных blobs: один content-addressed
контейнер позволяет держать bounded SQLite cache и один decoded block, не
собирать в RAM миллионный JSON/ZIP index. Внутри сохранены geometry revision,
input version/SHA, sampling/overlap policy, grid и все pixel/coverage blocks.
PNG хранит pixels; zlib uint32 little-endian хранит coverage counts без потери
перекрытий. У блоков отдельные SHA, у всего файла — managed blob SHA.

Reader проверяет schema, plan fingerprint, grid/policy consistency, полный
состав blocks, размеры, PNG modes, bounded coverage decompression и checksum.
Неполный, лишний, повреждённый или чужой result не принимается. Writer проверяет
отмену между блоками и ограничивает output file budget; failed/cancelled worker
не публикует outputs. Существующий Agent теперь разрешает отменить результат
в IMPORTING, пока импорт ещё не подтверждён. SUCCEEDED остаётся terminal.

Local import и внутренний PostgreSQL queue-import adapter используют один
application handler. ArtifactSeriesCreated, ArtifactVersionCreated и physical
receipt записываются в **одной unit of work**. Artifact series имеет реальные
layer attachment scope и null frame/representation. Lineage содержит все
source versions и observation evidence. Cancellation/stale source/new plan
проверяются до публикации; поздний result отклоняется. Повтор уже принятого
receipt возможен без staging-файла, другой receipt того же job отклоняется.

Local handoff восстанавливает потерянный submit reply и import acknowledgement:
повтор сохраняет job/manifest; canonical import предшествует Agent ACK.
Отмена сохраняется прежде RPC. Файловая projection не является единственным
источником stitch state и может быть уничтожена/восстановлена.

## Пользовательский путь

В окне сшивки добавлена кнопка «Вычисляемая сшивка…». Можно импортировать JSON
наблюдения, выбрать существующую geometry revision, RGB/gray/labels, вычислить,
отменить и просмотреть сохранённый результат. JSON имеет поля команды
`plan_layer_stitch`, кроме `plan_version_id`: exact document автоматически
становится managed evidence. Физический stack и реальные source versions
настраиваются заранее; image dimensions и SHA перепроверяет worker.
В экспертном JSON явно указываются calibration, controls либо declared
transform/note; physical adjacency по именам файлов не подставляется.

Просмотр декодирует выбранный bounded block и отдельно показывает coverage;
это ещё не multiresolution обзор всей мозаики. IO, Agent и SQLite выполняются
в фоне. Потерянный mutation reply повторяется с прежним idempotency key.
Parentless QThreads удерживаются до завершения, callbacks проверяют weak Qt
owner: закрытие/удаление окна во время чтения не уничтожает рабочий поток.
Пока orchestration/polling привязан к открытому computed dialog; повторное
открытие восстанавливает работу. Общий background Pipeline ещё нужно подключить.

## Проверки

Каталоги под `.codex_tmp_scratch/production-audit-20261026/`; gates пересекаются,
их результаты не суммируются:

- `stitch-agent-native-02/`: 10 cases passed. Настоящий Agent subprocess для
  RGB/gray/labels, layer output, progress, полная проверка контейнера; corrupt
  pixels/coverage, missing/extra blocks, чужой job, cancel/budget/input guards.
- `stitch-managed-result-final/`: 7 модулей, 48 cases passed. Native/local
  HTTP Agent, атомарный импорт, source lineage, replay, reopen/rebuild, lost
  submit/ACK, durable cancel, stale refusal, rollback и соседние architecture/
  attachment/Agent cases.
- `stitch-pg-atomic/`: 2 модуля, 3 cases passed. Настоящий изолированный PG:
  HTTP/CAS/outbox и actual mosaic -> atomic import. Fault после artifact events
  откатывает и series/version, и physical receipt; повтор успешен.
- `stitch-ui-lifetime/`: 2 модуля, 8 cases passed. Настоящий Qt -> built-in
  Agent process -> worker -> import -> real bounded image viewer; coverage,
  no service IO on GUI thread, exact JSON evidence, mutation lost reply и
  удаление окна при действующем background read. Control faults не выдаются
  за отдельную hardware environment или physical-quality evidence.
- `stitch-runtime-registration/`: 4 модуля, 42 cases, 4 skips. Соседние
  source runtime/Desktop/GUI smoke tests. Это не новый installed gate.
- `stitch-backup-restore/receipt.json`: штатный canonical backup, import в
  отдельный профиль, destroy cache/reopen, metadata/lineage/blob SHA и все
  6 pixel/coverage blocks одного настоящего worker result. 17 canonical
  events, исходные события не изменены; повторное вычисление не запускалось.

Итоговый root gate `stitch-local-root-functional/`: 165 модулей,
1131 cases, 0 failures/errors, 13 skips (1118 выполнены). Все модули прошли;
PostgreSQL cases работали в отдельных UUID schemas disposable audit instance.
Выбранные изменённые модули прошли Ruff; `uv lock --check --offline` проходит.
Этот gate включает источник реализации, а не новый frozen installer.

Первые red fixtures сохранены в `stitch-agent-native/` (fixture не создавала
staging directory) и `stitch-local-atomic/` (Agent запрещал cancel IMPORTING).
Fixture исправлена, отсутствовавший Agent transition реализован; это не skips.

## Открыто

GAP-STITCH01 не закрывается. Ещё нужны server queue submission/upload/lease
generation/reclaim и remote UI, общий typed layer Pipeline scheduling/recovery,
neighbor control graph/cycle consistency, vector geometry и широкая RU/EN UI.
Не проверены большие workload/RSS/IO и latency отмены полного импорта. Source
decode/hash сейчас повторяется между блоками, bounded cache и spatial locality
ещё нужно измерить/улучшить. Валидация большого artifact выполняется внутри
unit of work; для production cancellation/concurrency нужно вынести bulk IO
перед коротким CAS commit, сохранив immutable verified content contract.
Final frozen installer, clean OS/service и полный межподсистемный 20-step E2E
остаются отдельными обязательными gates.

Позднее server queue/native execution и lease/recovery подключены и проверены:
[серверный протокол](2026-10-26-server-stitch-functional.md). Remote UI и общий
typed layer Pipeline остаются открытыми; ранний список server queue выше
описывает состояние на момент локального gate.
