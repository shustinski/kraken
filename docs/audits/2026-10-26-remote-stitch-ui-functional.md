# Remote Qt сшивка и ограниченный просмотр серверной мозаики

Промежуточная функциональная приёмка GAP-STITCH01. Качество векторизации и
физическая точность реального корпуса не оцениваются; обучение не запускается.

## Реализованный путь

Окно «Вычисляемая сшивка» поддерживает RemoteServerProjectService: exact JSON
наблюдения загружается как managed evidence, создаётся geometry revision,
запрос/отмена идут через существующие multilayer commands. Вычисление выполняет
server Agent. Desktop не запускает local Agent для remote проекта. Polling и
все HTTP/service calls идут вне GUI thread.

Просмотр использует project-authorized
`GET /api/v1/projects/{project_id}/stitch-renders/{job_id}/blocks/{column}/{row}`
с `coverage=true` для покрытия. Возвращается один bounded PNG и небольшой
заголовок: render grid, sampling/overlap, request/geometry/version и SHA.
Полный mosaic container и полный tile plan не скачиваются для просмотра блока.
Metadata/geometry/policy сверяются с canonical accepted request; output artifact
обязан принадлежать тому же project/layer attachment. Reader проверяет PNG,
bounded coverage decompression и checksums блока. Corrupt stored block -> 409;
unknown request -> 404; неверный индекс -> 422. Route наследует существующий
project_reader; без authentication запрос отвергается. Viewer membership даёт
read access, но не RUN_PLUGIN; политика доступа не ослабляется.

Logical `catalog_xy` добавлено как необязательное поле tile placement.
Оно проверяется против настоящего `project.frame_id_at`, сохраняется вместе
с revision и используется только для V2 input transport. Это не physical
pose. Старые observations без поля читаются; источник координат при request
остаётся explicit frame_coordinates либо source version metadata.

Reader вычисляет plan fingerprint по исходному сериализованному plan, до
добавления новых optional defaults. Поэтому старый v1 mosaic container не
переписывается и сохраняет прежний SHA. Типизированная geometry сравнивается
после нормализации. Выполнены и schema fixture, и backup/restore настоящего
результата предыдущего этапа.

Потерянный reply после geometry/request commit повторяется с прежним ключом.
Observation import сохраняет ключ между повторными попытками; успешный import
выбирает новую revision. Definitive validation/authorization conflict освобождает
pending command для исправления; error остаётся видимым при последующем refresh.
Так исправлена обнаруженная UI блокировка: раньше отказ на unresolved geometry
мог бесконечно повторяться и не позволял исправить выбранную команду.
Ошибки связи сохраняют idempotency key, поскольку commit может быть неизвестен.

## Проверки

Evidence под `.codex_tmp_scratch/production-audit-20261026/`:

- `stitch-remote-foundation/`: local UI/geometry/native container compatibility.
- `stitch-remote-native-ui-03/`: remote actual Qt + uvicorn HTTP + PG + Agent
  subprocess, observation upload, request, import, real image/coverage viewer;
  local UI and native container neighbors.
- `stitch-remote-integrity-final/`: тот же путь с damaged stored coverage -> 409,
  bounded response, invalid/foreign IDs и PG/legacy/local UI cases.
- `stitch-remote-lost-replies/`: control faults после **настоящего server commit**
  для geometry и compute request. Повторяет тот же ключ, создаёт одну geometry
  revision и один request; local definitive rejection и catalog/frame mismatch.
  Faults моделируют утраченный ответ клиента и не выдаются за физическую
  сетевую аварию/другую hardware environment.
- `stitch-backup-restore-schema-compat/receipt.json`: сохранённый native result
  предыдущего этапа восстановлен штатным backup/import после schema extension;
  17 events, 6 blocks, прежний blob SHA, lineage и physical snapshot проверены
  после уничтожения cache/reopen. Recompute и training не выполнялись.

Начальный `stitch-remote-native-ui/` завершился **двумя connection timeouts**:
при проверке состояния ранее используемый audit container отсутствовал.
Стенд восстановлен отдельным контейнером `kraken-audit-20261026-74ef973a`,
tmpfs и свободным loopback port. Current connection metadata:
`postgres-current.json`, URL включает `connect_timeout=5`.
Это не application test pass; timeout evidence сохранена.

В `stitch-remote-native-ui-02/` actual remote Qt case прошёл, отрицательная
fixture ошибочно ожидала read access без project membership. Она исправлена:
сначала доказан 403 без membership, затем назначен VIEWER; read=200,
compute=403. Существующая политика application не изменялась.

Последний root gate `stitch-remote-root-functional/`: 167 модулей,
1143 cases, 0 failures/errors, 13 skips (1130 выполнены). Изменённые focused
модули прошли Ruff; `uv lock --check --offline` проходит. Это проверка
исходников и реального native workflow на текущем стенде, не новый installed
или clean OS gate. Следующий integration plan:
[typed layer Pipeline](2026-10-26-layer-pipeline-stitch-plan.md).

## Открыто

Общий typed layer Pipeline и autonomous local orchestration вне computed
dialog, соседский constraint graph/cycle consistency, vector geometry,
большие workload/RSS/IO, short-CAS bulk import/cancellation, runtime provenance
stitch worker, RU/EN всего нового UI и final installed/clean OS остаются
незавершёнными. Новый API даёт bounded block preview, а не multiresolution
обзор всей большой мозаики. GAP-STITCH01 и полная readiness ещё открыты.
