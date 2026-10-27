# Следующий шаг GAP-STITCH01: существующий Pipeline и layer operations

Исходный read-only план scheduler и stitch adapters. После него реализован
растровый layer Pipeline и выполнены первые native gates:
[фактический протокол](2026-10-26-layer-pipeline-stitch-functional.md).
Перечисленные ниже инварианты остаются критериями приёмки; наличие этого
плана само по себе не является passing доказательством.

Текущий `PipelineOperation` требует source/target representation IDs и frame
coordinates. `infrastructure/pipeline_runtime.py` собирает FrameSelectionV1/
SubmitPluginJobCommand, а observe/cancel/recovery ищут PluginJob projection.
Для layer mosaic фиктивные frame/representation IDs неприемлемы.

Нужен tagged union прежнего frame operation и typed PipelineLayerOperation в
`domain/pipeline.py`. Legacy JSON без kind остаётся frame operation;
его serialization/fingerprint должны сохраниться. Layer operation хранит
stitch ID/revision, input/output kinds, mode/block size, dependencies,
algorithm expectation, resources, approval/quality flags, но не fake frames.

Workflow statuses, deterministic attempt IDs, retry budget, approvals,
reservations, invalidation и CAS остаются в прежнем
`application/pipeline_execution.py`. Runtime выбирает frame/stitch adapter
для prepare/submit/observe/cancel/exists/current. Существующий StitchRender
служит native job данного attempt, не параллельным execution graph.

Обязательные инварианты:

1. Scheduler attempt job UUID используется как StitchRender.job_id. Нужен
   внутренний helper создания request с заданным UUID, стабильным created_at,
   actor/fingerprint и provenance. Не подбирать другой ID через публичную
   physical command UUID formula.
2. Reservation и canonical stitch request сохраняются одной UOW с CAS по
   layer-pipeline и physical model revisions. После commit доставляется один
   неизменный native manifest. Recovery не создаёт новый attempt.
3. Request имеет immutable owner run/node/attempt. `run_stitch_cycle` проверяет
   состояние владельца: не отправляет новую работу при pause, доставляет
   durable cancel, не вводит отдельную policy. Уже действующая работа следует
   существующей pause semantics Pipeline.
4. Success требует canonical imported receipt и действительного layer output
   правильного project/layer/hash/tool version. Agent finish сам по себе не
   означает success. Retry создаёт новый UUID и сохраняет старую историю.
5. Quality gate применяется к geometry evidence и всем content inputs при
   reserve, dispatch и import. Stale/unresolved план не вызывает бесконечный
   auto retry одной и той же невалидной геометрии.
6. Frozen resources совпадают в workflow reservations и global queue admission.
   Текущую fixed reservation submit_stitch нельзя оставлять для Pipeline node
   с другой стоимостью.

Отдельно важен **реальный поток данных upstream producer**. Нельзя поставить
зависимость frame->layer, а фактически отрендерить прежние image anchors и
выдать это за использование нового segmentation/vector output. Geometry
evidence и sampled derived content при необходимости должны иметь разные
typed bindings, SHA/lineage и явную проверку pixel geometry/physical mapping.
Подстановка новых masks/vectors в calibration.artifact без доказательства
эквивалентной системы координат запрещена. Статический image-plan node может
быть промежуточным, но не закрывает окончательное требование layer Pipeline
и vector stitching.

Meaningful native gates после реализации: mixed frame->layer workflow с
реально использованным upstream output; reserve/enqueue/import crash recovery;
pause/cancel race с обоими schedulers; новое attempt при retry и bounded budget;
stale input/geometry/QualityIssue до и во время выполнения; PostgreSQL CAS race
двух tick/command процессов; старые frame-only и legacy stitching workflows.
Final installed/full E2E и нагрузка остаются отдельными обязательными проверками.
