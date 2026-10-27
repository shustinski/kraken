# Сшивка в существующем Pipeline: функциональная проверка

Это промежуточный инженерный результат. Полная эксплуатационная готовность и
GAP-STITCH01 пока не закрыты. Длительное обучение не запускалось, качество
векторизации не оценивалось.

## Реализовано

В существующий LayerPipelineDefinition добавлен typed layer operation. Старые
frame operations сохраняют JSON и fingerprint. Layer node не требует
фиктивных координат кадров, frame ID или target representation. Workflow,
approvals, deterministic attempt IDs, reservations, retries и история остаются
в прежнем scheduler. Канонический StitchRender использует UUID attempt.
Reservation и physical request сохраняются одной unit of work.

Geometry evidence и sampled content разделены. Layer node принимает исходные
изображения плана, закреплённое raster representation или действительные
выходные версии upstream mask/confidence producer. Проверяются project/layer/
frame scope, текущие heads и lineage до geometry anchor через известные
grid-preserving capabilities. Произвольная картинка того же размера не
принимается. Binary mask использует nearest sampling. Content IDs/SHA
сохраняются в manifest, mosaic metadata и output lineage; исходные calibration
anchors не переписываются.

Local и server runtime доставляют native V2 manifest прежним Agent transport.
В server queue передаются frozen resources workflow. Проверка owner действует
и в независимом run_stitch_cycle: пауза блокирует новую доставку, отмена
фиксируется канонически и запрещает поздний import. Success требует
канонической import receipt и layer output, одного Agent finish недостаточно.
Проверки input heads/quality действуют при reservation, dispatch и import.

Qt редактор позволяет добавить/изменить layer node. Повторное сохранение
сохраняет статический content binding, algorithm version, CPU/GPU/RAM,
block size и disk budget. Исправлен поиск tuple item data в QComboBox,
из-за которого ранее выбранная geometry становилась недоступной.
Background controller исполняет layer-only workflow после закрытия диалога,
без создания фиктивной frame selection.

## Фактически пройденные gates

Каталоги ниже находятся под `.codex_tmp_scratch/production-audit-20261026/`.

| Gate | Что проверено |
|---|---|
| `pipeline-layer-native-local/` | Native layer-only Agent/import/reopen; outage перед enqueue, pause/resume с тем же attempt; cancel перед поздним import; rollback physical request и reservation вместе |
| `pipeline-stitch-live-content/` | Реальный CPU binary inference на явно необученном checkpoint → frame producer → layer consumer → native mosaic/import. В mosaic использованы действительные mask versions, geometry anchors сохранены. Обучения нет |
| `pipeline-stitch-ui-server-second/` (server module) | Реальные PostgreSQL, HTTP ServerAgentWorker и native subprocess; итог completed только после import; queue resources соответствуют node; глобальное восстановление соблюдает pause/cancel |
| `pipeline-stitch-editor-fixed/` | Qt RU/EN сохранение static representation binding и budgets, отказ недоступной geometry; прежние frame UI сценарии |
| `pipeline-stitch-desktop-native/` | Предыдущие три Qt cases и background desktop controller → настоящий Agent/native computation → completed, без frame gateway |
| `pipeline-stitch-failure-boundaries/` | Поздний QualityIssue запрещает импорт настоящего вычисленного результата; explicit exception/retry создаёт новый attempt и затем проходит native import; два конкурентных PostgreSQL ticks создают один request/queue job; ошибка после physical append откатывает оба event streams |
| `pipeline-stitch-backup-restore/receipt.json` | Canonical backup/restore настоящего mixed inference/mosaic проекта: 38 events, один workflow, шесть mosaic blocks; разрушение проекции и rebuild; сохранены workflow, artifact versions/SHA/lineage/coverage. Без обучения и пересчёта |

Ранние неудачные прогоны сохранены отдельно: ошибка capacity в тестовой
конфигурации server gate, timeout modal при выявленном отказе валидации,
затем обнаруженный дефект Qt geometry selection. Они не считаются passing
доказательствами. Финальные scoped gates выше выполнены после исправлений.

## Открытые проверки и границы

Общий root gate `stitch-pipeline-root-functional/` завершён: 171 модуль,
1155 cases, 0 failures/errors, 13 skipped (1142 выполнены). Новый модуль
`test_pipeline_stitch_failures.py` добавлен после начала root collection и
проверен отдельно: 3 cases passed в `pipeline-stitch-failure-boundaries/`.
Наборы остальных scoped gates пересекаются с root и не суммируются.
Ruff изменённых модулей, diff whitespace check и offline lock check прошли.

Нужны дополнительные concurrent command/stale failure scenarios,
нагрузка/RSS/IO, финальная установленная сборка и общий E2E.
Vector/CIF stitching с physical geometry здесь не реализован: проверен
растровый downstream content. Необученный checkpoint доказывает работу
контрактов, а не качество масок или инженерную пригодность геометрии.
