# Вычисляемая внутрислойная сшивка: подтверждённый пробел

Промежуточный read-only аудит исходного задания, разделов 5/8 и шага 13 E2E.
Это план устранения GAP-STITCH01, не результат готовой сшивки.

Текущие `domain/stitching.py`, `set_layer_stitching_status` и StitchingDialog
принимают managed attachment и отмечают состояние. Это сохраняемый legacy
workflow внешних файлов; он не доказывает вычисленное геометрическое сведение.
Поиск по source/Contour/CSliser не нашёл межкадрового mosaic renderer.
NeuralImage `sew_image` собирает patches одного изображения, CSliser переносит
файлы. Их нельзя выдавать за физическую регистрацию разных кадров.

Есть пригодный `infrastructure/registration.py::fit_affine`: нормализованный
SVD fit, rank/conditioning/residual guards и честная неизвестная uncertainty.
`RegistrationFit` повторно используется. `RegistrationRevision` специально
запрещает одинаковые слои, а current calibration уникальна на слой; ослаблять
эти инварианты ради tile poses нельзя.

Реализация должна расширить существующий MultilayerModel versioned
`LayerStitchRevision` с managed plan ArtifactAnchor, а не создать второй граф
исполнения. Tile placements фиксируют реальные image dimensions, version/SHA,
калибровку/Y orientation, reference и наблюдаемые control points либо явно
объявленный transform. Declared transform не получает выдуманные residuals.
Disconnected/недоопределённые/conflicting observations остаются unresolved.
Заданный bound относится только к control points, не ко всей мозаике.

Вычисление — capability существующего Agent/Pipeline с layer-scoped V2 outputs.
Рендер — выходными блоками, с spatial footprint index и inverse affine sampling.
RGB/grayscale interpolation/blending задаются явно; binary/labels требуют
nearest без промежуточных классов. Coverage mask отделяет пропуск от чёрного
пикселя. Блоки, coverage, report и итоговый manifest — immutable blobs с полной
проверкой состава публикации; нельзя маскировать layer output фиктивным frame ID.
RBAC/CAS, input-head/calibration stale checks, lease generation, cancellation и
replay используют существующие механизмы. Legacy attachments не превращаются
автоматически в independently verified registration.

Корпус: 5498 × 2000² RGB = 21,992 млрд пикселей, около 61,45 GiB raw RGB.
Float32 RGB accumulator + weight map составили бы около 327,71 GiB. Это
арифметическая оценка без перекрытия, **не измеренный benchmark**. Полный dense
canvas непригоден: нужны bounded cache, tiled output и preflight canvas/disk.
Нагрузочный прогон должен измерять RSS, IO и latency отмены.

Существующий corpus inventory прямо говорит, что координаты каталога не
доказывают физическое соседство. Без observations/confirmed stage transforms
нельзя честно сшить этот корпус по номерам файлов. Контролируемая раскладка
допустима как functional/stress fixture с явной маркировкой; реальная physical
acceptance при этом не заявляется. Длительное обучение для устранения этого
пробела не требуется.

## Промежуточная реализация в исходниках

Добавлены `domain/stitch_plan.py`, `application/stitch_plans.py` и
`infrastructure/stitching.py`. `MultilayerModel` хранит версии
`LayerStitchRevision` в прежнем каноническом физическом stream; старые snapshots
без нового поля читаются. Команда `plan_layer_stitch` использует существующие
local/HTTP/PG adapters, RBAC MANAGE_STRUCTURE, CAS, idempotency и outbox.
Managed frame inputs и layer attachment evidence закреплены version/SHA.
Новая версия source/evidence или stack, а также последующий план, помечают
старую геометрию stale. Legacy отметки сшивки этим не заменяются.

Первый solver поддерживает явно объявленный transform либо контрольные точки
непосредственно относительно reference frame. Используется прежний normalized
SVD в SI, residuals и честный unknown uncertainty. Недостаточные/вырожденные
наблюдения сохраняются unresolved и блокируют renderer. Declared transforms
сохраняют provenance/note и не получают выдуманного измеренного residual.
Control points за границами соответствующих изображений отклоняются.
Связный solver соседских constraint edges с cycle consistency ещё не выполнен.

Renderer выдаёт ограниченные блоки с отдельным uint32 coverage, не dense canvas.
Reference calibration задаёт grid; учтены origin, pixel scale, Y direction,
affine rotation/shear/scale/subpixel placement. RGB/gray: bilinear и равновесное
смешивание перекрытий. Labels: nearest и детерминированный первый frame по ID,
без промежуточных классов. Preflight ограничивает source pixels, canvas pixels,
количество блоков/кадров и размер блока. Проверка SHA и decode dimensions есть
в VerifiedTileSources; отмена проверяется между блоками/источниками и при hash.
Это вычислительное ядро; consumer callback обязан передавать immutable inputs.

Проверки (каталоги под `.codex_tmp_scratch/production-audit-20261026/`):

- `computed-stitching-core/`: 3 модуля, 32 cases passed. Геометрические fixtures,
  roundtrip snapshots и прежние multilayer workflows.
- `computed-stitching-managed/`: 2 модуля, 17 cases passed. Настоящие managed
  images -> measured control fit -> реальный RGB block renderer; повторное
  открытие, stale source, corrupted SHA и unresolved history.
- `computed-stitching-pg-02/`: 1 case passed. Настоящий изолированный PostgreSQL,
  authenticated HTTP, обязательный If-Match, idempotent replay, reopen,
  конкурентный CAS и transactional outbox. Первый прогон retained в
  `computed-stitching-pg/`: тест ошибочно ожидал 428 вместо существующего 422
  для отсутствующего If-Match; ожидание исправлено, API не изменён.

GAP-STITCH01 **остаётся открытым**. Ещё нужны Agent/Pipeline V2 capability,
полная атомарная публикация blocks/coverage/report/manifest как layer outputs,
import/recovery/cancel fencing и UI просмотра/запуска. `render_status` намеренно
остаётся `not_computed` в долговечной assessment: прямой тест renderer не
превращает вычисление в сохранённый производственный результат. Не проверены
vector geometry clipping/merge и реальные большие workloads/RSS/IO. Реальные
данные не раскладывались по номерам файлов и physical adjacency не выдумывалась.
Длительное обучение не запускалось.

Итоговый корневой gate после перечисленных изменений:
`stitch-foundation-root-functional/` — 162 модуля, 1113 test cases,
0 failures/errors, 13 skipped (1100 выполнены). Проверки изолированы по
модулям, PostgreSQL использует UUID schemas в disposable audit instance.
Это регрессия функционального основания, не приёмка незаконченного
Agent/Pipeline/UI workflow сшивки и не оценка качества векторизации.

Следующий этап реализовал native Agent V2, indexed immutable mosaic artifact,
atomic local/PG result import, backup/restore и локальный Qt путь:
[дополнительный функциональный протокол](2026-10-26-local-stitch-functional.md).
Ранний список отсутствующих частей выше исторический; актуальные открытые
части перечислены в новом протоколе. Общий Pipeline и remote execution ещё
не готовы, GAP-STITCH01 остаётся открытым.
