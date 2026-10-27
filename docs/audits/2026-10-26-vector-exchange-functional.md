# Экспорт/чтение геометрии слоя и диагностика CIF

Продолжение [native vector stitching](2026-10-26-vector-stitch-functional.md).
Полная эксплуатационная готовность не достигнута. Обучение и оценка качества
векторизации не запускались.

## Реализовано

Принятую мозаику можно экспортировать из Qt через local или remote service,
а также CLI. [Контракт обмена](../layer-vector-exchange.md) сохраняет исходные
опорные пиксельные координаты, вычисленные SI координаты без квантования,
отверстия, категории, source versions/SHA и обе стороны disagreement.
Новые файлы включают проверенную raster coverage: no-data не превращается в
измеренный пустой фон. Ранний v1 без маски остаётся читаемым с unknown coverage.

Header и каждый блок проверяются по project/layer/job/output identity,
каноническому manifest и принятому artifact. Remote API использует прежний
project reader access; geometry передаётся по блокам, без full mosaic download.
Локальный exporter разрешает immutable source в короткой UOW и отпускает
project write lock до сериализации. Это исторический snapshot результата,
не декларация текущей физической или электрической приёмки.

Writer создаёт временный файл, проверяет весь stream отдельным reader и
публикует новое имя. Существующий destination не заменяется. Отмена, budget,
прерванная сеть, изменённая output identity и неверный seal не оставляют
принятого частичного файла. Cleanup проверяет identity собственного staging
file. Все blocks, включая пустые, обязательны. Потребитель должен полностью
исчерпать iterator и проверить `reader.verified`, прежде чем принять результат.
Объекты — фрагменты блоков, не автоматически признанные проводники или nets.

## Выявленные дефекты

1. Permissive Contour converter мог пропустить вырожденный P primitive и
   вернуть неполный набор. Строгий adapter теперь проверяет distinct vertices
   и совпадение числа primitives с decoded polygons; такой файл целиком
   отклоняется. Повреждение геометрии автоматически не исправляется.
2. Отказ CIF ранее не указывал место. `CifStitchError` содержит error code,
   polygon ID и command index; native worker дополняет сообщение frame ID и
   artifact version ID. Полный отчёт отказа доходит в прежний canonical error
   и виден в существующем UI. Для `0093.cif`: polygon 82, command 86, ring
   self-intersection `[1842 1870]`; это не автоматическое решение о правильном
   заполнении неоднозначной геометрии.
3. Первый общий gate этого этапа упал: Windows запретила `os.replace` progress
   JSON, пока Agent держал read handle. В core вынесен существующий bounded
   Windows retry атомарного rename, используемый filesystem и stitch worker.
   In-place write и предварительное удаление destination не применяются;
   постоянная ошибка permissions не скрывается. Native Windows test держит
   настоящий открытый handle, проверяет старые bytes, затем освобождает его
   и проверяет полную замену для обоих writers. Временные files очищаются.

## Свежие доказательства

Все каталоги — `.codex_tmp_scratch/production-audit-20261026/`.

| Gate | Проверка |
|---|---|
| `vector-exchange-second/` | SI origin/anisotropy/y-axis, holes/global union, provenance/conflicts; resealed wrong SI/source, duplicate/truncated/extra stream; cancel/budget/source identity; существующий destination сохранён |
| `vector-exchange-live-first/` | Local Contour→mosaic→export/re-read без новых событий; PostgreSQL/HTTP native chain→Qt export→complete reader, без скачивания полного mosaic |
| `vector-exchange-remote-qt-recovery/` | Настоящий HTTP/PG/Qt; потерян ответ второго блока — нет final/part file, UI разблокирован; существующий файл не изменён; повтор дал идентичные bytes |
| `vector-cif-refusal-native-fixed/` | Вырожденные и ambiguous CIF не пропускаются; настоящий subprocess failed publication без output, с frame/version/polygon/command |
| `vector-exchange-sharing-fixed/` | Реальная Windows reader sharing regression, native local pipeline, worker и security tests после исправления |
| `vector-exchange-coverage/` | Новая и ранняя v1 compatibility; holes/no-data coverage, corrupt mask; повторные actual local и remote Qt export |
| `vector-cif-real-corpus-strict-complete/receipt.json` | Повторный read-only проход всех 5498 пар: 4944 прочитаны / 554 отказа, 137.564 s. 529 ambiguous_topology, 3 malformed_polygon, 5 degenerate_polygon, 17 empty_layer. Источники по SHA неизменны; gate не passing |

CLI действительно экспортировал ранее вычисленный проект
`629fce68-495b-4a32-8401-d9c65438b38d`, job
`cdcece84-8ab4-5d56-a47d-8dce6a6c0b0e`: 8 blocks, 6 feature fragments и
2 disagreement fragments. События исходного проекта не изменились.
Файл с coverage — `vector-exchange-coverage-cli.jsonl`, SHA
`bf28eca1f3ab0d63eb0f2eeede9106f3f87038f9dc5415e1a009921e0f6b5ee8`.
Это реальный accepted mosaic, без пересчёта и обучения.

Первый Linux source exchange gate успешно установил pinned wheels, но не смог
загрузить NumPy shared object с temporary filesystem. Стенд исправлен явным
exec mapping для этого isolated tmpfs. В `vector-exchange-linux-roundtrip-fixed/`
Windows→Linux→Windows ранний v1 прочитан и повторно записан byte-identical,
затем полностью проверен Windows reader. CPU=1, RAM=768 MiB, pids=64,
read-only root, cap-drop ALL; пользовательские сервисы и репозиторий не
монтировались, использовался source snapshot. Это не installed wheel/desktop
gate и не сертификат всех платформ.

`vector-exchange-linux-coverage/receipt.json` подтвердил тот же round trip
для актуального v1 с coverage. Восемь блоков и весь файл byte-identical,
Windows readback прошёл; SHA совпал с указанным выше CLI экспортом. Legacy и
новый stream проверены на NumPy 2.4.4 / Shapely 2.1.2 из действующего lock.
Первый failed Linux gate сохранён отдельно и не засчитывается.

`stitch-vector-exchange-root-functional/` завершился с 1 failure (sharing
progress JSON). После исправления `stitch-vector-exchange-root-fixed/`
прошёл: 177 модулей, 1182 cases, 0 failures/errors, 13 skipped (1169 выполнены).
Последний общий gate после coverage — `stitch-vector-exchange-coverage-root/`:
**177 модулей, 1184 cases, 0 failures/errors, 13 skipped (1171 выполнены)**.
Ruff изменённых modules, diff whitespace и offline lock checks прошли.
Scoped наборы пересекаются с root и не суммируются; полный real corpus gate
с 554 отказами остаётся отдельным непрошедшим доказательством совместимости.

## Оставшаяся работа

Нужны downstream electrical/IR consumers с отдельной инженерной приёмкой,
разбор 554 CIF отказов, большой нагрузочный/E2E gate, OS hard resource limits,
новые дистрибутивы и их установленная приёмка. NDJSON geometric exchange не
объявляется готовой интеграцией каждого внешнего CAD/editor или netlist.
