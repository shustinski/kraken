# Векторная сшивка в native layer Pipeline

Промежуточный функциональный результат, без обучения модели и без оценки
качества векторизации. GAP-STITCH01 и полная эксплуатационная готовность
остаются открытыми.

## Вычисление и контракты

Layer node существующего Pipeline принимает действительные CIF versions
upstream `frames.vectorize.v1`. Scope, current heads, SHA и image lineage
проверяются до исходных geometry anchors. CIF не подставляется вместо
калиброванного изображения. Manifest отдельно хранит `content_kind=vector`
и immutable content inputs. Калибровки и measured/declared poses сохранены.

Контуры преобразуются тем же SI→reference-pixel transform, что используется
растровым renderer. SQLite R-tree служит временным spatial index; union,
clipping и сравнение выполняются по ограниченным блокам. Сохраняются категории
и отверстия. В физическом перекрытии кадров вычисляется symmetric difference
геометрии одной категории; эти области остаются явными conflicts с обеими
source versions. Union не является доказательством электрического контакта.
На границах блоков результат хранится фрагментами в общих координатах;
новая полноценная экспортируемая CIF/IR модель этим не объявляется.

Существующий immutable mosaic container получил optional `vector_blocks`
extension с GeoJSON polygons, provenance и SHA каждого блока. Legacy raster
containers сохраняют прежнюю схему и чтение. Import проверяет полноту всех
blocks, policy, sources, checksums, topology и bounds. Канонические artifact
и physical receipt фиксируются прежней unit of work. Backup сохраняет весь
контейнер как один content-addressed layer attachment.

Бюджеты явные: до 1 млн source features/10 млн source vertices, до 10000
candidate features/1 млн входных и выходных vertices на block, до 10000
overlap pairs и 8 MiB serialized vector block. Неограниченная глобальная
геометрия в память не загружается. Превышение бюджета — отказ, не усечение.
OS hard limits и большой нагрузочный gate ещё не приняты.

## CIF compatibility и неопределённость

Contour adapter использует существующие primitive conversion и authored
keyhole ring decoding. Строгий профиль принимает один `DS 1 1 1`, один
непустой слой, точный image size, P polygons и B boxes (включая поворот).
Неизвестные команды, scaling/calls, неоднозначная топология и неполный файл
дают отказ целого результата. В отличие от permissive interactive reading,
никакая неизвестная команда не пропускается с выпуском частичного результата.
Нормализация произвольных самопересечений не выполняется автоматически.

Это существенное ограничение совместимости для части реальных данных.
Официальная [документация KLayout о полигонах](https://www.klayout.org/klayout-pypi/overview/geometry/polygons/)
также предупреждает о неодинаковом отображении самопересечений и недопустимых
отверстий разными системами. Это не доказывает ошибочность конкретного CIF:
каждый отказ требует анализа кодирования или инженерного исправления.

`tools/verify_stitch_vector_corpus.py` фактически прочитал все **5498** пар
из `D:\ozi\NN`. Результат: **4944 passed / 554 rejected**, 219.059 s;
SHA исходных CIF и JPEG после чтения совпали. Из отказов 529 относятся к
invalid/ambiguous polygon topology, 3 к malformed P, 5 к вырожденному ring,
17 к exchange command (проверенные примеры содержат `L ;`).
Это gate чтения CIF и размеров, без регистрации, bounds acceptance для
всего корпуса и без изменения исходников. Общий success этому gate не
присваивается; exit code 1 и полный список отказов сохранены.

Для первого случая `0093.cif`, polygon 82, native Contour parser вернул ok,
но после explicit keyhole decoding осталась ring self-intersection.
По этой проверке нельзя объявить файл испорченным или автоматически выбрать
правильную интерпретацию заполнения. Поддержка сложного authored CIF остаётся
открытой частью совместимости.

## UI и фактические gates

Qt editor предлагает vector producer port. Block viewer рисует contour rings
зелёным и границы disagreement красным; показывает число конфликтных блоков
и неподтверждённость электрической связности. Remote API передаёт прежний
bounded PNG и маленькую extension metadata; full mosaic download не нужен.

Все каталоги — под `.codex_tmp_scratch/production-audit-20261026/`.

| Gate | Доказательство |
|---|---|
| `vector-stitch-foundation-third/` | Независимые analytic union/overlap, holes, category separation, anisotropic scale/y-axis, resource/cancel/refusal, round-trip authored CIF hole |
| `vector-stitch-container-guard/` | Включает предыдущие проверки; полный vector mosaic read, tampered provenance с новым SHA, missing block отказ |
| `vector-stitch-live-first/` | Настоящий локальный Contour → CIF versions → Pipeline → native vector mosaic → canonical import; nonempty source features/conflicts; исходные изображения неизменны |
| `vector-stitch-server-second/` | PostgreSQL graph → настоящий HTTP ServerAgentWorker/Contour → V2 stitch subprocess → full import → block API с vector metadata; adjacent Agent/media/PG regression |
| `vector-stitch-qt-first/` | Показанный Qt editor сохраняет vector edge; показанный block viewer выводит реальный PNG и uncertainty text |
| `vector-stitch-backup-restore/receipt.json` | 39 canonical events, один workflow, восемь mosaic blocks; restore/rebuild сохраняет vectors, conflicts, lineage и SHA без пересчёта |
| `vector-cif-real-corpus-full/receipt.json` | Все 5498 пар прочитаны, 554 отказа сохранены; исходники неизменны; gate не считается успешным |
| `stitch-vector-root-functional/summary.json` | 176 модулей, 1168 cases, 0 failures/errors, 13 skipped (1155 выполнены). Включены все новые native local/server/vector/Qt cases и прежние regression modules |

Ruff всех изменённых Python modules, offline lock check и diff whitespace
check прошли. Shapely добавлен в stitching extra с уже закреплённой версией
из uv.lock; новые версии библиотек не подбирались. Scoped gates пересекаются
с root и не суммируются. Полный real corpus gate остаётся отдельным failed
доказательством совместимости.

Удалённая цепочка выявила реальный defect: domain V1 manifest содержит nested
mappingproxy parameters, а ServerAgentWorker делал shallow dict перед JSON.
Теперь используется существующий recursive encode_model на границе транспорта.
Первый server gate падал на execution_resources; повторный настоящий gate
прошёл. Ранние failed gates сохранены, не заменены passing результатами.

## Осталось

Разобрать сложные CIF без молчаливого изменения геометрии; добавить полноценные
layer vector export/consumption и инженерную приёмку; проверить remote Qt
vector workflow в установленной сборке, large load/RSS/IO, общий E2E и новые
дистрибутивы. Реальных stage/control-point данных для физической приёмки
приведённые CIF сами по себе не предоставляют. Долгое обучение не требуется
и не запускалось.
