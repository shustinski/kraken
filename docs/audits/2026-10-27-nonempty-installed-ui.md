# Непустая матрица в установленном Desktop

Продолжение [пространственной проверки](2026-10-27-spatial-matrix.md).
Работа направлена на функциональную готовность; обучение и оценка качества
векторизации не выполняются. Полная приёмка всех требований остаётся открытой.

## Пробел предыдущей проверки

Прежний `--self-test-dir` открывал пустой проект. Он проверял запуск Qt,
панели и Agent, но не доказывал получение управляемых кадров и миниатюр
установленным интерфейсом. Bytecode-сверка сама по себе не заменяла этот сценарий.

Новый `deployment_matrix_probe.py` импортирует настоящие managed artifacts
через обычные команды и BlobStore, создаёт новый head без собственных координат,
сверяет пространственный viewport, открывает Qt workspace, дожидается результата
фонового чтения и всех raster thumbnails. Затем создаётся ещё один head,
проверяется обновление видимого payload и повторное открытие workspace с новым
экземпляром EmbeddedProjectService. Проверяются SHA изображений, immutable ID
текущих векторов и сохранность исходных файлов после сценария.

У новых векторных head намеренно те же байты: BlobStore может переиспользовать
контент, но UI обязан показать новый version ID, а индекс — получить координаты
через цепочку родителей. Для этого сценария проверки одного SHA недостаточно.

## Два воспроизводимых режима

- Обычный `--self-test-dir`: диагностические 64 × 64 PNG red/blue и CIF,
  две занятые ячейки сетки 2 × 2. Кроме metadata, проверены центральные пиксели
  настоящих Qt thumbnail pixmap.
- `--self-test-reference-manifest`: до 16 пар из указанного JSONL manifest,
  только вместе с `--self-test-dir`. Читаются и валидируются только первые
  16 записей; SHA входных файлов обязателен. Пары копируются в новый audit-профиль
  сетки 4 × 4. Оригиналы не изменяются. Это позиции каталога, не физическая смежность.

Installer verifier принимает `--desktop-reference-manifest`. Он требует
непустую квитанцию матрицы, две смены head и проверку неизменности исходников;
пустой старый self-test больше не считается успешной Desktop acceptance.
Server такой аргумент отвергает, поскольку он относится к Desktop probe.

В этом этапе использованы первые 16 реальных JPEG/CIF из разрешённого `D:\ozi\NN`:
`production-audit-20261027/reference-load/pairs-16.jsonl`.
Пути доказательств здесь относительно `.codex_tmp_scratch/`.

## Дефект легенды, обнаруженный на реальных данных

Первый source reference probe успешно проверил данные, но его снимок
`nonempty-real-workspace.png` показал наложение подписей легенды. Этот снимок
не засчитывается как успешная визуальная приёмка.

`MatrixLegendPanel.set_rows` пересоздавал секции после каждого `cellsChanged`,
включая завершения thumbnail задач. Старые QWidget удалялись через
`deleteLater`, продолжая отображаться вне layout до deferred deletion.
Новые секции также могли ожидать отложенного show/layout.

Теперь при неизменном содержимом legend sections виджеты переиспользуются;
при изменении старые секции сразу скрываются, новые показываются в видимой
панели, layout активируется. Qt-регрессии проверяют отсутствие старых видимых
виджетов, непересекающиеся заголовки и отсутствие накопления при 100 повторных
thumbnail-only refresh. Probe дополнительно проверяет четыре видимые
непересекающиеся секции перед сохранением каждого screenshot.

Первый новый тест выявил ещё отложенный показ новых секций
(`nonempty-legend-regressions.xml`, 1 failed, 10 passed). Исправление включает
явный show; повтор `nonempty-legend-fixed.xml` — 11 passed.
Последний scoped run — 7 passed, 2 optional live skips (`nonempty-final-scoped.xml`).

## Source-проверки

- `nonempty-source-diagnostic/`: passed, два диагностических кадра,
  реальные managed import, Qt thumbnail pixels, обновление и reopen, затем
  native affine/components/stitch checks и настоящий локальный Agent.
- `nonempty-source-reference/`: 16 реальных пар, данные прошли; screenshot
  выявил описанный визуальный дефект.
- `nonempty-source-reference-fixed/`: passed после исправления легенды,
  16 реальных пар, проверены metadata, Qt thumbnails, видимый head refresh
  и fresh-service/workspace reopen. Снимок `nonempty-real-workspace-fixed.png`
  просмотрен: микрофотографии видимы, подписи легенды читаемы и не перекрываются.

Источник и установленный пакет различаются как виды доказательств. Source
результаты сами по себе не закрывают установленную проверку. Свежий экземпляр
service и повторное открытие workspace происходят в том же процессе probe;
полный restart обычного UI/login и двадцатиэтапный installed UI E2E остаются
отдельными требованиями. Этот bounded fixture также не подтверждает качество
разметки, электрическую/физическую ground truth, нагрузку нескольких пользователей,
SCM/reboot/clean OS или production update.

## Общий gate и фиксация исходников

`production-audit-20261027/nonempty-final-root/summary.json`: passed,
201 модуль (root + CSliser), 1300 cases, 0 failures/errors, 14 skipped,
1286 выполнены. Настоящий audit PostgreSQL включён. В `source-before.json`
и `source-after.json` совпадают все 2657 закреплённых source/test/tool/spec/
dependency файлов. Причины пропусков сохранены в `skipped-tests.json`;
отдельные live/model/training/E2E gate и POSIX/symlink ограничения не
засчитываются как выполненные проверки. Scoped Ruff и whitespace gate прошли.

Новый `nonempty-final-build` собран с кодами 0 (Desktop и Server отдельно).
Server также пересобран: он включает изменённый модуль легенды. Полная
bytecode-сверка упакованных модулей проекта/Contour с исходниками прошла:
Desktop 464, Server 374; `nonempty-final-{desktop,server}-sources.json`.

## Настоящие установленные проверки

Все четыре InnoSetup variants скомпилированы с кодом 0.
Production AppId variants не устанавливались и не публиковались:
`production-audit-20261027/nonempty-final-installers/production/`.

| Установщик | Bytes | SHA-256 |
| --- | ---: | --- |
| KrakenDesktopSetup-0.1.0.exe | 74 511 821 | `495c6e942f6d087808e9f9fb211802d77efd6512323a2ef87fe7b2047d619688` |
| KrakenServerSetup-0.1.0.exe | 114 846 024 | `d94c131877ffa32b10e8602e4521de5eaeb9e3fe20b37a6668baf7b301f5c8ad` |

Один отдельно идентифицированный Desktop audit installer последовательно
установлен, проверен и удалён в двух новых каталогах:

- `nonempty-final-desktop-diagnostic-installed/report.json`: passed,
  11,145 s, два диагностических кадра, проверка реальных thumbnail pixels.
- `nonempty-final-desktop-reference-installed/report.json`: passed,
  15,051 s, 16 настоящих JPEG/CIF, metadata/SHA/head refresh/reopen,
  все Qt thumbnails. `source_files_unchanged=true`. Manifest SHA-256:
  `f416dcedd4a6b6bfef97bba42186a80a6ef25f7de4f380e8d27d3d96673557f5`.

Установленный exe исполнял собственный упакованный self-test без source-tree
импортов; это новый непустой сценарий, не повтор прежней пустой матрицы.
Во втором режиме screenshot установленного приложения скопирован без изменений
в `nonempty-installed-real-workspace.png` и просмотрен: видны 16 микрофотографий,
легенда читается, старых пересекающихся подписей нет. PNG initial/update/reopen
сохранены в отдельном `desktop-acceptance/desktop-self-test-*/` каталоге каждого прогона.

`nonempty-final-server-installed/report.json`: passed, 17,987 s. Frozen
Server/Admin/Gateway: миграции, bootstrap, аутентификация, upload/Range,
SciPy extraction, graceful stop/restart и сохранение данных с настоящим
изолированным PostgreSQL. Схема UUID принадлежит только данному прогону.

Во всех трёх проверках удалены собственная регистрация audit installation и
exe, состояние существующей Windows-службы не изменилось. Server проверен
как установка файлов текущего пользователя; SCM service/reboot этим не доказаны.
Все 2657 закреплённых файлов остались неизменными после сборок и установленных
проверок; изменения аудиторских Markdown-файлов в source pin не входят.

Закрыт конкретный пробел пустой installed матрицы и исправлен найденный дефект
легенды. Полный обычный UI/login/process restart, двадцатиэтапный installed E2E,
все plugins, clean OS/Linux, production update, многопользовательская нагрузка
и остальные строки общей матрицы требований остаются открытыми.
