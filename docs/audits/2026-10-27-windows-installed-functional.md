# Windows: актуальные сборки и функциональная проверка 27 октября

Длительное обучение и оценка качества не запускались. Работа выполнялась в
отдельных каталогах `.codex_tmp_scratch/production-audit-20261027/`.
Исходные рабочие установки, службы и пользовательские проекты не обновлялись.

## Найденные дефекты упаковки

Встроенный `layer.stitch.v1` использует строгий CIF-кодек Contour. Прежний spec
исключал `contour` целиком. После включения достижимых codec modules frozen
проверка выявила eager импорт `contour.application` через infrastructure settings.
После его исправления настоящий install/self-test обнаружил второй eager импорт:
геометрический helper загружал permissive reader с `cv2`.

Исправления: lazy infrastructure exports с сохранением публичных имён settings;
отложенные импорты load/save vector adapters в `cif_box_polygon.py`; codec modules
доступны в frozen сборке, а Contour GUI/entry points и вычислительные Torch/OpenCV
пакеты исключены. Для контекстных ошибок CIF включены RU/EN codec resources.
Алгоритмы геометрии, точность и содержимое CIF этим изменением не заменялись.

`test_headless_cif_deployment.py` запускает свежий subprocess, запрещающий GUI,
Contour application, OpenCV, Torch и Matplotlib: hull/hole и rotated box читаются
с ожидаемыми геометрией и категориями. В self-test Desktop добавлен отдельный
private worker для raster, vector и invalid-vector. Последний должен вернуть
контекстную ошибку frame/version без output и без оставленной mosaic.

Failed evidence сохранены: `current-desktop-user-self-test/` (application import)
и `current-desktop-installed/` (cv2 import). Неудачная audit-установка удалена:
uninstall=0, registration/executable removed, service hash unchanged.

## Актуальные артефакты и доказательства

| Артефакт | Проверенный объём |
| --- | --- |
| `codec-final-build/` + `codec-final-build.log` | Rust locked release + PyInstaller Server/Admin/Gateway и Desktop на текущих исходниках |
| `codec-final-desktop-sources.json`, `codec-final-server-sources.json` | Нормализованный frozen bytecode выбранных worker/exchange/CIF/helper modules совпал с исходниками, записаны SHA |
| `codec-final-all-desktop-sources.json`, `codec-final-all-server-sources.json` | Все присутствующие в PYZ pure-Python модули проекта/Contour: 453 Desktop и 368 Server совпали с workspace source |
| `codec-final-desktop-portable/` | Frozen Desktop: OS secret store, local account/project/event readback, Qt workspace/planning/pipeline/quality rendering, NumPy/SciPy, три private stitch workers, authenticated local Agent start/stop — passed |
| `headless-codec-source-regression/` | 31 cases: 0 failures/errors, 2 skipped; actual workers, vector geometry, Agent и Desktop smoke |
| `codec-final-contour-full/` | 53 cases: 0 failures/errors, 1 skipped; settings, KLayout reader, CIF/holes/boxes regressions |
| `codec-final-desktop-installed/report.json` | Install/self-test/uninstall passed, 10.510 s; новый AppId, registration/executable removed, service hash unchanged |
| `codec-final-server-installed/report.json` | Install/frozen Server+Admin+Gateway acceptance/uninstall passed, 17.517 s; migrations/auth/upload/Range/SciPy/restart/persistence, registration/executable removed, service hash unchanged |
| `codec-final-root/summary.json` | 189 модулей root + CSliser, 1254 cases, 0 failures/errors, 14 skipped, 1240 выполнены; PostgreSQL включён, обучение не запускалось |

До и после финального root gate закреплены SHA 2630 Python source/test/tool,
spec/installer и dependency-manifest files (`source-before.json`/`source-after.json`).
Изменений в этом наборе нет. Ранний `deployment-integrated-root/` завершился
успешно, но во время него исправлялся codec; его результаты не используются
вместо финального прогона на неизменных исходниках.

Production build outputs в `codec-final-installers/production/`:

| Установщик | Bytes | SHA-256 |
| --- | --- | --- |
| KrakenDesktopSetup-0.1.0.exe | 74,432,885 | `f9103592e01b045bab8ce3c0edf1154470e6bd3b57d6e41eb6335b7410b5b13a` |
| KrakenServerSetup-0.1.0.exe | 114,776,550 | `868ff05fe3ffc261145b6a32ace793e80332a6b32927a5aaa3f36e433797b92d` |

Audit packages имеют другие SHA/AppId; их config/report хранит точные значения.
Компиляция обоих вариантов выполнялась одним сохранённым
`compile-current-installers.ps1`; pinned compiler/spec SHA есть в build configs.

Первый codec regression был запущен с default fast-filter Contour: два модуля
дали exit 5 (полностью deselected). Это не passing gate. Повтор `-m full`
выполнил нужные codec tests; сохранён отдельно. Scoped наборы пересекаются с
root и не складываются.

Portable frozen Desktop проверен настоящим ожиданием exit code дочернего
процесса; запуск GUI executable через PowerShell сам по себе не считается
доказательством его успешного завершения. Снимки workspace и planning
просмотрены: текст и контролы отображаются. Пустой deployment project не
подтверждает весь рабочий UI и массовые пользовательские операции.

Сборка фиксирует актуальные исходники, но побитовая воспроизводимость полного
установщика двумя одинаковыми сборками пока отдельно не доказана. Версия
артефактов остаётся 0.1.0; это проверяемые build outputs, не опубликованный релиз.

## Границы приёмки

Audit installers имеют отдельный AppId и устанавливаются в свежий workspace
каталог для install/acceptance/uninstall. Их SHA закреплены build config.
Production installers компилируются отдельно с обычными AppId и не запускаются
на рабочей машине. Server audit variant проверяет current-user файловую
установку; он не устанавливает SCM service и не подтверждает reboot/service
recovery. Нужны обычные production installation/update, чистая ОС, Linux,
актуальные standalone plugin distributions и полный desktop/remote E2E.

В sandbox-сеансе Windows OS secret store вернул CredWrite 1312; повтор в
пользовательском сеансе прошёл с уникальными временными credentials и их
очисткой. Эти результаты относятся к разным средам и не подменяют друг друга.

Ruff новых/диагностических modules проходит. В изменённом legacy
`cif_box_polygon.py` остаётся I001: тот же unsorted import block воспроизведён
на HEAD. Общий чистый Ruff для этого legacy файла не заявляется.
