# CSliser: передача контекста, ограничение диапазонов и Windows installation

Дата: 27 октября 2026. Этап функциональной проверки; обучение модели не запускалось.
Продолжение [предыдущего аудита](2026-10-27-csliser-functional.md).
Полная производственная готовность Kraken этим отчётом не объявляется.

## Исправления

GUI раньше принимал CLI-флаги, но не передавал их в окно. Теперь проверенный
`LaunchContext` заполняет источники, расширения, кадры, ширину, режим, операцию
и назначение. Отсутствующие параметры сохраняют настройки окна. Передача
контекста не запускает файловые операции; сохраняются preview и подтверждения.
`--settings-file` позволяет отделить INI экземпляра.

Не подключённые к окну legacy actions `open_path`, `open_folder`, `run_action`
больше не возвращают ложный успех. Они помечены deprecated в обоих manifests.
Discovery `--get-capabilities` сообщает только реально зарегистрированное
`get_capabilities`. Новый IPC-сервер или Agent job capability не добавлялись.
Поддерживаемая интеграция — запуск через Hub с аргументами CLI.

Диапазоны ограничены 65536 символами, 4096 элементами, числами до `2**63-1`,
миллионом уникальных кадров и четырьмя миллионами посещений при развёртывании.
Размер каждого диапазона/прямоугольника и общий бюджет посещений проверяются
до итерации; лимит объединения — при ограниченном накоплении. Сохраняются
порядок, дедупликация и прежняя геометрия прямоугольников. Это не обещание
постоянной памяти для произвольного числа папок и missing-frame sets.

В Windows bundle добавлен `CSliserCli.exe` для JSON/stdout; обычное окно
запускается через `CSliser.exe`. Spec содержит пути обоих source roots.
Inno поддерживает явный dist directory и отдельный audit AppId. Audit-вариант
не меняет регистрацию плагина `Software\Kraken\Plugins\csliser`, не создаёт
ярлыков и не запускает приложение после установки. Production AppId сохранён.

## Проверки

Evidence root: `.codex_tmp_scratch/production-audit-20261027/`.

| Проверка | Результат / evidence |
|---|---|
| Исходная регрессия | Четыре исходных случая воспроизвели дефекты; `csliser-context-before.xml` |
| CSliser tests | Все 7 модулей прошли; новые 14 cases: окно/CLI, отсутствие автоматических операций, legacy failure, границы/перекрытия/огромные диапазоны, discovery и CLI validation; `csliser-context-suite-fixed/` |
| Source Qt worker | 2 диагностических файла, затем 16 реальных JPEG/CIF пар — 32 файла / 33 295 580 bytes; SHA readback, отказ повторной записи, остановка QThread; `csliser-reference-source/` |
| Hub launch | Реальный `launch_plugin` создал subprocess, CLI вернул план 32 файлов без записи назначения; перехвачен только stdout/stderr. Собственный uv cache, существующий workspace venv, offline/no-sync; `csliser-hub-launch-plan.json` |
| Linux | 31 native source file-operation checks, 0 failures/errors, image digest и source manifest сохранены; сеть отключена; `csliser-context-linux/` |
| Итоговая сборка Windows | PyInstaller authoritative exit 0; `csliser-standalone-final-build.log` |
| Сверка EXE/source | По 31 модулю CSliser/kraken_core в каждом EXE совпали с текущими исходниками; `CSliser-final-frozen-sources.json`, `CSliserCli-final-frozen-sources.json` |
| Installed acceptance | Итоговый audit installer установлен в отдельную папку: discovery, 2 диагностических и 32 реальных файла через настоящие окно/планировщик/Qt worker; копии и источники SHA-проверены, overwrite отказан; `csliser-standalone-final-installed/report.json` |
| Удаление | Audit registration и EXE удалены; service ownership hash не изменён; тот же report |
| Оригиналы | Все 32 исходника из пользовательского корпуса сохранили SHA; операции делались с отдельными копиями, mapping — `csliser-reference-input/source-map.json` |
| Итоговый общий прогон | `csliser-context-final-root/summary.json`: 203 модуля, 1317 XML checks (1294 top-level + 23 subtests), 0 failures/errors, 14 skipped; PostgreSQL включён, обучение отключено; все 2671 pinned files неизменны |

Причины пропусков сохранены в `csliser-context-final-root/skips.json`: opt-in
server/Agent/UI/E2E/model gates, POSIX fork и недоступные Windows symlinks.
Scoped gates пересекаются с общим прогоном и к его числу проверок не прибавляются.
Ruff всех изменённых Python-модулей CSliser и scoped diff whitespace прошли.

Диагностический режим `--self-test-dir` копирует только в свежую собственную
папку с отдельным INI; лимит 64 файла / 512 MiB. Обработчики уведомлений
QMessageBox заменены сборщиком сообщений, файловый backend и QThread настоящие.
Move/delete или внешнее назначение в этом режиме отклоняются. Обычные
подтверждения пользовательского окна не менялись.

Первый installed screenshot выявил отсутствие шрифта у offscreen Qt. Проверка
теперь загружает Windows Segoe UI, если font database пуста, и требует glyphs
для русского/латинского текста. Кнопка настроек получила читаемую подпись.
Итоговый screenshot с реальными файлами просмотрен: русский текст читаем,
поля и кнопки видны. Старые изображения сохранены как evidence дефекта.
Это исправление диагностического окружения не доказывает дефект шрифта
в обычном Windows GUI.

Production installer с исходным AppId скомпилирован, **не установлен и не опубликован**:
28 859 678 bytes, SHA-256
`f7b885d9f0c70eb5b12c386808f90835e8aa50bec5d9eac80348841589805151`.
Audit installer: SHA-256
`51f8c65214e3e019791315363e1820b96698ad1ca5a00f603f13f3e6203a9043`.
Они лежат в `csliser-standalone-final-installer/production/` и
`csliser-standalone-final-installer/` соответственно.

## Оставшаяся работа

Последующее обновление: [source-quarantine recovery](2026-10-27-csliser-removal-recovery.md)
добавляет write-ahead receipt, OS lock, CLI/Qt восстановление и installed
process-exit proof. Следующий абзац сохраняет состояние этого более раннего этапа.

Нет durable журнала/автоматического восстановления после отключения питания;
quarantine требует разбора оператором. Не проведены clean-OS установка,
production update, reboot и обычный интерактивный сценарий Windows desktop.
Native Computer Use pipe недоступен; offscreen и внутренний Qt test не заменяют
внешний ввод в обычное окно. Не заявляются полная совместимость всех файловых
систем, большой многопользовательский load, GUI/remote/installed 20-stage E2E.
Предыдущие Desktop/Server installers `profile-final` предшествуют этой правке
каталога CSliser; их evidence не считается проверкой обновлённого каталога.
