# Фоновые операции административной панели

Этап функциональной готовности, без обучения модели.
Исходное требование §10: отзывчивость интерфейса и отсутствие блокировки
главного потока длительными операциями.

## Подтверждённый дефект

`admin-background-before.xml`: регрессия воспроизвела вызов `export_backup`
в том же потоке, что и Qt. По текущему коду аналогично выполнялись
`scan_integrity`, `import_backup`, `attach_project`, чтение каталога и
истории всех проектов. Ошибка проверки целостности также могла выйти
необработанной через Qt slot.

Это отдельный дефект от stale backup, исправленного предыдущим этапом.
Причина исходного системного окна Windows access violation не доказана.

## Изменения

- `AdministrationTasks` выполняет один service call в непривязанном к
  QWidget потоке. Worker удерживается до завершения; он захватывает только
  сервис и входные данные. Qt результаты доставляются bound slots владельца.
- После принудительного удаления родителя Qt отключает receiver slots;
  работа завершается без обращения к удалённым виджетам. Глобальное
  удержание потока и fallback ожидание `aboutToQuit` защищают завершение.
- Обычное закрытие занятой панели/её окна отклоняется без остановки event
  loop. Показаны операция, путь архива, indeterminate progress и просьба
  дождаться завершения. Обрывать запись кнопкой закрытия не предлагается.
- Конфликтующие действия и выбор проекта выключены до полного завершения
  потока. Повторный запуск не допускается; ошибки возвращают управление
  и возможность повтора. Добавлена кнопка обновления каталога после ошибки.
- Каталог и история читаются в фоне; Qt render получает не более 500
  строк аудита. Выбор проекта сохраняется при обновлении.
- Backup, restore, attach и integrity scan выполняются через тот же runner.
  Восстановление обновляет каталог. Кнопки размещены в двух колонках.

Не добавлены фиктивная отмена и имитация процента копирования. Прямое
завершение QApplication ждёт безопасного окончания начатого service call;
это fallback при обходе обычного close, а не принудительное завершение job.
При физическом сбое сохраняются существующие контракты exporter/importer.

## Фактические проверки

`admin-background-final-scoped.xml`: 17 проверок прошли, включая:

- отличный от GUI thread ID, Qt timer heartbeat при удерживаемом backup;
- отказ busy close, возврат кнопок после завершения;
- ошибки scan/import/attach и повтор каждого действия на GUI thread;
- восстановление через кнопку в свежий каталог и автоматическое обновление;
- ошибка каталога и успешный повтор кнопкой обновления;
- реальные child processes: удаление QObject parent при живом worker и
  QApplication quit с ожиданием его завершения, без QThread abort;
- предыдущие stale backup, legacy migration/lifetime и matrix probe tests.

Восемь изменённых focused modules прошли Ruff.

`admin-source-desktop/desktop-self-test-782d10258db34024b2d1e0292368ffca/report.json`:
16 реальных JPEG/CIF импортированы в собственный профиль, реальные кнопки
backup/scan/restore вызвали service/files/replay, timer Qt сработал во время
удерживаемого backup. Восстановление в независимый профиль сохранило все
managed versions. SHA исходников совпали; временный OS credential очищен.

В этом внутреннем тесте выбор пути и закрытие message boxes заменены
фиксированными ответами для своих fixtures; `ordinary_native_input=false`.
Это не ordinary native login и не полный 20-stage desktop E2E.
Снимок busy панели просмотрен; после проверки кнопки перенесены в две
колонки, статус при попытке закрытия сохраняет описание операции и путь.

`admin-nested-window.json`: дополнительная настоящая Qt проверка runner в
дочернем QWidget подтвердила отказ закрытия внешнего окна во время работы
и нормальное закрытие после завершения.

`admin-build-exit.json`: новый Desktop собран за 60.124 s, exit 0.
`admin-desktop-sources.json`: все 471 project/Contour modules, включая два
новых модуля runner/probe, совпали с source. Server не менялся и не
пересобирался: `admin-server-sources.json` повторно сверил 377 его модулей
в ранее принятой `backup-build` с текущим source.

`admin-desktop-installed/report.json`: actual audit install/acceptance/
uninstall прошли за 25.277 s. AppId
`KrakenDesktopAudit-acbe3866a1564a4c8150c7ee03d173ff`, собственные регистрация
и EXE удалены; service ownership hash не изменился.
Installed `desktop-self-test-ce03206151b74c538ad5157efced8ef3` подтвердил
`gui_heartbeat/export_outside_gui/busy_close_guarded/scan_passed=true`,
восстановил 16 реальных пар, сохранил SHA исходников и settings isolation.
Прежние matrix/head/reopen, legacy migration, freshness/restore,
native SciPy/stitch и Agent gates также прошли.
Просмотрен final installed PNG: кнопки в двух колонках, операция, полный
переносимый путь и progress читаемы, наложений нет.

Production Desktop installer скомпилирован, не установлен и не опубликован:
`admin-installers/production/KrakenDesktopSetup-0.1.0.exe`, 74 555 435 bytes,
SHA-256 `0715248f3be1c4f97752433b896026cbb530c551d701d6e08986dbab7cb805af`.
Текущие Server/CSliser installers остаются из предыдущих принятых этапов.

`admin-root/summary.json`: 210 модулей, 1379 XML checks (1348 top-level
tests + 31 reported subtests), 0 failures/errors, 15 skipped, 1364 выполнены.
PostgreSQL включён; фактическое обучение отключено. Причины пропусков —
`admin-root/skips.json`. Все 2687 pinned files не изменились во время
проверок и после сборки/установки (`post-install-source-check.json`).
Bundled Desktop plugin catalog также совпал с source. Завершение этого
прогона — 28 октября 2026; evidence directory сохранён с датой начала этапа.
Перекрывающиеся scoped/source/installed проверки не суммируются с root.

Полный 20-stage UI,
WAN/concurrent/load, native login, SCM/production AppId update/reboot/clean
OS остаются отдельными открытыми проверками общей готовности.
