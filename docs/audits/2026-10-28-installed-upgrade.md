# Обновление установленной системы без потери данных

Этап функциональной готовности, без обучения моделей. Артефакты:
`.codex_tmp_scratch/production-audit-20261028/`.
Инструкция оператора: [updating](../deployment/updating.md).

## Изменения

Установщики Desktop/Server ранее не задавали CloseApplications, то есть
полагались на стандартный Restart Manager. По
[документации Inno Setup](https://jrsoftware.org/ishelp/topic_setup_closeapplications.htm)
тихий установщик по умолчанию сам закрывает приложения, использующие его
файлы. Это установленное по исходному коду и контракту поведение; предыдущий
установщик с работающим пользовательским приложением здесь не запускался.

Теперь `CloseApplications=no`, `RestartApplications=no`. Общий
`packaging/installer/DeploymentGuard.iss` проверяет каталог приложения перед
первой заменой файлов: существующие EXE/DLL/PYD открываются без записи bytes
через OPEN_EXISTING с запросом write/exclusive sharing. Занятые/защищённые
файлы и reparse entries дают отказ до стадии [Files]. Uninstaller также
отклоняет удаление занятого приложения. Проверка не останавливает процессы,
не вызывает SCM и не удаляет файлы. Длинные/UNC paths передаются как extended
Windows paths. Действующие ownership checks удаления службы сохранены.

AppVersion можно задать при компиляции; production default остался 0.1.0.
Для теста скомпилированы пары 0.1.0 → 0.1.1 с одинаковым собственным audit
AppId каждой пары. Это labels фикстур, не публикация выпуска 0.1.1.
Payloads различны: прежние принятые EXE из `20261027/backup-build` и
`20261027/admin-build` заменяются текущими Server и Desktop.

Добавлен `tools/verify_windows_upgrade.py`: проверка SHA пакетов, собственных
регистраций/путей, всех payload files, работы старых данных до/после замены,
отказа на busy update/uninstall, удаления только своего приложения.
Непоставляемый файл в каталоге приложения сохраняется и после uninstall.

Desktop получил private deployment-only `--self-test-reopen-report` с
обязательным `--self-test-dir`. Принимается только завершённый изолированный
Desktop self-test report, связанный со своим каталогом и bounded matrix.
Новая EXE читает старый profile, аутентифицирует его аккаунт, проверяет
прежние версии и реальные blobs, открывает Qt workspace. Не создаёт замену
старому проекту. После чтения сверяются canonical events/objects/snapshots/
descriptors и прежние settings. Пространство Qt/secret lookups изолировано.
Login добавляет обычную локальную session; перестраиваемые SQLite caches не
выдаются за неизменяемую историю. Этот режим не является ordinary native UI
login; `ordinary_native_input=false` записано в report.

## Фактические установленные проверки

`server-upgrade-installed/report.json`: passed, 79.414 s,
AppId `KrakenServerAudit-92060bf527ac4d67ada8613d1333555f`.

- Прежние настоящие Server/Admin/Gateway установлены в свой каталог с пробелом;
  все payload SHA сверены. Созданы только две свои UUID PostgreSQL базы.
- Прежний Admin выполнил migrations/bootstrap; прежний Server принял аккаунт,
  проект и 32 файла из 16 разрешённых JPEG/CIF пар `D:\ozi\NN` через Gateway.
  Здесь файлы являются project attachments: проверяется история/хранение,
  а не физическая раскладка или векторизация.
- При работающем Server+Gateway новый installer вернул **7** в PrepareToInstall,
  Win32 sharing error 32 на `KrakenBlobGateway.exe`. PID/HTTP сервера и все
  старые payload SHA сохранились; установка не закрыла приложение.
- После штатного stdin stop обновление вернуло 0. Все новые payload SHA совпали
  с новым build. Новые migrations, прежний аккаунт, неизменная project history,
  все 32 SHA и HTTP Range после нового запуска прошли. Конфигурация и защищённые
  секреты совпали по SHA.
- Попытка удалить работающий обновлённый Server вернула **1**; его регистрация,
  PID и все новые payload SHA сохранились.
- Новый Admin сделал backup/verify/restore уже обновлённого старого проекта
  в свежую базу и новые файловые корни. Новый Server подтвердил прежний аккаунт,
  history и все 32 SHA/Range из восстановленной копии.
- Obsolete baseline files не осталось. Свои базы, регистрация и EXE удалены.
  Непоставляемый пользовательский файл и все исходники сохранены; service
  ownership hash до/после null. Windows service не создавалась.

`desktop-upgrade-installed/report.json`: passed, 55.674 s,
AppId `KrakenDesktopAudit-9fe7d57f798e4fc3b9ac2e1eba24d8be`.

- Прежняя EXE реально работала с 16 парами в собственном self-test profile.
  Попытка обновления вернула 7 до замены файлов; PID продолжил работу,
  исходный payload совпал по SHA. Прежний self-test завершился успешно.
- После обновления все файлы новой установки совпали с build.
  **Установленная новая EXE** открыла именно прежний profile: аккаунт прошёл
  аутентификацию, 16 matrix cells/32 bytes SHA и два historical heads прочитаны,
  78 canonical history events доступны. Настоящая Qt matrix получила правильные
  heads и непустые raster thumbnails. Канонические файлы и старые settings
  остались идентичны. `previous-project-reopened.png` просмотрен: 16 кадров
  видны, разделы легенды не перекрываются. Это программная Qt fixture.
- Полный self-test новой EXE дополнительно проверил свежие 16-pair import/head/
  reopen, legacy/freshness, administration backup/scan/restore, heartbeat/busy
  close, native SciPy/stitch и Agent. Он не подменяет отдельный reopen старого
  проекта. Secret-store keys собственных fixtures очищены.
- Obsolete baseline files отсутствуют. Audit registration/EXE удалены,
  непоставляемый файл и оригиналы сохранены, чужая служба не затронута.

`upgrade-source-reopen` отдельно проверил чтение прежнего принятого profile
от 27 октября текущим source runtime. Он не суммируется с installed acceptance.
`upgrade-desktop-build-exit.json`: exit 0, 58.097 s.
`upgrade-desktop-sources.json`: все 473 frozen project modules совпали с source.
Server/Admin runtime не менялся после предыдущего этапа: по 380 modules;
изменена их installer policy, а не их Python runtime.

`upgrade-root/summary.json`: 213 изолированных модулей, 1417 XML checks
(1386 top-level + 31 reported subtests), 0 failures/errors, 15 skipped,
1402 выполнены. PostgreSQL/native PG tools включены, все training flags=0.
`skips.json` сохраняет причины пропусков. Все 2700 pinned files, включая
installer specs/include, новые probe/verifier и Rust source/Cargo, совпали
с SHA до прогона и после установок (`post-install-source-check.json`).
Новые CLI и проверка отказа чужому profile входят в этот root. Ruff новых
Python modules/tests/verifier прошёл. Результаты предыдущего root и отдельных
installed gates к этому числу не прибавляются.

Production installers перекомпилированы с общей проверкой занятости,
не установлены и не опубликованы. Каталог `upgrade-production-installers`:

- `KrakenDesktopSetup-0.1.0.exe`, 74 563 180 bytes,
  SHA `fb034589cbdaec560c6380d1a5dc95b0b219a3f691560abdd03e572cb04374bc`;
- `KrakenServerSetup-0.1.0.exe`, 114 982 954 bytes,
  SHA `05fad345a3a31315c7aebe1b754fc24f6d8e2ba67569d7ba42066c0c51f58308`.

## Открытые границы

Проверена замена файлов в audit current-user установках, использующих общую
с production политику проверки занятости. Production AppId, elevated SCM,
обновление запущенной службы с drain/restart, clean OS/reboot и физический
power-cut не приняты. Preliminary file probe не исключает запуск другого
процесса после проверки; установка остаётся операцией в окно обслуживания.
Atomic directory switch/rollback всего runtime и произвольных schema migrations
не реализованы этим изменением. Совместимость всех исторических выпусков этим
одним baseline не объявляется. Обычный native UI/login и исходный python.exe
access violation остаются непроверенными; Computer Use pipe недоступен.
Полная производственная готовность по §§1–19 не заявляется.
