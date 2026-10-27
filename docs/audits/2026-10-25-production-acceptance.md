# Инженерная приёмка 25 октября 2026

Полная эксплуатационная готовность ещё не подтверждена. Этот протокол дополняет
[матрицу требований](2026-10-23-production-readiness.md), а не отменяет её открытые пункты.
Все пути доказательств ниже находятся относительно корня репозитория.

## Реальные изображения и разметка

Пользователь разрешил использовать `D:\ozi\NN`. Исходные JPEG/CIF не меняются;
рабочий проект создаётся отдельно через штатные application handlers.

`tools/audit_reference_dataset.py` прочитал и декодировал все **5498 JPEG** и
разобрал парные **5498 CIF** существующим загрузчиком Contour. Все изображения
имеют размер 2000×2000 RGB; непарных имён и одинаковых image SHA-256 нет.
Общий размер источников — 9 274 425 858 байт. Manifest SHA-256:
`5fda4269ed082d19f6e027fc68a2ed670ef769006a4c2d7998830f4f42200580`.

Найдены 17 CIF, которые текущий загрузчик не читает (`L ;` без имени слоя),
47 несовпадений имён изображения в заголовке и 96 пустых геометрий, требующих
проверки. Категории пересекаются: из предложенного чистого split исключены
156 записей. Пустая разметка может быть корректным отрицательным примером;
автоматически дефектом она не объявляется. Train/validation/heldout:
3148/1100/1094, последовательные блоки 60/20/20. Независимость по микросхемам
не установлена. Идентичные image bytes никогда не попадают в разные split;
группы с разными reference bytes требуют review и исключаются целиком.

Доказательства: `.codex_tmp_scratch/production-audit-20261025/real-reference-inventory/`.
`tools/materialize_reference_project.py` продолжает импорт в отдельный профиль
`real-reference-project/`. SHA каждого источника сверяется штатным BlobStore;
пары с находками получают QualityIssue. Координаты этой матрицы обозначают
позиции каталога, а не физическое соседство. CIF — предоставленная разметка,
а не подтверждённая электрическая ground truth.

## Массовый импорт и журнал событий

Реальный импорт обнаружил повторное полное чтение JSONL на каждый append,
CAS, idempotency lookup и stream read. Исправление использует временный индекс
проверенных записей в памяти процесса; канонический журнал остаётся authority.
Каждый lock hold сверяет identities сегментов, новые сегменты проходят полную
проверку последовательности/ревизий/ID. Изменённый или удалённый старый сегмент
сбрасывает индекс. `verify()` читает весь журнал. Внешние потребители получают
копии payload. Бюджет кэша — 128 MiB encoded log bytes и до четырёх проектов;
это не ограничение RSS. Для более крупных журналов работает полный replay.

Остались линейные обходы метаданных сегментов при новом lock hold; полная
асимптотика массовой загрузки пока не считается закрытой. Наблюдения отдельных
партий не являются SLA. Сохранены старый slow-import log, cProfile до/после
и resumed-import logs. Прерывание принадлежавшего аудиту процесса проверило
возобновление с сохранённых данных. Автоматическое восстановление файлового
lock проверено отдельными процессами, как описано ниже.

`test_filesystem_event_read_cache.py` проверяет новые instances, tail-only parse,
CAS, повреждение/удаление старых сегментов, неудачную частичную загрузку и
защиту общего кэша от изменения возвращённого payload. Целевой запуск с
filesystem/storage/pipeline integration: 17 passed; отдельный последующий
filesystem/migration/storage запуск: 13 passed и 5 subtests.

## Настоящие Windows дистрибутивы

Desktop собран PyInstaller и прошёл восемь фактических self-test проверок:
Windows Keyring, локальный event store, показ Qt workspace/planning/workflow/quality,
SVD registration и отдельный Agent с authenticated health и graceful shutdown.
Файл `.codex_tmp_scratch/production-audit-20261025/desktop-installer-verification.json`
фиксирует install=0, self-test=0, uninstall=0, удалённые audit registration/executable.
Использовались отдельные AppId/каталог; пользовательская установка не удалялась.
Это проверка на текущем Windows, а не clean OS. Позднейшие изменения требуют
повторной сборки и проверки.

Собранные `KrakenServer.exe`, `KrakenAdmin.exe`, `KrakenBlobGateway.exe` прошли
`tools/verify_windows_server_distribution.py`: настоящие packaged migrations,
bootstrap, production HTTP health/PostgreSQL, local authentication, blob upload,
Range download, graceful stop и повторный старт с теми же данными. **10 проверок**,
7.956 s. Отдельная UUID schema удалена; Windows service не создавалась.
Report с SHA бинарников:
`.codex_tmp_scratch/production-audit-20261025/server-distribution-audit-2/report.json`.
Первая попытка выявила ошибку fixture setup (неподдерживаемая global role),
исправлен сценарий; вторая использует штатное назначение maintainer создателю проекта.

Реальный Rust cargo test: 3 passed. Release Gateway integration: 8 passed без skip.
Исправлена дедупликация повреждённого blob того же размера: теперь сверяется
SHA-256 и возвращается conflict, чужие байты не перезаписываются.

Дополнительно закрыт риск чужой Windows service: Admin проверяет точный executable
до install/recovery-settings/uninstall; installer проверяет принадлежность перед
stop/delete. 20 targeted admin/service tests passed. Проверка собранного Server
installer, upgrade, service/reboot и чистых ОС остаётся открытой.

## Продолжающаяся работа

Board/analytics, automatic AnalysisRun quality и электрическая модель проходят
интеграционные проверки. До приёмки нужны свежая общая regression после их
интеграции, реальный corpus E2E, проверенные физические/электрические эталоны,
глобальные resource limits и длительные конкурентные/fault сценарии.

## Последующие подтверждённые проверки

Обычный общий root pytest: **846 passed, 7 skipped, 23 subtests passed**, 263.83 s.
Доказательства: `root-aggregate.log`, `root-aggregate/results.xml` в каталоге
аудита 20261025. Семь skip — Windows symlink privilege error1314; эти варианты
не объявляются проверенными. Отдельный модульный прогон обнаружил два дефекта:
общую недоступную TEMP (исправлен verifier: отдельные TEMP/basetemp/cache) и
ошибочное применение display-name лимита255 к WorkOrder destination. Поле теперь
сохраняет длинные пути без усечения; NUL/пустые/неограниченные значения запрещены.
Повторные work-order security/edge/destination проверки прошли. Новые изменения
после aggregate требуют собственных целевых проверок и следующего общего gate.

Файловый lock использует persistent `.gate` inode с kernel byte/flock locking.
ОС освобождает gate при гибели процесса. JSON record совместим с legacyO_EXCL;
живой legacy owner, чужой host и неизвестный record не похищаются. Reclaim
проходит только под kernel gate, исключая гонку между новыми reclaimers;
атомарный no-overwrite уважает конкурентного legacy writer. Собственная v2
диагностическая запись может быть очищена новым gate owner даже при живом PID,
если предыдущий release не смог удалить её из-за Windows sharing. Gate-файл не
удаляется, чтобы не создавать разные kernel objects для одного lock.
`test_filesystem_lock_recovery.py`: настоящая гибель владельца с сохранёнными
событиями, четыре процесса/60 изменений, live/dead legacy, reentrancy и временный
cleanup failure. Вместе с filesystem/cache/atomic — 17 passed.

Checkpoint импорта один раз остановился на Windows WinError5 при замене state.json.
Переход на fsynced random-temp atomic write и ограниченный retry только Windows
sharing failures проверен на действительно открытом файле. Повторный импорт
сверяет ранее записанную квитанцию и не дублирует immutable versions/receipts.
Исходная failed log сохранена.

Same-input benchmark `tools/benchmark_filesystem_event_reads.py` сравнил HEAD
reader с новым reader на одинаковых изолированных реальных сегментах, по три
повтора для current_revision+load_stream+idempotency lookup. Медианы:

| Segments/events | HEAD, s | Новый, s | Cold новый, s |
|---|---|---|---|
|256|0.0499131|0.0019047|0.0214393|
|2048|0.4325762|0.0137708|0.1731591|
|8192|1.6729617|0.0520619|0.7512215|

Ускорение warm reads26.2–32.1×; это не скорость полного импорта и не SLA.
Сохранены baseline source/SHA, identity всех входных событий и все samples:
`event-read-comparison/report.json`. SQLite/fsync/blob copy в эту меру не входят.

Frozen Server выявил дополнительный дефект: первый NumPy import внутри AnyIO
worker зависал в native `create_module`. Подтверждён повтором60s и dump всех
потоков (`server-runtime-diagnostics/threads-1.txt`), а не предположением по тайм-ауту.
Явная main-thread initialization NumPy/SciPy до worker startup исправила тот же
сценарий: HTTP extraction0.1293s; **12 packaged checks passed**, 11.412s;
`server-main-thread-verified/report.json`. Пересобранный Desktop также прошёл
девять self-test действий (`initialized-desktop-selftest/`). Source tests сами
по себе не доказывали работоспособность этого frozen runtime.

Оба обновлённых audit installer установлены, запустили настоящие приложения и
удалены: Desktop9checks, Server12checks; registration/executable removal проверены.
Доказательства: `desktop-installer-current-acceptance/report.json` и
`server-installer-file-acceptance/report.json`. SHA installers связаны с config
в `current-installers/`. Audit AppId, ярлыки и каталоги отдельные; hash существующей
service ImagePath до/после совпал. Server использовал current-user file-install
audit variant; elevated production SCM/service/reboot и clean OS этим не закрыты.


## Управляемый корпус и исторические проверки

Текст этого раздела восстановлен 26 октября из сохранённых receipts: предыдущая
запись содержала буквальные `?` вместо русских букв. Это восстановление фактов,
а не новый запуск обучения или оценка качества векторизации. По актуальному
указанию пользователя дальнейшая приёмка ограничена функциональностью;
длительное обучение и оценка качества векторизации не выполняются.

`real-reference-verification-2/report.json`: **passed**, 60.083 s. Проверены
5498 пар, 10996 серий артефактов и активных версий, 22153 события, 11155 streams.
SHA-256 исходников и управляемых blobs совпали; исходники не изменены.
Объём исходников и отдельно blobs — 9274425858 bytes. Неиспользуемых blobs и
повторяющихся receipts нет. В проекте 156 открытых quality issues: 139 minor,
17 major. Позиции кадров являются каталогом; физическая смежность неизвестна.

Ранее подготовленный dataset ZIP имеет размер 7099508659 bytes и SHA-256
`43d0534c65fa4bd0254f5a2d5336930908ad040b6ff50f166d38d0417ff2c78b`.
Он содержит 3148 train и 1100 validation пар, provenance из 8496 входных версий;
1094 heldout пары исключены. Immutable version:
`46fef765-c941-455d-8fcd-423dd0e113c1`, `artifact_role=dataset`.
Исторический job `432d19b4-cee9-49f2-896d-25392739fa7f` относится к прежней
приёмке Agent/NeuralImage; его наличие не означает нового разрешения на обучение
или подтверждённого качества. Supplied CIF не является электрической ground truth.

Исторические receipts сохраняются в `.codex_tmp_scratch/production-audit-20261025/`:

- `current-device-admission-training/`: PostgreSQL, HTTP fencing, Qt review,
  replay и малый training fixture с намерением ограничить запуск CPU. Это функциональные проверки на fixture,
  а не качество CUDA-модели на полном корпусе.
- `managed-epoch-budget/`: проверки ограничений и early stopping; подробный
  контракт в `docs/managed-neural-training.md`. Нового обучения в текущем этапе нет.
- `current-neuralimage-regression/` и `current-karakal-regression/`: исторические
  XML/logs модульных проверок, не приёмка качества на heldout.

Compute admission резервирует configured jobs/CPU/GPU/memory для очереди.
Неподдерживаемые ограничения помечены `unenforced`; это не доказательство
OS-enforced memory/CPU limits или общего распределённого per-host allocator.
Server проверяет claim/lease до выполнения worker-команд; это отдельный
контракт от контроля фактического потребления ресурсов.

Уточнение 26 октября: на данном Windows runtime Torch 2.12/cu132 реальный
subprocess с `CUDA_VISIBLE_DEVICES=''` возвращает `cuda.is_available()==True`,
а с `-1` — `False`. Поэтому прежнее название «CPU gate» по одной пустой переменной
не доказывает CPU, включая прямой запуск без Agent. Для каждого такого receipt
нужна проверка фактического `managed_execution.cuda_available`. Дополнительно
Agent раньше отбрасывал compute environment; исправленный allowlist пропускает
только CUDA visibility и три thread limits. Новые bounded model gates проверяют
фактическое отсутствие CUDA и не увеличивают лимит одной эпохи.

## Регрессии, упаковка и оставшиеся ограничения

В Linux Python 3.14.9 fork-пробы выявили наследование kernel gate и RLock.
Доказательства до/после: `linux-fork-cache-before.log`,
`linux-fork-cache-after.log`, `linux-stdlib-lock-regression/tests-2.xml`.
Это проверка механизма блокировок; полная Linux GUI/installation приёмка
этими пробами не закрыта. Wheel-прогоны в `linux-wheel-acceptance/`,
`linux-headless-wheel-acceptance/`, `linux-wheel-retry/` столкнулись с
сетевыми timeout/медленной загрузкой зависимостей; это не успешная приёмка Linux.

`current-root-integration/` фиксирует исходный прогон 975 collected cases,
1 failure, 8 skipped. Отдельные исправления и повторы сохранены в
`current-analytics-contract/`, `fork-lock-windows-regression/`,
`current-root-recovery-regression/`; первоначальный красный receipt не удалён.

Первоначальный Contour gate `current-contour-full/` имел failures.
Последующие receipts: `contour-facade-regression/`,
`contour-widget-facade-regression/`, `real-cif-regression/`,
`current-contour-full-verified/`. Фактические результаты модулей находятся в
их XML/logs/results.json. Это функциональные и регрессионные проверки;
они не заменяют приёмку качества векторизации на пользовательском корпусе.

`fenced-installers/` фиксирует изолированные Desktop 9 checks/9.039 s и
Server 12 checks/19.817 s. Audit registrations/executables удалены;
состояние существующей service проверялось до/после. Elevated SCM, reboot,
upgrade и clean OS остаются отдельными незакрытыми проверками.

Исторические исправления lease attempt/history и повторного handoff подтверждают
`lease-attempt-history-verified/`, `agent-handoff-idempotency-verified/`.
Эти receipts не являются проверкой геометрии или электрической связности.

## Функциональная приёмка качества 26 октября

На одной исходной управляемой паре `1415` выполнена изолированная цепочка:
finding → назначение связанному исполнителю → новая parent-linked версия CIF →
независимая перепроверка → приёмка → экспорт → повторное открытие и rebuild проекции.
Проверены реальные local accounts/ACL, запрет self-review, SHA-256 исходников,
сохранение старой версии и совпадение всех байтов вне исправленного заголовка `R`.
Receipt: `.codex_tmp_scratch/production-audit-20261026/quality-functional-01/receipt.json`,
**passed**, 1.267 s. Подробности и команда: [протокол](2026-10-26-quality-functional-acceptance.md).

Исправлен дефект экспорта: ошибка exclusive-create существующего destination
больше не удаляет чужой файл. Прерванный экспорт очищает только собственный файл.
Четыре отдельные регрессии прошли, включая сохранение файла другого writer,
подменившего частичный результат до cleanup. Этот сценарий проверяет только техническую
идентичность ссылки CIF на JPEG; геометрия, электрическая корректность,
UI/PostgreSQL-путь и массовая приёмка корпуса им не подтверждаются.
Отдельного API экспорта quality report нет: receipt создаёт диагностический tool;
штатные issue/history/artifact export остаются существующими API.
