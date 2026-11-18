# Реестр пользовательской функциональности Kraken

Дата инвентаризации: 2026-10-30. Источник — исходный код Hub, Manager, Server, Agent, Core, Blob Gateway и плагинов. Прежние отчёты в `docs/audits/` не использовались как доказательство готовности.

Статус выполнения сценариев — в `acceptance-matrix.md`. Здесь перечислено, что пользователь или оператор может инициировать.

Область текущей приёмки уточнена пользователем после `63ff7cb` (исправление `f749893`): KateGB не проверять,
реальный проект принимать пока только через серверное окружение. Перечень возможностей
плагина ниже сохраняется как описание кода, а не как обязательство его тестировать.

Сопоставление продолжения 2026-11-01: базовый коммит `8cf3ea6`, исправления `71ce8dc`;
режим D4 ниже актуализирован. Исполнение, пределы доказательств и непройденные обязательства
перечислены в актуальной секции матрицы и [checkpoint](checkpoint-2026-11-01.md).

<!-- quality-registry:generated-begin (source: docs/quality/registry/actions.json — do not edit inside) -->
## Автоматическая сводка из реестра (generated)

| ID | Поверхность | Статус | Обязательный | Тесты | Исключение |
|---|---|---|---|---|---|
| HUB-BOOTSTRAP-ACCOUNT | HUB | PASS | да | 0 | — |
| HUB-LOGIN | HUB | PASS | да | 0 | — |
| HUB-DEV-SESSION | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-CONNECT-SAVED-SERVER | HUB | NOT_TESTED | да | 0 | Отдельной строки нет; подключение имплицируется в live-блоках (live-catalog, real-project-qt-view), но само действие не принималось |
| HUB-PREFERENCES | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-UPDATE-SOURCE | HUB | NOT_TESTED | да | 0 | Собственного прогона нет; контракт URL/секрета проверен в блоке UPDATER-INSTALL / REMOTE-HTTPS (BLOCKED) |
| HUB-CHECK-UPDATES | HUB | BLOCKED | да | 0 | Не предоставлены отдельный тестовый установщик и удалённый TLS-стенд. Проверки контракта URL/секрета прошли, скачивание/запуск реального установщика и удалённый HTTPS не выполнялись |
| MGR-MATRIX-VISUAL | MGR | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-CREATE-PROJECT | HUB | PASS | да | 2 (tests/test_project_create_idempotency.py) | — |
| HUB-REFRESH-CATALOG | HUB | NOT_TESTED | да | 0 | Отдельной строки нет; перечитывание каталога имплицируется в live-catalog, но само действие не принималось |
| HUB-OPEN-PROJECT | HUB | PASS | да | 0 | — |
| HUB-BIND-WORKSPACE | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-RENAME-PROJECT | HUB | PASS | да | 0 | — |
| HUB-ARCHIVE-PROJECT | HUB | PASS | да | 0 | — |
| HUB-RESTORE-PROJECT | HUB | PASS | да | 0 | — |
| HUB-DELETE-PROJECT | HUB | PASS | да | 0 | — |
| HUB-PROJECT-PROPERTIES | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-SCAN-INTEGRITY | HUB | PASS | да | 0 | — |
| HUB-ADMIN-REFRESH | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-CONFIGURE-ROOTS | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-EXPORT-BACKUP | HUB | PASS | да | 0 | — |
| HUB-IMPORT-BACKUP | HUB | PASS | да | 0 | — |
| HUB-ATTACH-PROJECT | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-MIGRATE-LEGACY-WORK | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-PROJECT-ROLES | HUB | PASS | да | 2 (tests/test_remaining_qt_acceptance.py) | — |
| HUB-ADD-PERFORMER | HUB | PASS | да | 1 (tests/test_remaining_qt_acceptance.py) | — |
| HUB-EDIT-PERFORMER | HUB | PASS | да | 1 (tests/test_remaining_qt_acceptance.py) | — |
| HUB-DELETE-PERFORMER | HUB | PASS | да | 1 (tests/test_remaining_qt_acceptance.py) | — |
| HUB-MATRIX-CONTEXT | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-REVIEW-RETURN | HUB | PASS | да | 1 (tests/test_remaining_qt_acceptance.py) | — |
| HUB-ADD-LAYER | HUB | PASS | да | 3 (tests/test_workspace_files.py::test_png_folder_scans_and_imports_as_images) | — |
| HUB-REORDER-LAYERS | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-LAYER-CONTEXT | HUB | PARTIAL | да | 1 (tests/test_metadata_layer_deletion.py) | — |
| HUB-ADD-REPRESENTATION | HUB | PARTIAL | да | 1 (tests/test_workspace_files.py::test_png_folder_scans_and_imports_as_images) | — |
| HUB-REPRESENTATION-LIFECYCLE | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-LAYER-WORKFLOW | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-PIPELINE-STITCH-STEP | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-ISSUE-WORK | HUB | PASS | да | 1 (tests/test_remaining_qt_acceptance.py) | — |
| HUB-MY-WORK | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-WORK-PLAN | HUB | PASS | да | 0 | — |
| HUB-QUALITY | HUB | PASS | да | 0 | — |
| HUB-MULTILAYER | HUB | PASS | да | 0 | — |
| HUB-ELECTRICAL | HUB | PASS | да | 0 | — |
| HUB-STITCH-FILES | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-COMPUTED-STITCH | HUB | PASS | да | 0 | — |
| HUB-ANALYSIS | HUB | PASS | да | 3 (tests/test_karakal_agent_headless.py) | — |
| HUB-PLUGINS | HUB | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этого действия нет; сценарий не прогнан |
| HUB-STATISTICS | HUB | PASS | да | 3 (tests/test_statistics_reports.py::test_preflight_managed_import_counts_files_and_bytes_without_counting_other_versions) | — |
| SRV-HEALTH-SESSION | SRV | PASS | да | 1 (tests/test_server_administration.py) | — |
| SRV-PROJECTS | SRV | PASS | да | 0 | — |
| SRV-PRINCIPALS | SRV | PASS | да | 1 (tests/test_registered_principal_before_session.py) | — |
| SRV-LAYERS | SRV | NOT_TESTED | да | 0 | В SRV-POSTGRES-PERSISTENT проверены только создание и сохранность слоя; сканирование, импорт, переименование, порядок, архив и удаление серверных слоёв в прогоне не вызывались |
| SRV-REPRESENTATIONS | SRV | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этой группы маршрутов нет; сценарий не прогнан |
| SRV-ARTIFACTS | SRV | PASS | да | 1 (tests/test_server_managed_upload_preflight.py::test_persistent_upload_preflight_checks_revision_and_replays_existing_version) | — |
| SRV-COMMANDS | SRV | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этой группы маршрутов нет; сценарий не прогнан |
| SRV-WEBSOCKET | SRV | PASS | да | 0 | — |
| SRV-HISTORY-STATISTICS | SRV | PARTIAL | да | 0 | — |
| SRV-NOTES-CHECKS-PACKAGES | SRV | PARTIAL | да | 0 | — |
| SRV-JOBS-AGENT-PROTOCOL | SRV | PASS | да | 0 | — |
| SRV-ANALYSIS | SRV | PASS | да | 2 (tests/test_server_analysis_workflow.py) | — |
| SRV-QUALITY | SRV | PASS | да | 0 | — |
| SRV-LAYER-WORKFLOW | SRV | NOT_TESTED | да | 0 | В секции матрицы «Продолжение 2026-11-01» строки для этой группы маршрутов нет; сценарий не прогнан |
| SRV-PLANS | SRV | PASS | да | 0 | — |
| SRV-PHYSICS | SRV | PASS | да | 0 | — |
| SRV-ELECTRICAL | SRV | PASS | да | 0 | — |
| AGT-HEALTH | AGT | PASS | да | 0 | — |
| AGT-REGISTRY-CONFIDENCE | AGT | PASS | да | 1 (tests/test_karakal_agent_headless.py) | — |
| PLUGIN-CONTOUR-HELP | PLG | PARTIAL | да | 0 | — |
| PLUGIN-KRONA-HELP | PLG | PARTIAL | да | 0 | — |
| PLUGIN-CSLISER-CAPABILITIES | PLG | PARTIAL | да | 0 | — |
| PLUGIN-KATEGB-HELP | PLG | PARTIAL | нет | 0 | — |
| PLUGIN-NEURALIMAGE-VERSION | PLG | PARTIAL | да | 0 | — |
| PLUGIN-CSLISER-SELF-TEST | PLG | PASS | да | 0 | — |
| PLUGIN-CSLISER-COPY-CANCEL-RECOVERY | PLG | PASS | да | 0 | — |
| PLUGIN-KATEGB-CALCULATION | PLG | PASS | нет | 0 | — |
| PLUGIN-KRONA-LOGIC-VIEWER | PLG | PASS | да | 2 (plugins/krona/tests/test_explicit_connectivity.py) | — |
| PLUGIN-KARAKAL-COMPARE-MAP | PLG | PASS | да | 2 (plugins/karakal/tests/unit/test_headless_analysis.py) | — |
| PLUGIN-CONTOUR-DATASET-INTERACTIVE | PLG | PARTIAL | да | 0 | — |
| BLOB-PUT-GET-RESTART | GW | PASS | да | 0 | — |
| PERSISTENT-GATEWAY-UPLOAD | GW | PASS | да | 1 (tests/test_server_managed_upload_preflight.py::test_persistent_upload_preflight_checks_revision_and_replays_existing_version) | — |
| HUB-LOGIN-TABBED | HUB | NOT_TESTED | да | 0 | сценарий ещё не прогнан; палитра W3 |
| ADMIN-SERVER-BACKUP-RESTORE | SRV | NOT_TESTED | да | 0 | сценарий ещё не прогнан; палитра W3 |
| AGT-OP-VECTORIZE | AGT | NOT_TESTED | да | 0 | сценарий ещё не прогнан; палитра W3 |
| AGT-OP-DATASET-PREPARE | AGT | NOT_TESTED | да | 0 | сценарий ещё не прогнан; палитра W3 |
| AGT-OP-TRAIN | AGT | NOT_TESTED | да | 0 | сценарий ещё не прогнан; палитра W3 |
| AGT-OP-BINARY-SEGMENT | AGT | NOT_TESTED | да | 0 | сценарий ещё не прогнан; палитра W3 |
<!-- quality-registry:generated-end -->

## Как устроены поверхности

Один процесс `kraken-hub` показывает оболочку менеджера проектов. Команды пишут append-only журнал событий. Локальный каталог хранит файлы и SQLite-индексы. Общий каталог ходит в `kraken-server` по HTTP. `kraken-agent` исполняет зарегистрированные операции плагинов. Blob Gateway хранит неизменяемые объекты SHA-256. Плагины `krona`, `csliser` и `kategb` запускаются отдельным процессом и не входят в реестр возможностей агента.

## Hub и локальный менеджер

| ID | Как запускается | Ожидаемый результат |
|---|---|---|
| HUB-BOOTSTRAP-ACCOUNT | Окно «Создание аккаунта Kraken», если аккаунтов нет | Первый локальный аккаунт и сессия |
| HUB-LOGIN | «Вход в Kraken» → «Войти» | Сессия. Неверный пароль или блокировка: «Неверные учётные данные или аккаунт временно заблокирован.» |
| HUB-DEV-SESSION | Старт при development-сессии, без кнопки | Диалог входа пропускается |
| HUB-CONNECT-SAVED-SERVER | После входа, если сохранены адрес и учётная запись | Клиент сервера подключён; иначе сессия остаётся локальной |
| HUB-PREFERENCES | Вид → «Настройки…» | Адрес сервера, учётные записи, тема. Секреты в keyring |
| HUB-UPDATE-SOURCE | Вид → «Источник обновлений…» | URL или путь манифеста в QSettings |
| HUB-CHECK-UPDATES | Вид → «Проверить обновления» и тихая проверка при старте | Предложение скачать установщик или сообщение, что обновления нет |
| MGR-MATRIX-VISUAL | Вид → рамка и заливка ячейки | Режим матрицы в QSettings |
| HUB-CREATE-PROJECT | «Создать проект» | Проект в журнале, каталоги исходных и производных данных |
| HUB-REFRESH-CATALOG | «Обновить» | Список перечитан |
| HUB-OPEN-PROJECT | Двойной щелчок по проекту | Страница «Проект», матрица, слои. Меню «Действия» и «Слои…» становятся доступны |
| HUB-BIND-WORKSPACE | Вопрос «Настроить хранилище проекта» | Привязка каталогов |
| HUB-RENAME-PROJECT | «Переименовать» | Новое имя в журнале и в папках локального хранилища |
| HUB-ARCHIVE-PROJECT / HUB-RESTORE-PROJECT | «В архив» / «Восстановить» | Состояние проекта. Архивный скрыт, пока не включено «Показывать архивные» |
| HUB-DELETE-PROJECT | «Удалить» или «Запросить удаление» для сервера | Локально: имя вводится целиком, проект исчезает из каталога, внешние папки сохраняются. На сервере: заявка, проект остаётся |
| HUB-PROJECT-PROPERTIES | «Свойства» | Только просмотр |
| HUB-SCAN-INTEGRITY | Администрирование → «Проверить целостность» | Диалог «Проверка завершена» или «Обнаружены ошибки» |
| HUB-ADMIN-REFRESH | «Обновить каталог» | Проекты и до 500 строк аудита |
| HUB-CONFIGURE-ROOTS | «Настроить хранилища…» | Корни в QSettings |
| HUB-EXPORT-BACKUP / HUB-IMPORT-BACKUP | «Создать резервную копию» / «Восстановить резервную копию» | Каталог `kraken-backup-…` и проект в каталоге после импорта |
| HUB-ATTACH-PROJECT | «Подключить проект из локального каталога…» | Существующий каталог зарегистрирован |
| HUB-MIGRATE-LEGACY-WORK | «Миграция старых поручений…» | События миграции в журнале проекта |
| HUB-PROJECT-ROLES | «Участники и роли» | Назначение и отзыв ролей: наблюдатель, корректор, элементщик, сшивальщик, сопровождающий, администратор |
| HUB-ADD-PERFORMER / HUB-EDIT-PERFORMER / HUB-DELETE-PERFORMER | «Исполнители» | Исполнитель в локальном хранилище |
| HUB-MATRIX-CONTEXT | Контекстное меню ячейки | Выгрузка файлов, «Выдать работу…», «Свойства кадра…», «Свойства вектора…» |
| HUB-REVIEW-RETURN | Действия → «Загрузить проверенные файлы…» | Возврат записан в журнал и blob-хранилище |
| HUB-ADD-LAYER | «Добавить слой» / «+» | После успешного сканирования — слой и файлы. Без привязки хранилища — слой только по имени и типу |
| HUB-REORDER-LAYERS | Перетаскивание списка слоёв | Новый порядок в журнале |
| HUB-LAYER-CONTEXT | Меню слоя | Изображения, Karakal, сшивка, переименование, архив, удаление по имени, свойства |
| HUB-ADD-REPRESENTATION | «Добавить изображение» / «Добавить вектор» / «Добавить CIF из внешнего источника…» | Репрезентация и импорт |
| HUB-REPRESENTATION-LIFECYCLE | Меню узла доски | Переименование, заметка, активация, деактивация, архив |
| HUB-LAYER-WORKFLOW | «Технологический процесс…» | Определение процесса и команды запуск, пауза, продолжение, отмена, разрешение этапа, повтор ветви |
| HUB-PIPELINE-STITCH-STEP | «Добавить сшивку…» | Шаг сшивки, если геометрия слоя уже есть |
| HUB-ISSUE-WORK | «Выдать работу…» | Поручение: исправление векторов, векторизация, обучение, сшивка; размещение на накопителе или на сервере |
| HUB-MY-WORK | «Моя работа» | Обновить, запустить плагин, повторная выгрузка, принять, на доработку, отменить, повторить задание, импортировать, удалить задание, открыть, завершить |
| HUB-WORK-PLAN | «План восстановления…» | Планы, шаблоны, работы, базовый план, критерии, доска и выгрузка CSV/XLSX |
| HUB-QUALITY | «Контроль качества…» | Дефект и переходы: назначить, исправлено, проверка, принять, отклонить, исключение; дефекты из анализа |
| HUB-MULTILAYER | «Физические слои и контакты…» | Стек, калибровка, affine, кандидаты контактов |
| HUB-ELECTRICAL | «Электрическая связность…» | Компоненты, материал, контакты, выводы, частичный EDIF в панели диалога, модели MOS |
| HUB-STITCH-FILES | «Управление сшивкой…» | Версии артефакта и признак «Слой сшит» |
| HUB-COMPUTED-STITCH | «Вычисляемая сшивка…» | Импорт наблюдений, расчёт, просмотр, экспорт векторов |
| HUB-ANALYSIS | «Оценить результат», «История анализа» | Запуск Karakal, повтор, отмена, карта, экспорт, среда worker |
| HUB-PLUGINS | «Плагины» → «Открыть» | Отдельный процесс плагина, без задания агента |
| HUB-STATISTICS | «Статистика» | Расчёт и экспорт CSV/XLSX |

Команды проекта, которые локальный сервис и `POST /projects/{id}/commands` принимают из пользовательского набора: принятие и отмена проверки, активация артефакта и репрезентации, внешняя версия артефакта, архив серии, слоя, проекта и репрезентации, роли, серия артефактов, слой, заметка, репрезентация, удаление слоя, пакет проверки, переименования, порядок слоёв, возврат на доработку, восстановление проекта, правка заметки.

Заметки узлов доски, положение узлов, развёрнутая дорожка и режимы матрицы пишутся в QSettings, не в журнал проекта.

## Сервер и `kraken-admin`

Общий контракт мутаций: Bearer-сессия, для записи часто `Idempotency-Key` и `If-Match`. Ошибки — `application/problem+json`. OpenAPI: `GET /api/docs`, `GET /api/openapi.json`.

Группы маршрутов `/api/v1`:

- здоровье и сессия: `GET /health`, `POST /auth/sessions`, `POST /auth/accounts`, `GET /session`;
- проекты: список, создание, чтение, рабочее пространство, переименование, архив, восстановление, заявка на удаление;
- участники: `GET /principals`, `GET /performers`, ACL проекта;
- слои, сканирование, внешний слой, импорт, переименование, порядок, архив, удаление;
- репрезентации, план импорта, импорт, подготовка входов, кадры и viewport;
- артефакты, выгрузка, загрузка через прокси или ticket Blob Gateway, внешняя версия, скачивание;
- `POST /projects/{id}/commands`;
- `WS /ws`: ping, subscribe, unsubscribe;
- история и статистика;
- заметки, проверки, пакет и возврат ZIP, поручения и миграция старых поручений;
- задания проекта и протокол агента: lease, heartbeat, состояние, входы, публикация, выходы, complete, fail;
- прогоны анализа и карты;
- качество;
- технологический процесс слоя;
- планы работ, шаблоны, аналитика;
- физическая модель, блоки сшивки и векторы;
- электрическая модель и экспорт.

Команды `kraken-admin`: `init`, `setup-server`, `bootstrap-admin`, `recover-admin`, `bootstrap-local`, `doctor`, `blob-benchmark`, `project-create`, `project-list`, учётные записи и роли сервера, `revoke-maintainer`, `audit`, `migrate`, токены агента, служба Windows, `backup-server`, `backup-verify`, `restore-server`.

Утверждение и исполнение удаления проекта есть в коде администрирования и не выведены отдельной подкомандой `kraken-admin` и отдельным HTTP-методом. `init` записывает конфигурацию шлюза и не запускает процесс `KrakenBlobGateway.exe` сам по себе. Скрипты локального стенда лежат в `packaging/scripts/`.

## Agent

`kraken-agent` слушает loopback и пишет файл подключения. Режим `--server-url` берёт задания с сервера. HTTP: health, resources, постановка задания, статус, результат, публикации, отмена, подтверждение частичного импорта, завершение импорта, остановка.

Операции, которые Hub кладёт в реестр, если плагин включён и запускается:

| Операция | Плагин | Режим реестра (D4 актуализирован 2026-11-01) |
|---|---|---|
| `frames.vectorize.v1` | Contour | по объявлению плагина |
| `frames.dataset.prepare.v1` | Contour | interactive |
| `dataset.model.train.v1` | NeuralImage | по объявлению плагина |
| `frames.binary-segment.v1` и `v2` | NeuralImage | по объявлению плагина |
| `frames.confidence.v1` | NeuralImage | headless |
| `layer.confidence.analyze.v1` | Karakal | headless (исправление 2026-11-01; legacy JSON пока содержит только число входов) |
| `karakal.analyze.v1` | добавляет Hub | анализ |
| `layer.stitch.v1` | встроенный worker | headless |

## Core, обновления и Blob Gateway

Core задаёт контракты заданий v1/v2, контекст прямого запуска плагина, протокол анализа, тему, каталог плагинов, хранилища миниатюр и проверку обновлений Hub. TCP-реестр действий (`get_capabilities`, `open_path`, `open_folder`, `run_action`) в рабочих процессах Hub не обслуживается. `run_action` у Contour, NeuralImage и Krona всегда отвечает отказом.

Клиент обновлений плагинов (`src/updater`) даёт проверку, «Позже», скачивание, ссылку и выбор версии.

Blob Gateway: `GET /health`, `PUT/GET/HEAD /v1/blobs/{digest}` с HMAC-билетом. Объект фиксируется жёсткой ссылкой. Нет билета, неверный HMAC, чужая операция, просрочка — 401. Удаляемый проект — 409. Пауза резервного копирования — 503.

## Плагины

Запуск из списка «Плагины» не ставит задание агента.

- NeuralImage: окно, выборка, обучение, распознавание, профили проводников, контактов и памяти, старт и остановка, GPU, язык, тема, простой и расширенный режим, справка, обновления. Отдельные команды `neuralimage-client`, `neuralimage-server`, `neuralimage-config`.
- Contour: загрузка кадров и векторов, пакет, выборка, конвейер фильтров, сглаживание, возврат в Kraken по Ctrl+Shift+Return при подключённой сессии. Фильтры — шаги редактора, не операции агента: сглаживание, контраст, пороги, бинарные операции, морфология, геометрия, детекторы краёв.
- Krona: просмотр EDF и CLI `--json-out`, `--logic`, `--structural`.
- CSliser: копирование, перемещение, удаление через карантин, остановка, пресеты, восстановление, `--plan`, `--get-capabilities`, `--recover-source`, `--self-test-dir`.
- Karakal: режимы валидации и проверки ячеек, параметры JSON, история, публикация в Kraken только после управляемого запуска. Отдельный `karakal-worker`.
- KateGB: разметка Excel, генерация кадров, зашифрованный файл проверки, копирование, расчёт ошибок.

`_launch_karakal` в Hub не вызывается из живого меню. Действие Karakal со слоя идёт через агента.

Продолжение серверной приёмки после `63ff7cb` (исправление `f749893`): Server/Gateway200, 200 артефактов и
два открытия Qt PASS. D7 (native abort на HTTP500 очереди) исправлен, 36 регрессий PASS.
По явному согласию рабочая схема обновлена0011: оба job сохранены, очередь200 и Qt PASS.
Leased worker рабочего сервера BLOCKED: нет machine token (read-only проверка).
Текущие статусы и точный следующий шаг имеют приоритет в [checkpoint](checkpoint-2026-11-01.md).

Новый worker этап:393bca4/be1ca58/95ec8f2 — D8 upload fix, targeted lease, настоящий
Contour+Gateway+Hub и отказоустойчивость PASS6; токен отозван/HTTP401. Прежнее отсутствие
machine token не блокирует уже выполненный Contour блок. Следующие обязательные
NeuralImage/Karakal/stitch и пользовательский GUI запуск пока NOT TESTED, этап PARTIAL.
Точные команды/доказательства/next step — в checkpoint.
