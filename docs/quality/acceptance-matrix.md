# Матрица приёмки

## Продолжение 2026-11-01 (база `8cf3ea6`, исправления `71ce8dc`)

Прежние PASS ниже сохраняются. Уточнения этой секции имеют приоритет над историческим
«Не прогнано». Сырые журналы и команды: [checkpoint](checkpoint-2026-11-01.md).

Уточнение области после `63ff7cb` (исправление `f749893`): KateGB исключён из обязательных проверок по
прямому указанию пользователя. Существующие результаты ниже сохраняются исторически;
новые проверки KateGB не выполняются. Реальный проект принимается пока только
через серверное окружение; локальные PASS его приёмку не заменяют.

| ID / блок | Статус | Подтверждение и граница |
|---|---|---|
| LIVE-CONTOUR-MACHINE-WORKER | PASS | 393bca4: новый серверный project2×1, реальные upload tickets/Gateway/Contour subprocess, heartbeat/publication/complete,2CIF versions/hash/size/type/input lineage и2HTTP+Qt reopen. [Receipt](evidence/2026-11-01/live-contour-worker-receipt.json), [JUnit](evidence/2026-11-01/live-contour-worker-junit.xml). User test/jobs unchanged |
| PERSISTENT-GATEWAY-UPLOAD / D8 | PASS | Реальный HTTP500, PG regressionRED→GREEN; preflight передан зарегистрированному handler. Новая регрессия1PASS, смежные19PASS. [GREEN](evidence/2026-11-01/upload-preflight-green.txt) |
| MACHINE-TARGETED-LEASE | PASS | be1ca58: unknown/invalid/denied не меняют очередь, конкретный второй job выдаётся без изменения первого. [14PASS](evidence/2026-11-01/target-lease-green.txt); review finding closed |
| LIVE-WORKER-FAULTS | PASS | 95ec8f2: [6-test block](evidence/2026-11-01/live-worker-contour-faults.txt): unavailable/connection, auth401, CAS409, real Qt cancel/errors, actual plugin failure/bad PNG, real lease expiry/reclaim/fencing, duplicate result, TCP disconnect after commit, no duplicate versions; protected project/jobs unchanged |
| MACHINE-TOKEN-REVOKE | PASS | Dedicated Contour token revoked after block; [HTTP401](evidence/2026-11-01/machine-token-revocation-check.json). No secret in evidence |
| AGT-REGISTRY-CONFIDENCE / D4 | PASS | `test_karakal_agent_headless.py`: headless принимается, worker subprocess завершается, новый DurableJobStore читает результат, result.json и confidence.json перечитаны, SHA-256 совпал. Действие Hub доступно. `d4-focused.log`: 7 PASS вместе с режимами агента |
| KARAKAL-CONFIDENCE-METRICS | PARTIAL | Выход legacy confidence содержит только operation/job_id/input_count; запуск и JSON не доказывают расчёт уверенности |
| SRV-BACKUP-RESTORE-ISOLATED | PASS | Собственный временный кластер PostgreSQL 16 на loopback:55439. 20 + 6 PASS в `p0-backup-live.log`, `p0-backup-remaining.log`. Native backup/verify/restore, CLI, защита непустой цели, испорченные архивы, восстановление путей и барьеры шлюза. Рабочая база не использовалась |
| SRV-REGISTRATION-POLICY | PASS | Политика репозитория допускает самостоятельную ordinary-регистрацию: нет global roles, чужие проекты скрыты, неверный пароль отклонён. `test_server_administration.py` в `p0-policy-network.log`. Приглашения не являются текущим контрактом |
| QT-PLAN / QT-PHYSICS / QT-ELECTRICAL / QT-COMPUTED-STITCH | PASS | `qt-acceptance.log`: реальные показанные окна, щелчки, запись/переоткрытие и ошибочные данные. Весь блок 33 PASS; часть вспомогательных проверок использует адаптеры, они не подменяют тесты настоящего сервиса |
| QT-QUALITY | PASS | `quality-clicks.log`: видимые кнопки создания/назначения/ошибочного исправления, refresh, новый EmbeddedProjectService и повторное окно читают assigned и историю |
| SRV-AGENT-KARAKAL / SRV-AGENT-STITCH | PASS | `server-acceptance.log`: настоящий PostgreSQL, HTTP, ServerAgentWorker, subprocess Karakal/сшивка, повторное чтение, retry/cancel/runtime refusal, Qt удалённой сшивки. Блок 127 PASS |
| PLUGIN-CSLISER-COPY-CANCEL-RECOVERY | PASS | `plugins-acceptance-importlib.log`: показанное окно, копирование с проверкой байтов, отказ перезаписи, отмена, карантин/восстановление, архив и повторное открытие проекта |
| PLUGIN-KATEGB-CALCULATION | PASS | Историческая проверка до уточнения пользователя: Alice/Bob — по 50%; зашифрованный манифест перечитан. Дальнейшие проверки KateGB исключены |
| PLUGIN-KRONA-LOGIC-VIEWER | PASS | `test_explicit_connectivity.py`, `test_quality_qt.py`: реальный CLI на EDF, таблица NOT, hash исходника; показанный viewer отображает неопределённость. Не приёмка реального прибора |
| PLUGIN-KARAKAL-COMPARE-MAP | PASS | `test_headless_analysis.py`, `test_server_analysis_workflow.py`: контролируемые пары масок, численные XOR/IoU/Dice, карта кадра и повторное чтение |
| PLUGIN-CONTOUR-DATASET-INTERACTIVE | PARTIAL | Интерактивные proxy/gateway пути до возврата прошли в `server-acceptance.log`; headless выборка не заявлена и отдельно не выполнена |
| LOCAL-PROJECT-ORPHAN-RECOVERY | PASS | Независимое ревью нашло удаление файлов orphan workspace. Три RED; исправлено сохранением совпадающей привязки. `d1-review-green`: 37 PASS с регрессиями создания |
| QT-ROLES / QT-PERFORMER / QT-ISSUE-WORK / QT-REVIEW-RETURN | PASS | `test_remaining_qt_acceptance.py`: реальные модальные окна, assign/revoke и reread, исполнитель после reopen, отказ без каталога, пакет с байтами; изменённый CIF через preview → принять → подтверждение → completed и новый blob после reopen |
| QT-INTEGRITY-MESSAGES | PASS | Настоящие QMessageBox «Проверка завершена» и «Обнаружены ошибки» после порчи собственного временного blob, видимая кнопка |
| QT-STATISTICS-EXPORT | PASS | Два подтверждённых отказа исправлены (D6): импорт 2 файлов считался как 0, CSV падал на nested mappingproxy. `statistics-verified`: 18 PASS, включая calculate/CSV/XLSX кнопками |
| HUB-DUAL-LIVE-WS-DELETE-REQUEST | PASS | `live-catalog`: PostgreSQL + живой HTTP + DualCatalogService, local/remote вместе, websocket/outbox wake и catalog_changed, rename не трогает local; заявка pending, проект не удалён |
| SRV-QUALITY-PHYSICS-ELECTRICAL-PLANNING-QUEUE | PASS | `server-domain-remaining`: 30 PASS на отдельном кластере, HTTP/RBAC/reopen/CAS/rollback; 11 пропусков queue из первого agent-прогона затем исполнены |
| AGENT-REGRESSION | PASS | `agent-regression`: 74 PASS, 2 subtests; 11 PG skips закрыты отдельным `server-domain-remaining` |
| CONTOUR-EDITOR-FILTERS | PARTIAL | `contour-editor-full`: 323 PASS, 1 FAIL теста преждевременного доступа к async display, 1 skip отсутствующего 0518.cif. Ожидание теста исправлено; отдельный `contour-hole-recheck` PASS. Отсутствующая фикстура не подменена |
| KRONA-JSON-OUT | PASS | CLI `--json-out` на `resources/test_edifs/explicit_cmos_inverter.edf`, JSON повторно прочитан, содержит cell/design/instances/library/nets/ports/view |
| PNG-SCANNER-MANAGED-PARITY | PASS | Один PNG через два публичных пути импорта, после нового сервиса одинаковые SHA-256 и байты; `png-parity` |
| WINDOWS-SERVICE-INSTALL | BLOCKED | `WindowsPrincipal.IsInRole(Administrator)=False` вне sandbox. Служба рабочего сервера не изменялась; отдельная elevated VM/учётная запись не предоставлена |
| WINDOWS-HELLO-CREDENTIALS | BLOCKED | Аппаратное/пользовательское подтверждение Hello невозможно заменить offscreen. Нужен интерактивный пользовательский проход; тесты с подменой проверки не считаются приёмкой Hello |
| UPDATER-INSTALL / REMOTE-HTTPS | BLOCKED | Не предоставлены отдельный тестовый установщик и удалённый TLS-стенд. Проверки контракта URL/секрета прошли, скачивание/запуск реального установщика и удалённый HTTPS не выполнялись |
| REAL-PROJECT-SERVER-READ | PASS | Server/Gateway200; 100 PNG + 100 CIF полностью прочитаны через Gateway, hash/размер/декодирование/размеры пар; история и revision2 сохранены. [Результат](evidence/2026-11-01/real-project-artifact-verification.json) |
| REAL-PROJECT-SERVER-QT | PASS | Реальный Hub/DualCatalog/HTTP, два открытия двойным щелчком после закрытия окна; слой и две репрезентации, нет uncaught/errors, ревизия сохранена. [Квитанция](evidence/2026-11-01/real-project-qt-view.json) |
| HUB-QUEUE-HTTP-FAILURE / D7 | PASS | Прежний native abort воспроизведён; ошибка plugin_jobs теперь внутри try. [36 PASS](evidence/2026-11-01/hub-resume-green.txt), повторный успешный запрос200, независимое ревью |
| REAL-SERVER-JOBS-SCHEMA | PASS | После явного согласия0011 применена транзакционно; оба задания сохранены, JSON nullable, очередь HTTP200. [Квитанция](evidence/2026-11-01/schema-upgrade-working.json), [Qt после миграции](evidence/2026-11-01/real-project-qt-after-migration.json) |
| REAL-PROJECT-SERVER-ACCEPTANCE | PARTIAL | Чтение, очередь и Qt PASS; Contour machine worker PASS на отдельном qualityproject; остальные обязательные worker/plugin/fault/GUI-запуск сценарии этого нового этапа ещё NOT TESTED. Качество модели и электрическая приёмка реального прибора NOT TESTED |

Остальные непройденные части исторических групп сохраняют NOT TESTED/PARTIAL.
Kraken пока не объявляется готовым.

Дата прогона: 2026-10-30. Среда: Windows, Python 3.14.4 из `uv`, PyQt 6.11.0, платформа Qt `offscreen`, PostgreSQL 16 запущен локально, пароль администратора не задан, Rust-сборки в PATH нет, готовый `blob_gateway/target/release/kraken-blob-gateway.exe` есть.

Проверялись настоящие `EmbeddedProjectService`, диалоги Qt, процесс Blob Gateway, `kraken-server --development`, `kraken-agent` и процессы плагинов. Основные сервисы моками не подменялись. Успешный старый тест сам по себе здесь не считается готовностью функции.

Сырые журналы этого сеанса: `.test_runtime/functional-audit-results.json`, `.test_runtime/functional-audit-followup.json`. Повтор видимой кнопки целостности и слоя с диска выполнен отдельным процессом; его вывод включён в строки ниже.

Аудит всего реестра не закончен. Группы со статусом NOT TESTED ещё нужно пройти.

## Исторический прогон 2026-10-30–31

| ID | Сценарий и критерий | Статус | Факт |
|---|---|---|---|
| LOCAL-ACCOUNT-LOGIN | Первый аккаунт создаётся. Второй аккаунт отклоняется. Неверный пароль не открывает сессию. Верный пароль работает в новом экземпляре сервиса на том же каталоге | PASS | Второй аккаунт: `A local account already exists`. Повторный `login` успешен |
| LOCAL-PROJECT-BLANK-NAME | Пустое имя проекта отклоняется, каталог не меняется | PASS | `Название проекта: значение обязательно.` |
| LOCAL-PROJECT-CREATE-REOPEN | Проект 2×1 с именем «Аудит» читается новым экземпляром сервиса | PASS | Тот же id, имя и ширина |
| LOCAL-PROJECT-IDEMPOTENT | Повтор `create_project` с тем же `idempotency_key` возвращает тот же проект | PASS | 2026-10-31. `tests/test_project_create_idempotency.py`: тот же id, одна папка «Аудит», чтение новым сервисом, старый случайный id, одно событие привязки и полный шаблон слоёв после обрыва. Диалог: `tests/test_create_project_dialog_retry.py`. До правки был `FileExistsError` |
| LOCAL-PROJECT-RENAME-ARCHIVE-RESTORE | Переименование сохраняется, архив скрывает проект из обычного списка, восстановление возвращает `active` | PASS | После переоткрытия имя «Аудит-2», состояние active |
| LOCAL-IMPORT-REOPEN | PNG `1_1.png` импортируется в репрезентацию. Байты после переоткрытия совпадают | PASS | SHA-256 `6fe986738c50bebe789cc833c8a31ccf33730b959b507d3c560245ad0713d050`, одна ячейка с содержимым |
| LOCAL-NOTE-QUALITY | Заметка и дефект качества читаются после переоткрытия. Неизвестная серьёзность отклоняется | PASS | Заметка на месте, дефект «Обрыв», серьёзность `nope` даёт `DomainValidationError` |
| LOCAL-PERFORMER-WORK-ORDER | Векторизация на существующий каталог пишет пакет. Отмена сохраняется после переоткрытия | PASS | В пакете `1_1.png`, `frames.json`, `job.json`, `kraken-work.json`. Состояние `cancelled` |
| LOCAL-PLAN-STACK-ELECTRICAL | План становится active. Физический стек из одного проводящего слоя читается после переоткрытия. Экспорт электрической модели возвращает документ | PASS | Состояние плана active, в стеке 1 слой, у экспорта есть `content` и `sha256` |
| LOCAL-ROLE-GRANT | Назначение наблюдателя с ревизией ACL читается после переоткрытия | PASS | Роль `viewer` |
| LOCAL-BACKUP-RESTORE | Экспорт открывается в новом каталоге с `take_ownership`. Испорченная копия отклоняется | PASS | Имя совпало. Порча файла дала ошибку проверки, импорт не прошёл |
| LOCAL-INTEGRITY-DELETE | `scan_integrity` выполняется. Чужое имя не удаляет проект. Верное имя удаляет, после переоткрытия проекта нет | PASS | Неверное подтверждение отклонено |
| LOCAL-LAYER-DELETE-METADATA | Слой, созданный `create_layer`, с импортированным PNG, удаляется по точному имени и отсутствует после переоткрытия | PASS | 2026-10-31. `tests/test_metadata_layer_deletion.py`: активных слоёв нет, в истории `LayerDeletedV1`, после нового сервиса слой archived. До правки был `Layer has no two-root file binding` |
| LOCAL-LAYER-DELETE-DISK | Слой из JPG через `scan_layer_source` и `create_layer_from_disk` удаляется по имени «Metal» | PASS | Активных слоёв нет, привязка файлов пустая, в журнале есть `LayerDeletedV1`. В полной истории слой остаётся archived |
| LOCAL-LAYER-SCAN-PNG | Папка только с PNG принимается сканером нового слоя | PASS | 2026-10-31. `scan.ready`, копия `frame_0.png` в слое, кадр читается новым сервисом. До правки сканер сообщал, что изображений нет |
| QT-CREATE-PROJECT-DIALOG | Кнопка создания проекта, имя «Проект из окна», корень хранилища уже в настройках. Проект открыт и читается новым сервисом | PASS | Диалог Qt, offscreen |
| QT-INTEGRITY-BUTTON | Видимая кнопка «Проверить целостность» на странице «Администрирование» завершает проверку | PASS | После щелчка статус «Операция завершена.`. Текст модального окна перехватчиком не получен |
| BLOB-PUT-GET-RESTART | Билет с мусором даёт 401. Верный PUT записывает байты. После перезапуска процесса GET возвращает те же байты | PASS | Health 200, 401 на плохой билет, `already_existed: false`, байты совпали |
| SRV-POSTGRES-PERSISTENT | Постоянный сервер на PostgreSQL: миграции, регистрация, проект, слой, заметка, план, конфликт ревизии, роль, перезапуск | PASS | Подключение `kraken_local_app` к `kraken_local` подтверждено. `rolcreatedb=false`, отдельная база не создана. Прогон шёл в схеме `kraken_audit_*` и схема удалена. Health `metadata=postgresql`. Регистрация 201, неверный пароль 401, повтор входа 200. Повтор `Idempotency-Key` вернул тот же проект. Слой `M1`, заметка и план работ сохранились. Чужой `If-Match` — 409 `revision.conflict`. После нового процесса сервера проект, слой, заметка и план на месте. Счётчики 35 таблиц `public` до и после совпали |
| SRV-ROLE-BEFORE-SESSION | Назначить роль сразу после `POST /auth/accounts`, до любого запроса с токеном нового аккаунта | PASS | 2026-10-31. `tests/test_registered_principal_before_session.py` на отдельной схеме PostgreSQL: `viewer` до предъявления нового токена, затем `GET` проекта 200. Повтор того же пароля тоже записывает участника. До правки был 404 |
| SRV-VIEWER-BOUNDARY | Наблюдатель читает проект и не переименовывает его | PASS | `GET` проекта — 200. `PATCH` имени — 403, нет действия `rename_project` |
| SRV-DEV-MEMORY-PROJECT | `kraken-server --development`: сессия, создание проекта, перезапуск, список | PARTIAL | Создание в памяти прошло. После перезапуска проекта нет. PostgreSQL не участвовал |
| AGT-HEALTH | Агент с пустым реестром отвечает на health только с токеном из файла подключения | PASS | `status=ok`. Запрос без токена отклонён |
| AGT-REGISTRY-CONFIDENCE | `layer.confidence.analyze.v1` для исходного Karakal регистрируется как headless | FAIL | В `plugin-registry.json` операция interactive, modes `["interactive"]`, команда `python -m karakal.worker`. Задание не отправлялось: агент открыл бы процесс с ожиданием пользователя |
| PLUGIN-CONTOUR-HELP | `python -m contour --help` | PARTIAL | Процесс ответил. Редактор, пакет и возврат в Kraken не выполнялись |
| PLUGIN-KRONA-HELP | `python -m krona --help` | PARTIAL | Процесс ответил. EDF не открывался |
| PLUGIN-CSLISER-CAPABILITIES | `python -m csliser --get-capabilities` | PARTIAL | Процесс ответил. Копирование, перемещение и удаление не выполнялись |
| PLUGIN-KATEGB-HELP | `python -m kategb --help` | PARTIAL | Процесс ответил. Разметка и расчёт не выполнялись |
| PLUGIN-NEURALIMAGE-VERSION | `python -m neuralimage --version` | PARTIAL | Процесс ответил. Обучение и распознавание не запускались |
| PLUGIN-CSLISER-SELF-TEST | `python -m csliser --self-test-dir` | PASS | Код выхода 0 |
| E2E-SYNTHETIC-LIFECYCLE | `tests/test_end_to_end_functional.py` при `KRAKEN_RUN_END_TO_END_FUNCTIONAL=1`: 20 стадий на синтетике 3×128×128, настоящие сервис, агент, Contour, NeuralImage, сшивка, Karakal, Krona и backup | PARTIAL | Прогон 50,2 с, код 0. Квитанция `status=passed`, стадии 1–20. Векторизация, одно эпохное CPU-обучение, маски, анализ с подменой пустой маски, приёмка дефекта, мозаика, синтетическая регистрация, EDIF и таблица истинности Krona, CSV плана, архив после переоткрытия, 19 версий и целая копия. В квитанции `quality_assessed=false` и `real_device_electrical_acceptance=false`. Первый запуск не дошёл до теста: `PermissionError` на `AppData\Local\Temp\pytest-of-shust`. Повтор с `--basetemp` в `.test_runtime` прошёл |

Первый прогон поручения, плана и роли упал из-за неверных аргументов прогона: несуществующий каталог, ключ `revision` вместо `model.revision`, ревизия проекта вместо ревизии ACL. Повтор с аргументами, которые использует интерфейс, дал PASS. Эти три первых отказа дефектами продукта не считаются.

Первый щелчок «Проверить целостность» попал в невидимую кнопку. Повтор на открытой странице «Администрирование» дал статус завершения.

## Исторический список пробелов 2026-10-30

Исходный список сохранён для трассировки. Его текущие статусы уточнены в секции 2026-11-01 выше; он не является вторым актуальным набором NOT TESTED.

| Группа | Почему NOT TESTED |
|---|---|
| Остальные маршруты сервера: артефакты, команды, websocket, проверки, качество, физика, электрика, задания агента | На PostgreSQL проверены регистрация, проект, слой, заметка, план работ, конфликт ревизии, роль наблюдателя и перезапуск. Эти маршруты в прогоне не вызывались |
| `backup-server`, `backup-verify`, `restore-server` | Копия сняла бы всю базу `kraken_local`, включая рабочие данные. Отдельной пустой базы нет |
| Установка службы Windows, DPAPI `init`, удалённый HTTPS | Нет отдельного пустого кластера и прав службы. Боевую базу не трогал |
| Вход Windows Hello и сохранённый пароль в диспетчере учётных данных | Нужен интерактивный рабочий стол пользователя, не offscreen |
| Скачивание и запуск обновления Hub и плагинов | Нет манифеста установщика, запуск установщика прервал бы сессию |
| Диалоги Qt: план, качество, физика, электрика, сшивка, анализ, роли, поручение, возврат проверки, статистика, исполнители | Сервисные операции плана, качества, стека, электрики, ролей, поручения и целостности прогнаны. Сами диалоги, кроме создания проекта и кнопки целостности, не нажимались |
| Конвейер с доски слоя и диалогов Hub: обучение, распознавание, уверенность, выборка | Программный синтетический прогон эти шаги, кроме уверенности и отдельной выборки, уже прошёл. Кнопки доски и диалоги не нажимались |
| Вычисляемая сшивка из диалога и обмен векторами через окно | Синтетический прогон записал мозаику. Диалог «Вычисляемая сшивка…» не открывался |
| Karakal: сравнение, карта расхождений, публикация в Kraken | Worker и окно не запускались. Реестр уверенности слоя проверен отдельно и получил FAIL |
| Окна Contour, Krona, KateGB и рабочие команды CSliser кроме self-test | Интерактивный GUI на столе пользователя не открывался |
| Загрузка через сервер в Blob Gateway, Range из клиента Hub, барьер удаления и пауза backup | Проверен прямой PUT/GET шлюза. Связка сервер → билет → шлюз не собиралась |
| Двойной каталог Hub с живым сервером, подписка websocket, заявка на удаление | Сервер PostgreSQL недоступен для теста |
| Контурные фильтры, инструменты редактора, EDF-граф Krona, расчёт KateGB | Нет сквозного сценария в этом сеансе |

Операция `frames.dataset.prepare.v1` в headless-ветке Contour отвергается `contour/worker.py`: поддерживается только `frames.vectorize.v1`. Задание выборки в этом сеансе не ставилось, поэтому строка остаётся NOT TESTED, а ограничение записано в `defects.md` как наблюдение по коду.
