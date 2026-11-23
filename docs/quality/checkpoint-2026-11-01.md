# Стабилизация 2026-11-01: checkpoint

Начальный HEAD `8cf3ea6`, исправления и новые проверки — `71ce8dc`; продолжение D7 — `f749893`.
Ветка `codex/stabilization-2026-11-01`, checkout `D:/code/kraken`.
Все пять исходных документов прочитаны. D1–D3 и D5 уже были в `8cf3ea6`;
историческое «без коммита» устарело. Новый аудит с нуля не начинался.

Уточнение пользователя после `63ff7cb` (исправление `f749893`): **KateGB не проверять; приёмку реального
проекта выполнять пока только через серверное окружение**. Прежние результаты KateGB
остаются историческими, полный Excel-путь исключён из обязательств этого продолжения.
Локальные проверки не являются приёмкой реального проекта.

## Сохранённые проверки

Числа относятся к блокам, некоторые тесты пересекаются; это не сумма уникальных тестов.

| Блок | Результат | Доказательства |
|---|---|---|
| D4 | 7 PASS, worker subprocess, SHA-256, reread JSON/SQLite | [GREEN](evidence/2026-11-01/d4-focused.txt), RED, RED Hub, result, [report](evidence/2026-11-01/d4-confidence.json) |
| Native backup/verify/restore + gateway | 20 + 6 PASS, отдельный кластер | 20 PASS и stale import FAIL, весь оставшийся блок после исправления импорта |
| Регистрация | Ordinary account без global roles, ACL isolation, неверный пароль отклонён | [26 PASS и stale catalog test FAIL](evidence/2026-11-01/p0-policy-network.txt); перенос теста на фактический DualCatalogService проверен в следующем Qt-блоке |
| Qt план/физика/электрика/сшивка/анализ/шаблоны | 33 PASS | [Журнал](evidence/2026-11-01/qt-acceptance.txt) |
| Qt качество с кликами и новым сервисом | PASS | [Журнал](evidence/2026-11-01/quality-clicks.txt) |
| PG/HTTP/Hub/Agent/Karakal/сшивка | 127 PASS | Журнал |
| PG качество/физика/электрика/планы/очередь | 30 PASS | [Журнал](evidence/2026-11-01/server-domain-remaining.txt) |
| Live DualCatalog + websocket/outbox + pending удаления | PASS | Журнал |
| Generic /commands и websocket authorization | 9 PASS | Журнал |
| Плагины CSliser/KateGB/Krona/Karakal/Contour | 109 PASS, 1 неприменимый POSIX skip | [Журнал](evidence/2026-11-01/plugins-acceptance-importlib.txt) |
| Hub/поручения/возврат/выбор модели/отказ без модели | 85 PASS, 1 opt-in training skip | Журнал; успешное CPU-обучение уже подтверждено прежним E2E, без причины не повторялось |
| Agent | 74 PASS, 2 subtests, 11 PG skips затем исполнены в PG-блоке | Журнал |
| D1: сохранность непустого orphan workspace | 3 RED → GREEN, 37 PASS с созданием и четырьмя реальными Qt-сценариями | RED, GREEN |
| D6: импорт и CSV immutable payload | 18 PASS, включая Qt calculate/CSV/XLSX | RED счётчика, RED CSV, [GREEN](evidence/2026-11-01/statistics-verified.txt) |
| Qt возврат изменённого CIF → принять → reread | PASS | Новый Qt-блок: 22 PASS; NameError только нового CSV-теста исправлен и перепроверен в statistics-verified |
| Contour редактор/фильтры/CIF | PARTIAL: 323 PASS, 1 stale async test failure, 1 missing fixture skip | Блок, исправленное ожидание — PASS |
| Krona --json-out | PASS, JSON перечитан | Журнал, JSON |
| Lint изменённых Python-файлов | PASS | Журнал |
| Один PNG через scanner и managed import | PASS, хеш/байты после нового сервиса совпали | [Журнал](evidence/2026-11-01/png-parity.txt), `test_png_disk_import_survives_reopen` |
| Qt ошибочные действия ролей/исполнителей | 2 PASS, реальные предупреждения и отсутствие записи | Журнал |

Независимое ревью: протокол.
Копии JSON проверены парсером; SHA-256 артефактов.

## Воспроизведение

Из корня: `& docs/quality/reproduce-stabilization.ps1 -Block <block>`.
Блоки: `d4`, `d1`, `backup`, `server`, `server-domain`, `live-catalog`, `qt`,
`statistics`, `agent`, `plugins`, `policy`, `schema-upgrade`, `hub-resume`.
Каждый запуск пишет новый `quality-<uuid>` с командой/exit code в receipt, pytest.txt и JUnit.
PG-блоки создают собственный кластер и UUID-базы, не читают секрет рабочей базы.
Запуск сервера и тестов в одном процессе инструмента, остановка — finally;
каталоги остаются для разбора. Скрипт проверен блоком d4, 7 PASS:
receipt, JUnit.

Ручной PG-прогон: `.test_runtime/pg-stabilization-20261101`, loopback:55439,
роль `kraken_audit`, базы `kraken_backup_test_<uuid>` и отдельные схемы.
Все тестовые PostgreSQL-кластеры остановлены. Это утверждение относится к изолированным
регрессиям; позже по указанию пользователя запущен рабочий Server, см. блок ниже.
Sandbox запрещает loopback, включая asyncio socketpair; сетевые тесты потребовали
require_escalated. Эти отказы среды не объявлялись дефектами Kraken.

## Осталось / точный следующий шаг

1. Рабочий Server запущен из сохранённого `LocalServer/server.toml`, Gateway поднят
   штатным lifecycle. Health обоих — 200.
   Начальный connection refused сохранён как исторический preflight, больше не BLOCKED.
   Сохранённый аккаунт аутентифицирован через штатный keyring; секреты в evidence не копировались.
   Проект `test`, `7c622c09-73fa-552f-9a52-e80b64aeb3ef`, 10×10, слой m1:
   100 изображений + 100 CIF.
   Все 200 файлов получены через HTTP/Gateway, SHA-256/размер совпали, изображения декодированы,
   CIF прочитаны, размеры пар совпали; [полный результат](evidence/2026-11-01/real-project-artifact-verification.json).
   Два открытия настоящего Qt Hub двойным щелчком и повторное чтение —
   PASS, история/ревизия 2 не изменились.
   Подтверждён локальный loopback Agent; leased worker рабочего сервера ещё NOT TESTED.
   Это проверка доступности/чтения, а не качества модели или реального прибора.

   **P1 PASS после явного согласия пользователя:** миграция0011 применена к рабочей
   `kraken_local` одной транзакцией. Заблокированы background_jobs и строка версии;
   исходная версия0010 проверена. Все поля двух существующих заданий сравнены до/после,
   новый nullable JSON проверен как NULL. После commit схема0011, `/projects/{id}/jobs`
   отвечает200, проект остаётся на revision2:
   квитанция,
   выполненный script как текст,
   окончательный read-only verifier.
   Начальная post-commit проверка ошиблась сигнатурой `plugin_jobs(pid)`; исправлен только
   read-only probe на `plugin_jobs()`, миграция повторно не выполнялась.
   Два открытия Qt повторены после миграции: PASS,
   журнал, нет uncaught/errors, история неизменна.
   Независимый reviewer проверил DDL/transaction/receipt/Qt, замечаний нет.

   **Точный следующий шаг / worker BLOCKED:** read-only preflight
   обнаружил 0 server_agent_tokens и отсутствие KRAKEN_AGENT_TOKEN. Существующие jobs
   succeeded и waiting_for_user не менялись. Для leased worker нужен отдельный scoped
   machine token и тестовое задание в согласованной серверной области; не подменять его
   локальным loopback Agent. Не публиковать секрет и не возобновлять чужое waiting_for_user.
   Полный сценарий на отдельном PostgreSQL/HTTP с ServerAgentWorker уже PASS в127-тестовом
   блоке; это не означает приёмку worker в рабочем серверном окружении.

   Отдельный дефект D7: HTTP500 очереди выходил из startup Qt callback и завершал Hub
   с 0xc0000409. Выбор задания перенесён внутрь существующего try. RED,
   36 PASS; успешный retry подтверждён HTTP200.
   Независимое ревью: Critical/Important нет; Minor про assertion retry устранён.
   Проверка lint трёх новых/изменённых файлов — PASS.

   Воспроизведение: блоки `schema-upgrade` и `hub-resume` в reproduce-stabilization.ps1.
   Сценарии только чтения сохранены в `docs/quality/probes/`: manifest_live_artifacts.py,
   verify_live_artifacts.py, view_live_project_qt.py. Им нужен текущий защищённый DPAPI
   session-файл `.test_runtime/live-server/server-session.secret` от штатного входа;
   исходные пользовательские данные и токен в git не копировать. Запускать из корня через
   `.venv/Scripts/python.exe docs/quality/probes/<script>.py`, manifest перед verify.
   Снимки Qt: `.test_runtime/live-server/project-view-{1,2}.png`; в git не включены.
2. Получить отсутствующую эталонную `0518.cif` и выполнить только её пропущенную
   регрессию Contour. Остальные 324 проверки повторять без причины не нужно.
3. Legacy confidence исполняется headless, но выдаёт только счётчик: метрики PARTIAL.
   Не выдавать его JSON за оценку модели. Headless-выборка Contour не заявлена;
   положительный interactive путь до возврата проверен.
4. Windows service: `Administrator=False` вне sandbox. Нужен отдельный elevated стенд;
   Hello — интерактивное подтверждение пользователя; updater — тестовый установщик;
   remote HTTPS — выделенный TLS endpoint. BLOCKED.
5. KateGB исключён по прямому указанию пользователя. Локальные файлы реального проекта
   для этой приёмки не запрашивать: нужен его серверный доступ (пункт 1).
   Синтетический E2E по-прежнему не подтверждает качество модели и электрическую
   приёмку реального прибора.

В завершённых блоках воспроизведённые отказы исправлены. PARTIAL, BLOCKED и NOT TESTED
остаются: стабилизация не завершена, Kraken готовым не объявляется.

## Новый серверный блок: 393bca4

По новому заданию и явному согласованию пользователя создан временный machine token
только frames.vectorize.v1. Секрет — DPAPI вне git, receipt содержит только identity.
Прежний BLOCKED «нет токена» для этого capability снят. Токен ещё действует до конца
блока отказоустойчивости; затем штатный revoke. Полномочия на остальные необходимые
тестовые операции пользователь дал сообщением «делай, всё, что надо и меняй всё что надо».
Защита проекта test и существующих jobs этим не отменяется.

Найден D8: persistent upload gateway preflight вызывался у function после перехода
на ProjectCommandService. Реальный HTTP RED,
PG RED,
[PG GREEN](evidence/2026-11-01/upload-preflight-green.txt),
JUnit.
Исправлена передача preflight зарегистрированному handler. Смежные gateway/command
transport:19 PASS; отдельная новая регрессия:1 PASS. Server/Gateway перезапущены с
сохранённой конфигурацией, без миграций и изменений настроек.

[Полный Server→machine worker→Contour→версии→Hub](evidence/2026-11-01/live-contour-worker-receipt.json)
PASS, лог,
JUnit, 1 PASS за7.77s.
Новый project 8aaee5c1-b3d6-5bd0-b624-7ccf74b18d66, 2×1, два PNG64×64;
headless Contour действительно запущен subprocess. Две CIF-version87bytes каждая,
SHA/size/application-x-cif/parse/dimensions/input lineage, heartbeat/upload/publication/
complete подтверждены. Два новых HTTP-клиента и два Qt-окна открыли проект двойным
щелчком и перечитали результаты/историю. Это не проверка качества реального прибора.
Protected test revision2, history и оба jobs полными payload-хешами не изменились.

Промежуточные отказы нового тестового драйвера: не переданы x/y в upload metadata;
суффикс проверялся у hash-cache path вместо version.filename. Это исправления фикстуры,
не дефекты продукта. Первая неуспешная receipt ошибочно писала passed из teardown;
успех теперь устанавливается только последней строкой сценария. Доказательством RED
остаётся pytest/JUnit, а не прежнее неверное поле.

**Точный следующий шаг:** добавить и исполнить тесты отказов на отдельном quality
проекте, затем NeuralImage/Karakal/stitch через machine worker, пользовательский GUI
запуск и история. Пользовательский test и два прежних jobs не изменять. Независимое
ревью393bca4 запрошено; весь новый этап пока PARTIAL, не COMPLETED.
Текущий план: [server worker acceptance](server-worker-acceptance-plan.md).

## Адресный lease и отказоустойчивость: be1ca58 / 95ec8f2

Независимое ревью D8 приняло production-fix и выявило Important в самой live-приёмке:
глобальный lease менял очередь до проверки ID, предварительное чтение не атомарно.
Это устранено в be1ca58: отдельный POST /agent/jobs/{UUID}/lease фильтрует SQL до
выбора/изменения; capability auth остаётся прежней. На старом Server маршрут даст404,
не превратится в незаметный global polling. run_once(job_id=...) использует этот маршрут.
2RED,
14PASS со смежными queue/stitch,
JUnit.
Minor о lineage закрыт: теперь source-version проверяется для конкретной координаты,
MIME CIF проверяется явно. Рецензент перечитал изменения и закрыл оба finding, новых нет.

После этого полный блок: 6PASS за55.40s,
JUnit, source tests —95ec8f2.
Результаты: normal Contour/Gateway/versions/2HTTP+Qt reopen;
worker unavailable/connection refusal → queued без output; wrong token401;
stale cancel409; настоящий Hub cancel → cancelled и lease не выдаётся;
реальные ошибки Contour (неверный тип параметров, не-PNG с корректным blob-hash)
→ failed+diagnostic, MyWork показывает ошибку; неконтрактный retry failed409 без изменения;
15s lease реально истёк, old owner403, новый worker получил новый claim и завершил;
повторные complete и та же фактическая publication403 без новых history/versions;
TCP proxy оборвал ответ после серверного complete200 — результат остался succeeded,
повторное подключение не пересчитало и не продублировало его.

Все output-поля и protected hashes.
Каждый fault receipt сохранён рядом, prefix worker-fault-. Пользовательский test и оба
прежних jobs сохранили revision/history/payload hashes. Предыдущие partial attempts
оставлены в отдельных quality-проектах для разбора, пользовательские файлы не удалялись.

Contour machine token отозван: receipt,
HTTP401 подтверждён.
Секреты и исходные пользовательские артефакты в git не копировались.

Воспроизведение: gateway-preflight/targeted-lease/live-worker в существующем PS script.
live-worker требует явного KRAKEN_LIVE_WORKER_CONFIG: server_url, пути session_secret и
machine_secret (DPAPI), protected_project_id, results_directory. Значения секретов вJSON
запрещены. Для нового live-run нужен новый действующий frames.vectorize.v1 token;
отозванный не восстанавливать. Запуск на общей очереди только через targeted route.

**Следующий шаг:** scoped NeuralImage token и настоящая CPU inference через
frames.binary-segment.v2 с уже обученной моделью из прежнего E2E (повторное обучение
без причины не нужно); затем Karakal через настоящий диалог Hub и machine worker,
сшивка. Точная модель-кандидат:
.test_runtime/pytest-e2e/test_single_project_twenty_sta0/export/
479fcf1a-b71d-5725-9755-6acaaad6c5ae-S 660k_shift_128_epoch1.pth.
Перед использованием сверить hash/receipt и безопасную загрузку. Общий этап PARTIAL.
