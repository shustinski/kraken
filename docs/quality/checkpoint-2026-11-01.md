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
| D4 | 7 PASS, worker subprocess, SHA-256, reread JSON/SQLite | [GREEN](evidence/2026-11-01/d4-focused.txt), [RED](evidence/2026-11-01/d4-red.txt), [RED Hub](evidence/2026-11-01/d4-ui-red.txt), [result](evidence/2026-11-01/d4-result.json), [report](evidence/2026-11-01/d4-confidence.json) |
| Native backup/verify/restore + gateway | 20 + 6 PASS, отдельный кластер | [20 PASS и stale import FAIL](evidence/2026-11-01/p0-backup-live.txt), [весь оставшийся блок после исправления импорта](evidence/2026-11-01/p0-backup-remaining.txt) |
| Регистрация | Ordinary account без global roles, ACL isolation, неверный пароль отклонён | [26 PASS и stale catalog test FAIL](evidence/2026-11-01/p0-policy-network.txt); перенос теста на фактический DualCatalogService проверен в следующем Qt-блоке |
| Qt план/физика/электрика/сшивка/анализ/шаблоны | 33 PASS | [Журнал](evidence/2026-11-01/qt-acceptance.txt) |
| Qt качество с кликами и новым сервисом | PASS | [Журнал](evidence/2026-11-01/quality-clicks.txt) |
| PG/HTTP/Hub/Agent/Karakal/сшивка | 127 PASS | [Журнал](evidence/2026-11-01/server-acceptance.txt) |
| PG качество/физика/электрика/планы/очередь | 30 PASS | [Журнал](evidence/2026-11-01/server-domain-remaining.txt) |
| Live DualCatalog + websocket/outbox + pending удаления | PASS | [Журнал](evidence/2026-11-01/live-catalog.txt) |
| Generic /commands и websocket authorization | 9 PASS | [Журнал](evidence/2026-11-01/server-transport.txt) |
| Плагины CSliser/KateGB/Krona/Karakal/Contour | 109 PASS, 1 неприменимый POSIX skip | [Журнал](evidence/2026-11-01/plugins-acceptance-importlib.txt) |
| Hub/поручения/возврат/выбор модели/отказ без модели | 85 PASS, 1 opt-in training skip | [Журнал](evidence/2026-11-01/hub-remaining-run.txt); успешное CPU-обучение уже подтверждено прежним E2E, без причины не повторялось |
| Agent | 74 PASS, 2 subtests, 11 PG skips затем исполнены в PG-блоке | [Журнал](evidence/2026-11-01/agent-regression.txt) |
| D1: сохранность непустого orphan workspace | 3 RED → GREEN, 37 PASS с созданием и четырьмя реальными Qt-сценариями | [RED](evidence/2026-11-01/d1-review-red-edges.txt), [GREEN](evidence/2026-11-01/d1-review-green.txt) |
| D6: импорт и CSV immutable payload | 18 PASS, включая Qt calculate/CSV/XLSX | [RED счётчика](evidence/2026-11-01/statistics-red.txt), [RED CSV](evidence/2026-11-01/statistics-csv-red.txt), [GREEN](evidence/2026-11-01/statistics-verified.txt) |
| Qt возврат изменённого CIF → принять → reread | PASS | [Новый Qt-блок](evidence/2026-11-01/statistics-final-green.txt): 22 PASS; NameError только нового CSV-теста исправлен и перепроверен в statistics-verified |
| Contour редактор/фильтры/CIF | PARTIAL: 323 PASS, 1 stale async test failure, 1 missing fixture skip | [Блок](evidence/2026-11-01/contour-editor-full.txt), [исправленное ожидание — PASS](evidence/2026-11-01/contour-hole-recheck.txt) |
| Krona --json-out | PASS, JSON перечитан | [Журнал](evidence/2026-11-01/krona-json-out-valid.txt), [JSON](evidence/2026-11-01/krona-result.json) |
| Lint изменённых Python-файлов | PASS | [Журнал](evidence/2026-11-01/stabilization-lint-verified.txt) |
| Один PNG через scanner и managed import | PASS, хеш/байты после нового сервиса совпали | [Журнал](evidence/2026-11-01/png-parity.txt), `test_png_disk_import_survives_reopen` |
| Qt ошибочные действия ролей/исполнителей | 2 PASS, реальные предупреждения и отсутствие записи | [Журнал](evidence/2026-11-01/qt-roles-denied.txt) |

Независимое ревью: [протокол](evidence/2026-11-01/review.md).
Копии JSON проверены парсером; [SHA-256 артефактов](evidence/2026-11-01/checksums.json).

## Воспроизведение

Из корня: `& docs/quality/reproduce-stabilization.ps1 -Block <block>`.
Блоки: `d4`, `d1`, `backup`, `server`, `server-domain`, `live-catalog`, `qt`,
`statistics`, `agent`, `plugins`, `policy`, `schema-upgrade`, `hub-resume`.
Каждый запуск пишет новый `quality-<uuid>` с командой/exit code в receipt, pytest.txt и JUnit.
PG-блоки создают собственный кластер и UUID-базы, не читают секрет рабочей базы.
Запуск сервера и тестов в одном процессе инструмента, остановка — finally;
каталоги остаются для разбора. Скрипт проверен блоком d4, 7 PASS:
[receipt](evidence/2026-11-01/reproducer-d4-receipt.json), [JUnit](evidence/2026-11-01/reproducer-d4-junit.xml).

Ручной PG-прогон: `.test_runtime/pg-stabilization-20261101`, loopback:55439,
роль `kraken_audit`, базы `kraken_backup_test_<uuid>` и отдельные схемы.
Все тестовые PostgreSQL-кластеры остановлены. Это утверждение относится к изолированным
регрессиям; позже по указанию пользователя запущен рабочий Server, см. блок ниже.
Sandbox запрещает loopback, включая asyncio socketpair; сетевые тесты потребовали
require_escalated. Эти отказы среды не объявлялись дефектами Kraken.

## Осталось / точный следующий шаг

1. Рабочий Server запущен из сохранённого `LocalServer/server.toml`, Gateway поднят
   штатным lifecycle. [Health обоих — 200](evidence/2026-11-01/real-server-startup.json).
   Начальный connection refused сохранён как исторический preflight, больше не BLOCKED.
   Сохранённый аккаунт аутентифицирован через штатный keyring; секреты в evidence не копировались.
   Проект `test`, `7c622c09-73fa-552f-9a52-e80b64aeb3ef`, 10×10, слой m1:
   [100 изображений + 100 CIF](evidence/2026-11-01/real-project-artifact-manifest.json).
   Все 200 файлов получены через HTTP/Gateway, SHA-256/размер совпали, изображения декодированы,
   CIF прочитаны, размеры пар совпали; [полный результат](evidence/2026-11-01/real-project-artifact-verification.json).
   Два открытия настоящего Qt Hub двойным щелчком и повторное чтение —
   [PASS](evidence/2026-11-01/real-project-qt-view.json), история/ревизия 2 не изменились.
   Подтверждён локальный loopback Agent; leased worker рабочего сервера ещё NOT TESTED.
   Это проверка доступности/чтения, а не качества модели или реального прибора.

   **P1 PASS после явного согласия пользователя:** миграция0011 применена к рабочей
   `kraken_local` одной транзакцией. Заблокированы background_jobs и строка версии;
   исходная версия0010 проверена. Все поля двух существующих заданий сравнены до/после,
   новый nullable JSON проверен как NULL. После commit схема0011, `/projects/{id}/jobs`
   отвечает200, проект остаётся на revision2:
   [квитанция](evidence/2026-11-01/schema-upgrade-working.json),
   [выполненный script как текст](evidence/2026-11-01/apply_approved_schema_upgrade.py.txt),
   [окончательный read-only verifier](evidence/2026-11-01/verify_approved_schema_upgrade.py.txt).
   Начальная post-commit проверка ошиблась сигнатурой `plugin_jobs(pid)`; исправлен только
   read-only probe на `plugin_jobs()`, миграция повторно не выполнялась.
   Два открытия Qt повторены после миграции: [PASS](evidence/2026-11-01/real-project-qt-after-migration.json),
   [журнал](evidence/2026-11-01/qt-after-migration.txt), нет uncaught/errors, история неизменна.
   Независимый reviewer проверил DDL/transaction/receipt/Qt, замечаний нет.

   **Точный следующий шаг / worker BLOCKED:** [read-only preflight](evidence/2026-11-01/server-worker-preflight.json)
   обнаружил 0 server_agent_tokens и отсутствие KRAKEN_AGENT_TOKEN. Существующие jobs
   succeeded и waiting_for_user не менялись. Для leased worker нужен отдельный scoped
   machine token и тестовое задание в согласованной серверной области; не подменять его
   локальным loopback Agent. Не публиковать секрет и не возобновлять чужое waiting_for_user.
   Полный сценарий на отдельном PostgreSQL/HTTP с ServerAgentWorker уже PASS в127-тестовом
   блоке; это не означает приёмку worker в рабочем серверном окружении.

   Отдельный дефект D7: HTTP500 очереди выходил из startup Qt callback и завершал Hub
   с 0xc0000409. Выбор задания перенесён внутрь существующего try. [RED](evidence/2026-11-01/hub-resume-red.txt),
   [36 PASS](evidence/2026-11-01/hub-resume-green.txt); успешный retry подтверждён HTTP200.
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
