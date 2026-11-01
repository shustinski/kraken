# Стабилизация 2026-11-01: checkpoint

Начальный HEAD `8cf3ea6`, исправления и новые проверки — `71ce8dc`.
Ветка `codex/stabilization-2026-11-01`, checkout `D:/code/kraken`.
Все пять исходных документов прочитаны. D1–D3 и D5 уже были в `8cf3ea6`;
историческое «без коммита» устарело. Новый аудит с нуля не начинался.

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
`statistics`, `agent`, `plugins`, `policy`.
Каждый запуск пишет новый `quality-<uuid>` с командой/exit code в receipt, pytest.txt и JUnit.
PG-блоки создают собственный кластер и UUID-базы, не читают секрет рабочей базы.
Запуск сервера и тестов в одном процессе инструмента, остановка — finally;
каталоги остаются для разбора. Скрипт проверен блоком d4, 7 PASS:
[receipt](evidence/2026-11-01/reproducer-d4-receipt.json), [JUnit](evidence/2026-11-01/reproducer-d4-junit.xml).

Ручной PG-прогон: `.test_runtime/pg-stabilization-20261101`, loopback:55439,
роль `kraken_audit`, базы `kraken_backup_test_<uuid>` и отдельные схемы.
Все тестовые серверы остановлены; рабочая `kraken_local` не использовалась.
Sandbox запрещает loopback, включая asyncio socketpair; сетевые тесты потребовали
require_escalated. Эти отказы среды не объявлялись дефектами Kraken.

## Осталось / точный следующий шаг

1. При получении безопасного пути к эталонной Excel-разметке KateGB открыть её
   настоящим окном, выполнить расчёт по известному XML и сверить исполнителей/проценты,
   затем ошибочный ввод и повторное открытие. Вопрос о пути задан, ответа пока нет.
   Без фикстуры не выдумывать формат, сохранить NOT TESTED полного пути.
2. Получить отсутствующую эталонную `0518.cif` и выполнить только её пропущенную
   регрессию Contour. Остальные 324 проверки повторять без причины не нужно.
3. Legacy confidence исполняется headless, но выдаёт только счётчик: метрики PARTIAL.
   Не выдавать его JSON за оценку модели. Headless-выборка Contour не заявлена;
   положительный interactive путь до возврата проверен.
4. Windows service: `Administrator=False` вне sandbox. Нужен отдельный elevated стенд;
   Hello — интерактивное подтверждение пользователя; updater — тестовый установщик;
   remote HTTPS — выделенный TLS endpoint. BLOCKED.
5. Приёмочный реальный прибор/модель/проект не предоставлены. Синтетический E2E
   по-прежнему не подтверждает качество модели и электрическую приёмку прибора.

В завершённых блоках воспроизведённые отказы исправлены. PARTIAL, BLOCKED и NOT TESTED
остаются: стабилизация не завершена, Kraken готовым не объявляется.
