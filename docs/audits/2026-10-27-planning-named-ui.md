# Именованное планирование и доказательства приёмки

В `WorkPlanningDialog` ручной ввод UUID исполнителя, слоёв и версий заменён
выбором существующих объектов. Идентификаторы остаются в transport/domain
контрактах; формат WorkPlan, критерии, CAS, idempotency и проверки доступа
не менялись. Обучение и оценка качества векторизации не запускались.

Исполнители показываются по имени и берутся из backend выбранного проекта
через существующий `list_project_performers`, если используется DualCatalog.
Объединённый local/remote каталог не используется для назначения. Архивные
названия сохраняются для отображения истории, новые назначения используют
активных исполнителей. Слои выбираются флажками; отсутствие выбора явно означает
весь проект. Таблица показывает имена и выделяет ширину для названия работы.
Форма помещена в scroll area для небольших окон.

Доказательства выбираются отдельным диалогом:

- Кадр: слой и X/Y, существующий scoped `quality_frame_versions`.
- Результаты слоя/проекта: выбор artifact series, затем её истории версий.
- Версия показывается как имя файла, время и SHA-префикс; UUID не вводится.
- Выбранные версии сохраняются между загрузками, доступны в отдельном списке
  и могут удаляться из выбора по одной. Отмена диалога не меняет выбор родителя.
- Сохранённые критерии восстанавливают version IDs, metadata загружается по
  общей local/remote `artifact_version` API. Это просмотр той же immutable версии.

Все service calls выполняются вне GUI thread. При смене слоя/кадра/вида старые
кандидаты очищаются; выбранные доказательства сохраняются отдельно. Смена работы,
критерия или scope меняет контекст выбора. Ошибка чтения сохраняет выбранные
версии. Во время запроса picker и родитель участвуют в existing Hub close guard.
Клиентская фильтрация scope не заменяет server-side проверку evidence/ACL.

## Проверки

Артефакты находятся в `.codex_tmp_scratch/production-audit-20261027/`.

`planning-named-readable/` и `planning-named-final-scoped/` выполнили настоящие
shown Qt workflows. `test_postgres_http_named_work_and_attachment_acceptance_reopen`
поднял loopback uvicorn с реальной PostgreSQL fixture-схемой: создание плана,
выбор исполнителя/слоя, работа→running→progress→in_review, выбор именованного
layer attachment, criterion evidence→accepted→completed, повторное открытие и
чтение сохранённой версии. Snapshot/metadata проверяются, а не только тексты кнопок.

`test_named_performers_are_read_from_the_project_backend` запрещает merged
catalog. Остальные новые проверки воспроизводят отказ out-of-scope frame,
потерю ответа, сохранность/удаление выбора и close guard для дочернего worker.
Существующие dialog/template/board regressions сохранены и адаптированы к новым
виджетам. `planning-named-backend-scoped/` содержит повтор после routing fix.

Ранний `planning-named-remote-first/` выявил использование local-only имени
`get_artifact_version`; исправлено на общую `artifact_version` API.
В `planning-named-context/` один тест завершился timeout: fixture выбирала
произвольный frame series, а picker открывал (1,1). Fixture теперь явно выбирает
тот же кадр, а modal automation закрывает окно при ошибке. Это исправление
сценария, не изменение правил продукта; failed gate сохранён отдельно.

Снимки из изолированного pytest APPDATA сначала содержали отсутствующие glyphs.
Для визуальной fixture явно загружен установленный Segoe UI, как в deployment
probe. Читаемые PNG просмотрены; применение шрифта ограничено тестовым процессом.

## Среда и границы

Первый общий `planning-named-root/` прерван после исчезновения Docker Engine:
подтверждено отсутствие Linux-engine pipe и PostgreSQL ConnectionTimeout.
`environment-abort.json` сохраняет причину; этот прогон не passing.
Затем Engine стал доступен; собственный контейнер
`kraken-audit-20261026-74ef973a` был подтверждён остановленным, запущен, а SQL
`select 1` прошёл. Полный PostgreSQL прогон повторён отдельно от offline gate.
Пользовательские сервисы этими командами не управлялись.

Выбор evidence выполняется по требованию; всё содержимое blobs и все версии
проекта при открытии окна не скачиваются. Для attachments существующая API
возвращает metadata series выбранного слоя/проекта; большая нагрузка и pagination
этого пути требуют отдельной приёмки. Единый двадцатиэтапный desktop UI E2E,
массовые операции и многопользовательская нагрузка этим блоком не объявляются
завершёнными. Существующий representative service/Agent E2E остаётся отдельным gate.

## Итоговые артефакты текущей ревизии

| Gate | Результат |
| --- | --- |
| `planning-final-postgres-root/summary.json` | 197 modules, 1279 cases, 0 failures/errors, 14 skipped; 1265 выполнены, PostgreSQL включён |
| `planning-final-offline-root/summary.json` | Тот же набор без PostgreSQL: 1189 выполнены, 90 skipped; не замена PG gate и не суммируется с ним |
| `planning-final-desktop-sources.json`, `planning-final-server-sources.json` | Все present project/Contour pure modules совпали с source: 462 Desktop / 373 Server |
| `planning-final-desktop-installed/report.json` | Новый audit AppId: install/self-test/uninstall passed, 12.310 s, registration/executable removed |
| `planning-final-server-installed/report.json` | Новый audit AppId: install/frozen migrations/auth/Gateway/Range/SciPy/restart/uninstall passed, 20.762 s, registration/executable removed |

2651 source/test/tool/spec/dependency-manifest files закреплены до и после
финальных offline/PG gates; изменений нет. Все 14 пропусков PG gate перечислены
в `skipped-tests.json`. Обучение не запускалось. Ruff изменённых UI/test modules,
scoped whitespace и offline lock прошли; legacy lint debt не включается в это утверждение.

Production build outputs, обычные AppId, не установлены на рабочей машине:
`planning-final-installers/production/`.

| Установщик | Bytes | SHA-256 |
| --- | --- | --- |
| KrakenDesktopSetup-0.1.0.exe | 74,496,943 | `42229cde4f5c51788d4f06365cef3ee21b7945adee246cae794d9721f0a83307` |
| KrakenServerSetup-0.1.0.exe | 114,843,858 | `f221007ac8b352b8eb59aecbc794466f362da83f7a1be61eace647f25cd30111` |

Актуальный Server был собран из текущего worktree и перенесён в общий build root
с проверкой SHA всех 1181 файлов (`planning-final-server-copy.json`). Последующая
установленная проверка выполнена на этом payload, а не на старом Server.
SCM service, production AppId update, clean OS/Linux, standalone plugins и
побитовая воспроизводимость остаются отдельными открытыми задачами.
