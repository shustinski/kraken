# Реструктуризация docs/ и недостающие базовые документы

Дата: 2026-11-13. Статус: на согласовании.

## Намерение и почему

Репозиторий Kraken (clean-architecture project manager для sparse frame grids,
PyQt desktop + PostgreSQL server + Rust blob gateway + плагины) за полгода
накопил в `docs/` 212 закоммиченных файлов (~2.2 МБ). Проверка показала:

- Живая документация (13 фичевых контрактов в корне `docs/`, `architecture/`,
  `deployment/`) в целом актуальна и ссылается из README — её не трогаем.
- ~1.4 МБ — датированные артефакты процесса: `quality/evidence/` (900 KB сырых
  логов pytest/JUnit, 122 файла) и `audits/` (43 разовых отчёта, 544 KB).
  Они перемешаны с живыми доками, индекса нет — отсюда ощущение «мусора».
- Реальные дефекты: README.md строка 238 повреждена кодировкой
  (`**?????????? ? ???? ???????????????**` — в файле буквально вопросники),
  дубликат номера ADR (два файла `0002-*`).
- Отсутствуют базовые документы: индекс документации, сводный API-референс
  `/api/v1`, модель данных, README для `kraken_core` (wire-протокол плагинов),
  пользовательское руководство по Hub.

Пользователь выбрал вариант A: архивирование внутри репозитория + написание
недостающих документов.

## Цели

1. Починить README.md (строка 238) и коллизию ADR 0002.
2. Перенести старые датированные артефакты (audits до 2026-11, evidence
   2026-11-01) в `docs/archive/`, не рвя ссылки и сохранив git-историю.
3. Создать `docs/README.md` — индекс с картой структуры и политиками.
4. Написать четыре недостающих документа: API-референс, модель данных,
   README `kraken_core`, пользовательское руководство Hub.

## Не цели

- Не менять политику evidence: свежие прогоны продолжают складываться в
  `docs/quality/evidence/<date>/`, политика из плана 2026-11-06 действует.
- Не переписывать содержимое 13 фичевых доков, quality-документов и аудитов.
- Ничего не удалять — только `git mv` в архив.
- Не менять логику `tools/quality/*`, не добавлять CI.

## Анти-цели

- Не переводить существующие документы, не переименовывать фичевые доки,
  не переносить `quality/registry/` и сгенерированные quality-документы.

## Ограничения

- `docs/quality/registry/*.json` — single source of truth, потребляется
  `tools/quality/*` и `tests/test_quality_registry.py`. Схема требует поле
  `evidence` как список строк; существование файлов нигде не проверяется.
  Обновление путей в registry безопасно, но после этого обязательна
  перегенерация трёх quality-md (`defects.md`, `functional-inventory.md`,
  `acceptance-matrix.md`) через `tools/quality/generate_quality_docs.py` —
  тест `test_quality_registry.py` падает, если сгенерированные секции не
  совпадают с registry.
- Переносы — только `git mv` (сохранение истории).
- Окружение: Windows, PowerShell 5.1, uv.
- После всех переносов каждая относительная md-ссылка в `README.md` и внутри
  `docs/` должна резолвиться.

## Подход

Выбран **A — архивирование внутри репозитория**: старые артефакты уезжают в
`docs/archive/`, ссылки (в т.ч. в registry JSON) обновляются, документы
пересобираются. Альтернативы: **B** — evidence вне git (gitignore + внешний
каталог; отклонено: отходит от зафиксированной quality-политики) и **C** —
минимум, evidence не трогать (отклонено: не решает задачу с «мусором»).

## Часть 1. Починки и архив

1. **README.md:238** — заменить
   `**?????????? ? ???? ???????????????**` на `**План восстановления…**`
   (реальная строка меню: `src/kraken_manager/presentation/qt/shell.py:142`).
2. **ADR-коллизия** — `git mv docs/architecture/adr/0002-shared-project-application.md
   docs/architecture/adr/0004-shared-project-application.md` (хронологически это
   ADR от 2026-10-29, после 0003 от 2026-07-17); в файле заголовок
   `# ADR 0002:` → `# ADR 0004:`. Единственная ссылка на него — в
   `docs/audits/2026-10-29-catalog-operation-parity.md:18` (обновить номер и
   относительный путь — см. п. 3).
3. **Архив аудитов** — `git mv docs/audits/* docs/archive/audits/`, кроме
   `2026-11-05-workflow-audit.md` и `2026-11-06-workflow-fixes.md` (остаются
   на месте). Последствия:
   - `README.md:242`: `docs/audits/2026-10-23-production-readiness.md` →
     `docs/archive/audits/2026-10-23-production-readiness.md`.
   - Относительные ссылки внутри перенесённых аудитов, выходящие за пределы
     `audits/` (например `../architecture/adr/0002-shared-project-application.md`,
     `../work-planning-templates.md`, `../quality-workflow.md`,
     `../physical-registration.md`), получить префикс `../../`. Полный список
     ссылок выявляет проверка п. 8.
   - `tests/test_catalog_operation_parity.py` читает с диска
     `docs/audits/2026-10-29-catalog-operations.json` (переносится в архив):
     обновить путь в тесте на `docs/archive/audits/...` (одна строка, файл
     входит в разрешённый diff).
   - `plugins/csliser/README.md` ссылается на
     `../../docs/audits/2026-10-27-csliser-launch-and-install.md`: обновить на
     путь в архиве.
4. **Архив evidence** — `git mv docs/quality/evidence/2026-11-01
   docs/archive/quality-evidence/2026-11-01` (79 файлов, 375 KB). В
   `docs/quality/evidence/` остаются `2026-11-09` и `2026-11-10`. Последствия:
   - Механическая замена в `docs/quality/registry/defects.json` и
     `actions.json`: `"docs/quality/evidence/2026-11-01/` →
     `"docs/archive/quality-evidence/2026-11-01/` (единственные носители этих
     путей в репозитории — подтверждено поиском; скрипты
     `reproduce-stabilization.ps1` на них не ссылаются).
   - Перегенерация: `uv run python tools/quality/generate_quality_docs.py`
     (обновит сгенерированные секции трёх quality-md).
5. **docs/archive/README.md** — короткая страница: что лежит в архиве
   (разовые отчёты аудитов до 2026-11, evidence до 2026-11-09), почему
   (датированные артефакты процесса, не живая документация), что с ним делать
   (читать как историю, новые ссылки сюда не создавать; свежие прогоны — в
   `docs/quality/evidence/<date>/`).
6. **docs/README.md** — индекс: таблица «раздел — что это — для кого»,
   политика evidence, правило «архив для истории, живые доки для работы»,
   языковая конвенция (dev-документация — английский, операторские — русский).
   Ссылки на все живые разделы, включая новые доки из Части 2.
7. **Опциональная чистка** — удалить с диска неотслеживаемый
   `docs/quality/probes/__pycache__/` (покрыт `.gitignore`).
8. **Проверка ссылок** — скрипт: для всех `.md` репозитория (кроме `.venv`,
   `node_modules`, `docs/quality/evidence/`) собрать относительные ссылки вида
   `[text](path)` и убедиться, что цели существуют. Запускается до и после
   переносов; критерий — **нет новых** битых ссылок. Базовая линия (зафиксировать
   в начале): 37 битых ссылок на `../../.audit/...` — scratch-каталог, который
   не существует на диске и никогда не коммитился; они лежат в трёх аудитах
   (включая два защищённых не-целью), поэтому не чинятся, а фиксируются как
   известные исключения.

## Часть 2. Четыре недостающих документа

1. **`docs/api-reference.md` (EN)** — сводный референс HTTP-интерфейса
   `/api/v1` (REST + WebSocket `/api/v1/ws`). Источник: 20 route-модулей
   `src/kraken_server/interface/http/routes/` (`projects`, `layers`,
   `representations`, `artifacts`, `jobs`, `agent`, `analysis`, `electrical`,
   `multilayer`, `quality`, `reviews`, `notes`, `work_orders`,
   `work_planning`, `pipeline`, `auth`, `session`, `commands`, `project_reads`,
   `realtime`). Для каждого ресурса: метод, путь, аутентификация, обязательные
   заголовки (`If-Match`, `Idempotency-Key`), краткая семантика, ссылка на
   фичевый док за подробностями. Документ дополняет фичевые доки, не заменяет.
2. **`docs/data-model.md` (EN)** — модель данных: локальный проект (JSONL
   event-сегменты, `.index/read.sqlite3` как disposable read model, BlobStore
   SHA-256, project lock, single-writer), shared-проект (PostgreSQL authority:
   events, projections, identity, ACL, jobs; transactional outbox; BlobStore
   отдельно; HMAC-тикет на gateway), нотация event-стримов
   (`<entity>:<id>`, напр. `multilayer:{project_id}`,
   `quality-issue:<UUID>`, `layer-pipeline:<layer UUID>`), иммутабельные
   artifact versions, независимые версии схем событий/джобов/бандлов,
   upcasters. Источники: `docs/architecture/README.md` (Data authority),
   `migrations/`, доменная кодовая база.
3. **`src/kraken_core/README.md` (EN)** — что такое `kraken_core` (technical
   runtime и wire-протокол плагинов): модули (`plugin_protocol`, `ipc`,
   `analysis_bundle`, `analysis_run_protocol`, `analysis_protocol`,
   `frame_matrix`, `safe_files`, `runtime`, `process_lifetime`, `media_types`,
   `model_input`, `settings`, …), что плагины используют, контракты
   versioned manifests, правила версионирования. Формат — как у README
   `kraken_server`/`kraken_hub`/`kraken_agent`. Источник: код самого пакета.
4. **`docs/hub-user-guide.ru.md` (RU)** — операторское руководство по
   Hub (PyQt desktop): вход (локальные аккаунты, домен/LDAP), каталог
   проектов (создание, импорт директорий, rename/archive/restore), frame
   matrix, управление слоями и репрезентациями, рабочие планы и board,
   quality issues, технологические процессы (workflow), electrical-модель,
   бэкап/восстановление, обновления Hub. Скоуп: основные потоки оператора
   со ссылками на фичевые доки за техническими подробностями; не исчерпывающий
   справочник по каждому диалогу. Источники: `src/kraken_manager/presentation/qt/`,
   `src/kraken_hub/presentation/qt/parts/login.py` и `domain_login.py`
   (вход: локальные и доменные/LDAP-аккаунты), фичевые доки.

Языки: 1–3 — английский (соответствие существующей dev-документации), 4 —
русский (UI Hub русский).

## Оставлено реализатору

- Точный список относительных ссылок, требующих правки при переносе
  (механика, находит проверка п. 8).
- Внутридокументная структура и глубина четырёх новых доков в рамках
  заданного скоупа (например, таблица против секций на ресурс в API-референсе).
- Порядок разделов в `docs/README.md`.

## Проверка

0. **Базовая линия** (до начала): зафиксировать список незакоммиченных
   изменений и битых ссылок. В дереве уже есть чужие незавершённые правки
   (`.gitignore`, 8 evidence-файлов `2026-11-10`, 3 неотслеживаемых файла) —
   они не трогаются; критерий diff — «нет новых незапланированных изменений»
   относительно базовой линии.
1. Скрипт проверки md-ссылок: до переносов — фиксация битых ссылок (п. 8),
   после — нет новых.
2. `uv run python -m pytest tests/test_quality_registry.py
   tests/test_catalog_operation_parity.py` — проходят (после перегенерации
   quality-md и обновления пути в parity-тесте).
3. `uv run python tools/quality/generate_quality_docs.py --check` — без diff.
4. `git status` / `git diff --stat` — относительно базовой линии только
   запланированные файлы: README.md, ADR 0004, переносы в `docs/archive/`,
   registry JSON, перегенерированные quality-md,
   `tests/test_catalog_operation_parity.py`, `plugins/csliser/README.md`,
   новые `docs/README.md`, `docs/archive/README.md`, четыре новых дока.
