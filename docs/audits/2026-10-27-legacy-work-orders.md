# Старые поручения: перенос из проекции в канонический журнал

27 октября 2026. Закрывается известный сценарий GAP-W02 для поддерживаемых
WorkOrder v1: старое поручение существует в проекции, но ещё не имеет событий.
Обучение модели и оценка качества векторизации не запускались.

## Поведение и контракт

Перестроение SQLite, удаление кэша и создание **нового** canonical bundle
отказываются терять такие поручения. Старый индекс сохраняется. Каталожное
имя проекта остаётся видимым, чтобы открыть административную миграцию;
обычное открытие отстающего проекта продолжает требовать восстановления.

**Администрирование → Миграция старых поручений…** открывает предварительный
план. В нём доступны снимки поручений, назначение, инструкции, состояния и
число старых наблюдений. Полный JSON можно сохранить отдельно. Подтверждение —
кнопка «Сохранить поручения в журнале» после загрузки плана. Изменённый снимок
или даже дополнительное наблюдение с той же aggregate revision меняет digest;
старый план больше не применяется.

Сохраняются исходные поля, автор, исполнитель, revision, timestamps и известные
наблюдения `typed_model_history` / `projections_temporal`. Для них указано
происхождение `legacy_projection`, а история прежних операций остаётся
`unknown`. Наблюдения не выдаются за доказанные события и не подставляются
задним числом в recorded-time history. Новый actor — оператор миграции;
`effective_at` содержит заявленное старое время, `recorded_at` — время импорта.

На каждое поручение записывается `WorkOrderImportedFromProjectionV1` с SHA-256
снимка и наблюдений. Отдельный `LegacyWorkOrdersMigrationRecordedV1` сохраняет
результат и idempotency key. Ключ нельзя использовать для другого digest или
оператора. Snapshot SHA и принадлежность наблюдений проверяются также при
replay; повторно подписанное наблюдение другого проекта отвергается.

Требуется `MIGRATE_PROJECT`. При применении заново проверяются активность
учётной записи, роли и storage profile. Durable ACL revocation перекрывает
отставшее материальное ACL. Обновление principal не возвращает консольную
роль администратора, снятую сетевой композицией. Выбранная acting role
проверяется как `migrate_project`; viewer отказан даже владельцу проекта.
Неизвестные ACL/schema versions не интерпретируются как действующий grant.

Миграция выполняется фоновым Qt worker. Закрытие ждёт завершения;
родительское удаление не уничтожает работающий QThread. Bound Qt slots
отсоединяются при уничтожении диалога, поток удерживается отдельно до finish,
при выходе приложения ожидается его завершение.

В PostgreSQL импорт и receipt — одна транзакция. Локально сначала атомарно
публикуется append-only batch, затем под тем же project lock перестраивается
индекс. Сбой replay оставляет старый индекс и обнаруживаемое отставание;
повтор с тем же ключом восстанавливает индекс без повторных import events.

Лимиты: до 10000 неперенесённых поручений, до 10000 cached observations на
поручение, до 64 MiB сериализованных исходных снимков/наблюдений за операцию.
Превышение отказывает до записи событий. Эти лимиты не являются OS memory SLA.
Пакеты по сохранённым destination не открываются, jobs не запускаются,
размещение external/server не меняется. Повреждённые/неизвестные схемы и
непредставленные чужие streams требуют отдельного разбора; данные не удаляются.
Уже исчезнувшую старую проекцию миграция не может восстановить из ничего.

## Local SDK и HTTP

`plan_legacy_work_order_migration(principal=..., project_id=...)` возвращает
`legacy`, `already_recorded`, `prior_history` и `digest`.
`migrate_legacy_work_orders(..., expected_plan_digest=..., idempotency_key=...)`
применяет только заново проверенный план и возвращает исходный receipt при retry.

GET и POST `/api/v1/projects/{id}/work-orders/legacy-migration` используют
обычную authenticated session. POST принимает `expected_plan_digest` и header
`Idempotency-Key`. Heavy-operation gate ограничивает параллельные операции.
Remote SDK подключён к этим endpoint; sparse in-memory backend сообщает об
отсутствии persistent migration вместо AttributeError/500.

## Доказательства

Evidence root: `.codex_tmp_scratch/production-audit-20261027/`.

| Gate | Результат |
|---|---|
| `work-order-migration-before/` | Исходные failing regressions: потеря при rebuild и отсутствие рабочего пути миграции |
| `work-order-migration-final-targeted/` и `work-order-migration-identity-boundary/` | Сохранность индекса, snapshots/observations, stale digest, retry, backup/import, будущая ACL schema, revoked ACL, actor provenance, follow-up lifecycle/CAS, архитектурные границы |
| `work-order-migration-http-fixed/` / итоговый root | Настоящий PostgreSQL, rollback, конкурентные запросы — один победитель; обычная password/session authentication через HTTP, viewer отказан, Qt retry после отброшенного клиентом уже полученного ответа — один import. Это client-boundary fault injection, не обрыв сокета |
| `test_legacy_work_order_dialog.py` | Кнопки административной панели и настоящий backend/worker; отдельный процесс проверил удаление Qt parent при живом worker; непригодный к отображению план не разрешает Apply |
| `legacy-source-desktop/` | Собственный профиль и 16 реальных JPEG/CIF; миграция двух старых наблюдений, cache destruction/replay и verified backup прошли |
| `legacy-root/summary.json` | 208 модулей, 1360 XML checks (1337 top-level + 23 subtests), 0 failures/errors, 15 skipped, PostgreSQL включён; 2683 pinned files неизменны |
| `legacy-build.log` | Authoritative build exit 0 |
| `legacy-desktop-sources.json` / `legacy-server-sources.json` | Все 469 / 377 frozen project/Contour modules совпали с текущими source; новые модули проверены на обязательное присутствие |
| `legacy-desktop-catalog-resource.json` | Bundled plugin catalog совпал с source, включая актуальный CSliser manifest |
| `legacy-desktop-installed/report.json` | Audit install/acceptance/uninstall прошли за 18.948 s; 16 реальных пар, Qt preview/import, observations, cache replay, backup, raster thumbnails/head refresh и OS secret-store roundtrip |
| `legacy-server-installed/report.json` | Audit install/acceptance/uninstall прошли за 18.421 s; настоящий frozen HTTP/PG import + idempotent retry + process restart persistence, migrations/auth/Gateway/Range/SciPy |

Обе audit registrations и EXE удалены; service ownership hash не изменён.
Server UUID schema удалена. Desktop snapshot окна просмотрен: русский текст,
назначение и управляющие кнопки читаемы; JPEG/CIF originals SHA не изменились.
Installed tests используют собственный projection-only diagnostic, а не
неизвестный реальный архив старой рабочей базы. Qt-ввод внутренний; native
Computer Use pipe остаётся недоступным, ordinary login/20-stage UI не засчитаны.

Ruff изменённых focused modules и scoped whitespace прошли. У transport facade
`section_4.py` сохранены 18 старых F401 aliases: HEAD/current diagnostics
совпали (`legacy-lint-baseline/`). Полный repository Ruff этим не объявляется.
Причины 15 skips сохранены в `legacy-root/skips.json`; overlapping gates к
общему числу не суммируются.

Production installers **скомпилированы, не установлены и не опубликованы**:

- Desktop: 74 529 007 bytes, SHA-256
  `6d0e4ed4d5c70124d04ea1dc615066535f355951af7586e24115112be4656362`.
- Server: 114 872 846 bytes, SHA-256
  `903b225a1f5ae3c8a3a5e04d1dd021ceff0f9617e661aecc82163b1758b0bb8d`.

Они лежат в `legacy-installers/production/`. Audit variants имеют отдельные AppId.
SCM installation/reboot/production update/clean OS/full desktop-remote E2E
остаются отдельными открытыми gates.

## Следующий выявленный риск

`backup-reuse-observation/report.json` воспроизвёл поведение существующего
backup UI: фиксированное имя каталога допускает reuse старого завершённого
bundle. После нового поручения источник имел 9 событий, возвращённый архив —
8; новое поручение отсутствовало, хотя исторический bundle оставался valid.
Это не повреждение старого архива и не новый snapshot. Требуется исправить
семантику повторного создания backup/проверку current head и пользовательский
результат. GAP не закрыт данным этапом; общая готовность Kraken не объявляется.

Последующий этап исправил этот конкретный дефект; проверки и новые сборки
описаны в [протоколе актуальности backup](2026-10-27-backup-freshness.md).
