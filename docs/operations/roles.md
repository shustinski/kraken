# Учетные записи, роли и токены

Здесь «роль» означает три разных механизма. **Роль PostgreSQL** дает доступ к БД, **аккаунт Kraken** выполняет вход, **роль проекта** разрешает действия. Машинный токен Agent — отдельная сущность.

## PostgreSQL

| Имя | Где | Назначение и создание |
|---|---|---|
| `postgres` (типичное имя) | Основной кластер | Администратор установки PostgreSQL; фактическое имя может отличаться |
| `kraken_local_app` (по умолчанию) | Основной кластер | `kraken-admin init` создает `LOGIN`-роль и принадлежащую ей БД `kraken_local` |
| `kraken_audit` | Временный тестовый кластер | `reproduce-stabilization.ps1` вызывает `initdb -U kraken_audit`; кластер затем останавливается |
| `kraken_backup_test_owner` | Упомянут в плане W3 | Предлагался для S14 в основном кластере; существование не подтверждено |

`kraken_local_app` не получает `CREATEDB` при `init`. Поэтому обычный Server работает, но тест S14, создающий базы `kraken_backup_test_<uuid>`, с этой ролью пропускается. Для S14 уже есть [изолированный тест](runbook.md#изолированный-тест-backuprestore-s14). Наличие `kraken_audit` в старом журнале не означает, что она существует в основном кластере сейчас.

Выберите нужные host, port и database в `psql`, затем проверьте **этот** кластер без вывода паролей:

```sql
SELECT current_database(), current_user, inet_server_port();
SELECT rolname, rolcanlogin, rolcreatedb, rolsuper
FROM pg_roles
WHERE rolname IN ('postgres', 'kraken_local_app', 'kraken_audit', 'kraken_backup_test_owner')
ORDER BY rolname;
```

Имена рабочей роли и БД задаются мастером `init` и могут отличаться от примеров.

## Аккаунты Kraken

| Тип | Хранение и права |
|---|---|
| Локальный аккаунт Desktop | SQLite рабочей станции; `bootstrap-local`; вход в локальный Hub |
| Аккаунт Server | PostgreSQL Server; вход в общие проекты |
| `server_admin` | Системная роль аккаунта Server; управление сервером и назначение проектных ролей |
| GitLab / LDAP principal | Внешняя идентичность; проектные права выдаются отдельно |

Посмотреть существующие серверные аккаунты и метаданные токенов на машине Server:

```powershell
uv run kraken-admin account-list
uv run kraken-admin list-agent-tokens --config <путь-к-server.toml>
```

Создать обычный серверный аккаунт и затем дать ему роль администратора Server (пароль `account-create` спросит интерактивно):

```powershell
uv run kraken-admin account-create --username operator --display-name Operator
uv run kraken-admin grant-admin --username operator
```

Вторую команду выполняйте только если этому человеку действительно нужны административные права. Проектную роль она не создает.

Для аккаунтов есть `account-create`, `account-enable`, `account-disable`, `grant-admin`, `revoke-admin`, `reset-password`, `revoke-sessions`, `recover-admin`. `bootstrap-admin` создает первого администратора в подготовленном хранилище; `recover-admin` возвращает роль существующему логину и отзывает сессии. Точный синтаксис: `uv run kraken-admin <команда> --help`. [Справка CLI](../../src/kraken_server/README.md).

## Роли проекта

Правила заданы в `src/kraken_manager/domain/roles.py`. Создатель проекта становится `maintainer`. Клиент передает выбранную роль в `X-Kraken-Role`; Server проверяет и назначение пользователю, и разрешение действия.

| Роль | Включает и позволяет | Может назначать |
|---|---|---|
| `viewer` | Просмотр, история, статистика | Никого |
| `corrector` | `viewer`, заметки и возврат на доработку | Никого |
| `elementer` | `corrector`, импорт и плагины | Никого |
| `sewer` | `viewer`, импорт, плагины и заметки | Никого |
| `maintainer` | `elementer`, `sewer`, структура, работа, ACL | `maintainer`, `sewer`, `corrector`, `elementer`, `viewer` |
| `admin` | Все проектные разрешения | Любую проектную роль |

`server_admin` также может назначать проектные роли. Старые имена `owner`/`manager`, `contributor`, `reviewer` приводятся к `maintainer`, `elementer`, `corrector`; в новых инструкциях используйте актуальные. Роли назначаются через интерфейс проекта; у локального Admin есть команда `revoke-maintainer`.

## Машинные токены Agent

Токен Agent не является аккаунтом человека или ролью PostgreSQL. При выпуске задаются операции через повторяемый `--capability`; секрет выводится один раз:

```powershell
uv run kraken-admin create-agent-token --config <путь-к-server.toml> --name worker-1 --capability frames.vectorize.v1
uv run kraken-admin list-agent-tokens --config <путь-к-server.toml>
uv run kraken-admin revoke-agent-token --config <путь-к-server.toml> --token-id <id>
```

Это примеры, они ничего не создают до исполнения. Не записывайте в vault пароль, URL с паролем или значение токена. [Запуск Agent](runbook.md#agent-и-плагины).
