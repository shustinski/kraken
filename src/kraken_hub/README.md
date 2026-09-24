# Kraken Hub

Desktop-оболочка: каталог проектов, матрица кадров и запуск плагинов.
Отладка: **Run and Debug** → **Kraken Hub**. Конфигурация **Kraken Project Manager (dev auto-login)** открывает тот же Hub и входит как локальный пользователь `vscode`, если `KRAKEN_DEV_AUTO_LOGIN=1`. Перед обычным запуском:

```powershell
uv sync --extra desktop --extra dev
uv run kraken-admin bootstrap-local --username admin --display-name "Administrator"
```

`bootstrap-local` создаёт учётную запись рабочей станции в SQLite. Это не
администратор PostgreSQL-сервера. Команда описана в
[`kraken_server/README.md`](../kraken_server/README.md).

## Команды

```powershell
uv run kraken-hub --help
```

| Параметр | Пояснение |
|---|---|
| `--catalog` | Путь к `plugins.json`. Без флага Hub ищет каталог сам |
| `--plugins-dir` | Каталог установленных плагинов. Без флага используется стандартное расположение |
| `--list` | Напечатать каталог плагинов и выйти, окно не открывать |
| `--update-url` | URL или локальный путь манифеста обновления Hub. То же задаёт `KRAKEN_UPDATE_URL` |
| `--legacy-launcher` | Старое окно запуска отдельных плагинов |
| `--thumbnail-store-uri` | Кэш миниатюр: `sqlite:///путь`, `files:///путь` или `memory://` |

Подключение к общему серверу задаётся окружением, не флагами Hub. Шаблон:
[`config/templates/desktop.env.template`](../../config/templates/desktop.env.template).

```powershell
$env:KRAKEN_SERVER_URL = "http://127.0.0.1:8080"
$env:KRAKEN_GITLAB_TOKEN = "<gitlab-access-token>"
$env:KRAKEN_GITLAB_ISSUER = "https://gitlab.example.com"
uv run kraken-hub
```

Страница администрирования Hub относится к локальному каталогу рабочей станции.
Учётные записи сервера, роль администратора и отзыв maintainer выполняются
только локальной консолью `kraken-admin` на машине сервера.
