# Kraken Agent

Локальный агент плагинов. Держит очередь заданий в SQLite и слушает только
loopback. В режиме воркера забирает задания с Kraken Server по отзывному
машинному токену.

Отладка: **Run and Debug** → **Kraken Agent**.

```powershell
uv sync --extra server
uv run kraken-agent --help
```

Токен для режима воркера выпускает `kraken-admin create-agent-token`. Команда
описана в [`kraken_server/README.md`](../kraken_server/README.md).

## Команды

| Параметр | По умолчанию | Пояснение |
|---|---|---|
| `--data-dir` | `%USERPROFILE%\.kraken\agent` или `KRAKEN_AGENT_DATA` | Каталог очереди и промежуточных файлов |
| `--host` | `127.0.0.1` | Только loopback: `127.0.0.1` или `::1` |
| `--port` | `0` | Порт контрольного HTTP. `0` — выбрать свободный |
| `--token` | случайный | Одноразовый токен локального управления. Если не задан, агент генерирует его сам |
| `--plugins-config` | нет | JSON-реестр «операция → команда». В режиме воркера обязателен |
| `--connection-file` | нет | Куда записать JSON с URL и токеном после старта. Права файла `0600` |
| `--server-url` | нет | Адрес Kraken Server. Включает режим воркера |
| `--server-token` | `KRAKEN_AGENT_TOKEN` | Машинный токен воркера |
| `--lease-seconds` | `60` | На сколько секунд воркер берёт задание |

Локальный запуск без сервера:

```powershell
uv run kraken-agent --plugins-config plugins.json
```

Воркер:

```powershell
uv run kraken-agent --server-url http://127.0.0.1:8080 --plugins-config plugins.json
```

Шаблон переменных: [`config/templates/agent.env.template`](../../config/templates/agent.env.template).
