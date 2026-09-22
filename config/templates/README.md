# Шаблоны конфигурации

Это справочные файлы. Kraken не подхватывает их сам.

| Файл | Назначение |
|---|---|
| `server.local.template.toml` | Локальный сервер на `127.0.0.1:8080`, без пароля БД в файле |
| `server.production.template.toml` | Сетевой сервер с TLS и HTTPS Blob Gateway |
| `desktop.env.template` | Адрес сервера и GitLab для Kraken Hub |
| `agent.env.template` | Каталог данных и машинный токен Kraken Agent |

Рабочий `server.toml` создаёт админская консоль. Секреты она пишет рядом, в
`database-url.secret` и `blob-gateway.secret`.

Локальный путь по умолчанию:

- Windows: `%LOCALAPPDATA%\Kraken\LocalServer\server.toml`
- Linux и macOS: `~/.config/kraken/local-server.toml`

```powershell
uv run kraken-admin init
uv run kraken-server --config "$env:LOCALAPPDATA\Kraken\LocalServer\server.toml"
```

Каталог `config/local/` в git не входит: туда можно положить свою копию шаблона
и сгенерированные секреты.

```powershell
New-Item -ItemType Directory -Force config\local | Out-Null
Copy-Item config\templates\server.local.template.toml config\local\server.toml
uv run kraken-admin init --config config\local\server.toml
```

Тот же набор полей описан в [`src/kraken_server/README.md`](../../src/kraken_server/README.md).
Сборка установщика по-прежнему кладёт короткие примеры в `packaging/config/`.
