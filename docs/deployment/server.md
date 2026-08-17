# Развёртывание Kraken Server

Для обычной локальной установки не требуются SQL, `psql`, Python или `uv`.
Достаточно установленной и запущенной PostgreSQL и команды `KrakenAdmin`.

## Самый простой локальный запуск

В каталоге установленного Kraken Server выполните:

```powershell
.\KrakenAdmin.exe init
```

CLI откроет пошаговый мастер и запросит:

1. адрес и порт PostgreSQL;
2. логин и пароль администратора PostgreSQL;
3. имя создаваемой БД;
4. логин и пароль служебного пользователя БД;
5. логин, отображаемое имя и пароль первого администратора Kraken.

Для несекретных значений мастер показывает безопасные локальные значения по
умолчанию. Их можно принять клавишей Enter. Пароли вводятся скрыто и требуют
подтверждения там, где создаётся новая учётная запись.

Остальное выполняется автоматически:

- создаётся отдельный служебный пользователь PostgreSQL;
- используется заданный в мастере пароль доступа к БД;
- создаётся БД `kraken_local`;
- выполняются все миграции;
- создаётся файловое хранилище;
- настраивается и автоматически запускается `KrakenBlobGateway.exe`;
- записывается комментированный `server.toml`;
- строка подключения и ключ файловых разрешений шифруются Windows DPAPI;
- создаётся первый пользователь Kraken с ролью `server_admin`.

Запуск сервера:

```powershell
.\KrakenServer.exe --config "$env:LOCALAPPDATA\Kraken\LocalServer\server.toml"
```

Проверка:

```powershell
.\KrakenAdmin.exe doctor
```

Проверка параллельной пропускной способности файлового контура:

```powershell
.\KrakenAdmin.exe blob-benchmark --clients 30 --size-mib 1024
```

Команда не создаёт проект и не обращается к PostgreSQL. Она передаёт 30
одновременных потоков непосредственно через Blob Gateway и показывает
суммарную скорость. Для проверки реальной 10GbE-сети запускайте её на отдельной
машине; loopback-тест дополнительно ограничен генератором данных и локальным
диском клиента.

Создание проекта без Desktop:

```powershell
.\KrakenAdmin.exe project-create --name "Тестовый проект" --width 20 --height 30
```

CLI запросит пароль Kraken и выведет идентификатор проекта. Список проектов:

```powershell
.\KrakenAdmin.exe project-list
```

## Готовые PowerShell-сценарии

В поставку входят сценарии из каталога `scripts`:

| Сценарий | Назначение |
|---|---|
| `Initialize-KrakenLocal.ps1` | Создать БД, конфигурацию и первого администратора |
| `Start-KrakenLocal.ps1` | Запустить локальный сервер в текущем окне |
| `Test-KrakenLocal.ps1` | Проверить конфигурацию, БД и хранилище |
| `Test-KrakenBlobGateway.ps1` | Измерить параллельную пропускную способность файлового шлюза |
| `New-KrakenProject.ps1` | Создать проект без ручной работы в UI |

Пример полного сценария:

```powershell
.\scripts\Initialize-KrakenLocal.ps1
.\scripts\Start-KrakenLocal.ps1
```

Во втором окне:

```powershell
.\scripts\New-KrakenProject.ps1 -Name "Демонстрация" -Width 10 -Height 10
```

## Автоматизация без интерактивных вопросов

Секреты не следует передавать аргументами командной строки. Для CI или
установочного скрипта используются переменные окружения:

```powershell
$env:KRAKEN_POSTGRES_ADMIN_PASSWORD = "..."
$env:KRAKEN_DATABASE_PASSWORD = "..."
$env:KRAKEN_INITIAL_ADMIN_PASSWORD = "..."

.\KrakenAdmin.exe init `
  --database-host 127.0.0.1 `
  --database-port 5432 `
  --postgres-admin postgres `
  --database-name kraken_local `
  --database-user kraken_local_app `
  --username admin `
  --display-name Administrator `
  --non-interactive
```

Для операций с проектами:

```powershell
$env:KRAKEN_ACCOUNT_PASSWORD = "..."

.\KrakenAdmin.exe project-create `
  --name "CI project" `
  --width 5 `
  --height 5 `
  --non-interactive `
  --json
```

`--json` выдаёт машиночитаемый результат. Нулевой код возврата означает
успех, `2` — ошибку проверки или операции, `130` — отмену пользователем.

## Справка CLI

Список команд и их назначение:

```powershell
.\KrakenAdmin.exe --help
```

Аргументы, значения по умолчанию и примеры конкретной команды:

```powershell
.\KrakenAdmin.exe init --help
.\KrakenAdmin.exe project-create --help
.\KrakenAdmin.exe doctor --help
```

## Конфигурация

Сгенерированный `server.toml` содержит пояснение каждого параметра. Пароля
PostgreSQL в нём нет: поле `database.url_secret` ссылается на отдельный файл,
зашифрованный Windows DPAPI.

## Высокоскоростной файловый контур

`KrakenServer.exe` обслуживает авторизацию, проекты, события и PostgreSQL.
Гигабайтные файлы Desktop и Agent передают напрямую в Rust-компонент
`KrakenBlobGateway.exe` по краткоживущему HMAC-разрешению. Gateway за один
проход записывает временный объект, вычисляет SHA-256, выполняет `fsync` и
атомарно публикует объект через hard link. Python не копирует файл повторно.

Gateway поддерживает параллельные потоки, HTTPS, `HEAD` и одиночные HTTP Range.
Незавершённые временные загрузки старше 24 часов очищаются при запуске.
PostgreSQL хранит только метаданные объекта.

Для локального теста используются адреса `127.0.0.1:8080` и
`127.0.0.1:8081`. Для сетевого доступа задайте, например:

```powershell
.\KrakenAdmin.exe setup-server `
  --host 0.0.0.0 `
  --port 8080 `
  --blob-gateway-host 0.0.0.0 `
  --blob-gateway-port 8081 `
  --blob-gateway-public-url https://kraken-files.example.org `
  --tls-cert-file C:\Kraken\tls\server.crt `
  --tls-key-file C:\Kraken\tls\server.key
```

Удалённый `public_url` обязан использовать HTTPS. Сертификат и ключ применяются
как к основному серверу, так и к Blob Gateway. Секрет Gateway хранится в
`blob-gateway.secret`, защищённом DPAPI, и никогда не передаётся Desktop.

В каталоге `config` поставляются два справочных файла:

- `server.local.example.toml` — безопасные настройки локального теста;
- `server.production.example.toml` — TLS, reverse proxy, хранилище и GitLab.

Это справочные шаблоны. Рабочую конфигурацию создаёт `init` или
расширенная команда `setup-server`.

## Установка Windows-службы

Установщик Kraken Server запускает автоматическую настройку и передаёт
`--install-service`. Для ручной регистрации уже настроенного сервера:

```powershell
.\KrakenAdmin.exe install-service `
  --config "C:\ProgramData\Kraken\Server\server.toml" `
  --server-executable "C:\Program Files\Kraken Server\KrakenServer.exe"
```

## Подключение Kraken Desktop

Для Desktop на том же компьютере укажите:

```text
http://127.0.0.1:8080
```

Способ входа — «Учётная запись Kraken», логин по умолчанию — `admin`.
Пароль Desktop не сохраняет. Раздел «Администрирование» отображается только
пользователю с ролью `server_admin`.

Для сетевого доступа используйте HTTPS/WSS. PostgreSQL и файловое хранилище
не публикуются в сеть и доступны только Kraken Server.

## Расширенный режим

Если БД создаёт внешняя инфраструктура, используйте:

```powershell
.\KrakenAdmin.exe setup-server --help
```

Эта команда принимает URL уже существующей PostgreSQL. Для обычной локальной
установки она не нужна.

## Резервные копии

Полная резервная копия включает PostgreSQL, каталог blobs, каталог `.server`,
`server.toml`, `database-url.secret` и `blob-gateway.secret`. Перенос DPAPI-секретов на
другой компьютер требует повторной генерации конфигурации.
