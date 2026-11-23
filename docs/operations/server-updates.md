# Обновления через Kraken Server

Server читает опубликованные файлы из отдельного каталога. Публикация производится оператором на серверной машине; HTTP-маршруты доступны только для чтения и не требуют входа.

## Настройка Server

Добавьте в `server.toml`:

```toml
[updates]
root = "D:/KrakenUpdates"
```

Путь может быть относительным к каталогу `server.toml`. Если секция отсутствует, маршруты обновлений возвращают 404. Храните эту папку отдельно от BlobStore проектов. Перезапустите Server после изменения `server.toml`.

## Структура папки

```text
D:/KrakenUpdates/
  kraken-hub/stable/version.json
  kraken-hub/stable/KrakenDesktopSetup.exe
  contour/stable/version.json
  contour/stable/ContourSetup.exe
  neuralimage/stable/version.json
  neuralimage/stable/NeuralImageSetup.exe
```

Для канала `beta` создайте аналогичную подпапку. Имена файлов внутри канала должны быть простыми, без вложенных путей. Пример `version.json`:

```json
{
  "version": "2.0.0",
  "channel": "stable",
  "releases": [
    {
      "version": "2.0.0",
      "platform": "windows",
      "download_url": "ContourSetup.exe",
      "sha256": "<64 шестнадцатеричных символа SHA-256 установщика>"
    }
  ]
}
```

Перед публикацией вычислите хеш: `Get-FileHash .\ContourSetup.exe -Algorithm SHA256`. Вставьте значение `Hash` в `sha256`. Сначала скопируйте и проверьте установщик, затем атомарно замените `version.json`. Сервер возвращает 404 для манифеста с отсутствующим файлом, внешним адресом или без SHA-256. Клиенты откажутся запускать файл при несовпадении хеша.

Проверка без входа (подставьте адрес Server):

```powershell
Invoke-RestMethod 'https://kraken.example.org/api/v1/updates/contour/stable/version.json'
```

Адрес файла `download_url` разрешается относительно адреса манифеста. Server не принимает загрузку файлов через API. На сетевом Server используйте HTTPS: манифест и хеш, доставленные по незащищенному соединению, не подтверждают источник обновления.

## Клиенты

Hub использует адрес уже подключенного или сохраненного Server, либо `KRAKEN_SERVER_URL`. Contour и NeuralImage используют `KRAKEN_SERVER_URL`; при его отсутствии — локальный `http://127.0.0.1:8080`. Для отдельного клиента на другой машине задайте `KRAKEN_SERVER_URL` адресом доступного Server. Явные `KRAKEN_UPDATE_URL`, `CONTOUR_UPDATE_URL` и `NEURALIMAGE_UPDATE_URL` сохраняют приоритет для совместимости. Проверка манифеста доступна до входа в Server. Установка по-прежнему требует подтверждения в клиенте.
