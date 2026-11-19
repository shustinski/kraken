# Офлайн-перенос Kraken между двумя Windows

Нужны Git, uv и Python 3.14 на обоих компьютерах. Каталог `.venv` не копируется: в нём прописан путь к исходникам этой машины.

Contour собирает расширение на C++. На офлайн-ПК один раз, от имени администратора, ставится Visual C++ Build Tools из уже скачанного layout. Папка `offline\vs-build-tools-layout-win10-21h2` копируется на флешку, например в `E:\vs-build-tools-layout`:

```powershell
powershell -ExecutionPolicy Bypass -File E:\install-buildtools.ps1 -Layout E:\vs-build-tools-layout
```

Файл `install-buildtools.ps1` тоже нужно положить на флешку: его берут из `scripts\offline`. После установки откройте новый терминал и делайте import.

На компьютере с интернетом:

```powershell
.\scripts\offline\sync.ps1 -Mode export -TransferPath E:\kraken-transfer
```

Скрипт пишет `repo.bundle` и складывает скачанные пакеты в `uv-cache`. Повторный экспорт не качает заново то, что уже лежит в этом кэше.

Скопируйте папку `E:\kraken-transfer` на флешку целиком. Рядом положите `scripts\offline\sync.ps1`: на чистом компьютере репозитория ещё нет, и взять скрипт оттуда нельзя.

На компьютере без интернета:

```powershell
powershell -ExecutionPolicy Bypass -File E:\sync.ps1 -Mode import -TransferPath E:\kraken-transfer -RepoRoot C:\code\kraken
```

Import не двигает текущую ветку и не трогает незакоммиченные файлы. Он только добавляет привезённые коммиты как `kraken-offline/<ветка>`. Влить их:

```powershell
git merge kraken-offline/main
```

Обратно так же: закоммитить, `export` на флешку, `import` и `git merge` на другой машине. Полный набор пакетов нужен в первый раз и после изменения `uv.lock`.
