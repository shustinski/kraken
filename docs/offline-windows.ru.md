# Офлайн-разработка Kraken на двух компьютерах Windows

Скрипт `scripts/offline/sync.ps1` переносит историю Git и замкнутый набор зависимостей uv между компьютером с интернетом (ПК A) и компьютером без интернета (ПК B). Каталог `.venv` проекта не копируется. Импорт не делает `reset`, `clean` и не перезаписывает незакоммиченные файлы.

Оба компьютера: 64-битная Windows, Git. Для сборки нативных частей Contour и Blob Gateway дополнительно нужны Visual Studio Build Tools с C++ и Rust; uv их не ставит.

## Почему в обычном `uv sync` нет PyTorch

Команда

```powershell
uv sync --locked --all-groups --all-extras
```

синхронизирует только корневой пакет `kraken`. У него нет групп зависимостей, поэтому `--all-groups` ничего не добавляет. PyTorch не входит в зависимости корня.

`torch==2.12.0+cu132` и `torchvision==0.27.0+cu132` — необязательные зависимости пакета `neuralimage` (extras `ml`, `compute`, `server`, `full`). На Windows и Linux они берутся только из индекса CUDA 13.2:

`https://download.pytorch.org/whl/cu132`

Сборка `+cpu` — другой дистрибутив. Скрипт её не подставляет. Полный набор, включая CUDA-сборку, выбирается так:

```powershell
uv sync --frozen --all-packages --all-extras --all-groups
```

Проверено на Python 3.14.2. Требование проекта — `>=3.14`. Python 3.15 в этом контуре не проверялся.

## Первичная настройка

На ПК A, в каталоге репозитория:

```powershell
.\scripts\offline\sync.ps1 -Mode export -TransferPath E:\kraken-transfer
```

По умолчанию `-ToolMode Bundle`: в перенос попадают `uv.exe` и управляемый интерпретатор той же версии, что и локальный Python 3.14. Повторный экспорт не скачивает зависимости заново, если `uv.lock` не менялся.

Если Python 3.14 и uv уже стоят на обоих компьютерах и везти интерпретатор не нужно:

```powershell
.\scripts\offline\sync.ps1 -Mode export -TransferPath E:\kraken-transfer -ToolMode Validate
```

Такой перенос не поставит Python на чистый ПК B.

Скопируйте каталог `E:\kraken-transfer` на съёмный диск целиком. Не копируйте `.venv`.

На ПК B, если репозитория ещё нет:

```powershell
E:\kraken-transfer\tools\sync.ps1 -Mode import -TransferPath E:\kraken-transfer -RepoRoot D:\code\kraken
```

Импорт клонирует бандл и выполняет `uv sync --offline` в `D:\code\kraken\.venv`. Если каталог уже является репозиторием, история подтягивается в `refs/remotes/kraken-offline/*`, локальные ветки и незакоммиченные файлы остаются на месте.

Проверка, что среда собралась только из переноса:

```powershell
.\scripts\offline\sync.ps1 -Mode verify -TransferPath E:\kraken-transfer
```

Команда клонирует бандл во временный каталог, создаёт новый venv и ставит зависимости с `--offline`. Отчёт: файл рядом с переносом, `E:\kraken-transfer.verify-report.json`.

## Ежедневная работа в обе стороны

1. Закоммитьте изменения. Незакоммиченные файлы в бандл не входят; экспорт об этом пишет `OFFLINE_INFO: dirty-export`.
2. На исходном компьютере снова выполните `export` в каталог на диске.
3. На втором компьютере выполните `import` с тем же `-TransferPath` и `-RepoRoot`.
4. Влейте принесённую ветку сами. Скрипт печатает одну из меток:
   - `OFFLINE_INFO: fast-forward` — локальная ветка является предком принесённой. Команда: `git merge kraken-offline/<ветка>`.
   - `OFFLINE_INFO: diverged` — ветки разошлись. Смотрите оба набора коммитов и выполните тот же `git merge`. Конфликты решайте вручную и сделайте merge-коммит.
   - `OFFLINE_INFO: local-ahead` — в переносе нет новых коммитов этой ветки. Ничего сливать не нужно.
   - `OFFLINE_INFO: already-integrated` — указатели совпадают.

Принудительный push и reset скрипт не выполняет. Повторять import безопасно: он обновляет только remote-tracking refs `kraken-offline/*`.

Пример после import на ветке `main`:

```powershell
git log --oneline HEAD..kraken-offline/main
git merge kraken-offline/main
```

## Зависимости

Источник версий — `pyproject.toml` и `uv.lock`. На ПК A после изменения зависимостей:

```powershell
uv lock
.\scripts\offline\sync.ps1 -Mode export -TransferPath E:\kraken-transfer
```

`uv lock` нужен сети. Не редактируйте кэш uv руками. Если в переносе уже есть кэш для того же `uv.lock`, экспорт печатает `OFFLINE_INFO: dependencies-reused`.

PyTorch ставится только как `2.12.0+cu132` вместе с остальными пакетами workspace при `--all-packages --all-extras`. Наличие CUDA-устройства проверяется отдельно: колёса CUDA ставятся и без видеокарты, а `torch.cuda.is_available()` будет ложным, пока на машине нет совместимого драйвера. Это не повод ставить CPU-сборку.

## Сбои

| Сообщение | Что сделать |
|---|---|
| `missing-transfer` | Нет каталога переноса. Проверьте букву диска. |
| `invalid-manifest` | `manifest.json` повреждён или не совпадает с `manifest.sha256`. Повторите экспорт. |
| `missing-file` | Копирование на диск оборвалось: в манифесте есть файл, которого нет. Скопируйте каталог заново. |
| `checksum` | Файл есть, но SHA-256 не совпадает. Повторите копирование. |
| `python-incompatible` | На компьютере Python старше, чем `requires-python` (сейчас `>=3.14`). |
| `platform-incompatible` | Перенос собран не для этой ОС или архитектуры. Нужна 64-битная Windows. |
| `missing-git-objects` | Бандл неполный. Нужен полный экспорт `--branches --tags`, а не инкрементальный бандл без базы. |
| `offline-install` | В кэше нет замкнутого набора колёс, либо сборка пакета не удалась без сети. Повторите экспорт на ПК A. |
| `torch-cpu` | В среде оказалась CPU-сборка. Не заменяйте индекс `cu132`. |
| `submodules` | В репозитории есть `.gitmodules`. Этот перенос субмодули не возит. |
| `lfs-missing` | Git LFS включён, но объекты не скачаны. На ПК A выполните `git lfs pull` и экспортируйте снова. |

Незакоммиченная работа при ошибке import не удаляется: fetch либо проходит целиком, либо не двигает `HEAD`.

## Что проверка не доказывает

`verify` запускает uv с `--offline` и направляет `HTTP(S)_PROXY` на локальный счётчик. Это ловит обращения uv к прокси. Прямые соединения мимо прокси и правила брандмауэра Windows здесь не включаются: в отчёте `os_firewall` равен `false`. Отдельный физический ПК B в этой проверке не использовался; изоляция воспроизводится локально новым каталогом, новым venv и кэшем только из переноса.

Blob Gateway (`blob_gateway`) требует Rust и сетевые crates.io либо заранее подготовленный cargo vendor. В этот перенос vendor не кладётся, поэтому `cargo build` офлайн не обещается. Сборка расширений Contour на ПК B требует компилятор C++ из Visual Studio Build Tools.
