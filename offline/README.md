# Offline kit — шаги

## 0. Один раз на ПК с интернетом: toolchains

Папка (пример `D:\Kraken-offline-toolchains-2026-09-16`) + `toolchain-manifest.json`:

- Python 3.14.2 x64 installer  
- uv 0.10.2 (zip или `uv.exe`)  
- Git + Git LFS installers  
- Rust (`rustup-init` / toolchain)  
- `vs-build-tools-layout\` (VCTools + SDK)  
- Inno Setup installer  

Шаблон манифеста: `toolchain-manifest.example.json`.

---

## 1. Первый раз / обновление deps: собрать kit (ПК с интернетом)

Репозиторий чистый (`git status` пустой).

```powershell
.\offline\Create-OfflineKit.ps1 `
  -OutputPath D:\Kraken-offline-kit-2026-09-16 `
  -ToolchainDirectory D:\Kraken-offline-toolchains-2026-09-16
```

Кэш wheels/uv/cargo: **`offline/dep-cache/`** в проекте (не в git).  
При следующем kit докачивается только новое. Другой путь: `-CacheDirectory ...`.  
Если сборка упала — тот же `-OutputPath` снова. В корне kit: `README.md`, `Install-OfflineKit.ps1`.  
SHA-256 по умолчанию **выключен**; включить: `-IncludeHashes` (сборка) / `-VerifyHashes` (установка).

---

## 2. Первый раз: поставить toolchain на ПК без интернета

Все установщики в `toolchains\` (имена смотрите в `toolchains\toolchain-manifest.json`).  
Часть шагов — **от администратора**. После установки откройте **новый** терминал.

Ниже `$TC` = путь к `...\toolchains` внутри kit, например  
`E:\Kraken-offline-kit-2026-09-16\toolchains`.

### 2.1 Python

```powershell
# GUI: галка "Add python.exe to PATH", Install for all users (по желанию)
& "$TC\python-*-amd64.exe" /passive InstallAllUsers=1 PrependPath=1 Include_test=0
```

Проверка: `python --version`

### 2.2 uv

Если в kit лежит **zip**:

```powershell
Expand-Archive "$TC\uv-*.zip" -DestinationPath "$env:USERPROFILE\.local\bin" -Force
# добавить в PATH пользователя:
$uvBin = "$env:USERPROFILE\.local\bin"
[Environment]::SetEnvironmentVariable(
  "Path",
  ($env:Path + ";$uvBin"),
  "User"
)
$env:Path += ";$uvBin"
```

Если лежит один **`uv.exe`** — скопируйте его в папку из PATH (например `C:\Windows` нельзя без админа; лучше `$env:USERPROFILE\.local\bin` и добавить в PATH как выше).

Проверка: `uv --version`

### 2.3 Git

```powershell
& "$TC\Git-*-64-bit.exe" /VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS `
  /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"
```

Проверка: `git --version`

### 2.4 Git LFS

```powershell
& "$TC\git-lfs-windows-*.exe" /VERYSILENT /NORESTART
git lfs install
```

Проверка: `git lfs version`

### 2.5 Rust

Нужен уже установленный VS Build Tools (MSVC), иначе toolchain для Windows не соберётся.  
Если сети нет — `rustup` должен работать из уже подготовленного offline‑набора; для типового `rustup-init.exe` из kit на полностью офлайн‑ПК часто нужен заранее скачанный rustup dist. Практичный вариант:

```powershell
# Если на этой машине уже есть доступ к кэшу/файлам rust из kit (папка rust\):
# иначе сначала поставьте VS Build Tools (§2.6), затем:
& "$TC\rustup-init.exe" -y --default-toolchain stable --default-host x86_64-pc-windows-msvc
```

Если `rustup-init` требует сеть — на ПК с интернетом поставьте тот же toolchain и скопируйте `%USERPROFILE%\.cargo` и `%USERPROFILE%\.rustup` на офлайн‑ПК (или используйте offline rustup mirror, если готовили).

Проверка: `rustc --version`, `cargo --version`

### 2.6 VS Build Tools (обязательно `--noWeb` + сертификаты)

```powershell
# Админ PowerShell
cd "$TC\vs-build-tools-layout\certificates"
Get-ChildItem *.cer,*.crt | ForEach-Object { certutil -addstore -f Root $_.FullName }

cd "$TC\vs-build-tools-layout"
.\vs_BuildTools.exe --noWeb --noUpdateInstaller --wait `
  --add Microsoft.VisualStudio.Workload.VCTools `
  --includeRecommended
```

Если установщик сразу закрывается / signature error — на ПК с интернетом скачайте  
https://www.microsoft.com/pkiops/certs/Microsoft%20Windows%20Code%20Signing%20PCA%202024.crt  
в `certificates\` и снова `certutil` на офлайн‑ПК.  
Лог: `%TEMP%\dd_bootstrapper_*.log`.

### 2.7 Inno Setup

```powershell
& "$TC\innosetup-*.exe" /VERYSILENT /NORESTART /SUPPRESSMSGBOXES
```

Проверка: в меню Пуск есть Inno Setup; для CLI часто `ISCC.exe` в `C:\Program Files (x86)\Inno Setup 6\` или `...\Inno Setup 7\`.

### Порядок

1. Python → uv → Git → Git LFS  
2. VS Build Tools  
3. Rust  
4. Inno Setup  

---

## 3. Установить репозиторий из kit

В корне kit уже лежат `README.md` и `Install-OfflineKit.ps1`.

```powershell
cd E:\Kraken-offline-kit-2026-09-16
pwsh .\Install-OfflineKit.ps1 -KitPath . -DestinationPath D:\code\kraken
```

`DestinationPath` ещё не должен существовать.

Создать пустой проект в локальном GitLab, затем:

```powershell
cd D:\code\kraken
git remote add origin http://<gitlab>/<group>/kraken.git
git push -u origin --all
git push origin --tags
```

Rust при сборке blob_gateway:

```powershell
$env:CARGO_HOME = "E:\Kraken-offline-kit-2026-09-16\dependencies\cargo"
```

---

## 4. Дальше только бандлы (не копировать папку проекта)

Не переносить весь `kraken\` с флешки поверх существующего репо. Только `.bundle`.

Перед export: всё закоммичено, `git status` чистый.

### 4a. Офлайн → онлайн (правки в GitHub)

**Офлайн** (после `git pull` с GitLab):

```powershell
pwsh .\offline\Export-GitUpdate.ps1 -Since <общий-коммит> -OutputPath E:\updates\offline-to-online.bundle
```

**Онлайн:**

```powershell
cd D:\code\kraken
git pull origin
pwsh .\offline\Import-GitUpdate.ps1 -BundlePath E:\updates\offline-to-online.bundle
git merge offline-bundle/<ветка>
git push origin
```

### 4b. Онлайн → офлайн (правки с GitHub в GitLab)

**Онлайн** (после `git pull` с GitHub):

```powershell
pwsh .\offline\Export-GitUpdate.ps1 -Since <общий-коммит> -OutputPath E:\updates\online-to-offline.bundle
```

**Офлайн:**

```powershell
cd D:\code\kraken
git pull origin
pwsh .\offline\Import-GitUpdate.ps1 -BundlePath E:\updates\online-to-offline.bundle
git merge offline-bundle/<ветка>
git push origin
```

Общий коммит после merge:

```powershell
git merge-base HEAD offline-bundle/<ветка>
```

Этот SHA — `-Since` для следующего круга.

---

## Когда снова полный kit

Только если сменились зависимости (`uv.lock`), toolchain или нужен новый чистый ПК. Код между ПК — всегда через §4.
