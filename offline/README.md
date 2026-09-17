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

## 1. Первый раз: собрать kit (ПК с интернетом)

Репозиторий чистый (`git status` пустой).

```powershell
.\offline\Create-OfflineKit.ps1 `
  -OutputPath D:\Kraken-offline-kit-2026-09-16 `
  -ToolchainDirectory D:\Kraken-offline-toolchains-2026-09-16
```

Если упало — тот же `-OutputPath` снова (докачка). Скопировать kit на флешку.

---

## 2. Первый раз: поставить на ПК без интернета

1. Из `kit\toolchains\` поставить Python, uv, Git, Git LFS, Rust, VS Build Tools, Inno Setup.  
2. Установить репо:

```powershell
.\offline\Install-OfflineKit.ps1 `
  -KitPath E:\Kraken-offline-kit-2026-09-16 `
  -DestinationPath D:\code\kraken
```

`DestinationPath` ещё не должен существовать.

3. Создать пустой проект в локальном GitLab, затем:

```powershell
cd D:\code\kraken
git remote add origin http://<gitlab>/<group>/kraken.git
git push -u origin --all
git push origin --tags
```

Rust при сборке:

```powershell
$env:CARGO_HOME = "E:\Kraken-offline-kit-2026-09-16\dependencies\cargo"
```

---

## 3. Дальше только бандлы (не копировать папку проекта)

Не переносить весь `kraken\` с флешки поверх существующего репо. Только `.bundle`.

Перед export: всё закоммичено, `git status` чистый.

### 3a. Офлайн → онлайн (правки в GitHub)

**Офлайн** (после `git pull` с GitLab):

```powershell
# общий коммит с онлайн-ПК (после первого sync — merge-base)
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

### 3b. Онлайн → офлайн (правки с GitHub в GitLab)

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

Этот SHA — `-Since` для следующего круга. Повторять 3a/3b сколько нужно.

---

## Когда снова полный kit

Только если сменились зависимости (`uv.lock`), toolchain или нужен новый чистый ПК. Код между ПК — всегда через §3.
