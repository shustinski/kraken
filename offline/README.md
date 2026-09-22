# Offline kit — что запускать

Все скрипты в папке `offline\`.  
Запуск (админ, если скрипт сам скажет):

```powershell
powershell -ExecutionPolicy Bypass -File D:\code\kraken\offline\00-Menu.ps1
```

Или по одному:

| # | Скрипт | Где | Что делает |
|---|---|---|---|
| 01 | `01-Build-VsLayout.ps1` | ПК с интернетом, **админ** | Качает VS Build Tools layout (Win10 21H2 / SDK 19041). Окно Installer не закрывать. |
| 02 | `02-Create-Kit.ps1` | ПК с интернетом | Собирает kit (спросит пути). Нужен чистый `git status` и папка toolchains. |
| 03 | `03-Install-Toolchains.ps1` | ПК без интернета, **админ**, из корня kit | Ставит Python, uv, Git, Git LFS, VS Build Tools, Rust, Inno Setup. |
| 04 | `04-Install-Repo.ps1` | ПК без интернета | Ставит репозиторий и `.venv` из kit. |
| 05 | `05-Export-Update.ps1` | любой ПК с репо | Пишет `.bundle` на флешку. |
| 06 | `06-Import-Update.ps1` | любой ПК с репо | Читает `.bundle`, потом сами: `git merge offline-bundle/<ветка>`. |

## Короткий цикл

**Первый раз (онлайн):**  
подготовить папку toolchains (после 01 положить layout как `vs-build-tools-layout`) → `02-Create-Kit.ps1` → скопировать kit на диск.

**Первый раз (офлайн):**  
из корня kit → `03-Install-Toolchains.ps1` → новый терминал → `04-Install-Repo.ps1` → при желании `git remote` на GitLab.

**Потом только код:** `05` на одном ПК → флешка → `06` на другом → `git merge` → `git push`.

Полный kit снова — только если сменились зависимости/toolchain или новый ПК.

## Заметки

- Кэш wheels: `offline\dep-cache\` (докачка только нового).  
- SHA-256 выключен по умолчанию.  
- Не копируйте папку проекта поверх git-репо — только bundle (05/06).  
- VS layout: если прервать окно Installer — в логе `%TEMP%\dd_bootstrapper_*.log` будет `User Cancelled Download`.
