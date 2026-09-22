# Меню. Запуск одной строкой:
# powershell -ExecutionPolicy Bypass -File D:\code\kraken\offline\00-Menu.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Start-Script([string]$Name) {
    $path = Join-Path $PSScriptRoot $Name
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "Нет файла: $Name"
        Pause
        return
    }
    Write-Host ""
    Write-Host ">>> $Name"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $path
}

while ($true) {
    Clear-Host
    Write-Host "Kraken offline — выбери номер и Enter"
    Write-Host ""
    Write-Host "  1  Скачать VS layout (Win10 21H2)     [интернет, админ]"
    Write-Host "  2  Собрать kit                         [интернет]"
    Write-Host "  3  Установить toolchain из kit         [офлайн, админ]"
    Write-Host "  4  Установить репозиторий из kit       [офлайн]"
    Write-Host "  5  Экспорт обновления (.bundle)"
    Write-Host "  6  Импорт обновления (.bundle)"
    Write-Host "  0  Выход"
    Write-Host ""
    switch (Read-Host "Номер") {
        "1" { Start-Script "01-Build-VsLayout.ps1" }
        "2" { Start-Script "02-Create-Kit.ps1" }
        "3" { Start-Script "03-Install-Toolchains.ps1" }
        "4" { Start-Script "04-Install-Repo.ps1" }
        "5" { Start-Script "05-Export-Update.ps1" }
        "6" { Start-Script "06-Import-Update.ps1" }
        "0" { break }
        default { Write-Host "Нет такого пункта"; Start-Sleep 1 }
    }
}
