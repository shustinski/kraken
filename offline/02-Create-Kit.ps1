# 02 — собрать offline kit (нужен интернет, чистый git status).
. "$PSScriptRoot\_lib.ps1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defaultOut = Join-Path $PSScriptRoot ("kit-" + (Get-Date -Format "yyyy-MM-dd"))
$defaultTc = Resolve-FirstExisting @(
    (Join-Path $PSScriptRoot "16092026\offline_toolchain"),
    (Join-Path $PSScriptRoot "offline_toolchain"),
    "D:\Kraken-offline-toolchains-2026-09-16",
    "D:\offline_toolchain"
)
if ($null -eq $defaultTc) { $defaultTc = Join-Path $PSScriptRoot "offline_toolchain" }

Write-Host "Сборка offline kit"
Write-Host "Репозиторий: $repoRoot"
Write-Host ""

$outputPath = Read-PathOrDefault "Папка kit (новая или для resume)" $defaultOut
$toolchainDirectory = Read-PathOrDefault "Папка toolchains" $defaultTc

if (-not (Test-Path -LiteralPath $toolchainDirectory)) {
    throw "Папка toolchains не найдена: $toolchainDirectory"
}

& (Join-Path $PSScriptRoot "Create-OfflineKit.ps1") `
    -OutputPath $outputPath `
    -ToolchainDirectory $toolchainDirectory

Write-Host ""
Write-Host "DONE. Скопируйте папку kit на флешку/диск."
Write-Host "На офлайн-ПК: 03-Install-Toolchains.ps1 затем 04-Install-Repo.ps1"
Pause
