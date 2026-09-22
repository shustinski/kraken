# 06 — импорт git-обновления с флешки, затем сами сделайте git merge.
. "$PSScriptRoot\_lib.ps1"

$defaultRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = Read-PathOrDefault "Корень репозитория kraken" $defaultRepo
Push-Location $repo
try {
    $bundle = Read-Host "Путь к .bundle файлу"
    if ([string]::IsNullOrWhiteSpace($bundle)) { throw "Нужен путь к bundle" }
    $bundle = $bundle.Trim().Trim('"')
    & (Join-Path $PSScriptRoot "Import-GitUpdate.ps1") -BundlePath $bundle
    Write-Host ""
    Write-Host "Импорт готов. Дальше вручную:"
    Write-Host "  git branch -r"
    Write-Host "  git merge offline-bundle/<ветка>"
    Write-Host "  git push"
} finally {
    Pop-Location
}
Pause
