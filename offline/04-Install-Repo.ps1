# 04 — установить репозиторий из kit (после 03, новый терминал).
. "$PSScriptRoot\_lib.ps1"

$kitGuess = $null
foreach ($c in @((Get-Location).Path, $PSScriptRoot, (Join-Path $PSScriptRoot ".."))) {
    if ((Test-Path (Join-Path $c "source")) -and (Test-Path (Join-Path $c "dependencies"))) {
        $kitGuess = $c
        break
    }
}
if ($null -eq $kitGuess) { $kitGuess = (Get-Location).Path }

$kitPath = Read-PathOrDefault "Корень kit" $kitGuess
$destination = Read-PathOrDefault "Куда поставить репозиторий" "D:\code\kraken"

$installer = Join-Path $kitPath "Install-OfflineKit.ps1"
if (-not (Test-Path -LiteralPath $installer)) {
    $installer = Join-Path $PSScriptRoot "Install-OfflineKit.ps1"
}

& $installer -KitPath $kitPath -DestinationPath $destination

Write-Host ""
Write-Host "DONE: $destination"
Write-Host "Дальше при желании: git remote add origin http://<gitlab>/... && git push -u origin --all"
Pause
