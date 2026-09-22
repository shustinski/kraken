# 05 — экспорт git-обновления на флешку (чистый git status).
. "$PSScriptRoot\_lib.ps1"

$defaultRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = Read-PathOrDefault "Корень репозитория kraken" $defaultRepo
Push-Location $repo
try {
    $since = Read-Host "Общий коммит (-Since), например результат git merge-base"
    if ([string]::IsNullOrWhiteSpace($since)) { throw "Нужен коммит -Since" }
    $defaultOut = "E:\updates\update-{0}.bundle" -f (Get-Date -Format "yyyyMMdd-HHmm")
    $output = Read-PathOrDefault "Куда сохранить .bundle" $defaultOut
    & (Join-Path $PSScriptRoot "Export-GitUpdate.ps1") -Since $since.Trim() -OutputPath $output
    Write-Host "DONE: $output"
} finally {
    Pop-Location
}
Pause
