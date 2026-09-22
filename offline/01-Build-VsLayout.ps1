# 01 — скачать VS Build Tools layout для Windows 10 21H2 (нужен интернет, админ).
# Не закрывайте окно Visual Studio Installer до сообщения DONE.
. "$PSScriptRoot\_lib.ps1"
Assert-Admin

$bootstrapper = Resolve-FirstExisting @(
    "$env:USERPROFILE\Downloads\vs_BuildTools.exe",
    "$env:TEMP\vs_BuildTools.exe"
)
if ($null -eq $bootstrapper) {
    Write-Host "Скачиваю vs_BuildTools.exe ..."
    $bootstrapper = Join-Path $env:TEMP "vs_BuildTools.exe"
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile $bootstrapper
}

$defaultLayout = Join-Path $PSScriptRoot "vs-build-tools-layout-win10-21h2"
$layout = Read-PathOrDefault "Куда сохранить layout" $defaultLayout
New-Item -ItemType Directory -Force -Path $layout | Out-Null

Write-Host ""
Write-Host "Layout: $layout"
Write-Host "Bootstrapper: $bootstrapper"
Write-Host "Это займёт много времени и трафика. Окно Installer НЕ закрывать."
Write-Host ""

$argumentList = @(
    "--layout", $layout,
    "--lang", "en-US",
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041",
    "--includeRecommended",
    "--wait"
)
$process = Start-Process -FilePath $bootstrapper -ArgumentList $argumentList -Wait -PassThru
$code = $process.ExitCode
Write-Host "Exit code: $code"
if ($null -eq $code -or $code -ne 0) {
    $latestLog = Get-ChildItem "$env:TEMP\dd_bootstrapper_*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -ne $latestLog) {
        Write-Host "Лог: $($latestLog.FullName)"
        Get-Content -LiteralPath $latestLog.FullName -Tail 30
    }
    throw "Layout не собрался (код $code). Если в логе User Cancelled Download — вы закрыли окно."
}

# PCA 2024 cert for offline install
$certs = Join-Path $layout "certificates"
New-Item -ItemType Directory -Force -Path $certs | Out-Null
$certPath = Join-Path $certs "Microsoft Windows Code Signing PCA 2024.crt"
if (-not (Test-Path -LiteralPath $certPath)) {
    Write-Host "Скачиваю сертификат PCA 2024 ..."
    Invoke-WebRequest -Uri "https://www.microsoft.com/pkiops/certs/Microsoft%20Windows%20Code%20Signing%20PCA%202024.crt" -OutFile $certPath
}

$fileCount = @(Get-ChildItem -LiteralPath $layout -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host "DONE: $layout ($fileCount files)"
if ($fileCount -lt 50) { throw "Layout почти пустой ($fileCount файлов)." }

Write-Host ""
Write-Host "Дальше: положите эту папку в toolchains как vs-build-tools-layout и запустите 02-Create-Kit.ps1"
Pause
