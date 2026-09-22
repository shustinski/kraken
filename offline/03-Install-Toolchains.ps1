# 03 — установить все toolchain из kit (офлайн-ПК, админ).
. "$PSScriptRoot\_lib.ps1"
Assert-Admin

$defaultKit = Resolve-FirstExisting @(
    (Join-Path $PSScriptRoot ".."),
    $PSScriptRoot
)
# Prefer kit root that contains toolchains\
$kitGuess = $null
foreach ($c in @(
    (Get-Location).Path,
    $PSScriptRoot,
    (Join-Path $PSScriptRoot "..")
)) {
    if (Test-Path (Join-Path $c "toolchains")) { $kitGuess = $c; break }
}
if ($null -eq $kitGuess) { $kitGuess = (Get-Location).Path }

$kitPath = Read-PathOrDefault "Корень kit (там есть папка toolchains)" $kitGuess
$tc = Join-Path $kitPath "toolchains"
if (-not (Test-Path -LiteralPath $tc)) { throw "Не найдено: $tc" }

function Get-One {
    param([string]$Pattern)
    $item = Get-ChildItem -LiteralPath $tc -File -Filter $Pattern -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $item) { throw "В toolchains нет файла: $Pattern" }
    return $item.FullName
}

Write-Host "=== 1/7 Python ==="
$python = Get-One "python-*-amd64.exe"
Start-Process -FilePath $python -ArgumentList "/passive","InstallAllUsers=1","PrependPath=1","Include_test=0" -Wait

Write-Host "=== 2/7 uv ==="
$uvZip = Get-ChildItem -LiteralPath $tc -File -Filter "uv-*.zip" -EA SilentlyContinue | Select-Object -First 1
$uvExe = Get-ChildItem -LiteralPath $tc -File -Filter "uv.exe" -EA SilentlyContinue | Select-Object -First 1
$uvBin = Join-Path $env:USERPROFILE ".local\bin"
New-Item -ItemType Directory -Force -Path $uvBin | Out-Null
if ($null -ne $uvZip) {
    Expand-Archive -LiteralPath $uvZip.FullName -DestinationPath $uvBin -Force
} elseif ($null -ne $uvExe) {
    Copy-Item -LiteralPath $uvExe.FullName -Destination (Join-Path $uvBin "uv.exe") -Force
} else {
    throw "Нет uv.zip / uv.exe в toolchains"
}
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$uvBin*") {
    [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(';') + ";" + $uvBin), "User")
}
$env:Path += ";$uvBin"

Write-Host "=== 3/7 Git ==="
$git = Get-One "Git-*-64-bit.exe"
Start-Process -FilePath $git -ArgumentList "/VERYSILENT","/NORESTART","/NOCANCEL","/SP-","/CLOSEAPPLICATIONS","/RESTARTAPPLICATIONS","/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait

Write-Host "=== 4/7 Git LFS ==="
$lfs = Get-ChildItem -LiteralPath $tc -File -Filter "git-lfs-windows-*.exe" -EA SilentlyContinue | Select-Object -First 1
if ($null -eq $lfs) { throw "Нет git-lfs-windows-*.exe" }
Start-Process -FilePath $lfs.FullName -ArgumentList "/VERYSILENT","/NORESTART" -Wait
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
& git lfs install

Write-Host "=== 5/7 VS Build Tools (долго) ==="
$layout = Join-Path $tc "vs-build-tools-layout"
if (-not (Test-Path -LiteralPath $layout)) { throw "Нет $layout" }
$certs = Join-Path $layout "certificates"
if (Test-Path -LiteralPath $certs) {
    Get-ChildItem -LiteralPath $certs -File | Where-Object { $_.Extension -in ".cer",".crt" } | ForEach-Object {
        & certutil -addstore -f Root $_.FullName | Out-Null
    }
}
$vsExe = Join-Path $layout "vs_BuildTools.exe"
if (-not (Test-Path -LiteralPath $vsExe)) { $vsExe = Join-Path $layout "vs_setup.exe" }
$vsArgs = @(
    "--noWeb","--noUpdateInstaller","--wait",
    "--add","Microsoft.VisualStudio.Workload.VCTools",
    "--add","Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    "--add","Microsoft.VisualStudio.Component.Windows10SDK.19041",
    "--includeRecommended"
)
$vs = Start-Process -FilePath $vsExe -ArgumentList $vsArgs -Wait -PassThru
if ($null -ne $vs.ExitCode -and $vs.ExitCode -ne 0) {
    throw "VS Build Tools завершились с кодом $($vs.ExitCode). Лог: $env:TEMP\dd_bootstrapper_*.log"
}

Write-Host "=== 6/7 Rust ==="
$rustup = Join-Path $tc "rustup-init.exe"
if (-not (Test-Path -LiteralPath $rustup)) { throw "Нет rustup-init.exe" }
Start-Process -FilePath $rustup -ArgumentList "-y","--default-toolchain","stable","--default-host","x86_64-pc-windows-msvc" -Wait

Write-Host "=== 7/7 Inno Setup ==="
$inno = Get-ChildItem -LiteralPath $tc -File -Filter "innosetup-*.exe" -EA SilentlyContinue | Select-Object -First 1
if ($null -eq $inno) { throw "Нет innosetup-*.exe" }
Start-Process -FilePath $inno.FullName -ArgumentList "/VERYSILENT","/NORESTART","/SUPPRESSMSGBOXES" -Wait

Write-Host ""
Write-Host "DONE. Закройте терминал и откройте новый, затем запустите 04-Install-Repo.ps1"
Pause
