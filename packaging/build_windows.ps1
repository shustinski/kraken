param(
    [switch]$SkipInstallers
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = Join-Path $RepositoryRoot "dist\windows"
$Cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
if ($null -eq $Cargo) {
    $CargoPath = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
    if (-not (Test-Path $CargoPath)) {
        throw "Rust toolchain is required to build KrakenBlobGateway.exe (https://rustup.rs)"
    }
} else {
    $CargoPath = $Cargo.Source
}
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    throw "Visual Studio Build Tools with the C++ workload and Windows SDK are required"
}
$VisualStudio = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VisualStudio) {
    throw "Visual Studio C++ Build Tools are required"
}
$VcVars = Join-Path $VisualStudio "VC\Auxiliary\Build\vcvars64.bat"
$CargoCommand = 'call "' + $VcVars + '" >nul && "' + $CargoPath + '" build --locked --release --manifest-path "' + (Join-Path $RepositoryRoot "blob_gateway\Cargo.toml") + '"'
& cmd.exe /d /s /c $CargoCommand
if ($LASTEXITCODE -ne 0) {
    throw "Kraken Blob Gateway build failed with exit code $LASTEXITCODE"
}

Push-Location $PSScriptRoot
try {
    python -m PyInstaller --noconfirm --clean --distpath $OutputRoot --workpath (Join-Path $RepositoryRoot "build\pyinstaller-server") KrakenServer.spec
    python -m PyInstaller --noconfirm --clean --distpath $OutputRoot --workpath (Join-Path $RepositoryRoot "build\pyinstaller-desktop") KrakenDesktop.spec
} finally {
    Pop-Location
}

if (-not $SkipInstallers) {
    $Compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($null -eq $Compiler) {
        throw "Inno Setup 6 (ISCC.exe) is required to create Windows installers"
    }
    & $Compiler.Source (Join-Path $PSScriptRoot "installer\KrakenServer.iss")
    & $Compiler.Source (Join-Path $PSScriptRoot "installer\KrakenDesktop.iss")
}
