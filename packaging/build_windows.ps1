param(
    [switch]$SkipInstallers
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = Join-Path $RepositoryRoot "dist\windows"

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
