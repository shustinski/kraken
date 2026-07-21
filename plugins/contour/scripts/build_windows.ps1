param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$PluginRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $PluginRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Push-Location $PluginRoot
try {
    if (-not $SkipTests) {
        & $Python -m pytest
    }
    & $Python -m PyInstaller .\packaging\Contour.spec
    if (Get-Command iscc -ErrorAction SilentlyContinue) {
        iscc .\packaging\Contour.iss
    }
} finally {
    Pop-Location
}
