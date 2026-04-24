param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not $SkipTests) {
        python -m pytest
    }
    pyinstaller .\packaging\Contour.spec
    if (Get-Command iscc -ErrorAction SilentlyContinue) {
        iscc .\packaging\Contour.iss
    }
} finally {
    Pop-Location
}
