param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not $SkipTests) {
        python -m pytest
    }
    pyinstaller .\packaging\krona.spec
} finally {
    Pop-Location
}
