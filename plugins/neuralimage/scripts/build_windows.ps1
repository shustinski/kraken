param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not $SkipTests) {
        python -m pytest
    }
    pyinstaller .\packaging\NeuralImage.spec
    if (Get-Command iscc -ErrorAction SilentlyContinue) {
        iscc .\packaging\NeuralImageInstaller.iss
    }
} finally {
    Pop-Location
}
