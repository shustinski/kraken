# Kraken beta programme — W3 Task 6 helper: drive the S01 GUI scenario.
#
# Runs the given Python snippet (which drives the account-creation and login
# windows via pytest-qt's qtbot) inside an offscreen Qt session so the
# scenario works on a console session with no interactive desktop. The
# snippet must end with an assertion; a non-zero exit means the GUI did not
# behave as the registry promises.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s01_palette.ps1 `
#     -SnippetPath <file> [-TimeoutSec 180]
#
# This helper is a diagnostic aid for the scenario author; the committed
# scenario itself lives in tests/quality/test_s01_local_identity.py and runs
# under the gui tier.
param(
    [Parameter(Mandatory = $true)][string]$SnippetPath,
    [int]$TimeoutSec = 180
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $SnippetPath)) {
    throw "Snippet file not found: $SnippetPath"
}
$pythonExe = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python interpreter not found: $pythonExe"
}

$previousQpa = $env:QT_QPA_PLATFORM
try {
    $env:QT_QPA_PLATFORM = 'offscreen'
    & $pythonExe $SnippetPath
    $exitCode = $LASTEXITCODE
} finally {
    if ($null -eq $previousQpa) {
        Remove-Item Env:\QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    } else {
        $env:QT_QPA_PLATFORM = $previousQpa
    }
}
if ($exitCode -ne 0) {
    throw "S01 palette snippet failed with exit code $exitCode"
}
Write-Output 'S01 palette snippet OK'
