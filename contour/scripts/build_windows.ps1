<#
.SYNOPSIS
    Reproducible Windows build for the Contour installer.

.DESCRIPTION
    Runs lint + type check + tests, produces the PyInstaller bundle in
    dist/Contour and compiles the Inno Setup installer
    packaging/Contour-setup-<version>.exe.

.PARAMETER Version
    Version string to embed in the installer. Defaults to the value exposed
    by `polygon_widget.__version__`.

.PARAMETER SkipTests
    Skip the pytest phase (useful for quick local iteration).

.PARAMETER SkipLint
    Skip ruff + mypy (useful for quick local iteration).

.EXAMPLE
    .\scripts\build_windows.ps1

.EXAMPLE
    .\scripts\build_windows.ps1 -Version 0.4.1 -SkipTests
#>
[CmdletBinding()]
param(
    [string]$Version,
    [switch]$SkipTests,
    [switch]$SkipLint
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

if (-not $Version) {
    $Version = & $python -c "from polygon_widget.__version__ import __version__; print(__version__)"
    if (-not $Version) { throw "Failed to determine the package version." }
}

Write-Host ""
Write-Host "=== Building Contour $Version ===" -ForegroundColor Cyan
Write-Host "Repository : $repoRoot"
Write-Host "Python     : $(& $python --version)"
Write-Host ""

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ">>> $Name" -ForegroundColor Yellow
    & $Body
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
    Write-Host ""
}

if (-not $SkipLint) {
    Invoke-Step "ruff check"         { & $python -m ruff check polygon_widget tests examples }
    Invoke-Step "ruff format --check" { & $python -m ruff format --check polygon_widget tests examples }
    Invoke-Step "mypy"               { & $python -m mypy polygon_widget }
} else {
    Write-Host "Skipping lint/type checks." -ForegroundColor DarkYellow
}

if (-not $SkipTests) {
    $env:QT_QPA_PLATFORM = "offscreen"
    Invoke-Step "pytest" { & $python -m pytest -q }
} else {
    Write-Host "Skipping tests." -ForegroundColor DarkYellow
}

Invoke-Step "clean build/dist" {
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
}

Invoke-Step "pyinstaller" {
    & $python -m PyInstaller packaging/ViaLaNetPolygonWidget.spec --clean --noconfirm
}

$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $candidate) { $iscc = @{ Source = $candidate } }
}
if (-not $iscc) {
    throw "Inno Setup compiler (iscc.exe) not found on PATH or in the default install location."
}

Invoke-Step "iscc (installer)" {
    & $iscc.Source "/DMyAppVersion=$Version" packaging/ViaLaNetPolygonWidget.iss
}

$installer = Join-Path $repoRoot "packaging/Contour-setup-$Version.exe"
if (Test-Path $installer) {
    Write-Host "Installer ready: $installer" -ForegroundColor Green
} else {
    throw "Expected installer not produced at $installer."
}
