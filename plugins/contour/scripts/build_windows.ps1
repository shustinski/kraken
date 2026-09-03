#Requires -Version 5.1
<#
.SYNOPSIS
    Build the Contour Windows application bundle (PyInstaller) and installer (Inno Setup).

.EXAMPLE
    .\scripts\build_windows.ps1

.EXAMPLE
    .\scripts\build_windows.ps1 -SkipTests

.EXAMPLE
    .\scripts\build_windows.ps1 -Version 0.9.6 -Clean

.EXAMPLE
    .\scripts\build_windows.ps1 -PyInstallerOnly
    .\scripts\build_windows.ps1 -InstallerOnly -Version 0.9.5
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipInstaller,
    [switch]$PyInstallerOnly,
    [switch]$InstallerOnly,
    [switch]$Clean,
    [switch]$SkipSync,
    [string]$Version = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-ContourVersion {
    param([string]$PluginRoot)

    $versionFile = Join-Path $PluginRoot "src\contour\__version__.py"
    if (-not (Test-Path $versionFile)) {
        throw "Version file not found: $versionFile"
    }

    $content = Get-Content -Path $versionFile -Raw -Encoding UTF8
    if ($content -match '__version__\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }

    throw "Could not parse __version__ from $versionFile"
}

function Resolve-Uv {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $uv) {
        return $uv.Source
    }

    throw @"
uv was not found on PATH.
Install uv and sync the workspace from the repo root:
  uv sync --project plugins/contour --extra build --extra dev
"@
}

function Invoke-Uv {
    param(
        [string]$Uv,
        [string[]]$Arguments
    )

    Write-Host "uv $($Arguments -join ' ')"
    & $Uv @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv failed with exit code ${LASTEXITCODE}: uv $($Arguments -join ' ')"
    }
}

function Invoke-ContourUvRun {
    param(
        [string]$Uv,
        [string]$ProjectPath,
        [string[]]$Command
    )

    $arguments = @(
        "run",
        "--project", $ProjectPath,
        "--no-sync",
        "--extra", "build",
        "--"
    ) + $Command
    Invoke-Uv -Uv $Uv -Arguments $arguments
}

function Repair-BrokenNumpyDistInfo {
    param(
        [string]$Uv,
        [string]$RepoRoot,
        [string]$ProjectPath
    )

    $python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        return
    }

    # PyInstaller's hook-numpy.py calls importlib.metadata.version("numpy").
    # uv can leave a stale numpy-*.dist-info directory (METADATA/RECORD missing),
    # which makes version() return None and PyInstaller fail during analysis.
    $checkScript = @'
import importlib.metadata as m
import shutil
import sys
from pathlib import Path

try:
    numpy_version = m.version("numpy")
except m.PackageNotFoundError:
    numpy_version = None

if numpy_version is not None:
    sys.exit(0)

site = Path(sys.prefix) / "Lib" / "site-packages"
removed = []
for dist_info in site.glob("numpy-*.dist-info"):
    if not (dist_info / "METADATA").is_file():
        shutil.rmtree(dist_info)
        removed.append(dist_info.name)

if removed:
    print("Removed broken numpy dist-info: " + ", ".join(removed))
sys.exit(2 if removed else 1)
'@

    $checkScript | & $python - | Out-Host
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "Repairing numpy package metadata for PyInstaller..."
    Invoke-Uv -Uv $Uv -Arguments @(
        "sync",
        "--project", $ProjectPath,
        "--extra", "build",
        "--extra", "dev",
        "--frozen",
        "--reinstall-package", "numpy",
        "--link-mode", "copy"
    )
}

function Resolve-InnoSetupCompiler {
    $iscc = Get-Command iscc -ErrorAction SilentlyContinue
    if ($null -ne $iscc) {
        return $iscc.Source
    }

    $searchRoots = @(
        ${env:ProgramFiles},
        ${env:ProgramFiles(x86)}
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique

    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($root in $searchRoots) {
        $matches = Get-ChildItem -Path $root -Filter "ISCC.exe" -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\Inno Setup \d+\\ISCC\.exe$" } |
            Sort-Object { [int]($_.FullName -replace '.*Inno Setup (\d+).*', '$1') } -Descending |
            Select-Object -ExpandProperty FullName
        foreach ($match in $matches) {
            if (-not $candidates.Contains($match)) {
                [void]$candidates.Add($match)
            }
        }
    }

    if ($candidates.Count -gt 0) {
        return $candidates[0]
    }

    return $null
}

if ($PyInstallerOnly -and $InstallerOnly) {
    throw "Use only one of -PyInstallerOnly or -InstallerOnly."
}

$PluginRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $PluginRoot "..\..")).Path
$ProjectPath = Join-Path $RepoRoot "plugins\contour"
$Uv = Resolve-Uv
$SpecFile = Join-Path $PluginRoot "packaging\Contour.spec"
$IssFile = Join-Path $PluginRoot "packaging\Contour.iss"
$DistDir = Join-Path $PluginRoot "dist\Contour"
$BuildDir = Join-Path $PluginRoot "build\Contour"
$InstallerDir = Join-Path $PluginRoot "dist\installer"

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = Get-ContourVersion -PluginRoot $PluginRoot
}

Write-Host "Contour Windows build"
Write-Host "  repo root   : $RepoRoot"
Write-Host "  plugin root : $PluginRoot"
Write-Host "  uv          : $Uv"
Write-Host "  version     : $Version"

Push-Location $RepoRoot
try {
    if (-not $SkipSync) {
        Write-Step "Syncing uv environment (plugins/contour + build/dev extras)"
        Invoke-Uv -Uv $Uv -Arguments @(
            "sync",
            "--project", $ProjectPath,
            "--extra", "build",
            "--extra", "dev",
            "--link-mode", "copy"
        )
    }
    else {
        Write-Host "Skipping uv sync (-SkipSync)."
    }

    Push-Location $PluginRoot
    try {
        if ($Clean -and -not $InstallerOnly) {
            Write-Step "Cleaning previous PyInstaller output"
            if (Test-Path $DistDir) {
                Remove-Item -Path $DistDir -Recurse -Force
            }
            if (Test-Path $BuildDir) {
                Remove-Item -Path $BuildDir -Recurse -Force
            }
        }

        if (-not $InstallerOnly) {
            if (-not $SkipTests) {
                Write-Step "Running full test suite"
                Invoke-ContourUvRun -Uv $Uv -ProjectPath $ProjectPath -Command @(
                    "python", "-m", "pytest", "-m", "full"
                )
            }
            else {
                Write-Host "Skipping tests (-SkipTests)."
            }

            Write-Step "Building application bundle with PyInstaller"
            Repair-BrokenNumpyDistInfo -Uv $Uv -RepoRoot $RepoRoot -ProjectPath $ProjectPath
            Invoke-ContourUvRun -Uv $Uv -ProjectPath $ProjectPath -Command @(
                "python", "-m", "PyInstaller", "--noconfirm", "--clean", $SpecFile
            )

            $exePath = Join-Path $DistDir "Contour.exe"
            if (-not (Test-Path $exePath)) {
                throw "PyInstaller finished but executable was not found: $exePath"
            }

            Write-Host "PyInstaller output: $DistDir"
        }
        else {
            Write-Host "Skipping PyInstaller (-InstallerOnly)."
            $exePath = Join-Path $DistDir "Contour.exe"
            if (-not (Test-Path $exePath)) {
                throw "Installer build requires an existing bundle at '$DistDir'. Run PyInstaller first."
            }
        }

        if ($PyInstallerOnly -or $SkipInstaller) {
            if ($PyInstallerOnly) {
                Write-Host "Skipping Inno Setup (-PyInstallerOnly)."
            }
            else {
                Write-Host "Skipping Inno Setup (-SkipInstaller)."
            }
            return
        }

        Write-Step "Building installer with Inno Setup"
        $iscc = Resolve-InnoSetupCompiler
        if ($null -eq $iscc) {
            throw @"
Inno Setup compiler (ISCC.exe) was not found.
Install Inno Setup and either add ISCC to PATH or install under:
  $env:ProgramFiles\Inno Setup 7
"@
        }

        Write-Host "Using ISCC: $iscc"
        New-Item -ItemType Directory -Path $InstallerDir -Force | Out-Null

        $isccArgs = @(
            "/DMyAppVersion=$Version",
            $IssFile
        )

        Write-Host "Running: $iscc $($isccArgs -join ' ')"
        & $iscc @isccArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE"
        }

        $installerPattern = Join-Path $InstallerDir "Contour-setup-$Version*.exe"
        $installer = Get-ChildItem -Path $installerPattern -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($null -eq $installer) {
            throw "Inno Setup finished but installer was not found in '$InstallerDir'."
        }

        Write-Host ""
        Write-Host "Build complete." -ForegroundColor Green
        Write-Host "  app bundle : $DistDir"
        Write-Host "  installer  : $($installer.FullName)"
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
