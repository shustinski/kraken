#requires -Version 5.1
<#
.SYNOPSIS
  Copy Kraken to another Windows PC without internet, and bring commits back.

.EXAMPLE
  .\scripts\offline\sync.ps1 -Mode export -TransferPath E:\kraken-transfer

.EXAMPLE
  .\scripts\offline\sync.ps1 -Mode import -TransferPath E:\kraken-transfer -RepoRoot C:\code\kraken
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('export', 'import')]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$TransferPath,

    [string]$RepoRoot = ''
)

if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }

$bundle = Join-Path $TransferPath 'repo.bundle'
$cache = Join-Path $TransferPath 'uv-cache'

function Invoke-Uv {
    param([string[]]$UvArgs, [string]$Venv, [switch]$Offline)
    $savedCache = $env:UV_CACHE_DIR
    $savedVenv = $env:UV_PROJECT_ENVIRONMENT
    $savedOffline = $env:UV_OFFLINE
    try {
        $env:UV_CACHE_DIR = $cache
        $env:UV_PROJECT_ENVIRONMENT = $Venv
        if ($Offline) { $env:UV_OFFLINE = '1' } else { Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue }
        & uv @UvArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        if ($null -eq $savedCache) { Remove-Item Env:UV_CACHE_DIR -ErrorAction SilentlyContinue } else { $env:UV_CACHE_DIR = $savedCache }
        if ($null -eq $savedVenv) { Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue } else { $env:UV_PROJECT_ENVIRONMENT = $savedVenv }
        if ($null -eq $savedOffline) { Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue } else { $env:UV_OFFLINE = $savedOffline }
    }
}

$syncArgs = @(
    'sync', '--directory', $RepoRoot,
    '--all-packages', '--all-extras', '--all-groups',
    '--frozen', '--link-mode', 'copy'
)

if ($Mode -eq 'export') {
    New-Item -ItemType Directory -Force -Path $TransferPath | Out-Null
    if (Test-Path -LiteralPath $bundle) { Remove-Item -LiteralPath $bundle -Force }
    & git -C $RepoRoot bundle create $bundle HEAD --branches --tags
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (Test-Path -LiteralPath (Join-Path $RepoRoot 'pyproject.toml')) {
        $exportVenv = Join-Path ([System.IO.Path]::GetTempPath()) 'kraken-offline-export-venv'
        Invoke-Uv -UvArgs $syncArgs -Venv $exportVenv
        Remove-Item -LiteralPath $exportVenv -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Exported to $TransferPath"
    Write-Host 'On the other PC: import, then git merge kraken-offline/<branch>'
    return
}

if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot '.git'))) {
    & git clone $bundle $RepoRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    & git -C $RepoRoot fetch --no-tags $bundle '+refs/heads/*:refs/remotes/kraken-offline/*'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host 'Local commits were left in place.'
    Write-Host 'Integrate with: git merge kraken-offline/<branch>'
}

if (Test-Path -LiteralPath (Join-Path $RepoRoot 'pyproject.toml')) {
    Invoke-Uv -UvArgs ($syncArgs + @('--offline')) -Venv (Join-Path $RepoRoot '.venv') -Offline
}
Write-Host "Imported into $RepoRoot"
