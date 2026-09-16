[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$KitPath,

    [Parameter(Mandatory)]
    [string]$DestinationPath,

    [string]$Python,

    [string]$Uv
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-External {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')" }
}

$kit = (Resolve-Path -LiteralPath $KitPath).Path
$manifestPath = Join-Path $kit "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "manifest.json was not found in: $kit" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.platform -ne "windows-x64") { throw "This kit is for $($manifest.platform), not Windows x64." }

foreach ($entry in $manifest.files) {
    $filePath = Join-Path $kit $entry.path.Replace("/", "\\")
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) { throw "Kit file is missing: $($entry.path)" }
    $actual = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.sha256) { throw "SHA-256 mismatch: $($entry.path)" }
}

$destination = [System.IO.Path]::GetFullPath($DestinationPath)
if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
$parent = Split-Path $destination -Parent
if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }

$bundle = Join-Path $kit "source\\kraken-full.bundle"
Invoke-External git @("clone", "--branch", $manifest.repository.defaultBranch, $bundle, $destination)
Push-Location $destination
try {
    Invoke-External git @("bundle", "verify", $bundle)
    Invoke-External git @("fsck", "--full", "--no-reflogs")
    $actualCommit = (git rev-parse HEAD).Trim()
    if ($actualCommit -ne $manifest.repository.commit) {
        throw "Restored commit $actualCommit does not match kit commit $($manifest.repository.commit)."
    }
    foreach ($ref in $manifest.repository.refs) {
        if (-not $ref.StartsWith("refs/heads/")) { continue }
        $branch = $ref.Substring("refs/heads/".Length)
        & git show-ref --verify --quiet "refs/heads/$branch"
        if ($LASTEXITCODE -ne 0) {
            Invoke-External git @("branch", "--track", $branch, "origin/$branch")
        }
    }

    $resolvedPython = if ($Python) { $Python } else { (Get-Command python -ErrorAction Stop).Source }
    $resolvedUv = if ($Uv) { $Uv } else { (Get-Command uv -ErrorAction Stop).Source }
    $env:UV_CACHE_DIR = Join-Path $kit "dependencies\\uv-cache"
    $env:UV_OFFLINE = "1"
    $env:UV_PYTHON_DOWNLOADS = "never"
    Invoke-External $resolvedUv @(
        "sync", "--offline", "--frozen", "--all-packages", "--all-extras", "--all-groups", "--link-mode=copy",
        "--find-links", (Join-Path $kit "dependencies\\wheelhouse"),
        "--python", $resolvedPython
    )
} catch {
    Pop-Location
    Remove-Item -LiteralPath $destination -Recurse -Force -ErrorAction SilentlyContinue
    throw
} finally {
    if ((Get-Location).Path -eq $destination) { Pop-Location }
    Remove-Item Env:UV_CACHE_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue
    Remove-Item Env:UV_PYTHON_DOWNLOADS -ErrorAction SilentlyContinue
}

Write-Host "Offline repository installed: $destination"
