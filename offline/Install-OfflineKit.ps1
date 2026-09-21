[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$KitPath,

    [Parameter(Mandatory)]
    [string]$DestinationPath,

    [string]$Python,

    [string]$Uv,

    # Off by default. Use when the kit was built with Create-OfflineKit -IncludeHashes.
    [switch]$VerifyHashes
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-External {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')" }
}

function Get-RelativeKitPath {
    param([string]$Root, [string]$FullName)
    return $FullName.Substring($Root.Length).TrimStart([char]'\', [char]'/').Replace('\', '/')
}

function Get-TreeListingDigest {
    param([string]$Root, [string]$RelativeTreePath)

    $treeFullPath = Join-Path $Root ($RelativeTreePath.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $treeFullPath)) {
        throw "Expected kit tree is missing: $RelativeTreePath"
    }

    $lines = New-Object System.Collections.Generic.List[string]
    $fileCount = 0
    $totalBytes = [int64]0
    Get-ChildItem -LiteralPath $treeFullPath -File -Recurse |
        Sort-Object FullName |
        ForEach-Object {
            $rel = Get-RelativeKitPath -Root $Root -FullName $_.FullName
            $lines.Add("$rel`t$($_.Length)")
            $fileCount++
            $totalBytes += $_.Length
        }

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))
        $hash = ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join ""
    } finally {
        $sha.Dispose()
    }

    [ordered]@{
        fileCount = $fileCount
        totalBytes = $totalBytes
        listingSha256 = $hash
    }
}

$kit = (Resolve-Path -LiteralPath $KitPath).Path
$manifestPath = Join-Path $kit "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "manifest.json was not found in: $kit" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.platform -ne "windows-x64") { throw "This kit is for $($manifest.platform), not Windows x64." }

if ($VerifyHashes) {
    Write-Host "Verifying hashed kit files..."
    $fileIndex = 0
    $fileEntries = @($manifest.files)
    foreach ($entry in $fileEntries) {
        $fileIndex++
        if (($fileIndex % 25) -eq 1) {
            Write-Host "Verifying files: $fileIndex / $($fileEntries.Count) ..."
        }
        $filePath = Join-Path $kit $entry.path.Replace("/", "\\")
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) { throw "Kit file is missing: $($entry.path)" }
        $actual = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $entry.sha256) { throw "SHA-256 mismatch: $($entry.path)" }
    }

    if ($null -ne $manifest.PSObject.Properties["trees"] -and $null -ne $manifest.trees) {
        foreach ($tree in @($manifest.trees)) {
            Write-Host "Verifying tree listing: $($tree.path)"
            $actual = Get-TreeListingDigest -Root $kit -RelativeTreePath $tree.path
            if ($actual.fileCount -ne [int]$tree.fileCount -or $actual.totalBytes -ne [int64]$tree.totalBytes) {
                throw "Tree size mismatch: $($tree.path)"
            }
            if ($actual.listingSha256 -ne $tree.listingSha256) {
                throw "Tree listing SHA-256 mismatch: $($tree.path)"
            }
        }
    }
} else {
    Write-Host "Skipping hash verification (default). Pass -VerifyHashes to enable."
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
