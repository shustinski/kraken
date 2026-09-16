[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$BundlePath,

    [string]$RemoteName = "offline-bundle"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-External {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')" }
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repositoryRoot
try {
    if (@(git status --porcelain --untracked-files=all).Count -gt 0) {
        throw "The worktree is not clean. Commit, stash, or remove changes before importing an update bundle."
    }
    $bundle = (Resolve-Path -LiteralPath $BundlePath).Path
    $sidecar = "$bundle.json"
    if (-not (Test-Path -LiteralPath $sidecar)) { throw "Update manifest is missing: $sidecar" }
    $metadata = Get-Content -LiteralPath $sidecar -Raw | ConvertFrom-Json
    $actualHash = (Get-FileHash -LiteralPath $bundle -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $metadata.sha256) { throw "SHA-256 mismatch for $bundle" }
    Invoke-External git @("bundle", "verify", $bundle)

    $safeRemoteName = $RemoteName -replace "[^A-Za-z0-9._/-]", "-"
    if ([string]::IsNullOrWhiteSpace($safeRemoteName)) { throw "RemoteName contains no usable characters." }
    Invoke-External git @(
        "fetch", $bundle,
        "+refs/heads/*:refs/remotes/$safeRemoteName/*",
        "+refs/tags/*:refs/tags/*"
    )
    Write-Host "Imported refs are available as refs/remotes/$safeRemoteName/*. Merge or rebase them explicitly."
} finally {
    Pop-Location
}
