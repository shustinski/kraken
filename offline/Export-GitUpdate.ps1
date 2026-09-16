[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Since,

    [Parameter(Mandatory)]
    [string]$OutputPath
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
        throw "The worktree is not clean. Commit changes before exporting an update bundle."
    }
    Invoke-External git @("rev-parse", "--verify", "$Since^{commit}")
    $output = [System.IO.Path]::GetFullPath($OutputPath)
    if (Test-Path -LiteralPath $output) { throw "Output already exists: $output" }
    $directory = Split-Path $output -Parent
    if (-not (Test-Path -LiteralPath $directory)) { New-Item -ItemType Directory -Path $directory | Out-Null }
    $bundleRefs = @(git for-each-ref --format="%(refname)" refs/heads refs/tags)
    if ($bundleRefs.Count -eq 0) { throw "No local branches or tags are available to bundle." }

    # ^Since is recorded as a required prerequisite, so import fails safely on the wrong base.
    Invoke-External git (@("bundle", "create", $output) + $bundleRefs + "^$Since")
    $sha = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
    [ordered]@{
        formatVersion = 1
        baseCommit = (git rev-parse "$Since^{commit}").Trim()
        sourceCommit = (git rev-parse HEAD).Trim()
        sha256 = $sha
        createdUtc = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath "$output.json" -Encoding utf8NoBOM
    Write-Host "Update bundle created: $output"
} finally {
    Pop-Location
}
