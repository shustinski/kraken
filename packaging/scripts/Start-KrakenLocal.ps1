param(
    [string]$Config = (Join-Path $env:LOCALAPPDATA "Kraken\LocalServer\server.toml")
)

$ErrorActionPreference = "Stop"
$InstallRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ServerExecutable = Join-Path $InstallRoot "KrakenServer.exe"

if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Configuration not found: $Config. Run Initialize-KrakenLocal.ps1 first."
}

Write-Host "Kraken Server runs in this window. Press Ctrl+C to stop it."
& $ServerExecutable --config $Config
exit $LASTEXITCODE
