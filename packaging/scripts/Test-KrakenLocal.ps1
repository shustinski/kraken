param(
    [string]$Config = (Join-Path $env:LOCALAPPDATA "Kraken\LocalServer\server.toml"),
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$InstallRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AdminExecutable = Join-Path $InstallRoot "KrakenAdmin.exe"
$Arguments = @("doctor", "--config", $Config)
if ($Json) {
    $Arguments += "--json"
}

& $AdminExecutable @Arguments
exit $LASTEXITCODE
