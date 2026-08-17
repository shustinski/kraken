param(
    [string]$Config = "$env:LOCALAPPDATA\Kraken\LocalServer\server.toml",
    [ValidateRange(1, 256)]
    [int]$Clients = 30,
    [ValidateRange(1, 1048576)]
    [int]$SizeMiB = 64,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$InstallRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AdminExecutable = Join-Path $InstallRoot "KrakenAdmin.exe"
$Arguments = @(
    "blob-benchmark",
    "--config", $Config,
    "--clients", $Clients,
    "--size-mib", $SizeMiB
)
if ($Json) {
    $Arguments += "--json"
}

& $AdminExecutable @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Kraken Blob Gateway benchmark failed with exit code $LASTEXITCODE"
}
