param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 1000000)]
    [int]$Width,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 1000000)]
    [int]$Height,
    [string]$Server = "http://127.0.0.1:8080",
    [string]$Username = "admin",
    [switch]$NonInteractive,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$InstallRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AdminExecutable = Join-Path $InstallRoot "KrakenAdmin.exe"
$Arguments = @(
    "project-create",
    "--server", $Server,
    "--username", $Username,
    "--name", $Name,
    "--width", $Width,
    "--height", $Height
)
if ($NonInteractive) {
    $Arguments += "--non-interactive"
}
if ($Json) {
    $Arguments += "--json"
}

& $AdminExecutable @Arguments
exit $LASTEXITCODE
