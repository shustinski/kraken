param(
    [string]$DatabaseHost,
    [int]$DatabasePort,
    [string]$PostgresAdministrator,
    [string]$DatabaseName,
    [string]$DatabaseUser,
    [string]$KrakenAdministrator,
    [string]$AdministratorDisplayName,
    [int]$ServerPort = 8080,
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$InstallRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AdminExecutable = Join-Path $InstallRoot "KrakenAdmin.exe"
$LocalRoot = Join-Path $env:LOCALAPPDATA "Kraken\LocalServer"
$Arguments = @(
    "init",
    "--config", (Join-Path $LocalRoot "server.toml"),
    "--blob-root", (Join-Path $LocalRoot "blobs"),
    "--port", $ServerPort
)
$OptionalArguments = @{
    "DatabaseHost" = "--database-host"
    "DatabasePort" = "--database-port"
    "PostgresAdministrator" = "--postgres-admin"
    "DatabaseName" = "--database-name"
    "DatabaseUser" = "--database-user"
    "KrakenAdministrator" = "--username"
    "AdministratorDisplayName" = "--display-name"
}
foreach ($Name in $OptionalArguments.Keys) {
    if ($PSBoundParameters.ContainsKey($Name)) {
        $Arguments += $OptionalArguments[$Name]
        $Arguments += $PSBoundParameters[$Name]
    }
}
if ($NonInteractive) {
    $Arguments += "--non-interactive"
}

& $AdminExecutable @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Kraken initialization failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Ready. Start the server with scripts\Start-KrakenLocal.ps1"
