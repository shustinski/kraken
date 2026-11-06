#requires -Version 5.1
<#
.SYNOPSIS
  Export or import a Kraken offline development transfer.

.DESCRIPTION
  Collects a full Git bundle and a uv cache that can recreate the locked
  environment without network access. Import fetches remote-tracking refs and
  never resets, cleans, or checks out over local work.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('export', 'import', 'verify', 'import-bundle')]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$TransferPath,

    [string]$RepoRoot = '',

    [string]$UvCommand = 'uv',

    [string]$ProjectEnvironment = '',

    [string]$FindLinks = '',

    [string]$CachePath = '',

    [ValidateSet('Bundle', 'Validate')]
    [string]$ToolMode = 'Bundle',

    [string]$RepresentativeTests = ''
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$OutputEncoding = [Console]::OutputEncoding

Import-Module (Join-Path $PSScriptRoot 'KrakenOffline.psm1') -Force
$exitCode = Invoke-KrakenOffline -Mode $Mode -TransferPath $TransferPath -RepoRoot $RepoRoot -UvCommand $UvCommand -ProjectEnvironment $ProjectEnvironment -FindLinks $FindLinks -CachePath $CachePath -ToolMode $ToolMode -RepresentativeTests $RepresentativeTests -ScriptRoot $PSScriptRoot
exit $exitCode
