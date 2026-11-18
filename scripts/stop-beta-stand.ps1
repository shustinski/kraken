<#
.SYNOPSIS
    Tear down the permanent Kraken beta stand started by
    scripts/start-beta-stand.ps1.

.DESCRIPTION
    Order (see docs/quality/beta-stand.md):
      1. revoke the machine token while the Server is still up and PROVE
         it: a token-authenticated POST /api/v1/agent/lease must now
         answer HTTP 401;
      2. stop the agent worker and the Server/Gateway processes this
         stand started (PIDs from .test_runtime\beta-stand\state.json,
         each verified against its command line so a recycled PID of an
         unrelated process is never killed);
      3. wait until ports 8080/8081 are closed;
      4. drop the dedicated kraken_beta_test database (or, in the
         documented schema-fallback mode, drop the dedicated SCHEMA) -
         the working kraken_local data is never touched;
      5. remove the stand's secret/state files (logs and test results
         stay under .test_runtime\beta-stand for post-mortems).

    Idempotent: without a state file there is nothing owned to stop and
    the script exits 0. Prints PASS/FAIL per component; exits 0 only
    when everything passed.

.PARAMETER KeepDatabase
    Keep kraken_beta_test (and its schema) instead of dropping it.

.NOTES
    PowerShell 5.1; helper logic lives in tools/quality/beta_stand.py.
#>
[CmdletBinding()]
param(
    [switch]$KeepDatabase
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$python = Join-Path $repo '.venv\Scripts\python.exe'
$helper = Join-Path $repo 'tools\quality\beta_stand.py'

$standDir = Join-Path $repo '.test_runtime\beta-stand'
$statePath = Join-Path $standDir 'state.json'
$envJsonPath = Join-Path $repo '.test_runtime\beta-stand.env.json'
$serverToml = Join-Path $standDir 'server.toml'
$machineSecret = Join-Path $standDir 'machine.secret'
$baseDbSecret = Join-Path $repo 'config\local\database-url.secret'
$serverUrl = 'http://127.0.0.1:8080'

$script:failureReasons = @()
function Report {
    param([string]$Component, [bool]$Ok, [string]$Detail = '')
    if ($Ok) {
        Write-Host ("PASS {0}" -f $Component)
    }
    else {
        Write-Host ("FAIL {0}: {1}" -f $Component, $Detail)
        $script:failureReasons += ('{0}: {1}' -f $Component, $Detail)
    }
}
function Invoke-NativeCapture {
    param([string]$Exe, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Exe @Arguments 2>&1
        return [pscustomobject]@{ Exit = $LASTEXITCODE; Lines = @($output | ForEach-Object { [string]$_ }) }
    }
    finally {
        $ErrorActionPreference = $prev
    }
}
function Test-PortListening {
    param([int]$Port)
    $client = New-Object Net.Sockets.TcpClient
    try {
        $client.Connect('127.0.0.1', $Port)
        return $true
    }
    catch { return $false }
    finally { $client.Close() }
}
function Stop-StandProcess {
    <#
        Kill a PID only while its command line still matches the stand
        component (protects against recycled PIDs of unrelated processes).
        Returns $true when the process is gone (killed or already exited).
    #>
    param([int]$ProcessId, [string]$CommandLinePattern)
    if ($ProcessId -le 0) { return $true }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $true }
    $row = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    $commandLine = [string]$row.CommandLine
    if ($commandLine -notlike $CommandLinePattern) {
        Write-Host ("  pid $ProcessId command line does not match '$CommandLinePattern'; not killing")
        return $false
    }
    & taskkill /PID $ProcessId /T /F | Out-Null
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 250
    }
    return -not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

if (-not (Test-Path -LiteralPath $statePath)) {
    Write-Host 'beta-stand: no state file; nothing was started by start-beta-stand.ps1 on this machine'
    exit 0
}
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json

# ------------------------------------------- 1. revoke machine token + 401 proof
$machineTokenId = [string]$state.machine_token_id
$serverReachable = Test-PortListening -Port 8080
if ($machineTokenId -and (Test-Path -LiteralPath $machineSecret)) {
    $revokedAlready = $false
    $tokenAbsent = $false
    $list = Invoke-NativeCapture -Exe $python -Arguments @(
        '-m', 'kraken_admin', 'list-agent-tokens', '--config', $serverToml, '--json')
    if ($list.Exit -eq 0) {
        try {
            $items = (($list.Lines -join "`n") | ConvertFrom-Json).items
            $entry = @($items | Where-Object { $_.token_id -eq $machineTokenId }) | Select-Object -First 1
            $revokedAlready = [bool]($entry -and $entry.revoked)
            $tokenAbsent = [bool]($null -eq $entry)
        }
        catch { }
    }
    if ($revokedAlready) {
        Report 'revoke-machine-token' $true "token $machineTokenId already revoked"
    }
    elseif ($tokenAbsent) {
        Report 'revoke-machine-token' $true 'token row absent (the stand database was already reset)'
    }
    else {
        $revoke = Invoke-NativeCapture -Exe $python -Arguments @(
            '-m', 'kraken_admin', 'revoke-agent-token', '--config', $serverToml,
            '--token-id', $machineTokenId)
        if ($revoke.Exit -eq 0) {
            Report 'revoke-machine-token' $true "revoked $machineTokenId"
        }
        else {
            Report 'revoke-machine-token' $false ($revoke.Lines -join ' ')
        }
    }
    if ($serverReachable) {
        # The proof: the revoked token must now be rejected with HTTP 401.
        $probe = Invoke-NativeCapture -Exe $python -Arguments @(
            $helper, 'probe-machine-token', '--server-url', $serverUrl,
            '--token-secret', $machineSecret, '--expect', '401')
        if ($probe.Exit -eq 0) {
            Report 'revocation-probe' $true 'POST /api/v1/agent/lease answered 401'
        }
        else {
            Report 'revocation-probe' $false ($probe.Lines -join ' ')
        }
    }
    else {
        Write-Host '  revocation-probe skipped: the Server is not listening (revocation is still in the database)'
    }
}
else {
    Write-Host 'beta-stand: no machine token recorded; skipping revoke/probe'
}

# ---------------------------------------------------- 2. stop owned processes
# Stand-specific command-line patterns, identical to the ones start-beta-stand
# uses to identify these components (Find-StandPid): a recycled PID that now
# belongs to a WORKING kraken_agent/kraken_server must never match, so
# teardown can only ever kill processes started from this stand's directory.
$agentStopped = Stop-StandProcess -ProcessId ([int]$state.agent_pid) -CommandLinePattern '*kraken_agent*beta-stand*agent-data*'
if ($state.agent_pid) {
    Report 'stop-agent' $agentStopped "pid $($state.agent_pid)"
}
$serverStopped = Stop-StandProcess -ProcessId ([int]$state.server_pid) -CommandLinePattern '*kraken_server*beta-stand*server.toml*'
if ($state.server_pid) {
    Report 'stop-server' $serverStopped "pid $($state.server_pid)"
}
if ($state.gateway_pid) {
    $gatewayStopped = Stop-StandProcess -ProcessId ([int]$state.gateway_pid) -CommandLinePattern '*kraken-blob-gateway*'
    Report 'stop-gateway' $gatewayStopped "pid $($state.gateway_pid)"
}

# ------------------------------------------------------- 3. ports closed
$deadline = (Get-Date).AddSeconds(15)
$port8080Closed = $false
$port8081Closed = $false
while ((Get-Date) -lt $deadline) {
    $port8080Closed = -not (Test-PortListening -Port 8080)
    $port8081Closed = -not (Test-PortListening -Port 8081)
    if ($port8080Closed -and ($port8081Closed -or -not $state.gateway_enabled)) { break }
    Start-Sleep -Milliseconds 500
}
Report 'port-8080-closed' $port8080Closed ''
if ($state.gateway_enabled) {
    Report 'port-8081-closed' $port8081Closed ''
}

# ----------------------------------------------------- 4. drop test database
if ($KeepDatabase) {
    Write-Host "beta-stand: keeping database $($state.db_name) (-KeepDatabase)"
}
elseif (Test-Path -LiteralPath $baseDbSecret) {
    # Same shared guard as the start script (db-drop also refuses reserved
    # names itself): a recorded name pointing at the working database or the
    # public schema must never reach DROP.
    $dropMode = if ([string]$state.db_mode -eq 'schema') { 'schema' } else { 'database' }
    $nameCheck = Invoke-NativeCapture -Exe $python -Arguments @(
        $helper, 'check-database-name', '--source-secret', $baseDbSecret,
        '--database', [string]$state.db_name, '--mode', $dropMode)
    if ($nameCheck.Exit -ne 0) {
        Report 'drop-database' $false ('recorded name refused: ' + ($nameCheck.Lines -join ' '))
    }
    else {
        $dropArguments = @($helper, 'db-drop', '--source-secret', $baseDbSecret, '--database', [string]$state.db_name)
        if ($dropMode -eq 'schema') { $dropArguments += @('--schema', [string]$state.db_name) }
        $drop = Invoke-NativeCapture -Exe $python -Arguments $dropArguments
        if ($drop.Exit -eq 0) { Report 'drop-database' $true ($drop.Lines -join ' ') }
        else { Report 'drop-database' $false ($drop.Lines -join ' ') }
    }
}
else {
    Report 'drop-database' $false 'config/local/database-url.secret not found'
}

# ------------------------------------------------- 5. remove stand secrets
foreach ($file in @($statePath, $machineSecret,
                    (Join-Path $standDir 'session.secret'),
                    (Join-Path $standDir 'beta-database-url.secret'),
                    (Join-Path $standDir 'live-worker-config.json'),
                    (Join-Path $standDir 'agent-connection.json'),
                    $envJsonPath)) {
    Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue
}
Write-Host '  stand secrets/state/env removed (logs and results kept under .test_runtime\beta-stand)'

Write-Host ''
if ($script:failureReasons.Count -eq 0) {
    Write-Host 'BETA STAND TEARDOWN: PASS'
    exit 0
}
Write-Host 'BETA STAND TEARDOWN: FAIL'
foreach ($reason in $script:failureReasons) { Write-Host "  - $reason" }
exit 1
