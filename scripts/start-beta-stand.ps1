<#
.SYNOPSIS
    Bring up the permanent Kraken beta stand: PostgreSQL test database +
    Server + Blob Gateway + machine token (+ optional agent worker).

.DESCRIPTION
    Repeatable, idempotent bring-up for the live/integration gate tiers
    (see docs/quality/beta-stand.md). Never touches the working
    kraken_local database or config/local/server.toml: everything the
    stand owns lives in the dedicated PostgreSQL database
    kraken_beta_test and under .test_runtime\beta-stand (git-ignored).

    Steps: provision/reset the test database -> render the stand
    server.toml -> kraken-admin migrate -> start the Server (the Server
    spawns/adopts the Blob Gateway child) -> beta account + session token
    -> protected sentinel project -> machine token (frames.vectorize.v1)
    -> optional kraken-agent server worker -> write live-worker-config
    JSON + .test_runtime\beta-stand.env.json + state.json.

    Re-running converges: a healthy Server on 127.0.0.1:8080 and a live
    owned agent are reused, the account/project/token are reused while
    valid, and already-provisioned databases are reused when the Server
    is still up (reset when it is not). Use -Restart for a clean
    restart of the stand's own components (full teardown via
    stop-beta-stand.ps1, including dropping kraken_beta_test).

    The background agent worker is OPT-IN (-WithWorkerAgent): the live
    acceptance fixture IS a controlled one-shot machine worker, and a
    polling global-lease agent would take its job ("Unexpected lease;
    stop acceptance"). Keep the stand without the worker while running
    tests/test_live_server_worker_acceptance.py / plugin chain.

    Prints one "PASS <component>" or "FAIL <component>: <reason>" line per
    component and exits 0 only when all of them PASS. Secret VALUES are
    never printed; the plaintext KRAKEN_TEST_POSTGRES_URL goes only into
    .test_runtime\beta-stand.env.json (git-ignored).

.PARAMETER DatabaseName
    Dedicated PostgreSQL database for the stand. Default kraken_beta_test.

.PARAMETER WithWorkerAgent
    Also start kraken-agent in Server worker mode (leases queued jobs on
    its own). Leave off for live acceptance runs.

.PARAMETER Restart
    Run stop-beta-stand.ps1 first, then bring the stand up from scratch.

.NOTES
    PowerShell 5.1 conventions (scripts/quality-gates.ps1): native
    commands run via Invoke-Native with EAP=Continue; all DPAPI/TOML/HTTP
    logic lives in tools/quality/beta_stand.py.
    Blob Gateway prerequisite: blob_gateway\target\release\kraken-blob-gateway.exe
    (Rust build; on closed-network machines the crate sources must come
    from the offline kit's dependencies/cargo vendor, see
    offline/Create-OfflineKit.ps1). Without the exe every other component
    still comes up in proxy mode, but the live acceptance requires
    transfer mode "gateway" and will FAIL.
#>
[CmdletBinding()]
param(
    [string]$DatabaseName = 'kraken_beta_test',

    [switch]$WithWorkerAgent,

    [switch]$Restart
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
$dbSecret = Join-Path $standDir 'beta-database-url.secret'
$sessionSecret = Join-Path $standDir 'session.secret'
$machineSecret = Join-Path $standDir 'machine.secret'
$pluginsJson = Join-Path $standDir 'worker-plugins.json'
$liveConfig = Join-Path $standDir 'live-worker-config.json'
$resultsDir = Join-Path $standDir 'results'
$logsDir = Join-Path $standDir 'logs'
$baseToml = Join-Path $repo 'config\local\server.toml'
$baseDbSecret = Join-Path $repo 'config\local\database-url.secret'
$serverUrl = 'http://127.0.0.1:8080'
$gatewayHealth = 'http://127.0.0.1:8081/health'
$betaUsername = 'beta-stand'
$protectedProjectName = 'beta-stand-protected'

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
function Skip {
    param([string]$Component, [string]$Detail)
    Write-Host ("SKIP {0}: {1}" -f $Component, $Detail)
}
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments, [string]$LogFile = '')
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($LogFile) {
            & $Exe @Arguments 2>&1 | Out-File -LiteralPath $LogFile -Encoding utf8
        }
        else {
            & $Exe @Arguments 2>&1 | Write-Host
        }
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prev
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
function Wait-Health {
    param([string]$Url, [int]$TimeoutSeconds = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return $true }
        }
        catch { }
        Start-Sleep -Milliseconds 500
    }
    return $false
}
function New-BetaPassword {
    -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })
}
function Start-DetachedComponent {
    <#
        Kraken processes bind their lifetime to their launching console tree
        (kraken_core.process_lifetime exits when an ancestor dies), so a
        component started as a plain child of this script would die when the
        caller's shell command ends. A two-stage `cmd /c start` launcher
        orphans the component: its stale ancestor entries are already-exited
        PIDs, which process_lifetime ignores, so the stand survives the
        console and is stopped only by stop-beta-stand.ps1.
    #>
    param([string]$LauncherPath, [string]$CommandLine, [string]$OutLog)
    $lines = @(
        '@echo off',
        ('start "kraken-beta-stand" /min cmd /c ""{0}" {1} > "{2}" 2>&1"' -f $python, $CommandLine, $OutLog)
    )
    [IO.File]::WriteAllLines($LauncherPath, $lines)
    Start-Process -FilePath $LauncherPath -WindowStyle Hidden | Out-Null
}
function Find-StandPid {
    <#
        Locate the real component PID from its command line (the launcher cmd
        exits immediately, so no Start-Process PID exists to keep).
    #>
    param([string]$LikePattern, [int]$TimeoutSeconds = 30)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $row = Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
            Where-Object { [string]$_.CommandLine -like $LikePattern } |
            Select-Object -First 1
        if ($row) { return [int]$row.ProcessId }
        Start-Sleep -Milliseconds 500
    }
    return 0
}

# ------------------------------------------------------------------ restart
if ($Restart -and (Test-Path -LiteralPath $statePath)) {
    Write-Host '[beta-stand] -Restart: running stop-beta-stand.ps1 first'
    & (Join-Path $PSScriptRoot 'stop-beta-stand.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'stop-beta-stand.ps1 failed; refusing to start over a half-stopped stand' }
}

# ------------------------------------------------------------- preconditions
$preconditionsOk = $true
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Report 'preconditions' $false "venv python not found: $python"
    $preconditionsOk = $false
}
if (-not (Test-Path -LiteralPath $baseToml -PathType Leaf)) {
    Report 'preconditions' $false "missing config/local/server.toml (git-ignored, machine-owned)"
    $preconditionsOk = $false
}
if (-not (Test-Path -LiteralPath $baseDbSecret -PathType Leaf)) {
    Report 'preconditions' $false 'missing config/local/database-url.secret'
    $preconditionsOk = $false
}
if (-not $preconditionsOk) { exit 1 }
# Guard the dedicated name BEFORE any DDL: it must never be the working
# database, a maintenance/template database, or (schema-fallback mode) a
# reserved schema such as public. Checked with the strictest mode because
# provisioning may fall back to schema mode.
$nameCheck = Invoke-NativeCapture -Exe $python -Arguments @(
    $helper, 'check-database-name', '--source-secret', $baseDbSecret,
    '--database', $DatabaseName, '--mode', 'schema')
if ($nameCheck.Exit -ne 0) {
    Report 'preconditions' $false ('-DatabaseName refused: ' + ($nameCheck.Lines -join ' '))
    exit 1
}
New-Item -ItemType Directory -Path $standDir, $logsDir, $resultsDir -Force | Out-Null

$state = $null
if (Test-Path -LiteralPath $statePath) {
    try { $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } catch { $state = $null }
}

# --------------------------------------------------------- 1. test database
$serverAlreadyUp = Test-PortListening -Port 8080
$dbMode = 'database'
$provisioned = $false
if ($serverAlreadyUp -and $state -and $state.db_provisioned) {
    Report 'postgres' $true 'stand Server is live; reusing database (no reset while the Server holds it)'
    $dbMode = [string]$state.db_mode
    $provisioned = $true
}
else {
    $provision = Invoke-NativeCapture -Exe $python -Arguments @(
        $helper, 'db-provision', '--source-secret', $baseDbSecret,
        '--destination-secret', $dbSecret, '--database', $DatabaseName, '--reset')
    if ($provision.Exit -ne 0) {
        Write-Host ('[beta-stand] CREATE DATABASE failed (' + ($provision.Lines -join ' ') +
            '); falling back to a dedicated SCHEMA')
        $provision = Invoke-NativeCapture -Exe $python -Arguments @(
            $helper, 'db-provision', '--source-secret', $baseDbSecret,
            '--destination-secret', $dbSecret, '--database', $DatabaseName, '--reset', '--mode', 'schema')
        if ($provision.Exit -eq 0) { $dbMode = 'schema' }
    }
    if ($provision.Exit -eq 0) {
        $provisioned = $true
        Report 'postgres' $true ($provision.Lines -join ' ')
    }
    else {
        Report 'postgres' $false ('database provisioning failed: ' + ($provision.Lines -join ' '))
    }
}

# ------------------------------------------------- 2. stand server.toml
$gatewayExe = ''
foreach ($candidate in @(
    (Join-Path $repo 'blob_gateway\target\release\kraken-blob-gateway.exe'),
    ((Get-Content -LiteralPath $baseToml -Raw) | Select-String -Pattern 'executable\s*=\s*"([^"]+)"' |
        ForEach-Object { $_.Matches[0].Groups[1].Value -replace '\\\\', '\' }))) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { $gatewayExe = $candidate; break }
}
$render = Invoke-NativeCapture -Exe $python -Arguments @(
    $helper, 'render-server-config', '--base', $baseToml, '--out', $serverToml,
    '--database-url-secret-name', 'beta-database-url.secret',
    '--gateway-executable', $(if ($gatewayExe) { $gatewayExe } else { 'none' }))
if ($render.Exit -eq 0 -and (Test-Path -LiteralPath $serverToml)) {
    Report 'server-config' $true ($render.Lines -join ' ')
}
else {
    Report 'server-config' $false ($render.Lines -join ' ')
    Write-Host '[beta-stand] cannot continue without the stand server.toml'
    exit 1
}

# --------------------------------------------------------- 3. migrations
if ($provisioned) {
    $migrateLog = Join-Path $logsDir 'migrate.log'
    $migrateExit = Invoke-Native -Exe $python -Arguments @(
        '-m', 'kraken_admin', 'migrate', '--config', $serverToml) -LogFile $migrateLog
    if ($migrateExit -eq 0) { Report 'migrate' $true }
    else {
        Report 'migrate' $false "exit $migrateExit; see $migrateLog"
    }
}
else {
    Report 'migrate' $false 'skipped because the database was not provisioned'
}

# ------------------------------------------------------------- 4. server
$serverPid = $null
if ($serverAlreadyUp) {
    # Safety: only reuse a Server THIS stand started (its command line
    # references the stand server.toml). A working/foreign Server on 8080
    # must never be adopted - later steps issue tokens and drop databases.
    $standPid = Find-StandPid -LikePattern '*kraken_server*beta-stand*server.toml*' -TimeoutSeconds 2
    $healthy = Wait-Health -Url "$serverUrl/api/v1/health" -TimeoutSeconds 10
    if ($healthy -and $standPid -gt 0) {
        Report 'server' $true "already serving on 127.0.0.1:8080 (pid $standPid); reusing"
        $serverPid = $standPid
    }
    elseif ($healthy) {
        Report 'server' $false ('127.0.0.1:8080 is served by a process this stand did not start; ' +
            'stop it first - the stand refuses to adopt a foreign Server')
    }
    else {
        Report 'server' $false 'port 8080 is listening but /api/v1/health is not 200'
    }
}
else {
    $stderr = Join-Path $logsDir 'server.err.log'
    Start-DetachedComponent -LauncherPath (Join-Path $standDir 'run-server.cmd') `
        -CommandLine @('-m', 'kraken_server', '--config', ('"{0}"' -f $serverToml)) `
        -OutLog $stderr
    if (Wait-Health -Url "$serverUrl/api/v1/health" -TimeoutSeconds 60) {
        # The launcher cmd exits at once; find the real python PID by the
        # stand-specific command line so stop-beta-stand.ps1 can own it.
        $serverPid = Find-StandPid -LikePattern '*kraken_server*beta-stand*server.toml*'
        Report 'server' $true "pid $serverPid"
    }
    else {
        $serverPid = Find-StandPid -LikePattern '*kraken_server*beta-stand*server.toml*' -TimeoutSeconds 2
        Report 'server' $false "not healthy within 60s; see $stderr"
    }
}

# ---------------------------------------------------------- 5. blob gateway
$gatewayPid = $null
$gatewayEnabled = [bool]$gatewayExe
if (-not $gatewayEnabled) {
    Report 'gateway' $false ('kraken-blob-gateway.exe not found (build blob_gateway with cargo; on closed ' +
        'networks the crates must come from the offline kit dependencies/cargo vendor). Uploads stay in ' +
        'proxy mode and the live acceptance requires mode "gateway".')
}
else {
    $deadline = (Get-Date).AddSeconds(20)
    $gatewayOk = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $gatewayHealth -UseBasicParsing -TimeoutSec 2
            $service = ($response.Content | ConvertFrom-Json).service
            if ($response.StatusCode -eq 200 -and $service -eq 'kraken-blob-gateway') { $gatewayOk = $true; break }
        }
        catch { }
        Start-Sleep -Milliseconds 500
    }
    if ($gatewayOk) {
        $owner = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty OwningProcess
        if ($owner) { $gatewayPid = [int]$owner }
        Report 'gateway' $true "127.0.0.1:8081 healthy (pid $gatewayPid, child of the Server)"
    }
    else {
        Report 'gateway' $false "exe exists but $gatewayHealth is not healthy; if the Server was reused from a " +
            'run without the gateway, stop and start the stand again'
    }
}

# ------------------------------------------------- 6. account + session token
$betaPassword = if ($state -and $state.beta_password) { [string]$state.beta_password } else { New-BetaPassword }
if ($provisioned -or $serverAlreadyUp) {
    $login = Invoke-NativeCapture -Exe $python -Arguments @(
        $helper, 'login', '--server-url', $serverUrl, '--username', $betaUsername,
        '--password', $betaPassword, '--display-name', 'Beta Stand',
        '--destination-secret', $sessionSecret)
    if ($login.Exit -eq 0) { Report 'account' $true ($login.Lines -join ' ') }
    else { Report 'account' $false ($login.Lines -join ' ') }
}
else {
    Report 'account' $false 'skipped because the Server is not up'
}

# ------------------------------------------------ 7. protected sentinel project
$projectId = ''
if (Test-Path -LiteralPath $sessionSecret) {
    $projectArgs = @($helper, 'ensure-project', '--server-url', $serverUrl,
        '--token-secret', $sessionSecret, '--name', $protectedProjectName,
        '--width', '2', '--height', '1')
    if ($state -and $state.protected_project_id) {
        $projectArgs += @('--project-id', [string]$state.protected_project_id)
    }
    $project = Invoke-NativeCapture -Exe $python -Arguments $projectArgs
    if ($project.Exit -eq 0) {
        $projectId = [string](($project.Lines | Select-Object -Last 1 | ConvertFrom-Json).project_id)
        Report 'protected-project' $true "$projectId ($protectedProjectName)"
    }
    else { Report 'protected-project' $false ($project.Lines -join ' ') }
}
else {
    Report 'protected-project' $false 'skipped because no session token is available'
}

# ------------------------------------------------------- 8. machine token
$machineTokenId = ''
$tokenUsable = (Test-Path -LiteralPath $machineSecret) -and $state -and $state.machine_token_id
if ($tokenUsable) {
    $listJson = Invoke-NativeCapture -Exe $python -Arguments @(
        '-m', 'kraken_admin', 'list-agent-tokens', '--config', $serverToml, '--json')
    $stillActive = $false
    if ($listJson.Exit -eq 0) {
        try {
            $tokens = (($listJson.Lines -join "`n") | ConvertFrom-Json).items
            $stillActive = [bool](@($tokens | Where-Object {
                $_.token_id -eq [string]$state.machine_token_id -and -not $_.revoked }).Count -gt 0)
        }
        catch { $stillActive = $false }
    }
    if ($stillActive) {
        $machineTokenId = [string]$state.machine_token_id
        Report 'machine-token' $true "reusing active token $machineTokenId"
    }
}
if (-not $machineTokenId) {
    # Capture the one-time output in memory only; never write it to a log.
    $created = Invoke-NativeCapture -Exe $python -Arguments @(
        '-m', 'kraken_admin', 'create-agent-token', '--config', $serverToml,
        '--name', 'beta-stand-worker', '--capability', 'frames.vectorize.v1')
    $idMatch = [regex]::Match(($created.Lines -join "`n"), '(?m)^Agent token id: (.+)$')
    $tokenValue = ''
    if ($idMatch.Success) {
        $machineTokenId = $idMatch.Groups[1].Value.Trim()
        $promptIndex = [array]::IndexOf($created.Lines, 'Copy this token now; it cannot be shown again:')
        if ($promptIndex -ge 0) {
            foreach ($line in $created.Lines[($promptIndex + 1)..($created.Lines.Count - 1)]) {
                if ($line.Trim()) { $tokenValue = $line.Trim(); break }
            }
        }
    }
    if ($tokenValue) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $tokenValue | & $python $helper 'protect' --destination $machineSecret | Out-Null
            $protectExit = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $prev
            $tokenValue = ''
        }
        if ($protectExit -eq 0) { Report 'machine-token' $true "issued frames.vectorize.v1 token $machineTokenId" }
        else { Report 'machine-token' $false 'failed to protect the issued token' }
    }
    else {
        $machineTokenId = ''
        Report 'machine-token' $false ($created.Lines -join ' ')
    }
}

# ---------------------------------------------------------- 9. agent worker
$agentPid = $null
if (-not $WithWorkerAgent) {
    Skip 'agent' ('not requested; the live acceptance fixture is itself a one-shot machine worker ' +
        '(-WithWorkerAgent starts a polling agent; do not run it during acceptance)')
}
elseif ($state -and $state.agent_pid -and (Get-Process -Id ([int]$state.agent_pid) -ErrorAction SilentlyContinue)) {
    $agentPid = [int]$state.agent_pid
    Report 'agent' $true "pid $agentPid already running; reusing"
}
elseif ($machineTokenId -and (Test-Path -LiteralPath $machineSecret)) {
    @{ plugins = @(@{ operation = 'frames.vectorize.v1'; command = @($python, '-m', 'contour.worker') }) } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pluginsJson -Encoding UTF8
    $agentData = Join-Path $standDir 'agent-data'
    $agentConnection = Join-Path $standDir 'agent-connection.json'
    Remove-Item -LiteralPath $agentConnection -ErrorAction SilentlyContinue
    $env:KRAKEN_AGENT_TOKEN = (& $python $helper 'unprotect' --source $machineSecret) -join ''
    try {
        Start-DetachedComponent -LauncherPath (Join-Path $standDir 'run-agent.cmd') `
            -CommandLine @('-m', 'kraken_agent', '--server-url', $serverUrl,
                           '--plugins-config', ('"{0}"' -f $pluginsJson),
                           '--data-dir', ('"{0}"' -f $agentData), '--independent') `
            -OutLog (Join-Path $logsDir 'agent.err.log')
    }
    finally {
        $env:KRAKEN_AGENT_TOKEN = ''
    }
    $agentPid = Find-StandPid -LikePattern '*kraken_agent*beta-stand*agent-data*'
    if ($agentPid -gt 0) {
        Report 'agent' $true "server worker pid $agentPid (leases jobs on its own)"
    }
    else {
        Report 'agent' $false 'worker did not come up; see logs\agent.err.log'
    }
}
else {
    Report 'agent' $false 'no usable machine token; cannot start the worker'
}

# ------------------------------------------- 10. live worker config + env json
# The outputs are written whenever the core stand (Server, account,
# sentinel project, machine token) is up, even if the Blob Gateway is
# missing: the integration tier only needs KRAKEN_TEST_POSTGRES_URL. The
# overall verdict still FAILs when any component failed.
$componentsOk = ($script:failureReasons.Count -eq 0)
$coreOk = -not ($script:failureReasons | Where-Object { $_ -notmatch '^gateway:' })
if ($coreOk) {
    @{
        server_url = $serverUrl
        session_secret = $sessionSecret
        machine_secret = $machineSecret
        protected_project_id = $projectId
        results_directory = $resultsDir
    } | ConvertTo-Json | Set-Content -LiteralPath $liveConfig -Encoding UTF8
    $envExit = Invoke-Native -Exe $python -Arguments @(
        $helper, 'env-json', '--database-url-secret', $dbSecret,
        '--live-config', $liveConfig, '--out', $envJsonPath)
    if ($envExit -eq 0) { Report 'env-outputs' $true "live config: $liveConfig" }
    else { Report 'env-outputs' $false "env-json exited $envExit" }
}
else {
    Report 'env-outputs' $false 'skipped because the core stand (server/account/token) is not up'
}

# --------------------------------------------------------------- 11. state
@{
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
    server_url = $serverUrl
    server_pid = $serverPid
    gateway_pid = $gatewayPid
    agent_pid = $agentPid
    gateway_enabled = $gatewayEnabled
    db_name = $DatabaseName
    db_mode = $dbMode
    db_provisioned = $provisioned
    beta_username = $betaUsername
    # Plaintext is acceptable here: state.json lives under .test_runtime (git-ignored
    # '.*/' rule) and this is a throwaway loopback-only stand account, not a user secret.
    beta_password = $betaPassword
    protected_project_id = $projectId
    machine_token_id = $machineTokenId
    live_worker_config = $liveConfig
    env_json = $envJsonPath
} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8

Write-Host ''
Write-Host '==================== beta stand ===================='
Write-Host ("  caller env: `$env:KRAKEN_LIVE_WORKER_CONFIG = '{0}'" -f $liveConfig)
Write-Host ("  caller env: KRAKEN_TEST_POSTGRES_URL is in {0} (plaintext URL, git-ignored)" -f $envJsonPath)
Write-Host '  import both with: Get-Content .test_runtime\beta-stand.env.json | ConvertFrom-Json'
if ($componentsOk) {
    Write-Host ''
    Write-Host 'BETA STAND RESULT: PASS'
    exit 0
}
Write-Host ''
Write-Host 'BETA STAND RESULT: FAIL'
foreach ($reason in $script:failureReasons) { Write-Host "  - $reason" }
exit 1
