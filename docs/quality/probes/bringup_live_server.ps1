<#
.SYNOPSIS
    Brings up the live Kraken Server and Blob Gateway from the saved user
    configuration, verifies health, snapshots the protected user project state
    (read-only) and writes a receipt.

.DESCRIPTION
    Task 4.1 of docs/superpowers/plans/2026-11-06-kraken-verification-continuation.md.

    Steps:
      1. Pre-check 127.0.0.1:8080 / 8081. A port already serving a healthy
         Kraken Server (GET /api/v1/health with metadata=postgresql) or Blob
         Gateway (GET /health with service=kraken-blob-gateway) is reused and
         recorded; any other owner aborts the probe.
      2. Start the Blob Gateway, then the Server (server last, so an enabled
         gateway dependency would see the gateway up). With
         [blob_gateway] enabled = false the server does not require or start
         the gateway itself (ServerConfig.apply_to_environment clears all
         KRAKEN_BLOB_GATEWAY_* variables), which is why the gateway is started
         separately, mirroring the server's own BlobGatewayManager lifecycle
         via launch_blob_gateway.py (same executable, same child environment,
         so ticket secrets match).
      3. Poll GET http://127.0.0.1:8080/api/v1/health and
         GET http://127.0.0.1:8081/health (60 s timeout each).
      4. Authenticate with the established live-probe pattern (DPAPI session
         file .test_runtime/live-server/server-session.secret, see
         live_server_context.py) and snapshot the protected project
         7c622c09-73fa-552f-9a52-e80b64aeb3ef read-only: revision, history
         length, job count/states and SHA-256 of each job's payload body as
         returned. If the session file is missing or expired the baseline is
         recorded as BLOCKED with the exact error; no credentials are created.
      5. Write docs/quality/evidence/<date>/live-bringup.json (health statuses,
         PIDs, protected baseline, config paths without secret values, redacted
         startup commands). The receipt never contains secret or token values.
      6. With -KeepRunning the processes are left running (exit 0, PIDs are
         printed); otherwise both are stopped in a finally block after the
         receipt.

    Launcher note: kraken_core.process_lifetime makes the server terminate
    when a watched ancestor exits. This script therefore re-executes itself
    with -LaunchServer as a persistent hidden launcher that is the server's
    direct parent; KRAKEN_PROCESS_LIFETIME_ROOT_PID=<launcher pid> makes the
    server watch only the launcher, so the server survives this probe's exit.
    The launcher waits on the server and exits when the server stops.

    Exit codes: 0 = server healthy and receipt written (baseline may be
    BLOCKED, see receipt status PARTIAL); 1 = pre-check or bring-up failure.

.PARAMETER KeepRunning
    Leave the started server and gateway running after the receipt is written.

.PARAMETER ReceiptDir
    Receipt directory. Default: docs/quality/evidence/<yyyy-MM-dd>/ relative
    to the repository root.
#>
[CmdletBinding()]
param(
    [switch]$KeepRunning,
    [string]$ReceiptDir = "",

    # Internal parameters: used when this script is re-executed as the
    # persistent server launcher. Not meant for direct invocation.
    [switch]$LaunchServer,
    [string]$ServerPidFile = "",
    [string]$ServerLogFile = "",
    [string]$ServerStdErrFile = ""
)

$ErrorActionPreference = "Stop"

# Python helpers print JSON; force UTF-8 on both sides of the process boundary
# so non-ASCII values (e.g. display names) survive capture and the receipt.
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ProbesDir = $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ServerConfig = Join-Path $RepoRoot "config\local\server.toml"
$ServerHealthUrl = "http://127.0.0.1:8080/api/v1/health"
$GatewayHealthUrl = "http://127.0.0.1:8081/health"
$ProtectedProjectId = "7c622c09-73fa-552f-9a52-e80b64aeb3ef"

function Write-Step([string]$Message) {
    Write-Host "[bringup] $Message"
}

# --- internal launcher mode ---------------------------------------------------
if ($LaunchServer) {
    if (-not $ServerPidFile -or -not $ServerLogFile -or -not $ServerStdErrFile) {
        throw "Launcher mode requires -ServerPidFile, -ServerLogFile and -ServerStdErrFile"
    }
    # Watch only this launcher (see the note in the header): the probe's own
    # process is excluded from the server's ancestor watch.
    $env:KRAKEN_PROCESS_LIFETIME_ROOT_PID = "$PID"
    $server = Start-Process -FilePath $Python `
        -ArgumentList @('-m', 'kraken_server', '--config', $ServerConfig) `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $ServerLogFile -RedirectStandardError $ServerStdErrFile
    [System.IO.File]::WriteAllText($ServerPidFile, "$($server.Id)")
    Wait-Process -Id $server.Id -ErrorAction SilentlyContinue
    exit 0
}

# --- helpers --------------------------------------------------------------------

function Test-TcpPort([string]$Computer, [int]$Port, [int]$TimeoutMs) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $asyncResult = $client.BeginConnect($Computer, $Port, $null, $null)
        if ($asyncResult.AsyncWaitHandle.WaitOne($TimeoutMs)) {
            try { $client.EndConnect($asyncResult); return $true } catch { return $false }
        }
        return $false
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Invoke-JsonGet([string]$Url, [int]$TimeoutSeconds) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds
        $body = $null
        if ($response.Content) { $body = $response.Content | ConvertFrom-Json }
        return [pscustomobject]@{ Ok = $true; StatusCode = [int]$response.StatusCode; Body = $body; Error = $null }
    }
    catch {
        $statusCode = $null
        $body = $null
        if ($_.Exception.Response) {
            $statusCode = [int]$_.Exception.Response.StatusCode
            try {
                $stream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
                $body = $reader.ReadToEnd()
            }
            catch { }
        }
        return [pscustomobject]@{ Ok = $false; StatusCode = $statusCode; Body = $body; Error = $_.Exception.Message }
    }
}

# --- run directory and receipt location ------------------------------------------

$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path (Join-Path $env:LOCALAPPDATA "Temp\opencode") "live-bringup-$runStamp"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$serverLogFile = Join-Path $runDir "server.log"
$serverStdErrFile = Join-Path $runDir "server-stderr.log"
$gatewayLogFile = Join-Path $runDir "gateway.log"
$serverPidFile = Join-Path $runDir "server.pid"
$gatewayPidFile = Join-Path $runDir "gateway.pid"

if (-not $ReceiptDir) {
    $ReceiptDir = Join-Path $RepoRoot "docs\quality\evidence\$(Get-Date -Format 'yyyy-MM-dd')"
}
elseif (-not [System.IO.Path]::IsPathRooted($ReceiptDir)) {
    $ReceiptDir = Join-Path $RepoRoot $ReceiptDir
}
New-Item -ItemType Directory -Path $ReceiptDir -Force | Out-Null
$receiptPath = Join-Path $ReceiptDir "live-bringup.json"

# --- state ------------------------------------------------------------------------

$serverPid = $null
$launcherProcess = $null
$gatewayPid = $null
$serverReused = $false
$gatewayReused = $false
$gatewayBlocked = $false
$gatewaySpec = $null
$serverHealth = $null
$gatewayHealth = $null
$baseline = $null
$probeFailed = $false

function Stop-ProbeProcesses {
    if ($serverPid -and -not $serverReused) {
        Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
    }
    if ($launcherProcess -and -not $launcherProcess.HasExited) {
        Stop-Process -Id $launcherProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($gatewayPid -and -not $gatewayReused) {
        Stop-Process -Id $gatewayPid -Force -ErrorAction SilentlyContinue
    }
}

try {
    # --- 1. pre-check: ports and reusable healthy services -------------------------

    if (Test-TcpPort "127.0.0.1" 8080 750) {
        $probe = Invoke-JsonGet $ServerHealthUrl 3
        if ($probe.Ok -and $probe.Body -and $probe.Body.status -eq "ok" -and
            $probe.Body.metadata -eq "postgresql" -and $probe.Body.api_version -eq "v1") {
            Write-Step "Port 8080 already serves a healthy Kraken Server - reusing it"
            $serverReused = $true
            $serverHealth = $probe
            $owner = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($owner) { $serverPid = [int]$owner.OwningProcess }
        }
        else {
            Write-Step "ERROR: port 8080 is occupied by a process that is not a healthy Kraken Server; aborting"
            $probeFailed = $true
        }
    }

    if (-not $probeFailed -and (Test-TcpPort "127.0.0.1" 8081 750)) {
        $probe = Invoke-JsonGet $GatewayHealthUrl 3
        if ($probe.Ok -and $probe.Body -and $probe.Body.service -eq "kraken-blob-gateway") {
            Write-Step "Port 8081 already serves a healthy Kraken Blob Gateway - reusing it"
            $gatewayReused = $true
            $gatewayHealth = $probe
            $owner = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($owner) { $gatewayPid = [int]$owner.OwningProcess }
        }
        else {
            Write-Step "ERROR: port 8081 is occupied by a process that is not a Kraken Blob Gateway; aborting"
            $probeFailed = $true
        }
    }

    # --- 2a. gateway first ----------------------------------------------------------

    if (-not $probeFailed -and -not $gatewayReused) {
        Write-Step "Starting Blob Gateway (mirrors the server's own BlobGatewayManager lifecycle)"
        $gatewaySpecRaw = & $Python (Join-Path $ProbesDir "launch_blob_gateway.py") --config $ServerConfig --log $gatewayLogFile --pid-file $gatewayPidFile
        $gatewaySpec = ($gatewaySpecRaw -join "`n") | ConvertFrom-Json
        if ($gatewaySpec.status -eq "ok") {
            $gatewayPid = [int]$gatewaySpec.pid
            Write-Step "Blob Gateway started, PID $gatewayPid"
        }
        else {
            Write-Step "Blob Gateway BLOCKED: $($gatewaySpec.error)"
            $gatewayBlocked = $true
        }
    }

    # --- 2b. server last, through the persistent launcher ---------------------------

    if (-not $probeFailed -and -not $serverReused) {
        Write-Step "Starting Kraken Server through the persistent launcher"
        $launcherProcess = Start-Process -FilePath (Join-Path $PSHOME "powershell.exe") `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
                '-LaunchServer', '-ServerPidFile', $serverPidFile,
                '-ServerLogFile', $serverLogFile, '-ServerStdErrFile', $serverStdErrFile) `
            -WindowStyle Hidden -PassThru
        $deadline = (Get-Date).AddSeconds(30)
        while (-not (Test-Path $serverPidFile) -and (Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 100
        }
        if (Test-Path $serverPidFile) {
            $serverPid = [int]([System.IO.File]::ReadAllText($serverPidFile).Trim())
            Write-Step "Kraken Server started, PID $serverPid (launcher PID $($launcherProcess.Id))"
        }
        elseif ($launcherProcess.HasExited) {
            Write-Step "ERROR: launcher exited before starting the server (exit code $($launcherProcess.ExitCode))"
            if (Test-Path $serverStdErrFile) {
                Get-Content $serverStdErrFile -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[server-stderr] $_" }
            }
            $probeFailed = $true
        }
        else {
            Write-Step "ERROR: server PID file did not appear within 30 seconds"
            $probeFailed = $true
        }
    }

    # --- 3. health polling (60 s timeout) --------------------------------------------

    if (-not $probeFailed -and -not $serverReused -and $serverHealth.Ok -ne $true) {
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $deadline) {
            $check = Invoke-JsonGet $ServerHealthUrl 3
            if ($check.Ok -and $check.Body -and $check.Body.status -eq "ok") {
                $serverHealth = $check
                break
            }
            Start-Sleep -Milliseconds 500
        }
        if ($serverHealth.Ok -ne $true -or $serverHealth.Body.status -ne "ok") {
            Write-Step "ERROR: Kraken Server did not become healthy at $ServerHealthUrl within 60 s"
            if (Test-Path $serverStdErrFile) {
                Get-Content $serverStdErrFile -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[server-stderr] $_" }
            }
            if (Test-Path $serverLogFile) {
                Get-Content $serverLogFile -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[server-log] $_" }
            }
            $probeFailed = $true
        }
        else {
            Write-Step "Server healthy: $($serverHealth.Body | ConvertTo-Json -Compress -Depth 6)"
        }
    }

    if (-not $probeFailed -and -not $gatewayReused -and -not $gatewayBlocked) {
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $deadline) {
            $check = Invoke-JsonGet $GatewayHealthUrl 3
            if ($check.Ok -and $check.Body -and $check.Body.service -eq "kraken-blob-gateway" -and $check.Body.status -eq "ok") {
                $gatewayHealth = $check
                break
            }
            Start-Sleep -Milliseconds 500
        }
        if ($gatewayHealth.Ok -ne $true -or $gatewayHealth.Body.status -ne "ok") {
            Write-Step "ERROR: Blob Gateway did not become healthy at $GatewayHealthUrl within 60 s"
            if (Test-Path $gatewayLogFile) {
                Get-Content $gatewayLogFile -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[gateway-log] $_" }
            }
            $probeFailed = $true
        }
        else {
            Write-Step "Gateway healthy: $($gatewayHealth.Body | ConvertTo-Json -Compress -Depth 6)"
        }
    }

    # --- 4. authenticated read-only baseline of the protected project ----------------

    if (-not $probeFailed) {
        Write-Step "Snapshotting protected project $ProtectedProjectId (read-only)"
        $baselineRaw = & $Python (Join-Path $ProbesDir "snapshot_live_baseline.py")
        $baseline = ($baselineRaw -join "`n") | ConvertFrom-Json
        if ($baseline.status -eq "ok") {
            Write-Step "Baseline captured: revision $($baseline.project.revision), history $($baseline.history_length) events, $($baseline.jobs.count) jobs"
        }
        else {
            Write-Step "Baseline BLOCKED ($($baseline.error_type)): $($baseline.error)"
        }
    }

    # --- 5. receipt --------------------------------------------------------------------

    $gatewayHealthy = $gatewayReused -or ($gatewayHealth -and $gatewayHealth.Ok -and $gatewayHealth.Body.status -eq "ok")
    $baselineOk = $baseline -and $baseline.status -eq "ok"
    $overall = "FAIL"
    if (-not $probeFailed -and $serverHealth -and $serverHealth.Ok -and $serverHealth.Body.status -eq "ok") {
        if ($gatewayHealthy -and $baselineOk) { $overall = "PASS" } else { $overall = "PARTIAL" }
    }

    $configDir = Split-Path $ServerConfig -Parent
    if ($gatewaySpec) {
        $gatewayExecutable = [string]$gatewaySpec.executable
    }
    else {
        $gatewayExecutable = ""
    }
    $gatewayCommand = if ($gatewayExecutable) {
        "<gateway executable> with environment KRAKEN_BLOB_ROOT / KRAKEN_BLOB_GATEWAY_BIND / KRAKEN_BLOB_GATEWAY_SECRET=<redacted> (exact BlobGatewayManager child environment, no CLI arguments)"
    }
    else {
        "not started (no gateway configuration available)"
    }

    $serverProcessReceipt = [ordered]@{}
    if ($serverReused) {
        $serverProcessReceipt = [ordered]@{ reused_existing_process = $true; pid = $null }
    }
    elseif ($serverPid) {
        $serverProcessReceipt = [ordered]@{
            started_by_probe = $true; pid = $serverPid; launcher_pid = $launcherProcess.Id
            log = $serverLogFile; stderr_log = $serverStdErrFile
        }
    }
    else {
        $serverProcessReceipt = [ordered]@{ started_by_probe = $false; pid = $null }
    }

    $gatewayProcessReceipt = [ordered]@{}
    if ($gatewayReused) {
        $gatewayProcessReceipt = [ordered]@{ reused_existing_process = $true; pid = $null }
    }
    elseif ($gatewayPid) {
        $gatewayProcessReceipt = [ordered]@{
            started_by_probe = $true; pid = $gatewayPid; executable = $gatewayExecutable
            log = $gatewayLogFile
            environment_note = "exact BlobGatewayManager child environment (KRAKEN_BLOB_ROOT, KRAKEN_BLOB_GATEWAY_BIND, KRAKEN_BLOB_GATEWAY_SECRET redacted)"
        }
    }
    else {
        $gatewayProcessReceipt = [ordered]@{ started_by_probe = $false; pid = $null }
        if ($gatewayBlocked) {
            $gatewayProcessReceipt.status = "BLOCKED"
            $gatewayProcessReceipt.error = [string]$gatewaySpec.error
            $gatewayProcessReceipt.executable_expected = [string]$gatewaySpec.executable
        }
    }

    $serverHealthReceipt = if ($serverHealth) {
        [ordered]@{ url = $ServerHealthUrl; http_status = $serverHealth.StatusCode; body = $serverHealth.Body }
    }
    else {
        [ordered]@{ url = $ServerHealthUrl; http_status = $null; body = $null; note = "not reached" }
    }

    $gatewayHealthReceipt = if ($gatewayHealth) {
        [ordered]@{ url = $GatewayHealthUrl; http_status = $gatewayHealth.StatusCode; body = $gatewayHealth.Body }
    }
    elseif ($gatewayBlocked) {
        [ordered]@{ url = $GatewayHealthUrl; status = "BLOCKED"; note = "gateway not started; see processes.gateway" }
    }
    else {
        [ordered]@{ url = $GatewayHealthUrl; http_status = $null; body = $null; note = "not reached" }
    }

    $receipt = [ordered]@{
        probe = "bringup_live_server"
        plan_task = "4.1"
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
        status = $overall
        server_url = "http://127.0.0.1:8080"
        gateway_url = "http://127.0.0.1:8081"
        keep_running = [bool]$KeepRunning
        health = [ordered]@{ server = $serverHealthReceipt; gateway = $gatewayHealthReceipt }
        processes = [ordered]@{ server = $serverProcessReceipt; gateway = $gatewayProcessReceipt }
        protected_baseline = $baseline
        configuration = [ordered]@{
            server_config = $ServerConfig
            database_url_secret = Join-Path $configDir "database-url.secret"
            blob_gateway_secret = Join-Path $configDir "blob-gateway.secret"
            blob_gateway_executable = $gatewayExecutable
            note = "paths only; no secret or token values are recorded in this receipt"
        }
        startup_commands = [ordered]@{
            server = "$Python -m kraken_server --config $ServerConfig"
            gateway = $gatewayCommand
        }
        logs = [ordered]@{ run_dir = $runDir }
        protected_project_id = $ProtectedProjectId
    }

    $receiptJson = $receipt | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText($receiptPath, $receiptJson + "`n", (New-Object System.Text.UTF8Encoding($false)))
    Write-Step "Receipt written: $receiptPath (status $overall)"
}
finally {
    # Reused (pre-existing) processes are never touched; only probe-started ones.
    if (-not $KeepRunning) {
        Stop-ProbeProcesses
    }
}

# --- 6. result ---------------------------------------------------------------------

if ($probeFailed) {
    Write-Step "FAILED"
    exit 1
}

if ($KeepRunning) {
    if ($serverReused) {
        Write-Step "KeepRunning: reusing existing Kraken Server on 127.0.0.1:8080"
    }
    else {
        Write-Step "KeepRunning: Kraken Server PID $serverPid (launcher PID $($launcherProcess.Id))"
    }
    if ($gatewayReused) {
        Write-Step "KeepRunning: reusing existing Blob Gateway on 127.0.0.1:8081"
    }
    elseif ($gatewayPid) {
        Write-Step "KeepRunning: Blob Gateway PID $gatewayPid"
    }
    else {
        Write-Step "KeepRunning: Blob Gateway not running ($([string]$gatewaySpec.error))"
    }
    Write-Step "DONE (status $overall)"
    exit 0
}

Write-Step "Stopped probe-started processes; server and gateway no longer running"
exit 0
