<#
.SYNOPSIS
    Kraken local quality gate: one command per verification tier.

.DESCRIPTION
    Runs the requested verification tiers, writes per-tier evidence
    (receipt.json + junit.xml + pytest.txt) under
    docs/quality/evidence/<yyyy-MM-dd>/gate-<tier>/, enforces the quality
    registry first, and (unless -SkipCoverage) rebuilds the coverage report
    from all of today's JUnit and FAILS the gate when a mandatory action
    that was PASS in the previous committed coverage report is FAIL now.

    The gate exits 0 only when:
      * the registry enforcement suite (tests/test_quality_registry.py) passed;
      * every requested tier is PASS, PARTIAL, or NOT_TESTED
        (PASS = exit 0, 0 skips; PARTIAL = exit 0 with skips, skips are never
        counted as pass; FAIL = non-zero exit, including interrupted runs and
        "no tests collected" - a mandatory suite that could not run fails);
      * the coverage comparison found no PASS -> FAIL regression
        (newly NOT_TESTED actions are informational only).

.PARAMETER Tier
    One or more of: smoke, unit, integration, gui, stateful, e2e, extended,
    regression. Repeatable (comma- or space-separated).

.PARAMETER OutDir
    Evidence root for this run. Default: docs/quality/evidence/<yyyy-MM-dd>/.

.PARAMETER SkipCoverage
    Skip the coverage report rebuild + regression comparison.

.PARAMETER Base
    Git base for the coverage report's source metric only (default HEAD~1).

.NOTES
    TIER -> SUITE MAPPING (refined against the actual test layout):

    smoke       tests/test_desktop_runtime_smoke.py. Two of its tests are
                env-gated (KRAKEN_SMOKE_SERVER_URL / KRAKEN_SMOKE_SERVER_TOKEN
                and KRAKEN_SMOKE_AGENT_URL / KRAKEN_SMOKE_AGENT_TOKEN) and
                self-skip without an isolated server/agent, so a run without
                those env vars is PARTIAL by design (skips are never pass).

    unit        The fast default suite WITHOUT network/live/PG:
                `pytest tests plugins/contour/tests/unit` with explicit
                --ignore patterns for every env-gated file. Refinements:
                (1) the plan proposed marker exclusion
                (`-m "not live and not pg"`) but no such markers are
                registered (pyproject only defines `performance`), so the
                gated files are identified instead by their env gates
                (KRAKEN_LIVE_*, KRAKEN_UI_TEST_CONFIG,
                KRAKEN_TEST_POSTGRES_URL, KRAKEN_RUN_* opt-in gates) and
                ignored explicitly (see $unitIgnores). The PostgreSQL gate is
                the shared `database` fixture in
                tests/test_admin_project_workflows.py (the only definition
                in the repo).
                (2) the two halves run as two pytest invocations (see
                "Import domains" below) and their JUnit files are merged in
                the tier receipt.
                (3) the plugin half inherits the contour plugin's own
                pytest configuration (plugins/contour/pyproject.toml:
                addopts "-ra --strict-markers -m fast",
                pythonpath ["src"]): pytest resolves the rootdir to
                plugins/contour for that invocation, so only the plugin's
                `fast` suite runs there (its conftest assigns `fast` to the
                everyday behavioral files; vectorization/full/slow files
                are collected but deselected).

    integration The PG-gated suites: every file that uses the `database`
                fixture or KRAKEN_TEST_POSTGRES_URL (see $integrationFiles).
                They self-skip without KRAKEN_TEST_POSTGRES_URL; the gate
                does NOT fail on skip and reports PARTIAL with the skip list.
                Refinement: the gate does not manage PostgreSQL clusters
                itself (that stays with the PG blocks of
                docs/quality/reproduce-stabilization.ps1); if
                KRAKEN_TEST_POSTGRES_URL is set in the environment it is
                inherited and the suites run for real.

    gui         pytest-qt offscreen suites, capped to an explicit file list
                (see $guiFiles): the proven offscreen `qt` block of
                reproduce-stabilization.ps1 plus the local layer_* Qt suites.
                Live/PG-gated Qt files stay out of this tier; the remaining
                Qt files are covered by the unit tier.

    stateful    tests/stateful/ (absent until plan Task 3.x lands it) ->
                NOT_TESTED with a note; an absent tier is not a gate failure.

    e2e         tests/test_end_to_end_functional.py with
                KRAKEN_RUN_END_TO_END_FUNCTIONAL=1 set for the run. The
                live-* files are NOT part of e2e (they need the live server).

    extended    Explorer campaigns (absent until plan Task 6.1) -> NOT_TESTED
                with a note, same as stateful.

    regression  Every file carrying a module-level `kraken_defect = "D<n>"`
                constant, collected at run time via
                tools/quality/list_defect_files.py (a thin CLI over the AST
                helper tools/quality/test_nodes.py; scans tests/ and
                plugins/). The list is split into import domains (see below);
                each domain is one pytest invocation, merged in the receipt.

    IMPORT DOMAINS (why some tiers run two pytest invocations):
        plugins/contour/tests/ is a Python package named `tests`
        (it has __init__.py). When a single pytest collection mixes tests/
        files with plugins/contour/tests files, `sys.modules['tests']` binds
        to the contour package and every `from tests.<x> import ...` inside
        tests/ files breaks with ModuleNotFoundError (verified: the four
        kraken_defect files that import tests.* fail collection in a mixed
        run, and the same files collect fine alone). Therefore any tier whose
        file set spans both roots is executed as one pytest invocation per
        import domain - tests/ first, then the plugin files with
        --import-mode=importlib (the convention of the `plugins` block in
        reproduce-stabilization.ps1) - and the receipts merge both JUnit
        files. Tiers that stay inside one domain (smoke, integration, gui,
        e2e, registry) are single invocations.

    CONVENTIONS (from docs/quality/reproduce-stabilization.ps1):
      * receipt.json per run carries the exact command(s) and exit code(s);
      * JUnit XML per run;
      * raw pytest output captured to pytest.txt (UTF-8); the tail is also
        echoed to the console;
      * a uuid scratch run root under .test_runtime\quality-<uuid>
        (git-ignored) holds pytest --basetemp, so the committed evidence
        dir contains only receipt + JUnit + raw log;
      * re-running a tier replaces its stale junit/log/receipt first (a
        crashed run's JUnit must never mix into the coverage comparison).
      * env vars QT_QPA_PLATFORM=offscreen, PYTHONUNBUFFERED=1, PYTHONUTF8=1,
        PYTHONIOENCODING=utf-8 are set for the runs and restored afterwards;
      * plugin-domain invocations (contour plugin files) run with PYTHONPATH
        extended by plugins\contour so their `from scripts.* import ...`
        resolves against plugins\contour\scripts (the `scripts` namespace
        package merges both); it is set per invocation and restored after,
        and is NEVER applied to the tests/ domain - the contour `tests`
        package is a regular package and would shadow the tests/ namespace
        package (PEP 420: a regular package later on sys.path wins).

    COVERAGE BASELINE SELECTION:
      1. newest working-tree docs/quality/evidence/<date>/coverage-report.json
         with <date> strictly before today;
      2. otherwise the newest git-committed coverage-report.json (any date),
         read via `git show` (needed on the day the first report was
         committed, when no strictly-earlier report exists).
    The comparison itself reuses the coverage report's own verdict code via
    tools/quality/gate_compare.py.

    PowerShell 5.1 notes (why the script looks the way it does):
      * native stderr under $ErrorActionPreference='Stop' aborts the script
        even when redirected, so every native invocation runs in an
        'Invoke-Native' scope with EAP=Continue and checks $LASTEXITCODE;
      * `python -c` with embedded double quotes mangles arguments on 5.1, so
        all Python logic lives in tools/quality/*.py modules.

    Requires Windows PowerShell 5.1+ and the repo venv (.venv\Scripts\python.exe).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('smoke', 'unit', 'integration', 'gui', 'stateful', 'e2e', 'extended', 'regression')]
    [string[]]$Tier,

    [string]$OutDir = '',

    [switch]$SkipCoverage,

    [string]$Base = 'HEAD~1'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python interpreter not found: $python" }

$today = Get-Date -Format 'yyyy-MM-dd'
$evidenceRoot = Join-Path $repo 'docs\quality\evidence'
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $evidenceRoot $today }
$OutDir = [IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

# Scratch run root (git-ignored, same convention as reproduce-stabilization.ps1):
# pytest --basetemp lives here so the committed evidence dir holds only
# receipt + JUnit + raw log.
$runRoot = Join-Path $repo ('.test_runtime\quality-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$Tier = @($Tier | Select-Object -Unique)
$gateFailed = $false
$script:failureReasons = @()

# ------------------------------------------------------------------- helpers
function Invoke-Native {
    <#
        Run a native command without EAP=Stop aborting on stderr.
        Returns the process exit code. With -LogFile, all output goes to the
        file (UTF-8); otherwise it is echoed to the console.
    #>
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [string]$LogFile = ''
    )
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

function Write-Receipt {
    param([string]$Dir, [object]$Receipt)
    $Receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Dir 'receipt.json') -Encoding UTF8
}

function Read-JUnitSummary {
    param([string]$Path)
    [xml]$xml = Get-Content -LiteralPath $Path -Encoding UTF8
    $suite = $xml.DocumentElement
    if ($suite.LocalName -eq 'testsuites') {
        $suite = @($suite.ChildNodes | Where-Object { $_.LocalName -eq 'testsuite' })[0]
    }
    $skippedIds = @()
    foreach ($case in @($suite.SelectNodes('.//testcase'))) {
        if ($null -ne $case.SelectSingleNode('skipped')) {
            $skippedIds += '{0}::{1}' -f $case.GetAttribute('classname'), $case.GetAttribute('name')
        }
    }
    [pscustomobject]@{
        Tests      = [int]$suite.GetAttribute('tests')
        Failures   = [int]$suite.GetAttribute('failures')
        Errors     = [int]$suite.GetAttribute('errors')
        Skipped    = [int]$suite.GetAttribute('skipped')
        Duration   = [double]$suite.GetAttribute('time')
        SkippedIds = $skippedIds
    }
}

function Invoke-GateTier {
    <#
        Run one tier. -Runs is a list of runs, one pytest invocation each
        (per import domain). A run is a pscustomobject with .Args (string[])
        and an optional .Env (hashtable, e.g. PYTHONPATH for the plugin
        import domain), or a plain argument array. A flat file list is a
        single invocation. Each invocation gets its own junit/log file and
        the receipt merges the results. A tier is FAIL when any invocation
        fails, when no JUnit evidence was produced, PASS when everything
        passed with zero skips, PARTIAL when everything passed but skips
        occurred.
    #>
    param(
        [string]$Name,
        [object[]]$Runs = @(),
        [string]$AbsentMessage = ''
    )
    # a flat string list is ONE pytest invocation; otherwise each element
    # is one invocation
    $runs = @($Runs)
    if ($runs.Count -gt 0 -and $runs[0] -is [string]) {
        $runs = , @($Runs)
    }
    $dir = Join-Path $OutDir "gate-$Name"
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    # a tier re-run replaces the previous evidence; never merge with stale
    # JUnit (a crashed run's junit could poison the coverage comparison)
    Get-ChildItem -LiteralPath $dir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(junit.*\.xml|pytest.*\.txt|receipt\.json)$' } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    if ($AbsentMessage) {
        $receipt = [ordered]@{
            gate = 'quality-gates'
            timestamp = (Get-Date).ToUniversalTime().ToString('o')
            tier = $Name
            command = ''
            exit_code = $null
            tests = 0
            passed = 0
            failed = 0
            errors = 0
            skipped = 0
            skipped_tests = @()
            duration_seconds = $null
            status = 'NOT_TESTED'
            note = $AbsentMessage
        }
        Write-Receipt -Dir $dir -Receipt $receipt
        Write-Host ("[gate] {0} : NOT_TESTED - {1}" -f $Name, $AbsentMessage)
        return 'NOT_TESTED'
    }

    $single = $runs.Count -le 1
    $runReceipts = @()
    $totTests = 0; $totPassed = 0; $totFailed = 0; $totErrors = 0; $totSkipped = 0
    $totDuration = [double]0; $worstExit = 0; $anyNonZero = $false; $skippedAll = @()
    $anyUnreadable = $false
    $runIndex = 0
    foreach ($run in $runs) {
        $runIndex++
        $suffix = if ($single) { '' } else { "($runIndex)" }
        if ($run -is [pscustomobject]) {
            $runArgs = @($run.Args)
            $runEnv = if ($run.Env -and $run.Env.Count -gt 0) { $run.Env } else { @{} }
        }
        else {
            $runArgs = @($run)
            $runEnv = @{}
        }
        $junit = Join-Path $dir ('junit{0}.xml' -f $suffix)
        $log = Join-Path $dir ('pytest{0}.txt' -f $suffix)
        $basetemp = Join-Path $runRoot ('basetemp-{0}{1}' -f $Name, $suffix)
        $arguments = @('-m', 'pytest') + @($runArgs) +
            @('-q', '-p', 'no:cacheprovider', '-o', 'faulthandler_timeout=60',
              "--basetemp=$basetemp", "--junitxml=$junit")
        $command = '{0} {1}' -f $python, ($arguments -join ' ')
        Write-Host ("[gate] {0}{1} : running: {2}" -f $Name, $suffix, $command)
        # per-invocation environment (e.g. PYTHONPATH for the plugin import
        # domain only - it must NOT apply to the tests/ domain, whose
        # namespace package `tests` would be shadowed by the contour
        # package); always restored, even when the run crashes
        $savedRunEnv = @{}
        foreach ($key in $runEnv.Keys) { $savedRunEnv[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        foreach ($key in $runEnv.Keys) { [Environment]::SetEnvironmentVariable($key, [string]$runEnv[$key], 'Process') }
        try {
            $exitCode = Invoke-Native -Exe $python -Arguments $arguments -LogFile $log
        }
        finally {
            foreach ($key in $savedRunEnv.Keys) { [Environment]::SetEnvironmentVariable($key, $savedRunEnv[$key], 'Process') }
        }

        $tests = 0; $passed = 0; $failed = 0; $errors = 0; $skipped = 0
        $skippedIds = @(); $duration = $null; $unreadable = $false
        if (Test-Path -LiteralPath $junit) {
            try {
                $j = Read-JUnitSummary -Path $junit
                $tests = $j.Tests
                $failed = $j.Failures + $j.Errors
                $errors = $j.Errors
                $skipped = $j.Skipped
                $passed = [Math]::Max(0, $j.Tests - $j.Failures - $j.Errors - $j.Skipped)
                $skippedIds = @($j.SkippedIds)
                $duration = $j.Duration
            }
            catch { $unreadable = $true }
        }
        else {
            $unreadable = $true
        }
        if ($unreadable) { $anyUnreadable = $true }

        $totTests += $tests; $totPassed += $passed; $totFailed += $failed
        $totErrors += $errors; $totSkipped += $skipped
        $totDuration += ([double]$duration); $skippedAll += $skippedIds
        if ($exitCode -ne 0) {
            $anyNonZero = $true
            # Windows crash codes are negative (e.g. -1073741819
            # STATUS_ACCESS_VIOLATION) and always rank worse than positive
            # pytest exit codes; among crashes the most negative wins
            if ($worstExit -eq 0) {
                $worstExit = $exitCode
            }
            elseif ($exitCode -lt 0 -and ($worstExit -gt 0 -or $exitCode -lt $worstExit)) {
                $worstExit = $exitCode
            }
            elseif ($worstExit -gt 0 -and $exitCode -gt $worstExit) {
                $worstExit = $exitCode
            }
        }

        $note = ''
        if ($unreadable) { $note = 'no readable JUnit produced (the suite may have crashed)' }
        elseif ($exitCode -eq 1) { $note = 'tests failed' }
        elseif ($exitCode -eq 2) { $note = 'run interrupted (collection error/crash)' }
        elseif ($exitCode -eq 3) { $note = 'internal pytest error' }
        elseif ($exitCode -eq 5) { $note = 'no tests collected; the suite could not run' }
        elseif ($exitCode -lt 0) { $note = 'pytest process crashed (exit code {0})' -f $exitCode }
        elseif ($exitCode -ne 0) { $note = 'unexpected pytest exit code {0}' -f $exitCode }
        if (Test-Path -LiteralPath $log) {
            Get-Content -LiteralPath $log -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
        }
        $runReceipts += [ordered]@{
            command = $command
            exit_code = $exitCode
            tests = $tests
            passed = $passed
            failed = $failed
            errors = $errors
            skipped = $skipped
            skipped_tests = $skippedIds
            duration_seconds = $duration
            junit = (Split-Path -Leaf $junit)
            log = (Split-Path -Leaf $log)
            note = $note
        }
        Write-Host ("[gate] {0}{1} : exit {2}; {3} passed, {4} failed, {5} skipped; {6}s" -f `
            $Name, $suffix, $exitCode, $passed, $failed, $skipped, $duration)
    }

    $status = 'FAIL'
    $note = ''
    if ($anyUnreadable) {
        $note = 'no readable JUnit produced (the suite may have crashed)'
    }
    elseif ($anyNonZero) {
        $bad = @($runReceipts | Where-Object { $_.exit_code -ne 0 })
        $note = ('invocation(s) failed: ' + (($bad | ForEach-Object { '{0} ({1})' -f $_.exit_code, $_.note }) -join '; '))
    }
    elseif ($totTests -eq 0) {
        $note = 'pytest exited 0 but produced no JUnit test evidence; the mandatory suite did not run'
    }
    elseif ($totSkipped -gt 0) {
        $status = 'PARTIAL'
        $note = '{0} test(s) skipped; skipped tests are never counted as pass' -f $totSkipped
    }
    else {
        $status = 'PASS'
    }

    $receipt = [ordered]@{
        gate = 'quality-gates'
        timestamp = (Get-Date).ToUniversalTime().ToString('o')
        tier = $Name
        command = @($runReceipts | ForEach-Object { $_.command })
        exit_code = $worstExit
        tests = $totTests
        passed = $totPassed
        failed = $totFailed
        errors = $totErrors
        skipped = $totSkipped
        skipped_tests = @($skippedAll)
        duration_seconds = $totDuration
        status = $status
        note = $note
        runs = $runReceipts
    }
    Write-Receipt -Dir $dir -Receipt $receipt
    Write-Host ("[gate] {0} : {1} (worst exit {2}; {3} passed, {4} failed, {5} skipped; {6}s)" -f `
        $Name, $status, $worstExit, $totPassed, $totFailed, $totSkipped, $totDuration)
    if ($status -eq 'FAIL') { $script:failureReasons += "{0} tier: FAIL ({1})" -f $Name, $note }
    return $status
}

function Find-BaselineReport {
    # 1) newest working-tree report in a date directory strictly before today
    $dirs = Get-ChildItem -LiteralPath $evidenceRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' -and $_.Name -lt $today } |
        Sort-Object Name -Descending
    foreach ($d in $dirs) {
        $p = Join-Path $d.FullName 'coverage-report.json'
        if (Test-Path -LiteralPath $p) { return $p }
    }
    # 2) newest git-committed report (any date), via git show
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $tracked = @(& git ls-files -- 'docs/quality/evidence' 2>$null)
    }
    finally { $ErrorActionPreference = $prev }
    $tracked = @($tracked | Where-Object { $_ -match '^docs/quality/evidence/\d{4}-\d{2}-\d{2}/coverage-report\.json$' } | Sort-Object -Descending)
    if ($tracked.Count -gt 0) {
        $dest = Join-Path $env:TEMP ('kraken-gate-base-' + [guid]::NewGuid().ToString('N') + '.json')
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & git show ("HEAD:{0}" -f $tracked[0]) 2>$null | Out-File -LiteralPath $dest -Encoding utf8
            $code = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $prev }
        if ($code -eq 0 -and (Test-Path -LiteralPath $dest)) { return $dest }
        Remove-Item -LiteralPath $dest -ErrorAction SilentlyContinue
    }
    return $null
}

# ---------------------------------------------------------------- suite maps
# Files excluded from the unit tier: every env-gated file (live/UI/opt-in) and
# every PostgreSQL-gated file (they belong to the integration tier or a live
# environment). See the header for the detection method.
$unitIgnores = @(
    # live / UI / opt-in env gates (never run without their environment)
    'tests/test_live_server_plugin_chain.py',        # KRAKEN_LIVE_WORKER_CONFIG
    'tests/test_live_server_worker_acceptance.py',   # KRAKEN_LIVE_WORKER_CONFIG
    'tests/test_live_server_worker_faults.py',       # gated via the acceptance plugin
    'tests/test_live_ui_workflow.py',                # KRAKEN_UI_TEST_CONFIG
    'tests/test_neuralimage_headless_training.py',   # KRAKEN_RUN_TRAINING_LIVE
    'tests/test_pipeline_live_contour.py',           # KRAKEN_RUN_PIPELINE_LIVE
    'tests/test_pipeline_live_neural_model.py',      # KRAKEN_RUN_MODEL_FUNCTIONAL
    'tests/test_end_to_end_functional.py',           # KRAKEN_RUN_END_TO_END_FUNCTIONAL (e2e tier)
    # PostgreSQL-gated (database fixture / KRAKEN_TEST_POSTGRES_URL) -> integration tier
    'tests/test_admin_project_workflows.py',
    'tests/test_agent_execution_schema_upgrade.py',
    'tests/test_agent_queue_admission.py',
    'tests/test_cif_media_upload.py',
    'tests/test_device_candidate_postgres.py',
    'tests/test_electrical_postgres.py',
    'tests/test_legacy_work_order_postgres.py',
    'tests/test_live_catalog_acceptance.py',
    'tests/test_model_input_postgres.py',
    'tests/test_multilayer_postgres.py',
    'tests/test_performer_store.py',
    'tests/test_pipeline_postgres.py',
    'tests/test_pipeline_stitch_failures.py',
    'tests/test_pipeline_stitch_postgres.py',
    'tests/test_planning_named_selection.py',
    'tests/test_planning_postgres.py',
    'tests/test_planning_required_arguments.py',
    'tests/test_quality_analysis_postgres.py',
    'tests/test_quality_postgres.py',
    'tests/test_registered_principal_before_session.py',
    'tests/test_remote_stitch_dialog.py',
    'tests/test_server_analysis_acceptance.py',
    'tests/test_server_analysis_workflow.py',
    'tests/test_server_backup.py',
    'tests/test_server_managed_upload_preflight.py',
    'tests/test_server_stitch_workflow.py',
    'tests/test_server_vector_stitch.py',
    'tests/test_stitch_plan_postgres.py',
    'tests/test_targeted_machine_lease.py',
    'tests/test_worker_runtime_postgres.py',
    'tests/test_work_plan_analytics_postgres.py',
    'tests/test_work_templates_postgres.py'
)

# PostgreSQL-gated suites (integration tier): all users of the `database`
# fixture / KRAKEN_TEST_POSTGRES_URL. Self-skip without the env var.
$integrationFiles = @(
    'tests/test_admin_project_workflows.py',
    'tests/test_agent_execution_schema_upgrade.py',
    'tests/test_agent_queue_admission.py',
    'tests/test_cif_media_upload.py',
    'tests/test_device_candidate_postgres.py',
    'tests/test_electrical_postgres.py',
    'tests/test_legacy_work_order_postgres.py',
    'tests/test_live_catalog_acceptance.py',
    'tests/test_model_input_postgres.py',
    'tests/test_multilayer_postgres.py',
    'tests/test_performer_store.py',
    'tests/test_pipeline_postgres.py',
    'tests/test_pipeline_stitch_failures.py',
    'tests/test_pipeline_stitch_postgres.py',
    'tests/test_planning_named_selection.py',
    'tests/test_planning_postgres.py',
    'tests/test_planning_required_arguments.py',
    'tests/test_quality_analysis_postgres.py',
    'tests/test_quality_postgres.py',
    'tests/test_registered_principal_before_session.py',
    'tests/test_remote_stitch_dialog.py',
    'tests/test_server_analysis_acceptance.py',
    'tests/test_server_analysis_workflow.py',
    'tests/test_server_backup.py',
    'tests/test_server_managed_upload_preflight.py',
    'tests/test_server_stitch_workflow.py',
    'tests/test_server_vector_stitch.py',
    'tests/test_stitch_plan_postgres.py',
    'tests/test_targeted_machine_lease.py',
    'tests/test_worker_runtime_postgres.py',
    'tests/test_work_plan_analytics_postgres.py',
    'tests/test_work_templates_postgres.py'
)

# Offscreen pytest-qt suites (gui tier): the proven `qt` block of
# reproduce-stabilization.ps1 plus the local layer_* Qt suites.
$guiFiles = @(
    'tests/test_work_planning_dialog.py',
    'tests/test_quality_dialog.py',
    'tests/test_multilayer_dialog.py',
    'tests/test_electrical_dialog.py',
    'tests/test_computed_stitch_dialog.py',
    'tests/test_analysis_qt_ui.py',
    'tests/test_local_analysis_map_ui.py',
    'tests/test_work_templates_dialog.py',
    'tests/test_vector_stitch_dialog.py',
    'tests/test_remaining_qt_acceptance.py',
    'tests/test_layer_creation_dialog.py',
    'tests/test_layer_management_features.py',
    'tests/test_layer_menu_permissions.py',
    'tests/test_layer_menu_profiling.py',
    'tests/test_layer_pipeline_dialog.py',
    'tests/test_layer_workflow_repairs.py'
)

# ------------------------------------------------------------------------ run
$savedEnv = @{}
foreach ($key in @('QT_QPA_PLATFORM', 'PYTHONUNBUFFERED', 'PYTHONUTF8', 'PYTHONIOENCODING', 'KRAKEN_RUN_END_TO_END_FUNCTIONAL')) {
    $savedEnv[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}

# PYTHONPATH for the plugin import domain only: the contour plugin tests do
# `from scripts.<x> import ...`, and `scripts` is a namespace package whose
# repo-root portion does not contain those modules (they live in
# plugins/contour/scripts/). It is applied per invocation (see Invoke-GateTier
# -RunEnv), never to the tests/ domain.
$prevPp = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$contourPp = @((Join-Path $repo 'plugins\contour'))
if ($prevPp) { $contourPp += $prevPp }
$contourPp = $contourPp -join ';'

try {
    $env:QT_QPA_PLATFORM = 'offscreen'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'

    Write-Host ''
    Write-Host ('Kraken quality gate | tiers: {0} | evidence: {1}' -f ($Tier -join ', '), $OutDir)
    Write-Host ''

    # 1) registry enforcement - always first, failure aborts the gate
    $regStatus = Invoke-GateTier -Name 'registry' -Runs @(@('tests/test_quality_registry.py'))
    if ($regStatus -eq 'FAIL') {
        $gateFailed = $true
        $script:failureReasons = @('registry enforcement failed; the gate aborts regardless of tier')
        Write-Host '[gate] REGISTRY ENFORCEMENT FAILED - aborting, no tiers will run'
    }
    else {
        Write-Host '[gate] registry enforcement: OK'
    }

    # 2) requested tiers; a FAILing tier fails the gate (a mandatory suite
    #    that could not run to a green result)
    if (-not $gateFailed) {
        foreach ($t in $Tier) {
            $st = $null
            switch ($t) {
                'smoke' {
                    $st = Invoke-GateTier -Name 'smoke' -Runs @(@('tests/test_desktop_runtime_smoke.py'))
                }
                'unit' {
                    $unitArgs = @('tests')
                    foreach ($f in $unitIgnores) { $unitArgs += "--ignore=$f" }
                    # two import domains: tests/ first, then the contour
                    # plugin with plugins/contour on PYTHONPATH (see header)
                    $st = Invoke-GateTier -Name 'unit' -Runs @(
                        [pscustomobject]@{ Args = $unitArgs },
                        [pscustomobject]@{
                            Args = @('plugins/contour/tests/unit', '--import-mode=importlib')
                            Env = @{ PYTHONPATH = $contourPp }
                        }
                    )
                }
                'integration' {
                    $st = Invoke-GateTier -Name 'integration' -Runs @(, $integrationFiles)
                }
                'gui' {
                    $st = Invoke-GateTier -Name 'gui' -Runs @(, $guiFiles)
                }
                'stateful' {
                    $hasStateful = $false
                    if (Test-Path -LiteralPath (Join-Path $repo 'tests\stateful')) {
                        $hasStateful = @((Get-ChildItem -LiteralPath (Join-Path $repo 'tests\stateful') -Filter 'test_*.py' -Recurse -ErrorAction SilentlyContinue)).Count -gt 0
                    }
                    $msg = if ($hasStateful) { '' } else { 'tests/stateful/ does not exist yet (plan Task 3.x); an absent tier is not a gate failure' }
                    $st = Invoke-GateTier -Name 'stateful' -Runs @(@('tests/stateful')) -AbsentMessage $msg
                }
                'e2e' {
                    $env:KRAKEN_RUN_END_TO_END_FUNCTIONAL = '1'
                    try {
                        $st = Invoke-GateTier -Name 'e2e' -Runs @(@('tests/test_end_to_end_functional.py'))
                    }
                    finally {
                        [Environment]::SetEnvironmentVariable('KRAKEN_RUN_END_TO_END_FUNCTIONAL', $savedEnv['KRAKEN_RUN_END_TO_END_FUNCTIONAL'], 'Process')
                    }
                }
                'extended' {
                    $hasExplorer = Test-Path -LiteralPath (Join-Path $repo 'tools\explorer\explorer.py')
                    $msg = if ($hasExplorer) { '' } else { 'explorer campaigns do not exist yet (plan Task 6.1); an absent tier is not a gate failure' }
                    $st = Invoke-GateTier -Name 'extended' -Runs @(@()) -AbsentMessage $msg
                }
                'regression' {
                    $defectFiles = @()
                    $prev = $ErrorActionPreference
                    $ErrorActionPreference = 'Continue'
                    try {
                        $defectFiles = @(& $python (Join-Path $repo 'tools\quality\list_defect_files.py') 2>$null)
                        $listExit = $LASTEXITCODE
                    }
                    finally { $ErrorActionPreference = $prev }
                    if ($listExit -ne 0) {
                        throw "tools/quality/list_defect_files.py failed with exit code $listExit; the regression tier cannot be built"
                    }
                    $defectFiles = @($defectFiles | Where-Object { $_ -and $_.Trim() } | ForEach-Object { $_.Trim() })
                    # split into import domains: tests/ first, then plugin
                    # files (the plugin run gets plugins/contour on PYTHONPATH)
                    $testFiles = @($defectFiles | Where-Object { $_ -like 'tests/*' })
                    $pluginFiles = @($defectFiles | Where-Object { $_ -notlike 'tests/*' })
                    $runs = @()
                    if ($testFiles.Count -gt 0) {
                        $runs += [pscustomobject]@{ Args = $testFiles }
                    }
                    if ($pluginFiles.Count -gt 0) {
                        $runs += [pscustomobject]@{
                            Args = (@($pluginFiles) + @('--import-mode=importlib'))
                            Env = @{ PYTHONPATH = $contourPp }
                        }
                    }
                    $msg = if ($runs.Count -eq 0) { 'no files carry a kraken_defect marker yet; nothing to run' } else { '' }
                    $st = Invoke-GateTier -Name 'regression' -Runs $runs -AbsentMessage $msg
                }
            }
            if ($st -eq 'FAIL') { $gateFailed = $true }
        }
    }

    # 3) coverage report + regression comparison
    if (-not $gateFailed -and -not $SkipCoverage) {
        Write-Host ''
        $junits = @(Get-ChildItem -LiteralPath $OutDir -Recurse -File -Filter 'junit*.xml' -ErrorAction SilentlyContinue |
            ForEach-Object { $_.FullName })
        $junitArgs = @()
        foreach ($j in $junits) { $junitArgs += $j }

        $baseReport = Find-BaselineReport
        $compareDoc = Join-Path $OutDir 'coverage-compare.json'
        if ($null -eq $baseReport) {
            [ordered]@{ status = 'SKIPPED'; reason = 'no previous committed coverage report found'; base_report = $null; regressions = @() } |
                ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $compareDoc -Encoding UTF8
            Write-Host '[gate] coverage comparison: SKIPPED (no previous coverage report to compare against)'
        }
        else {
            $baseCopy = Join-Path $OutDir 'coverage-base-report.json'
            [IO.File]::Copy($baseReport, $baseCopy, $true)
            if ($baseReport -ne $baseCopy) { Remove-Item -LiteralPath $baseReport -ErrorAction SilentlyContinue }
            Write-Host ('[gate] coverage baseline: {0}' -f $baseCopy)
            # both CLIs take the JUnit list after a single flag (argparse nargs='*')
            $cmpExit = Invoke-Native -Exe $python -Arguments (@((Join-Path $repo 'tools\quality\gate_compare.py'), '--registry', 'docs/quality/registry', '--base-report', $baseCopy, '--out', $compareDoc, '--new-junit') + $junitArgs)
            if ($cmpExit -eq 1) {
                $gateFailed = $true
                $doc = Get-Content -LiteralPath $compareDoc -Raw | ConvertFrom-Json
                $script:failureReasons += 'coverage regression (PASS before, FAIL now): {0}' -f ($doc.regressions -join ', ')
            }
            elseif ($cmpExit -ne 0) {
                $gateFailed = $true
                $script:failureReasons += "coverage comparison failed with exit code $cmpExit"
            }
            else {
                $doc = Get-Content -LiteralPath $compareDoc -Raw | ConvertFrom-Json
                if ($doc.newly_not_tested) {
                    Write-Host ('[gate] informational: newly NOT_TESTED (no gate failure): {0}' -f ($doc.newly_not_tested -join ', '))
                }
            }
            $repExit = Invoke-Native -Exe $python -Arguments (@((Join-Path $repo 'tools\quality\coverage_report.py'), '--registry', 'docs/quality/registry', '--base', $Base, '--out', $OutDir, '--junit') + $junitArgs)
            if ($repExit -ne 0) {
                $gateFailed = $true
                $script:failureReasons += "coverage report generation failed with exit code $repExit"
            }
            else {
                Write-Host ('[gate] coverage report written to {0}' -f $OutDir)
            }
        }
    }
    elseif ($SkipCoverage) {
        Write-Host ''
        Write-Host '[gate] coverage comparison: SKIPPED (-SkipCoverage)'
    }
}
finally {
    foreach ($key in $savedEnv.Keys) {
        [Environment]::SetEnvironmentVariable($key, $savedEnv[$key], 'Process')
    }
}

# --------------------------------------------------------------------- verdict
Write-Host ''
Write-Host '==================== gate summary ===================='
foreach ($t in @('registry') + $Tier) {
    $receiptPath = Join-Path $OutDir "gate-$t\receipt.json"
    if (Test-Path -LiteralPath $receiptPath) {
        $r = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        $counts = '{0}/{1}/{2}' -f $r.passed, $r.failed, $r.skipped
        Write-Host ('  {0,-13} {1,-10} passed/failed/skipped={2,-12} exit={3}' -f $t, $r.status, $counts, $r.exit_code)
    }
}
if ($gateFailed) {
    Write-Host ''
    Write-Host 'GATE RESULT: FAIL'
    foreach ($reason in $script:failureReasons) { Write-Host "  - $reason" }
    exit 1
}
Write-Host ''
Write-Host 'GATE RESULT: PASS'
exit 0
