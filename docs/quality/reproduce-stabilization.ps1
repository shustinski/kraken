param(
    [ValidateSet('backup', 'server', 'server-domain', 'live-catalog', 'agent', 'qt', 'plugins', 'statistics', 'd1', 'd4', 'policy', 'schema-upgrade', 'hub-resume')]
    [string]$Block = 'd4',
    [string]$PostgresBin = 'C:\PostgreSQL\16\pgsql\bin'
)
$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location -LiteralPath $repository
$python = Join-Path $repository '.venv\Scripts\python.exe'
$runRoot = Join-Path $repository ('.test_runtime\quality-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $runRoot | Out-Null
$groups = @{
    'backup' = @('tests/test_server_backup.py', 'tests/test_server_backup_gateway.py')
    'server' = @('tests/test_registered_principal_before_session.py', 'tests/test_server_analysis_workflow.py', 'tests/test_server_analysis_acceptance.py', 'tests/test_server_stitch_workflow.py', 'tests/test_remote_stitch_dialog.py', 'tests/test_server_extended_api.py', 'tests/test_server_route_contracts.py', 'tests/test_server_workspace_inputs.py', 'tests/test_server_asset_access.py', 'tests/test_blob_gateway.py', 'tests/test_remote_catalog.py')
    'server-domain' = @('tests/test_agent_queue_admission.py', 'tests/test_server_services.py', 'tests/test_server_vector_stitch.py', 'tests/test_quality_postgres.py', 'tests/test_multilayer_postgres.py', 'tests/test_electrical_postgres.py', 'tests/test_work_templates_postgres.py', 'tests/test_work_plan_analytics_postgres.py', 'tests/test_planning_postgres.py')
    'live-catalog' = @('tests/test_live_catalog_acceptance.py')
    'agent' = @((Get-ChildItem tests/test_agent*.py).FullName) + @('tests/test_work_plugin_launch.py', 'tests/test_karakal_agent_headless.py')
    'qt' = @('tests/test_work_planning_dialog.py', 'tests/test_quality_dialog.py', 'tests/test_multilayer_dialog.py', 'tests/test_electrical_dialog.py', 'tests/test_computed_stitch_dialog.py', 'tests/test_analysis_qt_ui.py', 'tests/test_local_analysis_map_ui.py', 'tests/test_work_templates_dialog.py', 'tests/test_vector_stitch_dialog.py', 'tests/test_remaining_qt_acceptance.py')
    'plugins' = @('tests/test_csliser_functional.py', 'plugins/csliser/tests', 'plugins/krona/tests/test_explicit_connectivity.py', 'plugins/krona/tests/test_quality_qt.py', 'plugins/karakal/tests/unit/test_headless_analysis.py', 'tests/test_contour_headless_worker.py', 'tests/test_headless_cif_deployment.py', '--import-mode=importlib')
    'statistics' = @('tests/test_statistics_reports.py', 'tests/test_remaining_qt_acceptance.py::test_statistics_calculate_and_export_buttons_use_real_history')
    'd1' = @('tests/test_project_create_idempotency.py', 'tests/test_create_project_dialog_retry.py', 'tests/test_hub_embedded_service.py')
    'd4' = @('tests/test_karakal_agent_headless.py', 'tests/test_agent_execution_modes.py')
    'schema-upgrade' = @('tests/test_agent_execution_schema_upgrade.py')
    'hub-resume' = @('tests/test_hub_resume_remote_failure.py', 'tests/test_hub_manager_app.py')
    'policy' = @('tests/test_server_administration.py', 'tests/test_registered_principal_before_session.py')
}
$savedEnvironment = @{}
foreach ($key in @('QT_QPA_PLATFORM', 'PYTHONUNBUFFERED', 'PYTHONUTF8', 'PYTHONIOENCODING', 'KRAKEN_TEST_POSTGRES_URL', 'KRAKEN_TEST_PG_BIN')) {
    $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
$started = $false
$testExitCode = 1
$arguments = @('-m', 'pytest') + $groups[$Block] + @('-v', '-p', 'no:cacheprovider', '-o', 'faulthandler_timeout=60', "--basetemp=$runRoot\pytest", "--junitxml=$runRoot\junit.xml")
try {
    $env:QT_QPA_PLATFORM = 'offscreen'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    if ($Block -in @('backup', 'server', 'server-domain', 'live-catalog', 'agent', 'policy', 'schema-upgrade')) {
        # This is always a new cluster, never the configured working database.
        $cluster = Join-Path $runRoot 'postgres'
        & (Join-Path $PostgresBin 'initdb.exe') -D $cluster -U kraken_audit --auth=trust --encoding=UTF8 --no-locale *> (Join-Path $runRoot 'initdb.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Temporary initdb failed; see initdb.txt' }
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
        $listener.Start()
        $port = $listener.LocalEndpoint.Port
        $listener.Stop()
        & (Join-Path $PostgresBin 'pg_ctl.exe') -D $cluster -l (Join-Path $runRoot 'postgres.txt') -o "-h 127.0.0.1 -p $port" -w start
        if ($LASTEXITCODE -ne 0) { throw 'Temporary PostgreSQL failed to start' }
        $started = $true
        $env:KRAKEN_TEST_POSTGRES_URL = "postgresql+psycopg://kraken_audit@127.0.0.1:$port/postgres?connect_timeout=5"
        $env:KRAKEN_TEST_PG_BIN = $PostgresBin
    }
    & $python @arguments *> (Join-Path $runRoot 'pytest.txt')
    $testExitCode = $LASTEXITCODE
} finally {
    if ($started) { & (Join-Path $PostgresBin 'pg_ctl.exe') -D $cluster -w stop }
    foreach ($key in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($key, $savedEnvironment[$key], 'Process')
    }
    @{
        block = $Block; python = $python; arguments = $arguments; exit_code = $testExitCode
        git_head = (& git rev-parse HEAD); results = $runRoot
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'receipt.json') -Encoding utf8
    Write-Output "Evidence: $runRoot"
}
exit $testExitCode
