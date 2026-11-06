#requires -Version 5.1
Set-StrictMode -Version Latest

$script:Serializer = $null

function Get-JsonSerializer {
    if ($null -eq $script:Serializer) {
        Add-Type -AssemblyName System.Web.Extensions
        $serializer = New-Object System.Web.Script.Serialization.JavaScriptSerializer
        $serializer.MaxJsonLength = [int]::MaxValue
        $serializer.RecursionLimit = 64
        $script:Serializer = $serializer
    }
    return $script:Serializer
}

function ConvertTo-Serializable {
    param($Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [psobject]) {
        $base = $Value.PSObject.BaseObject
        if ($base -isnot [psobject]) { return ConvertTo-Serializable $base }
    }
    if ($Value -is [string]) { return [string]$Value }
    if ($Value -is [bool]) { return [bool]$Value }
    if ($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]) { return $Value }
    if ($Value -is [System.Collections.IDictionary]) {
        $dict = New-Object 'System.Collections.Generic.Dictionary[string,object]'
        foreach ($key in @($Value.Keys)) {
            $dict[[string]$key] = ConvertTo-Serializable $Value[$key]
        }
        return $dict
    }
    $isList = $Value -is [System.Array] -or $Value -is [System.Collections.ArrayList] -or $Value.GetType().FullName.StartsWith('System.Collections.Generic.List')
    if ($isList) {
        $list = New-Object System.Collections.Generic.List[object]
        foreach ($item in @($Value)) {
            $list.Add((ConvertTo-Serializable $item))
        }
        return $list
    }
    return [string]$Value
}

function Test-MapKey {
    param($Map, [string]$Name)
    if ($null -eq $Map) { return $false }
    if ($Map -is [System.Collections.IDictionary]) { return [bool]$Map.ContainsKey($Name) }
    return $null -ne $Map.PSObject.Properties[$Name]
}

function Get-MapValue {
    param($Map, [string]$Name)
    if ($null -eq $Map) { return $null }
    if ($Map -is [System.Collections.IDictionary]) { return $Map[$Name] }
    $property = $Map.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function ConvertTo-OfflineJson {
    param($Value)
    $serializer = Get-JsonSerializer
    return $serializer.Serialize((ConvertTo-Serializable $Value))
}

function Read-OfflineJson {
    param([string]$Path)
    $text = [System.IO.File]::ReadAllText($Path)
    return ($text | ConvertFrom-Json)
}

function Write-OfflineJson {
    param([string]$Path, $Value)
    $json = ConvertTo-Json -InputObject $Value -Depth 20 -Compress
    [System.IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding $false))
}

function Write-OfflineLine {
    param([string]$Message)
    [Console]::Out.WriteLine($Message)
}

function Stop-Offline {
    param([string]$Message)
    throw $Message
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-RelativePath {
    param([string]$Root, [string]$FullName)
    $base = [System.IO.Path]::GetFullPath($Root)
    if (-not $base.EndsWith('\')) { $base += '\' }
    $full = [System.IO.Path]::GetFullPath($FullName)
    if (-not $full.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)) {
        Stop-Offline "OFFLINE_ERROR: failed`nPath is outside the transfer: $full"
    }
    return ($full.Substring($base.Length) -replace '\\', '/')
}

function Invoke-Native {
    param(
        [string]$File,
        [string[]]$Arguments,
        [string]$ErrorCode
    )
    $output = & $File @Arguments 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Stop-Offline "$ErrorCode`n$File $($Arguments -join ' ')`n$output"
    }
    return $output
}

function Get-GitOutput {
    param([string]$Repo, [string[]]$Arguments)
    $output = & git -C $Repo @Arguments 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Stop-Offline "OFFLINE_ERROR: failed`ngit $($Arguments -join ' ')`n$output"
    }
    return $output.Trim()
}

function Test-WorkspaceProject {
    param([string]$Repo)
    $project = Join-Path $Repo 'pyproject.toml'
    if (-not (Test-Path -LiteralPath $project)) { return $false }
    $text = [System.IO.File]::ReadAllText($project)
    return $text -match '(?m)^\[tool\.uv\.workspace\]'
}

function Get-PythonRequires {
    param([string]$Repo, [string]$ScriptRoot)
    $project = Join-Path $Repo 'pyproject.toml'
    if (Test-Path -LiteralPath $project) {
        $text = [System.IO.File]::ReadAllText($project)
        $match = [regex]::Match($text, 'requires-python\s*=\s*"([^"]+)"')
        if ($match.Success) { return $match.Groups[1].Value }
    }
    $profile = Join-Path $ScriptRoot 'sync-profile.json'
    if (Test-Path -LiteralPath $profile) {
        $parsed = Read-OfflineJson $profile
        if (Test-MapKey $parsed 'python_requires') { return [string](Get-MapValue $parsed 'python_requires') }
    }
    return '>=3.14'
}

function Get-WorkspaceSyncArgs {
    param([string]$Repo, [string]$ScriptRoot)
    if (-not (Test-WorkspaceProject $Repo)) { return @() }
    $profile = Join-Path $ScriptRoot 'sync-profile.json'
    if (Test-Path -LiteralPath $profile) {
        $parsed = Read-OfflineJson $profile
        if (Test-MapKey $parsed 'workspace_sync_args') {
            return @(Get-MapValue $parsed 'workspace_sync_args')
        }
    }
    return @('--all-packages', '--all-extras', '--all-groups')
}

function Get-LocalPython {
    $candidates = @(
        @('py', '-3.14', '-c', 'import sys; print(sys.executable); print(sys.version.split()[0])'),
        @('python', '-c', 'import sys; print(sys.executable); print(sys.version.split()[0])')
    )
    foreach ($candidate in $candidates) {
        $file = $candidate[0]
        $pythonArguments = @()
        if ($candidate.Length -gt 1) { $pythonArguments = $candidate[1..($candidate.Length - 1)] }
        $command = Get-Command $file -ErrorAction SilentlyContinue
        if ($null -eq $command) { continue }
        $output = & $file @pythonArguments 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { continue }
        $lines = @($output -split "`r?`n" | Where-Object { $_ -and $_.Trim() -ne '' })
        if ($lines.Count -ge 2) {
            return @{ Executable = $lines[0].Trim(); Version = $lines[1].Trim() }
        }
    }
    Stop-Offline "OFFLINE_ERROR: python-incompatible`nPython 3.14 was not found. Install it or export with -ToolMode Bundle."
}

function Test-PythonCompatible {
    param([string]$Version, [string]$Requires)
    if ($Requires -notmatch '^>=(\d+)\.(\d+)$') {
        Stop-Offline "OFFLINE_ERROR: invalid-manifest`nOnly requires-python of the form >=X.Y is supported. Found: $Requires"
    }
    $needMajor = [int]$Matches[1]
    $needMinor = [int]$Matches[2]
    if ($Version -notmatch '^(\d+)\.(\d+)') {
        Stop-Offline "OFFLINE_ERROR: python-incompatible`nCould not parse Python version '$Version'."
    }
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    $compatible = ($major -gt $needMajor) -or (($major -eq $needMajor) -and ($minor -ge $needMinor))
    if (-not $compatible) {
        Stop-Offline "OFFLINE_ERROR: python-incompatible`nThis transfer requires Python $Requires. This computer has $Version."
    }
}

function Get-NormalizedArchitecture {
    switch ($env:PROCESSOR_ARCHITECTURE) {
        'AMD64' { return 'amd64' }
        'ARM64' { return 'arm64' }
        'x86' { return 'x86' }
        default { return ([string]$env:PROCESSOR_ARCHITECTURE).ToLowerInvariant() }
    }
}

function Import-VisualStudioEnvironment {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cl) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { return }
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $install) { return }
    $vcvars = Join-Path $install.Trim() 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) { return }
    $captured = & cmd.exe /c "`"$vcvars`" >nul && set" 2>$null
    foreach ($line in @($captured)) {
        $text = [string]$line
        $split = $text.IndexOf('=')
        if ($split -lt 1) { continue }
        $name = $text.Substring(0, $split)
        $value = $text.Substring($split + 1)
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

function Invoke-UvCommand {
    param(
        [string]$File,
        [string[]]$Arguments,
        [string]$Cache,
        [string]$ProjectEnvironment,
        [bool]$Offline,
        [string]$PythonInstallDir,
        [string]$Proxy
    )
    $names = @(
        'UV_CACHE_DIR', 'UV_PROJECT_ENVIRONMENT', 'UV_OFFLINE', 'UV_PYTHON_DOWNLOADS',
        'UV_PYTHON_INSTALL_DIR', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'UV_LINK_MODE'
    )
    $saved = @{}
    foreach ($name in $names) {
        $saved[$name] = [System.Environment]::GetEnvironmentVariable($name, 'Process')
    }
    try {
        [System.Environment]::SetEnvironmentVariable('UV_CACHE_DIR', $Cache, 'Process')
        [System.Environment]::SetEnvironmentVariable('UV_LINK_MODE', 'copy', 'Process')
        [System.Environment]::SetEnvironmentVariable('UV_PYTHON_DOWNLOADS', 'never', 'Process')
        if ($ProjectEnvironment) {
            [System.Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $ProjectEnvironment, 'Process')
        } else {
            [System.Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $null, 'Process')
        }
        if ($Offline) {
            [System.Environment]::SetEnvironmentVariable('UV_OFFLINE', '1', 'Process')
        } else {
            [System.Environment]::SetEnvironmentVariable('UV_OFFLINE', $null, 'Process')
        }
        if ($PythonInstallDir) {
            [System.Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_DIR', $PythonInstallDir, 'Process')
        } else {
            [System.Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_DIR', $null, 'Process')
        }
        foreach ($proxyName in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')) {
            if ($Proxy) {
                [System.Environment]::SetEnvironmentVariable($proxyName, $Proxy, 'Process')
            } else {
                [System.Environment]::SetEnvironmentVariable($proxyName, $null, 'Process')
            }
        }
        $script:LastUvOutput = & $File @Arguments 2>&1 | Out-String
        $script:LastUvExit = $LASTEXITCODE
    } finally {
        foreach ($name in $names) {
            [System.Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
        }
    }
}

function Get-TorchSpec {
    param([string]$Repo)
    $lock = Join-Path $Repo 'uv.lock'
    if (-not (Test-Path -LiteralPath $lock)) { return $null }
    $text = [System.IO.File]::ReadAllText($lock)
    $match = [regex]::Match($text, 'name = "torch"\s+version = "([^"]+)"')
    if (-not $match.Success) { return $null }
    return $match.Groups[1].Value
}

function Assert-TorchDistribution {
    param(
        [string]$ExpectedVersion,
        [string]$Cache,
        [string]$Environment
    )
    if (-not $ExpectedVersion) { return }
    if ($ExpectedVersion -like '*+cpu*') {
        Stop-Offline "OFFLINE_ERROR: torch-cpu`nThe lockfile pins a CPU PyTorch build ($ExpectedVersion). Kraken requires the CUDA build declared in plugins/neuralimage."
    }
    $needle = "torch-$ExpectedVersion"
    $wheels = @()
    if (Test-Path -LiteralPath $Cache) {
        $wheels = @(Get-ChildItem -LiteralPath $Cache -Recurse -Filter '*.whl' -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like 'torch-*' })
    }
    $cpu = @($wheels | Where-Object { $_.Name -like '*+cpu*' })
    if ($cpu.Count -gt 0 -and -not ($wheels | Where-Object { $_.Name -like "$needle*" })) {
        Stop-Offline "OFFLINE_ERROR: torch-cpu`nThe transfer contains a CPU torch wheel. Expected $ExpectedVersion from the CUDA 13.2 index."
    }
    $python = $null
    if ($Environment) {
        $candidate = Join-Path $Environment 'Scripts\python.exe'
        if (Test-Path -LiteralPath $candidate) { $python = $candidate }
    }
    if ($python) {
        $probe = "import importlib.metadata as metadata; version = metadata.version('torch'); print(version); raise SystemExit(0 if '+cpu' not in version and version == '$ExpectedVersion' else 1)"
        $output = & $python -c $probe 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            Stop-Offline "OFFLINE_ERROR: torch-cpu`nExpected torch $ExpectedVersion. The environment reported:`n$output"
        }
        return
    }
    $matched = @($wheels | Where-Object { $_.Name -like "$needle*" -and $_.Name -like '*win_amd64*' })
    $distInfo = @(Get-ChildItem -LiteralPath $Cache -Recurse -Directory -Filter "torch-$ExpectedVersion.dist-info" -ErrorAction SilentlyContinue)
    if ($matched.Count -lt 1 -and $distInfo.Count -lt 1) {
        Stop-Offline "OFFLINE_ERROR: offline-install`nCUDA wheel $ExpectedVersion for Windows amd64 is not in the transfer cache."
    }
}

function Export-GitBundle {
    param([string]$Repo, [string]$Transfer)
    $gitDir = Join-Path $Transfer 'git'
    New-Item -ItemType Directory -Force -Path $gitDir | Out-Null
    $bundle = Join-Path $gitDir 'repo.bundle'
    if (Test-Path -LiteralPath $bundle) { Remove-Item -LiteralPath $bundle -Force }
    $inside = Get-GitOutput $Repo @('rev-parse', '--is-inside-work-tree')
    if ($inside -ne 'true') {
        Stop-Offline "OFFLINE_ERROR: not-a-repository`n$Repo is not a Git work tree."
    }
    $gitmodules = Join-Path $Repo '.gitmodules'
    if (Test-Path -LiteralPath $gitmodules) {
        Stop-Offline "OFFLINE_ERROR: submodules`n.gitmodule is present. Vendor submodules before exporting; this workflow does not transfer submodule checkouts."
    }
    $lfs = & git -C $Repo lfs ls-files --long 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0 -and $lfs.Trim()) {
        foreach ($line in @($lfs -split "`r?`n")) {
            if ($line -notmatch '^([0-9a-f]{64})\s+([-|*])\s+') { continue }
            $oid = $Matches[1]
            $marker = $Matches[2]
            $objectPath = Join-Path $Repo (".git\lfs\objects\{0}\{1}\{2}" -f $oid.Substring(0, 2), $oid.Substring(2, 2), $oid)
            if ($marker -eq '-' -or -not (Test-Path -LiteralPath $objectPath)) {
                Stop-Offline "OFFLINE_ERROR: lfs-missing`nGit LFS object $oid is not in the local store.`n$line"
            }
        }
        $destination = Join-Path $Transfer 'lfs\objects'
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Copy-Item -Path (Join-Path $Repo '.git\lfs\objects\*') -Destination $destination -Recurse -Force
    } elseif ($LASTEXITCODE -ne 0 -and (Test-Path -LiteralPath (Join-Path $Repo '.gitattributes'))) {
        $attributes = [System.IO.File]::ReadAllText((Join-Path $Repo '.gitattributes'))
        if ($attributes -match 'filter=lfs') {
            Stop-Offline "OFFLINE_ERROR: lfs-missing`ngit lfs ls-files failed and LFS attributes are present.`n$lfs"
        }
    }
    [void](Invoke-Native 'git' @('-C', $Repo, 'bundle', 'create', $bundle, 'HEAD', '--branches', '--tags') 'OFFLINE_ERROR: missing-git-objects')
    [void](Invoke-Native 'git' @('-C', $Repo, 'bundle', 'verify', $bundle) 'OFFLINE_ERROR: missing-git-objects')
    $head = Get-GitOutput $Repo @('rev-parse', 'HEAD')
    $branch = Get-GitOutput $Repo @('rev-parse', '--abbrev-ref', 'HEAD')
    return @{ head = $head; branch = $branch; bundle = 'git/repo.bundle' }
}

function Import-GitBundleFile {
    param([string]$Repo, [string]$Bundle)
    if (-not (Test-Path -LiteralPath $Bundle)) {
        Stop-Offline "OFFLINE_ERROR: missing-git-objects`nBundle was not found: $Bundle"
    }
    $gitDir = Join-Path $Repo '.git'
    if (-not (Test-Path -LiteralPath $gitDir)) {
        $parent = Split-Path -Parent $Repo
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
        }
        if ((Test-Path -LiteralPath $Repo) -and @(Get-ChildItem -LiteralPath $Repo -Force).Count -gt 0) {
            Stop-Offline "OFFLINE_ERROR: not-a-repository`n$Repo exists and is not a Git repository. Refusing to overwrite it."
        }
        [void](Invoke-Native 'git' @('clone', $Bundle, $Repo) 'OFFLINE_ERROR: missing-git-objects')
        Write-OfflineLine 'OFFLINE_INFO: cloned'
        Write-OfflineLine 'Integrate with the cloned branch. No merge is required on a fresh clone.'
        return
    }
    $status = & git -C $Repo status --porcelain 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Stop-Offline "OFFLINE_ERROR: not-a-repository`n$status"
    }
    if ($status.Trim()) {
        Write-OfflineLine 'OFFLINE_INFO: dirty'
        Write-OfflineLine 'Uncommitted work was left untouched.'
    }
    $before = Get-GitOutput $Repo @('rev-parse', 'HEAD')
    $verify = & git -C $Repo bundle verify $Bundle 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Stop-Offline "OFFLINE_ERROR: missing-git-objects`n$verify"
    }
    $fetch = & git -C $Repo fetch --no-tags $Bundle '+refs/heads/*:refs/remotes/kraken-offline/*' 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        $after = Get-GitOutput $Repo @('rev-parse', 'HEAD')
        if ($after -ne $before) {
            Stop-Offline "OFFLINE_ERROR: failed`nHEAD changed during a failed fetch. Local HEAD is $after."
        }
        Stop-Offline "OFFLINE_ERROR: missing-git-objects`n$fetch"
    }
    $tagFetch = & git -C $Repo fetch --no-tags $Bundle 'refs/tags/*:refs/tags/*' 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-OfflineLine 'OFFLINE_INFO: tags-not-updated'
        Write-OfflineLine $tagFetch.Trim()
    }
    $after = Get-GitOutput $Repo @('rev-parse', 'HEAD')
    if ($after -ne $before) {
        Stop-Offline "OFFLINE_ERROR: failed`nImport moved HEAD from $before to $after. Local commits must stay in place."
    }
    $branch = Get-GitOutput $Repo @('rev-parse', '--abbrev-ref', 'HEAD')
    if ($branch -eq 'HEAD') {
        Write-OfflineLine 'Detached HEAD was left unchanged. Inspect refs/remotes/kraken-offline.'
        return
    }
    $remoteRef = "refs/remotes/kraken-offline/$branch"
    $remoteExists = & git -C $Repo rev-parse --verify --quiet $remoteRef 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-OfflineLine "OFFLINE_INFO: branch-not-in-bundle"
        Write-OfflineLine "The bundle has no '$branch' branch. Other branches are under refs/remotes/kraken-offline/."
        return
    }
    $local = Get-GitOutput $Repo @('rev-parse', 'HEAD')
    $remote = Get-GitOutput $Repo @('rev-parse', $remoteRef)
    if ($local -eq $remote) {
        Write-OfflineLine 'OFFLINE_INFO: already-integrated'
        return
    }
    $baseOutput = & git -C $Repo merge-base $local $remote 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-OfflineLine 'OFFLINE_INFO: diverged'
        Write-OfflineLine "Histories are unrelated. Inspect both sides, then merge explicitly: git merge kraken-offline/$branch"
        return
    }
    $base = $baseOutput.Trim()
    if ($base -eq $remote) {
        Write-OfflineLine 'OFFLINE_INFO: local-ahead'
        Write-OfflineLine "Local '$branch' contains commits that are not in the transfer. Nothing was merged."
        return
    }
    if ($base -eq $local) {
        Write-OfflineLine 'OFFLINE_INFO: fast-forward'
    } else {
        Write-OfflineLine 'OFFLINE_INFO: diverged'
    }
    Write-OfflineLine "Review the incoming commits, then integrate them yourself: git merge kraken-offline/$branch"
    Write-OfflineLine 'Do not reset or force-push. Resolve conflicts in the work tree and commit the merge.'
}

function Publish-TransferManifest {
    param(
        [string]$Transfer,
        $GitInfo,
        [string]$PythonRequires,
        [string]$ToolMode,
        [string]$PythonVersion,
        [string]$UvVersion,
        [string]$LockHash,
        [string]$TorchVersion,
        [object[]]$External
    )
    $files = New-Object System.Collections.ArrayList
    $children = @(Get-ChildItem -LiteralPath $Transfer -Recurse -File -Force)
    foreach ($file in $children) {
        $relative = Get-RelativePath $Transfer $file.FullName
        if ($relative -in @('manifest.json', 'manifest.sha256')) { continue }
        [void]$files.Add(@{
            path = $relative
            sha256 = (Get-Sha256 $file.FullName)
            bytes = [int64]$file.Length
        })
    }
    $manifest = @{
        schema_version = 1
        created_at = [DateTime]::UtcNow.ToString('o')
        platform = @{
            os = 'windows'
            architecture = (Get-NormalizedArchitecture)
        }
        python_requires = $PythonRequires
        python_version = $PythonVersion
        uv_version = $UvVersion
        tool_mode = $ToolMode
        lock_sha256 = $LockHash
        torch_version = $TorchVersion
        git = $GitInfo
        external = @($External)
        files = @($files)
    }
    $manifestPath = Join-Path $Transfer 'manifest.json'
    Write-OfflineJson $manifestPath $manifest
    $digest = Get-Sha256 $manifestPath
    [System.IO.File]::WriteAllText((Join-Path $Transfer 'manifest.sha256'), $digest, (New-Object System.Text.UTF8Encoding $false))
}

function Test-TransferManifest {
    param([string]$Transfer)
    if (-not (Test-Path -LiteralPath $Transfer)) {
        Stop-Offline "OFFLINE_ERROR: missing-transfer`nTransfer directory was not found: $Transfer"
    }
    $manifestPath = Join-Path $Transfer 'manifest.json'
    $digestPath = Join-Path $Transfer 'manifest.sha256'
    if (-not (Test-Path -LiteralPath $manifestPath) -or -not (Test-Path -LiteralPath $digestPath)) {
        Stop-Offline "OFFLINE_ERROR: invalid-manifest`nmanifest.json and manifest.sha256 are required."
    }
    $expected = ([System.IO.File]::ReadAllText($digestPath)).Trim().ToLowerInvariant()
    $actual = Get-Sha256 $manifestPath
    if ($expected -ne $actual) {
        Stop-Offline "OFFLINE_ERROR: invalid-manifest`nmanifest.sha256 does not match manifest.json."
    }
    try {
        $manifest = Read-OfflineJson $manifestPath
    } catch {
        Stop-Offline "OFFLINE_ERROR: invalid-manifest`n$($_.Exception.Message)"
    }
    foreach ($key in @('schema_version', 'platform', 'python_requires', 'git', 'files')) {
        if (-not (Test-MapKey $manifest $key)) {
            Stop-Offline "OFFLINE_ERROR: invalid-manifest`nMissing manifest field '$key'."
        }
    }
    if ([int](Get-MapValue $manifest 'schema_version') -ne 1) {
        Stop-Offline "OFFLINE_ERROR: invalid-manifest`nUnsupported schema $(Get-MapValue $manifest 'schema_version')."
    }
    $platform = Get-MapValue $manifest 'platform'
    if ([string](Get-MapValue $platform 'os') -ne 'windows') {
        Stop-Offline "OFFLINE_ERROR: platform-incompatible`nTransfer platform is $(Get-MapValue $platform 'os')."
    }
    $architecture = [string](Get-MapValue $platform 'architecture')
    $localArchitecture = Get-NormalizedArchitecture
    if ($architecture -ne $localArchitecture) {
        Stop-Offline "OFFLINE_ERROR: platform-incompatible`nTransfer architecture is $architecture. This computer is $localArchitecture."
    }
    $python = Get-LocalPython
    Test-PythonCompatible $python.Version ([string](Get-MapValue $manifest 'python_requires'))
    $files = @(Get-MapValue $manifest 'files')
    if ($files.Count -lt 1) {
        Stop-Offline "OFFLINE_ERROR: invalid-manifest`nThe manifest does not list any files."
    }
    foreach ($entry in $files) {
        if ($null -eq $entry -or -not (Test-MapKey $entry 'path')) {
            Stop-Offline "OFFLINE_ERROR: invalid-manifest`nA file entry is not an object."
        }
        foreach ($key in @('path', 'sha256', 'bytes')) {
            if (-not (Test-MapKey $entry $key)) {
                Stop-Offline "OFFLINE_ERROR: invalid-manifest`nA file entry is missing '$key'."
            }
        }
        $relative = [string](Get-MapValue $entry 'path')
        if ($relative -match '(^[\\/])|(\.\.)') {
            Stop-Offline "OFFLINE_ERROR: invalid-manifest`nRefusing unsafe path $relative."
        }
        $full = Join-Path $Transfer ($relative -replace '/', '\')
        if (-not (Test-Path -LiteralPath $full)) {
            Stop-Offline "OFFLINE_ERROR: missing-file`nThe transfer is incomplete. Missing: $relative"
        }
        $digest = Get-Sha256 $full
        if ($digest -ne [string](Get-MapValue $entry 'sha256')) {
            Stop-Offline "OFFLINE_ERROR: checksum`nSHA-256 does not match for $relative."
        }
    }
    return @{ Manifest = $manifest; Python = $python }
}

function Get-ExternalTools {
    $names = @('git', 'git-lfs', 'rustc', 'cargo', 'cl')
    $items = New-Object System.Collections.ArrayList
    foreach ($name in $names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        $source = $null
        if ($null -ne $command) { $source = [string]$command.Source }
        [void]$items.Add(@{ name = $name; available = ($null -ne $command); path = $source })
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    [void]$items.Add(@{ name = 'vswhere'; available = (Test-Path -LiteralPath $vswhere); path = $vswhere })
    return @($items)
}

function Export-PythonDependencies {
    param(
        [string]$Repo,
        [string]$Transfer,
        [string]$UvCommand,
        [string]$FindLinks,
        [string]$ScriptRoot
    )
    $project = Join-Path $Repo 'pyproject.toml'
    if (-not (Test-Path -LiteralPath $project)) {
        Write-OfflineLine 'OFFLINE_INFO: no-python-project'
        return @{ lock = ''; torch = $null; reused = $false }
    }
    $lock = Join-Path $Repo 'uv.lock'
    if (-not (Test-Path -LiteralPath $lock)) {
        [void](Invoke-Native $UvCommand @('lock', '--directory', $Repo) 'OFFLINE_ERROR: offline-install')
    }
    $lockHash = Get-Sha256 $lock
    $cache = Join-Path $Transfer 'uv-cache'
    $stampPath = Join-Path $Transfer 'dependency-stamp.txt'
    $reused = $false
    if ((Test-Path -LiteralPath $stampPath) -and (Test-Path -LiteralPath $cache)) {
        $stamp = ([System.IO.File]::ReadAllText($stampPath)).Trim()
        if ($stamp -eq $lockHash) {
            $reused = $true
            Write-OfflineLine 'OFFLINE_INFO: dependencies-reused'
        }
    }
    $torch = Get-TorchSpec $Repo
    if (-not $reused) {
        if (Test-WorkspaceProject $Repo) { Import-VisualStudioEnvironment }
        New-Item -ItemType Directory -Force -Path $cache | Out-Null
        $tempEnv = Join-Path ([System.IO.Path]::GetTempPath()) ('kraken-offline-export-' + [guid]::NewGuid().ToString('n'))
        $syncArgs = @('sync', '--directory', $Repo, '--frozen', '--link-mode', 'copy')
        $syncArgs += @(Get-WorkspaceSyncArgs $Repo $ScriptRoot)
        if (-not $FindLinks) {
            $defaultLinks = Join-Path $Repo 'offline\dep-cache\wheelhouse'
            if (Test-Path -LiteralPath $defaultLinks) { $FindLinks = $defaultLinks }
        }
        if ($FindLinks) {
            if (-not (Test-Path -LiteralPath $FindLinks)) {
                Stop-Offline "OFFLINE_ERROR: offline-install`nFind-links directory was not found: $FindLinks"
            }
            $syncArgs += @('--find-links', $FindLinks)
        }
        $python = Get-LocalPython
        $syncArgs += @('--python', $python.Executable, '--no-python-downloads')
        try {
            Invoke-UvCommand -File $UvCommand -Arguments $syncArgs -Cache $cache -ProjectEnvironment $tempEnv -Offline $false -PythonInstallDir '' -Proxy ''
            if ($script:LastUvExit -ne 0) {
                Stop-Offline "OFFLINE_ERROR: offline-install`nuv sync failed while preparing the transfer cache.`n$script:LastUvOutput"
            }
            Write-OfflineLine $script:LastUvOutput
            Assert-TorchDistribution -ExpectedVersion $torch -Cache $cache -Environment $tempEnv
        } finally {
            if (Test-Path -LiteralPath $tempEnv) {
                Remove-Item -LiteralPath $tempEnv -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        [System.IO.File]::WriteAllText($stampPath, $lockHash, (New-Object System.Text.UTF8Encoding $false))
    } else {
        Assert-TorchDistribution -ExpectedVersion $torch -Cache $cache -Environment ''
    }
    $requirements = Join-Path $Transfer 'requirements.txt'
    $exportArgs = @('export', '--directory', $Repo, '--frozen', '--no-emit-workspace', '--output-file', $requirements)
    $exportArgs += @(Get-WorkspaceSyncArgs $Repo $ScriptRoot)
    [void](Invoke-Native $UvCommand $exportArgs 'OFFLINE_ERROR: offline-install')
    return @{ lock = $lockHash; torch = $torch; reused = $reused }
}

function Install-OfflineEnvironment {
    param(
        [string]$Repo,
        [string]$Cache,
        [string]$ProjectEnvironment,
        [string]$UvCommand,
        [string]$ScriptRoot,
        [string]$PythonExecutable,
        [string]$Proxy,
        [string]$PythonInstallDir,
        [bool]$LockExpected = $false
    )
    if (-not (Test-Path -LiteralPath (Join-Path $Repo 'pyproject.toml'))) {
        if ($LockExpected) {
            Stop-Offline "OFFLINE_ERROR: offline-install`nThe cloned repository has no pyproject.toml. The bundle does not check out the exported HEAD."
        }
        Write-OfflineLine 'OFFLINE_INFO: no-dependencies'
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Repo 'uv.lock'))) {
        Stop-Offline "OFFLINE_ERROR: offline-install`n$Repo has no uv.lock. Export the lockfile in the Git bundle before installing offline."
    }
    if (Test-WorkspaceProject $Repo) { Import-VisualStudioEnvironment }
    New-Item -ItemType Directory -Force -Path $ProjectEnvironment | Out-Null
    $syncArgs = @('sync', '--directory', $Repo, '--frozen', '--offline', '--link-mode', 'copy', '--python', $PythonExecutable, '--no-python-downloads')
    $syncArgs += @(Get-WorkspaceSyncArgs $Repo $ScriptRoot)
    Invoke-UvCommand -File $UvCommand -Arguments $syncArgs -Cache $Cache -ProjectEnvironment $ProjectEnvironment -Offline $true -PythonInstallDir $PythonInstallDir -Proxy $Proxy
    Write-OfflineLine $script:LastUvOutput
    if ($script:LastUvExit -ne 0) {
        Stop-Offline "OFFLINE_ERROR: offline-install`nuv sync --offline failed. The transfer cache does not contain every locked distribution, or the package set is incompatible with this Python.`n$script:LastUvOutput"
    }
    $torch = Get-TorchSpec $Repo
    Assert-TorchDistribution -ExpectedVersion $torch -Cache $Cache -Environment $ProjectEnvironment
    Write-OfflineLine 'OFFLINE_INFO: offline-install-ok'
}

function Start-ProxyProbe {
    $listener = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $state = @{ Listener = $listener; Count = 0; Stop = $false }
    $runspace = [runspacefactory]::CreateRunspace()
    $runspace.Open()
    $powershell = [powershell]::Create()
    $powershell.Runspace = $runspace
    [void]$powershell.AddScript({
        param($State)
        while (-not $State.Stop) {
            if ($State.Listener.Pending()) {
                $client = $State.Listener.AcceptTcpClient()
                $State.Count = [int]$State.Count + 1
                $client.Close()
            } else {
                Start-Sleep -Milliseconds 50
            }
        }
    }).AddArgument($state)
    $handle = $powershell.BeginInvoke()
    return @{ State = $state; PowerShell = $powershell; Handle = $handle; Port = $port }
}

function Stop-ProxyProbe {
    param($Probe)
    if ($null -eq $Probe) { return 0 }
    $Probe.State.Stop = $true
    try { $Probe.PowerShell.EndInvoke($Probe.Handle) } catch { }
    try { $Probe.State.Listener.Stop() } catch { }
    $Probe.PowerShell.Dispose()
    $Probe.PowerShell.Runspace.Close()
    return [int]$Probe.State.Count
}

function Invoke-OfflineVerification {
    param(
        [string]$Transfer,
        [string]$RepoRoot,
        [string]$UvCommand,
        [string]$ProjectEnvironment,
        [string]$CachePath,
        [string]$ScriptRoot,
        [string]$RepresentativeTests
    )
    $checked = Test-TransferManifest $Transfer
    $manifest = $checked.Manifest
    $bundleRelative = [string](Get-MapValue (Get-MapValue $manifest 'git') 'bundle')
    $bundle = Join-Path $Transfer ($bundleRelative -replace '/', '\')
    $cloneRoot = $RepoRoot
    $temporaryClone = $false
    if (-not $cloneRoot) {
        $cloneRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('kraken-offline-verify-' + [guid]::NewGuid().ToString('n'))
        $temporaryClone = $true
    }
    if (-not (Test-Path -LiteralPath (Join-Path $cloneRoot '.git'))) {
        [void](Invoke-Native 'git' @('clone', $bundle, $cloneRoot) 'OFFLINE_ERROR: missing-git-objects')
        $temporaryClone = $true
    }
    $cache = if ($CachePath) { $CachePath } else { Join-Path $Transfer 'uv-cache' }
    $keepCheckout = [bool]$ProjectEnvironment
    if (-not $ProjectEnvironment) {
        $ProjectEnvironment = Join-Path ([System.IO.Path]::GetTempPath()) ('kraken-offline-verify-venv-' + [guid]::NewGuid().ToString('n'))
    }
    $pythonInstall = ''
    $pythonExecutable = $checked.Python.Executable
    if ([string](Get-MapValue $manifest 'tool_mode') -eq 'Bundle') {
        $pythonInstall = Join-Path $Transfer 'python'
        $bundled = @(Get-ChildItem -LiteralPath $pythonInstall -Recurse -Filter 'python.exe' -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '\\Scripts\\' } |
            Select-Object -First 1)
        if ($bundled.Count -eq 1) { $pythonExecutable = $bundled[0].FullName }
    }
    $probe = $null
    $report = @{
        ok = $false
        cache_dir = [System.IO.Path]::GetFullPath($cache)
        project_environment = $ProjectEnvironment
        repository = $cloneRoot
        torch = @{ checked = $false; version = [string](Get-MapValue $manifest 'torch_version') }
        tests = @{ ran = $false; exit_code = $null; command = '' }
        build = @{ compileall = $false; native = @() }
        external = @(Get-ExternalTools)
        network_isolation = @{
            uv_offline = $true
            proxy = ''
            proxy_connections = 0
            proxy_listener_started = $false
            os_firewall = $false
            note = 'uv was started with --offline and HTTP(S)_PROXY pointed at a local listener. Connections that bypass the proxy are not observed. An OS firewall rule was not installed.'
        }
    }
    try {
        $probe = Start-ProxyProbe
        $report.network_isolation.proxy_listener_started = $true
        $report.network_isolation.proxy = "http://127.0.0.1:$($probe.Port)"
        $lockExpected = -not [string]::IsNullOrWhiteSpace([string](Get-MapValue $manifest 'lock_sha256'))
        Install-OfflineEnvironment -Repo $cloneRoot -Cache $cache -ProjectEnvironment $ProjectEnvironment -UvCommand $UvCommand -ScriptRoot $ScriptRoot -PythonExecutable $pythonExecutable -Proxy $report.network_isolation.proxy -PythonInstallDir $pythonInstall -LockExpected $lockExpected
        $venvPython = Join-Path $ProjectEnvironment 'Scripts\python.exe'
        $torch = Get-TorchSpec $cloneRoot
        if ($torch) {
            $probeCode = "import torch; print(torch.__version__); print(int(torch.cuda.is_available()))"
            $torchOutput = & $venvPython -c $probeCode 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) {
                Stop-Offline "OFFLINE_ERROR: torch-cpu`n$torchOutput"
            }
            $lines = @($torchOutput -split "`r?`n" | Where-Object { $_ -and $_.Trim() -ne '' })
            $report.torch = @{
                checked = $true
                version = $lines[0].Trim()
                cuda_available = ($lines.Count -gt 1 -and $lines[1].Trim() -eq '1')
                cuda_build = $lines[0].Trim()
            }
        }
        $krakenCore = Join-Path $cloneRoot 'src\kraken_core'
        if (Test-Path -LiteralPath $krakenCore) {
            $importOutput = & $venvPython -c "import kraken_core, kraken_manager" 2>&1 | Out-String
            Write-OfflineLine $importOutput
            if ($LASTEXITCODE -ne 0) {
                Stop-Offline "OFFLINE_ERROR: offline-install`nKraken packages did not import from the offline environment."
            }
            $compileOutput = & $venvPython -m compileall -q $krakenCore 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) {
                Stop-Offline "OFFLINE_ERROR: offline-install`ncompileall failed for src/kraken_core."
            }
            $report.build.compileall = $true
        }
        $tests = $RepresentativeTests
        if (-not $tests -and (Test-Path -LiteralPath (Join-Path $cloneRoot 'plugins\csliser\tests\unit\test_ranges.py'))) {
            $tests = 'plugins/csliser/tests/unit/test_ranges.py plugins/csliser/tests/unit/test_planner.py'
        }
        if ($tests) {
            $pytest = & $venvPython -c "import pytest" 2>&1 | Out-String
            if ($LASTEXITCODE -eq 0) {
                $testArgs = @('-m', 'pytest') + @($tests -split '\s+') + @('-q', '--tb=short')
                $report.tests.command = "$venvPython $($testArgs -join ' ')"
                & $venvPython @testArgs 2>&1 | ForEach-Object { Write-OfflineLine ([string]$_) }
                $report.tests.ran = $true
                $report.tests.exit_code = $LASTEXITCODE
                if ($LASTEXITCODE -ne 0) {
                    Stop-Offline "OFFLINE_ERROR: offline-install`nRepresentative tests failed."
                }
            } else {
                Write-OfflineLine 'OFFLINE_INFO: tests-skipped'
                Write-OfflineLine $pytest.Trim()
            }
        }
        $cargo = Join-Path $cloneRoot 'blob_gateway\Cargo.toml'
        if (Test-Path -LiteralPath $cargo) {
            $rustc = Get-Command rustc -ErrorAction SilentlyContinue
            $report.build.native += @{
                name = 'blob_gateway'
                rustc = $(if ($rustc) { $rustc.Source } else { $null })
                offline_build = $false
                reason = 'Rust crates are not vendored in this transfer. cargo build would need crates.io or a vendor directory.'
            }
            Write-OfflineLine 'OFFLINE_INFO: native-skipped blob_gateway'
        }
        $report.ok = $true
    } finally {
        $connections = Stop-ProxyProbe $probe
        $report.network_isolation.proxy_connections = $connections
        $reportPath = $Transfer + '.verify-report.json'
        Write-OfflineJson $reportPath $report
        Write-OfflineLine "OFFLINE_REPORT: $reportPath"
        if ($temporaryClone -and -not $keepCheckout -and $cloneRoot -and (Test-Path -LiteralPath $cloneRoot)) {
            Remove-Item -LiteralPath $cloneRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if ($report.network_isolation.proxy_connections -gt 0) {
        Stop-Offline "OFFLINE_ERROR: offline-install`nThe offline installer opened $($report.network_isolation.proxy_connections) proxy connection(s)."
    }
}

function Invoke-KrakenOffline {
    param(
        [string]$Mode,
        [string]$TransferPath,
        [string]$RepoRoot,
        [string]$UvCommand,
        [string]$ProjectEnvironment,
        [string]$FindLinks,
        [string]$CachePath,
        [string]$ToolMode,
        [string]$RepresentativeTests,
        [string]$ScriptRoot
    )
    try {
        if ($Mode -eq 'import-bundle') {
            Import-GitBundleFile -Repo $RepoRoot -Bundle $TransferPath
            return 0
        }
        if ($Mode -eq 'export') {
            if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }
            if (-not (Test-Path -LiteralPath $RepoRoot)) {
                Stop-Offline "OFFLINE_ERROR: not-a-repository`nRepository was not found: $RepoRoot"
            }
            New-Item -ItemType Directory -Force -Path $TransferPath | Out-Null
            $dirty = & git -C $RepoRoot status --porcelain 2>&1 | Out-String
            if ($LASTEXITCODE -eq 0 -and $dirty.Trim()) {
                Write-OfflineLine 'OFFLINE_INFO: dirty-export'
                Write-OfflineLine 'Uncommitted files are not part of the Git bundle.'
            }
            $gitInfo = Export-GitBundle -Repo $RepoRoot -Transfer $TransferPath
            $python = Get-LocalPython
            $uvVersion = (& $UvCommand --version 2>&1 | Out-String).Trim()
            if ($LASTEXITCODE -ne 0) {
                Stop-Offline "OFFLINE_ERROR: offline-install`nuv is not available: $uvVersion"
            }
            $tools = Join-Path $TransferPath 'tools'
            New-Item -ItemType Directory -Force -Path $tools | Out-Null
            foreach ($scriptName in @('sync.ps1', 'KrakenOffline.psm1', 'sync-profile.json')) {
                Copy-Item -LiteralPath (Join-Path $ScriptRoot $scriptName) -Destination (Join-Path $tools $scriptName) -Force
            }
            if ($ToolMode -eq 'Bundle') {
                $uvCommandInfo = Get-Command $UvCommand -ErrorAction SilentlyContinue
                if ($null -eq $uvCommandInfo -or -not $uvCommandInfo.Source) {
                    Stop-Offline "OFFLINE_ERROR: offline-install`nCould not copy the uv executable."
                }
                Copy-Item -LiteralPath $uvCommandInfo.Source -Destination (Join-Path $tools 'uv.exe') -Force
                $pythonDir = Join-Path $TransferPath 'python'
                $existing = @(Get-ChildItem -LiteralPath $pythonDir -Recurse -Filter 'python.exe' -ErrorAction SilentlyContinue)
                if ($existing.Count -eq 0) {
                    [void](Invoke-Native $UvCommand @('python', 'install', $python.Version, '--install-dir', $pythonDir, '--no-bin') 'OFFLINE_ERROR: offline-install')
                } else {
                    Write-OfflineLine 'OFFLINE_INFO: python-reused'
                }
            } else {
                Write-OfflineLine "OFFLINE_INFO: tools-validated Python $($python.Version); uv $uvVersion"
            }
            $dependency = Export-PythonDependencies -Repo $RepoRoot -Transfer $TransferPath -UvCommand $UvCommand -FindLinks $FindLinks -ScriptRoot $ScriptRoot
            Publish-TransferManifest -Transfer $TransferPath -GitInfo $gitInfo -PythonRequires (Get-PythonRequires $RepoRoot $ScriptRoot) -ToolMode $ToolMode -PythonVersion $python.Version -UvVersion $uvVersion -LockHash $dependency.lock -TorchVersion $dependency.torch -External (Get-ExternalTools)
            $checked = Test-TransferManifest $TransferPath
            Write-OfflineLine "OFFLINE_INFO: export-ok files=$((@(Get-MapValue $checked.Manifest 'files')).Count)"
            return 0
        }
        if ($Mode -eq 'import') {
            $checked = Test-TransferManifest $TransferPath
            if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }
            $bundle = Join-Path $TransferPath ([string](Get-MapValue (Get-MapValue $checked.Manifest 'git') 'bundle') -replace '/', '\')
            Import-GitBundleFile -Repo $RepoRoot -Bundle $bundle
            if (Test-Path -LiteralPath (Join-Path $TransferPath 'uv-cache')) {
                $environment = if ($ProjectEnvironment) { $ProjectEnvironment } else { Join-Path $RepoRoot '.venv' }
                $pythonInstall = ''
                if ([string](Get-MapValue $checked.Manifest 'tool_mode') -eq 'Bundle') {
                    $pythonInstall = Join-Path $TransferPath 'python'
                }
                Install-OfflineEnvironment -Repo $RepoRoot -Cache (Join-Path $TransferPath 'uv-cache') -ProjectEnvironment $environment -UvCommand $UvCommand -ScriptRoot $ScriptRoot -PythonExecutable $checked.Python.Executable -Proxy '' -PythonInstallDir $pythonInstall
            } else {
                Write-OfflineLine 'OFFLINE_INFO: no-dependencies'
            }
            return 0
        }
        if ($Mode -eq 'verify') {
            Invoke-OfflineVerification -Transfer $TransferPath -RepoRoot $RepoRoot -UvCommand $UvCommand -ProjectEnvironment $ProjectEnvironment -CachePath $CachePath -ScriptRoot $ScriptRoot -RepresentativeTests $RepresentativeTests
            return 0
        }
        Stop-Offline "OFFLINE_ERROR: failed`nUnknown mode $Mode"
    } catch {
        $text = $_.Exception.Message
        if ($text -notlike 'OFFLINE_ERROR:*') {
            $text = "OFFLINE_ERROR: failed`n$text"
        }
        Write-OfflineLine $text
        [Console]::Error.WriteLine($text)
        return 1
    }
}

Export-ModuleMember -Function Invoke-KrakenOffline
