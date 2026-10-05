# Build Karakal and publish it to the update folder (beta / stable channel).
#
# Beta for testers (version and beta number are picked automatically):
#   .\publish.ps1 -Channel beta
# Stable release (the same code number without -betaN):
#   .\publish.ps1 -Channel stable
# Trial run into %TEMP%\karakal_share, nothing committed:
#   .\publish.ps1 -Channel beta -DryRun
# The update folder moved: leave a pointer in the old place for installed copies:
#   .\publish.ps1 -MoveTo \\new\share\karakal -ShareRoot \\old\share\karakal
#
# The update folder comes from "update_root" in resources\update_client.json
# (or -ShareRoot). Release notes come from the "Не выпущено" section of CHANGELOG.md.
[CmdletBinding(DefaultParameterSetName = "Publish")]
param(
    [Parameter(ParameterSetName = "Publish", Mandatory = $true)]
    [ValidateSet("beta", "stable")]
    [string]$Channel,

    # Normally left out: taken from the code and the update folder.
    [Parameter(ParameterSetName = "Publish")]
    [string]$Version = "",

    [Parameter(ParameterSetName = "Publish")]
    [Parameter(ParameterSetName = "Move", Mandatory = $true)]
    [string]$ShareRoot = "",

    # Local copy of every build (installer + portable folder); "" to skip.
    [Parameter(ParameterSetName = "Publish")]
    [string]$CopyRoot = "G:\",

    [Parameter(ParameterSetName = "Publish")]
    [switch]$DryRun,

    [Parameter(ParameterSetName = "Publish")]
    [switch]$SkipTests,

    [Parameter(ParameterSetName = "Publish")]
    [switch]$SkipBuild,

    # Publish from uncommitted code (only for trying the script out).
    [Parameter(ParameterSetName = "Publish")]
    [switch]$AllowDirty,

    # Do not commit CHANGELOG.md / version.py after publishing.
    [Parameter(ParameterSetName = "Publish")]
    [switch]$NoCommit,

    [Parameter(ParameterSetName = "Move")]
    [string]$MoveTo = "",

    [string]$IsccPath = ""
)

$ErrorActionPreference = "Stop"
$PluginRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $PluginRoot "..\..")).Path
$IssPath = Join-Path $PluginRoot "packaging\Karakal.iss"
$Tools = Join-Path $PSScriptRoot "release_tools.py"
$Changelog = Join-Path $PluginRoot "CHANGELOG.md"
$VersionFile = Join-Path $PluginRoot "src\karakal\version.py"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Build profile per channel. The release interface is not ready yet, so stable is built
# with the tester interface too; switch "stable" to "dev" (or a release profile) here.
$ChannelProfiles = @{ beta = "tester"; stable = "tester" }

function Get-Iscc {
    param([string]$Hint)
    if ($Hint -and (Test-Path -LiteralPath $Hint)) { return (Resolve-Path $Hint).Path }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) { return $path }
    }
    $reg = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1" -ErrorAction SilentlyContinue
    if ($reg -and $reg.InstallLocation) {
        $fromReg = Join-Path $reg.InstallLocation "ISCC.exe"
        if (Test-Path -LiteralPath $fromReg) { return $fromReg }
    }
    return $null
}

function Get-PyInstaller {
    $venv = Join-Path $RepoRoot ".venv\Scripts\pyinstaller.exe"
    if (Test-Path -LiteralPath $venv) { return $venv }
    $onPath = Get-Command pyinstaller -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "PyInstaller not found: expected $venv"
}

function Invoke-Tools {
    $output = & python $Tools @args
    if ($LASTEXITCODE -ne 0) { throw "release_tools $($args[0]) failed" }
    return $output
}

function Get-NumericVersion([string]$Ver) {
    $core = ($Ver -split "[-+]", 2)[0]
    $parts = @($core.Split(".") | Where-Object { $_ -match "^\d+$" })
    while ($parts.Count -lt 4) { $parts += "0" }
    return ($parts[0..3] -join ".")
}

function Write-VersionJson {
    param(
        [string]$Path,
        [hashtable]$Payload
    )
    $tmp = "$Path.tmp"
    $json = $Payload | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($tmp, $json, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Read-Manifest([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null }
}

function Assert-ChannelVersion([string]$Ch, [string]$Ver) {
    $hasSuffix = $Ver -match "[-.][A-Za-z]"
    if ($Ch -eq "stable" -and $hasSuffix) {
        throw "stable channel rejects prerelease version '$Ver'"
    }
    if ($Ch -eq "beta" -and -not $hasSuffix) {
        throw "beta channel requires a prerelease suffix (e.g. 0.1.0-beta0), got '$Ver'"
    }
}

function Test-Newer([string]$Left, [string]$Right) {
    $py = "from updater.client import compare_versions; import sys; sys.exit(0 if compare_versions(sys.argv[1], sys.argv[2]) > 0 else 1)"
    $env:PYTHONPATH = (Join-Path $RepoRoot "src") + ";" + (Join-Path $PluginRoot "src") + ";" + $env:PYTHONPATH
    & python -c $py $Left $Right
    return $LASTEXITCODE -eq 0
}

function Publish-ToChannel {
    # Copy the installer into <root>\<channel> and put the version on top of its manifest.
    param(
        [string]$Root,
        [string]$Ch,
        [string]$InstallerSrc,
        [string]$Ver,
        [string]$Sha,
        [string]$Notes
    )
    $dir = Join-Path $Root $Ch
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $name = Split-Path $InstallerSrc -Leaf
    $staging = Join-Path $dir ($name + ".partial")
    Copy-Item -LiteralPath $InstallerSrc -Destination $staging -Force
    Move-Item -LiteralPath $staging -Destination (Join-Path $dir $name) -Force

    $manifestPath = Join-Path $dir "version.json"
    $prev = Read-Manifest $manifestPath
    $releases = @(@{
        version      = $Ver
        download_url = $name
        notes        = $Notes
        sha256       = $Sha
        channel      = $Ch
    })
    if ($prev -and $prev.releases) {
        foreach ($item in @($prev.releases)) {
            if ([string]$item.version -eq $Ver) { continue }
            $releases += @{
                version      = [string]$item.version
                download_url = [string]$item.download_url
                notes        = [string]$item.notes
                sha256       = [string]$item.sha256
                channel      = $Ch
            }
        }
    }
    Write-VersionJson -Path $manifestPath -Payload @{
        version                = $Ver
        channel                = $Ch
        download_url           = $name
        sha256                 = $Sha
        release_notes_markdown = $Notes
        releases               = $releases
    }
    Write-Host "  $Ch\version.json -> $Ver"
}

# --- Move the update folder -------------------------------------------------------
if ($PSCmdlet.ParameterSetName -eq "Move" -or $MoveTo) {
    if (-not $MoveTo) { throw "-MoveTo requires a destination root" }
    if ([string]::IsNullOrWhiteSpace($ShareRoot)) { throw "-ShareRoot is required for -MoveTo" }
    foreach ($ch in @("beta", "stable")) {
        $dir = Join-Path $ShareRoot $ch
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $manifestPath = Join-Path $dir "version.json"
        $payload = @{
            version                = "0.0.0"
            channel                = $ch
            moved_to               = $MoveTo
            download_url           = ""
            release_notes_markdown = "Updates moved to $MoveTo"
            releases               = @()
        }
        $existing = Read-Manifest $manifestPath
        if ($existing) {
            if ($existing.version) { $payload.version = [string]$existing.version }
            if ($existing.download_url) { $payload.download_url = [string]$existing.download_url }
        }
        Write-VersionJson -Path $manifestPath -Payload $payload
    }
    Write-Host "Wrote moved_to=$MoveTo into $ShareRoot\beta and $ShareRoot\stable"
    exit 0
}

# --- Where and what -----------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($ShareRoot)) {
    $ShareRoot = [string](Invoke-Tools update-root)
}
if ($DryRun) {
    $ShareRoot = Join-Path $env:TEMP "karakal_share"
    $CopyRoot = ""
    $NoCommit = $true
}
if ([string]::IsNullOrWhiteSpace($ShareRoot)) {
    throw "Update folder is not set: write it into ""update_root"" of plugins\karakal\resources\update_client.json or pass -ShareRoot"
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $ShareRoot)) {
    throw "Update folder is not reachable: $ShareRoot"
}

if (-not $AllowDirty -and -not $DryRun) {
    $dirty = & git -C $RepoRoot status --porcelain -- plugins/karakal src/updater
    if ($dirty) {
        throw "Uncommitted changes in plugins/karakal or src/updater: commit them first (or -AllowDirty to try the script)"
    }
}

foreach ($ch in @("beta", "stable")) {
    $dir = Join-Path $ShareRoot $ch
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $manifest = Join-Path $dir "version.json"
    if (-not (Test-Path -LiteralPath $manifest)) {
        Write-VersionJson -Path $manifest -Payload @{
            version                = "0.0.0"
            channel                = $ch
            download_url           = ""
            releases               = @()
            release_notes_markdown = "placeholder"
        }
    }
}

if (-not $Version) {
    $Version = [string](Invoke-Tools next-version --channel $Channel --share $ShareRoot)
}
Assert-ChannelVersion -Ch $Channel -Ver $Version
$prev = Read-Manifest (Join-Path $ShareRoot "$Channel\version.json")
if ($prev -and $prev.version -and $prev.version -ne "0.0.0" -and -not $prev.moved_to) {
    if (-not (Test-Newer $Version ([string]$prev.version))) {
        throw "Version $Version is not newer than $($prev.version) already in $Channel"
    }
}
Invoke-Tools check-notes --channel $Channel | Out-Null
$buildProfile = $ChannelProfiles[$Channel]
Write-Host "Publishing Karakal $Version to $Channel (profile $buildProfile) -> $ShareRoot"

# --- Tests and build ----------------------------------------------------------------
Push-Location $RepoRoot
try {
    if (-not $SkipTests) {
        Write-Host "Running pytest..."
        $env:KARAKAL_BUILD_PROFILE = ""
        & python -m pytest plugins/karakal/tests/unit -q --tb=line
        if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
    }

    if (-not $SkipBuild) {
        $env:KARAKAL_BUILD_PROFILE = $buildProfile
        $env:KARAKAL_VERSION = $Version
        Write-Host "Building exe profile=$buildProfile version=$Version"
        Push-Location $PluginRoot
        try {
            & (Get-PyInstaller) --noconfirm karakal.spec
            if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
            $builtName = if ($buildProfile -eq "tester") { "karakal-$Version" } else { "karakal" }
            $built = Join-Path $PluginRoot "dist\$builtName"
            if (-not (Test-Path -LiteralPath $built)) { throw "PyInstaller output not found: $built" }
            $stage = Join-Path $PluginRoot "dist\windows\karakal"
            if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
            New-Item -ItemType Directory -Force -Path (Split-Path $stage) | Out-Null
            Copy-Item -LiteralPath $built -Destination $stage -Recurse
            $exe = Get-ChildItem -LiteralPath $stage -Filter "karakal*.exe" | Select-Object -First 1
            if ($exe -and $exe.Name -ne "karakal.exe") {
                Rename-Item -LiteralPath $exe.FullName -NewName "karakal.exe"
            }
        } finally {
            Pop-Location
            Remove-Item Env:\KARAKAL_VERSION -ErrorAction SilentlyContinue
        }

        $iscc = Get-Iscc -Hint $IsccPath
        if (-not $iscc) { throw "ISCC.exe not found: install Inno Setup 6 or pass -IsccPath" }
        $override = Join-Path $PluginRoot "packaging\version_override.iss"
        $numeric = Get-NumericVersion $Version
        $overrideText = "#define AppVersion `"$Version`"`r`n#define AppVersionNumeric `"$numeric`"`r`n"
        Set-Content -LiteralPath $override -Value $overrideText -Encoding ASCII
        try {
            & $iscc /Q $IssPath
            if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
        } finally {
            Remove-Item -LiteralPath $override -Force -ErrorAction SilentlyContinue
        }
    }
} finally {
    Pop-Location
}

$installerName = "Karakal-setup-$Version.exe"
$installerSrc = Join-Path $PluginRoot "dist\installers\$installerName"
if (-not (Test-Path -LiteralPath $installerSrc)) {
    if (-not $DryRun) { throw "Installer not found: $installerSrc" }
    Write-Warning "Installer not found at $installerSrc - writing a stub for the dry run"
    New-Item -ItemType Directory -Force -Path (Split-Path $installerSrc) | Out-Null
    [System.IO.File]::WriteAllBytes($installerSrc, [byte[]](1..64))
}
$sha = (Get-FileHash -LiteralPath $installerSrc -Algorithm SHA256).Hash.ToLowerInvariant()

# --- Release notes: "Не выпущено" becomes the version section ------------------------
$changelogBackup = Get-Content -LiteralPath $Changelog -Raw -Encoding UTF8
$notesFile = Join-Path $env:TEMP "karakal_notes_$Version.md"
Invoke-Tools release-notes --version $Version --out $notesFile | Out-Null
$notesText = (Get-Content -LiteralPath $notesFile -Raw -Encoding UTF8).Trim()

# --- Publish ------------------------------------------------------------------------
Publish-ToChannel -Root $ShareRoot -Ch $Channel -InstallerSrc $installerSrc -Ver $Version -Sha $sha -Notes $notesText
if ($Channel -eq "stable") {
    # Testers stay on beta: give them the release too while it is the newest version.
    $beta = Read-Manifest (Join-Path $ShareRoot "beta\version.json")
    if (-not $beta -or -not $beta.version -or $beta.version -eq "0.0.0" -or (Test-Newer $Version ([string]$beta.version))) {
        Publish-ToChannel -Root $ShareRoot -Ch "beta" -InstallerSrc $installerSrc -Ver $Version -Sha $sha -Notes $notesText
    }
}

if ($DryRun) {
    [System.IO.File]::WriteAllText($Changelog, $changelogBackup, [System.Text.UTF8Encoding]::new($false))
    Write-Host "DryRun: CHANGELOG.md left as it was; release notes in $notesFile"
} else {
    Invoke-Tools set-beta --version $Version | Out-Null
}

if ($CopyRoot -and (Test-Path -LiteralPath $CopyRoot) -and -not $SkipBuild) {
    $copyDir = Join-Path $CopyRoot ("karakal_" + ($Version -replace "[.-]", "_"))
    New-Item -ItemType Directory -Force -Path $copyDir | Out-Null
    Copy-Item -LiteralPath $installerSrc -Destination $copyDir -Force
    $portable = Join-Path $copyDir "karakal_portable"
    if (Test-Path -LiteralPath $portable) { Remove-Item -LiteralPath $portable -Recurse -Force }
    Copy-Item -LiteralPath (Join-Path $PluginRoot "dist\windows\karakal") -Destination $portable -Recurse
    Write-Host "Local copy: $copyDir"
}

if (-not $NoCommit) {
    & git -C $RepoRoot add -- $Changelog $VersionFile
    & git -C $RepoRoot commit -q -m "chore(karakal): выпуск $Version ($Channel)" -- $Changelog $VersionFile
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    & git -C $RepoRoot tag "karakal-v$Version"
    Write-Host "Committed CHANGELOG.md and version.py, tag karakal-v$Version (not pushed)"
}

Write-Host "Published $Version to $Channel in $ShareRoot"
Write-Host "sha256=$sha"
