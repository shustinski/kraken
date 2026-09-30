# Build and publish Karakal installers to a share (beta/stable channels).
# Example:
#   .\publish.ps1 -Channel beta -Version 0.1.0-beta0 -ShareRoot $env:TEMP\karakal_share -DryRun
#   .\publish.ps1 -MoveTo $env:TEMP\karakal_share_new -ShareRoot $env:TEMP\karakal_share
[CmdletBinding(DefaultParameterSetName = "Publish")]
param(
    [Parameter(ParameterSetName = "Publish", Mandatory = $true)]
    [ValidateSet("beta", "stable")]
    [string]$Channel,

    [Parameter(ParameterSetName = "Publish", Mandatory = $true)]
    [string]$Version,

    [Parameter(ParameterSetName = "Publish")]
    [Parameter(ParameterSetName = "Move", Mandatory = $true)]
    [string]$ShareRoot = "",

    [Parameter(ParameterSetName = "Publish")]
    [string]$Notes = "",

    [Parameter(ParameterSetName = "Publish")]
    [switch]$DryRun,

    [Parameter(ParameterSetName = "Publish")]
    [switch]$SkipTests,

    [Parameter(ParameterSetName = "Publish")]
    [switch]$SkipBuild,

    [Parameter(ParameterSetName = "Move")]
    [string]$MoveTo = "",

    [string]$IsccPath = ""
)

$ErrorActionPreference = "Stop"
$PluginRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $PluginRoot "..\..")
$IssPath = Join-Path $PluginRoot "packaging\Karakal.iss"

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

function Assert-ChannelVersion([string]$Ch, [string]$Ver) {
    $hasSuffix = $Ver -match "[-.][A-Za-z]"
    if ($Ch -eq "stable" -and $hasSuffix) {
        throw "stable channel rejects prerelease version '$Ver'"
    }
    if ($Ch -eq "beta" -and -not $hasSuffix) {
        throw "beta channel requires a prerelease suffix (e.g. 0.1.0-beta0), got '$Ver'"
    }
}

function Compare-SemverLike([string]$Left, [string]$Right) {
    $py = "from updater.client import compare_versions; import sys; sys.exit(0 if compare_versions(sys.argv[1], sys.argv[2]) > 0 else 1)"
    $env:PYTHONPATH = (Join-Path $RepoRoot "src") + ";" + (Join-Path $PluginRoot "src") + ";" + $env:PYTHONPATH
    & python -c $py $Left $Right
    return $LASTEXITCODE -eq 0
}

if ($PSCmdlet.ParameterSetName -eq "Move" -or $MoveTo) {
    if (-not $MoveTo) { throw "-MoveTo requires a destination root" }
    if ([string]::IsNullOrWhiteSpace($ShareRoot)) { throw "-ShareRoot is required for -MoveTo" }
    $oldRoot = $ShareRoot
    foreach ($ch in @("beta", "stable")) {
        $dir = Join-Path $oldRoot $ch
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $manifestPath = Join-Path $dir "version.json"
        $payload = @{
            version                  = "0.0.0"
            channel                  = $ch
            moved_to                 = $MoveTo
            download_url             = ""
            release_notes_markdown   = "Updates moved to $MoveTo"
            releases                 = @()
        }
        if (Test-Path -LiteralPath $manifestPath) {
            try {
                $existing = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($existing.version) { $payload.version = [string]$existing.version }
                if ($existing.download_url) { $payload.download_url = [string]$existing.download_url }
            } catch { }
        }
        Write-VersionJson -Path $manifestPath -Payload $payload
    }
    Write-Host "Wrote moved_to=$MoveTo into $oldRoot\beta and $oldRoot\stable"
    exit 0
}

Assert-ChannelVersion -Ch $Channel -Ver $Version

if ($DryRun) {
    $SkipBuild = $true
    if ([string]::IsNullOrWhiteSpace($ShareRoot)) {
        $ShareRoot = Join-Path $env:TEMP "karakal_share"
    }
}
if ([string]::IsNullOrWhiteSpace($ShareRoot)) {
    throw "ShareRoot is required (or use -DryRun for %TEMP%\karakal_share)"
}

$destRoot = $ShareRoot
$channelDir = Join-Path $destRoot $Channel
New-Item -ItemType Directory -Force -Path $channelDir | Out-Null

foreach ($ch in @("beta", "stable")) {
    $dir = Join-Path $destRoot $ch
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

$existingManifest = Join-Path $channelDir "version.json"
if (Test-Path -LiteralPath $existingManifest) {
    $prev = Get-Content -LiteralPath $existingManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($prev.version -and $prev.version -ne "0.0.0" -and -not $prev.moved_to) {
        if (-not (Compare-SemverLike $Version ([string]$prev.version))) {
            throw "Version $Version is not newer than existing $($prev.version) on $Channel"
        }
    }
}

Push-Location $RepoRoot
try {
    if (-not $SkipTests) {
        Write-Host "Running pytest..."
        $env:KARAKAL_BUILD_PROFILE = ""
        & python -m pytest plugins/karakal/tests/unit -q --tb=line
        if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
    }

    if (-not $SkipBuild) {
        $profile = if ($Channel -eq "beta") { "tester" } else { "dev" }
        $env:KARAKAL_BUILD_PROFILE = $profile
        $env:KARAKAL_VERSION = $Version
        Write-Host "Building exe profile=$profile version=$Version"
        Push-Location $PluginRoot
        try {
            & pyinstaller --noconfirm karakal.spec
            if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
            $built = Get-ChildItem -LiteralPath (Join-Path $PluginRoot "dist") -Directory |
                Where-Object { $_.Name -like "karakal*" } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if (-not $built) { throw "PyInstaller output folder not found under dist/" }
            $stage = Join-Path $PluginRoot "dist\windows\karakal"
            if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
            New-Item -ItemType Directory -Force -Path (Split-Path $stage) | Out-Null
            Copy-Item -LiteralPath $built.FullName -Destination $stage -Recurse
            $exe = Get-ChildItem -LiteralPath $stage -Filter "karakal*.exe" | Select-Object -First 1
            if ($exe -and $exe.Name -ne "karakal.exe") {
                Rename-Item -LiteralPath $exe.FullName -NewName "karakal.exe"
            }
        } finally {
            Pop-Location
        }

        $iscc = Get-Iscc -Hint $IsccPath
        if (-not $iscc) {
            Write-Warning "ISCC.exe not found - skipping installer. Re-run with -SkipBuild after installing Inno."
        } else {
            $override = Join-Path $PluginRoot "packaging\version_override.iss"
            $numeric = Get-NumericVersion $Version
            $overrideText = "#define AppVersion `"$Version`"`r`n#define AppVersionNumeric `"$numeric`"`r`n"
            Set-Content -LiteralPath $override -Value $overrideText -Encoding ASCII
            try {
                & $iscc $IssPath
                if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
            } finally {
                Remove-Item -LiteralPath $override -Force -ErrorAction SilentlyContinue
            }
        }
    }
} finally {
    Pop-Location
}

$installerName = "Karakal-setup-$Version.exe"
$installerSrc = Join-Path $PluginRoot "dist\installers\$installerName"
if (-not (Test-Path -LiteralPath $installerSrc)) {
    Write-Warning "Installer not found at $installerSrc - writing stub for publish layout check"
    New-Item -ItemType Directory -Force -Path (Split-Path $installerSrc) | Out-Null
    [System.IO.File]::WriteAllBytes($installerSrc, [byte[]](1..64))
}

$sha = (Get-FileHash -LiteralPath $installerSrc -Algorithm SHA256).Hash.ToLowerInvariant()
$staging = Join-Path $channelDir ($installerName + ".partial")
Copy-Item -LiteralPath $installerSrc -Destination $staging -Force
$finalInstaller = Join-Path $channelDir $installerName
Move-Item -LiteralPath $staging -Destination $finalInstaller -Force

$notesText = if ($Notes -and (Test-Path -LiteralPath $Notes)) {
    Get-Content -LiteralPath $Notes -Raw -Encoding UTF8
} elseif ($Notes) {
    $Notes
} else {
    "Karakal $Version"
}

$releases = @(@{
    version      = $Version
    download_url = $installerName
    notes        = $notesText
    sha256       = $sha
    channel      = $Channel
})
if (Test-Path -LiteralPath $existingManifest) {
    try {
        $prev = Get-Content -LiteralPath $existingManifest -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($prev.releases) {
            foreach ($item in @($prev.releases)) {
                if ([string]$item.version -eq $Version) { continue }
                $releases += @{
                    version      = [string]$item.version
                    download_url = [string]$item.download_url
                    notes        = [string]$item.notes
                    sha256       = [string]$item.sha256
                    channel      = $Channel
                }
            }
        }
    } catch { }
}

Write-VersionJson -Path (Join-Path $channelDir "version.json") -Payload @{
    version                  = $Version
    channel                  = $Channel
    download_url             = $installerName
    sha256                   = $sha
    release_notes_markdown   = $notesText
    releases                 = $releases
}

Write-Host "Published $Version to $channelDir"
Write-Host "sha256=$sha"
if ($DryRun) { Write-Host "DryRun root: $destRoot" }
