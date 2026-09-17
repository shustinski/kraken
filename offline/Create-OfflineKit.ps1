[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputPath,

    [string]$ToolchainDirectory,

    # Persistent local cache under offline/ (wheels + uv-cache + cargo vendor).
    [string]$CacheDirectory = "",

    [string]$Python = ".venv\\Scripts\\python.exe",

    [string]$Uv = "uv"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$outputFullPath = $null
$stagingEnvironment = $null

function Invoke-External {
    param([string]$FilePath, [string[]]$Arguments)

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Get-Sha256ManifestEntries {
    param([string]$Root)

    Get-ChildItem -LiteralPath $Root -File -Recurse |
        Where-Object {
            $_.FullName -ne (Join-Path $Root "manifest.json") -and
            $_.Name -notlike "*.partial" -and
            $_.FullName -notlike "*\.staging-venv\*"
        } |
        Sort-Object FullName |
        ForEach-Object {
            $relativePath = $_.FullName.Substring($Root.Length).TrimStart([char]'\', [char]'/').Replace('\', '/')
            [ordered]@{
                path = $relativePath
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                size = $_.Length
            }
        }
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Content)

    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Copy-ToolchainTree {
    param([string]$Source, [string]$Destination)

    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        $target = Join-Path $Destination $_.Name
        Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force
    }
}

function Copy-DirectoryContents {
    param([string]$Source, [string]$Destination)

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Get-ChildItem -LiteralPath $Source -Force -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Recurse -Force
    }
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repositoryRoot
try {
    $status = @(git status --porcelain --untracked-files=all)
    if ($status.Count -gt 0) {
        throw "The worktree is not clean. Commit, stash, or remove every change before creating an offline kit."
    }

    $pythonPath = if ([System.IO.Path]::IsPathRooted($Python)) { $Python } else { Join-Path $repositoryRoot $Python }
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw "Python interpreter was not found: $pythonPath"
    }
    $uvCommand = Get-Command $Uv -ErrorAction Stop

    $outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
    if (Test-Path -LiteralPath $outputFullPath) {
        Write-Host "Resuming existing kit directory: $outputFullPath"
        $staleManifest = Join-Path $outputFullPath "manifest.json"
        if (Test-Path -LiteralPath $staleManifest) {
            Remove-Item -LiteralPath $staleManifest -Force
            Write-Host "Removed previous manifest.json; it will be rewritten when the kit completes."
        }
    } else {
        New-Item -ItemType Directory -Path $outputFullPath | Out-Null
    }

    $cacheFullPath = if ([string]::IsNullOrWhiteSpace($CacheDirectory)) {
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "dep-cache"))
    } else {
        [System.IO.Path]::GetFullPath($CacheDirectory)
    }
    $cacheWheelhouse = Join-Path $cacheFullPath "wheelhouse"
    $cacheUv = Join-Path $cacheFullPath "uv-cache"
    $cacheCargo = Join-Path $cacheFullPath "cargo"
    New-Item -ItemType Directory -Path $cacheWheelhouse, $cacheUv, $cacheCargo -Force | Out-Null
    Write-Host "Persistent dependency cache: $cacheFullPath"

    $sourceDirectory = Join-Path $outputFullPath "source"
    $dependencyDirectory = Join-Path $outputFullPath "dependencies"
    $toolDirectory = Join-Path $outputFullPath "toolchains"
    New-Item -ItemType Directory -Path $sourceDirectory, $dependencyDirectory, $toolDirectory -Force | Out-Null

    $branch = (git branch --show-current).Trim()
    if ([string]::IsNullOrWhiteSpace($branch)) {
        throw "HEAD is detached. Check out a branch before creating the kit."
    }
    $commit = (git rev-parse HEAD).Trim()
    $bundleRefs = @(git for-each-ref --format="%(refname)" refs/heads refs/tags)
    if ($bundleRefs.Count -eq 0) {
        throw "No local branches or tags are available to bundle."
    }

    $bundlePath = Join-Path $sourceDirectory "kraken-full.bundle"
    Write-Host "Creating Git bundle..."
    Invoke-External git (@("bundle", "create", $bundlePath) + $bundleRefs)
    Invoke-External git @("bundle", "verify", $bundlePath)
    $headsPath = Join-Path $sourceDirectory "bundle-heads.txt"
    Write-Utf8NoBom -Path $headsPath -Content ((& git bundle list-heads $bundlePath) -join [Environment]::NewLine)

    # This is an auditable, exact dependency list. Local workspace packages are restored from the bundle.
    $requirementsPath = Join-Path $dependencyDirectory "requirements-windows-x64.txt"
    Write-Host "Exporting locked requirements..."
    Invoke-External $uvCommand.Source @(
        "export", "--locked", "--all-packages", "--all-extras", "--all-groups", "--no-emit-local",
        "--output-file", $requirementsPath
    )
    $wheelhouse = Join-Path $dependencyDirectory "wheelhouse"
    Write-Host "Building wheelhouse (uses $cacheWheelhouse; downloads only missing/changed)..."
    Invoke-External $pythonPath @(
        (Join-Path $PSScriptRoot "build_wheelhouse.py"),
        "--lock", (Join-Path $repositoryRoot "uv.lock"),
        "--cache", $cacheWheelhouse,
        "--output", $wheelhouse
    )

    # Populate UV cache on this PC, then copy into the kit.
    $uvCache = Join-Path $dependencyDirectory "uv-cache"
    $stagingEnvironment = Join-Path $outputFullPath ".staging-venv"
    if (Test-Path -LiteralPath $stagingEnvironment) {
        Remove-Item -LiteralPath $stagingEnvironment -Recurse -Force
    }
    $previousCache = $env:UV_CACHE_DIR
    $previousEnvironment = $env:UV_PROJECT_ENVIRONMENT
    try {
        $env:UV_CACHE_DIR = $cacheUv
        $env:UV_PROJECT_ENVIRONMENT = $stagingEnvironment
        Write-Host "Populating UV cache via sync (reuses $cacheUv)..."
        Invoke-External $uvCommand.Source @(
            "sync", "--frozen", "--all-packages", "--all-extras", "--all-groups", "--link-mode=copy",
            "--python", $pythonPath
        )
    } finally {
        $env:UV_CACHE_DIR = $previousCache
        $env:UV_PROJECT_ENVIRONMENT = $previousEnvironment
        if (Test-Path -LiteralPath $stagingEnvironment) {
            Remove-Item -LiteralPath $stagingEnvironment -Recurse -Force
        }
        $stagingEnvironment = $null
    }
    Write-Host "Copying UV cache into the kit..."
    if (Test-Path -LiteralPath $uvCache) {
        Remove-Item -LiteralPath $uvCache -Recurse -Force
    }
    Copy-DirectoryContents -Source $cacheUv -Destination $uvCache

    # Cargo vendor is deliberately generated from Cargo.lock so an offline build cannot update Rust dependencies.
    $cargo = Get-Command cargo -ErrorAction SilentlyContinue
    if ($null -eq $cargo) {
        $cargoPath = Join-Path $env:USERPROFILE ".cargo\\bin\\cargo.exe"
        if (Test-Path -LiteralPath $cargoPath -PathType Leaf) {
            $cargo = [pscustomobject]@{ Source = $cargoPath }
        } else {
            throw "cargo was not found. Install the pinned Rust toolchain, then run this script again."
        }
    }
    $cacheCargoVendor = Join-Path $cacheCargo "vendor"
    $cacheCargoConfig = Join-Path $cacheCargo "config.toml"
    $cacheCargoLockStamp = Join-Path $cacheCargo "Cargo.lock.sha256"
    $cargoLockPath = Join-Path $repositoryRoot "blob_gateway\\Cargo.lock"
    $cargoLockHash = (Get-FileHash -LiteralPath $cargoLockPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $cargoConfigDirectory = Join-Path $dependencyDirectory "cargo"
    New-Item -ItemType Directory -Path $cacheCargo, $cargoConfigDirectory -Force | Out-Null
    $cargoCacheFresh = (
        (Test-Path -LiteralPath $cacheCargoVendor) -and
        (Test-Path -LiteralPath $cacheCargoConfig) -and
        (Test-Path -LiteralPath $cacheCargoLockStamp) -and
        ((Get-Content -LiteralPath $cacheCargoLockStamp -Raw).Trim() -eq $cargoLockHash)
    )
    if ($cargoCacheFresh) {
        Write-Host "Cargo vendor already in cache for this Cargo.lock; copying into the kit."
    } else {
        if (Test-Path -LiteralPath $cacheCargoVendor) {
            Remove-Item -LiteralPath $cacheCargoVendor -Recurse -Force
        }
        Write-Host "Vendoring Cargo dependencies into cache..."
        $null = & $cargo.Source vendor --locked --manifest-path (Join-Path $repositoryRoot "blob_gateway\\Cargo.toml") $cacheCargoVendor
        if ($LASTEXITCODE -ne 0) {
            throw "cargo vendor failed with exit code $LASTEXITCODE"
        }
        @(
            "[source.crates-io]",
            "replace-with = 'offline-vendor'",
            "",
            "[source.offline-vendor]",
            "directory = 'vendor'"
        ) | ForEach-Object { $_ } | Out-String | ForEach-Object {
            Write-Utf8NoBom -Path $cacheCargoConfig -Content $_
        }
        Write-Utf8NoBom -Path $cacheCargoLockStamp -Content $cargoLockHash
    }
    Copy-DirectoryContents -Source $cacheCargo -Destination $cargoConfigDirectory

    if ([string]::IsNullOrWhiteSpace($ToolchainDirectory)) {
        throw "ToolchainDirectory is required. It must contain the pinned offline installers and the VS Build Tools layout listed in offline/README.md."
    }
    $toolchainSource = (Resolve-Path -LiteralPath $ToolchainDirectory).Path
    $toolchainManifestPath = Join-Path $toolchainSource "toolchain-manifest.json"
    if (-not (Test-Path -LiteralPath $toolchainManifestPath -PathType Leaf)) {
        throw "ToolchainDirectory must contain toolchain-manifest.json. Start from offline/toolchain-manifest.example.json."
    }
    $toolchainManifest = Get-Content -LiteralPath $toolchainManifestPath -Raw | ConvertFrom-Json
    $requiredToolchainRoles = @("python", "uv", "git", "git-lfs", "rust", "vs-build-tools-layout", "inno-setup")
    $declaredRoles = @($toolchainManifest.files | ForEach-Object { $_.role })
    foreach ($role in $requiredToolchainRoles) {
        if ($declaredRoles -notcontains $role) {
            throw "toolchain-manifest.json does not declare required role '$role'."
        }
    }
    foreach ($entry in $toolchainManifest.files) {
        if ([string]::IsNullOrWhiteSpace($entry.path) -or [string]::IsNullOrWhiteSpace($entry.version)) {
            throw "Each toolchain manifest entry must have path and version."
        }
        $entryPath = Join-Path $toolchainSource $entry.path
        if (-not (Test-Path -LiteralPath $entryPath)) {
            throw "Toolchain manifest entry is missing: $($entry.path)"
        }
    }
    Write-Host "Copying toolchain installers into the kit..."
    Copy-ToolchainTree -Source $toolchainSource -Destination $toolDirectory

    $layoutCerts = Join-Path $toolDirectory "vs-build-tools-layout\\certificates"
    $extraCerts = Join-Path $PSScriptRoot "vs-build-tools-extra-certs"
    if ((Test-Path -LiteralPath $layoutCerts) -and (Test-Path -LiteralPath $extraCerts)) {
        Get-ChildItem -LiteralPath $extraCerts -File | Where-Object { $_.Extension -in '.cer', '.crt' } | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $layoutCerts $_.Name) -Force
        }
        Write-Host "Merged extra VS layout certificates into toolchains\\vs-build-tools-layout\\certificates."
    }

    # Instructions and installer must sit in the kit root (not only inside the git bundle).
    $readmeSource = Join-Path $PSScriptRoot "README.md"
    $installSource = Join-Path $PSScriptRoot "Install-OfflineKit.ps1"
    Copy-Item -LiteralPath $readmeSource -Destination (Join-Path $outputFullPath "README.md") -Force
    Copy-Item -LiteralPath $installSource -Destination (Join-Path $outputFullPath "Install-OfflineKit.ps1") -Force

    $lfsFiles = @(git lfs ls-files -n)
    if ($lfsFiles.Count -gt 0) {
        $lfsArchive = Join-Path $sourceDirectory "lfs-objects.tar"
        $lfsStorage = (git rev-parse --git-path lfs/objects).Trim()
        if (-not (Test-Path -LiteralPath $lfsStorage)) {
            throw "Git LFS tracks files, but its object store is unavailable: $lfsStorage"
        }
        Invoke-External tar @("-cf", $lfsArchive, "-C", (Split-Path $lfsStorage -Parent), (Split-Path $lfsStorage -Leaf))
        Write-Utf8NoBom -Path (Join-Path $sourceDirectory "lfs-files.txt") -Content ($lfsFiles -join [Environment]::NewLine)
    }

    $pythonVersion = (& $pythonPath --version).Trim()
    $uvVersion = (& $uvCommand.Source --version).Trim()
    $rustVersion = (& $cargo.Source --version).Trim()
    $manifest = [ordered]@{
        formatVersion = 1
        createdUtc = (Get-Date).ToUniversalTime().ToString("o")
        platform = "windows-x64"
        repository = [ordered]@{
            commit = $commit
            defaultBranch = $branch
            refs = @(git for-each-ref --format="%(refname)" refs/heads refs/tags)
        }
        tools = [ordered]@{
            python = $pythonVersion
            uv = $uvVersion
            rust = $rustVersion
        }
        toolchainManifest = $toolchainManifest
        lfsObjectArchiveIncluded = ($lfsFiles.Count -gt 0)
        files = @(Get-Sha256ManifestEntries -Root $outputFullPath)
    }
    Write-Utf8NoBom -Path (Join-Path $outputFullPath "manifest.json") -Content ($manifest | ConvertTo-Json -Depth 8)
    Write-Host "Offline kit created: $outputFullPath"
} catch {
    if ($null -ne $stagingEnvironment -and (Test-Path -LiteralPath $stagingEnvironment)) {
        Remove-Item -LiteralPath $stagingEnvironment -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $outputFullPath -and (Test-Path -LiteralPath $outputFullPath)) {
        Write-Host "Kit left incomplete at $outputFullPath. Re-run with the same -OutputPath to resume."
    }
    throw
} finally {
    Pop-Location
}
