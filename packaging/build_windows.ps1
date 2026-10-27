param(
    [switch]$SkipInstallers,
    [ValidateSet('Server', 'Desktop')][string[]]$Targets = @('Server', 'Desktop'),
    [string]$PythonExecutable = '',
    [string]$CargoExecutable = '',
    [string]$InnoCompilerExecutable = '',
    [string]$OutputRoot = ''
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepositoryRoot 'dist\windows' }
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (-not $PythonExecutable) {
    $WorkspacePython = Join-Path $RepositoryRoot '.venv\Scripts\python.exe'
    $PythonExecutable = if (Test-Path -LiteralPath $WorkspacePython) { $WorkspacePython } else { (Get-Command python.exe -ErrorAction Stop).Source }
}
if (-not $SkipInstallers -and -not $InnoCompilerExecutable) {
    $InnoCompilerExecutable = (Get-Command ISCC.exe -ErrorAction Stop).Source
}
if ('Server' -in $Targets) {
    if (-not $CargoExecutable) {
        $Cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
        $CargoExecutable = if ($null -ne $Cargo) { $Cargo.Source } else { Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe' }
    }
    if (-not (Test-Path -LiteralPath $CargoExecutable)) { throw 'Rust toolchain is required to build KrakenBlobGateway.exe' }
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $VsWhere)) { throw 'Visual Studio Build Tools and Windows SDK are required' }
    $VisualStudio = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $VisualStudio) { throw 'Visual Studio C++ Build Tools are required' }
    $VcVars = Join-Path $VisualStudio 'VC\Auxiliary\Build\vcvars64.bat'
    $CargoCommand = 'call "' + $VcVars + '" >nul && "' + $CargoExecutable + '" build --locked --release --manifest-path "' + (Join-Path $RepositoryRoot 'blob_gateway\Cargo.toml') + '"'
    & cmd.exe /d /s /c $CargoCommand
    if ($LASTEXITCODE -ne 0) { throw "Kraken Blob Gateway build failed with exit code $LASTEXITCODE" }
}

Push-Location $PSScriptRoot
try {
    foreach ($Target in $Targets) {
        & $PythonExecutable -c 'from scipy.ndimage import label; import numpy, PIL, PyQt6, psutil'
        if ($LASTEXITCODE -ne 0) { throw 'Install agent/registration/electrical extras and PyQt6 before building; an incomplete distribution is not accepted' }
        & $PythonExecutable -m PyInstaller --noconfirm --clean --distpath $OutputRoot --workpath (Join-Path $RepositoryRoot "build\pyinstaller-$($Target.ToLowerInvariant())") "Kraken$Target.spec"
        if ($LASTEXITCODE -ne 0) { throw "Kraken $Target packaging failed with exit code $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}

if (-not $SkipInstallers) {
    foreach ($Target in $Targets) {
        & $InnoCompilerExecutable "/DBuildRoot=$OutputRoot" (Join-Path $PSScriptRoot "installer\Kraken$Target.iss")
        if ($LASTEXITCODE -ne 0) { throw "Kraken $Target installer compilation failed with exit code $LASTEXITCODE" }
    }
}
