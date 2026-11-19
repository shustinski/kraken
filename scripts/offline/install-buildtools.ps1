#requires -Version 5.1
<#
.SYNOPSIS
  Install Visual C++ Build Tools from an offline layout. Run once, as administrator.

.EXAMPLE
  .\scripts\offline\install-buildtools.ps1 -Layout E:\vs-build-tools-layout
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Layout
)

$vsExe = Join-Path $Layout 'vs_BuildTools.exe'
if (-not (Test-Path -LiteralPath $vsExe)) {
    $vsExe = Join-Path $Layout 'vs_setup.exe'
}

$certificates = Join-Path $Layout 'certificates'
if (Test-Path -LiteralPath $certificates) {
    Get-ChildItem -LiteralPath $certificates -File | Where-Object { $_.Extension -in '.cer', '.crt' } | ForEach-Object {
        & certutil -addstore -f Root $_.FullName | Out-Null
    }
}

$arguments = @(
    '--noWeb', '--noUpdateInstaller', '--wait',
    '--add', 'Microsoft.VisualStudio.Workload.VCTools',
    '--add', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
    '--add', 'Microsoft.VisualStudio.Component.Windows10SDK.19041',
    '--includeRecommended'
)
$installer = Start-Process -FilePath $vsExe -ArgumentList $arguments -Wait -PassThru
exit $installer.ExitCode
