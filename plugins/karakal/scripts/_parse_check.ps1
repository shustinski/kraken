$ wh = $null
$errs = $null
[void][System.Management.Automation.Language.Parser]::ParseFile(
    'D:\WORK\git\kraken\plugins\karakal\scripts\publish.ps1',
    [ref]$wh,
    [ref]$errs
)
if (-not $errs -or $errs.Count -eq 0) {
    Write-Host 'PARSE_OK'
} else {
    foreach ($e in $errs) {
        Write-Host ("{0}:{1} {2}" -f $e.Extent.StartLineNumber, $e.Extent.StartColumnNumber, $e.Message)
    }
}
