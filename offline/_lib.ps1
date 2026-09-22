# Shared helpers for offline numbered scripts.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-OfflineRoot {
    return $PSScriptRoot
}

function Read-PathOrDefault {
    param(
        [Parameter(Mandatory)][string]$Prompt,
        [Parameter(Mandatory)][string]$Default
    )
    $value = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value.Trim().Trim('"')
}

function Assert-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Запустите этот скрипт от имени администратора (ПКМ по PowerShell -> Запуск от имени администратора)."
    }
}

function Invoke-External {
    param([string]$FilePath, [string[]]$Arguments = @())
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Команда завершилась с кодом $LASTEXITCODE : $FilePath $($Arguments -join ' ')"
    }
}

function Resolve-FirstExisting {
    param([string[]]$Candidates)
    foreach ($path in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return $null
}
