# Register the opportunistic option-chain capture (vol_snapshots/auto_capture.py).
#
# Fires every 30 minutes and 3 minutes after logon, as the current user, only
# while logged on, with no console window (pythonw). Runs missed while the
# machine slept start on wake. Outside US market windows each run exits in
# well under a second; see auto_capture.py for the close/intraday logic.
#
#   powershell -ExecutionPolicy Bypass -File vol_snapshots\register_auto_capture.ps1
#   Unregister-ScheduledTask -TaskName options-auto-capture -Confirm:$false   # remove
#
# ASCII only: PowerShell 5.1 misreads non-ASCII in a BOM-less script.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "auto_capture.py"

$pyw = Join-Path $repo ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { throw "pythonw.exe not found (no repo .venv and none on PATH)" }
    $pyw = $cmd.Source
}

$action = New-ScheduledTaskAction -Execute $pyw -Argument ('"' + $script + '"') -WorkingDirectory $repo

$atLogon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$atLogon.Delay = "PT3M"
$every30 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 40) -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName "options-auto-capture" -Action $action `
    -Trigger @($atLogon, $every30) -Settings $settings -Principal $principal `
    -Description "Opportunistic option-chain snapshots (options-toolkit vol_snapshots/auto_capture.py)" `
    -Force | Out-Null

Write-Output "Registered options-auto-capture -> $pyw $script"
Write-Output "Log: $(Join-Path $PSScriptRoot 'capture.log')"
