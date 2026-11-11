param([Parameter(Mandatory = $true)][int]$QueuePid)

$statePath = Join-Path $PSScriptRoot 'remaining_models_state.json'
$Host.UI.RawUI.WindowTitle = 'BatteryLife remaining main-table models'
Write-Host "BatteryLife model queue PID $QueuePid"
Write-Host "State: $statePath"
Write-Host 'Updates every 15 seconds. Closing this monitor does not stop training.'
while ($true) {
    $queue = Get-Process -Id $QueuePid -ErrorAction SilentlyContinue
    $state = Get-Content -LiteralPath $statePath -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
    $stamp = Get-Date -Format 'HH:mm:ss'
    $name = $state.current_model
    if ($name) {
        $entry = $state.models.$name
        Write-Host "[$stamp] $name; stage=$($entry.last_stage); epoch=$($entry.last_epoch); val MAPE=$($entry.last_val_mape); queue=$($state.state)"
    } else {
        Write-Host "[$stamp] queue=$($state.state)"
    }
    if (-not $queue) { break }
    Start-Sleep -Seconds 15
}
