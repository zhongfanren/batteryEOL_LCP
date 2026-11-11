param(
    [Parameter(Mandatory = $true)][int]$DownloadPid,
    [string]$LogName = 'stage_stdout.log',
    [string]$ProcessLabel = 'downloader'
)

$logPath = Join-Path $PSScriptRoot $LogName
$errorPath = Join-Path $PSScriptRoot ($LogName -replace '_stdout\.log$', '_stderr.log')
$stateFile = if ($LogName -eq 'no_matr_dlinear_console.log') {
    'heldout_pipeline_state.json'
} elseif ($LogName -like 'cpmlp_*_console.log') {
    'cpmlp_pipeline_state.json'
} else {
    'pipeline_state.json'
}
$statePath = Join-Path $PSScriptRoot $stateFile
$hfStatePath = Join-Path $PSScriptRoot 'hf_download_state.json'
$Host.UI.RawUI.WindowTitle = "BatteryLife $ProcessLabel monitor - $LogName"

Write-Host "BatteryLife $ProcessLabel monitor (PID $DownloadPid)"
Write-Host "Log: $logPath"
Write-Host "Updates every 10 seconds. Closing this monitor does not stop $ProcessLabel."

while ($true) {
    $process = Get-Process -Id $DownloadPid -ErrorAction SilentlyContinue
    $lastLine = Get-Content -LiteralPath $logPath -Tail 1 -ErrorAction SilentlyContinue
    $state = Get-Content -LiteralPath $statePath -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
    $hfState = if ($LogName -eq 'hf_stdout.log') {
        Get-Content -LiteralPath $hfStatePath -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
    }
    $stamp = Get-Date -Format 'HH:mm:ss'
    if ($process) {
        Write-Host "[$stamp] $ProcessLabel running; pipeline=$($state.state)"
    } else {
        Write-Host "[$stamp] $ProcessLabel stopped; pipeline=$($state.state)"
    }
    if ($lastLine) { Write-Host $lastLine }
    if ($hfState) {
        Write-Host "HF verified: $($hfState.verified_files)/$($hfState.total_files) files; $($hfState.verified_gb)/$($hfState.total_gb) GB"
    }
    if (-not $process) {
        Get-Content -LiteralPath $errorPath -Tail 10 -ErrorAction SilentlyContinue
        break
    }
    Start-Sleep -Seconds 10
}
