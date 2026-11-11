$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $workspaceRoot
$Host.UI.RawUI.WindowTitle = 'ISU stability: 3 splits x 3 seeds x 2 predictors (3h cap)'
Write-Host 'ISU stability batch: 18 predictor settings; 2 existing fits reused.'
Write-Host 'Each fit is frozen before CP analysis. All preset methods and bandwidths are reported.'
Write-Host 'Progress: isu_stability_20261028\state.json; results: SUMMARY.md'
& 'D:\Anaconda3\envs\my_torch\python.exe' -u (Join-Path $PSScriptRoot 'run_isu_stability.py') --queue
if ($LASTEXITCODE -ne 0) {
    Write-Host "Batch exited with code $LASTEXITCODE. Inspect state.json and analysis\job\console.log." -ForegroundColor Red
} else {
    Write-Host 'Batch stopped normally; SUMMARY.md includes all completed fits. Check state.json for complete vs partial.' -ForegroundColor Green
}
