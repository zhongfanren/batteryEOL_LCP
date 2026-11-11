$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $workspaceRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$Host.UI.RawUI.WindowTitle = 'ISU train-only crossfit: 5 folds x 2 models; 90 minute cap'
Write-Host 'Original main predictors and calibration/test membership remain frozen.'
Write-Host '10 auxiliary jobs: inner epoch selection -> outer refit -> held-out T100 predictions.'
Write-Host 'Automatic risk and CP/LCP evaluation follows all 10 completed jobs.'
Write-Host 'Progress: simple_cptransformer_cp\batterylife_main_table\isu_crossfit_risk_20261028\state.json'
& python -u (Join-Path $PSScriptRoot 'run_isu_crossfit_risk.py') --queue
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Queue failed. Inspect state.json and folds\job\console.log.' -ForegroundColor Red
} else {
    Write-Host 'Queue stopped. Check state.json for complete vs timeout/partial; BRIEF.md appears after complete evaluation.' -ForegroundColor Green
}
