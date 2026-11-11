$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $workspaceRoot
$Host.UI.RawUI.WindowTitle = 'ISU-ILCC: CPMLP -> CPLSTM -> CP/LCP reports'
Write-Host 'Fixed ISU-ILCC split: train 97 / dev 20 / calibration 61 / test 61'
Write-Host 'Models run serially on the GPU; each model stops early using development MAPE.'
Write-Host 'Progress and reports: simple_cptransformer_cp\batterylife_main_table\curated_isu_pilot'
& python -u (Join-Path $PSScriptRoot 'run_curated_isu.py') --queue
if ($LASTEXITCODE -ne 0) {
    Write-Host "Queue exited with code $LASTEXITCODE. Inspect queue_state.json and the model log." -ForegroundColor Red
} else {
    Write-Host 'Queue finished. Open curated_isu_pilot\COMPARISON.md and each model REPORT.md.' -ForegroundColor Green
}
