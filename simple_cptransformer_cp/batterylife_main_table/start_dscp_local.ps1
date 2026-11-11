$ErrorActionPreference = 'Stop'
$pythonPath = 'D:/Anaconda3/envs/my_torch/python.exe'
$runPath = Join-Path $PSScriptRoot 'run_dscp_local.py'
$plotPath = Join-Path $PSScriptRoot 'plot_dscp_local.py'
Set-Location -LiteralPath 'D:/Battery-LCP'
$Host.UI.RawUI.WindowTitle = 'DSCP-Local: 6 frozen-predictor residual encoders'
& $pythonPath -u $runPath
if ($LASTEXITCODE -ne 0) { throw 'DSCP-Local stopped; inspect state.json and console.log.' }
& $pythonPath -u $plotPath
if ($LASTEXITCODE -ne 0) { throw 'Training/evaluation completed, plotting failed; inspect output.' }
Write-Host 'Finished. Read dscp_local_20261111/RESULTS.md and state.json.' -ForegroundColor Green
