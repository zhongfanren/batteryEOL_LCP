$ErrorActionPreference = 'Stop'
$pythonPath = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$runPath = Join-Path $PSScriptRoot 'run_dscp_local.py'
$plotPath = Join-Path $PSScriptRoot 'plot_dscp_local.py'
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $workspaceRoot
$Host.UI.RawUI.WindowTitle = 'DSCP-Local: 6 frozen-predictor residual encoders'
& $pythonPath -u $runPath
if ($LASTEXITCODE -ne 0) { throw 'DSCP-Local stopped; inspect state.json and console.log.' }
& $pythonPath -u $plotPath
if ($LASTEXITCODE -ne 0) { throw 'Training/evaluation completed, plotting failed; inspect output.' }
Write-Host 'Finished. Read dscp_local_20261111/RESULTS.md and state.json.' -ForegroundColor Green
