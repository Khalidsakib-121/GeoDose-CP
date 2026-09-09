$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  & py -3.10 -m venv .venv
  & .venv\Scripts\python.exe -m pip install --upgrade pip
  & .venv\Scripts\python.exe -m pip install -r requirements_py310.txt
}
& .venv\Scripts\python.exe verify_runtime.py
& .venv\Scripts\python.exe tests\test_integration.py
& .venv\Scripts\python.exe stage3d_d3_integration_run.py --overwrite
& .venv\Scripts\python.exe verify_stage3d_d3_integration.py
& .venv\Scripts\python.exe package_stage3d_d3_integration_outputs.py
Write-Host "SUCCESS. Send GeoDose_Stage3D_D3_M1_M6_EXACT_INTEGRATION_OUTPUTS.zip back to ChatGPT."
Write-Host "Do NOT start the 20-rep pilot yet."
