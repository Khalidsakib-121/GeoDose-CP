$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dest = "D:\GeoDose_Stage3E_N2_N3_Scalable_Certification_v1_0"
function Invoke-NativeChecked {
    param([scriptblock]$Command, [string]$Label)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}
if ($Root -match "\\AppData\\Local\\Temp\\" -or $Root -match "\.zip\.") {
    Write-Host "Detected compressed-folder execution. Relocating to $Dest"
    if (-not (Test-Path "D:\")) { throw "D:\ is unavailable. Use Extract All to a normal folder and run again." }
    foreach ($needed in @("preflight_windows.py","stage3e_run.py","configs\stage3e_contract.yaml","src\geodose_stage3e\runner.py","tests\test_stage3e.py")) {
        if (-not (Test-Path (Join-Path $Root $needed))) { throw "Windows did not materialize the full ZIP contents. Use Extract All before running." }
    }
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    robocopy $Root $Dest /E /R:2 /W:1 /XD ".venv" "outputs_stage3e" /XF "GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip" | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Robocopy relocation failed with exit code $LASTEXITCODE." }
    & "$Dest\RUN_STAGE3E.ps1"
    exit $LASTEXITCODE
}
Set-Location $Root
Write-Host "[PRECHECK] Verifying accepted inputs before environment setup..."
& py -3.10 preflight_windows.py
if ($LASTEXITCODE -ne 0) { throw "Stage3E preflight failed with exit code $LASTEXITCODE" }
if (Test-Path ".venv\Scripts\python.exe") {
    & .venv\Scripts\python.exe verify_runtime.py *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Existing package-local virtual environment is incomplete or incompatible. Rebuilding..."
        Remove-Item -Recurse -Force .venv
    }
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.10 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.10 venv creation failed with exit code $LASTEXITCODE" }
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed with exit code $LASTEXITCODE" }
    & .venv\Scripts\python.exe -m pip install -r requirements_py310.txt
    if ($LASTEXITCODE -ne 0) { throw "dependency installation failed with exit code $LASTEXITCODE" }
}
& .venv\Scripts\python.exe verify_runtime.py
if ($LASTEXITCODE -ne 0) { throw "runtime verification failed" }
& .venv\Scripts\python.exe tests\test_stage3e.py
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
& .venv\Scripts\python.exe stage3e_run.py --overwrite
if ($LASTEXITCODE -ne 0) { throw "Stage3E scientific runner failed" }
& .venv\Scripts\python.exe verify_stage3e.py
if ($LASTEXITCODE -ne 0) { throw "Stage3E output verification failed" }
& .venv\Scripts\python.exe package_stage3e_outputs.py
if ($LASTEXITCODE -ne 0) { throw "Stage3E output packaging failed" }
Write-Host "SUCCESS. Send GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip back to ChatGPT."
Write-Host "Do NOT start the 20-rep pilot or production run until Stage3E output is independently accepted."
