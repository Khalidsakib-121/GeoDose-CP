$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dest = "D:\GeoDose_Stage3F_Pilot_S1_S10_20Rep_Threshold_Freeze_v1_0"
if ($Root -match "\\AppData\\Local\\Temp\\" -or $Root -match "\.zip\.") {
    Write-Host "Detected compressed-folder execution. Relocating to $Dest"
    if (-not (Test-Path "D:\")) { throw "D:\ is unavailable. Use Extract All to a normal folder and run again." }
    foreach ($needed in @("preflight_windows.py","stage3f_run.py","configs\pilot_contract.yaml","src\geodose_stage3f\runner.py","tests\test_stage3f.py")) {
        if (-not (Test-Path (Join-Path $Root $needed))) { throw "Windows did not materialize the full ZIP contents. Use Extract All before running." }
    }
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    robocopy $Root $Dest /E /R:2 /W:1 /XD ".venv" "outputs_stage3f" /XF "GeoDose_Stage3F_PILOT_S1_S10_20REP_THRESHOLD_FREEZE_OUTPUTS.zip" | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Robocopy relocation failed with exit code $LASTEXITCODE." }
    & "$Dest\RUN_STAGE3F.ps1"
    exit $LASTEXITCODE
}
Set-Location $Root
Write-Host "[PRECHECK] Verifying accepted archives and exact source trees before environment setup..."
& py -3.10 preflight_windows.py
if ($LASTEXITCODE -ne 0) { throw "Stage3F preflight failed with exit code $LASTEXITCODE" }
if (Test-Path ".venv\Scripts\python.exe") {
    & .venv\Scripts\python.exe verify_runtime.py *> $null
    if ($LASTEXITCODE -ne 0) { Write-Host "Rebuilding incompatible package-local virtual environment..."; Remove-Item -Recurse -Force .venv }
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.10 -m venv .venv; if ($LASTEXITCODE -ne 0) { throw "Python 3.10 venv creation failed" }
    & .venv\Scripts\python.exe -m pip install --upgrade pip; if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & .venv\Scripts\python.exe -m pip install -r requirements_py310.txt; if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
}
& .venv\Scripts\python.exe verify_runtime.py; if ($LASTEXITCODE -ne 0) { throw "runtime verification failed" }
& .venv\Scripts\python.exe tests\test_stage3f.py; if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
& .venv\Scripts\python.exe stage3f_run.py --overwrite; if ($LASTEXITCODE -ne 0) { throw "Stage3F pilot runner failed" }
& .venv\Scripts\python.exe verify_stage3f.py; if ($LASTEXITCODE -ne 0) { throw "Stage3F output verification failed" }
& .venv\Scripts\python.exe package_stage3f_outputs.py; if ($LASTEXITCODE -ne 0) { throw "Stage3F output packaging failed" }
Write-Host "SUCCESS. Send GeoDose_Stage3F_PILOT_S1_S10_20REP_THRESHOLD_FREEZE_OUTPUTS.zip back to ChatGPT."
Write-Host "Do NOT start production until the Stage3F output is independently accepted."
