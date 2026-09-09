$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
$Dest="D:\GeoDose_Stage4_Production_F06_Ablation_Efficiency_v1_0"
if ($Root -match "\\AppData\\Local\\Temp\\" -or $Root -match "\.zip\.") {
  Write-Host "Detected compressed-folder execution. Relocating to $Dest"
  if (-not (Test-Path "D:\")) { throw "D:\ is unavailable. Use Extract All to a normal folder." }
  foreach($n in @("preflight_windows.py","stage4_run.py","configs\production_contract.yaml","src\geodose_stage4\runner.py","tests\test_stage4.py")){if(-not(Test-Path(Join-Path $Root $n))){throw "Full ZIP contents are not materialized. Use Extract All."}}
  New-Item -ItemType Directory -Force -Path $Dest|Out-Null
  robocopy $Root $Dest /E /R:2 /W:1 /XD ".venv" "outputs_stage4" /XF "GeoDose_Stage4_PRODUCTION_F06_ABLATION_EFFICIENCY_OUTPUTS.zip"|Out-Null
  if($LASTEXITCODE -ge 8){throw "Robocopy relocation failed: $LASTEXITCODE"}
  & "$Dest\RUN_STAGE4.ps1";exit $LASTEXITCODE
}
Set-Location $Root
Write-Host "[PRECHECK] Verifying frozen inputs before environment setup...";& py -3.10 preflight_windows.py;if($LASTEXITCODE -ne 0){throw "preflight failed"}
if(Test-Path ".venv\Scripts\python.exe"){& .venv\Scripts\python.exe verify_runtime.py *> $null;if($LASTEXITCODE -ne 0){Remove-Item -Recurse -Force .venv}}
if(-not(Test-Path ".venv\Scripts\python.exe")){& py -3.10 -m venv .venv;if($LASTEXITCODE -ne 0){throw "venv failed"};& .venv\Scripts\python.exe -m pip install --upgrade pip;if($LASTEXITCODE -ne 0){throw "pip failed"};& .venv\Scripts\python.exe -m pip install -r requirements_py310.txt;if($LASTEXITCODE -ne 0){throw "dependencies failed"}}
& .venv\Scripts\python.exe verify_runtime.py;if($LASTEXITCODE -ne 0){throw "runtime verification failed"}
& .venv\Scripts\python.exe tests\test_stage4.py;if($LASTEXITCODE -ne 0){throw "unit tests failed"}
Write-Host "Long production run begins. Checkpoints make it interruption-safe.";& .venv\Scripts\python.exe stage4_run.py;if($LASTEXITCODE -ne 0){throw "Stage4 runner failed; rerun to resume after correcting Stage4 only"}
& .venv\Scripts\python.exe verify_stage4.py;if($LASTEXITCODE -ne 0){throw "verification failed"}
& .venv\Scripts\python.exe package_stage4_outputs.py;if($LASTEXITCODE -ne 0){throw "packaging failed"}
Write-Host "SUCCESS. Send GeoDose_Stage4_PRODUCTION_F06_ABLATION_EFFICIENCY_OUTPUTS.zip back to ChatGPT. Do not start MineDoseBench yet."
