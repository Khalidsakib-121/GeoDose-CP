$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "GeoDose-CP Stage5A External Context Production v1.0.1 - FREEZE CANDIDATE"
Write-Host "This run uses only the exact source bytes already frozen by the successful support census."
Write-Host "It does NOT redownload/reselect sources, impute the 92 exclusions, change 0.99, or read MineDoseBench outcomes/M2-M6 performance."

$Local = Get-Content (Join-Path $Root "configs\local_paths_windows.json") -Raw | ConvertFrom-Json
$Base = $Local.base_acquisition_root
$DefaultBase = "D:\GeoDose_Stage5A_External_Context_Acquisition_v1_0_1_FREEZE\GeoDose_Stage5A_External_Context_Acquisition_v1_0_1_FREEZE"
if ([string]::IsNullOrWhiteSpace($Base)) {
  if (Test-Path (Join-Path $DefaultBase "PACKAGE_MANIFEST.json")) { $Base = $DefaultBase }
  else {
    Write-Host "Paste the existing INNER acquisition v1.0.1 folder containing .venv and CONTEXT_SOURCE_CACHE:"
    $Base = Read-Host
  }
}
$Base = $Base.Trim('"')
$Base = (Resolve-Path $Base).Path
$Py = Join-Path $Base ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "Frozen acquisition Python environment not found: $Py" }
if (!(Test-Path (Join-Path $Base "CONTEXT_SOURCE_CACHE"))) { throw "CONTEXT_SOURCE_CACHE not found. Do not redownload; point to the exact folder used by the successful support census." }
& $Py (Join-Path $Root "verify_package.py")
if ($LASTEXITCODE -ne 0) { throw "Production package verification failed." }
& $Py (Join-Path $Base "verify_runtime.py")
if ($LASTEXITCODE -ne 0) { throw "Frozen runtime verification failed." }
& $Py (Join-Path $Root "tests\run_tests.py") --base-root $Base
if ($LASTEXITCODE -ne 0) { throw "Production unit tests failed." }
$Out = $Local.output_dir
$RunArgs = @((Join-Path $Root "final_context_production.py"), "--base-root", $Base)
if (![string]::IsNullOrWhiteSpace($Out)) { $RunArgs += @("--output-dir", $Out) }
Write-Host "[PRODUCTION] Freezing 23,618 census-authorized context-complete blocks and computing the final 13 covariates..."
& $Py @RunArgs
if ($LASTEXITCODE -ne 0) { throw "Final context production failed. Existing source cache and production checkpoints are retained." }
Write-Host "FINAL CONTEXT PRODUCTION PASSED. Stop here and send GeoDose_Stage5A_EXTERNAL_CONTEXT_FINAL_OUTPUTS.zip for independent audit."
Write-Host "DO NOT run the unchanged MineDoseBench Generator v1.1.0; it still requires 23,710 rows and must be prospectively revised next."
