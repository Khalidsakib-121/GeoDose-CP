$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Local = Get-Content (Join-Path $Root "configs\local_paths_windows.json") -Raw | ConvertFrom-Json
$Base = $Local.base_acquisition_root
$DefaultBase = "D:\GeoDose_Stage5A_External_Context_Acquisition_v1_0_1_FREEZE\GeoDose_Stage5A_External_Context_Acquisition_v1_0_1_FREEZE"
if ([string]::IsNullOrWhiteSpace($Base)) { if (Test-Path $DefaultBase) {$Base=$DefaultBase} else {$Base=Read-Host "Paste inner acquisition v1.0.1 folder"} }
$Base=$Base.Trim('"');$Base=(Resolve-Path $Base).Path;$Py=Join-Path $Base ".venv\Scripts\python.exe"
$RunArgs=@((Join-Path $Root "verify_final_context_only.py"),"--base-root",$Base)
if (![string]::IsNullOrWhiteSpace($Local.output_dir)) {$RunArgs+=@("--output-dir",$Local.output_dir)}
& $Py @RunArgs
if ($LASTEXITCODE -ne 0) {throw "Final context verification failed."}
