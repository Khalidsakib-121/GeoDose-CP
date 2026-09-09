$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$tempLike = ($root -match '\\AppData\\Local\\Temp\\') -or ($root -match '\.zip\.')

if ($tempLike) {
    Write-Host "Detected execution from a ZIP/compressed-folder temporary location."
    Write-Host "Relocating the package to D:\ before creating the virtual environment."
    if (-not (Test-Path 'D:\')) { throw 'D:\ is not available. Extract the ZIP manually to D:\ and run again.' }
    $required = @(
        'preflight_windows.py','stage3d_d4_run.py','requirements_py310.txt','verify_runtime.py','verify_stage3d_d4.py',
        'configs\d4_contract.yaml','src\geodose_stage3d_d4\runner.py','tests\test_d4.py'
    )
    foreach ($rel in $required) {
        if (-not (Test-Path (Join-Path $root $rel))) {
            throw "Windows did not materialize the complete ZIP package; missing $rel. Use Extract All to D:\."
        }
    }
    $dest = 'D:\GeoDose_Stage3D_D4_M6_NewData_Nuisance_Integration_v1_1_0'
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Get-ChildItem -LiteralPath $root -Force | Where-Object { $_.Name -notin @('.venv','outputs_stage3d_d4','GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip') } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
    }
    if (-not (Test-Path (Join-Path $dest 'stage3d_d4_run.py'))) { throw 'Relocation verification failed.' }
    & (Join-Path $dest 'RUN_STAGE3D_D4.ps1')
    exit $LASTEXITCODE
}

Set-Location $root
Write-Host "[PRECHECK] Verifying frozen Windows paths, hashes, ZIP integrity, and source trees before environment setup..."
& py -3.10 preflight_windows.py
if ($LASTEXITCODE -ne 0) { throw "D4 Windows input preflight failed." }
if (Test-Path '.venv\Scripts\python.exe') {
    & .venv\Scripts\python.exe verify_runtime.py *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Existing package-local virtual environment is incomplete or incompatible. Rebuilding...'
        Remove-Item -Recurse -Force '.venv'
    }
}
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host "Creating package-local Python 3.10 virtual environment..."
    py -3.10 -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements_py310.txt
}
Write-Host "[1/5] Verifying frozen Python 3.10 runtime..."
.venv\Scripts\python.exe verify_runtime.py
Write-Host "[2/5] Running D4 unit tests..."
.venv\Scripts\python.exe tests\test_d4.py
Write-Host "[3/5] Running source-level M6 new-data + nuisance integration gate..."
.venv\Scripts\python.exe stage3d_d4_run.py --overwrite
Write-Host "[4/5] Independently verifying D4 outputs..."
.venv\Scripts\python.exe verify_stage3d_d4.py
Write-Host "[5/5] Packaging verified outputs..."
.venv\Scripts\python.exe package_stage3d_d4_outputs.py
Write-Host "SUCCESS. Send GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip back to ChatGPT."
Write-Host "Do NOT start Stage3E or the 20-rep pilot until independent review."
Read-Host "Press Enter to close"
