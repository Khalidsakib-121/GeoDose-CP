$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$preferred = "D:\GeoDose_Stage2B_Local_STAC_v1_0\.venv\Scripts\python.exe"
if (Test-Path $preferred) {
    $py = $preferred
} else {
    $py = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "Python: $py"
$pyVersion = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pyVersion.Trim() -ne "3.10") {
    throw "Stage 3B requires Python 3.10; found $pyVersion"
}

& $py -c "import numpy,pandas; assert numpy.__version__=='2.2.6', numpy.__version__; assert pandas.__version__=='2.3.3', pandas.__version__; print('Frozen package versions passed')"
if ($LASTEXITCODE -ne 0) { throw "Frozen Python package versions do not match." }

& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw "Python environment has broken requirements." }

& $py .\stage3b_generate.py --self-test
if ($LASTEXITCODE -ne 0) { throw "Stage 3B self-test failed." }

& $py .\stage3b_generate.py `
  --input .\inputs\stage3a\GeoDose_Stage3A_OUTPUTS.zip `
  --output .\outputs_stage3b `
  --overwrite `
  --replication 1
if ($LASTEXITCODE -ne 0) { throw "Stage 3B generation failed." }

& $py .\verify_stage3b.py `
  --input .\inputs\stage3a\GeoDose_Stage3A_OUTPUTS.zip `
  --output .\outputs_stage3b
if ($LASTEXITCODE -ne 0) { throw "Stage 3B verification failed." }

$archive = Join-Path $PSScriptRoot "GeoDose_Stage3B_OUTPUTS.zip"
if (Test-Path $archive) { Remove-Item $archive -Force }
Compress-Archive -Path .\outputs_stage3b\* -DestinationPath $archive -Force
Write-Host "Created: $archive"
