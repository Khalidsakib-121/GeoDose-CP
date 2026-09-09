$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "D:\GeoDose_Stage2B_Local_STAC_v1_0\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -c "import sys,importlib.metadata as m,numpy,pandas,geopandas,yaml; assert sys.version_info[:2]==(3,10), 'Python 3.10 is required'; expected={'numpy':'2.2.6','pandas':'2.3.3','geopandas':'1.1.4','PyYAML':'6.0.3','shapely':'2.1.2','pyogrio':'0.13.0'}; actual={k:m.version(k) for k in expected}; assert actual==expected, f'Frozen environment mismatch: {actual}'; print('Python',sys.version); print('Frozen environment check passed')"
if ($LASTEXITCODE -ne 0) { throw "Stage 3A environment check failed" }

& $py tests/test_generator.py
if ($LASTEXITCODE -ne 0) { throw "Stage 3A generator unit tests failed" }

& $py stage3a_prepare.py --input inputs --output outputs_stage3a --overwrite
if ($LASTEXITCODE -ne 0) { throw "Stage 3A preparation failed" }

& $py verify_stage3a.py --input inputs --output outputs_stage3a
if ($LASTEXITCODE -ne 0) { throw "Stage 3A verification failed" }
