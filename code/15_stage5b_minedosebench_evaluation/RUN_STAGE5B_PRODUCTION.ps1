$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Log = Join-Path $Root 'STAGE5B_PRODUCTION_POWERSHELL_LOG.txt'
Start-Transcript -Path $Log -Append | Out-Null
try {
  Write-Host 'GeoDose-CP Stage5B MineDoseBench Production Evaluation v1.0.2 numerical closure (scientific contract v1.0.0)' -ForegroundColor Cyan
  Write-Host 'Frozen input: accepted Stage5A MineDoseBench v1.2.0.'
  Write-Host 'No Stage5A bytes, targets, scenarios, thresholds, or spatial splits are modified.'
  Write-Host 'This runner is resumable at one atomic case-replication checkpoint.'

  if ($Root.Length -gt 105) {
    throw "Windows path is too long ($($Root.Length) chars). Extract this package under a short folder such as D:\G5B and rerun. Do NOT modify the package contents."
  }
  $env:PYTHONHASHSEED='0'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'

  & py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:3]==(3,10,0) else 4)"
  if ($LASTEXITCODE -ne 0) { throw 'Exact Python 3.10.0 is required. The py -3.10 interpreter is absent or is not exactly 3.10.0.' }

  Write-Host '[1/7] Verifying immutable Stage5B source package before environment setup...'
  & py -3.10 (Join-Path $Root 'verify_package.py')
  if ($LASTEXITCODE -ne 0) { throw 'Package verification failed.' }

  $Venv = Join-Path $Root '.venv'
  $VPy = Join-Path $Venv 'Scripts\python.exe'
  $rebuild=$false
  if (Test-Path $Venv) {
    if (-not (Test-Path $VPy)) { $rebuild=$true }
    else {
      & $VPy -c "import sys; raise SystemExit(0 if sys.version_info[:3]==(3,10,0) else 4)"
      if ($LASTEXITCODE -ne 0) { $rebuild=$true }
    }
  }
  if ($rebuild) { Write-Host '[ENV] Removing incomplete/wrong local venv...'; Remove-Item -Recurse -Force $Venv }
  if (-not (Test-Path $Venv)) { Write-Host '[ENV] Creating package-local Python 3.10.0 environment...'; & py -3.10 -m venv $Venv; if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' } }

  Write-Host '[2/7] Installing/verifying frozen dependencies...'
  & $VPy -m pip install --disable-pip-version-check --no-input pip==26.2.1
  if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
  & $VPy -m pip install --disable-pip-version-check --no-input -r (Join-Path $Root 'requirements_py310.txt')
  if ($LASTEXITCODE -ne 0) { throw 'Frozen dependency installation failed' }

  Write-Host '[3/7] Verifying exact frozen runtime...'
  & $VPy (Join-Path $Root 'verify_runtime.py')
  if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime verification failed' }

  Write-Host '[4/7] Running Stage5B unit/authority tests...'
  & $VPy (Join-Path $Root 'tests\test_stage5b.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage5B unit/authority tests failed' }

  Write-Host '[5/7] Running/resuming all 540 frozen Stage5B case-replication evaluations...' -ForegroundColor Yellow
  Write-Host 'If Windows or the terminal stops, rerun this SAME BAT. Completed checkpoints are reused.'
  & $VPy (Join-Path $Root 'stage5b_run.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage5B production failed. Completed atomic checkpoints are retained in _STAGE5B_WORK\checkpoints.' }

  Write-Host '[6/7] Running independent post-production verifier...'
  & $VPy (Join-Path $Root 'verify_stage5b.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage5B post-production verification failed. Do not delete checkpoints or outputs.' }

  Write-Host '[7/7] Packaging verified Stage5B result bundle...'
  & $VPy (Join-Path $Root 'package_stage5b_release.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage5B result packaging failed.' }

  Write-Host ''
  Write-Host 'STAGE5B MINE DOSE BENCH PRODUCTION EVALUATION PASSED.' -ForegroundColor Green
  Write-Host 'STOP HERE. Send STAGE5B_PRODUCTION_OUTPUTS\GeoDose_Stage5B_MineDoseBench_PRODUCTION_RESULTS.zip and this PowerShell log for independent audit.' -ForegroundColor Yellow
  Write-Host 'Do NOT start the NSW real-demonstration stage until the independent Stage5B audit passes.' -ForegroundColor Yellow
}
catch {
  Write-Host ''
  Write-Host ('STAGE5B FAILED: ' + $_.Exception.Message) -ForegroundColor Red
  Write-Host 'Do not delete _STAGE5B_WORK or STAGE5B_PRODUCTION_OUTPUTS. Rerun the same BAT after resolving the reported issue.' -ForegroundColor Yellow
  throw
}
finally { Stop-Transcript | Out-Null }
