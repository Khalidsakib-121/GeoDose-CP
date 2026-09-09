$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Log = Join-Path $Root 'STAGE6A_NSW_REAL_POWERSHELL_LOG.txt'
Start-Transcript -Path $Log -Append | Out-Null
try {
  Write-Host 'GeoDose-CP Stage6A Real NSW Demonstration v1.0.0' -ForegroundColor Cyan
  Write-Host 'Frozen design: Mt Arthur Coal + Hunter Valley Operations + Bulga Complex.'
  Write-Host 'Stage1 treatment gate is binding: no annual treatment is fabricated; M2/M4/M6 real treatment routes are not operational.'
  Write-Host 'This runner is resumable at 12 atomic mine x scale x predictor checkpoints.'
  if ($Root.Length -gt 105) { throw "Windows path is too long ($($Root.Length) chars). Extract under a short folder such as D:\G6A." }
  $env:PYTHONHASHSEED='0';$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:NUMEXPR_NUM_THREADS='1'
  & py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:3]==(3,10,0) else 4)"
  if ($LASTEXITCODE -ne 0) { throw 'Exact Python 3.10.0 is required. Install/use the exact frozen Python runtime before continuing.' }

  Write-Host '[1/7] Verifying immutable Stage6A source package...'
  & py -3.10 (Join-Path $Root 'verify_package.py')
  if ($LASTEXITCODE -ne 0) { throw 'Package verification failed.' }

  $Venv=Join-Path $Root '.venv';$VPy=Join-Path $Venv 'Scripts\python.exe';$rebuild=$false
  if (Test-Path $Venv) {
    if (-not (Test-Path $VPy)) {$rebuild=$true} else { & $VPy -c "import sys; raise SystemExit(0 if sys.version_info[:3]==(3,10,0) else 4)"; if ($LASTEXITCODE -ne 0) {$rebuild=$true} }
  }
  if ($rebuild) { Write-Host '[ENV] Removing incomplete/wrong local venv...';Remove-Item -Recurse -Force $Venv }
  if (-not (Test-Path $Venv)) { Write-Host '[ENV] Creating package-local Python 3.10.0 environment...'; & py -3.10 -m venv $Venv; if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' } }

  Write-Host '[2/7] Installing/verifying frozen dependencies...'
  & $VPy -m pip install --disable-pip-version-check --no-input pip==26.2.1
  if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
  & $VPy -m pip install --disable-pip-version-check --no-input -r (Join-Path $Root 'requirements_py310.txt')
  if ($LASTEXITCODE -ne 0) { throw 'Frozen dependency installation failed' }

  Write-Host '[3/7] Verifying exact frozen runtime...'
  & $VPy (Join-Path $Root 'verify_runtime.py')
  if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime verification failed' }

  Write-Host '[4/7] Running Stage6A authority/unit/regression gates...'
  & $VPy (Join-Path $Root 'tests\test_stage6a.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage6A authority/unit tests failed' }

  Write-Host '[5/7] Running/resuming the real NSW production...' -ForegroundColor Yellow
  Write-Host 'If interrupted, rerun this SAME BAT. Do not delete _STAGE6A_WORK.'
  & $VPy (Join-Path $Root 'stage6a_run.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage6A production failed. Existing checkpoints are retained.' }

  Write-Host '[6/7] Running independent terminal verifier...'
  & $VPy (Join-Path $Root 'verify_stage6a.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage6A verification failed. Keep checkpoints and outputs.' }

  Write-Host '[7/7] Packaging verified Stage6A result bundle...'
  & $VPy (Join-Path $Root 'package_stage6a_release.py')
  if ($LASTEXITCODE -ne 0) { throw 'Stage6A result packaging failed.' }

  Write-Host ''
  Write-Host 'STAGE6A REAL NSW DEMONSTRATION PASSED.' -ForegroundColor Green
  Write-Host 'STOP HERE. Send STAGE6A_NSW_REAL_OUTPUTS\GeoDose_Stage6A_NSW_REAL_DEMONSTRATION_RESULTS.zip and STAGE6A_NSW_REAL_POWERSHELL_LOG.txt for independent audit.' -ForegroundColor Yellow
}
catch {
  Write-Host ''
  Write-Host ('STAGE6A FAILED: '+$_.Exception.Message) -ForegroundColor Red
  Write-Host 'Do not delete _STAGE6A_WORK or STAGE6A_NSW_REAL_OUTPUTS. Keep the traceback/log for diagnosis.' -ForegroundColor Yellow
  throw
}
finally { Stop-Transcript | Out-Null }
