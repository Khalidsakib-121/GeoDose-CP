$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$required = @{
  "numpy" = "2.2.6"
  "pandas" = "2.3.3"
  "scipy" = "1.15.3"
  "scikit-learn" = "1.7.2"
  "xgboost" = "3.0.5"
  "joblib" = "1.5.2"
  "PyYAML" = "6.0.3"
}

function Test-Environment([string]$pythonExe) {
  if (-not (Test-Path $pythonExe)) { return $false }
  $json = $required | ConvertTo-Json -Compress
  & $pythonExe -c "import sys,json,importlib.metadata as m; req=json.loads(r'''$json'''); assert sys.version_info[:2]==(3,10),sys.version; bad={k:(m.version(k),v) for k,v in req.items() if m.version(k)!=v}; assert not bad,bad; print('Exact Stage3C environment verified')" | Out-Host
  return ($LASTEXITCODE -eq 0)
}

$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$valid = $false
try { $valid = Test-Environment $py } catch { $valid = $false }

if (-not $valid) {
  if (Test-Path $venv) {
    $resolvedRoot = (Resolve-Path $root).Path
    $resolvedVenv = (Resolve-Path $venv).Path
    if (-not $resolvedVenv.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar)) {
      throw "Unsafe virtual-environment path: $resolvedVenv"
    }
    Write-Host "Existing .venv is absent or incompatible; recreating only the package-local environment."
    Remove-Item -Recurse -Force $venv
  }
  Write-Host "Creating isolated Python 3.10 environment..."
  py -3.10 -m venv .venv
  $py = Join-Path $venv "Scripts\python.exe"
  & $py -m pip install --disable-pip-version-check -r requirements_py310.txt
  if ($LASTEXITCODE -ne 0) { throw "Pinned package installation failed" }
  if (-not (Test-Environment $py)) { throw "Exact environment verification failed after installation" }
}

& $py tests\test_stage3c.py
if ($LASTEXITCODE -ne 0) { throw "Stage3C unit tests failed" }

& $py stage3c_run.py --overwrite
if ($LASTEXITCODE -ne 0) { throw "Stage3C reference generation failed" }

& $py verify_stage3c.py
if ($LASTEXITCODE -ne 0) { throw "Stage3C independent verification failed" }

& $py package_stage3c_outputs.py
if ($LASTEXITCODE -ne 0) { throw "Deterministic Stage3C output packaging failed" }
