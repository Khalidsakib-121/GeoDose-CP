$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$required = @{
  "numpy" = "2.2.6"
  "pandas" = "2.3.3"
  "scipy" = "1.15.3"
  "PyYAML" = "6.0.3"
}

function Test-Environment([string]$pythonExe) {
  if (-not (Test-Path $pythonExe)) { return $false }
  $checkCode = @'
import sys
import importlib.metadata as metadata
required = {
    'numpy': '2.2.6',
    'pandas': '2.3.3',
    'scipy': '1.15.3',
    'PyYAML': '6.0.3',
}
if sys.version_info[:2] != (3, 10):
    raise RuntimeError(f'Python 3.10 is required; found {sys.version.split()[0]}')
observed = {}
for name in required:
    try:
        observed[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        observed[name] = '<missing>'
mismatch = {name: {'observed': observed[name], 'required': required[name]} for name in required if observed[name] != required[name]}
if mismatch:
    raise RuntimeError(f'Exact Stage3D environment mismatch: {mismatch}')
print('Exact Stage3D environment verified')
'@
  $tempScript = [System.IO.Path]::GetTempFileName()
  try {
    [System.IO.File]::WriteAllText($tempScript, $checkCode, [System.Text.Encoding]::ASCII)
    & $pythonExe $tempScript | Out-Host
    return ($LASTEXITCODE -eq 0)
  }
  finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $tempScript
  }
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
    Write-Host "Existing package-local .venv is absent or incompatible; recreating it."
    Remove-Item -Recurse -Force $venv
  }
  Write-Host "Creating isolated Python 3.10 environment..."
  py -3.10 -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw "Python 3.10 virtual-environment creation failed" }
  $py = Join-Path $venv "Scripts\python.exe"
  & $py -m pip install --disable-pip-version-check -r requirements_py310.txt
  if ($LASTEXITCODE -ne 0) { throw "Pinned package installation failed" }
  if (-not (Test-Environment $py)) { throw "Exact environment verification failed after installation" }
}

& $py tests\test_stage3d_d1.py
if ($LASTEXITCODE -ne 0) { throw "Stage3D D0/D1 unit tests failed" }

& $py stage3d_d1_run.py --overwrite
if ($LASTEXITCODE -ne 0) { throw "Stage3D D0/D1 generation failed" }

& $py verify_stage3d_d1.py
if ($LASTEXITCODE -ne 0) { throw "Stage3D D0/D1 independent verification failed" }

& $py package_stage3d_outputs.py
if ($LASTEXITCODE -ne 0) { throw "Stage3D D0/D1 output packaging failed" }
