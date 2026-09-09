$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

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
for package_name in required:
    try:
        observed[package_name] = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        observed[package_name] = '<missing>'
mismatches = {
    package_name: {'observed': observed[package_name], 'required': required_version}
    for package_name, required_version in required.items()
    if observed[package_name] != required_version
}
if mismatches:
    raise RuntimeError(f'Exact Stage3D D2 environment mismatch: {mismatches}')
print('Exact Stage3D D2 environment verified')
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
    Write-Host "Existing .venv is absent or incompatible; recreating only the package-local environment."
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

& $py tests\test_stage3d_d2.py
if ($LASTEXITCODE -ne 0) { throw "Stage3D D2 unit tests failed" }

& $py stage3d_d2_run.py --overwrite
if ($LASTEXITCODE -ne 0) { throw "Stage3D D2 generation failed" }

& $py verify_stage3d_d2.py
if ($LASTEXITCODE -ne 0) { throw "Stage3D D2 independent verification failed" }

& $py package_stage3d_d2_outputs.py
if ($LASTEXITCODE -ne 0) { throw "Stage3D D2 output packaging failed" }
