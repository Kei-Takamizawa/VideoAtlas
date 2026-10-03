param([switch]$Gpu)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Requirements = if ($Gpu) { 'requirements-windows-gpu.txt' } else { 'requirements-windows.txt' }

Push-Location $RepoRoot
try {
    & py -3.12 --version
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 was not found. Install 64-bit Python 3.12 and the Python Launcher.' }

    if (-not (Test-Path $VenvPython)) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    }

    $VenvPythonInfo = & $VenvPython -c 'import struct,sys; print("%d.%d:%d" % (sys.version_info[0], sys.version_info[1], struct.calcsize("P") * 8))'
    if ($LASTEXITCODE -ne 0 -or $VenvPythonInfo.Trim() -ne '3.12:64') {
        throw "The existing .venv must use 64-bit Python 3.12; found $VenvPythonInfo. Remove .venv and rerun setup."
    }

    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }

    & $VenvPython -m pip uninstall -y onnxruntime onnxruntime-gpu
    if ($LASTEXITCODE -ne 0) { throw 'Could not remove an existing ONNX Runtime package.' }

    & $VenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Could not install dependencies from $Requirements." }

    & $VenvPython Scripts\download_face01.py
    if ($LASTEXITCODE -ne 0) { throw 'Could not download and verify the face models.' }

    Write-Host 'Setup finished. Start VideoAtlas with Scripts\run_windows.cmd.'
}
finally {
    Pop-Location
}
