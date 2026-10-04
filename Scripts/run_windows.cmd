@echo off
set "APP_ROOT=%~dp0.."
if not exist "%APP_ROOT%\.venv\Scripts\python.exe" (
    echo Run Scripts\setup_windows.cmd first.
    exit /b 1
)
pushd "%APP_ROOT%"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m videoatlas %*
set "APP_EXIT=%ERRORLEVEL%"
popd
exit /b %APP_EXIT%
