@echo off
setlocal
cd /d "%~dp0"

set "MOGWAI=build\windows-ninja-msvc\bin\Release\Mogwai.exe"
if not exist "%MOGWAI%" (
    echo HSTRCloud is not built. Build the Release HSTRCloud target first.
    pause
    exit /b 1
)

"%MOGWAI%" --script="scripts/HSTR/RunWDASCloud.py" %*
if errorlevel 1 pause
