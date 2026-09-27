@echo off
setlocal
cd /d "%~dp0"
set "MOGWAI=%~dp0build\windows-ninja-msvc\bin\Release\Mogwai.exe"
if not exist "%MOGWAI%" (
    echo Mogwai.exe is missing. Build Mogwai first.
    pause
    exit /b 1
)
rem Interactive launcher: deliberately omit --headless and do not forward flags.
"%MOGWAI%" --script="%~dp0scripts\HSTR\IntelCloudSeaFull.py"
if errorlevel 1 pause
