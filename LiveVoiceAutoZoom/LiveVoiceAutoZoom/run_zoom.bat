@echo off
setlocal

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPTS_PATH=%SCRIPT_DIR%scripts"

REM Check if scripts directory exists
if not exist "%SCRIPTS_PATH%" (
    echo [ERROR] Scripts directory not found: %SCRIPTS_PATH%
    echo Please ensure the 'scripts' folder exists in the same directory as this batch file.
    pause
    exit /b 1
)

REM Change to scripts directory
cd /d "%SCRIPTS_PATH%"

REM Check if Python script exists
if not exist "live_zoom_record_and_analyze.py" (
    echo [ERROR] Script not found: live_zoom_record_and_analyze.py
    echo Please ensure the script exists in: %SCRIPTS_PATH%
    pause
    exit /b 1
)

REM Run the Python script
python live_zoom_record_and_analyze.py
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo [ERROR] Script exited with error code: %EXITCODE%
)

pause
exit /b %EXITCODE%
