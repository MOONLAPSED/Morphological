@echo off
REM This batch file launches the PowerShell script to prepare and start the Windows Sandbox.

REM Get the directory where this batch file is located.
SET "SCRIPT_DIR=%~dp0"

echo Launching Sandbox Environment via PowerShell...
echo Please ensure Git is installed and in your PATH.
echo.

REM Execute the PowerShell script.
REM -NoProfile: Speeds up PowerShell startup slightly.
REM -ExecutionPolicy Bypass: Temporarily bypasses execution policy for this script run.
REM -File: Specifies the script file to run.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%launch_sandbox.ps1"

REM Check if PowerShell script executed successfully (basic check)
IF ERRORLEVEL 1 (
    echo.
    echo PowerShell script encountered an error. Please check the output above.
    pause
    exit /b 1
)

echo.
echo PowerShell script has initiated the sandbox launch.
REM Add a small pause so the user can see messages if PowerShell closes too quickly.
REM timeout /t 3 /nobreak >nul
pause