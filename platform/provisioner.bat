@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: provisioner.bat - Windows Sandbox Provisioner
:: Sets up Scoop, configures environment, and logs everything to sandbox_log.txt
:: ─────────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion
pushd "%~dp0"

set "LOGFILE=C:\Users\WDAGUtilityAccount\Desktop\sandbox_log.txt"
echo [%TIME%] Starting provisioner.bat in Windows Sandbox... > "%LOGFILE%" 2>&1
echo [%TIME%] Working directory: %CD% >> "%LOGFILE%"

set "SANDBOX_USER=WDAGUtilityAccount"
set "SCOOP_ROOT=C:\Users\%SANDBOX_USER%\scoop"
set "SCOOP_SHIMS=%SCOOP_ROOT%\shims"
set "PATH=%SCOOP_SHIMS%;%PATH%"

:: Check Internet connectivity
echo [%TIME%] Checking internet connectivity... >> "%LOGFILE%"
ping -n 1 github.com | findstr TTL >nul
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] No internet connection. >> "%LOGFILE%"
    timeout /t 10
    popd
    exit /b 1
)
echo [%TIME%] Internet connection OK >> "%LOGFILE%"

:: Try to detect Scoop
where scoop >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [%TIME%] Scoop already installed. >> "%LOGFILE%"
    goto SetupComplete
)

:: Scoop not found – install it
echo [%TIME%] Scoop not found, installing... >> "%LOGFILE%"
powershell.exe -ExecutionPolicy Bypass -NoProfile -Command ^
    "$env:SCOOP='%SCOOP_ROOT%'; [Environment]::SetEnvironmentVariable('SCOOP', $env:SCOOP, 'User'); iwr -useb get.scoop.sh | iex" >> "%LOGFILE%" 2>&1

timeout /t 5 >nul

:: Recheck Scoop
where scoop >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] Scoop installation failed. >> "%LOGFILE%"
    popd
    exit /b 1
)

echo [%TIME%] Scoop installed successfully. >> "%LOGFILE%"

:SetupComplete
:: Persist environment
echo [%TIME%] Persisting PATH to user env... >> "%LOGFILE%"
setx PATH "%PATH%" >> "%LOGFILE%" 2>&1

echo [%TIME%] Setting registry PATH (redundant safety)... >> "%LOGFILE%"
reg add "HKCU\Environment" /f /v PATH /t REG_EXPAND_SZ /d "%PATH%" >> "%LOGFILE%" 2>&1

:: Detect host IP (used for bridging in sandboxed net)
for /f "tokens=3" %%a in ('route print ^| findstr /C:" 0.0.0.0"') do set HOST_IP=%%a
echo [%TIME%] Detected host IP: %HOST_IP% >> "%LOGFILE%"

:: Optional: Run invoke_setup.bat here if you want auto provisioning
REM call "%~dp0invoke_setup.bat" >> "%LOGFILE%" 2>&1

:: Switch to Desktop for usability
cd /d "%USERPROFILE%\Desktop"
echo [%TIME%] Switched to desktop directory. >> "%LOGFILE%"

echo.
echo === Provisioning Complete ===
echo Log: %LOGFILE%
echo Type 'cmd' to launch a shell here, or press any key to exit...
pause >nul
%SystemRoot%\System32\cmd.exe /K cd /d "%USERPROFILE%\Desktop"

:: Final diagnostics
popd
echo [%TIME%] Final directory: %CD% >> "%LOGFILE%"
exit /b 0
