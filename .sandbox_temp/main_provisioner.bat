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

echo [%TIME%] Waiting for Sandbox to stabilize (15 seconds)... >> "%LOGFILE%"
timeout /t 10 >nul
echo [%TIME%] Initial delay complete. >> "%LOGFILE%"

set "SANDBOX_USER=WDAGUtilityAccount"
set "SCOOP_ROOT=C:\Users\%SANDBOX_USER%\scoop"
set "SCOOP_SHIMS=%SCOOP_ROOT%\shims"

:: Check Internet connectivity
echo [%TIME%] Checking internet connectivity... >> "%LOGFILE%"
ping -n 1 github.com -w 1000 -n 1 | findstr TTL >nul
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] No internet connection. Aborting. >> "%LOGFILE%"
    timeout /t 10
    popd
    exit /b 1
)
echo [%TIME%] Internet connection OK. >> "%LOGFILE%"

echo [%TIME%] Waiting for network initialization... >> "%LOGFILE%"
timeout /t 10 >nul

:: Try to detect Scoop
where scoop >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [%TIME%] Scoop already installed. >> "%LOGFILE%"
    goto ScoopInstalled
)

:: Scoop not found – install it
echo [%TIME%] Scoop not found, installing... >> "%LOGFILE%"

:: Create a temporary PowerShell script for Scoop installation
set "TEMP_PS_SCRIPT=C:\Users\%SANDBOX_USER%\Desktop\install_scoop_temp.ps1"
(
    echo $env:SCOOP='%SCOOP_ROOT%'
    echo [Environment]::SetEnvironmentVariable('SCOOP', $env:SCOOP, 'User')
    echo [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    echo Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')
) > "%TEMP_PS_SCRIPT%"

:: Execute the temporary PowerShell script
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%TEMP_PS_SCRIPT%" >> "%LOGFILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] Temporary Scoop installation script failed. >> "%LOGFILE%"
    del "%TEMP_PS_SCRIPT%" 2>nul
    popd
    exit /b 1
)
del "%TEMP_PS_SCRIPT%" 2>nul :: Clean up temp script

timeout /t 5 >nul :: Give Scoop a moment to settle

:ScoopInstalled
:: IMPORTANT: Ensure Scoop shims are in the current session's PATH.
:: This line ensures 'scoop' and 'uv' (if installed via scoop) are found.
set "PATH=%SCOOP_SHIMS%;%PATH%"
echo [%TIME%] Added %SCOOP_SHIMS% to PATH for current session. Current PATH: %PATH% >> "%LOGFILE%"

:: Recheck Scoop after PATH update
where scoop >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] Scoop installation failed or not found in PATH after update. >> "%LOGFILE%"
    popd
    exit /b 1
)
echo [%TIME%] Scoop confirmed installed and in PATH. >> "%LOGFILE%"

:: Now, directly call scoop_setup.ps1
echo [%TIME%] Launching scoop_setup.ps1... >> "%LOGFILE%"
:: Pass the current directory of provisioner.bat to scoop_setup.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scoop_setup.ps1" -ScriptDirectory "%~dp0" >> "%LOGFILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] scoop_setup.ps1 failed with errorlevel %ERRORLEVEL%. >> "%LOGFILE%"
    popd
    exit /b 1
)
echo [%TIME%] scoop_setup.ps1 completed successfully. >> "%LOGFILE%"

echo [%TIME%] Provisioning complete. Exiting Sandbox. >> "%LOGFILE%"
popd
exit /b 0