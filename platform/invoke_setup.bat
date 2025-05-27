@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM invoke_setup.bat  
REM   - Called by provisioner.bat with one argument: VERSION  
REM   - Installs Scoop if needed, then runs scoop_setup.ps1  
REM ─────────────────────────────────────────────────────────────────────────────
setlocal

if "%~1"=="" (
  echo [WARN] No version passed in; continuing without version context.
) else (
  set "PROJECT_VERSION=%~1"
  echo [INFO] Running setup for version %PROJECT_VERSION%
)

REM Make sure we’re in this script’s directory
pushd "%~dp0"

REM Install Scoop if somehow missing (double-check)
if not exist "%USERPROFILE%\scoop" (
  echo [INFO] Scoop missing—installing via PowerShell
  powershell -NoProfile -ExecutionPolicy RemoteSigned -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')"
  if errorlevel 1 (
    echo [ERROR] Failed to install Scoop.  
    popd & exit /b 1
  )
)

REM Run full PowerShell installer
echo [INFO] Launching scoop_setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scoop_setup.ps1" -Version "%PROJECT_VERSION%"
if errorlevel 1 (
  echo [ERROR] scoop_setup.ps1 failed.
  popd & exit /b 1
)

:ForwardWSL2Port
  set PORT=%1
  for /f "tokens=*" %%I in ('wsl hostname -I') do set "WSL_IP=%%I"
  echo [%TIME%] Forwarding port %PORT% to WSL2 at %WSL_IP% >> "%USERPROFILE%\Desktop\sandbox_log.txt"
  netsh interface portproxy delete v4tov4 listenport=%PORT% listenaddress=0.0.0.0 >> "%USERPROFILE%\Desktop\sandbox_log.txt" 2>&1
  netsh interface portproxy add v4tov4 listenport=%PORT% listenaddress=0.0.0.0 connectport=%PORT% connectaddress=%WSL_IP% >> "%USERPROFILE%\Desktop\sandbox_log.txt" 2>&1
  goto :eof

echo [INFO] invoke_setup.bat completed.
popd
exit /b 0