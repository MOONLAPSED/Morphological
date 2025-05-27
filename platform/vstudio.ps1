Write-Host "Starting Visual Studio minimal install script..."

# Define the desired Visual Studio version and components
$VSVersion = "2022" # Or "2019", etc. - choose the one you prefer
$VSEdition = "Community" # Or "Professional", "Enterprise"
$InstallDir = "C:\Program Files (x86)\Microsoft Visual Studio\$VSVersion\$VSEdition" # Adjust if needed

# The workload ID for Desktop development with C++ (includes native debugging)
$WorkloadId = "Microsoft.VisualStudio.Workload.NativeDesktop"

# Optional individual components - we can refine this if needed
# For basic debugging, the workload might be enough.
# Let's start with just the workload for simplicity.
# $ComponentIds = @(
#     "Microsoft.VisualStudio.Component.Debugger.JustInTime",
#     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" # Common tools
# )

# Construct the command-line arguments for the installer
$InstallerArgs = @(
    "--installPath", $InstallDir,
    "--add", $WorkloadId,
    "--passive", # No user interaction
    "--norestart", # Don't restart automatically
    "--wait"      # Wait for the installation to complete
    # If you had specific components:
    # "--add", "$($ComponentIds -join ',')"
)

# Construct the full installer command
#$InstallerCommand = Join-Path -Path $env:TEMP -ChildPath "vs_installer.exe"
$DownloadDir = Join-Path -Path $env:USERPROFILE -ChildPath "Desktop\vs_installer"
if (-not (Test-Path $DownloadDir)) {
    New-Item -ItemType Directory -Path $DownloadDir | Out-Null
}
$InstallerCommand = Join-Path -Path $DownloadDir -ChildPath "vs_installer.exe"
$DownloadUrl = "https://aka.ms/vs/$VSVersion/release/vs_Community.exe" # Adjust for edition

# If installer exists, remove it to avoid corrupted file issues
if (Test-Path $InstallerCommand) {
    Write-Host "Removing existing Visual Studio Installer to avoid corruption..."
    Remove-Item -Path $InstallerCommand -Force
}

Write-Host "Downloading Visual Studio Installer..."
try {
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $InstallerCommand
    if (-not $?) {
        Write-Error "Failed to download Visual Studio Installer."
        exit 1
    }
} catch {
    Write-Host "Error downloading Visual Studio Installer: $($_.Exception.Message)"
    exit 1
}

# Verify installer file exists and is not empty
if (-not (Test-Path $InstallerCommand)) {
    Write-Error "Installer file not found after download."
    exit 1
}
$installerSize = (Get-Item $InstallerCommand).Length
if ($installerSize -eq 0) {
    Write-Error "Installer file is empty, download failed."
    exit 1
}

Write-Host "Starting Visual Studio $VSEdition $VSVersion installation (minimal)..."
Write-Host "Installer path: $InstallerCommand"
Write-Host "Installer arguments: $InstallerArgs"

# Unblock the installer file to avoid execution issues
Unblock-File -Path $InstallerCommand

# Suggest manual verification
Write-Host "Please verify the installer file manually at: $InstallerCommand"
Write-Host "You can try running it manually to check for corruption or sandbox restrictions."

# Convert argument array to a single string for call operator
$argString = $InstallerArgs -join ' '

# Use call operator to run the installer with arguments
Write-Host "Running installer with call operator..."
& $InstallerCommand $argString

if ($LASTEXITCODE -eq 0) {
    Write-Host "Visual Studio minimal installation completed successfully."
} else {
    Write-Error "Visual Studio installation failed with exit code: $($LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Visual Studio minimal install script finished."
