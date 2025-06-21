# launch_sandbox.ps1

Write-Host "Initializing Sandbox Launcher"

# --- Determine the Host Folder Path using Git ---
$HostFolderPath = $null
$RawGitPath = $null
try {
    Write-Host "[DEBUG] Current PSScriptRoot: $PSScriptRoot"
    Write-Host "[DEBUG] Attempting to find repository root using 'git rev-parse --show-toplevel'..."
    
    Push-Location $PSScriptRoot # Ensure git runs in script's dir context
    $gitOutput = git rev-parse --show-toplevel 2>&1
    Pop-Location

    if ($LASTEXITCODE -ne 0) {
        throw "git command failed. Output: $gitOutput"
    }
    $RawGitPath = $gitOutput.Trim()
    
    $HostFolderPath = $RawGitPath.Replace('/', '\') # Convert to backslashes

    if (-not (Test-Path $HostFolderPath -PathType Container)) {
        $errorMessage = "Path determined by git and converted for WSB is not a valid directory."
        $errorMessage += " RawGitPath: '$RawGitPath'."
        $errorMessage += " Converted HostFolderPath: '$HostFolderPath'."
        throw $errorMessage
    }
    Write-Host "[SUCCESS] Determined HostFolder path (raw from git): $RawGitPath"
    Write-Host "[SUCCESS] Converted HostFolder path for WSB: $HostFolderPath"
} catch {
    Write-Warning "[ERROR] Could not determine repository root. Details: $($_.Exception.Message)"
    Write-Warning "[INFO] Please ensure Git is installed, in your system PATH, and this script is run from within the repository or one of its subdirectories."
    exit 1
}

# --- Paths for template and temporary WSB file ---
$ScriptDirectory = $PSScriptRoot
$TemplateWsbPath = Join-Path $ScriptDirectory "sandbox_config.template.wsb"
$TempWsbPath = Join-Path $ScriptDirectory "_active_sandbox.wsb"

Write-Host "[DEBUG] ScriptDirectory: $ScriptDirectory"
Write-Host "[DEBUG] TemplateWsbPath: $TemplateWsbPath"
Write-Host "[DEBUG] TempWsbPath: $TempWsbPath"

if (-not (Test-Path $TemplateWsbPath -PathType Leaf)) {
    Write-Error "[FATAL] Template WSB file not found or is not a file at: $TemplateWsbPath"
    exit 1
}

# --- Read Template, Replace Placeholder, Write Temporary WSB ---
Write-Host "[INFO] Generating dynamic sandbox configuration..."
try {
    $templateContent = Get-Content -Path $TemplateWsbPath -Raw
    Write-Host "[DEBUG] Successfully read template content from: $TemplateWsbPath (Length: $($templateContent.Length))"

    $placeholder = "__HOST_FOLDER_PATH__"
    Write-Host "[DEBUG] Placeholder to replace: '$placeholder'"
    Write-Host "[DEBUG] Value to insert: '$HostFolderPath'"

    if ($templateContent -notmatch [regex]::Escape($placeholder)) {
        Write-Warning "[WARNING] Placeholder '$placeholder' NOT FOUND in template file content from '$TemplateWsbPath'. Replacement will not occur."
        Write-Host "[DEBUG] Snippet of template content (first 300 chars): $($templateContent.Substring(0, [System.Math]::Min($templateContent.Length, 300)))"
    }
    
    $modifiedContent = $templateContent.Replace($placeholder, $HostFolderPath)
    Write-Host "[DEBUG] Content modification attempted. Length of templateContent: $($templateContent.Length), Length of modifiedContent: $($modifiedContent.Length)"

    if ($modifiedContent -like "# launch_sandbox.ps1*") {
        Write-Error "[FATAL] CRITICAL ERROR: \$modifiedContent appears to be the PowerShell script itself, not XML. ABORTING before writing."
        Write-Host "[DEBUG] \$modifiedContent (first 300 chars): $($modifiedContent.Substring(0, [System.Math]::Min($modifiedContent.Length, 300)))"
        exit 1
    }

    Set-Content -Path $TempWsbPath -Value $modifiedContent -Encoding UTF8 -Force
    Write-Host "[SUCCESS] Generated temporary WSB file: $TempWsbPath"
    
    Write-Host "---------------------------------------------------------------------"
    Write-Host "--- ACTUAL CONTENT OF THE GENERATED FILE: $($TempWsbPath) ---"
    Write-Host "---------------------------------------------------------------------"
    Get-Content -Path $TempWsbPath | ForEach-Object { Write-Host $_ }
    Write-Host "---------------------------------------------------------------------"
    Write-Host "--- END OF GENERATED FILE CONTENT ---"
    Write-Host "---------------------------------------------------------------------"

} catch {
    Write-Error "[FATAL] Error during template processing or file writing: $($_.Exception.Message)"
    exit 1
}

# --- Launch the Sandbox ---
Write-Host "[INFO] Launching Windows Sandbox with generated configuration: $TempWsbPath"
try {
    Start-Process -FilePath $TempWsbPath
    Write-Host "[SUCCESS] Windows Sandbox launched."
} catch {
    Write-Error "[FATAL] Failed to launch the sandbox: $($_.Exception.Message)"
    Write-Warning "[INFO] You can try manually opening the generated file to see a more specific error from Sandbox: $TempWsbPath"
    exit 1
}

Write-Host "[INFO] Launcher script finished. The sandbox should be starting."