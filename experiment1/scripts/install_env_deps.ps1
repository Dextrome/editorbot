<#
Script: install_env_deps.ps1
Purpose: Create a Python venv, install project dependencies from requirements.txt,
         and optionally install system FFmpeg using winget or choco (requires admin privileges).

Usage examples:
powershell -ExecutionPolicy Bypass -File .\scripts\install_env_deps.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\install_env_deps.ps1 -InstallSystemFFmpeg
#>
param(
    [switch]$InstallSystemFFmpeg,
    [switch]$PreferWinget
)

function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Write-Warn($msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Write-Err($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

Write-Info "Creating virtual environment 'venv' (if not exists)..."
if (-Not (Test-Path -Path './venv/Scripts/activate.ps1')) {
    python -m venv venv
} else {
    Write-Info "Virtual environment already exists. Skipping venv creation."
}

Write-Info "Upgrading pip, setuptools, wheel in venv..."
& .\venv\Scripts\Activate.ps1; python -m pip install --upgrade pip setuptools wheel

Write-Info "Installing Python dependencies from requirements.txt..."
& .\venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt

if ($InstallSystemFFmpeg) {
    Write-Info "Attempting to install system ffmpeg. This requires admin privileges."
    if ($PreferWinget -or (-not (Get-Command choco -ErrorAction SilentlyContinue))) {
        # Use winget if requested or choco is not available
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Info "Installing ffmpeg using winget..."
            winget install --id=FFmpeg.FFmpeg -e --source winget
            if ($LASTEXITCODE -ne 0) { Write-Warn "winget reported non-zero exit code ($LASTEXITCODE). Please verify manually." }
        } else {
            Write-Err "winget is not available. Try running the script with -PreferWinget, or install ffmpeg manually or via chocolatey (choco)."
        }
    } else {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-Info "Installing ffmpeg using chocolatey (choco)..."
            choco install ffmpeg -y
            if ($LASTEXITCODE -ne 0) { Write-Warn "choco reported non-zero exit code ($LASTEXITCODE). Please verify manually." }
        } else {
            Write-Err "chocolatey is not installed. Try running the script with -PreferWinget, or install ffmpeg manually."
        }
    }
}

Write-Info "Environment setup complete. Activate your environment with: .\venv\Scripts\Activate.ps1 (PowerShell)"
Write-Info "Verify FFmpeg: ffmpeg -version (add to PATH if necessary)"

Write-Info "Note: If you didn't install system ffmpeg, the project uses 'imageio-ffmpeg' as a fallback for ffmpeg binaries when possible."
