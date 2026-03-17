# moxing-server Installation Script for Windows
#
# Usage:
#   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/moxing/scripts/install.ps1" | Invoke-Expression
#
# Or:
#   .\install.ps1 -Backend cuda
#

param(
    [ValidateSet("auto", "cuda", "vulkan", "cpu")]
    [string]$Backend = "auto",
    
    [switch]$NoVenv,
    [switch]$Help
)

# Colors
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Info { Write-ColorOutput Cyan $args }
function Write-Warn { Write-ColorOutput Yellow $args }
function Write-Err { Write-ColorOutput Red $args }

if ($Help) {
    Write-Host "moxing-server Installation Script for Windows"
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Backend    GPU backend: cuda, vulkan, cpu (default: auto-detect)"
    Write-Host "  -NoVenv     Don't create a virtual environment"
    Write-Host "  -Help       Show this help message"
    exit 0
}

Write-Info "============================================================"
Write-Info "  moxing-server Installation Script"
Write-Info "============================================================"
Write-Host ""

# Check Python
function Find-Python {
    $pythonCmd = $null
    
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCmd = "python"
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $pythonCmd = "python3"
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCmd = "py -3"
    }
    
    if (-not $pythonCmd) {
        Write-Err "Python not found!"
        Write-Host ""
        Write-Host "Please install Python 3.8+ from:"
        Write-Host "  - https://www.python.org/downloads/"
        Write-Host "  - Microsoft Store: 'Python 3.11'"
        Write-Host "  - winget install Python.Python.3.12"
        exit 1
    }
    
    # Check version
    $version = Invoke-Expression "$pythonCmd --version 2>&1"
    Write-Info "Python: $version"
    
    # Verify version >= 3.8
    $versionParts = $version -replace "Python ", "" -split "\."
    $major = [int]$versionParts[0]
    $minor = [int]$versionParts[1]
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Err "Python 3.8+ required. Found $version"
        exit 1
    }
    
    return $pythonCmd
}

# Detect GPU
function Detect-GPU {
    Write-Host ""
    Write-Info "Detecting GPU..."
    
    $gpuVendor = ""
    $gpuName = ""
    
    # Check NVIDIA via nvidia-smi
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try {
            $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($gpuInfo) {
                $gpuName = $gpuInfo.Split("`n")[0].Trim()
                $gpuVendor = "nvidia"
                Write-Success "  NVIDIA GPU: $gpuName"
            }
        } catch {}
    }
    
    # Check via WMI
    if (-not $gpuVendor) {
        try {
            $gpus = Get-WmiObject Win32_VideoController
            foreach ($gpu in $gpus) {
                $name = $gpu.Name.ToLower()
                if ($name -match "nvidia|geforce|rtx|gtx") {
                    $gpuVendor = "nvidia"
                    $gpuName = $gpu.Name
                    Write-Success "  NVIDIA GPU: $gpuName"
                    break
                } elseif ($name -match "amd|radeon") {
                    $gpuVendor = "amd"
                    $gpuName = $gpu.Name
                    Write-Success "  AMD GPU: $gpuName"
                    break
                } elseif ($name -match "intel.*arc") {
                    $gpuVendor = "intel"
                    $gpuName = $gpu.Name
                    Write-Success "  Intel GPU: $gpuName"
                    break
                }
            }
        } catch {}
    }
    
    # Check Vulkan support
    if (-not $gpuVendor) {
        $vulkanDll = "C:\Windows\System32\vulkan-1.dll"
        if (Test-Path $vulkanDll) {
            $gpuVendor = "vulkan"
            Write-Success "  Vulkan support detected"
        }
    }
    
    if (-not $gpuVendor) {
        Write-Warn "  No GPU detected, using CPU backend"
        $gpuVendor = "cpu"
    }
    
    # Determine backend
    if ($Backend -eq "auto") {
        switch ($gpuVendor) {
            "nvidia" { $Backend = "cuda" }
            "amd" { $Backend = "vulkan" }
            "intel" { $Backend = "vulkan" }
            "vulkan" { $Backend = "vulkan" }
            default { $Backend = "cpu" }
        }
    }
    
    Write-Host ""
    Write-Info "Selected backend: $Backend"
}

# Check Visual C++ Redistributable
function Check-VCRedist {
    Write-Host ""
    Write-Info "Checking Visual C++ Redistributable..."
    
    $vcRedists = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\*\VC\Runtimes\X64" -ErrorAction SilentlyContinue
    
    if ($vcRedists) {
        Write-Success "  Visual C++ Redistributable installed"
    } else {
        Write-Warn "  Visual C++ Redistributable not found"
        Write-Host ""
        Write-Host "  Installing via winget..."
        
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements
        } else {
            Write-Host "  Please install from: https://aka.ms/vs/17/release/vc_redist.x64.exe"
        }
    }
}

# Create virtual environment
function Setup-Venv {
    param($pythonCmd)
    
    if (-not $NoVenv) {
        $venvDir = "$env:USERPROFILE\moxing-env"
        
        if (-not (Test-Path $venvDir)) {
            Write-Host ""
            Write-Info "Creating virtual environment at $venvDir..."
            Invoke-Expression "$pythonCmd -m venv `"$venvDir`""
        }
        
        Write-Host ""
        Write-Info "Activating virtual environment..."
        
        $activateScript = "$venvDir\Scripts\Activate.ps1"
        if (Test-Path $activateScript) {
            . $activateScript
        }
        
        return "python"
    }
    
    return $pythonCmd
}

# Install moxing-server
function Install-Pyllama {
    param($pythonCmd)
    
    Write-Host ""
    Write-Info "Installing moxing-server..."
    
    Invoke-Expression "$pythonCmd -m pip install --upgrade pip --quiet"
    Invoke-Expression "$pythonCmd -m pip install moxing-server --quiet"
    
    Write-Success "moxing-server installed"
}

# Download binaries
function Download-Binaries {
    param($pythonCmd)
    
    Write-Host ""
    Write-Info "Downloading pre-built binaries ($Backend backend)..."
    
    try {
        Invoke-Expression "$pythonCmd -m moxing.cli download-binaries --backend $Backend"
        Write-Success "Binaries downloaded"
    } catch {
        Write-Warn "Binary download failed. You may need to download manually."
        Write-Host "  Run: moxing download-binaries --backend $Backend"
    }
}

# Verify installation
function Verify-Installation {
    param($pythonCmd)
    
    Write-Host ""
    Write-Info "Verifying installation..."
    Write-Host ""
    
    Invoke-Expression "$pythonCmd -m moxing.cli devices"
    
    Write-Host ""
    Write-Success "============================================================"
    Write-Success "  Installation Complete!"
    Write-Success "============================================================"
    Write-Host ""
    
    if (-not $NoVenv) {
        Write-Host "To use moxing, activate the virtual environment:"
        Write-Info "  $env:USERPROFILE\moxing-env\Scripts\Activate.ps1"
        Write-Host ""
    }
    
    Write-Host "Quick start commands:"
    Write-Info "  moxing devices"        ; Write-Host "        # List GPUs"
    Write-Info "  moxing models"         ; Write-Host "         # List popular models"
    Write-Info "  moxing download llama-3.2-3b"  ; Write-Host "  # Download a model"
    Write-Info "  moxing run llama-3.2-3b"       ; Write-Host "       # Run inference"
    Write-Info "  moxing serve llama-3.2-3b"     ; Write-Host "     # Start API server"
    Write-Host ""
}

# Main
Write-Info "Detecting system..."
$pythonCmd = Find-Python
Detect-GPU
Check-VCRedist
$pythonCmd = Setup-Venv $pythonCmd
Install-Pyllama $pythonCmd
Download-Binaries $pythonCmd
Verify-Installation $pythonCmd