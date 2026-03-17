# Installation Guide

Complete installation guide for pyllm across all platforms, operating systems, and hardware configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Windows](#windows)
  - [Linux](#linux)
  - [macOS](#macos)
- [GPU Backend Setup](#gpu-backend-setup)
  - [Vulkan (Recommended)](#vulkan-recommended)
  - [CUDA (NVIDIA)](#cuda-nvidia)
  - [ROCm (AMD)](#rocm-amd)
  - [Metal (Apple Silicon)](#metal-apple-silicon)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Install pyllm
pip install pyllm

# Download pre-built binaries
pyllm download-binaries

# Verify installation
pyllm devices
```

---

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.11+ |
| RAM | 4GB | 16GB+ |
| Disk Space | 2GB | 50GB+ (for models) |
| CPU | x86_64 / ARM64 | Multi-core |

### GPU Requirements (Optional but Recommended)

| Backend | Hardware | VRAM |
|---------|----------|------|
| Vulkan | AMD, Intel, NVIDIA | 4GB+ |
| CUDA | NVIDIA GTX 10xx+ | 6GB+ |
| ROCm | AMD RDNA2+ | 8GB+ |
| Metal | Apple M1/M2/M3 | 8GB+ |

---

## Platform-Specific Instructions

### Windows

#### Windows 10/11 (x64)

```powershell
# 1. Install Python 3.11+ from python.org or Microsoft Store
winget install Python.Python.3.12

# 2. Install pyllm
pip install pyllm

# 3. Download binaries (auto-selects Vulkan backend)
pyllm download-binaries

# 4. Verify GPU detection
pyllm devices
```

#### Windows Hardware Detection

| GPU | Recommended Backend | Notes |
|-----|---------------------|-------|
| NVIDIA RTX/GTX | CUDA | Best performance |
| AMD Radeon RX | Vulkan | Good performance, no CUDA |
| Intel Arc | Vulkan | Good performance |
| Integrated GPU | CPU | Slow but works |

#### Windows Vulkan Setup

```powershell
# Check Vulkan support
pyllm devices

# If no Vulkan devices found, install Vulkan Runtime
# Download from: https://vulkan.lunarg.com/sdk/home
```

#### Windows CUDA Setup (NVIDIA)

```powershell
# 1. Install CUDA Toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# 2. Verify CUDA installation
nvidia-smi

# 3. Download CUDA binaries
pyllm download-binaries --backend cuda
```

#### Windows Troubleshooting

```powershell
# If binary fails to start:
# 1. Install Visual C++ Redistributable
winget install Microsoft.VCRedist.2015+.x64

# 2. Check DLL dependencies
pyllm download-binaries --force

# 3. Run with verbose output
pyllm run model.gguf -v
```

---

### Linux

#### Ubuntu/Debian

```bash
# 1. Install Python and dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 2. Create virtual environment (recommended)
python3 -m venv ~/pyllm-env
source ~/pyllm-env/bin/activate

# 3. Install pyllm
pip install pyllm

# 4. Install Vulkan support (recommended for AMD/Intel)
sudo apt install -y vulkan-tools libvulkan1

# 5. Download binaries
pyllm download-binaries

# 6. Verify GPU detection
pyllm devices
```

#### Fedora/RHEL

```bash
# 1. Install Python
sudo dnf install -y python3 python3-pip

# 2. Install Vulkan
sudo dnf install -y vulkan-loader vulkan-tools

# 3. Install and setup pyllm
pip install pyllm
pyllm download-binaries
```

#### Arch Linux

```bash
# 1. Install dependencies
sudo pacman -S python python-pip vulkan-tools vulkan-icd-loader

# 2. Install pyllm
pip install pyllm
pyllm download-binaries
```

#### Linux GPU Support Matrix

| GPU Vendor | Backend | Package |
|------------|---------|---------|
| NVIDIA | CUDA | `nvidia-driver`, `cuda` |
| NVIDIA | Vulkan | `nvidia-driver` |
| AMD | Vulkan | `mesa-vulkan-drivers` |
| AMD | ROCm | See ROCm setup |
| Intel | Vulkan | `mesa-vulkan-drivers` |

---

### macOS

#### macOS 12+ (Monterey and later)

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python@3.11

# 3. Create virtual environment
python3.11 -m venv ~/pyllm-env
source ~/pyllm-env/bin/activate

# 4. Install pyllm
pip install pyllm

# 5. Download Metal binaries
pyllm download-binaries --backend metal

# 6. Verify Metal detection
pyllm devices
```

#### Apple Silicon (M1/M2/M3/M4)

```bash
# Metal is automatically detected and used
pyllm devices

# Expected output:
# 0 | Apple M2 | metal | 8.0GB
```

#### Intel Mac

```bash
# Uses CPU backend on Intel Macs
pyllm download-binaries --backend cpu
```

---

## GPU Backend Setup

### Vulkan (Recommended)

**Best for**: AMD GPUs, Intel GPUs, cross-platform compatibility

#### Windows Vulkan Setup

1. Install GPU drivers (AMD Adrenalin, Intel Arc, or NVIDIA GeForce)
2. Vulkan Runtime is usually included with drivers
3. Verify: `pyllm devices`

#### Linux Vulkan Setup

```bash
# Ubuntu/Debian
sudo apt install -y vulkan-tools libvulkan1 mesa-vulkan-drivers

# Verify Vulkan support
vulkaninfo | head -20

# If using AMD GPU
sudo apt install -y amdvlk  # Optional alternative Vulkan driver

# If using Intel GPU
sudo apt install -y intel-media-va-driver-non-free
```

#### Verify Vulkan Detection

```bash
pyllm devices

# Expected output for AMD:
# 0 | AMD Radeon RX 7900 XTX | vulkan | 24.0GB
```

---

### CUDA (NVIDIA)

**Best for**: NVIDIA GPUs (RTX 20xx, RTX 30xx, RTX 40xx, GTX 16xx)

#### Requirements

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 525.60.13+ |
| CUDA Toolkit | 12.0+ |
| cuDNN | 8.9+ (optional) |

#### Windows CUDA Setup

```powershell
# 1. Install NVIDIA Driver
# https://www.nvidia.com/Download/index.aspx

# 2. Install CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads
# Select: Windows > x86_64 > Version > exe (local)

# 3. Verify installation
nvidia-smi

# 4. Install pyllm with CUDA binaries
pyllm download-binaries --backend cuda
```

#### Linux CUDA Setup

```bash
# Ubuntu 22.04
# 1. Install NVIDIA driver
sudo apt install -y nvidia-driver-535

# 2. Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda

# 3. Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. Verify
nvidia-smi

# 5. Download CUDA binaries
pyllm download-binaries --backend cuda
```

#### CUDA GPU Memory Requirements

| Model Size | Minimum VRAM | Recommended |
|------------|--------------|-------------|
| 3B (Q4) | 4GB | 6GB |
| 7B (Q4) | 6GB | 8GB |
| 13B (Q4) | 10GB | 12GB |
| 70B (Q4) | 40GB | 48GB (2x24GB) |

---

### ROCm (AMD)

**Best for**: AMD GPUs on Linux (RX 6000, RX 7000, Instinct)

#### Supported GPUs

| GPU Series | ROCm Support |
|------------|--------------|
| RX 7900 XTX/XT | ✅ ROCm 5.7+ |
| RX 7800/7700 | ✅ ROCm 5.7+ |
| RX 7600 | ✅ ROCm 6.0+ |
| RX 6900/6800/6700 | ✅ ROCm 5.5+ |
| RX 6600/6500 | ⚠️ Limited |
| RX 5700/5600/5500 | ⚠️ Use Vulkan |

#### ROCm Installation (Ubuntu)

```bash
# 1. Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

# 2. Add user to render group
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# 3. Reboot
sudo reboot

# 4. Verify
rocminfo

# 5. Download ROCm binaries
pyllm download-binaries --backend rocm
```

#### ROCm Environment Variables

```bash
# Add to ~/.bashrc
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RDNA2
# or
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RDNA3
```

---

### Metal (Apple Silicon)

**Best for**: Apple M1/M2/M3/M4 chips

#### Requirements

- macOS 12.0+ (Monterey)
- Apple Silicon Mac (M1, M2, M3, M4 series)

#### Setup (Automatic)

```bash
# Metal is automatically detected on Apple Silicon
pyllm devices

# No additional setup needed
pyllm download-binaries  # Auto-selects Metal
```

#### Apple Silicon Memory

| Chip | Unified Memory | Recommended Model Size |
|------|----------------|----------------------|
| M1/M2 8GB | 8GB | Up to 7B Q4 |
| M1/M2 16GB | 16GB | Up to 13B Q4 |
| M1/M2 32GB | 32GB | Up to 30B Q4 |
| M3 Max 36GB | 36GB | Up to 34B Q4 |
| M2 Ultra 192GB | 192GB | Up to 120B Q4 |

---

## Troubleshooting

### Common Issues

#### 1. "Binary not found" Error

```bash
# Solution: Download binaries
pyllm download-binaries --force
```

#### 2. No GPU Detected

```bash
# Check GPU drivers
# Windows: nvidia-smi (NVIDIA) or Device Manager
# Linux: lspci | grep -i vga

# Check Vulkan support
# Windows/Linux: vulkaninfo

# Force CPU backend
pyllm download-binaries --backend cpu
```

#### 3. Out of Memory (OOM)

```bash
# Use smaller context
pyllm run model.gguf -c 2048

# Use smaller quantization
# Q4_K_M instead of Q5_K_M or Q8_0
```

#### 4. Slow Performance

```bash
# Check GPU is being used
pyllm devices

# Verify GPU offloading
pyllm config model.gguf

# Should show GPU layers > 0
```

### Platform-Specific Issues

#### Windows: DLL Missing

```
Error: libomp.dll not found
```

**Solution**: Install Visual C++ Redistributable
```powershell
winget install Microsoft.VCRedist.2015+.x64
```

#### Linux: Permission Denied

```
Error: Permission denied
```

**Solution**:
```bash
chmod +x ~/.cache/pyllm/binaries/linux/*
```

#### macOS: "App is damaged"

```
Error: "llama-server" is damaged and can't be opened
```

**Solution**:
```bash
xattr -d com.apple.quarantine ~/.cache/pyllm/binaries/darwin/*
```

### Diagnostic Commands

```bash
# Show all detected devices
pyllm devices

# Show binary location
pyllm --help

# Test with small model
pyllm run tinyllama.q4_k_m.gguf -p "Hello"

# Show system info
pyllm config model.gguf
```

---

## Offline Installation

For air-gapped systems:

```bash
# On internet-connected machine:
pip download pyllm -d ./packages
pyllm download-binaries
cp -r ~/.cache/pyllm ./pyllm-cache

# Transfer to offline machine
pip install --no-index --find-links=./packages pyllm
cp -r ./pyllm-cache ~/.cache/pyllm
```

---

## Next Steps

After installation:

1. **Download a model**: `pyllm download llama-3.2-3b`
2. **Run inference**: `pyllm run llama-3.2-3b -p "Hello"`
3. **Start server**: `pyllm serve llama-3.2-3b`
4. **Use with OpenAI SDK**: See [API Documentation](API.md)