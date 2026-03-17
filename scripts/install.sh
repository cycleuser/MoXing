#!/bin/bash
#
# moxing-server Installation Script for Linux/macOS
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ggml-org/llama.cpp/master/moxing/scripts/install.sh | bash
#
# Or:
#   ./install.sh [--backend cuda|vulkan|rocm|metal|cpu] [--no-venv]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
BACKEND="auto"
NO_VENV=false
PYTHON_CMD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend|-b)
            BACKEND="$2"
            shift 2
            ;;
        --no-venv)
            NO_VENV=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend, -b   GPU backend: cuda, vulkan, rocm, metal, cpu (default: auto-detect)"
            echo "  --no-venv       Don't create a virtual environment"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  moxing-server Installation Script${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     OS_TYPE="linux";;
    Darwin*)    OS_TYPE="macos";;
    *)          echo -e "${RED}Unsupported OS: $OS${NC}"; exit 1;;
esac

echo -e "${BLUE}Detected OS: ${GREEN}$OS_TYPE${NC}"

# Detect architecture
ARCH="$(uname -m)"
echo -e "${BLUE}Architecture:  ${GREEN}$ARCH${NC}"

# Find Python
detect_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Python not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    
    # Check version
    PY_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${BLUE}Python:        ${GREEN}$PYTHON_CMD (version $PY_VERSION)${NC}"
    
    # Check if version >= 3.8
    $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" || {
        echo -e "${RED}Python 3.8+ required. Found version $PY_VERSION${NC}"
        exit 1
    }
}

detect_python

# Detect GPU
detect_gpu() {
    echo ""
    echo -e "${BLUE}Detecting GPU...${NC}"
    
    GPU_VENDOR=""
    GPU_NAME=""
    
    # Check NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
        if [ -n "$GPU_NAME" ]; then
            GPU_VENDOR="nvidia"
            echo -e "  ${GREEN}✓${NC} NVIDIA GPU: $GPU_NAME"
        fi
    fi
    
    # Check AMD (ROCm on Linux)
    if [ -z "$GPU_VENDOR" ] && [ "$OS_TYPE" = "linux" ] && [ -d "/opt/rocm" ]; then
        if command -v rocm-smi &> /dev/null; then
            GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP "Card series:\s*\K.*" | head -1 || echo "")
            if [ -n "$GPU_NAME" ]; then
                GPU_VENDOR="amd"
                echo -e "  ${GREEN}✓${NC} AMD GPU (ROCm): $GPU_NAME"
            fi
        fi
    fi
    
    # Check Apple Silicon
    if [ -z "$GPU_VENDOR" ] && [ "$OS_TYPE" = "macos" ] && [ "$ARCH" = "arm64" ]; then
        GPU_NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        GPU_VENDOR="apple"
        echo -e "  ${GREEN}✓${NC} Apple Silicon: $GPU_NAME"
    fi
    
    # Check Vulkan
    if [ -z "$GPU_VENDOR" ] && command -v vulkaninfo &> /dev/null; then
        VULKAN_GPU=$(vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -1 | cut -d= -f2 | tr -d ' ,' || echo "")
        if [ -n "$VULKAN_GPU" ]; then
            GPU_VENDOR="vulkan"
            GPU_NAME="$VULKAN_GPU"
            echo -e "  ${GREEN}✓${NC} Vulkan GPU: $GPU_NAME"
        fi
    fi
    
    if [ -z "$GPU_VENDOR" ]; then
        echo -e "  ${YELLOW}⚠${NC} No GPU detected, using CPU backend"
        GPU_VENDOR="cpu"
    fi
    
    # Determine backend
    if [ "$BACKEND" = "auto" ]; then
        case "$GPU_VENDOR" in
            nvidia)     BACKEND="cuda";;
            amd)        BACKEND="rocm";;
            apple)      BACKEND="metal";;
            vulkan)     BACKEND="vulkan";;
            *)          BACKEND="cpu";;
        esac
    fi
    
    echo ""
    echo -e "${BLUE}Selected backend: ${GREEN}$BACKEND${NC}"
}

detect_gpu

# Install system dependencies
install_deps() {
    echo ""
    echo -e "${BLUE}Installing system dependencies...${NC}"
    
    if [ "$OS_TYPE" = "linux" ]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3-pip python3-venv vulkan-tools 2>/dev/null || true
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3-pip python3-vulkan-loader 2>/dev/null || true
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm python-pip vulkan-tools 2>/dev/null || true
        fi
    elif [ "$OS_TYPE" = "macos" ]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
    fi
}

# Create virtual environment
setup_venv() {
    if [ "$NO_VENV" = false ]; then
        VENV_DIR="$HOME/moxing-env"
        
        if [ ! -d "$VENV_DIR" ]; then
            echo ""
            echo -e "${BLUE}Creating virtual environment at $VENV_DIR...${NC}"
            $PYTHON_CMD -m venv "$VENV_DIR"
        fi
        
        echo ""
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source "$VENV_DIR/bin/activate"
        
        PYTHON_CMD="python"
    fi
}

# Install moxing-server
install_moxing() {
    echo ""
    echo -e "${BLUE}Installing moxing-server...${NC}"
    
    $PYTHON_CMD -m pip install --upgrade pip --quiet
    $PYTHON_CMD -m pip install moxing-server --quiet
    
    echo -e "${GREEN}✓ moxing-server installed${NC}"
}

# Download binaries
download_binaries() {
    echo ""
    echo -e "${BLUE}Downloading pre-built binaries ($BACKEND backend)...${NC}"
    
    $PYTHON_CMD -m moxing.cli download-binaries --backend "$BACKEND" || {
        echo -e "${YELLOW}⚠ Binary download failed. You may need to download manually.${NC}"
        echo -e "  Run: moxing download-binaries --backend $BACKEND"
    }
}

# Verify installation
verify() {
    echo ""
    echo -e "${BLUE}Verifying installation...${NC}"
    
    echo ""
    $PYTHON_CMD -m moxing.cli devices || true
    
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  Installation Complete!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    
    if [ "$NO_VENV" = false ]; then
        echo -e "To use moxing, activate the virtual environment:"
        echo -e "  ${CYAN}source $VENV_DIR/bin/activate${NC}"
        echo ""
    fi
    
    echo -e "Quick start commands:"
    echo -e "  ${CYAN}moxing devices${NC}        # List GPUs"
    echo -e "  ${CYAN}moxing models${NC}         # List popular models"
    echo -e "  ${CYAN}moxing download llama-3.2-3b${NC}  # Download a model"
    echo -e "  ${CYAN}moxing run llama-3.2-3b${NC}       # Run inference"
    echo -e "  ${CYAN}moxing serve llama-3.2-3b${NC}     # Start API server"
    echo ""
}

# Main
main() {
    install_deps
    setup_venv
    install_moxing
    download_binaries
    verify
}

main