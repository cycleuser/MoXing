#!/usr/bin/env bash
# MoXing - Build llama.cpp binaries from source
#
# This script builds llama.cpp binaries from source for multiple backends.
# Each backend requires specific dependencies:
#   - CPU: cmake, gcc/g++
#   - CUDA: CUDA Toolkit (nvcc)
#   - Vulkan: Vulkan SDK
#   - ROCm: ROCm SDK
#   - Metal: macOS only (Xcode)
#
# Usage:
#   ./generate_binaries.sh                    # Build all available backends
#   ./generate_binaries.sh --backend cuda     # Build specific backend
#   ./generate_binaries.sh --version b8468    # Build specific version
#   ./generate_binaries.sh --clean            # Clean build directory first
#
# Output: moxing/bin/{platform}-{backend}/
#         dist/binaries/{platform}-{backend}.tar.gz

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
BIN_DIR="$SCRIPT_DIR/moxing/bin"
DIST_DIR="$SCRIPT_DIR/dist/binaries"
BUILD_DIR="$SCRIPT_DIR/build/llama.cpp"
LLAMA_CPP_REPO="ggml-org/llama.cpp"

# Detect current platform
detect_platform() {
    local os arch
    case "$(uname -s)" in
        Linux*)   os="linux" ;;
        Darwin*)  os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *)        os="linux" ;;
    esac
    
    case "$(uname -m)" in
        x86_64|amd64)  arch="x64" ;;
        arm64|aarch64) arch="arm64" ;;
        *)             arch="x64" ;;
    esac
    
    echo "${os}-${arch}"
}

PLATFORM=$(detect_platform)
LLAMA_CPP_DIR="$BUILD_DIR"

# Default values
VERSION=""
BACKENDS="all"
CLEAN=0
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version|-v)
            VERSION="$2"
            shift 2
            ;;
        --backend|-b)
            BACKENDS="$2"
            shift 2
            ;;
        --clean|-c)
            CLEAN=1
            shift
            ;;
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        --help|-h)
            echo "MoXing - Build llama.cpp binaries from source"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version, -v VERSION    Build specific llama.cpp version (tag)"
            echo "  --backend, -b BACKEND    Build specific backend (cpu,cuda,vulkan,rocm,metal,all)"
            echo "  --clean, -c              Clean build directory first"
            echo "  --jobs, -j N             Number of parallel jobs (default: $JOBS)"
            echo "  --help, -h               Show this help"
            echo ""
            echo "Backends:"
            echo "  cpu     - CPU only (requires: cmake, gcc)"
            echo "  cuda    - NVIDIA CUDA (requires: CUDA Toolkit)"
            echo "  vulkan  - Vulkan (requires: Vulkan SDK)"
            echo "  rocm    - AMD ROCm (requires: ROCm SDK, Linux only)"
            echo "  metal   - Apple Metal (macOS only)"
            echo "  all     - Build all available backends"
            echo ""
            echo "Output:"
            echo "  moxing/bin/{platform}-{backend}/"
            echo "  dist/binaries/{platform}-{backend}.tar.gz"
            echo ""
            echo "Examples:"
            echo "  $0 --backend cuda          # Build CUDA backend only"
            echo "  $0 --version b8468 --clean # Build specific version, clean first"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== MoXing Binary Builder ===${NC}"
echo ""
echo "Platform: $PLATFORM"
echo "Jobs: $JOBS"

# Get latest version if not specified
if [[ -z "$VERSION" ]]; then
    echo -e "${BLUE}[1/5] Fetching latest llama.cpp version...${NC}"
    VERSION=$($PYTHON -c "
import json
from urllib.request import urlopen, Request
req = Request('https://api.github.com/repos/$LLAMA_CPP_REPO/releases/latest',
              headers={'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'moxing'})
with urlopen(req, timeout=30) as r:
    data = json.loads(r.read().decode())
    print(data['tag_name'])
")
    echo "  Latest version: $VERSION"
else
    echo -e "${BLUE}[1/5] Using specified version: $VERSION${NC}"
fi

# Clone or update llama.cpp
echo -e "${BLUE}[2/5] Preparing llama.cpp source...${NC}"
if [[ "$CLEAN" -eq 1 ]] && [[ -d "$LLAMA_CPP_DIR" ]]; then
    echo "  Cleaning build directory..."
    rm -rf "$LLAMA_CPP_DIR"
fi

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    echo "  Cloning llama.cpp..."
    git clone --depth 1 --branch "$VERSION" "https://github.com/$LLAMA_CPP_REPO.git" "$LLAMA_CPP_DIR"
else
    echo "  Updating llama.cpp..."
    cd "$LLAMA_CPP_DIR"
    git fetch --tags
    git checkout "$VERSION" || git checkout "tags/$VERSION"
    git submodule update --init --recursive
    cd "$SCRIPT_DIR"
fi

# Create output directories
mkdir -p "$BIN_DIR"
mkdir -p "$DIST_DIR"

# Build function
build_backend() {
    local backend=$1
    local platform_name="${PLATFORM}-${backend}"
    local output_dir="$BIN_DIR/$platform_name"
    local build_dir="$LLAMA_CPP_DIR/build-$backend"
    
    echo ""
    echo -e "${GREEN}=== Building: $platform_name ===${NC}"
    
    # Clean previous build
    rm -rf "$build_dir" "$output_dir"
    mkdir -p "$output_dir"
    
    cd "$LLAMA_CPP_DIR"
    
    # Build options
    local cmake_opts=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="$output_dir"
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=ON
        -DLLAMA_BUILD_SERVER=ON
    )
    
    case "$backend" in
        cpu)
            cmake_opts+=(
                -DLLAMA_CUDA=OFF
                -DLLAMA_VULKAN=OFF
                -DLLAMA_ROCM=OFF
                -DLLAMA_METAL=OFF
            )
            ;;
        cuda)
            if ! command -v nvcc &> /dev/null; then
                echo -e "${YELLOW}  Skipping: CUDA not available (nvcc not found)${NC}"
                return 0
            fi
            cmake_opts+=(
                -DLLAMA_CUDA=ON
                -DLLAMA_VULKAN=OFF
                -DLLAMA_ROCM=OFF
                -DLLAMA_METAL=OFF
            )
            ;;
        vulkan)
            if [[ -z "$VULKAN_SDK" ]] && ! pkg-config --exists vulkan 2>/dev/null; then
                echo -e "${YELLOW}  Skipping: Vulkan not available (VULKAN_SDK not set)${NC}"
                return 0
            fi
            cmake_opts+=(
                -DLLAMA_CUDA=OFF
                -DLLAMA_VULKAN=ON
                -DLLAMA_ROCM=OFF
                -DLLAMA_METAL=OFF
            )
            ;;
        rocm)
            if [[ ! -d "/opt/rocm" ]] && ! command -v hipcc &> /dev/null; then
                echo -e "${YELLOW}  Skipping: ROCm not available (/opt/rocm not found)${NC}"
                return 0
            fi
            cmake_opts+=(
                -DLLAMA_CUDA=OFF
                -DLLAMA_VULKAN=OFF
                -DLLAMA_HIPBLAS=ON
                -DLLAMA_METAL=OFF
                -DCMAKE_C_COMPILER=hipcc
                -DCMAKE_CXX_COMPILER=hipcc
            )
            ;;
        metal)
            if [[ "$(uname -s)" != "Darwin" ]]; then
                echo -e "${YELLOW}  Skipping: Metal only available on macOS${NC}"
                return 0
            fi
            cmake_opts+=(
                -DLLAMA_CUDA=OFF
                -DLLAMA_VULKAN=OFF
                -DLLAMA_ROCM=OFF
                -DLLAMA_METAL=ON
            )
            ;;
        *)
            echo -e "${RED}  Unknown backend: $backend${NC}"
            return 1
            ;;
    esac
    
    echo "  CMake options: ${cmake_opts[*]}"
    
    # Configure
    cmake -B "$build_dir" -S . "${cmake_opts[@]}"
    
    # Build
    cmake --build "$build_dir" --config Release -j "$JOBS"
    
    # Copy binaries
    echo "  Copying binaries..."
    local bin_dir="$build_dir/bin"
    if [[ ! -d "$bin_dir" ]]; then
        bin_dir="$build_dir"
    fi
    
    # Find and copy llama-* binaries
    for f in "$bin_dir"/llama-*; do
        if [[ -f "$f" ]]; then
            cp "$f" "$output_dir/"
            chmod +x "$output_dir/$(basename "$f")"
            echo "    $(basename "$f")"
        fi
    done
    
    # Copy shared libraries
    case "$(uname -s)" in
        Linux*)
            for f in "$bin_dir"/*.so*; do
                if [[ -f "$f" ]]; then
                    cp -P "$f" "$output_dir/"
                    echo "    $(basename "$f")"
                fi
            done
            ;;
        Darwin*)
            for f in "$bin_dir"/*.dylib; do
                if [[ -f "$f" ]]; then
                    cp -P "$f" "$output_dir/"
                    echo "    $(basename "$f")"
                fi
            done
            ;;
        MINGW*|MSYS*|CYGWIN*)
            for f in "$bin_dir"/*.dll; do
                if [[ -f "$f" ]]; then
                    cp "$f" "$output_dir/"
                    echo "    $(basename "$f")"
                fi
            done
            ;;
    esac
    
    # Create VERSION file
    echo -e "$VERSION\n$backend" > "$output_dir/VERSION"
    
    # Count files
    local count=$(ls -1 "$output_dir" 2>/dev/null | wc -l)
    echo -e "${GREEN}  Done: $count files${NC}"
    
    cd "$SCRIPT_DIR"
}

# Build backends
echo -e "${BLUE}[3/5] Building binaries...${NC}"

if [[ "$BACKENDS" == "all" ]]; then
    # Build all backends (platform-specific)
    case "$PLATFORM" in
        linux-x64)
            build_backend "cpu"
            build_backend "cuda"
            build_backend "vulkan"
            build_backend "rocm"
            ;;
        linux-arm64)
            build_backend "cpu"
            ;;
        darwin-arm64)
            build_backend "cpu"
            build_backend "metal"
            ;;
        darwin-x64)
            build_backend "cpu"
            ;;
        windows-x64)
            build_backend "cpu"
            build_backend "cuda"
            build_backend "vulkan"
            ;;
        *)
            build_backend "cpu"
            ;;
    esac
else
    # Build specified backends
    IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"
    for backend in "${BACKEND_LIST[@]}"; do
        build_backend "$(echo "$backend" | tr -d ' ')"
    done
fi

# Package binaries
echo ""
echo -e "${BLUE}[4/5] Packaging binaries...${NC}"
$PYTHON scripts/upload_binaries.py --package

# Summary
echo ""
echo -e "${BLUE}[5/5] Summary...${NC}"
echo ""
echo "Binaries built to: moxing/bin/"
ls -la "$BIN_DIR"/*/VERSION 2>/dev/null || echo "  No binaries built"
echo ""
echo "Packages created in: dist/binaries/"
ls -la "$DIST_DIR" 2>/dev/null || echo "  No packages created"
echo ""
echo "Total size:"
du -sh "$BIN_DIR" 2>/dev/null || echo "  0"

echo ""
echo -e "${GREEN}=== Done! ===${NC}"
echo ""
echo "llama.cpp version: $VERSION"
echo "Platform: $PLATFORM"
echo ""
echo "To upload binaries to GitHub:"
echo "  ./upload_binaries.sh"