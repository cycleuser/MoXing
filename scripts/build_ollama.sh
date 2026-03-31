#!/bin/bash
#
# Build llama.cpp with Ollama patches
# This creates llama-server binaries with support for GLM-4.7-flash and other models
#

set -e

OLLAMA_PATH="/home/fred/Documents/GitHub/Others/ollama"
LLAMA_CPP_SRC="/home/fred/Documents/GitHub/cycleuser/llama.cpp"
BUILD_DIR="$HOME/.cache/moxing/build-ollama"
OUTPUT_DIR="$(dirname "$0")/../moxing/bin"
NPROC=$(nproc)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build llama.cpp binaries with Ollama patches"
    echo ""
    echo "Options:"
    echo "  --backend BACKEND    Build specific backend (cuda, rocm, vulkan, cpu, all)"
    echo "  --clean              Clean build directory before building"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --backend rocm"
    echo "  $0 --backend cuda"
    echo "  $0 --backend all"
    exit 0
}

BACKEND="all"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ ! -d "$OLLAMA_PATH" ]]; then
    log_error "Ollama repository not found at: $OLLAMA_PATH"
    exit 1
fi

if [[ ! -d "$LLAMA_CPP_SRC" ]]; then
    log_error "llama.cpp not found at: $LLAMA_CPP_SRC"
    exit 1
fi

mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

PATCHES_DIR="$OLLAMA_PATH/llama/patches"

detect_platform() {
    case "$(uname -s)" in
        Linux) echo "linux" ;;
        Darwin) echo "darwin" ;;
        *) echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) echo "unknown" ;;
    esac
}

get_amd_targets() {
    local targets=""
    if command -v rocminfo &> /dev/null; then
        targets=$(rocminfo 2>/dev/null | grep -oP 'Name:\s+\Kgfx[0-9a-z]+' | sort -u | tr '\n' ';')
    fi
    if [[ -z "$targets" ]]; then
        targets="gfx1100"
    fi
    echo "${targets%;}"
}

get_cuda_archs() {
    local archs=""
    if command -v nvidia-smi &> /dev/null; then
        archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.\n' | tr '\n' ';')
    fi
    if [[ -z "$archs" ]]; then
        archs="89;90"
    fi
    echo "${archs%;}"
}

apply_patches() {
    log_info "Applying Ollama patches to $LLAMA_CPP_SRC..."
    
    cd "$LLAMA_CPP_SRC"
    
    local patch_count=0
    local applied_count=0
    local skipped_count=0
    
    for patch in "$PATCHES_DIR"/*.patch; do
        [[ -f "$patch" ]] || continue
        ((patch_count++))
        
        patch_name=$(basename "$patch")
        
        if git apply --check "$patch" 2>/dev/null; then
            git apply "$patch" 2>/dev/null
            log_ok "Applied: $patch_name"
            ((applied_count++))
        else
            ((skipped_count++))
        fi
    done
    
    log_info "Applied: $applied_count, Skipped: $skipped_count, Total: $patch_count"
}

build_backend() {
    local backend=$1
    local platform=$(detect_platform)
    local arch=$(detect_arch)
    local output_name="${platform}-${arch}-${backend}"
    local output_path="$OUTPUT_DIR/$output_name"
    local build_path="$BUILD_DIR/build-$backend"
    
    echo ""
    echo "============================================"
    log_info "Building backend: $backend"
    echo "============================================"
    
    if [[ "$CLEAN" == "true" ]] && [[ -d "$build_path" ]]; then
        log_info "Cleaning build directory..."
        rm -rf "$build_path"
    fi
    
    mkdir -p "$build_path"
    mkdir -p "$output_path"
    
    cd "$LLAMA_CPP_SRC"
    
    local cmake_args=(
        -B "$build_path"
        -S "$LLAMA_CPP_SRC"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="$output_path"
        -DGGML_NATIVE=OFF
        -DGGML_BACKEND_DL=OFF
        -DGGML_SHARED=ON
        -DLLAMA_BUILD_SERVER=ON
    )
    
    case "$backend" in
        cuda)
            local archs=$(get_cuda_archs)
            log_info "CUDA architectures: $archs"
            cmake_args+=(
                -DGGML_CUDA=ON
                -DCMAKE_CUDA_ARCHITECTURES="${archs//;/,}"
                -DGGML_CUDA_GRAPHS=ON
                -DGGML_CUDA_FA=ON
            )
            ;;
        rocm)
            local targets=$(get_amd_targets)
            log_info "AMD GPU targets: $targets"
            cmake_args+=(
                -DGGML_HIP=ON
                -DAMDGPU_TARGETS="${targets//;/,}"
                -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
                -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
            )
            ;;
        vulkan)
            cmake_args+=(
                -DGGML_VULKAN=ON
            )
            ;;
        cpu)
            cmake_args+=(
                -DGGML_CUDA=OFF
                -DGGML_HIP=OFF
                -DGGML_VULKAN=OFF
            )
            ;;
    esac
    
    log_info "Running CMake..."
    cmake "${cmake_args[@]}"
    
    log_info "Building with $NPROC threads..."
    cmake --build "$build_path" --config Release -j "$NPROC"
    
    log_info "Copying binaries..."
    
    local binaries=("llama-server" "llama-cli" "llama-bench" "llama-quantize")
    
    for bin in "${binaries[@]}"; do
        if [[ -f "$build_path/bin/$bin" ]]; then
            cp "$build_path/bin/$bin" "$output_path/"
            chmod +x "$output_path/$bin"
            log_ok "Installed: $bin"
        elif [[ -f "$build_path/$bin" ]]; then
            cp "$build_path/$bin" "$output_path/"
            chmod +x "$output_path/$bin"
            log_ok "Installed: $bin"
        else
            log_warn "Not found: $bin"
        fi
    done
    
    for lib in "$build_path"/bin/*.so* "$build_path"/*.so*; do
        [[ -f "$lib" ]] || continue
        libname=$(basename "$lib")
        [[ -f "$output_path/$libname" ]] || cp "$lib" "$output_path/"
    done
    
    local timestamp=$(date +%Y%m%d)
    echo "ollama-patched-$timestamp" > "$output_path/VERSION"
    echo "$backend" >> "$output_path/VERSION"
    
    log_ok "Build complete: $output_path"
}

main() {
    log_info "Ollama path: $OLLAMA_PATH"
    log_info "llama.cpp path: $LLAMA_CPP_SRC"
    log_info "Build dir: $BUILD_DIR"
    log_info "Output dir: $OUTPUT_DIR"
    log_info "Backend: $BACKEND"
    
    apply_patches
    
    if [[ "$BACKEND" == "all" ]]; then
        for b in cuda rocm vulkan cpu; do
            build_backend "$b"
        done
    else
        build_backend "$BACKEND"
    fi
    
    echo ""
    echo "============================================"
    log_ok "All builds complete!"
    echo "============================================"
    ls -la "$OUTPUT_DIR"
}

main