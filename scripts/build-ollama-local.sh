#!/bin/bash
# 编译本地 Ollama llama.cpp 目录
# 使用 /Users/fred/Documents/GitHub/Others/ollama/llama/ 目录

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_LLAMA_DIR="/Users/fred/Documents/GitHub/Others/ollama/llama"
PATCHES_DIR="$OLLAMA_LLAMA_DIR/patches"
LLAMA_CPP_DIR="$OLLAMA_LLAMA_DIR/llama.cpp"
OUTPUT_DIR="$SCRIPT_DIR/../moxing/bin"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# 检查目录
if [ ! -d "$OLLAMA_LLAMA_DIR" ]; then
    log_error "Ollama llama 目录不存在：$OLLAMA_LLAMA_DIR"
    exit 1
fi

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    log_error "llama.cpp 目录不存在：$LLAMA_CPP_DIR"
    exit 1
fi

if [ ! -d "$PATCHES_DIR" ]; then
    log_error "patches 目录不存在：$PATCHES_DIR"
    exit 1
fi

log_info "使用目录:"
log_info "  Ollama llama: $OLLAMA_LLAMA_DIR"
log_info "  llama.cpp: $LLAMA_CPP_DIR"
log_info "  patches: $PATCHES_DIR"
log_info "  output: $OUTPUT_DIR"

# 应用补丁（如果需要）
apply_patches_if_needed() {
    if [ -f "$LLAMA_CPP_DIR/.patches-applied" ]; then
        log_info "补丁已经应用，跳过..."
        return
    fi
    
    log_info "应用 Ollama 补丁到 llama.cpp..."
    cd "$LLAMA_CPP_DIR"
    
    for patch in "$PATCHES_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            patch_name=$(basename "$patch")
            echo "  Applying: $patch_name"
            patch -p1 --forward < "$patch" 2>/dev/null || true
        fi
    done
    
    touch "$LLAMA_CPP_DIR/.patches-applied"
    log_success "补丁应用完成"
}

# 编译 macOS Metal
build_macos_metal() {
    log_info "编译 macOS Metal..."
    cd "$LLAMA_CPP_DIR"
    
    rm -rf build-metal
    mkdir build-metal && cd build-metal
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR/darwin-arm64-metal"
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    mkdir -p "$OUTPUT_DIR/darwin-arm64-metal"
    cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-metal/" 2>/dev/null || true
    cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-metal/" 2>/dev/null || true
    cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-metal/" 2>/dev/null || true
    cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-metal/" 2>/dev/null || true
    
    log_success "macOS Metal 编译完成"
}

# 编译 macOS CPU
build_macos_cpu() {
    log_info "编译 macOS CPU..."
    cd "$LLAMA_CPP_DIR"
    
    rm -rf build-cpu
    mkdir build-cpu && cd build-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR/darwin-arm64-cpu"
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    mkdir -p "$OUTPUT_DIR/darwin-arm64-cpu"
    cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-cpu/" 2>/dev/null || true
    cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-cpu/" 2>/dev/null || true
    cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-cpu/" 2>/dev/null || true
    cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-cpu/" 2>/dev/null || true
    
    log_success "macOS CPU 编译完成"
}

# 创建版本信息
create_version_info() {
    cat > "$OUTPUT_DIR/OLLAMA_VERSION.txt" << EOF
MoXing Ollama-Patched llama.cpp Binaries
==========================================

Source: /Users/fred/Documents/GitHub/Others/ollama/llama/
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Patches: $(ls -1 "$PATCHES_DIR"/*.patch 2>/dev/null | wc -l | tr -d ' ') patches applied

Available Binaries:
$(ls -1 "$OUTPUT_DIR" | grep -v VERSION | sed 's/^/  - /')

Built by: MoXing Build Script
EOF
    
    log_success "版本信息创建完成"
}

# 主流程
main() {
    echo "=============================================="
    echo "编译 Ollama llama.cpp (本地目录)"
    echo "=============================================="
    
    mkdir -p "$OUTPUT_DIR"
    
    # 应用补丁
    apply_patches_if_needed
    
    # 编译
    build_macos_metal
    build_macos_cpu
    
    # 版本信息
    create_version_info
    
    echo ""
    echo "=============================================="
    log_success "编译完成！"
    echo "=============================================="
    echo "输出目录：$OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
}

main "$@"
