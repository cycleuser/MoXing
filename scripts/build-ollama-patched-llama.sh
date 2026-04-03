#!/bin/bash
# MoXing Ollama 补丁版 llama.cpp 多平台编译脚本
# 基于 Ollama 官方的补丁和编译方式

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/ollama-llama-build"
OUTPUT_DIR="$SCRIPT_DIR/moxing/bin"
PATCHES_DIR="$WORK_DIR/patches"
LLAMA_CPP_REPO="$WORK_DIR/llama.cpp"

# 使用较旧的兼容版本
LLAMA_CPP_COMMIT="8e1e11d"  # 兼容 Ollama 补丁的版本

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# 清理
cleanup() {
    log_info "清理工作目录..."
    rm -rf "$WORK_DIR"
}

# 下载 Ollama 补丁
download_patches() {
    log_info "下载 Ollama 补丁..."
    mkdir -p "$PATCHES_DIR"
    
    # 从临时克隆的 Ollama 仓库复制补丁
    if [ ! -d "/tmp/ollama-llama" ]; then
        log_info "克隆 Ollama 补丁..."
        cd /tmp
        git clone --depth 1 --filter=blob:none --sparse https://github.com/ollama/ollama.git ollama-llama
        cd ollama-llama
        git sparse-checkout set llama/patches
    fi
    
    cp -r /tmp/ollama-llama/llama/patches/* "$PATCHES_DIR/"
    log_success "补丁下载完成 ($(ls -1 "$PATCHES_DIR"/*.patch 2>/dev/null | wc -l | tr -d ' ') 个)"
}

# 下载兼容的 llama.cpp 版本
download_llama_cpp() {
    log_info "下载 llama.cpp (兼容版本: $LLAMA_CPP_COMMIT)..."
    cd "$WORK_DIR"
    
    if [ -d "llama.cpp" ]; then
        rm -rf llama.cpp
    fi
    
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    git checkout "$LLAMA_CPP_COMMIT" || git checkout main
    
    log_success "llama.cpp 源码准备完成"
}

# 应用 Ollama 补丁
apply_ollama_patches() {
    log_info "应用 Ollama 补丁..."
    cd "$LLAMA_CPP_REPO"
    
    # 按顺序应用每个补丁
    for patch in "$PATCHES_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            patch_name=$(basename "$patch")
            log_info "应用：$patch_name"
            
            # 使用 git am 应用补丁（比 git apply 更宽松）
            if ! git am -3 "$patch" 2>/dev/null; then
                log_warning "补丁 $patch_name 应用失败，跳过..."
                git am --abort 2>/dev/null || true
            fi
        fi
    done
    
    log_success "补丁应用完成"
}

# 编译 macOS Metal 版本
build_macos_metal() {
    log_info "编译 macOS Metal 版本..."
    cd "$LLAMA_CPP_REPO"
    
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
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu) 2>&1 | tail -20
    
    # 检查编译结果
    if [ -f "bin/llama-server" ]; then
        mkdir -p "$OUTPUT_DIR/darwin-arm64-metal"
        cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-metal/"
        cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-metal/"
        cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-metal/"
        cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-metal/"
        log_success "macOS Metal 版本编译完成"
    else
        log_error "macOS Metal 版本编译失败"
    fi
}

# 编译 macOS CPU 版本
build_macos_cpu() {
    log_info "编译 macOS CPU 版本..."
    cd "$LLAMA_CPP_REPO"
    
    rm -rf build-cpu
    mkdir build-cpu && cd build-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    if [ -f "bin/llama-server" ]; then
        mkdir -p "$OUTPUT_DIR/darwin-arm64-cpu"
        cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-cpu/"
        cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-cpu/"
        cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-cpu/"
        cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-cpu/"
        log_success "macOS CPU 版本编译完成"
    else
        log_error "macOS CPU 版本编译失败"
    fi
}

# 编译 Linux CUDA 版本
build_linux_cuda() {
    log_info "编译 Linux CUDA 版本..."
    cd "$LLAMA_CPP_REPO"
    
    rm -rf build-cuda
    mkdir build-cuda && cd build-cuda
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    if [ -f "bin/llama-server" ]; then
        mkdir -p "$OUTPUT_DIR/linux-x64-cuda"
        cp bin/llama-server "$OUTPUT_DIR/linux-x64-cuda/"
        cp bin/llama-cli "$OUTPUT_DIR/linux-x64-cuda/"
        cp bin/llama-bench "$OUTPUT_DIR/linux-x64-cuda/"
        cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-cuda/"
        log_success "Linux CUDA 版本编译完成"
    else
        log_error "Linux CUDA 版本编译失败"
    fi
}

# 编译 Linux Vulkan 版本
build_linux_vulkan() {
    log_info "编译 Linux Vulkan 版本..."
    cd "$LLAMA_CPP_REPO"
    
    rm -rf build-vulkan
    mkdir build-vulkan && cd build-vulkan
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DGGML_CUDA=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    if [ -f "bin/llama-server" ]; then
        mkdir -p "$OUTPUT_DIR/linux-x64-vulkan"
        cp bin/llama-server "$OUTPUT_DIR/linux-x64-vulkan/"
        cp bin/llama-cli "$OUTPUT_DIR/linux-x64-vulkan/"
        cp bin/llama-bench "$OUTPUT_DIR/linux-x64-vulkan/"
        cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-vulkan/"
        log_success "Linux Vulkan 版本编译完成"
    else
        log_error "Linux Vulkan 版本编译失败"
    fi
}

# 编译 Linux ROCm 版本
build_linux_rocm() {
    log_info "编译 Linux ROCm 版本..."
    cd "$LLAMA_CPP_REPO"
    
    rm -rf build-rocm
    mkdir build-rocm && cd build-rocm
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIPBLAS=ON \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    if [ -f "bin/llama-server" ]; then
        mkdir -p "$OUTPUT_DIR/linux-x64-rocm"
        cp bin/llama-server "$OUTPUT_DIR/linux-x64-rocm/"
        cp bin/llama-cli "$OUTPUT_DIR/linux-x64-rocm/"
        cp bin/llama-bench "$OUTPUT_DIR/linux-x64-rocm/"
        cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-rocm/"
        log_success "Linux ROCm 版本编译完成"
    else
        log_error "Linux ROCm 版本编译失败"
    fi
}

# 编译 Linux CPU 版本
build_linux_cpu() {
    log_info "编译 Linux CPU 版本..."
    cd "$LLAMA_CPP_REPO"
    
    rm -rf build-cpu
    mkdir build-cpu && cd build-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    if [ -f "bin/llama-server" ]; then
        mkdir -p "$OUTPUT_DIR/linux-x64-cpu"
        cp bin/llama-server "$OUTPUT_DIR/linux-x64-cpu/"
        cp bin/llama-cli "$OUTPUT_DIR/linux-x64-cpu/"
        cp bin/llama-bench "$OUTPUT_DIR/linux-x64-cpu/"
        cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-cpu/"
        log_success "Linux CPU 版本编译完成"
    else
        log_error "Linux CPU 版本编译失败"
    fi
}

# 创建版本信息
create_version_info() {
    log_info "创建版本信息..."
    
    cat > "$OUTPUT_DIR/VERSION.txt" << EOF
MoXing Ollama-Patched llama.cpp Binaries
==========================================

Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
LLama.cpp Commit: $LLAMA_CPP_COMMIT
Patches: Ollama ($(ls -1 "$PATCHES_DIR"/*.patch 2>/dev/null | wc -l | tr -d ' ') patches)

Available Platforms:
$(ls -1 "$OUTPUT_DIR" | grep -v VERSION | sed 's/^/  - /')

Note: These binaries include Ollama's custom patches for enhanced model support.
Built by: MoXing Build Script
EOF
    
    log_success "版本信息创建完成"
}

# 主函数
main() {
    echo "=============================================="
    echo "编译 Ollama 补丁版 llama.cpp"
    echo "=============================================="
    
    mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
    
    # 下载补丁
    download_patches
    
    # 下载兼容的 llama.cpp 版本
    download_llama_cpp
    
    # 应用补丁
    apply_ollama_patches
    
    # 根据平台编译
    OS="$(uname -s)"
    
    case "$OS" in
        Darwin)
            log_info "检测到 macOS 系统"
            build_macos_metal
            build_macos_cpu
            ;;
        Linux)
            log_info "检测到 Linux 系统"
            build_linux_cuda
            build_linux_vulkan
            build_linux_rocm
            build_linux_cpu
            ;;
        *)
            log_error "不支持的操作系统：$OS"
            exit 1
            ;;
    esac
    
    # 创建版本信息
    create_version_info
    
    echo ""
    echo "=============================================="
    log_success "编译完成！"
    echo "=============================================="
    echo "输出目录：$OUTPUT_DIR"
    echo ""
    ls -la "$OUTPUT_DIR"
}

# 运行
main "$@"
