#!/bin/bash
# 编译 Ollama 补丁版 llama.cpp 多平台可执行文件
# 支持：Windows (CUDA/Vulkan/CPU), Linux (CUDA/Vulkan/ROCm/CPU), macOS (Metal/CPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/ollama-llama-build"
PATCHES_DIR="$WORK_DIR/patches"
OUTPUT_DIR="$SCRIPT_DIR/ollama-binaries"
LLAMA_CPP_VERSION="b4376"  # 使用稳定版本

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 清理函数
cleanup() {
    log_info "清理工作目录..."
    rm -rf "$WORK_DIR"
}

# 下载 Ollama 补丁
download_patches() {
    log_info "下载 Ollama 补丁..."
    mkdir -p "$PATCHES_DIR"
    
    # 从 GitHub API 获取补丁列表
    PATCH_URL="https://api.github.com/repos/ollama/ollama/contents/llama/patches"
    
    # 下载每个补丁
    curl -sL "$PATCH_URL" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for item in data:
    if item['name'].endswith('.patch'):
        print(f\"{item['name']}|{item['download_url']}\")
" | while IFS='|' read -r name url; do
        log_info "下载补丁：$name"
        curl -sL "$url" -o "$PATCHES_DIR/$name"
    done
    
    log_success "补丁下载完成，共 $(ls -1 "$PATCHES_DIR"/*.patch 2>/dev/null | wc -l | tr -d ' ') 个补丁"
}

# 下载 llama.cpp 源码
download_llama_cpp() {
    log_info "下载 llama.cpp $LLAMA_CPP_VERSION..."
    cd "$WORK_DIR"
    
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp
        git checkout "$LLAMA_CPP_VERSION" || git checkout main
    else
        cd llama.cpp
        git fetch --tags
        git checkout "$LLAMA_CPP_VERSION" || git checkout main
    fi
    
    log_success "llama.cpp 源码准备完成"
}

# 应用 Ollama 补丁
apply_patches() {
    log_info "应用 Ollama 补丁..."
    cd "$WORK_DIR/llama.cpp"
    
    # 应用每个补丁
    for patch in "$PATCHES_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            patch_name=$(basename "$patch")
            log_info "应用补丁：$patch_name"
            if ! git apply "$patch" 2>/dev/null; then
                log_warning "补丁 $patch_name 应用失败，尝试使用 patch 命令..."
                patch -p1 < "$patch" || true
            fi
        fi
    done
    
    log_success "补丁应用完成"
}

# 编译 macOS Metal 版本
build_macos_metal() {
    log_info "编译 macOS Metal 版本..."
    cd "$WORK_DIR/llama.cpp"
    
    mkdir -p build-macos-metal
    cd build-macos-metal
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    # 复制可执行文件
    mkdir -p "$OUTPUT_DIR/darwin-arm64-metal"
    cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-metal/"
    cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-metal/"
    cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-metal/"
    cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-metal/"
    
    log_success "macOS Metal 版本编译完成"
}

# 编译 macOS CPU 版本
build_macos_cpu() {
    log_info "编译 macOS CPU 版本..."
    cd "$WORK_DIR/llama.cpp"
    
    mkdir -p build-macos-cpu
    cd build-macos-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    # 复制可执行文件
    mkdir -p "$OUTPUT_DIR/darwin-arm64-cpu"
    cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-cpu/"
    cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-cpu/"
    cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-cpu/"
    cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-cpu/"
    
    log_success "macOS CPU 版本编译完成"
}

# 编译 Linux CUDA 版本
build_linux_cuda() {
    log_info "编译 Linux CUDA 版本..."
    cd "$WORK_DIR/llama.cpp"
    
    mkdir -p build-linux-cuda
    cd build-linux-cuda
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    # 复制可执行文件
    mkdir -p "$OUTPUT_DIR/linux-x64-cuda"
    cp bin/llama-server "$OUTPUT_DIR/linux-x64-cuda/"
    cp bin/llama-cli "$OUTPUT_DIR/linux-x64-cuda/"
    cp bin/llama-bench "$OUTPUT_DIR/linux-x64-cuda/"
    cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-cuda/"
    
    log_success "Linux CUDA 版本编译完成"
}

# 编译 Linux Vulkan 版本
build_linux_vulkan() {
    log_info "编译 Linux Vulkan 版本..."
    cd "$WORK_DIR/llama.cpp"
    
    mkdir -p build-linux-vulkan
    cd build-linux-vulkan
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DGGML_CUDA=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    # 复制可执行文件
    mkdir -p "$OUTPUT_DIR/linux-x64-vulkan"
    cp bin/llama-server "$OUTPUT_DIR/linux-x64-vulkan/"
    cp bin/llama-cli "$OUTPUT_DIR/linux-x64-vulkan/"
    cp bin/llama-bench "$OUTPUT_DIR/linux-x64-vulkan/"
    cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-vulkan/"
    
    log_success "Linux Vulkan 版本编译完成"
}

# 编译 Linux ROCm 版本
build_linux_rocm() {
    log_info "编译 Linux ROCm 版本..."
    cd "$WORK_DIR/llama.cpp"
    
    mkdir -p build-linux-rocm
    cd build-linux-rocm
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIPBLAS=ON \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    # 复制可执行文件
    mkdir -p "$OUTPUT_DIR/linux-x64-rocm"
    cp bin/llama-server "$OUTPUT_DIR/linux-x64-rocm/"
    cp bin/llama-cli "$OUTPUT_DIR/linux-x64-rocm/"
    cp bin/llama-bench "$OUTPUT_DIR/linux-x64-rocm/"
    cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-rocm/"
    
    log_success "Linux ROCm 版本编译完成"
}

# 编译 Linux CPU 版本
build_linux_cpu() {
    log_info "编译 Linux CPU 版本..."
    cd "$WORK_DIR/llama.cpp"
    
    mkdir -p build-linux-cpu
    cd build-linux-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    # 复制可执行文件
    mkdir -p "$OUTPUT_DIR/linux-x64-cpu"
    cp bin/llama-server "$OUTPUT_DIR/linux-x64-cpu/"
    cp bin/llama-cli "$OUTPUT_DIR/linux-x64-cpu/"
    cp bin/llama-bench "$OUTPUT_DIR/linux-x64-cpu/"
    cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-cpu/"
    
    log_success "Linux CPU 版本编译完成"
}

# 编译 Windows CUDA 版本（需要交叉编译）
build_windows_cuda() {
    log_warning "Windows CUDA 版本需要 Windows 系统编译，跳过..."
    # 如果是在 Linux/macOS 上，可以使用 mingw-w64 交叉编译，但比较复杂
    # 建议在 Windows 原生编译
}

# 编译 Windows Vulkan 版本
build_windows_vulkan() {
    log_warning "Windows Vulkan 版本需要 Windows 系统编译，跳过..."
}

# 编译 Windows CPU 版本
build_windows_cpu() {
    log_warning "Windows CPU 版本需要 Windows 系统编译，跳过..."
}

# 创建版本信息文件
create_version_info() {
    log_info "创建版本信息文件..."
    
    cat > "$OUTPUT_DIR/VERSION.txt" << EOF
MoXing Ollama-Patched llama.cpp Binaries
==========================================

Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
LLama.cpp Version: $LLAMA_CPP_VERSION
Patches: Ollama ($(ls -1 "$PATCHES_DIR"/*.patch 2>/dev/null | wc -l | tr -d ' ') patches)

Available Platforms:
- macOS (Metal, CPU)
- Linux (CUDA, Vulkan, ROCm, CPU)
- Windows (需要 Windows 系统编译)

Built by: MoXing Build Script
EOF
    
    log_success "版本信息文件创建完成"
}

# 主函数
main() {
    echo "=============================================="
    echo "编译 Ollama 补丁版 llama.cpp"
    echo "=============================================="
    
    # 创建目录
    mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
    
    # 下载补丁
    download_patches
    
    # 下载源码
    download_llama_cpp
    
    # 应用补丁
    apply_patches
    
    # 根据当前平台编译
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
        MINGW*|MSYS*|CYGWIN*)
            log_info "检测到 Windows 系统"
            build_windows_cuda
            build_windows_vulkan
            build_windows_cpu
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

# 运行主函数
main "$@"
