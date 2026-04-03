#!/bin/bash
# MoXing 多平台二进制编译脚本
# 策略：优先使用 Ollama 补丁版，失败则降级到 llama.cpp 官方版

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/ollama-llama-build"
OUTPUT_DIR="$SCRIPT_DIR/ollama-binaries"
LLAMA_CPP_VERSION="b4376"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 清理
cleanup() {
    log_info "清理工作目录..."
    rm -rf "$WORK_DIR"
}

# 下载并编译 llama.cpp 官方版（降级方案）
build_llama_cpp_official() {
    log_info "使用 llama.cpp 官方版（降级方案）..."
    cd "$WORK_DIR"
    
    if [ ! -d "llama.cpp" ]; then
        git clone --depth 1 --branch "$LLAMA_CPP_VERSION" https://github.com/ggerganov/llama.cpp.git
    fi
    
    cd llama.cpp
    
    # macOS Metal
    if [ "$(uname -s)" = "Darwin" ]; then
        log_info "编译 macOS Metal 版本..."
        mkdir -p build-metal && cd build-metal
        cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
        cmake --build . --config Release -j$(sysctl -n hw.ncpu)
        
        mkdir -p "$OUTPUT_DIR/darwin-arm64-metal"
        cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-metal/"
        cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-metal/"
        cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-metal/"
        cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-metal/"
        
        cd ..
        
        # macOS CPU
        log_info "编译 macOS CPU 版本..."
        mkdir -p build-cpu && cd build-cpu
        cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
        cmake --build . --config Release -j$(sysctl -n hw.ncpu)
        
        mkdir -p "$OUTPUT_DIR/darwin-arm64-cpu"
        cp bin/llama-server "$OUTPUT_DIR/darwin-arm64-cpu/"
        cp bin/llama-cli "$OUTPUT_DIR/darwin-arm64-cpu/"
        cp bin/llama-bench "$OUTPUT_DIR/darwin-arm64-cpu/"
        cp bin/llama-quantize "$OUTPUT_DIR/darwin-arm64-cpu/"
    elif [ "$(uname -s)" = "Linux" ]; then
        # Linux CPU
        log_info "编译 Linux CPU 版本..."
        mkdir -p build-cpu && cd build-cpu
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release -j$(nproc)
        
        mkdir -p "$OUTPUT_DIR/linux-x64-cpu"
        cp bin/llama-server "$OUTPUT_DIR/linux-x64-cpu/"
        cp bin/llama-cli "$OUTPUT_DIR/linux-x64-cpu/"
        cp bin/llama-bench "$OUTPUT_DIR/linux-x64-cpu/"
        cp bin/llama-quantize "$OUTPUT_DIR/linux-x64-cpu/"
    fi
    
    log_success "llama.cpp 官方版编译完成"
}

# 尝试下载 Ollama 预编译二进制
download_ollama_binaries() {
    log_info "尝试下载 Ollama 预编译二进制..."
    
    OS="$(uname -s)"
    
    case "$OS" in
        Darwin)
            log_info "下载 macOS 版本..."
            mkdir -p "$OUTPUT_DIR/darwin-arm64-metal"
            curl -sL "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin" -o "$OUTPUT_DIR/darwin-arm64-metal/ollama"
            chmod +x "$OUTPUT_DIR/darwin-arm64-metal/ollama"
            ;;
        Linux)
            log_info "下载 Linux 版本..."
            mkdir -p "$OUTPUT_DIR/linux-x64-cuda"
            curl -sL "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz" -o ollama-linux.tgz
            tar -xzf ollama-linux.tgz -C "$OUTPUT_DIR/linux-x64-cuda/"
            ;;
    esac
}

# 主函数
main() {
    echo "=============================================="
    echo "MoXing 多平台二进制编译"
    echo "=============================================="
    
    mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
    
    # 策略 1: 尝试下载 Ollama 预编译二进制
    download_ollama_binaries
    
    # 策略 2: 降级到 llama.cpp 官方版
    build_llama_cpp_official
    
    # 创建版本信息
    cat > "$OUTPUT_DIR/VERSION.txt" << EOF
MoXing Binaries
===============

Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
LLama.cpp Version: $LLAMA_CPP_VERSION
Source: llama.cpp official (fallback from Ollama patches)

Available Platforms:
$(ls -1 "$OUTPUT_DIR" | grep -v VERSION)

Built by: MoXing Build Script
EOF
    
    log_success "编译完成！输出目录：$OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
}

main "$@"
