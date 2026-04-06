#!/bin/bash
#
# 交叉编译 Windows CPU 版本
# 使用 MinGW-w64 在 Linux 上编译 Windows 可执行文件
#
# 用法: ./scripts/cross_compile_windows.sh
#
# 注意：
# - 仅支持 CPU 版本
# - CUDA/Vulkan 需要 Windows SDK，无法交叉编译
# - 推荐使用 GitHub Actions 自动化完整构建
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

check_mingw() {
    if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
        log_error "MinGW-w64 未安装"
        echo ""
        echo "安装方法:"
        echo "  sudo apt install -y mingw-w64"
        echo ""
        exit 1
    fi
    
    MINGW_VERSION=$(x86_64-w64-mingw32-gcc --version | head -1)
    log_success "MinGW-w64: $MINGW_VERSION"
}

prepare_llama_cpp() {
    LLAMA_DIR="$PROJECT_DIR/build/llama.cpp-windows"
    
    log_info "准备 llama.cpp..."
    
    if [ -d "$LLAMA_DIR" ]; then
        log_info "更新现有 llama.cpp..."
        cd "$LLAMA_DIR"
        git fetch --all
        git reset --hard origin/master
    else
        log_info "克隆 llama.cpp..."
        mkdir -p "$PROJECT_DIR/build"
        git clone --depth 1 --recursive https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi
    
    log_success "llama.cpp 准备完成"
}

cross_compile_cpu() {
    log_info "交叉编译 Windows CPU 版本..."
    
    cd "$LLAMA_DIR"
    
    rm -rf build-windows-cpu
    mkdir -p build-windows-cpu
    cd build-windows-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Windows \
        -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
        -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
        -DCMAKE_FIND_ROOT_PATH=/usr/x86_64-w64-mingw32 \
        -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
        -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
        -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    log_success "Windows CPU 构建完成"
}

copy_binaries() {
    log_info "复制 Windows 二进制..."
    
    WIN_DIR="$PROJECT_DIR/moxing/bin/windows-x64-cpu-cross"
    mkdir -p "$WIN_DIR"
    
    cd "$LLAMA_DIR/build-windows-cpu"
    
    cp bin/llama-server.exe "$WIN_DIR/" 2>/dev/null || log_warning "server 未找到"
    cp bin/llama-cli.exe "$WIN_DIR/" 2>/dev/null || true
    cp bin/llama-bench.exe "$WIN_DIR/" 2>/dev/null || true
    cp libggml.dll "$WIN_DIR/" 2>/dev/null || true
    cp libggml-base.dll "$WIN_DIR/" 2>/dev/null || true
    cp libggml-cpu.dll "$WIN_DIR/" 2>/dev/null || true
    cp libllama.dll "$WIN_DIR/" 2>/dev/null || true
    
    log_success "Windows 二进制复制完成"
    
    echo ""
    echo "已复制的文件:"
    ls -lh "$WIN_DIR/"
}

package_windows_cpu() {
    log_info "打包 Windows CPU..."
    
    VERSION=$(python3 -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from moxing import __version__; print(__version__)")
    PACKAGE_NAME="moxing-windows-cpu-cross-$VERSION"
    DIST_DIR="$PROJECT_DIR/dist"
    
    mkdir -p "$DIST_DIR/$PACKAGE_NAME/lib"
    
    WIN_DIR="$PROJECT_DIR/moxing/bin/windows-x64-cpu-cross"
    cp "$WIN_DIR/*" "$DIST_DIR/$PACKAGE_NAME/lib/"
    
    cat > "$DIST_DIR/$PACKAGE_NAME/install.bat" << 'BATEOF'
@echo off
echo Installing MoXing Windows CPU binaries...
echo.
echo Copying files to C:\moxing\cpu...
if not exist "C:\moxing\cpu" mkdir "C:\moxing\cpu"
xcopy lib "C:\moxing\cpu" /E /I /Y
echo.
echo Done!
echo Usage: moxing serve model.gguf -b cpu
BATEOF
    
    cat > "$DIST_DIR/$PACKAGE_NAME/README.md" << 'MDEOF'
# MoXing Windows CPU (Cross-compiled)

This package contains CPU-only binaries cross-compiled from Linux.

## Installation

```cmd
install.bat
```

## Usage

```cmd
moxing serve model.gguf -b cpu
```

## Limitations

- CPU only (no CUDA/Vulkan)
- Cross-compiled with MinGW-w64
- For full Windows support, use GitHub Actions build

## Size

~10MB (CPU binaries only)
MDEOF
    
    cd "$DIST_DIR"
    zip -r "$PACKAGE_NAME.zip" "$PACKAGE_NAME"
    
    log_success "Windows CPU 包已创建: $PACKAGE_NAME.zip"
    
    echo ""
    ls -lh "$DIST_DIR/$PACKAGE_NAME.zip"
}

show_summary() {
    echo ""
    echo -e "${BLUE}==============================================${NC}"
    echo -e "${GREEN}  Windows 交叉编译完成${NC}"
    echo -e "${BLUE}==============================================${NC}"
    echo ""
    echo "注意："
    echo "  - 仅编译了 CPU 版本"
    echo "  - CUDA/Vulkan 需要 Windows SDK，无法交叉编译"
    echo ""
    echo "完整 Windows 构建推荐使用 GitHub Actions:"
    echo "  git tag v0.1.27"
    echo "  git push origin v0.1.27"
    echo ""
    echo "GitHub Actions 会自动构建:"
    echo "  - Windows CUDA"
    echo "  - Windows Vulkan"
    echo "  - Windows CPU"
    echo ""
}

main() {
    echo ""
    echo -e "${BLUE}==============================================${NC}"
    echo -e "${GREEN}  MoXing Windows 交叉编译${NC}"
    echo -e "${BLUE}==============================================${NC}"
    echo ""
    
    check_mingw
    prepare_llama_cpp
    cross_compile_cpu
    copy_binaries
    package_windows_cpu
    show_summary
}

main "$@"