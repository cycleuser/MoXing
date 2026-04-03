#!/bin/bash
# MoXing 全平台 Ollama 补丁版 llama.cpp 编译脚本
# 支持：macOS (Metal/CPU), Linux (CUDA/Vulkan/ROCm/CPU), Windows (CUDA/Vulkan/CPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_LLAMA_DIR="/Users/fred/Documents/GitHub/Others/ollama/llama"
LLAMA_CPP_DIR="$OLLAMA_LLAMA_DIR/llama.cpp"
OUTPUT_BASE="$SCRIPT_DIR/../moxing/bin"
PATCHES_DIR="$OLLAMA_LLAMA_DIR/patches"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

# 设置完整的 CMake 构建系统
setup_cmake_system() {
    log_info "设置 CMake 构建系统..."
    cd "$LLAMA_CPP_DIR"
    
    # 从官方克隆 CMake 文件
    if [ ! -f "CMakeLists.txt" ]; then
        curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CMakeLists.txt" -o CMakeLists.txt
    fi
    
    if [ ! -d "cmake" ]; then
        mkdir -p cmake
        curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/cmake/common.cmake" -o cmake/common.cmake
        curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/cmake/license.cmake" -o cmake/license.cmake
        curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/cmake/build-info.cmake" -o cmake/build-info.cmake
    fi
    
    if [ ! -d "ggml/cmake" ]; then
        mkdir -p ggml/cmake
        curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/cmake/common.cmake" -o ggml/cmake/common.cmake
    fi
    
    log_success "CMake 构建系统设置完成"
}

# macOS Metal 编译
build_macos_metal() {
    log_info "编译 macOS Metal..."
    cd "$LLAMA_CPP_DIR"
    
    rm -rf build-metal-osx
    mkdir build-metal-osx && cd build-metal-osx
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DCMAKE_INSTALL_PREFIX="$OUTPUT_BASE/darwin-arm64-metal"
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    mkdir -p "$OUTPUT_BASE/darwin-arm64-metal"
    cp bin/llama-server "$OUTPUT_BASE/darwin-arm64-metal/" 2>/dev/null || true
    cp bin/llama-cli "$OUTPUT_BASE/darwin-arm64-metal/" 2>/dev/null || true
    cp bin/llama-bench "$OUTPUT_BASE/darwin-arm64-metal/" 2>/dev/null || true
    cp bin/llama-quantize "$OUTPUT_BASE/darwin-arm64-metal/" 2>/dev/null || true
    
    log_success "macOS Metal 编译完成"
}

# macOS CPU 编译
build_macos_cpu() {
    log_info "编译 macOS CPU..."
    cd "$LLAMA_CPP_DIR"
    
    rm -rf build-cpu-osx
    mkdir build-cpu-osx && cd build-cpu-osx
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIPBLAS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    mkdir -p "$OUTPUT_BASE/darwin-arm64-cpu"
    cp bin/llama-server "$OUTPUT_BASE/darwin-arm64-cpu/" 2>/dev/null || true
    cp bin/llama-cli "$OUTPUT_BASE/darwin-arm64-cpu/" 2>/dev/null || true
    cp bin/llama-bench "$OUTPUT_BASE/darwin-arm64-cpu/" 2>/dev/null || true
    cp bin/llama-quantize "$OUTPUT_BASE/darwin-arm64-cpu/" 2>/dev/null || true
    
    log_success "macOS CPU 编译完成"
}

# Linux CUDA 编译（需要 Linux 系统）
build_linux_cuda() {
    log_warning "Linux CUDA 编译需要在 Linux 系统上执行"
    log_info "创建 Linux CUDA 编译脚本..."
    
    cat > "$OUTPUT_BASE/build-linux-cuda.sh" << 'LINUXSCRIPT'
#!/bin/bash
set -e
OUTPUT_DIR="${1:-./moxing-binaries/linux-x64-cuda}"
LLAMA_CPP_DIR="${2:-/tmp/llama.cpp}"

echo "编译 Linux CUDA..."
cd "$LLAMA_CPP_DIR"
mkdir -p build-linux-cuda && cd build-linux-cuda

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_VULKAN=OFF \
    -DGGML_HIPBLAS=OFF \
    -DGGML_METAL=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_COMMON=ON

cmake --build . --config Release -j$(nproc)

mkdir -p "$OUTPUT_DIR"
cp bin/llama-server "$OUTPUT_DIR/"
cp bin/llama-cli "$OUTPUT_DIR/"
cp bin/llama-bench "$OUTPUT_DIR/"
cp bin/llama-quantize "$OUTPUT_DIR/"

echo "Linux CUDA 编译完成：$OUTPUT_DIR"
LINUXSCRIPT
    chmod +x "$OUTPUT_BASE/build-linux-cuda.sh"
    log_success "Linux CUDA 编译脚本已创建"
}

# Linux Vulkan 编译
build_linux_vulkan() {
    log_warning "Linux Vulkan 编译需要在 Linux 系统上执行"
    log_info "创建 Linux Vulkan 编译脚本..."
    
    cat > "$OUTPUT_BASE/build-linux-vulkan.sh" << 'LINUXSCRIPT'
#!/bin/bash
set -e
OUTPUT_DIR="${1:-./moxing-binaries/linux-x64-vulkan}"
LLAMA_CPP_DIR="${2:-/tmp/llama.cpp}"

echo "编译 Linux Vulkan..."
cd "$LLAMA_CPP_DIR"
mkdir -p build-linux-vulkan && cd build-linux-vulkan

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA=OFF \
    -DGGML_HIPBLAS=OFF \
    -DGGML_METAL=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_COMMON=ON

cmake --build . --config Release -j$(nproc)

mkdir -p "$OUTPUT_DIR"
cp bin/llama-server "$OUTPUT_DIR/"
cp bin/llama-cli "$OUTPUT_DIR/"
cp bin/llama-bench "$OUTPUT_DIR/"
cp bin/llama-quantize "$OUTPUT_DIR/"

echo "Linux Vulkan 编译完成：$OUTPUT_DIR"
LINUXSCRIPT
    chmod +x "$OUTPUT_BASE/build-linux-vulkan.sh"
    log_success "Linux Vulkan 编译脚本已创建"
}

# Linux ROCm 编译
build_linux_rocm() {
    log_warning "Linux ROCm 编译需要在 Linux 系统上执行"
    log_info "创建 Linux ROCm 编译脚本..."
    
    cat > "$OUTPUT_BASE/build-linux-rocm.sh" << 'LINUXSCRIPT'
#!/bin/bash
set -e
OUTPUT_DIR="${1:-./moxing-binaries/linux-x64-rocm}"
LLAMA_CPP_DIR="${2:-/tmp/llama.cpp}"

echo "编译 Linux ROCm..."
cd "$LLAMA_CPP_DIR"
mkdir -p build-linux-rocm && cd build-linux-rocm

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIPBLAS=ON \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_COMMON=ON

cmake --build . --config Release -j$(nproc)

mkdir -p "$OUTPUT_DIR"
cp bin/llama-server "$OUTPUT_DIR/"
cp bin/llama-cli "$OUTPUT_DIR/"
cp bin/llama-bench "$OUTPUT_DIR/"
cp bin/llama-quantize "$OUTPUT_DIR/"

echo "Linux ROCm 编译完成：$OUTPUT_DIR"
LINUXSCRIPT
    chmod +x "$OUTPUT_BASE/build-linux-rocm.sh"
    log_success "Linux ROCm 编译脚本已创建"
}

# Windows CUDA 编译（需要 Windows 系统）
build_windows_cuda() {
    log_warning "Windows CUDA 编译需要在 Windows 系统上执行"
    log_info "创建 Windows CUDA 编译脚本..."
    
    cat > "$OUTPUT_BASE/build-windows-cuda.ps1" << 'WINSCRIPT'
# Windows CUDA 编译脚本 (PowerShell)
$OutputDir = if ($args[0]) { $args[0] } else { ".\moxing-binaries\windows-x64-cuda" }
$LlamaCppDir = if ($args[1]) { $args[1] } else { "C:\llama.cpp" }

Write-Host "编译 Windows CUDA..."
Set-Location $LlamaCppDir
New-Item -ItemType Directory -Force -Path "build-windows-cuda" | Out-Null
Set-Location "build-windows-cuda"

cmake .. `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DGGML_VULKAN=OFF `
    -DGGML_HIPBLAS=OFF `
    -DGGML_METAL=OFF `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_COMMON=ON `
    -DCMAKE_GENERATOR="Visual Studio 17 2022"

cmake --build . --config Release -j $env:NUMBER_OF_PROCESSORS

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Copy-Item "bin\Release\llama-server.exe" "$OutputDir/"
Copy-Item "bin\Release\llama-cli.exe" "$OutputDir/"
Copy-Item "bin\Release\llama-bench.exe" "$OutputDir/"
Copy-Item "bin\Release\llama-quantize.exe" "$OutputDir/"

Write-Host "Windows CUDA 编译完成：$OutputDir"
WINSCRIPT
    log_success "Windows CUDA 编译脚本已创建"
}

# Windows Vulkan 编译
build_windows_vulkan() {
    log_warning "Windows Vulkan 编译需要在 Windows 系统上执行"
    log_info "创建 Windows Vulkan 编译脚本..."
    
    cat > "$OUTPUT_BASE/build-windows-vulkan.ps1" << 'WINSCRIPT'
# Windows Vulkan 编译脚本 (PowerShell)
$OutputDir = if ($args[0]) { $args[0] } else { ".\moxing-binaries\windows-x64-vulkan" }
$LlamaCppDir = if ($args[1]) { $args[1] } else { "C:\llama.cpp" }

Write-Host "编译 Windows Vulkan..."
Set-Location $LlamaCppDir
New-Item -ItemType Directory -Force -Path "build-windows-vulkan" | Out-Null
Set-Location "build-windows-vulkan"

cmake .. `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_VULKAN=ON `
    -DGGML_CUDA=OFF `
    -DGGML_HIPBLAS=OFF `
    -DGGML_METAL=OFF `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_COMMON=ON `
    -DCMAKE_GENERATOR="Visual Studio 17 2022"

cmake --build . --config Release -j $env:NUMBER_OF_PROCESSORS

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Copy-Item "bin\Release\llama-server.exe" "$OutputDir/"
Copy-Item "bin\Release\llama-cli.exe" "$OutputDir/"
Copy-Item "bin\Release\llama-bench.exe" "$OutputDir/"
Copy-Item "bin\Release\llama-quantize.exe" "$OutputDir/"

Write-Host "Windows Vulkan 编译完成：$OutputDir"
WINSCRIPT
    log_success "Windows Vulkan 编译脚本已创建"
}

# 创建版本信息
create_version_info() {
    cat > "$OUTPUT_BASE/OLLAMA_BUILD_INFO.txt" << EOF
MoXing Ollama-Patched llama.cpp Binaries
==========================================

Source: /Users/fred/Documents/GitHub/Others/ollama/llama/llama.cpp
Patches: Ollama 35 patches applied
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Build Script: build-all-platforms.sh

Available Binaries:
$(ls -1 "$OUTPUT_BASE" | grep -E "darwin|linux|windows" | sed 's/^/  - /')

Build Instructions for Other Platforms:
  Linux CUDA:    ./build-linux-cuda.sh    <output_dir> <llama.cpp_dir>
  Linux Vulkan:  ./build-linux-vulkan.sh  <output_dir> <llama.cpp_dir>
  Linux ROCm:    ./build-linux-rocm.sh    <output_dir> <llama.cpp_dir>
  Windows CUDA:  powershell .\build-windows-cuda.ps1   <output_dir> <llama.cpp_dir>
  Windows Vulkan: powershell .\build-windows-vulkan.ps1 <output_dir> <llama.cpp_dir>

Built by: MoXing Build Script
EOF
    
    log_success "版本信息创建完成"
}

# 主函数
main() {
    echo "=============================================="
    echo "MoXing Ollama 补丁版 llama.cpp 全平台编译"
    echo "=============================================="
    
    # 检查目录
    if [ ! -d "$OLLAMA_LLAMA_DIR" ]; then
        log_error "Ollama llama 目录不存在：$OLLAMA_LLAMA_DIR"
        exit 1
    fi
    
    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        log_error "llama.cpp 目录不存在：$LLAMA_CPP_DIR"
        exit 1
    fi
    
    mkdir -p "$OUTPUT_BASE"
    
    # 设置 CMake
    setup_cmake_system
    
    # macOS 编译
    build_macos_metal
    build_macos_cpu
    
    # 其他平台创建脚本
    build_linux_cuda
    build_linux_vulkan
    build_linux_rocm
    build_windows_cuda
    build_windows_vulkan
    
    # 版本信息
    create_version_info
    
    echo ""
    echo "=============================================="
    log_success "编译完成！"
    echo "=============================================="
    echo ""
    echo "已编译 (macOS):"
    ls -la "$OUTPUT_BASE/darwin-"*/ 2>/dev/null | head -20
    echo ""
    echo "其他平台编译脚本已创建，请在对应系统上运行:"
    echo "  Linux:   ./build-linux-*.sh"
    echo "  Windows: powershell .\build-windows-*.ps1"
}

main "$@"
