#!/bin/bash
#
# MoXing 全平台二进制构建脚本
#
# 用法:
#   ./scripts/build_all_binaries.sh           # 构建当前平台
#   ./scripts/build_all_binaries.sh --upload  # 构建并上传
#   ./scripts/build_all_binaries.sh --help    # 显示帮助
#
# 支持平台:
#   - Linux: CUDA / ROCm / Vulkan / CPU
#   - macOS: Metal / CPU
#   - Windows: 生成 PowerShell 脚本
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_step()    { echo -e "${CYAN}==>${NC} $1"; }

detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux"; ARCH=$(uname -m) ;;
        Darwin*)    PLATFORM="darwin"; ARCH=$(uname -m) ;;
        CYGWIN*|MINGW*|MSYS*)  PLATFORM="windows"; ARCH="x64" ;;
        *)          PLATFORM="unknown"; ARCH="unknown" ;;
    esac
    log_info "平台: $PLATFORM-$ARCH"
}

check_dependencies() {
    log_step "检查依赖..."
    
    local missing=()
    
    if ! command -v cmake &> /dev/null; then
        missing+=("cmake")
    fi
    
    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "缺少依赖: ${missing[*]}"
        echo ""
        echo "安装方法:"
        case "$PLATFORM" in
            linux)
                echo "  sudo apt install -y cmake git python3"
                ;;
            darwin)
                echo "  brew install cmake git python3"
                ;;
        esac
        exit 1
    fi
    
    log_success "依赖检查通过"
}

prepare_llama_cpp() {
    LLAMA_DIR="$PROJECT_DIR/build/llama.cpp"
    
    log_step "准备 llama.cpp 源码..."
    
    if [ -d "$LLAMA_DIR" ]; then
        log_info "更新现有 llama.cpp..."
        cd "$LLAMA_DIR"
        git fetch --all
        git reset --hard origin/master
        git submodule update --init --recursive
    else
        log_info "克隆 llama.cpp..."
        mkdir -p "$PROJECT_DIR/build"
        git clone --depth 1 --recursive https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi
    
    cd "$LLAMA_DIR"
    VERSION=$(git describe --tags 2>/dev/null || echo "unknown")
    log_success "llama.cpp 准备完成: $VERSION"
}

build_linux_cuda() {
    log_step "构建 Linux CUDA..."
    
    cd "$LLAMA_DIR"
    rm -rf build-cuda
    mkdir -p build-cuda
    cd build-cuda
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DLLAMA_CUBLAS=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
    
    cmake --build . --config Release -j$(nproc)
    
    log_success "CUDA 构建完成"
}

build_linux_rocm() {
    log_step "构建 Linux ROCm..."
    
    if ! command -v hipcc &> /dev/null; then
        log_warning "ROCm 未安装，跳过"
        return 0
    fi
    
    cd "$LLAMA_DIR"
    rm -rf build-rocm
    mkdir -p build-rocm
    cd build-rocm
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIPBLAS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DHIP_PATH=/opt/rocm
    
    cmake --build . --config Release -j$(nproc)
    
    log_success "ROCm 构建完成"
}

build_linux_vulkan() {
    log_step "构建 Linux Vulkan..."
    
    cd "$LLAMA_DIR"
    rm -rf build-vulkan
    mkdir -p build-vulkan
    cd build-vulkan
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    log_success "Vulkan 构建完成"
}

build_linux_cpu() {
    log_step "构建 Linux CPU..."
    
    cd "$LLAMA_DIR"
    rm -rf build-cpu
    mkdir -p build-cpu
    cd build-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(nproc)
    
    log_success "CPU 构建完成"
}

build_linux() {
    log_info "开始 Linux 构建..."
    
    build_linux_cuda
    build_linux_rocm
    build_linux_vulkan
    build_linux_cpu
    
    log_success "Linux 构建完成"
}

build_macos_metal() {
    log_step "构建 macOS Metal..."
    
    cd "$LLAMA_DIR"
    rm -rf build-metal
    mkdir -p build-metal
    cd build-metal
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    log_success "Metal 构建完成"
}

build_macos_cpu() {
    log_step "构建 macOS CPU..."
    
    cd "$LLAMA_DIR"
    rm -rf build-cpu
    mkdir -p build-cpu
    cd build-cpu
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    log_success "CPU 构建完成"
}

build_macos() {
    log_info "开始 macOS 构建..."
    
    build_macos_metal
    build_macos_cpu
    
    log_success "macOS 构建完成"
}

generate_windows_script() {
    log_step "生成 Windows 构建脚本..."
    
    WIN_SCRIPT="$PROJECT_DIR/scripts/build_windows.ps1"
    
    cat > "$WIN_SCRIPT" << 'PSEOF'
# MoXing Windows 构建脚本
# 用法: .\scripts\build_windows.ps1 [-Backend cuda|vulkan|cpu|all]

param(
    [ValidateSet("cuda", "vulkan", "cpu", "all")]
    [string]$Backend = "all"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$LlamaDir = Join-Path $ProjectDir "build\llama.cpp"

function Build-CUDA {
    Write-Host "构建 CUDA 后端..." -ForegroundColor Cyan
    Set-Location $LlamaDir
    
    if (Test-Path "build-cuda") { Remove-Item -Recurse -Force "build-cuda" }
    New-Item -ItemType Directory -Force -Path "build-cuda" | Out-Null
    Set-Location "build-cuda"
    
    cmake .. `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_COMMON=ON `
        -G "Visual Studio 17 2022" `
        -A x64
    
    cmake --build . --config Release --parallel $env:NUMBER_OF_PROCESSORS
    
    Write-Host "CUDA 构建完成" -ForegroundColor Green
}

function Build-Vulkan {
    Write-Host "构建 Vulkan 后端..." -ForegroundColor Cyan
    Set-Location $LlamaDir
    
    if (Test-Path "build-vulkan") { Remove-Item -Recurse -Force "build-vulkan" }
    New-Item -ItemType Directory -Force -Path "build-vulkan" | Out-Null
    Set-Location "build-vulkan"
    
    cmake .. `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_VULKAN=ON `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_COMMON=ON `
        -G "Visual Studio 17 2022" `
        -A x64
    
    cmake --build . --config Release --parallel $env:NUMBER_OF_PROCESSORS
    
    Write-Host "Vulkan 构建完成" -ForegroundColor Green
}

function Build-CPU {
    Write-Host "构建 CPU 后端..." -ForegroundColor Cyan
    Set-Location $LlamaDir
    
    if (Test-Path "build-cpu") { Remove-Item -Recurse -Force "build-cpu" }
    New-Item -ItemType Directory -Force -Path "build-cpu" | Out-Null
    Set-Location "build-cpu"
    
    cmake .. `
        -DCMAKE_BUILD_TYPE=Release `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_COMMON=ON `
        -G "Visual Studio 17 2022" `
        -A x64
    
    cmake --build . --config Release --parallel $env:NUMBER_OF_PROCESSORS
    
    Write-Host "CPU 构建完成" -ForegroundColor Green
}

function Copy-Binaries {
    Write-Host "复制二进制文件..." -ForegroundColor Cyan
    $BinDir = Join-Path $ProjectDir "moxing\bin"
    
    if ($Backend -eq "cuda" -or $Backend -eq "all") {
        $Dest = Join-Path $BinDir "windows-x64-cuda"
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Copy-Item "$LlamaDir\build-cuda\bin\Release\llama-server.exe" $Dest
        Copy-Item "$LlamaDir\build-cuda\bin\Release\llama-cli.exe" $Dest
        Copy-Item "$LlamaDir\build-cuda\bin\Release\llama-bench.exe" $Dest
        Copy-Item "$LlamaDir\build-cuda\*.dll" $Dest
    }
    
    if ($Backend -eq "vulkan" -or $Backend -eq "all") {
        $Dest = Join-Path $BinDir "windows-x64-vulkan"
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Copy-Item "$LlamaDir\build-vulkan\bin\Release\llama-server.exe" $Dest
        Copy-Item "$LlamaDir\build-vulkan\bin\Release\llama-cli.exe" $Dest
        Copy-Item "$LlamaDir\build-vulkan\bin\Release\llama-bench.exe" $Dest
    }
    
    if ($Backend -eq "cpu" -or $Backend -eq "all") {
        $Dest = Join-Path $BinDir "windows-x64-cpu"
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Copy-Item "$LlamaDir\build-cpu\bin\Release\llama-server.exe" $Dest
        Copy-Item "$LlamaDir\build-cpu\bin\Release\llama-cli.exe" $Dest
        Copy-Item "$LlamaDir\build-cpu\bin\Release\llama-bench.exe" $Dest
    }
    
    Write-Host "二进制复制完成" -ForegroundColor Green
}

# 主逻辑
if (-not (Test-Path $LlamaDir)) {
    Write-Error "llama.cpp 未找到: $LlamaDir"
    Write-Host "请先克隆: git clone https://github.com/ggml-org/llama.cpp.git build\llama.cpp"
    exit 1
}

if ($Backend -eq "cuda" -or $Backend -eq "all") { Build-CUDA }
if ($Backend -eq "vulkan" -or $Backend -eq "all") { Build-Vulkan }
if ($Backend -eq "cpu" -or $Backend -eq "all") { Build-CPU }

Copy-Binaries

Write-Host ""
Write-Host "Windows 构建完成!" -ForegroundColor Green
PSEOF
    
    log_success "Windows PowerShell 脚本已生成: $WIN_SCRIPT"
    log_warning "请在 Windows 上运行: .\\scripts\\build_windows.ps1"
}

copy_linux_binaries() {
    log_step "复制 Linux 二进制..."
    
    BIN_DIR="$PROJECT_DIR/moxing/bin"
    mkdir -p "$BIN_DIR"
    
    CUDA_DIR="$BIN_DIR/linux-x64-cuda"
    ROCM_DIR="$BIN_DIR/linux-x64-rocm"
    VULKAN_DIR="$BIN_DIR/linux-x64-vulkan"
    CPU_DIR="$BIN_DIR/linux-x64-cpu"
    
    mkdir -p "$CUDA_DIR" "$ROCM_DIR" "$VULKAN_DIR" "$CPU_DIR"
    
    cd "$LLAMA_DIR"
    
    cp build-cuda/bin/llama-server "$CUDA_DIR/" 2>/dev/null || log_warning "CUDA server 未找到"
    cp build-cuda/bin/llama-cli "$CUDA_DIR/" 2>/dev/null || true
    cp build-cuda/bin/llama-bench "$CUDA_DIR/" 2>/dev/null || true
    cp build-cuda/libggml.so* "$CUDA_DIR/" 2>/dev/null || true
    cp build-cuda/libggml-base.so* "$CUDA_DIR/" 2>/dev/null || true
    cp build-cuda/libggml-cpu.so* "$CUDA_DIR/" 2>/dev/null || true
    cp build-cuda/libggml-cuda.so* "$CUDA_DIR/" 2>/dev/null || true
    cp build-cuda/libllama.so* "$CUDA_DIR/" 2>/dev/null || true
    
    cp build-rocm/bin/llama-server "$ROCM_DIR/" 2>/dev/null || log_warning "ROCm server 未找到"
    cp build-rocm/bin/llama-cli "$ROCM_DIR/" 2>/dev/null || true
    cp build-rocm/bin/llama-bench "$ROCM_DIR/" 2>/dev/null || true
    cp build-rocm/libggml.so* "$ROCM_DIR/" 2>/dev/null || true
    cp build-rocm/libggml-base.so* "$ROCM_DIR/" 2>/dev/null || true
    cp build-rocm/libggml-cpu.so* "$ROCM_DIR/" 2>/dev/null || true
    cp build-rocm/libggml-hip.so* "$ROCM_DIR/" 2>/dev/null || true
    cp build-rocm/libllama.so* "$ROCM_DIR/" 2>/dev/null || true
    
    cp build-vulkan/bin/llama-server "$VULKAN_DIR/" 2>/dev/null || log_warning "Vulkan server 未找到"
    cp build-vulkan/bin/llama-cli "$VULKAN_DIR/" 2>/dev/null || true
    cp build-vulkan/bin/llama-bench "$VULKAN_DIR/" 2>/dev/null || true
    cp build-vulkan/libggml.so* "$VULKAN_DIR/" 2>/dev/null || true
    cp build-vulkan/libggml-base.so* "$VULKAN_DIR/" 2>/dev/null || true
    cp build-vulkan/libggml-cpu.so* "$VULKAN_DIR/" 2>/dev/null || true
    cp build-vulkan/libggml-vulkan.so* "$VULKAN_DIR/" 2>/dev/null || true
    cp build-vulkan/libllama.so* "$VULKAN_DIR/" 2>/dev/null || true
    
    cp build-cpu/bin/llama-server "$CPU_DIR/" 2>/dev/null || log_warning "CPU server 未找到"
    cp build-cpu/bin/llama-cli "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/bin/llama-bench "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libggml.so* "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libggml-base.so* "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libggml-cpu.so* "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libllama.so* "$CPU_DIR/" 2>/dev/null || true
    
    log_success "Linux 二进制复制完成"
    
    echo ""
    echo "已复制的文件:"
    echo "  CUDA:   $(ls $CUDA_DIR | wc -l) 个文件"
    echo "  ROCm:   $(ls $ROCM_DIR | wc -l) 个文件"
    echo "  Vulkan: $(ls $VULKAN_DIR | wc -l) 个文件"
    echo "  CPU:    $(ls $CPU_DIR | wc -l) 个文件"
}

copy_macos_binaries() {
    log_step "复制 macOS 二进制..."
    
    BIN_DIR="$PROJECT_DIR/moxing/bin"
    mkdir -p "$BIN_DIR"
    
    METAL_DIR="$BIN_DIR/darwin-arm64-metal"
    CPU_DIR="$BIN_DIR/darwin-arm64-cpu"
    
    mkdir -p "$METAL_DIR" "$CPU_DIR"
    
    cd "$LLAMA_DIR"
    
    cp build-metal/bin/llama-server "$METAL_DIR/" 2>/dev/null || log_warning "Metal server 未找到"
    cp build-metal/bin/llama-cli "$METAL_DIR/" 2>/dev/null || true
    cp build-metal/bin/llama-bench "$METAL_DIR/" 2>/dev/null || true
    cp build-metal/libggml.dylib "$METAL_DIR/" 2>/dev/null || true
    cp build-metal/libggml-base.dylib "$METAL_DIR/" 2>/dev/null || true
    cp build-metal/libggml-cpu.dylib "$METAL_DIR/" 2>/dev/null || true
    cp build-metal/libggml-metal.dylib "$METAL_DIR/" 2>/dev/null || true
    cp build-metal/libllama.dylib "$METAL_DIR/" 2>/dev/null || true
    
    cp build-cpu/bin/llama-server "$CPU_DIR/" 2>/dev/null || log_warning "CPU server 未找到"
    cp build-cpu/bin/llama-cli "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/bin/llama-bench "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libggml.dylib "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libggml-base.dylib "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libggml-cpu.dylib "$CPU_DIR/" 2>/dev/null || true
    cp build-cpu/libllama.dylib "$CPU_DIR/" 2>/dev/null || true
    
    log_success "macOS 二进制复制完成"
    
    echo ""
    echo "已复制的文件:"
    echo "  Metal: $(ls $METAL_DIR | wc -l) 个文件"
    echo "  CPU:   $(ls $CPU_DIR | wc -l) 个文件"
}

copy_binaries() {
    case "$PLATFORM" in
        linux)   copy_linux_binaries ;;
        darwin)  copy_macos_binaries ;;
        windows) log_warning "Windows 二进制请在 Windows 上复制" ;;
    esac
}

package_cuda_v13() {
    log_step "打包 CUDA v13..."
    
    if [ ! -d "/usr/lib/ollama/cuda_v13" ]; then
        log_warning "Ollama CUDA v13 库未找到，跳过"
        log_info "安装 Ollama: curl -fsSL https://ollama.ai/install.sh | sh"
        return 0
    fi
    
    bash "$SCRIPT_DIR/package_cuda_v13.sh"
    
    log_success "CUDA v13 包已创建"
}

package_rocm() {
    log_step "打包 ROCm..."
    
    ROCM_LIB="$PROJECT_DIR/moxing/bin/linux-x64-rocm/libggml-hip.so"
    if [ ! -f "$ROCM_LIB" ]; then
        log_warning "ROCm 库未找到，跳过"
        return 0
    fi
    
    bash "$SCRIPT_DIR/package_rocm.sh"
    
    log_success "ROCm 包已创建"
}

package_binaries() {
    log_step "打包二进制库..."
    
    mkdir -p "$PROJECT_DIR/dist"
    
    case "$PLATFORM" in
        linux)
            package_cuda_v13
            package_rocm
            ;;
        darwin)
            log_info "macOS 不需要额外二进制包"
            ;;
        windows)
            log_warning "Windows 二进制包请在 Windows 上创建"
            ;;
    esac
}

build_wheel() {
    log_step "构建 Python Wheel..."
    
    cd "$PROJECT_DIR"
    
    MANIFEST_FILE="$PROJECT_DIR/MANIFEST.in"
    
    if grep -q "recursive-include moxing/bin \*" "$MANIFEST_FILE"; then
        log_warning "修复 MANIFEST.in..."
        sed -i 's/recursive-include moxing.bin \*/recursive-exclude moxing.bin \*/' "$MANIFEST_FILE"
    fi
    
    rm -rf build/ dist/*.whl *.egg-info/
    
    python3 -m build --wheel
    
    WHEEL_FILE=$(ls dist/*.whl | head -1)
    WHEEL_SIZE=$(du -h "$WHEEL_FILE" | cut -f1)
    
    log_success "Wheel 构建完成: $WHEEL_SIZE"
    
    if [[ "$WHEEL_SIZE" == *"G"* ]] || [[ "$WHEEL_SIZE" == *"M"* && "${WHEEL_SIZE%M}" -gt 10 ]]; then
        log_error "Wheel 异常大小: $WHEEL_SIZE"
        log_warning "检查 MANIFEST.in 是否正确排除二进制"
        return 1
    fi
}

upload_release() {
    log_step "上传到 GitHub Release..."
    
    if ! command -v gh &> /dev/null; then
        log_error "gh CLI 未安装"
        log_info "安装: https://cli.github.com/"
        return 1
    fi
    
    if ! gh auth status &> /dev/null; then
        log_error "未登录 GitHub"
        log_info "运行: gh auth login"
        return 1
    fi
    
    bash "$SCRIPT_DIR/upload_to_github.sh"
    
    log_success "上传完成"
}

show_summary() {
    VERSION=$(python3 -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from moxing import __version__; print(__version__)")
    
    echo ""
    echo -e "${CYAN}==============================================${NC}"
    echo -e "${GREEN}  构建完成！${NC}"
    echo -e "${CYAN}==============================================${NC}"
    echo ""
    echo "版本: v$VERSION"
    echo ""
    echo "已生成文件:"
    echo ""
    ls -lh "$PROJECT_DIR/dist/" 2>/dev/null || true
    echo ""
    echo "下一步:"
    if [[ $UPLOAD -eq 0 ]]; then
        echo "  1. 检查 dist/ 目录中的文件"
        echo "  2. 运行: ./scripts/upload_to_github.sh"
        echo "  或"
        echo "  3. 运行: $0 --upload"
    else
        VERSION=$(python3 -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from moxing import __version__; print(__version__)")
        echo "  查看发布: https://github.com/cycleuser/MoXing/releases/tag/v$VERSION"
    fi
}

main() {
    echo ""
    echo -e "${CYAN}==============================================${NC}"
    echo -e "${GREEN}  MoXing 二进制构建脚本${NC}"
    echo -e "${CYAN}==============================================${NC}"
    echo ""
    
    BUILD_LLAMA=1
    COPY_BIN=1
    PACKAGE=1
    WHEEL=1
    UPLOAD=0
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-build)   BUILD_LLAMA=0 ;;
            --no-copy)    COPY_BIN=0 ;;
            --no-package) PACKAGE=0 ;;
            --no-wheel)   WHEEL=0 ;;
            --upload)     UPLOAD=1 ;;
            --all)        UPLOAD=1 ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --no-build    跳过 llama.cpp 构建"
                echo "  --no-copy     跳过二进制复制"
                echo "  --no-package  跳过二进制打包"
                echo "  --no-wheel    跳过 Wheel 构建"
                echo "  --upload      构建并上传到 GitHub"
                echo "  --all         完整构建并上传"
                echo "  --help        显示此帮助"
                echo ""
                echo "示例:"
                echo "  $0                    # 构建当前平台"
                echo "  $0 --upload           # 构建并上传"
                echo "  $0 --no-build --upload # 使用现有二进制上传"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
        shift
    done
    
    detect_platform
    check_dependencies
    
    if [[ $BUILD_LLAMA -eq 1 ]]; then
        prepare_llama_cpp
        
        case "$PLATFORM" in
            linux)
                build_linux
                ;;
            darwin)
                build_macos
                ;;
            windows)
                generate_windows_script
                ;;
            *)
                log_error "未知平台: $PLATFORM"
                exit 1
                ;;
        esac
    fi
    
    if [[ $COPY_BIN -eq 1 ]]; then
        copy_binaries
    fi
    
    if [[ $PACKAGE -eq 1 ]]; then
        package_binaries
    fi
    
    if [[ $WHEEL -eq 1 ]]; then
        build_wheel
    fi
    
    if [[ $UPLOAD -eq 1 ]]; then
        upload_release
    fi
    
    show_summary
}

main "$@"