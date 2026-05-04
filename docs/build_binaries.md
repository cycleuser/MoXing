# MoXing 二进制构建指南

本文档详细说明如何在 Linux、macOS、Windows 上构建和发布 MoXing 二进制包。

## 构建流程总览

```
┌─────────────────────────────────────────────────────────────┐
│                      MoXing 构建流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 准备 llama.cpp 源码                                      │
│     └─> 克隆 ollama/llama.cpp (含 Ollama 补丁)               │
│                                                             │
│  2. 按平台构建后端                                            │
│     ├─> Linux: CUDA / ROCm / Vulkan / CPU                   │
│     ├─> macOS: Metal / CPU                                   │
│     └─> Windows: CUDA / Vulkan / CPU                         │
│                                                             │
│  3. 打包二进制库                                              │
│     ├─> CUDA v13 (~750MB 压缩)                               │
│     ├─> ROCm (~12MB 压缩)                                    │
│     └─> Vulkan/CPU (由 Ollama 提供)                          │
│                                                             │
│  4. 构建 Python Wheel                                        │
│     └─> 仅含 Python 代码 (~126KB)                            │
│                                                             │
│  5. 发布到 GitHub Release                                    │
│     ├─> moxing-X.X.X-py3-none-any.whl                        │
│     ├─> moxing-cuda-v13-X.X.X.tar.gz                         │
│     └─> moxing-rocm-X.X.X.tar.gz                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 系统要求

### Linux (推荐用于 CUDA 和 ROCm)

| 要求 | CUDA | ROCm | Vulkan |
|------|------|------|--------|
| GPU | NVIDIA | AMD | 任意 |
| 驱动 | CUDA 12+ | ROCm 6.0+ | Vulkan 1.3+ |
| 工具链 | GCC 11+ | HIPCC | GCC 11+ |
| 包 | cuda-toolkit | rocm-dev | vulkan-sdk |

### macOS (仅 Metal/CPU)

| 要求 | Metal | CPU |
|------|-------|-----|
| macOS | 12.0+ | 任意 |
| Xcode | 14.0+ | 不需要 |
| 工具链 | Clang | Clang |

### Windows (CUDA/Vulkan/CPU)

| 要求 | CUDA | Vulkan | CPU |
|------|------|--------|-----|
| Visual Studio | 2022 | 2022 | 2022 |
| CUDA Toolkit | 12.0+ | 不需要 | 不需要 |
| Vulkan SDK | 不需要 | 1.3+ | 不需要 |

## 一键构建脚本

### 跨平台主脚本

```bash
# scripts/build_all_binaries.sh
#!/bin/bash
# MoXing 全平台二进制构建脚本
# 用法: ./scripts/build_all_binaries.sh [选项]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

# 检测平台
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux"; ARCH=$(uname -m) ;;
        Darwin*)    PLATFORM="darwin"; ARCH=$(uname -m) ;;
        CYGWIN*|MINGW*|MSYS*)  PLATFORM="windows"; ARCH="x64" ;;
        *)          PLATFORM="unknown" ;;
    esac
    log_info "检测到平台: $PLATFORM-$ARCH"
}

# 准备 llama.cpp 源码
prepare_llama_cpp() {
    LLAMA_DIR="$PROJECT_DIR/build/llama.cpp"
    
    if [ -d "$LLAMA_DIR" ]; then
        log_info "llama.cpp 已存在，更新..."
        cd "$LLAMA_DIR"
        git pull
    else
        log_info "克隆 llama.cpp..."
        mkdir -p "$PROJECT_DIR/build"
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi
    
    cd "$LLAMA_DIR"
    log_success "llama.cpp 准备完成: $(git describe --tags)"
}

# Linux 构建
build_linux() {
    log_info "开始 Linux 构建..."
    
    cd "$LLAMA_DIR"
    
    # CUDA
    log_info "构建 CUDA 后端..."
    rm -rf build-cuda && mkdir build-cuda && cd build-cuda
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DLLAMA_CUBLAS=ON
    cmake --build . --config Release -j$(nproc)
    cd "$LLAMA_DIR"
    
    # ROCm
    log_info "构建 ROCm 后端..."
    rm -rf build-rocm && mkdir build-rocm && cd build-rocm
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIPBLAS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    cmake --build . --config Release -j$(nproc)
    cd "$LLAMA_DIR"
    
    # Vulkan
    log_info "构建 Vulkan 后端..."
    rm -rf build-vulkan && mkdir build-vulkan && cd build-vulkan
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    cmake --build . --config Release -j$(nproc)
    cd "$LLAMA_DIR"
    
    # CPU
    log_info "构建 CPU 后端..."
    rm -rf build-cpu && mkdir build-cpu && cd build-cpu
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    cmake --build . --config Release -j$(nproc)
    
    log_success "Linux 构建完成"
}

# macOS 构建
build_macos() {
    log_info "开始 macOS 构建..."
    
    cd "$LLAMA_DIR"
    
    # Metal
    log_info "构建 Metal 后端..."
    rm -rf build-metal && mkdir build-metal && cd build-metal
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    cd "$LLAMA_DIR"
    
    # CPU
    log_info "构建 CPU 后端..."
    rm -rf build-cpu && mkdir build-cpu && cd build-cpu
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)
    
    log_success "macOS 构建完成"
}

# Windows 构建 (PowerShell)
build_windows() {
    log_warning "Windows 构建请在 Windows 上运行 PowerShell 脚本"
    log_info "生成 PowerShell 脚本..."
    
    cat > "$PROJECT_DIR/scripts/build_windows.ps1" << 'PSEOF'
# Windows 构建脚本
param(
    [string]$Backend = "all"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$LlamaDir = Join-Path $ProjectDir "build\llama.cpp"

# 检查 Visual Studio
if (-not (Get-Command "cmake" -ErrorAction SilentlyContinue)) {
    Write-Error "CMake 未安装"
    exit 1
}

# CUDA
if ($Backend -eq "cuda" -or $Backend -eq "all") {
    Write-Host "构建 CUDA 后端..."
    Set-Location $LlamaDir
    New-Item -ItemType Directory -Force -Path "build-cuda" | Out-Null
    Set-Location "build-cuda"
    
    cmake .. `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_COMMON=ON `
        -G "Visual Studio 17 2022"
    
    cmake --build . --config Release --parallel
}

# Vulkan
if ($Backend -eq "vulkan" -or $Backend -eq "all") {
    Write-Host "构建 Vulkan 后端..."
    Set-Location $LlamaDir
    New-Item -ItemType Directory -Force -Path "build-vulkan" | Out-Null
    Set-Location "build-vulkan"
    
    cmake .. `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_VULKAN=ON `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_COMMON=ON `
        -G "Visual Studio 17 2022"
    
    cmake --build . --config Release --parallel
}

# CPU
if ($Backend -eq "cpu" -or $Backend -eq "all") {
    Write-Host "构建 CPU 后端..."
    Set-Location $LlamaDir
    New-Item -ItemType Directory -Force -Path "build-cpu" | Out-Null
    Set-Location "build-cpu"
    
    cmake .. `
        -DCMAKE_BUILD_TYPE=Release `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_COMMON=ON `
        -G "Visual Studio 17 2022"
    
    cmake --build . --config Release --parallel
}

Write-Host "Windows 构建完成"
PSEOF
    
    log_success "PowerShell 脚本已生成: scripts/build_windows.ps1"
}

# 复制二进制到 moxing/bin
copy_binaries() {
    log_info "复制二进制文件到 moxing/bin..."
    
    BIN_DIR="$PROJECT_DIR/moxing/bin"
    mkdir -p "$BIN_DIR"
    
    case "$PLATFORM" in
        linux)
            # CUDA
            mkdir -p "$BIN_DIR/linux-x64-cuda"
            cp "$LLAMA_DIR/build-cuda/bin/llama-server" "$BIN_DIR/linux-x64-cuda/"
            cp "$LLAMA_DIR/build-cuda/bin/llama-cli" "$BIN_DIR/linux-x64-cuda/"
            cp "$LLAMA_DIR/build-cuda/bin/llama-bench" "$BIN_DIR/linux-x64-cuda/"
            cp "$LLAMA_DIR/build-cuda/lib/libggml-cuda.so*" "$BIN_DIR/linux-x64-cuda/"
            
            # ROCm
            mkdir -p "$BIN_DIR/linux-x64-rocm"
            cp "$LLAMA_DIR/build-rocm/bin/llama-server" "$BIN_DIR/linux-x64-rocm/"
            cp "$LLAMA_DIR/build-rocm/bin/llama-cli" "$BIN_DIR/linux-x64-rocm/"
            cp "$LLAMA_DIR/build-rocm/bin/llama-bench" "$BIN_DIR/linux-x64-rocm/"
            cp "$LLAMA_DIR/build-rocm/lib/libggml-hip.so*" "$BIN_DIR/linux-x64-rocm/"
            
            # Vulkan
            mkdir -p "$BIN_DIR/linux-x64-vulkan"
            cp "$LLAMA_DIR/build-vulkan/bin/llama-server" "$BIN_DIR/linux-x64-vulkan/"
            cp "$LLAMA_DIR/build-vulkan/bin/llama-cli" "$BIN_DIR/linux-x64-vulkan/"
            cp "$LLAMA_DIR/build-vulkan/bin/llama-bench" "$BIN_DIR/linux-x64-vulkan/"
            cp "$LLAMA_DIR/build-vulkan/lib/libggml-vulkan.so*" "$BIN_DIR/linux-x64-vulkan/"
            
            # CPU
            mkdir -p "$BIN_DIR/linux-x64-cpu"
            cp "$LLAMA_DIR/build-cpu/bin/llama-server" "$BIN_DIR/linux-x64-cpu/"
            cp "$LLAMA_DIR/build-cpu/bin/llama-cli" "$BIN_DIR/linux-x64-cpu/"
            cp "$LLAMA_DIR/build-cpu/bin/llama-bench" "$BIN_DIR/linux-x64-cpu/"
            ;;
        darwin)
            # Metal
            mkdir -p "$BIN_DIR/darwin-arm64-metal"
            cp "$LLAMA_DIR/build-metal/bin/llama-server" "$BIN_DIR/darwin-arm64-metal/"
            cp "$LLAMA_DIR/build-metal/bin/llama-cli" "$BIN_DIR/darwin-arm64-metal/"
            cp "$LLAMA_DIR/build-metal/bin/llama-bench" "$BIN_DIR/darwin-arm64-metal/"
            cp "$LLAMA_DIR/build-metal/lib/libggml-metal.dylib" "$BIN_DIR/darwin-arm64-metal/"
            
            # CPU
            mkdir -p "$BIN_DIR/darwin-arm64-cpu"
            cp "$LLAMA_DIR/build-cpu/bin/llama-server" "$BIN_DIR/darwin-arm64-cpu/"
            cp "$LLAMA_DIR/build-cpu/bin/llama-cli" "$BIN_DIR/darwin-arm64-cpu/"
            cp "$LLAMA_DIR/build-cpu/bin/llama-bench" "$BIN_DIR/darwin-arm64-cpu/"
            ;;
        windows)
            log_warning "Windows 二进制请在 Windows 上复制"
            ;;
    esac
    
    log_success "二进制复制完成"
}

# 打包二进制库
package_binaries() {
    log_info "打包二进制库..."
    
    DIST_DIR="$PROJECT_DIR/dist"
    mkdir -p "$DIST_DIR"
    
    VERSION=$(python3 -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from moxing import __version__; print(__version__)")
    
    case "$PLATFORM" in
        linux)
            # CUDA v13 包
            log_info "创建 CUDA v13 包..."
            bash "$PROJECT_DIR/scripts/package_cuda_v13.sh"
            
            # ROCm 包
            log_info "创建 ROCm 包..."
            bash "$PROJECT_DIR/scripts/package_rocm.sh"
            ;;
        darwin)
            log_warning "macOS 不需要额外二进制包 (Metal 由系统提供)"
            ;;
        windows)
            log_warning "Windows CUDA 包请在 Windows 上创建"
            ;;
    esac
    
    log_success "二进制打包完成"
}

# 构建 Wheel
build_wheel() {
    log_info "构建 Python Wheel..."
    
    cd "$PROJECT_DIR"
    
    # 确保 MANIFEST.in 正确
    if grep -q "recursive-include moxing/bin *" MANIFEST.in; then
        log_warning "MANIFEST.in 包含二进制，正在修复..."
        sed -i 's/recursive-include moxing.bin \*/recursive-exclude moxing.bin \*/' MANIFEST.in
    fi
    
    # 清理旧构建
    rm -rf build/ dist/*.whl moxing.egg-info/
    
    # 构建
    python3 -m build --wheel
    
    WHEEL_SIZE=$(du -h dist/*.whl | cut -f1)
    log_success "Wheel 构建完成: $WHEEL_SIZE"
}

# 发布到 GitHub
upload_release() {
    log_info "发布到 GitHub Release..."
    
    # 检查 gh CLI
    if ! command -v gh &> /dev/null; then
        log_error "gh CLI 未安装"
        log_info "安装: https://cli.github.com/"
        return 1
    fi
    
    # 检查认证
    if ! gh auth status &> /dev/null; then
        log_error "未登录 GitHub"
        log_info "运行: gh auth login"
        return 1
    fi
    
    bash "$PROJECT_DIR/scripts/upload_to_github.sh"
    
    log_success "发布完成"
}

# 主函数
main() {
    echo ""
    echo "=============================================="
    echo "  MoXing 二进制构建脚本"
    echo "=============================================="
    echo ""
    
    # 参数解析
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
            --help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --no-build    跳过 llama.cpp 构建"
                echo "  --no-copy     跳过二进制复制"
                echo "  --no-package  跳过二进制打包"
                echo "  --no-wheel    跳过 Wheel 构建"
                echo "  --upload      上传到 GitHub Release"
                echo "  --all         构建并上传"
                echo "  --help        显示帮助"
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
    
    if [[ $BUILD_LLAMA -eq 1 ]]; then
        prepare_llama_cpp
        
        case "$PLATFORM" in
            linux)   build_linux ;;
            darwin)  build_macos ;;
            windows) build_windows ;;
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
    
    echo ""
    echo "=============================================="
    log_success "构建完成！"
    echo "=============================================="
    echo ""
    
    if [[ $UPLOAD -eq 0 ]]; then
        echo "下一步："
        echo "  1. 检查 dist/ 目录中的文件"
        echo "  2. 运行 ./scripts/upload_to_github.sh 上传"
        echo "  或"
        echo "  3. 运行 $0 --upload 一键上传"
    fi
}

main "$@"
```

## 分平台详细步骤

### Linux 构建 (CUDA + ROCm)

```bash
# 1. 安装依赖
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip

# CUDA
sudo apt install -y cuda-toolkit-12-6 nvidia-cuda-dev

# ROCm (AMD GPU)
# 参考: https://rocm.docs.amd.com/
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install.deb
sudo apt install ./amdgpu-install.deb
sudo amdgpu-install --usecase=rocm

# Vulkan
sudo apt install -y vulkan-sdk libvulkan-dev

# 2. 运行构建脚本
cd MoXing
./scripts/build_all_binaries.sh --all
```

### macOS 构建 (Metal)

```bash
# 1. 安装依赖
xcode-select --install
brew install cmake python3 git

# 2. 运行构建脚本
cd MoXing
./scripts/build_all_binaries.sh

# macOS 不需要额外二进制包，Metal 库由系统提供
# 只需要上传 Wheel
./scripts/upload_to_github.sh
```

### Windows 构建 (CUDA/Vulkan)

```powershell
# 1. 安装依赖
# Visual Studio 2022: https://visualstudio.microsoft.com/
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
# Vulkan SDK: https://vulkan.lunarg.com/sdk/home

# 2. 准备源码
git clone https://github.com/cycleuser/MoXing.git
cd MoXing
git clone https://github.com/ggml-org/llama.cpp.git build\llama.cpp

# 3. 运行 PowerShell 构建脚本
.\scripts\build_windows.ps1 -Backend cuda
.\scripts\build_windows.ps1 -Backend vulkan
.\scripts\build_windows.ps1 -Backend cpu

# 4. 构建 Wheel
python -m build --wheel

# 5. 上传 (需要 GitHub CLI)
gh release create vX.X.X --title "MoXing vX.X.X" dist\*.whl
```

## Ollama 补丁版构建 (可选)

如果需要支持 Ollama 特定架构 (如 gemma4)，需要构建 Ollama 补丁版 llama.cpp:

```bash
# 克隆 Ollama llama.cpp (含 35 个补丁)
git clone https://github.com/ollama/llama.git build/ollama-llama
cd build/ollama-llama/llama.cpp

# 构建 CUDA v13 版本
mkdir build-cuda-v13 && cd build-cuda-v13
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
cmake --build . --config Release -j$(nproc)

# 复制到 moxing/bin
cp bin/ollama-runner ../../moxing/bin/linux-x64-cuda-ollama/
cp lib/libggml-cuda.so ../../moxing/bin/linux-x64-cuda-ollama/
```

## 打包说明

### CUDA v13 包 (~750MB)

```bash
# 从 Ollama 系统库复制
./scripts/package_cuda_v13.sh

# 内容:
# - libggml-cuda.so (CUDA 后端)
# - libcublas.so.12 (CUDA BLAS)
# - libcudart.so.12 (CUDA runtime)
# - libggml-base.so (GGML 基础)
# - libggml-cpu-*.so (CPU 变体)
```

### ROCm 包 (~12MB)

```bash
# 从构建产物复制
./scripts/package_rocm.sh

# 内容:
# - libggml-hip.so (ROCm 后端)
```

### Vulkan/CPU

Vulkan 和 CPU 库由 Ollama 自带，无需额外打包:
- `/usr/lib/ollama/vulkan/` (~55MB)
- `/usr/lib/ollama/libggml-cpu-*.so`

## 发布检查清单

| 步骤 | Linux | macOS | Windows |
|------|-------|-------|---------|
| 1. 构建 llama.cpp | ✓ | ✓ | ✓ |
| 2. 复制二进制 | ✓ | ✓ | ✓ |
| 3. 打包 CUDA | ✓ | - | ✓ |
| 4. 打包 ROCm | ✓ | - | - |
| 5. 构建 Wheel | ✓ | ✓ | ✓ |
| 6. Wheel 大小检查 | <200KB | <200KB | <200KB |
| 7. 上传 GitHub | ✓ | ✓ | ✓ |

## 常见问题

### Q: Wheel 为什么这么大?

检查 `MANIFEST.in` 是否正确排除二进制:

```diff
- recursive-include moxing/bin *
+ recursive-exclude moxing/bin *
```

### Q: CUDA v13 和 v12 有什么区别?

| 版本 | 大小 | 兼容性 |
|------|------|--------|
| v12 | 2.4GB | CUDA 12.0-12.4 |
| v13 | 932MB | CUDA 12.5+ |
| **节省** | **1.5GB** | 63% |

推荐使用 v13，更小更快。

### Q: ROCm 为什么需要手动安装?

Ollama 不自带 ROCm 库，需要从 llama.cpp 构建。AMD GPU 用户需要:
1. 安装 ROCm 6.0+ runtime
2. 安装 moxing-rocm 包
3. 设置 `OLLAMA_LLM_LIBRARY=rocm`

### Q: 如何测试构建?

```bash
# 检查二进制
moxing devices

# 测试后端
moxing ollama serve gemma4:e2b -b cuda -d gpu0
moxing ollama serve gemma4:e2b -b rocm -d gpu1
moxing ollama serve gemma4:e2b -b vulkan
```

## 相关文档

- [性能对比](./ollama_backend_comparison.md)
- [测试报告](./ollama_backend_test_report.md)
- [快速参考](./ollama_backend_quick_reference.md)
- [二进制包说明](./binary_packages.md)