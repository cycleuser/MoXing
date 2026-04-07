#!/bin/bash
#
# 重新构建 ROCm Runner，使用系统 ROCm
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR_DIR="$PROJECT_DIR/ollama/llama/vendor"
BUILD_DIR="$PROJECT_DIR/build"
MOXING_BIN="$PROJECT_DIR/moxing/bin"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_step()    { echo -e "${CYAN}==>${NC} $1"; }

# 检查 ROCm
check_rocm() {
    log_step "检查 ROCm..."
    
    if ! command -v hipcc &> /dev/null; then
        log_error "未找到 hipcc，请安装 ROCm"
        exit 1
    fi
    
    HIP_VERSION=$(hipcc --version 2>/dev/null | grep -o "[0-9]\+\.[0-9]\+" | head -1 || echo "unknown")
    log_info "HIP 版本: $HIP_VERSION"
    
    # 查找 ROCm 路径
    ROCM_PATH=""
    for path in "/opt/rocm" "/opt/rocm/core-7.12" "/opt/rocm-7.1.1"; do
        if [ -d "$path" ]; then
            ROCM_PATH="$path"
            break
        fi
    done
    
    if [ -z "$ROCM_PATH" ]; then
        log_error "未找到 ROCm 安装目录"
        exit 1
    fi
    
    log_info "ROCm 路径: $ROCM_PATH"
    
    # 导出环境变量
    export ROCM_PATH
    export PATH="$ROCM_PATH/bin:$PATH"
    if [ -d "$ROCM_PATH/llvm/bin" ]; then
        export PATH="$ROCM_PATH/llvm/bin:$PATH"
    fi
    
    # 查找 GPU 目标
    if [ -f "$ROCM_PATH/bin/rocminfo" ]; then
        log_info "GPU 信息:"
        rocminfo | grep "Name:" | head -5 || true
    fi
}

# 配置 CMake
configure_cmake() {
    log_step "配置 CMake..."
    
    cd "$VENDOR_DIR"
    
    local build_dir="$VENDOR_DIR/build-rocm-moxing"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    
    # 设置编译器
    if [ -f "$ROCM_PATH/llvm/bin/clang++" ]; then
        export CXX="$ROCM_PATH/llvm/bin/clang++"
        export CC="$ROCM_PATH/llvm/bin/clang"
    fi
    
    # 检测 GPU 架构
    local gpu_targets="gfx1100"
    if command -v rocminfo &> /dev/null; then
        local detected=$(rocminfo | grep " gfx" | head -1 | awk '{print $2}')
        if [ -n "$detected" ]; then
            gpu_targets="$detected"
            log_info "检测到的 GPU: $gpu_targets"
        fi
    fi
    
    # 配置
    cmake -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_C_COMPILER="$CC" \
        -DGGML_HIP=ON \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_HIP_ARCHITECTURES="$gpu_targets" \
        -DGPU_TARGETS="$gpu_targets" \
        -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
        2>&1 | tail -30
    
    log_success "CMake 配置完成"
}

# 编译
build() {
    log_step "编译 ROCm Runner..."
    
    local build_dir="$VENDOR_DIR/build-rocm-moxing"
    
    cd "$VENDOR_DIR"
    
    log_info "这可能需要几分钟..."
    cmake --build "$build_dir" -j$(nproc) 2>&1 | tail -30
    
    if [ ! -f "$build_dir/bin/llama-server" ]; then
        log_error "编译失败"
        exit 1
    fi
    
    log_success "编译完成"
}

# 安装
install_runner() {
    log_step "安装 Runner..."
    
    local build_dir="$VENDOR_DIR/build-rocm-moxing"
    local install_dir="$MOXING_BIN/ollama-linux-x64-rocm"
    
    # 备份旧的
    if [ -d "$install_dir" ]; then
        mv "$install_dir" "$install_dir.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    mkdir -p "$install_dir"
    
    # 复制二进制
    cp "$build_dir/bin/llama-server" "$install_dir/ollama-runner-rocm"
    cp "$build_dir/bin/llama-cli" "$install_dir/ollama-cli-rocm" 2>/dev/null || true
    cp "$build_dir/bin/llama-bench" "$install_dir/ollama-bench-rocm" 2>/dev/null || true
    
    # 复制库
    cp "$build_dir/bin"/libggml*.so* "$install_dir/"
    cp "$build_dir/bin"/libllama*.so* "$install_dir/"
    cp "$build_dir/bin"/libmtmd*.so* "$install_dir/"
    
    # 创建符号链接
    ln -sf "ollama-runner-rocm" "$install_dir/ollama-runner"
    
    # 设置权限
    chmod +x "$install_dir"/ollama-runner*
    
    log_success "安装到: $install_dir"
    
    # 同时复制到 build 目录
    local build_out="$BUILD_DIR/ollama-runner-rocm"
    mkdir -p "$build_out"
    cp -r "$install_dir"/* "$build_out/"
    log_success "复制到: $build_out"
}

# 测试
test_runner() {
    log_step "测试 Runner..."
    
    local runner="$MOXING_BIN/ollama-linux-x64-rocm/ollama-runner-rocm"
    
    log_info "检查库依赖..."
    ldd "$runner" | grep "not found" || log_success "所有库已找到"
    
    log_info "测试版本..."
    LD_LIBRARY_PATH="$MOXING_BIN/ollama-linux-x64-rocm" "$runner" --version 2>/dev/null | head -1 || log_warning "版本检查失败"
}

# 创建环境脚本
create_env_script() {
    log_step "创建环境脚本..."
    
    local install_dir="$MOXING_BIN/ollama-linux-x64-rocm"
    
    cat > "$install_dir/setup_env.sh" << EOF
#!/bin/bash
# ROCm 环境设置

export ROCM_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:\$PATH"

# 库路径
export LD_LIBRARY_PATH="$install_dir:$ROCM_PATH/lib:\$LD_LIBRARY_PATH"

# 如果使用 conda 的 ROCm
if [ -d "\$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
fi

echo "ROCm 环境已设置"
echo "ROCM_PATH: \$ROCM_PATH"
echo "LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
EOF
    
    chmod +x "$install_dir/setup_env.sh"
    log_success "环境脚本已创建"
}

# 主流程
main() {
    echo "=========================================="
    echo "  重新构建 ROCm Runner"
    echo "=========================================="
    echo ""
    
    check_rocm
    configure_cmake
    build
    install_runner
    create_env_script
    test_runner
    
    echo ""
    echo "=========================================="
    log_success "构建完成！"
    echo "=========================================="
    echo ""
    echo "使用方法:"
    echo "  source $MOXING_BIN/ollama-linux-x64-rocm/setup_env.sh"
    echo "  moxing ollama serve gemma4:31b -b rocm -d gpu1"
    echo ""
}

main "$@"
