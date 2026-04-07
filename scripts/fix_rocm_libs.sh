#!/bin/bash
#
# 修复 ROCm 库依赖
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

# 找到 ROCm 库
find_rocm_libs() {
    log_info "查找 ROCm 库..."
    
    ROCM_PATHS=(
        "/opt/rocm"
        "/opt/rocm/core-7.12"
        "/opt/rocm-7.1.1"
    )
    
    ROCM_LIB_PATH=""
    for path in "${ROCM_PATHS[@]}"; do
        if [ -d "$path/lib" ]; then
            ROCM_LIB_PATH="$path/lib"
            log_success "找到 ROCm 库: $ROCM_LIB_PATH"
            break
        fi
    done
    
    if [ -z "$ROCM_LIB_PATH" ]; then
        log_error "未找到 ROCm 库"
        exit 1
    fi
}

# 修复 runner 的库依赖
fix_runner_libs() {
    local runner_dir="$1"
    
    log_info "修复 runner 库依赖: $runner_dir"
    
    for binary in "$runner_dir"/ollama-runner*; do
        if [ -f "$binary" ] && [ ! -L "$binary" ]; then
            log_info "修复: $(basename "$binary")"
            
            # 使用 patchelf 修改 rpath（如果可用）
            if command -v patchelf &> /dev/null; then
                patchelf --set-rpath "$ROCM_LIB_PATH:$(dirname "$binary")" "$binary" 2>/dev/null || true
            fi
            
            # 检查缺失的库
            local missing_libs=$(ldd "$binary" 2>/dev/null | grep "not found" | awk '{print $1}')
            
            if [ -n "$missing_libs" ]; then
                log_warning "缺失的库:"
                echo "$missing_libs"
                
                # 尝试在 ROCm 路径中找到并创建符号链接
                for lib in $missing_libs; do
                    local lib_name=$(echo "$lib" | sed 's/://')
                    local found_lib=$(find "$ROCM_LIB_PATH" -name "$lib_name*" -type f 2>/dev/null | head -1)
                    
                    if [ -n "$found_lib" ]; then
                        log_info "创建符号链接: $lib_name -> $found_lib"
                        ln -sf "$found_lib" "$runner_dir/$lib_name"
                    fi
                done
            fi
        fi
    done
    
    log_success "修复完成"
}

# 设置环境变量脚本
create_env_script() {
    local runner_dir="$1"
    
    log_info "创建环境设置脚本..."
    
    cat > "$runner_dir/setup_env.sh" << EOF
#!/bin/bash
# ROCm 环境设置

export ROCM_PATH="$ROCM_LIB_PATH"
export LD_LIBRARY_PATH="$ROCM_LIB_PATH:\$LD_LIBRARY_PATH"

# 查找设备
if command -v rocminfo &> /dev/null; then
    echo "ROCm 设备:"
    rocminfo | grep "Name:" | head -10
fi

echo "环境已设置"
echo "LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
EOF
    
    chmod +x "$runner_dir/setup_env.sh"
    log_success "环境脚本已创建: $runner_dir/setup_env.sh"
}

# 主流程
main() {
    echo "=========================================="
    echo "  修复 ROCm 库依赖"
    echo "=========================================="
    echo ""
    
    find_rocm_libs
    
    # 修复 moxing/bin 中的 ROCm runner
    MOXING_ROCM="/home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm"
    
    if [ -d "$MOXING_ROCM" ]; then
        fix_runner_libs "$MOXING_ROCM"
        create_env_script "$MOXING_ROCM"
    else
        log_warning "未找到 ROCm runner 目录"
    fi
    
    # 修复 build 目录
    BUILD_ROCM="/home/fred/Documents/GitHub/cycleuser/MoXing/build/ollama-runner-rocm"
    
    if [ -d "$BUILD_ROCM" ]; then
        fix_runner_libs "$BUILD_ROCM"
    fi
    
    echo ""
    echo "=========================================="
    log_success "修复完成！"
    echo "=========================================="
    echo ""
    echo "使用方法:"
    echo "  source $MOXING_ROCM/setup_env.sh"
    echo "  moxing ollama serve gemma4:31b -b rocm -d gpu1"
    echo ""
}

main "$@"
