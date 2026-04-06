#!/bin/bash
#
# 完整的工具检测脚本 - 使用 which 和 whereis
#

# Don't exit on error - we want to check all tools
# set -e

echo "=============================================="
echo "  完整工具检测"
echo "=============================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_tool() {
    local name=$1
    local var_name=$2
    
    # 用 which 检查
    local which_path=$(which $name 2>/dev/null || echo "")
    
    # 用 whereis 检查
    local whereis_path=$(whereis $name 2>/dev/null | awk '{print $2}')
    
    if [ -n "$which_path" ]; then
        echo -e "${GREEN}✓${NC} $name: $which_path"
        if [ -n "$var_name" ]; then
            echo "  $var_name=$which_path"
        fi
        return 0
    elif [ -n "$whereis_path" ]; then
        echo -e "${GREEN}✓${NC} $name: $whereis_path (whereis)"
        if [ -n "$var_name" ]; then
            echo "  $var_name=$whereis_path"
        fi
        return 0
    else
        echo -e "${RED}✗${NC} $name: 未找到"
        return 1
    fi
}

echo "========================================"
echo "1. 编译工具"
echo "========================================"
check_tool cmake CMAKE
check_tool make MAKE
check_tool gcc GCC
check_tool g++ GPP
check_tool clang CLANG
check_tool clang++ CLANGPP

echo ""
echo "========================================"
echo "2. CUDA 工具"
echo "========================================"
check_tool nvcc NVCC
check_tool cuda-gdb CUDA_GDB

echo ""
if [ -d "/usr/local/cuda" ]; then
    CUDA_PATH=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
    echo -e "${GREEN}✓${NC} CUDA 安装路径: $CUDA_PATH"
    echo "  CUDA_PATH=$CUDA_PATH"
    
    if [ -f "$CUDA_PATH/bin/nvcc" ]; then
        echo -e "${GREEN}✓${NC} nvcc 版本:"
        $CUDA_PATH/bin/nvcc --version | grep release
    fi
else
    echo -e "${RED}✗${NC} CUDA 未安装"
fi

echo ""
echo "========================================"
echo "3. ROCm/HIP 工具"
echo "========================================"
check_tool hipcc HIPCC
check_tool hipconfig HIPCONFIG
check_tool clang-offload-bundler CLANG_OFFLOAD

echo ""
if [ -d "/opt/rocm" ]; then
    ROCM_PATH=$(ls -d /opt/rocm* 2>/dev/null | grep -v "^/opt/rocm$" | sort -V | tail -1)
    if [ -z "$ROCM_PATH" ]; then
        ROCM_PATH="/opt/rocm"
    fi
    echo -e "${GREEN}✓${NC} ROCm 安装路径: $ROCM_PATH"
    echo "  ROCM_PATH=$ROCM_PATH"
    
    if [ -f "$ROCM_PATH/bin/hipcc" ]; then
        echo -e "${GREEN}✓${NC} hipcc 版本:"
        $ROCM_PATH/bin/hipcc --version 2>&1 | grep -E "HIP version|clang version" | head -2
    fi
    
    if [ -f "$ROCM_PATH/bin/rocminfo" ]; then
        echo -e "${GREEN}✓${NC} ROCm 目标架构:"
        $ROCM_PATH/bin/rocminfo 2>/dev/null | grep -E "Name:.*gfx" | head -5
    fi
else
    echo -e "${RED}✗${NC} ROCm 未安装"
fi

echo ""
echo "========================================"
echo "4. Vulkan 工具"
echo "========================================"
check_tool vulkaninfo VULKANINFO
check_tool glslc GLSLC
check_tool glslangValidator GLSLANG

echo ""
echo "Vulkan 设备:"
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion" | head -10
fi

echo ""
echo "========================================"
echo "5. Python 工具"
echo "========================================"
check_tool python3 PYTHON
check_tool pip3 PIP

if command -v python3 &> /dev/null; then
    echo ""
    echo "Python 版本:"
    python3 --version
    echo "Python 路径:"
    which python3
fi

echo ""
echo "========================================"
echo "6. 构建依赖库"
echo "========================================"

# 检查 CUDA 库
if [ -d "/usr/local/cuda" ]; then
    CUDA_LIB=$(ls -d /usr/local/cuda*/lib64 2>/dev/null | sort -V | tail -1)
    if [ -d "$CUDA_LIB" ]; then
        echo -e "${GREEN}✓${NC} CUDA 库: $CUDA_LIB"
        echo "  库文件数: $(ls $CUDA_LIB/*.so 2>/dev/null | wc -l)"
    fi
fi

# 检查 ROCm 库
if [ -d "/opt/rocm" ]; then
    ROCM_LIB=$(ls -d /opt/rocm*/lib 2>/dev/null | sort -V | tail -1)
    if [ -d "$ROCM_LIB" ]; then
        echo -e "${GREEN}✓${NC} ROCm 库: $ROCM_LIB"
        echo "  库文件数: $(ls $ROCM_LIB/*.so 2>/dev/null | wc -l)"
    fi
fi

# 检查 Vulkan 库
VULKAN_LIB=$(ldconfig -p 2>/dev/null | grep libvulkan.so | awk '{print $NF}' | head -1)
if [ -n "$VULKAN_LIB" ]; then
    echo -e "${GREEN}✓${NC} Vulkan 库: $VULKAN_LIB"
else
    echo -e "${RED}✗${NC} Vulkan 库未找到"
fi

echo ""
echo "========================================"
echo "7. 环境变量"
echo "========================================"
echo "PATH:"
echo "  $PATH" | tr ':' '\n' | head -10

echo ""
echo "LD_LIBRARY_PATH:"
if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "  $LD_LIBRARY_PATH" | tr ':' '\n'
else
    echo "  (未设置)"
fi

echo ""
echo "========================================"
echo "8. 推荐的环境变量设置"
echo "========================================"

CUDA_PATH=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
ROCM_PATH=$(ls -d /opt/rocm* 2>/dev/null | grep -v "^/opt/rocm$" | sort -V | tail -1)
[ -z "$ROCM_PATH" ] && ROCM_PATH="/opt/rocm"

echo "export PATH=$CUDA_PATH/bin:\$PATH"
echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "export PATH=$ROCM_PATH/bin:\$PATH"
echo "export LD_LIBRARY_PATH=$ROCM_PATH/lib:\$LD_LIBRARY_PATH"

echo ""
echo "=============================================="
echo "  检测完成"
echo "=============================================="