#!/bin/bash
#
# MinGW-w64 兼容性补丁
# 修复 Windows API 类型定义缺失问题
#

LLAMA_DIR="$1"

if [ -z "$LLAMA_DIR" ]; then
    LLAMA_DIR="/home/fred/Documents/GitHub/cycleuser/MoXing/build/llama.cpp-windows"
fi

echo "应用 MinGW 兼容性补丁..."

# 补丁 1: 添加 THREAD_POWER_THROTTLING_STATE 定义
PATCH_FILE="$LLAMA_DIR/ggml/src/ggml-cpu/mingw_compat.h"

cat > "$PATCH_FILE" << 'EOF'
// MinGW-w64 compatibility patch for Windows thread power throttling
// MinGW headers may not define THREAD_POWER_THROTTLING_STATE

#ifndef MINGW_COMPAT_H
#define MINGW_COMPAT_H

#ifdef _WIN32
#ifdef __MINGW32__

// MinGW may not have these definitions
#ifndef THREAD_POWER_THROTTLING_CURRENT_VERSION
#define THREAD_POWER_THROTTLING_CURRENT_VERSION 1
#endif

#ifndef THREAD_POWER_THROTTLING_EXECUTION_SPEED
#define THREAD_POWER_THROTTLING_EXECUTION_SPEED 0x1
#endif

#ifndef THREAD_POWER_THROTTLING_VALID_FLAGS
#define THREAD_POWER_THROTTLING_VALID_FLAGS 0x1
#endif

// Define the structure if not present
#ifndef _THREAD_POWER_THROTTLING_STATE
typedef struct _THREAD_POWER_THROTTLING_STATE {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
} THREAD_POWER_THROTTLING_STATE;
#endif

#endif // __MINGW32__
#endif // _WIN32

#endif // MINGW_COMPAT_H
EOF

echo "  ✓ 创建 mingw_compat.h"

# 补丁 2: 在 ggml-cpu.c 中包含兼容头文件
CPU_FILE="$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.c"

if [ -f "$CPU_FILE" ]; then
    # 在文件开头添加 include
    if ! grep -q "mingw_compat.h" "$CPU_FILE"; then
        sed -i '1i\#include "mingw_compat.h"' "$CPU_FILE"
        echo "  ✓ 修改 ggml-cpu.c"
    fi
fi

echo "补丁应用完成"
echo ""
echo "重新编译请运行:"
echo "  cd $LLAMA_DIR/build-windows-cpu"
echo "  cmake --build . --config Release -j\$(nproc)"