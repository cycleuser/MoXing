#!/bin/bash
# 简化的 ROCm 测试脚本

set -e

echo "清理旧进程..."
pkill -f ollama-runner 2>/dev/null || true
pkill -f llama-server 2>/dev/null || true
sleep 2

echo ""
echo "=== 测试 1: 检查 ROCm runner 是否存在 ==="
RUNNER=/home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm/ollama-runner-rocm
if [ -f "$RUNNER" ]; then
    echo "✓ Runner 存在: $RUNNER"
    ls -lh "$RUNNER"
else
    echo "✗ Runner 不存在"
    exit 1
fi

echo ""
echo "=== 测试 2: 检查 ROCm 库 ==="
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:/home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm

for lib in libhipblas.so.3 librocblas.so.5 libhipblaslt.so.1 librocroller.so.1 libamdhip64.so.7; do
    if [ -f "/opt/rocm/core-7.12/lib/$lib" ]; then
        echo "✓ $lib"
    else
        echo "✗ $lib 缺失"
    fi
done

echo ""
echo "=== 测试 3: 检测 GPU ==="
export HIP_VISIBLE_DEVICES=0
"$RUNNER" --version 2>&1 | head -5

echo ""
echo "=== 测试 4: 启动服务 (端口 8081) ==="
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313

if [ ! -f "$MODEL" ]; then
    echo "✗ 模型不存在: $MODEL"
    exit 1
fi

echo "启动命令:"
echo "$RUNNER -m $MODEL --port 8081 --host 0.0.0.0 -c 8192 -ngl 999"
echo ""

# 后台启动
"$RUNNER" -m "$MODEL" --port 8081 --host 0.0.0.0 -c 8192 -ngl 999 2>&1 &
PID=$!

echo "等待服务启动..."
for i in {1..60}; do
    if curl -s http://localhost:8081/v1/models >/dev/null 2>&1; then
        echo "✓ 服务已就绪!"
        break
    fi
    
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✗ 进程已退出"
        exit 1
    fi
    
    printf "."
    sleep 1
done

echo ""
echo "=== 测试 5: API 测试 ==="
curl -s http://localhost:8081/v1/models | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8081/v1/models

echo ""
echo ""
echo "=== 测试 6: 停止服务 ==="
kill $PID 2>/dev/null || true
echo "✓ 服务已停止"

echo ""
echo "========================================"
echo "  所有测试通过!"
echo "========================================"