#!/bin/bash

echo "========================================"
echo "  MoXing ROCm 诊断脚本"
echo "========================================"
echo ""

# 1. 清理
echo "[1/6] 清理旧进程..."
pkill -f ollama-runner 2>/dev/null || true
pkill -f llama-server 2>/dev/null || true
sleep 2

# 2. 检查 runner
echo ""
echo "[2/6] 检查 runner 二进制..."
RUNNER=/home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm/ollama-runner-rocm
if [ ! -f "$RUNNER" ]; then
    echo "✗ Runner 不存在!"
    exit 1
fi
ls -lh "$RUNNER"

# 3. 检查库
echo ""
echo "[3/6] 检查 ROCm 库..."
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
ldd "$RUNNER" 2>&1 | grep -E "not found|hipblas|rocblas|amdhip" | head -10

# 4. 测试 GPU 检测
echo ""
echo "[4/6] 测试 GPU 检测..."
export HIP_VISIBLE_DEVICES=0
"$RUNNER" --version 2>&1 | grep -E "Device|ROCm|VRAM" | head -5

# 5. 启动服务
echo ""
echo "[5/6] 启动 ROCm 服务 (后台运行, 日志写入 /tmp/rocm_test.log)..."
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313

nohup "$RUNNER" \
    -m "$MODEL" \
    --port 8081 \
    --host 0.0.0.0 \
    -c 8192 \
    -ngl 999 \
    -fa on \
    -b 512 \
    > /tmp/rocm_test.log 2>&1 &

PID=$!
echo "PID: $PID"
echo "日志: /tmp/rocm_test.log"
echo ""

# 6. 等待并测试
echo "[6/6] 等待服务就绪 (最多 60 秒)..."
for i in {1..60}; do
    if curl -s http://localhost:8081/v1/models >/dev/null 2>&1; then
        echo ""
        echo "✓ 服务就绪! 测试 API..."
        echo ""
        curl -s http://localhost:8081/v1/models
        echo ""
        echo ""
        echo "=== 服务运行中 ==="
        echo "PID: $PID"
        echo "API: http://localhost:8081/v1"
        echo "日志: tail -f /tmp/rocm_test.log"
        echo ""
        echo "停止服务: kill $PID"
        exit 0
    fi
    
    # 检查进程
    if ! ps -p $PID > /dev/null 2>&1; then
        echo ""
        echo "✗ 进程已退出，日志:"
        tail -30 /tmp/rocm_test.log
        exit 1
    fi
    
    printf "."
    sleep 1
done

echo ""
echo "✗ 超时，日志:"
tail -30 /tmp/rocm_test.log
kill $PID 2>/dev/null || true
exit 1