#!/bin/bash

echo "========================================"
echo "  MoXing ROCm API 测试"
echo "========================================"
echo ""

# 清理
pkill -f ollama-runner 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

# 后台启动
echo "[1/4] 启动 ROCm 服务..."
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export HIP_VISIBLE_DEVICES=0

MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a

/home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm/ollama-runner-rocm \
  -m "$MODEL" \
  --port 8081 \
  --host 0.0.0.0 \
  -c 4096 \
  -ngl 999 \
  --fit off \
  > /tmp/rocm_server.log 2>&1 &

PID=$!
echo "PID: $PID"
echo "日志: /tmp/rocm_server.log"
echo ""

echo "[2/4] 等待服务启动 (最多 60 秒)..."
for i in {1..60}; do
    if curl -s http://localhost:8081/v1/models 2>/dev/null | grep -q "gemma"; then
        echo ""
        echo "✓ 服务就绪!"
        break
    fi
    
    if ! ps -p $PID > /dev/null 2>&1; then
        echo ""
        echo "✗ 进程已退出，查看日志:"
        tail -50 /tmp/rocm_server.log
        exit 1
    fi
    
    printf "."
    sleep 1
done

echo ""
echo ""
echo "[3/4] 测试 API..."
echo ""

echo "GET /v1/models:"
curl -s http://localhost:8081/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "API 无响应"
echo ""

echo ""
echo "POST /v1/chat/completions:"
curl -s http://localhost:8081/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gemma4:e4b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20}' 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "API 无响应"

echo ""
echo ""
echo "[4/4] 清理..."
echo "进程状态:"
ps aux | grep $PID | grep -v grep || echo "进程已退出"

if ps -p $PID > /dev/null 2>&1; then
    echo ""
    echo "服务仍在运行，按 Enter 停止..."
    read
    kill $PID 2>/dev/null
fi

echo ""
echo "日志尾部:"
tail -20 /tmp/rocm_server.log