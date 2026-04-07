#!/bin/bash

echo "========================================"
echo "  MoXing ROCm 完整测试"
echo "========================================"
echo ""

# 清理旧进程
pkill -f ollama-runner 2>/dev/null
sleep 2

echo "[1/5] 测试设备检测..."
moxing devices

echo ""
echo "[2/5] 启动 ROCm 服务 (gpu1 = RX 7900 XTX)..."
echo ""

# 后台启动服务，保存日志
moxing ollama serve gemma4:31b -d gpu1 -b rocm --host 0.0.0.0 -v > /tmp/moxing_test.log 2>&1 &
PID=$!

echo "服务 PID: $PID"
echo "日志文件: /tmp/moxing_test.log"
echo ""

echo "[3/5] 等待服务就绪 (最多 120 秒)..."
for i in {1..120}; do
    if curl -s http://localhost:8080/v1/models 2>/dev/null | grep -q "gemma"; then
        echo "✓ 服务就绪! (等待了 ${i} 秒)"
        break
    fi
    
    # 检查进程是否还在
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✗ 进程已退出，查看日志:"
        tail -50 /tmp/moxing_test.log
        exit 1
    fi
    
    printf "."
    sleep 1
done

echo ""
echo ""

if ! ps -p $PID > /dev/null 2>&1; then
    echo "服务启动失败"
    tail -50 /tmp/moxing_test.log
    exit 1
fi

echo "[4/5] 测试 API..."
echo ""

echo "GET /v1/models:"
curl -s http://localhost:8080/v1/models | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8080/v1/models

echo ""
echo ""
echo "POST /v1/chat/completions:"
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma4:31b", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}' \
  | python3 -m json.tool 2>/dev/null || \
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma4:31b", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}'

echo ""
echo ""
echo "[5/5] 停止服务..."
kill $PID 2>/dev/null
echo "✓ 服务已停止"

echo ""
echo "========================================"
echo "  测试完成"
echo "========================================"