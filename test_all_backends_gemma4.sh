#!/bin/bash
# 全面测试 gemma4:e4b 在所有设备和后端上的运行情况

set -e

MODEL="gemma4:e4b"
MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"
MOXING_DIR="/home/fred/Documents/GitHub/cycleuser/MoXing"
RESULTS_DIR="$MOXING_DIR/test_results"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "MoXing gemma4:e4b 全面测试"
echo "时间: $DATE"
echo "========================================"

test_backend() {
    local device=$1
    local backend=$2
    local port=$3
    local test_name="${device}_${backend}"
    
    echo ""
    echo "========================================"
    echo "测试: $test_name"
    echo "设备: $device, 后端: $backend, 端口: $port"
    echo "========================================"
    
    pkill -f llama-server 2>/dev/null || true
    sleep 3
    
    cd "$MOXING_DIR"
    
    echo "启动服务器..."
    timeout 60 moxing serve "$MODEL_PATH" -d "$device" -b "$backend" -p "$port" 2>&1 &
    SERVER_PID=$!
    
    sleep 12
    
    if curl -s http://127.0.0.1:$port/health > /dev/null 2>&1; then
        echo "服务器启动成功"
        
        echo "测试推理..."
        RESULT=$(curl -s http://127.0.0.1:$port/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model": "test", "messages": [{"role": "user", "content": "Hello, say hi briefly"}], "max_tokens": 20}' \
            2>&1)
        
        SPEED=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('predicted_per_second',0):.1f}\")" 2>/dev/null || echo "ERROR")
        RESPONSE=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content','ERROR')[:50])" 2>/dev/null || echo "ERROR")
        
        echo "响应: $RESPONSE"
        echo "速度: $SPEED tok/s"
        
        echo "$RESULT" > "$RESULTS_DIR/${test_name}_${DATE}.json"
        
        kill $SERVER_PID 2>/dev/null || true
    else
        echo "服务器启动失败"
        echo "FAILED" > "$RESULTS_DIR/${test_name}_${DATE}.json"
    fi
    
    pkill -f llama-server 2>/dev/null || true
    sleep 2
}

echo ""
echo "=== NVIDIA RTX 4070 测试 ==="
test_backend "gpu0" "cuda" 8080
test_backend "gpu0" "vulkan" 8081

echo ""
echo "=== AMD RX 7900 XTX 测试 ==="
test_backend "gpu1" "rocm" 8082
test_backend "gpu1" "vulkan" 8083

echo ""
echo "=== AMD Radeon 610M 测试 ==="
test_backend "gpu2" "rocm" 8084
test_backend "gpu2" "vulkan" 8085

echo ""
echo "=== CPU 测试 ==="
test_backend "cpu" "cpu" 8086

echo ""
echo "========================================"
echo "测试完成！结果保存在: $RESULTS_DIR"
echo "========================================"
echo ""
echo "=== 测试结果汇总 ==="
for f in "$RESULTS_DIR"/*_${DATE}.json; do
    name=$(basename "$f" _${DATE}.json)
    speed=$(cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('predicted_per_second',0):.1f}\")" 2>/dev/null || echo "FAILED")
    echo "$name: $speed tok/s"
done