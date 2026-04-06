#!/bin/bash
# MoXing 全面性能测试

set -e

MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"
MOXING_DIR="/home/fred/Documents/GitHub/cycleuser/MoXing"
RESULTS_DIR="$MOXING_DIR/benchmark_results"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/test_${DATE}.log"
SUMMARY_FILE="$RESULTS_DIR/summary_${DATE}.txt"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cleanup() {
    log "清理进程..."
    sudo killall -9 ollama 2>/dev/null || true
    killall -9 llama-server 2>/dev/null || true
    sleep 3
}

test_backend() {
    local device=$1
    local backend=$2
    local port=$3
    local cpu_offload=$4
    local ctx=$5
    local test_name="${device}_${backend}"
    
    log "========================================"
    log "测试: $test_name"
    log "设备: $device, 后端: $backend, CPU offload: $cpu_offload, 上下文: $ctx"
    log "========================================"
    
    cleanup
    
    cd "$MOXING_DIR"
    
    local cmd="moxing serve $MODEL_PATH -d $device -b $backend -p $port -c $ctx"
    if [ "$cpu_offload" != "0" ]; then
        cmd="$cmd --cpu-offload $cpu_offload"
    fi
    
    log "启动: $cmd"
    
    timeout 90 $cmd 2>&1 &
    SERVER_PID=$!
    
    sleep 15
    
    if curl -s http://127.0.0.1:$port/health > /dev/null 2>&1; then
        log "服务器启动成功，运行推理..."
        
        result=$(curl -s http://127.0.0.1:$port/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model": "test", "messages": [{"role": "user", "content": "Say hello briefly"}], "max_tokens": 20}')
        
        speed=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('predicted_per_second',0):.1f}\")" 2>/dev/null || echo "ERROR")
        prompt_speed=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('prompt_per_second',0):.1f}\")" 2>/dev/null || echo "ERROR")
        
        log "生成速度: $speed tok/s, 提示速度: $prompt_speed tok/s"
        
        echo "$test_name: 生成=$speed tok/s, 提示=$prompt_speed tok/s, CPU_offload=$cpu_offload" >> "$SUMMARY_FILE"
        echo "$result" > "$RESULTS_DIR/${test_name}_${DATE}.json"
        
        kill $SERVER_PID 2>/dev/null || true
        return 0
    else
        log "服务器启动失败!"
        echo "$test_name: FAILED" >> "$SUMMARY_FILE"
        return 1
    fi
}

log "========================================"
log "MoXing 全面性能测试 - gemma4:e4b (9GB)"
log "========================================"

echo "# MoXing Benchmark - $DATE" > "$SUMMARY_FILE"
echo "# Model: gemma4:e4b (9GB)" >> "$SUMMARY_FILE"

log ""
log "=== RTX 4070 (8GB, 需CPU offload) - CUDA ==="
test_backend "gpu0" "cuda" 8080 "15" "4096"

log ""
log "=== RTX 4070 (8GB, 需CPU offload) - Vulkan ==="
test_backend "gpu0" "vulkan" 8081 "15" "4096"

log ""
log "=== RX 7900 XTX (24GB, 全GPU) - ROCm ==="
test_backend "gpu1" "rocm" 8082 "0" "8192"

log ""
log "=== RX 7900 XTX (24GB, 全GPU) - Vulkan ==="
test_backend "gpu1" "vulkan" 8083 "0" "8192"

log ""
log "=== Radeon 610M (512MB, 重度CPU offload) - ROCm ==="
test_backend "gpu2" "rocm" 8084 "40" "2048"

log ""
log "=== Radeon 610M (512MB, 重度CPU offload) - Vulkan ==="
test_backend "gpu2" "vulkan" 8085 "40" "2048"

log ""
log "=== CPU (AMD 7945HX 16核) ==="
test_backend "cpu" "cpu" 8086 "0" "4096"

log ""
log "========================================"
log "测试完成！结果汇总:"
log "========================================"
cat "$SUMMARY_FILE"

cleanup