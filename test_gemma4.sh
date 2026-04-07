#!/bin/bash
#
# 测试 gemma4 模型 - 验证 Ollama Runner 功能
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOXING_BIN="$SCRIPT_DIR/moxing/bin"

# 标准测试问题
TEST_QUESTIONS=(
    "1+1等于几？请用中文回答。"
    "请用一句话总结机器学习是什么。"
    "什么是Python编程语言？"
)

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

# 查找模型
find_model() {
    local model_name=$1
    
    # 在 Ollama 目录查找
    local ollama_dir="$HOME/.ollama/models"
    
    if [ -d "$ollama_dir" ]; then
        # 尝试找到 manifest
        local manifests_dir="$ollama_dir/manifests/registry.ollama.ai"
        
        # 解析模型名称
        if [[ "$model_name" =~ : ]]; then
            local tag="${model_name##*:}"
            local name="${model_name%%:*}"
        else
            local tag="latest"
            local name="$model_name"
        fi
        
        # 标准化名称
        if [[ "$name" != */* ]]; then
            name="library/$name"
        fi
        
        local manifest="$manifests_dir/$name/$tag"
        
        if [ -f "$manifest" ]; then
            # 解析 manifest 获取 blob
            local digest=$(cat "$manifest" | python3 -c "import json,sys; d=json.load(sys.stdin); print([l['digest'] for l in d.get('layers',[]) if l.get('mediaType')=='application/vnd.ollama.image.model'][0])" 2>/dev/null || echo "")
            
            if [ -n "$digest" ]; then
                local blob_path="$ollama_dir/blobs/sha256-${digest#sha256:}"
                if [ -f "$blob_path" ]; then
                    echo "$blob_path"
                    return 0
                fi
            fi
        fi
    fi
    
    return 1
}

# 测试 runner
test_runner() {
    local backend=$1
    local model_path=$2
    local port=$3
    
    log_info "测试 $backend 后端..."
    
    local runner_dir="$MOXING_BIN/ollama-linux-x64-$backend"
    if [ ! -d "$runner_dir" ]; then
        log_warning "$backend 不可用，跳过"
        return 1
    fi
    
    # 设置库路径
    export LD_LIBRARY_PATH="$runner_dir:$LD_LIBRARY_PATH"
    
    # 设备选择
    local device_arg=""
    if [ "$backend" = "cuda" ]; then
        device_arg="CUDA0"
        export CUDA_VISIBLE_DEVICES=0
    elif [ "$backend" = "rocm" ]; then
        device_arg="ROCm0"
        export HIP_VISIBLE_DEVICES=0
    fi
    
    log_info "启动服务器 (端口 $port)..."
    
    # 启动服务器
    local runner="$runner_dir/ollama-runner-$backend"
    if [ ! -f "$runner" ]; then
        runner="$runner_dir/ollama-runner"
    fi
    
    # 启动并等待就绪
    "$runner" \
        -m "$model_path" \
        --port "$port" \
        --host 127.0.0.1 \
        -c 4096 \
        -ngl 999 \
        --no-webui &
    
    local server_pid=$!
    
    # 等待服务器就绪
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            break
        fi
        sleep 1
        retries=$((retries + 1))
    done
    
    if [ $retries -eq 30 ]; then
        log_error "服务器启动超时"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    
    log_success "服务器已启动 (PID: $server_pid)"
    
    # 测试问题
    local question="${TEST_QUESTIONS[0]}"
    log_info "测试问题: $question"
    
    # 调用 API
    local response=$(curl -s "http://127.0.0.1:$port/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$question\", \"n_predict\": 100}" 2>/dev/null | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('content','').strip())" 2>/dev/null || echo "")
    
    if [ -n "$response" ]; then
        log_success "响应成功:"
        echo "  $response"
    else
        log_error "无响应"
    fi
    
    # 停止服务器
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    
    return 0
}

# 主流程
main() {
    echo "=========================================="
    echo "  Gemma4 模型测试"
    echo "=========================================="
    echo ""
    
    # 查找模型
    local model_name="${1:-gemma4:31b}"
    log_info "查找模型: $model_name"
    
    local model_path=$(find_model "$model_name")
    
    if [ -z "$model_path" ]; then
        log_error "找不到模型: $model_name"
        echo "请确保模型已下载: ollama pull $model_name"
        exit 1
    fi
    
    log_success "找到模型: $model_path"
    
    # 测试每个后端
    local port=18080
    
    for backend in cuda rocm cpu; do
        echo ""
        echo "----------------------------------------"
        log_info "后端: $backend"
        echo "----------------------------------------"
        
        if test_runner "$backend" "$model_path" $port; then
            log_success "$backend 测试通过"
        else
            log_warning "$backend 测试失败"
        fi
        
        port=$((port + 1))
        sleep 2
    done
    
    echo ""
    echo "=========================================="
    log_success "测试完成！"
    echo "=========================================="
}

# 参数
if [ "$1" = "--help" ]; then
    echo "用法: $0 [模型名称]"
    echo "  $0                    # 测试 gemma4:31b"
    echo "  $0 gemma4:e4b         # 测试 gemma4:e4b"
    exit 0
fi

main "$@"
