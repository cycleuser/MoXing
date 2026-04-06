#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODELS=("gemma3:4b" "gemma4:e4b" "carstenuhlig/omnicoder-9b")
PROMPT="Hello, please introduce yourself briefly."
MAX_TOKENS=50

test_backend() {
    local model=$1
    local device=$2
    local backend=$3
    
    echo -e "${BLUE}Testing: $model with $backend on $device${NC}"
    
    LOG_FILE="/tmp/moxing_${model}_${backend}_${device}.log"
    
    timeout 120 moxing ollama run "$model" \
        -d "$device" \
        -b "$backend" \
        -p "$PROMPT" \
        -n "$MAX_TOKENS" \
        2>&1 > "$LOG_FILE"
    
    if grep -qi "error\|failed\|crash" "$LOG_FILE" 2>/dev/null; then
        echo -e "${RED}FAILED${NC}"
        cat "$LOG_FILE"
        return 1
    fi
    
    if grep -qi "server ready\|model loaded" "$LOG_FILE" 2>/dev/null; then
        echo -e "${GREEN}PASSED${NC}"
        grep -i "response\|tokens" "$LOG_FILE" | head -5
        rm -f "$LOG_FILE"
        return 0
    fi
    
    echo -e "${YELLOW}UNKNOWN${NC}"
    return 2
}

echo "========================================"
echo "MoXing Comprehensive Backend Test"
echo "========================================"

moxing devices

PASS=0
FAIL=0

echo ""
echo -e "${BLUE}=== ROCm Tests ===${NC}"

for model in "${MODELS[@]}"; do
    if test_backend "$model" "gpu1" "rocm"; then
        ((PASS++))
    else
        ((FAIL++))
    fi
    sleep 3
done

echo ""
echo -e "${BLUE}=== Vulkan Tests ===${NC}"

for model in "${MODELS[@]}"; do
    if test_backend "$model" "gpu1" "vulkan"; then
        ((PASS++))
    else
        ((FAIL++))
    fi
    sleep 3
done

echo ""
echo -e "${BLUE}=== CUDA Tests (small models only) ===${NC}"

for model in "gemma3:4b"; do
    if test_backend "$model" "gpu0" "cuda"; then
        ((PASS++))
    else
        ((FAIL++))
    fi
    sleep 3
done

echo ""
echo -e "${BLUE}=== Vulkan on NVIDIA ===${NC}"

for model in "gemma3:4b"; do
    if test_backend "$model" "gpu0" "vulkan"; then
        ((PASS++))
    else
        ((FAIL++))
    fi
    sleep 3
done

echo ""
echo "========================================"
echo -e "Summary: ${GREEN}Passed: $PASS${NC} ${RED}Failed: $FAIL${NC}"
echo "========================================"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi