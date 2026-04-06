#!/bin/bash
# 测试单个后端

DEVICE=$1
BACKEND=$2
CTX=$3
PORT=$4

QUESTIONS=(
    "What is 25 * 4?"
    "What is the capital of Japan?"
    "Write a Python function to check prime."
)

MAX_TOKENS=(30 50 80)

echo "Testing $BACKEND on $DEVICE (ctx=$CTX)"

./scripts/start_server.sh "$DEVICE" "$BACKEND" "$CTX" "$PORT"
if [ $? -ne 0 ]; then
    echo "FAILED to start server"
    exit 1
fi

sleep 2

RESULTS=""

for i in "${!QUESTIONS[@]}"; do
    Q="${QUESTIONS[$i]}"
    MT="${MAX_TOKENS[$i]}"
    
    RESULT=$(./scripts/test_speed.sh "$PORT" "$Q" "$MT")
    TOKENS=$(echo "$RESULT" | awk '{print $1}')
    SPEED=$(echo "$RESULT" | awk '{print $2}')
    TIME=$(echo "$RESULT" | awk '{print $3}')
    
    RESULTS="$RESULTS $SPEED"
    
    echo "  Q$((i+1)): $TOKENS tokens, $SPEED tok/s"
    
    sleep 1
done

./scripts/stop_server.sh "$BACKEND" "$DEVICE"

echo "$RESULTS"