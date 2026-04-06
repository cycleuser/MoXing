#!/bin/bash
# 运行推理测试并测量速度
PORT=$1
QUESTION="$2"
MAX_TOKENS=$3

START=$(date +%s.%N)

RESULT=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"test\", \"messages\": [{\"role\": \"user\", \"content\": \"$QUESTION\"}], \"max_tokens\": $MAX_TOKENS}")

END=$(date +%s.%N)
TIME=$(echo "$END - $START" | bc)

TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")

if [ "$TOKENS" -gt 0 ] && [ $(echo "$TIME > 0" | bc) -eq 1 ]; then
    SPEED=$(echo "scale=1; $TOKENS / $TIME" | bc)
else
    SPEED="0"
fi

echo "$TOKENS $SPEED $TIME"