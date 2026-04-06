#!/bin/bash
# 启动服务器
DEVICE=$1
BACKEND=$2
CTX=$3
PORT=$4
MODEL="gemma4:e4b"

echo "Starting server: $BACKEND on $DEVICE (port $PORT, ctx $CTX)"

moxing ollama serve "$MODEL" -d "$DEVICE" -b "$BACKEND" -c "$CTX" -p "$PORT" > /tmp/server_${BACKEND}_${DEVICE}.log 2>&1 &

echo $! > /tmp/server_${BACKEND}_${DEVICE}.pid

for i in {1..60}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready on port $PORT"
        exit 0
    fi
    sleep 1
done

echo "Server failed to start"
exit 1