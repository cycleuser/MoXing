#!/bin/bash
# 停止服务器
BACKEND=$1
DEVICE=$2

PID_FILE="/tmp/server_${BACKEND}_${DEVICE}.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill $PID 2>/dev/null
    rm -f "$PID_FILE"
    echo "Stopped server (PID: $PID)"
else
    echo "No server PID file found"
fi

pkill -f "llama-server.*${BACKEND}" 2>/dev/null || true