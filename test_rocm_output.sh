#!/bin/bash

echo "========================================"
echo "  MoXing ROCm 测试 (显示所有输出)"
echo "========================================"
echo ""

echo "清理旧进程..."
pkill -f ollama-runner 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

echo ""
echo "运行命令:"
echo "moxing ollama serve gemma4:e4b -d gpu1 -b rocm --host 0.0.0.0 --runner ollama"
echo ""

moxing ollama serve gemma4:e4b -d gpu1 -b rocm --host 0.0.0.0 --runner ollama