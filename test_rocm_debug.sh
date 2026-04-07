#!/bin/bash

echo "========================================"
echo "  MoXing ROCm 完整调试测试"
echo "========================================"
echo ""

# 清理旧进程
pkill -f ollama-runner 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

echo "新增选项:"
echo "  --runner-verbose  启用 runner 详细日志"
echo "  --fit off         禁用参数自动调优"
echo ""

echo "========================================"
echo "  测试 1: ROCm + 详细日志 + 禁用 fit"
echo "========================================"
echo ""
echo "命令: moxing ollama serve gemma4:e4b -d gpu1 -b rocm --runner-verbose --fit off --host 0.0.0.0"
echo ""

moxing ollama serve gemma4:e4b -d gpu1 -b rocm --runner-verbose --fit off --host 0.0.0.0