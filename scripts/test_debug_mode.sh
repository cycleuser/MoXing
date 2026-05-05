#!/bin/bash

echo "========================================"
echo "  MoXing ROCm 调试模式测试"
echo "========================================"
echo ""

# 清理
pkill -f ollama-runner 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

echo "测试命令:"
echo ""
echo "  moxing ollama serve gemma4:e4b \\"
echo "    -d gpu1 \\"
echo "    -b rocm \\"
echo "    --runner-verbose \\"
echo "    --fit off \\"
echo "    --host 0.0.0.0"
echo ""
echo "========================================"
echo ""

moxing ollama serve gemma4:e4b \
  -d gpu1 \
  -b rocm \
  --runner-verbose \
  --fit off \
  --host 0.0.0.0