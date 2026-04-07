#!/bin/bash

# 清理
pkill -f ollama-runner 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

echo "========================================"
echo "  测试 Official Vulkan Runner"
echo "========================================"
echo ""

echo "命令: moxing ollama serve gemma4:26b --runner official -b vulkan -d gpu1 --host 0.0.0.0"
echo ""

# 使用 -v 选项查看详细输出
moxing ollama serve gemma4:26b --runner official -b vulkan -d gpu1 --host 0.0.0.0 -v