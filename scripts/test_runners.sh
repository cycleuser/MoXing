#!/bin/bash

# 清理
pkill -f ollama-runner 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

echo "========================================"
echo "  MoXing Runner 测试脚本"
echo "========================================"
echo ""

MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313

echo "可用的 Runner 类型:"
echo ""
echo "1. Ollama Runner (ollama):"
ls -la /home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-*/ollama-runner-* 2>/dev/null | awk '{print "  " $NF}'
echo ""
echo "2. Official Runner (llama.cpp):"
ls -la /home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/official-linux-x64-*/llama-server 2>/dev/null | awk '{print "  " $NF}'
echo ""

echo "========================================"
echo "  测试 1: Official Vulkan Runner"
echo "========================================"
echo ""
echo "命令: moxing ollama serve gemma4:31b --runner official -b vulkan -d gpu1 -p 8081"
echo ""

moxing ollama serve gemma4:31b --runner official -b vulkan -d gpu1 -p 8081 -v 2>&1 &
PID1=$!

echo "等待启动..."
sleep 30

if curl -s http://localhost:8081/v1/models 2>/dev/null | grep -q "gemma"; then
    echo "✓ Official Vulkan 服务成功!"
    echo ""
    echo "测试 Chat:"
    curl -s http://localhost:8081/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gemma4:31b", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 20}' | head -20
else
    echo "✗ Official Vulkan 服务未响应"
    ps aux | grep $PID1 | grep -v grep
fi

echo ""
echo "停止 Official Vulkan 服务..."
kill $PID1 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

echo ""
echo "========================================"
echo "  测试 2: Ollama Vulkan Runner"
echo "========================================"
echo ""
echo "命令: moxing ollama serve gemma4:31b --runner ollama -b vulkan -d gpu1 -p 8082"
echo ""

moxing ollama serve gemma4:31b --runner ollama -b vulkan -d gpu1 -p 8082 -v 2>&1 &
PID2=$!

echo "等待启动..."
sleep 30

if curl -s http://localhost:8082/v1/models 2>/dev/null | grep -q "gemma"; then
    echo "✓ Ollama Vulkan 服务成功!"
    echo ""
    echo "测试 Chat:"
    curl -s http://localhost:8082/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gemma4:31b", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 20}' | head -20
else
    echo "✗ Ollama Vulkan 服务未响应"
fi

echo ""
echo "停止 Ollama Vulkan 服务..."
kill $PID2 2>/dev/null
pkill -f ollama-runner 2>/dev/null
sleep 2

echo ""
echo "========================================"
echo "  测试完成"
echo "========================================"
echo ""
echo "使用方法:"
echo "  moxing ollama serve gemma4:31b --runner ollama -b vulkan -d gpu1"
echo "  moxing ollama serve gemma4:31b --runner official -b vulkan -d gpu1"