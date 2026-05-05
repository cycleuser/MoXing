#!/bin/bash

# 清理所有旧进程
pkill -f ollama-runner 2>/dev/null
pkill -f llama-server 2>/dev/null
sleep 2

echo "========================================"
echo "  MoXing ROCm 测试脚本"
echo "========================================"
echo ""

echo "[1/5] 检查设备..."
moxing devices
echo ""

echo "[2/5] 检查 ROCm 库..."
echo "libhipblas:"
ls -la /opt/rocm/core-7.12/lib/libhipblas* 2>/dev/null || echo "  未找到"
echo ""
echo "librocblas:"
ls -la /opt/rocm/core-7.12/lib/librocblas* 2>/dev/null || echo "  未找到"
echo ""
echo "libhipblaslt:"
ls -la /opt/rocm/core-7.12/lib/libhipblaslt* 2>/dev/null || echo "  未找到"
echo ""
echo "librocroller:"
ls -la /opt/rocm/core-7.12/lib/librocroller* 2>/dev/null || echo "  未找到"
echo ""

echo "[3/5] 检查 runner 二进制..."
ls -la /home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm/ollama-runner-rocm
echo ""

echo "[4/5] 测试 runner 版本..."
LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:/home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm \
HIP_VISIBLE_DEVICES=0 \
/home/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/ollama-linux-x64-rocm/ollama-runner-rocm --version 2>&1 | head -10
echo ""

echo "[5/5] 启动 ROCm 服务 (端口 8081)..."
echo "运行命令:"
echo "moxing ollama serve gemma4:31b -d gpu1 -b rocm --host 0.0.0.0 -p 8081 -v"
echo ""
echo "然后在新终端测试:"
echo "curl http://localhost:8081/v1/models"
echo ""
echo "或运行完整测试:"
echo "moxing ollama serve gemma4:31b -d gpu1 -b rocm --host 0.0.0.0 -p 8081"