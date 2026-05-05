#!/bin/bash

echo "========================================"
echo "  MoXing ROCm 测试"
echo "========================================"
echo ""

export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib

echo "测试 moxing devices:"
moxing devices

echo ""
echo "========================================"
echo "  ROCm 7.12 库已安装完成"
echo "========================================"
echo ""
echo "已安装的库:"
ls -la /opt/rocm/core-7.12/lib/lib{hipblas,rocblas,hipblaslt,rocroller}* 2>/dev/null

echo ""
echo "测试命令:"
echo "  moxing ollama serve gemma4:31b -b rocm -d gpu1"
echo "  moxing ollama serve llama3.3:70b -b rocm -d gpu0"