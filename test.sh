#!/bin/bash
set -e

echo "=========================================="
echo "Testing Ollama ROCm 7.12 on RX 7900 XTX"
echo "=========================================="

OLLAMA_BIN="/home/fred/Documents/GitHub/Others/ollama/ollama-rocm-7.12"

echo ""
echo "!!! IMPORTANT !!!"
echo "Please stop system ollama first:"
echo "  sudo systemctl stop ollama"
echo "  pkill -f 'ollama serve'"
echo ""
read -p "Press Enter after stopping system ollama..."

echo ""
echo "GPU Selection:"
echo "  HIP_VISIBLE_DEVICES=0 (RX 7900 XTX only)"
amd-smi | grep "AMD Radeon"

echo ""
echo "[1/3] Starting server with explicit ROCm path..."
env \
  ROCM_PATH=/opt/rocm/core-7.12 \
  LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib \
  HIP_VISIBLE_DEVICES=0 \
  $OLLAMA_BIN serve &
sleep 5

echo ""
echo "[2/3] Pulling gemma4:31b..."
env \
  ROCM_PATH=/opt/rocm/core-7.12 \
  LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib \
  HIP_VISIBLE_DEVICES=0 \
  $OLLAMA_BIN pull gemma4:31b

echo ""
echo "[3/3] Running gemma4:31b - type your prompt:"
env \
  ROCM_PATH=/opt/rocm/core-7.12 \
  LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib \
  HIP_VISIBLE_DEVICES=0 \
  $OLLAMA_BIN run gemma4:31b

echo ""
echo "Check GPU usage: amd-smi"