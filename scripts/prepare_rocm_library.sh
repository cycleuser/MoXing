#!/bin/bash
# Prepare ROCm library for Ollama
# This script copies the ROCm library from llama.cpp build to Ollama's library directory

set -e

MOXING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_SOURCE="$MOXING_DIR/moxing/bin/linux-x64-rocm-ollama/libggml-hip.so"
OLLAMA_ROCM_DIR="/usr/lib/ollama/rocm"

echo "=== Preparing ROCm Library for Ollama ==="
echo ""

# Check if source exists
if [ ! -f "$ROCM_SOURCE" ]; then
    echo "❌ Error: ROCm library not found at $ROCM_SOURCE"
    echo "   Please build the ROCm binaries first:"
    echo "   cd moxing && ./scripts/build_ollama_backends.sh"
    exit 1
fi

echo "Source: $ROCM_SOURCE ($(du -h "$ROCM_SOURCE" | cut -f1))"
echo "Target: $OLLAMA_ROCM_DIR/"
echo ""

# Create directory
if [ ! -d "$OLLAMA_ROCM_DIR" ]; then
    echo "Creating directory: $OLLAMA_ROCM_DIR"
    sudo mkdir -p "$OLLAMA_ROCM_DIR"
fi

# Copy library
echo "Copying libggml-hip.so..."
sudo cp "$ROCM_SOURCE" "$OLLAMA_ROCM_DIR/"

# Set permissions
sudo chmod 755 "$OLLAMA_ROCM_DIR/libggml-hip.so"

echo ""
echo "✅ ROCm library installed successfully!"
echo ""
echo "Usage:"
echo "  moxing ollama serve model -b rocm -d gpu1"
echo ""
echo "Or with Ollama directly:"
echo "  HIP_VISIBLE_DEVICES=0 OLLAMA_LLM_LIBRARY=rocm ollama run model"
