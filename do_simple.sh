#!/bin/bash
set -e

echo "=========================================="
echo "Rebuild Ollama Runner with System ROCm 7.12"
echo "=========================================="

echo ""
echo "[1/3] Installing Go if not present..."
if ! command -v go &> /dev/null; then
    echo "Installing latest Go..."
    GO_VERSION="1.24.2"
    GO_URL="https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
    wget -q $GO_URL -O /tmp/go.tar.gz
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz
    export PATH="/usr/local/go/bin:$PATH"
    echo "Go installed: $(go version)"
else
    echo "Go already installed: $(go version)"
fi

echo ""
echo "[2/3] Cleaning old packages (optional)..."
sudo apt-get remove -y libamd-comgr-dev libamd-comgr2 libhsa-runtime-dev libhsa-runtime64-1 || true
sudo apt autoremove -y || true

echo ""
echo "[3/3] Rebuilding Ollama with system ROCm..."
cd /home/fred/Documents/GitHub/Others/ollama

export ROCM_PATH="/opt/rocm/core-7.12"
export PATH="/usr/local/go/bin:$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export HIP_PLATFORM="amd"

echo "ROCm version: $(cat $ROCM_PATH/.info/version)"
echo "ROCM_PATH: $ROCM_PATH"
echo "Go version: $(go version)"

go generate ./...
go build -o ollama-rocm-7.12 .

echo ""
echo "=========================================="
echo "Build completed!"
echo "New binary: /home/fred/Documents/GitHub/Others/ollama/ollama-rocm-7.12"
echo "=========================================="