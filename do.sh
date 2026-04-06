#!/bin/bash
set -e

echo "=========================================="
echo "ROCm Installation and Runner Compilation"
echo "=========================================="

echo ""
echo "[1/5] Cleaning old ROCm packages..."
sudo apt-get remove -y libamd-comgr-dev libamd-comgr2 libhsa-runtime-dev libhsa-runtime64-1 || true

echo ""
echo "[2/5] Setting up TheRock build environment..."
cd /home/fred/Documents/GitHub/Others/TheRock

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "[3/5] Fetching TheRock sources (this may take a while)..."
python3 ./build_tools/fetch_sources.py --progress --jobs 4

echo ""
echo "[4/5] Configuring TheRock for gfx1100 and gfx1035..."
cmake -B build -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTHEROCK_AMDGPU_FAMILIES="gfx1100" \
    -DTHEROCK_ENABLE_ALL=ON

echo ""
echo "[5/5] Building TheRock (this will take 1-2 hours)..."
ninja -C build

echo ""
echo "=========================================="
echo "TheRock build completed!"
echo "ROCm installed at: /home/fred/Documents/GitHub/Others/TheRock/build/dist/rocm"
echo "=========================================="

echo ""
echo "Now rebuilding Ollama runner with new ROCm..."
cd /home/fred/Documents/GitHub/Others/ollama

export PATH="/home/fred/Documents/GitHub/Others/TheRock/build/dist/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/home/fred/Documents/GitHub/Others/TheRock/build/dist/rocm/lib:$LD_LIBRARY_PATH"
export ROCM_PATH="/home/fred/Documents/GitHub/Others/TheRock/build/dist/rocm"

go generate ./...
go build -o ollama-rocm .

echo ""
echo "=========================================="
echo "Build completed!"
echo "New Ollama binary: /home/fred/Documents/GitHub/Others/ollama/ollama-rocm"
echo "=========================================="