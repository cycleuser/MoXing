#!/bin/bash
# Create MoXing release package
# This script creates a minimal release package with only necessary binaries

set -e

MOXING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=$(grep "__version__" "$MOXING_DIR/moxing/__init__.py" | cut -d'"' -f2)
DIST_DIR="$MOXING_DIR/dist"
RELEASE_DIR="$DIST_DIR/moxing-$VERSION"

echo "=== Creating MoXing Release Package v$VERSION ==="
echo ""

# Clean dist directory
rm -rf "$DIST_DIR"
mkdir -p "$RELEASE_DIR"

# Build wheel
echo "📦 Building wheel package..."
cd "$MOXING_DIR"
python -m build --wheel --outdir "$DIST_DIR"

# Copy ROCm library
echo ""
echo "📦 Preparing ROCm library..."
mkdir -p "$RELEASE_DIR/lib"
if [ -f "$MOXING_DIR/moxing/bin/linux-x64-rocm-ollama/libggml-hip.so" ]; then
    cp "$MOXING_DIR/moxing/bin/linux-x64-rocm-ollama/libggml-hip.so" "$RELEASE_DIR/lib/"
    echo "  ✅ ROCm library included (62MB)"
else
    echo "  ⚠️  ROCm library not found, skipping"
fi

# Copy llama.cpp binaries
echo ""
echo "📦 Preparing llama.cpp binaries..."
for backend in cuda rocm vulkan cpu; do
    src_dir="$MOXING_DIR/moxing/bin/linux-x64-$backend"
    if [ -d "$src_dir" ]; then
        mkdir -p "$RELEASE_DIR/bin/linux-x64-$backend"
        cp "$src_dir/llama-server" "$RELEASE_DIR/bin/linux-x64-$backend/" 2>/dev/null || true
        cp "$src_dir/llama-cli" "$RELEASE_DIR/bin/linux-x64-$backend/" 2>/dev/null || true
        echo "  ✅ $backend binaries included"
    fi
done

# Create setup script
cat > "$RELEASE_DIR/setup_rocm.sh" << 'SETUP_EOF'
#!/bin/bash
# Setup ROCm library for Ollama
sudo mkdir -p /usr/lib/ollama/rocm
sudo cp lib/libggml-hip.so /usr/lib/ollama/rocm/
sudo chmod 755 /usr/lib/ollama/rocm/libggml-hip.so
echo "✅ ROCm library installed!"
SETUP_EOF
chmod +x "$RELEASE_DIR/setup_rocm.sh"

# Create README
cat > "$RELEASE_DIR/README.md" << 'README_EOF'
# MoXing Release Package

## Installation

1. Install the wheel package:
   ```bash
   pip install moxing-*.whl
   ```

2. (AMD GPUs only) Setup ROCm:
   ```bash
   ./setup_rocm.sh
   ```

## Binary Sizes

- **CUDA**: Uses system Ollama's built-in libraries (cuda_v13)
- **ROCm**: 62MB (included in this package)
- **Vulkan**: Uses system Ollama's built-in libraries
- **CPU**: Uses system Ollama's built-in libraries

## Usage

```bash
# Check devices
moxing devices

# CUDA (NVIDIA)
moxing ollama serve model -b cuda -d gpu0

# ROCm (AMD)
moxing ollama serve model -b rocm -d gpu1

# Vulkan
moxing ollama serve model -b vulkan

# CPU
moxing ollama serve model -b cpu
```

## Why No CUDA Libraries?

Ollama already includes CUDA v13 libraries (932MB) which are 4x smaller than v12.
Using system libraries saves 2.5GB of download and disk space.

## Requirements

- Ollama installed (provides CUDA, Vulkan, CPU libraries)
- Python 3.8+
- For ROCm: AMD GPU with ROCm runtime
README_EOF

# Create tarball
echo ""
echo "📦 Creating release tarball..."
cd "$DIST_DIR"
tar -czf "moxing-$VERSION-release.tar.gz" "moxing-$VERSION"

# Calculate sizes
WHEEL_SIZE=$(du -h "$DIST_DIR"/moxing-*.whl | cut -f1)
ROCM_SIZE=$(du -h "$RELEASE_DIR/lib/libggml-hip.so" 2>/dev/null | cut -f1 || echo "N/A")
TOTAL_SIZE=$(du -h "$DIST_DIR/moxing-$VERSION-release.tar.gz" | cut -f1)

echo ""
echo "✅ Release package created successfully!"
echo ""
echo "📊 Package sizes:"
echo "  Wheel:      $WHEEL_SIZE"
echo "  ROCm lib:   $ROCM_SIZE"
echo "  Total:      $TOTAL_SIZE"
echo ""
echo "📁 Output files:"
echo "  $DIST_DIR/moxing-$VERSION-release.tar.gz"
echo "  $DIST_DIR/moxing-$VERSION-py3-none-any.whl"
echo ""
echo "🚀 Next steps:"
echo "  1. Upload wheel to PyPI: twine upload $DIST_DIR/*.whl"
echo "  2. Create GitHub release with tarball"
echo "  3. Update release notes"
