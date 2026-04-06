#!/bin/bash
# Upload release packages to GitHub
# This script creates a GitHub release with binary packages

set -e

MOXING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=$(python3 -c "import sys; sys.path.insert(0, '$MOXING_DIR'); from moxing import __version__; print(__version__)")
DIST_DIR="$MOXING_DIR/dist"

echo "=== Uploading MoXing v$VERSION to GitHub ==="
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ Error: GitHub CLI not found"
    echo "   Install: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Error: Not authenticated with GitHub"
    echo "   Run: gh auth login"
    exit 1
fi

# Build wheel
echo "📦 Building wheel package..."
cd "$MOXING_DIR"
python -m build --wheel --outdir "$DIST_DIR"

# Check files
WHEEL_FILE="$DIST_DIR/moxing-$VERSION-py3-none-any.whl"
CUDA_FILE="$DIST_DIR/moxing-cuda-v13-$VERSION.tar.gz"
ROCM_FILE="$DIST_DIR/moxing-rocm-$VERSION.tar.gz"

if [ ! -f "$WHEEL_FILE" ]; then
    echo "❌ Error: Wheel file not found"
    exit 1
fi

if [ ! -f "$CUDA_FILE" ]; then
    echo "⚠️  Warning: CUDA package not found, creating..."
    bash "$MOXING_DIR/scripts/package_cuda_v13.sh"
fi

if [ ! -f "$ROCM_FILE" ]; then
    echo "⚠️  Warning: ROCm package not found, creating..."
    bash "$MOXING_DIR/scripts/package_rocm.sh"
fi

# Create release notes
cat > /tmp/release_notes.md << 'NOTES_EOF'
## MoXing vVERSION

### 🎉 New Features

- **Multi-backend Ollama support**: CUDA, ROCm, Vulkan, CPU
- **Device selection**: Choose specific GPU with `-d gpu0` or `-d gpu1`
- **Backend selection**: Choose backend with `-b cuda`, `-b rocm`, etc.
- **Performance testing**: Comprehensive benchmark results for 11 models × 4 backends

### 📦 Binary Packages

This release includes optional binary packages:

| Package | Size | Purpose |
|---------|------|---------|
| **CUDA v13** | 750MB | NVIDIA GPU support (4x smaller than v12) |
| **ROCm** | 12MB | AMD GPU support |

### 🚀 Quick Start

```bash
# Install
pip install moxing-VERSION-py3-none-any.whl

# For NVIDIA CUDA
tar -xzf moxing-cuda-v13-VERSION.tar.gz
cd moxing-cuda-v13-VERSION && ./install.sh

# For AMD ROCm
tar -xzf moxing-rocm-VERSION.tar.gz
cd moxing-rocm-VERSION && ./install.sh
```

### 📊 Performance Highlights

| Backend | Success Rate | Avg Speed | Best For |
|---------|-------------|-----------|----------|
| **ROCm** | 95% | 12-20 tok/s | AMD GPUs, large models |
| **CUDA** | 60% | 10-15 tok/s | NVIDIA GPUs < 8GB |
| **Vulkan** | 50% | 7-12 tok/s | Cross-platform |
| **CPU** | 100% | 0.5-5 tok/s | Fallback |

### 🎯 Usage Examples

```bash
# Check available devices
moxing devices

# Serve Ollama model on CUDA
moxing ollama serve gemma4:e2b -b cuda -d gpu0

# Serve on ROCm (AMD)
moxing ollama serve gemma4:26b -b rocm -d gpu1

# Serve on Vulkan
moxing ollama serve model -b vulkan
```

### 📝 Documentation

- [Performance Comparison](./docs/ollama_backend_comparison.md)
- [Test Report](./docs/ollama_backend_test_report.md)
- [Binary Packages](./docs/binary_packages.md)

### 🔧 What's Changed

- Use CUDA v13 (932MB) instead of v12 (2.4GB) - 63% smaller!
- Automatic backend selection based on device
- Environment variable control (OLLAMA_LLM_LIBRARY, etc.)
- One-time ROCm setup for AMD GPUs

### ⚠️ Requirements

- Ollama installed (provides CUDA, Vulkan, CPU libraries)
- Python 3.8+
- For CUDA: NVIDIA GPU with CUDA 13 support
- For ROCm: AMD GPU with ROCm 6.0+ runtime

Full Changelog: https://github.com/cycleuser/MoXing/compare/vPREVIOUS...vVERSION
NOTES_EOF

# Update version in notes
sed -i "s/vVERSION/v$VERSION/g" /tmp/release_notes.md

# Get previous version
PREVIOUS=$(git describe --tags --abbrev=0 2>/dev/null || echo "0.1.25")
sed -i "s/vPREVIOUS/$PREVIOUS/g" /tmp/release_notes.md

echo ""
echo "📦 Files to upload:"
echo "  - $(basename $WHEEL_FILE)"
echo "  - $(basename $CUDA_FILE)"
echo "  - $(basename $ROCM_FILE)"
echo ""

# Create GitHub release
echo "🚀 Creating GitHub release..."
gh release create "v$VERSION" \
    --title "MoXing v$VERSION" \
    --notes-file /tmp/release_notes.md \
    "$WHEEL_FILE" \
    "$CUDA_FILE" \
    "$ROCM_FILE"

echo ""
echo "✅ Release v$VERSION uploaded successfully!"
echo ""
echo "🔗 View release:"
echo "   https://github.com/cycleuser/MoXing/releases/tag/v$VERSION"
EOF
chmod +x ~/Documents/GitHub/cycleuser/MoXing/scripts/upload_to_github.sh
cat ~/Documents/GitHub/cycleuser/MoXing/scripts/upload_to_github.sh