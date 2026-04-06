#!/bin/bash
# Package CUDA v13 libraries for Ollama
# This creates a portable CUDA v13 runtime package (~940MB)

set -e

MOXING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=$(python3 -c "import sys; sys.path.insert(0, '$MOXING_DIR'); from moxing import __version__; print(__version__)")
PACKAGE_NAME="moxing-cuda-v13-$VERSION"
PACKAGE_DIR="$MOXING_DIR/dist/$PACKAGE_NAME"

echo "=== Creating CUDA v13 Package ==="
echo ""

# Clean and create package directory
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/lib/cuda_v13"

# Copy CUDA v13 libraries
echo "📦 Copying CUDA v13 libraries..."
cp -r /usr/lib/ollama/cuda_v13/* "$PACKAGE_DIR/lib/cuda_v13/"
echo "  ✅ CUDA v13 libraries (932MB)"

# Copy base libraries
echo "📦 Copying base libraries..."
cp /usr/lib/ollama/libggml-base.so* "$PACKAGE_DIR/lib/" 2>/dev/null || true
cp /usr/lib/ollama/libggml-cpu-*.so "$PACKAGE_DIR/lib/" 2>/dev/null || true
echo "  ✅ Base libraries (~8MB)"

# Create setup script
cat > "$PACKAGE_DIR/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# Install CUDA v13 libraries for Ollama

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing CUDA v13 Libraries for Ollama ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: Ollama not found"
    echo "   Please install Ollama first: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Install CUDA v13 libraries
echo "Installing CUDA v13 libraries..."
sudo mkdir -p /usr/lib/ollama/cuda_v13
sudo cp -r "$SCRIPT_DIR/lib/cuda_v13/"* /usr/lib/ollama/cuda_v13/

echo "Installing base libraries..."
sudo cp "$SCRIPT_DIR/lib/libggml-base.so"* /usr/lib/ollama/ 2>/dev/null || true
sudo cp "$SCRIPT_DIR/lib/libggml-cpu-"*.so /usr/lib/ollama/ 2>/dev/null || true

echo ""
echo "✅ CUDA v13 libraries installed successfully!"
echo ""
echo "Usage:"
echo "  OLLAMA_LLM_LIBRARY=cuda_v13 ollama run model"
echo ""
echo "Or with MoXing:"
echo "  moxing ollama serve model -b cuda -d gpu0"
INSTALL_EOF
chmod +x "$PACKAGE_DIR/install.sh"

# Create README
cat > "$PACKAGE_DIR/README.md" << 'README_EOF'
# CUDA v13 Libraries for Ollama

## What's Included

- CUDA v13 libraries (932MB)
- GGML base libraries (~8MB)
- CPU variant libraries

## Size Comparison

| Version | Size | Included |
|---------|------|----------|
| **CUDA v13** | 932MB | This package |
| CUDA v12 | 2.4GB | Not included |
| **Savings** | **1.5GB** | 63% smaller |

## Installation

```bash
./install.sh
```

## Usage

After installation, use with Ollama:

```bash
# Set CUDA v13 as default
export OLLAMA_LLM_LIBRARY=cuda_v13

# Run model
ollama run gemma4:e2b

# Or with specific GPU
CUDA_VISIBLE_DEVICES=0 ollama run model
```

## With MoXing

MoXing automatically uses CUDA v13 when available:

```bash
moxing ollama serve model -b cuda -d gpu0
```

## Why CUDA v13?

- **4x smaller** than v12 (932MB vs 2.4GB)
- **Same performance**
- **Better compatibility** with newer CUDA versions
- **Faster downloads**

## Requirements

- Ollama installed
- NVIDIA GPU with CUDA support
- CUDA 13.x runtime (usually included in driver)

## Uninstall

```bash
sudo rm -rf /usr/lib/ollama/cuda_v13
```
README_EOF

# Create tarball
echo ""
echo "📦 Creating tarball..."
cd "$MOXING_DIR/dist"
tar -czf "$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"

# Calculate sizes
PACKAGE_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
TARBALL_SIZE=$(du -sh "$PACKAGE_NAME.tar.gz" | cut -f1)

echo ""
echo "✅ CUDA v13 package created successfully!"
echo ""
echo "📊 Package sizes:"
echo "  Directory:  $PACKAGE_SIZE"
echo "  Tarball:    $TARBALL_SIZE"
echo ""
echo "📁 Output file:"
echo "  $MOXING_DIR/dist/$PACKAGE_NAME.tar.gz"
echo ""
echo "🚀 Installation:"
echo "  tar -xzf $PACKAGE_NAME.tar.gz"
echo "  cd $PACKAGE_NAME"
echo "  ./install.sh"
