#!/bin/bash
# Package ROCm library for Ollama
# This creates a portable ROCm runtime package (~80MB)

set -e

MOXING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOXING_DIR="$(cd "$MOXING_DIR/.." && pwd)"
VERSION=$(python3 -c "import sys; sys.path.insert(0, '$MOXING_DIR'); from moxing import __version__; print(__version__)")
PACKAGE_NAME="moxing-rocm-$VERSION"
PACKAGE_DIR="$MOXING_DIR/dist/$PACKAGE_NAME"

echo "=== Creating ROCm Package ==="
echo ""

# Check source
ROCM_SOURCE="$MOXING_DIR/moxing/bin/linux-x64-rocm-ollama/libggml-hip.so"
if [ ! -f "$ROCM_SOURCE" ]; then
    echo "❌ Error: ROCm library not found at $ROCM_SOURCE"
    echo "   Please build ROCm binaries first"
    exit 1
fi

# Clean and create package directory
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/lib"

# Copy ROCm library
echo "📦 Copying ROCm library..."
cp "$ROCM_SOURCE" "$PACKAGE_DIR/lib/"
ROCM_SIZE=$(du -h "$PACKAGE_DIR/lib/libggml-hip.so" | cut -f1)
echo "  ✅ ROCm library ($ROCM_SIZE)"

# Create setup script
cat > "$PACKAGE_DIR/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# Install ROCm library for Ollama

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing ROCm Library for Ollama ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: Ollama not found"
    echo "   Please install Ollama first: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Install ROCm library
echo "Installing ROCm library..."
sudo mkdir -p /usr/lib/ollama/rocm
sudo cp "$SCRIPT_DIR/lib/libggml-hip.so" /usr/lib/ollama/rocm/
sudo chmod 755 /usr/lib/ollama/rocm/libggml-hip.so

echo ""
echo "✅ ROCm library installed successfully!"
echo ""
echo "Usage:"
echo "  OLLAMA_LLM_LIBRARY=rocm ollama run model"
echo ""
echo "Or with MoXing:"
echo "  moxing ollama serve model -b rocm -d gpu1"
INSTALL_EOF
chmod +x "$PACKAGE_DIR/install.sh"

# Create README
cat > "$PACKAGE_DIR/README.md" << 'README_EOF'
# ROCm Library for Ollama

## What's Included

- libggml-hip.so (~62MB) - ROCm backend for AMD GPUs

## Installation

```bash
./install.sh
```

## Usage

After installation, use with Ollama:

```bash
# Set ROCm as default
export OLLAMA_LLM_LIBRARY=rocm

# Run model
ollama run gemma4:e2b

# Or with specific GPU
HIP_VISIBLE_DEVICES=0 ollama run model
```

## With MoXing

```bash
moxing ollama serve model -b rocm -d gpu1
```

## Why This Package?

Ollama doesn't include ROCm libraries by default. This package provides:

- AMD GPU support for Ollama
- ROCm backend compiled from llama.cpp
- One-time setup for all AMD GPUs

## Requirements

- Ollama installed
- AMD GPU with ROCm 6.0+ support
- ROCm runtime installed

## Uninstall

```bash
sudo rm -rf /usr/lib/ollama/rocm
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
echo "✅ ROCm package created successfully!"
echo ""
echo "📊 Package sizes:"
echo "  Directory:  $PACKAGE_SIZE"
echo "  Tarball:    $TARBALL_SIZE"
echo ""
echo "📁 Output file:"
echo "  $MOXING_DIR/dist/$PACKAGE_NAME.tar.gz"
