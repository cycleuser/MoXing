#!/bin/bash
# download_binaries.sh - Download all llama.cpp binaries for MoXing
# Run this script when network is available

set -e

VERSION="b8461"
BIN_DIR="$(dirname "$0")/../moxing/bin"
BASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/$VERSION"

mkdir -p "$BIN_DIR"

echo "Downloading llama.cpp binaries version $VERSION"
echo "Target: $BIN_DIR"

# Function to download and extract
download_platform() {
    local name=$1
    local asset=$2
    local ext=$3
    
    echo ""
    echo "=== $name ==="
    
    local dest="$BIN_DIR/$name"
    mkdir -p "$dest"
    
    if [ -f "$dest/VERSION" ]; then
        echo "Already downloaded, skipping"
        return
    fi
    
    local url="$BASE_URL/$asset"
    local tmp="/tmp/moxing-$name.$ext"
    
    echo "Downloading: $url"
    
    # Try aria2c first (faster), then curl
    if command -v aria2c &> /dev/null; then
        aria2c -x 4 -s 4 -o "$tmp" "$url" || rm -f "$tmp"
    elif command -v wget &> /dev/null; then
        wget -O "$tmp" "$url" || rm -f "$tmp"
    else
        curl -L -o "$tmp" "$url" || rm -f "$tmp"
    fi
    
    if [ ! -f "$tmp" ]; then
        echo "Failed to download $name"
        return 1
    fi
    
    echo "Extracting..."
    cd /tmp
    if [ "$ext" = "zip" ]; then
        unzip -o "$tmp" -d "moxing-extract-$name"
    else
        mkdir -p "moxing-extract-$name"
        tar -xzf "$tmp" -C "moxing-extract-$name"
    fi
    
    # Copy binaries and libs
    find "moxing-extract-$name" -type f \( -name "llama-*" -o -name "*.so*" -o -name "*.dll" -o -name "*.dylib" \) \
        -exec cp {} "$dest/" \; 2>/dev/null || true
    
    # Copy symlinks
    find "moxing-extract-$name" -type l -exec cp -d {} "$dest/" \; 2>/dev/null || true
    
    # Make executables
    chmod +x "$dest"/llama-* 2>/dev/null || true
    
    # Cleanup
    rm -rf "$tmp" "moxing-extract-$name"
    
    # Create VERSION file
    echo -e "$VERSION\n${name##*-}\n" > "$dest/VERSION"
    
    echo "Done: $dest"
}

# Linux x64
download_platform "linux-x64-cpu" "llama-b8461-bin-ubuntu-x64.tar.gz" "tar.gz" &
download_platform "linux-x64-vulkan" "llama-b8461-bin-ubuntu-vulkan-x64.tar.gz" "tar.gz" &
download_platform "linux-x64-rocm" "llama-b8461-bin-ubuntu-rocm-7.2-x64.tar.gz" "tar.gz" &

# Windows x64
download_platform "windows-x64-cpu" "llama-b8461-bin-win-cpu-x64.zip" "zip" &
download_platform "windows-x64-cuda" "cudart-llama-bin-win-cuda-12.4-x64.zip" "zip" &
download_platform "windows-x64-vulkan" "llama-b8461-bin-win-vulkan-x64.zip" "zip" &

# macOS
download_platform "darwin-arm64-metal" "llama-b8461-bin-macos-arm64.tar.gz" "tar.gz" &
download_platform "darwin-x64-metal" "llama-b8461-bin-macos-x64.tar.gz" "tar.gz" &

# Wait for all downloads
wait

echo ""
echo "=== Summary ==="
ls -la "$BIN_DIR"/*/VERSION 2>/dev/null || echo "No binaries downloaded"

echo ""
echo "Total size:"
du -sh "$BIN_DIR" 2>/dev/null || echo "0"