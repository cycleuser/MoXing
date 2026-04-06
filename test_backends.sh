#!/bin/bash
#
# Test all backends with llama-server --version
#

echo "Testing all backends..."
echo ""

for backend in cuda rocm vulkan cpu; do
    echo "=== Testing $backend ==="
    binary="moxing/bin/linux-x64-$backend/llama-server"
    if [ -f "$binary" ]; then
        echo "Binary: $binary"
        echo "Size: $(ls -lh $binary | awk '{print $5}')"
        echo "Version:"
        $binary --version 2>&1 | grep -E "Device|version|built" | head -5
        echo ""
    else
        echo "ERROR: Binary not found: $binary"
        echo ""
    fi
done