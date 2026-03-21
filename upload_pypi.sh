#!/usr/bin/env bash
# MoXing - Upload wheel to PyPI
#
# This script uploads the built wheel to PyPI.
# Run generate_wheel.sh first to build the wheel.
#
# Usage:
#   ./upload_pypi.sh              # Upload to PyPI
#   ./upload_pypi.sh --test       # Upload to TestPyPI
#   ./upload_pypi.sh --check      # Check wheel only, don't upload
#
# Prerequisites:
#   pip install twine
#   Set TWINE_USERNAME and TWINE_PASSWORD (or use API token)

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"

# Parse arguments
TEST=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --test|-t)
            TEST=1
            shift
            ;;
        --check|-c)
            CHECK_ONLY=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test, -t    Upload to TestPyPI instead of PyPI"
            echo "  --check, -c   Check wheel only, don't upload"
            echo "  --help, -h    Show this help"
            echo ""
            echo "Prerequisites:"
            echo "  pip install twine"
            echo "  Set TWINE_USERNAME and TWINE_PASSWORD (or use API token)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== MoXing PyPI Uploader ==="

# Check for wheel
WHEEL=$(ls -t dist/*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
    echo "Error: No wheel found in dist/"
    echo "Run generate_wheel.sh first."
    exit 1
fi

echo "Wheel: $WHEEL"
WHEEL_SIZE=$(du -h "$WHEEL" | cut -f1)
echo "Size: $WHEEL_SIZE"

# Get version from wheel name
VERSION=$(basename "$WHEEL" | sed 's/moxing-\(.*\)-py3-none-any.whl/\1/')
echo "Version: $VERSION"

# Check wheel
echo ""
echo "[1/2] Checking wheel..."
"$PYTHON" -m pip install --upgrade twine -q
"$PYTHON" -m twine check "$WHEEL"

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    echo "=== Done (check only) ==="
    exit 0
fi

# Upload
echo ""
echo "[2/2] Uploading..."

if [[ $TEST -eq 1 ]]; then
    echo "Uploading to TestPyPI..."
    "$PYTHON" -m twine upload --repository testpypi "$WHEEL"
    echo ""
    echo "=== Done! ==="
    echo "TestPyPI URL: https://test.pypi.org/project/moxing/"
else
    echo "Uploading to PyPI..."
    "$PYTHON" -m twine upload "$WHEEL"
    echo ""
    echo "=== Done! ==="
    echo "PyPI URL: https://pypi.org/project/moxing/"
fi

echo ""
echo "Version: $VERSION"
echo "Wheel: $WHEEL"