#!/usr/bin/env bash
# MoXing - Generate wheel package
#
# Usage:
#   ./generate_wheel.sh              # Build wheel with current version
#   ./generate_wheel.sh --version 0.2.0   # Set specific version
#
# Output: dist/moxing-{version}-py3-none-any.whl

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
VERSION_FILE="moxing/__init__.py"

# Parse arguments
VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --version|-v)
            VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version, -v VERSION   Set specific version"
            echo "  --help, -h              Show this help"
            echo ""
            echo "Output: dist/moxing-{version}-py3-none-any.whl"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== MoXing Wheel Generator ==="

# Set version if specified
if [[ -n "$VERSION" ]]; then
    echo "[1/4] Setting version to $VERSION..."
    sed -i "s/__version__ = .*/__version__ = \"$VERSION\"/" "$VERSION_FILE"
else
    echo "[1/4] Using current version..."
fi

# Read current version
CURRENT_VERSION=$("$PYTHON" -c "exec(open('$VERSION_FILE').read()); print(__version__)")
echo "  Version: $CURRENT_VERSION"

echo "[2/4] Cleaning old builds..."
rm -rf dist/*.whl dist/*.tar.gz build/ *.egg-info moxing.egg-info

echo "[3/4] Installing build tools..."
"$PYTHON" -m pip install --upgrade build -q

echo "[4/4] Building wheel..."
"$PYTHON" scripts/build_platform_wheels.py

if [[ $? -ne 0 ]]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "=== Done! ==="
echo "Version: $CURRENT_VERSION"
echo "Wheel: dist/moxing-$CURRENT_VERSION-py3-none-any.whl"
echo ""
echo "To upload to PyPI:"
echo "  ./upload_pypi.sh"