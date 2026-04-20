#!/usr/bin/env bash
# MoXing - Build and upload to PyPI
# Auto-bumps patch version, builds wheel, and uploads to PyPI.
#
# Usage:
#   ./upload_pypi.sh              # Build + upload to PyPI
#   ./upload_pypi.sh --test       # Build + upload to TestPyPI
#   ./upload_pypi.sh --check      # Build + check wheel only, don't upload
#   ./upload_pypi.sh --no-bump    # Skip version bump (use current version)
#
# Prerequisites:
#   pip install build twine
#   Set TWINE_USERNAME and TWINE_PASSWORD (or use API token)

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
VERSION_FILE="moxing/__init__.py"

# Parse arguments
TEST=0
CHECK_ONLY=0
NO_BUMP=0

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
        --no-bump)
            NO_BUMP=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test, -t      Upload to TestPyPI instead of PyPI"
            echo "  --check, -c     Build + check wheel only, don't upload"
            echo "  --no-bump       Skip version bump (use current version)"
            echo "  --help, -h      Show this help"
            echo ""
            echo "Prerequisites:"
            echo "  pip install build twine"
            echo "  Set TWINE_USERNAME and TWINE_PASSWORD (or use API token)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== MoXing PyPI Upload ==="

# [1/5] Bump patch version
if [[ $NO_BUMP -eq 0 ]]; then
    echo "[1/5] Bumping patch version..."
    "$PYTHON" -c "
import re, sys
p = '$VERSION_FILE'
t = open(p, encoding='utf-8').read()
m = re.search(r'(__version__\s*=\s*\"(\d+\.\d+\.)(\d+)\")', t)
if not m:
    print('ERROR: cannot parse version')
    sys.exit(1)
old_v = m.group(2) + m.group(3)
new_v = m.group(2) + str(int(m.group(3)) + 1)
open(p, 'w', encoding='utf-8').write(t.replace(m.group(1), '__version__ = \"' + new_v + '\"'))
print(f'  {old_v} -> {new_v}')
"
else
    echo "[1/5] Skipping version bump (--no-bump)"
    CURRENT_VERSION=$("$PYTHON" -c "
import re
t = open('$VERSION_FILE', encoding='utf-8').read()
m = re.search(r'__version__\s*=\s*\"(\d+\.\d+\.\d+)\"', t)
print(m.group(1) if m else 'unknown')
")
    echo "  Current version: $CURRENT_VERSION"
fi

# [2/5] Clean old builds
echo "[2/5] Cleaning old builds..."
rm -rf dist/ build/ *.egg-info moxing.egg-info

# [3/5] Install build tools
echo "[3/5] Installing build tools..."
"$PYTHON" -m pip install --upgrade build twine -q

# [4/5] Build package
echo "[4/5] Building package..."
"$PYTHON" -m build
"$PYTHON" -m twine check dist/*

# Get version from built wheel
WHEEL=$(ls -t dist/*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
    echo "Error: No wheel found in dist/"
    exit 1
fi
VERSION=$(basename "$WHEEL" | sed 's/moxing-\(.*\)-py3-none-any.whl/\1/')
echo "  Built: $WHEEL ($(du -h "$WHEEL" | cut -f1))"
echo "  Version: $VERSION"

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    echo "=== Done (check only) ==="
    exit 0
fi

# [5/5] Upload
echo "[5/5] Uploading..."

if [[ $TEST -eq 1 ]]; then
    echo "Uploading to TestPyPI..."
    "$PYTHON" -m twine upload --repository testpypi dist/*
    echo ""
    echo "=== Done! ==="
    echo "TestPyPI URL: https://test.pypi.org/project/moxing/"
else
    echo "Uploading to PyPI..."
    "$PYTHON" -m twine upload dist/*
    echo ""
    echo "=== Done! ==="
    echo "PyPI URL: https://pypi.org/project/moxing/"
fi

echo "Version: $VERSION"
