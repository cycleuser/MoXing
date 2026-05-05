#!/usr/bin/env bash
# MoXing - Build and upload to PyPI
# Usage: ./upload_pypi.sh [--test|-t] [--check|-c] [--no-bump] [--help|-h]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
VERSION_FILE="moxing/__init__.py"
TWINE_REPO="pypi"
BUMP=1 CHECK=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --test|-t)   TWINE_REPO="testpypi"; shift ;;
        --check|-c)  CHECK=1; shift ;;
        --no-bump)   BUMP=0; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "  --test, -t   Upload to TestPyPI"
            echo "  --check, -c  Build + check only, skip upload"
            echo "  --no-bump    Skip version bump"
            echo "  --help, -h   Show this help"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=== MoXing PyPI Upload ==="

# [1] Version
if (( BUMP )); then
    echo "[1/4] Bumping patch version..."
    "$PYTHON" -c "
import re, sys
p = '$VERSION_FILE'
t = open(p, encoding='utf-8').read()
m = re.search(r'(__version__\s*=\s*\"(\d+\.\d+\.)(\d+)\")', t)
if not m: print('ERROR: cannot parse version'); sys.exit(1)
old = m.group(2) + m.group(3)
new = m.group(2) + str(int(m.group(3)) + 1)
open(p, 'w', encoding='utf-8').write(t.replace(m.group(1), '__version__ = \"' + new + '\"'))
print(f'  {old} -> {new}')
"
else
    CUR=$("$PYTHON" -c "import re; t=open('$VERSION_FILE').read(); m=re.search(r'__version__\s*=\s*\"(\S+)\"', t); print(m.group(1))")
    echo "[1/4] Version: $CUR (no bump)"
fi

# [2] Clean + build
echo "[2/4] Cleaning..."
rm -rf dist/ build/ *.egg-info moxing.egg-info

echo "[3/4] Building..."
"$PYTHON" -m pip install --upgrade build twine -q
"$PYTHON" -m build
"$PYTHON" -m twine check dist/*

WHEEL=$(ls -t dist/*.whl 2>/dev/null | head -1)
VER=$(basename "$WHEEL" | sed 's/moxing-\(.*\)-py3-none-any.whl/\1/')
echo "  $WHEEL ($(du -h "$WHEEL" | cut -f1))"

if (( CHECK )); then
    echo "=== Check done (v$VER) ==="
    exit 0
fi

# [4] Upload
echo "[4/4] Uploading to ${TWINE_REPO}..."
if [[ "$TWINE_REPO" == "testpypi" ]]; then
    "$PYTHON" -m twine upload --repository testpypi dist/*
else
    "$PYTHON" -m twine upload dist/*
fi

echo "=== Done (v$VER) ==="
if [[ "$TWINE_REPO" == "testpypi" ]]; then
    echo "TestPyPI: https://test.pypi.org/project/moxing/"
else
    echo "PyPI:    https://pypi.org/project/moxing/"
fi
