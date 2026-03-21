#!/usr/bin/env bash
# MoXing - Upload binaries to GitHub releases
#
# This script uploads the packaged binaries to GitHub releases.
# Run generate_binaries.sh first to download and package the binaries.
#
# Usage:
#   ./upload_binaries.sh              # Upload to GitHub
#   ./upload_binaries.sh --package    # Package only, don't upload
#
# Prerequisites:
#   gh auth login
#
# Release URL: https://github.com/cycleuser/MoXing/releases/tag/binaries

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
MOXING_REPO="cycleuser/MoXing"
RELEASE_TAG="binaries"
DIST_DIR="dist/binaries"

# Parse arguments
PACKAGE_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --package|-p)
            PACKAGE_ONLY=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --package, -p    Package binaries only, don't upload"
            echo "  --help, -h       Show this help"
            echo ""
            echo "Prerequisites:"
            echo "  gh auth login"
            echo ""
            echo "Release URL: https://github.com/$MOXING_REPO/releases/tag/$RELEASE_TAG"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== MoXing Binary Uploader ==="

# Check if binaries exist
if [[ ! -d "$DIST_DIR" ]]; then
    echo "Error: No packaged binaries found."
    echo "Run generate_binaries.sh first."
    exit 1
fi

# Package binaries
echo "[1/3] Packaging binaries..."
"$PYTHON" scripts/upload_binaries.py --package

if [[ $PACKAGE_ONLY -eq 1 ]]; then
    echo ""
    echo "=== Done (package only) ==="
    echo "Packages in: $DIST_DIR"
    ls -la "$DIST_DIR"
    exit 0
fi

# Check for gh CLI
echo "[2/3] Checking GitHub CLI..."
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) not installed."
    echo "Install from: https://cli.github.com/"
    echo ""
    echo "Then run: gh auth login"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub."
    echo "Run: gh auth login"
    exit 1
fi

echo "  GitHub CLI is ready"

# Upload to GitHub
echo "[3/3] Uploading to GitHub..."

# Get llama.cpp version from first VERSION file
LLAMA_VERSION=""
for vfile in moxing/bin/*/VERSION; do
    if [[ -f "$vfile" ]]; then
        LLAMA_VERSION=$(head -n1 "$vfile")
        break
    fi
done

# Get moxing version
MOXING_VERSION=$("$PYTHON" -c "exec(open('moxing/__init__.py').read()); print(__version__)")

# Create release notes
RELEASE_NOTES="llama.cpp: $LLAMA_VERSION
moxing: $MOXING_VERSION

Pre-built llama.cpp binaries for MoXing.

## Supported Platforms

| Platform | CPU | CUDA | Vulkan | ROCm | Metal |
|----------|-----|------|--------|------|-------|
| Linux x64 | ✅ | ✅ | ✅ | ✅ | - |
| Windows x64 | ✅ | ✅ | ✅ | - | - |
| macOS ARM64 | ✅ | - | - | - | ✅ |

## Installation

\`\`\`bash
pip install moxing
moxing serve model.gguf  # Binaries download automatically
\`\`\`

## Manual Download

Download the archive for your platform and extract to:
- Linux/macOS: \`~/.cache/moxing/binaries/{platform}-{backend}/\`
- Windows: \`%USERPROFILE%\\.cache\\moxing\\binaries\\{platform}-{backend}\\\`
"

# Check if release exists
if gh release view "$RELEASE_TAG" --repo "$MOXING_REPO" &> /dev/null; then
    echo "  Release $RELEASE_TAG exists, updating..."
    
    # Update release notes
    gh release edit "$RELEASE_TAG" --repo "$MOXING_REPO" --notes "$RELEASE_NOTES" || true
else
    echo "  Creating release $RELEASE_TAG..."
    gh release create "$RELEASE_TAG" --repo "$MOXING_REPO" --title "Binaries" --notes "$RELEASE_NOTES"
fi

# Upload assets
for archive_path in "$DIST_DIR"/*; do
    if [[ -f "$archive_path" ]]; then
        filename=$(basename "$archive_path")
        echo ""
        echo "  Uploading: $filename"
        gh release upload "$RELEASE_TAG" "$archive_path" --repo "$MOXING_REPO" --clobber || echo "    [warning] Failed to upload $filename"
    fi
done

echo ""
echo "=== Done! ==="
echo ""
echo "Release URL: https://github.com/$MOXING_REPO/releases/tag/$RELEASE_TAG"
echo ""
echo "llama.cpp version: $LLAMA_VERSION"
echo "moxing version: $MOXING_VERSION"