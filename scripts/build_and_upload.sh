#!/bin/bash
#
# moxing-server Build and Upload Script for Linux/macOS
#
# Usage:
#   ./scripts/build_and_upload.sh --all
#   ./scripts/build_and_upload.sh --test
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}  moxing-server Build and Upload Script${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Parse arguments
BUILD=0
UPLOAD=0
CLEAN=0
CHECK=0
TEST=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CLEAN=1
            BUILD=1
            CHECK=1
            UPLOAD=1
            shift
            ;;
        --build)
            BUILD=1
            shift
            ;;
        --upload)
            UPLOAD=1
            shift
            ;;
        --check)
            CHECK=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --test)
            TEST=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all      Clean, build, check and upload"
            echo "  --build    Build the package"
            echo "  --upload   Upload to PyPI"
            echo "  --check    Check the package"
            echo "  --clean    Clean build artifacts"
            echo "  --test     Upload to TestPyPI instead of PyPI"
            echo "  --help     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# If no options specified, show help
if [[ $BUILD -eq 0 && $UPLOAD -eq 0 && $CHECK -eq 0 && $CLEAN -eq 0 ]]; then
    echo "No action specified. Use --help for usage."
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo ""

# Clean
if [[ $CLEAN -eq 1 ]]; then
    echo -e "${BLUE}Cleaning build artifacts...${NC}"
    rm -rf "$PROJECT_DIR/build"
    rm -rf "$PROJECT_DIR/dist"
    rm -rf "$PROJECT_DIR/moxing_server.egg-info"
    find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}Clean complete${NC}"
    echo ""
fi

# Build
if [[ $BUILD -eq 1 ]]; then
    echo -e "${BLUE}Building package...${NC}"
    
    # Check if build is installed
    if ! python3 -c "import build" 2>/dev/null; then
        echo "Installing build dependency..."
        pip3 install build
    fi
    
    cd "$PROJECT_DIR"
    python3 -m build
    
    echo -e "${GREEN}Build complete!${NC}"
    echo ""
    echo "Built packages:"
    ls -1 "$PROJECT_DIR/dist"
    echo ""
fi

# Check
if [[ $CHECK -eq 1 ]]; then
    echo -e "${BLUE}Checking package...${NC}"
    
    # Check if twine is installed
    if ! python3 -c "import twine" 2>/dev/null; then
        echo "Installing twine..."
        pip3 install twine
    fi
    
    cd "$PROJECT_DIR"
    python3 -m twine check dist/*
    
    echo -e "${GREEN}Check passed${NC}"
    echo ""
fi

# Upload
if [[ $UPLOAD -eq 1 ]]; then
    echo -e "${BLUE}Uploading...${NC}"
    
    # Check if dist exists
    if [[ ! -f "$PROJECT_DIR/dist/"* ]]; then
        echo -e "${RED}No dist files found. Run --build first.${NC}"
        exit 1
    fi
    
    # Check if twine is installed
    if ! python3 -c "import twine" 2>/dev/null; then
        echo "Installing twine..."
        pip3 install twine
    fi
    
    cd "$PROJECT_DIR"
    
    if [[ $TEST -eq 1 ]]; then
        echo "Uploading to TestPyPI..."
        python3 -m twine upload --repository testpypi dist/*
    else
        echo "Uploading to PyPI..."
        python3 -m twine upload dist/*
    fi
    
    echo -e "${GREEN}Upload complete!${NC}"
fi

echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}  Done!${NC}"
echo -e "${CYAN}============================================================${NC}"