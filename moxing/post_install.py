#!/usr/bin/env python3
"""
Post-install script for moxing.

This script is called after pip install to pre-download llama.cpp binaries.
It uses the backend specified via environment variable or detects automatically.

Usage:
    MOXING_BACKEND=metal pip install moxing[metal]
    MOXING_BACKEND=cuda pip install moxing[cuda]
"""

import os
import sys
import platform
from pathlib import Path


def get_backend_from_env():
    """Get backend from environment variable."""
    return os.environ.get("MOXING_BACKEND", "auto").lower()


def get_default_backend():
    """Detect default backend for current system."""
    if sys.platform == "darwin":
        return "metal"
    elif sys.platform == "win32":
        return "vulkan"
    else:
        return "vulkan"


def download_binaries(backend="auto"):
    """Download llama.cpp binaries."""
    try:
        # Add the package path
        import moxing
        from moxing.binaries import get_binary_manager
        
        manager = get_binary_manager()
        
        if manager.is_downloaded():
            print("Binaries already downloaded.")
            return True
        
        print(f"Downloading llama.cpp binaries for {manager.platform}...")
        manager.download_binaries(backend=backend, force=False)
        print("Download complete!")
        return True
        
    except Exception as e:
        print(f"Warning: Could not download binaries: {e}")
        print("Binaries will be downloaded on first use.")
        return False


def main():
    """Main entry point."""
    backend = get_backend_from_env()
    if backend == "auto":
        backend = get_default_backend()
    
    print(f"MoXing post-install: setting up {backend} backend...")
    download_binaries(backend)


if __name__ == "__main__":
    main()