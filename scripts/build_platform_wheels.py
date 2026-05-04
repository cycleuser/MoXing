#!/usr/bin/env python3
"""
Build a small wheel for MoXing.

The wheel is small (~100 KB) without bundled binaries.
Binaries are downloaded from GitHub on first use.

PyPI file size limit is ~60 MB, so we can't bundle binaries.

Usage:
    python scripts/build_platform_wheels.py
    twine upload dist/*.whl
"""

import sys
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
DIST_DIR = REPO_ROOT / "dist"


def build_wheel():
    """Build a small wheel without bundled binaries."""
    
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean old wheels
    for f in DIST_DIR.glob("*.whl"):
        f.unlink()
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Copy moxing package without bin
        src_dir = REPO_ROOT / "moxing"
        dst_dir = tmpdir / "moxing"
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("bin"))
        
        # Copy pyproject.toml
        shutil.copy2(REPO_ROOT / "pyproject.toml", tmpdir / "pyproject.toml")
        
        # Build wheel
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(DIST_DIR)],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return None
    
    wheels = list(DIST_DIR.glob("*.whl"))
    if wheels:
        wheel = wheels[0]
        size_kb = wheel.stat().st_size / 1024
        print(f"\nCreated: {wheel.name}")
        print(f"Size: {size_kb:.1f} KB")
        print("\nUpload to PyPI:")
        print("  twine upload dist/*.whl")
        print("\nUsage:")
        print("  pip install moxing")
        print("  moxing serve model.gguf  # Downloads binaries on first run")
        return wheel
    
    return None


if __name__ == "__main__":
    build_wheel()