#!/usr/bin/env python3
"""
Build wheels for MoXing - one wheel per OS/arch with all backends bundled.

PyPI requires standard platform tags. We build:
- moxing-VERSION-py3-none-manylinux_2_17_x86_64.whl (Linux: CPU + Vulkan + ROCm + CUDA)
- moxing-VERSION-py3-none-win_amd64.whl (Windows: CPU + CUDA + Vulkan)
- moxing-VERSION-py3-none-macosx_11_0_arm64.whl (macOS: Metal)

Each wheel auto-detects the best backend at runtime.

Usage:
    python scripts/build_platform_wheels.py              # Build all
    python scripts/build_platform_wheels.py --platform linux-x64
    python scripts/build_platform_wheels.py --list
"""

import os
import sys
import argparse
import shutil
import subprocess
import tempfile
import zipfile
import hashlib
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict


REPO_ROOT = Path(__file__).parent.parent
DIST_DIR = REPO_ROOT / "dist"
BIN_DIR = REPO_ROOT / "moxing" / "bin"


@dataclass
class PlatformConfig:
    name: str
    os: str
    arch: str
    wheel_tag: str
    backends: List[str]
    description: str


PLATFORMS = [
    PlatformConfig(
        name="linux-x64",
        os="linux", arch="x64",
        wheel_tag="manylinux_2_17_x86_64",
        backends=["cpu", "vulkan", "rocm", "cuda"],
        description="Linux x64 (CPU, Vulkan, ROCm, CUDA)"
    ),
    PlatformConfig(
        name="windows-x64",
        os="windows", arch="x64",
        wheel_tag="win_amd64",
        backends=["cpu", "cuda", "vulkan"],
        description="Windows x64 (CPU, CUDA, Vulkan)"
    ),
    PlatformConfig(
        name="darwin-arm64",
        os="darwin", arch="arm64",
        wheel_tag="macosx_11_0_arm64",
        backends=["metal"],
        description="macOS ARM64 (Metal)"
    ),
]


def fix_wheel_platform(wheel_path: Path, platform_tag: str) -> Path:
    """Update the WHEEL file inside the wheel with the correct platform tag."""
    version = "0.1.7"
    new_name = f"moxing-{version}-py3-none-{platform_tag}.whl"
    new_path = wheel_path.parent / new_name
    dist_info = f"moxing-{version}.dist-info"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract wheel
        with zipfile.ZipFile(wheel_path, 'r') as zf:
            zf.extractall(tmpdir)
        
        # Update WHEEL file
        wheel_file = tmpdir / dist_info / "WHEEL"
        wheel_content = f"""Wheel-Version: 1.0
Generator: setuptools (82.0.1)
Root-Is-Purelib: true
Tag: py3-none-{platform_tag}
"""
        wheel_file.write_text(wheel_content)
        
        # Regenerate RECORD file
        record_file = tmpdir / dist_info / "RECORD"
        record_lines = []
        
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                file_path = Path(root) / file
                if file == "RECORD":
                    continue
                arcname = file_path.relative_to(tmpdir)
                
                # Calculate SHA256 hash
                with open(file_path, 'rb') as f:
                    content = f.read()
                    digest = hashlib.sha256(content).digest()
                    hash_str = "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
                
                record_lines.append(f"{arcname},{hash_str},{len(content)}")
        
        record_lines.append(f"{dist_info}/RECORD,,")
        record_file.write_text("\n".join(record_lines))
        
        # Recreate wheel
        with zipfile.ZipFile(new_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(tmpdir)
                    zf.write(file_path, arcname)
        
        # Remove old wheel
        wheel_path.unlink()
        
        return new_path


def build_platform_wheel(config: PlatformConfig) -> Optional[Path]:
    """Build a wheel for a specific platform with all available backends."""
    
    # Check available backends
    available = []
    for backend in config.backends:
        bin_dir = BIN_DIR / f"{config.name}-{backend}"
        if bin_dir.exists() and (bin_dir / "VERSION").exists():
            available.append(backend)
    
    if not available:
        print(f"No binaries for {config.name}, skipping")
        return None
    
    print(f"\nBuilding: {config.name}")
    print(f"  Backends: {', '.join(available)}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Copy moxing package without bin
        src_dir = REPO_ROOT / "moxing"
        dst_dir = tmpdir / "moxing"
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("bin"))
        
        # Copy all available backend binaries
        dst_bin = dst_dir / "bin"
        dst_bin.mkdir(parents=True, exist_ok=True)
        
        for backend in available:
            bin_dir = BIN_DIR / f"{config.name}-{backend}"
            dst_backend = dst_bin / f"{config.name}-{backend}"
            shutil.copytree(bin_dir, dst_backend)
        
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
            print(f"  Build failed: {result.stderr}")
            return None
        
        # Fix the wheel with correct platform tag
        version = "0.1.7"
        old_wheel = DIST_DIR / f"moxing-{version}-py3-none-any.whl"
        if old_wheel.exists():
            new_wheel = fix_wheel_platform(old_wheel, config.wheel_tag)
            
            size_mb = new_wheel.stat().st_size / (1024 * 1024)
            print(f"  Created: {new_wheel.name} ({size_mb:.1f} MB)")
            return new_wheel
    
    return None


def build_all_wheels(platforms: Optional[List[str]]) -> Dict[str, Path]:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean old wheels
    for f in DIST_DIR.glob("*.whl"):
        f.unlink()
    
    configs = PLATFORMS
    if platforms:
        configs = [c for c in configs if c.name in platforms]
    
    print(f"Building {len(configs)} wheel(s)")
    print(f"Output: {DIST_DIR}")
    
    results = {}
    for config in configs:
        wheel = build_platform_wheel(config)
        if wheel:
            results[config.name] = wheel
    
    # Summary
    print(f"\n{'='*60}")
    print("Build Summary")
    print('='*60)
    
    total_size = 0
    for name, wheel in results.items():
        size = wheel.stat().st_size / (1024 * 1024)
        total_size += size
        print(f"  {wheel.name} ({size:.1f} MB)")
    
    print(f"\nTotal: {len(results)} wheels, {total_size:.1f} MB")
    print("\nUpload to PyPI:")
    print("  twine upload dist/*.whl")
    print("\nInstall:")
    print("  pip install moxing    # Auto-detects best backend")
    return results


def main():
    parser = argparse.ArgumentParser(description="Build platform-specific wheels")
    parser.add_argument("--platform", "-p", default=None, help="Specific platform(s)")
    parser.add_argument("--list", "-l", action="store_true", help="List platforms")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable wheels:")
        print("-" * 60)
        for c in PLATFORMS:
            print(f"  {c.name:<15} {c.description}")
            print(f"                  -> moxing-VERSION-py3-none-{c.wheel_tag}.whl")
        
        print("\nBackend auto-detection:")
        print("  Linux:   CUDA > Vulkan > ROCm > CPU")
        print("  Windows: CUDA > Vulkan > CPU")
        print("  macOS:   Metal > CPU")
        return
    
    platforms = args.platform.split(",") if args.platform else None
    build_all_wheels(platforms)


if __name__ == "__main__":
    main()