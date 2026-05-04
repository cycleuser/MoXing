#!/usr/bin/env python3
"""
Build moxing wheels with pre-bundled llama.cpp binaries.

This script creates platform-specific wheels that include the llama.cpp binaries,
allowing offline installation without downloading binaries at runtime.

Usage:
    python scripts/build_with_binaries.py --backend metal --platform darwin --arch arm64
    python scripts/build_with_binaries.py --backend cuda --platform linux --arch x64
    python scripts/build_with_binaries.py --backend vulkan --platform windows --arch x64

The resulting wheel will be named like:
    moxing-0.1.3-cp312-cp312-macosx_11_0_arm64.whl
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from urllib.request import urlopen, Request

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
PROJECT_ROOT = Path(__file__).parent.parent


def get_latest_release():
    """Get the latest release info from GitHub API."""
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())


def find_asset(release, platform, arch, backend):
    """Find the appropriate asset for the given configuration."""
    platform_map = {
        "darwin": ["macos", "darwin", "osx"],
        "linux": ["linux", "ubuntu"],
        "windows": ["win", "windows", "msvc"],
    }
    
    arch_map = {
        "arm64": ["arm64", "aarch64"],
        "x64": ["x64", "x86_64", "amd64"],
    }
    
    backend_map = {
        "metal": ["metal", "apple"],
        "cuda": ["cuda", "cu", "gpu"],
        "vulkan": ["vulkan"],
        "cpu": ["cpu", "noavx"],
    }
    
    platform_pats = platform_map.get(platform, [])
    arch_pats = arch_map.get(arch, [])
    
    for asset in release["assets"]:
        name = asset["name"].lower()
        
        if not any(p in name for p in platform_pats):
            continue
        
        if arch_pats and not any(p in name for p in arch_pats):
            continue
        
        if backend != "auto":
            backend_pats = backend_map.get(backend, [])
            if backend_pats and not any(p in name for p in backend_pats):
                continue
        
        if name.endswith((".zip", ".tar.gz", ".tgz")):
            return asset
    
    return None


def download_and_extract(asset, dest_dir):
    """Download and extract binaries from an asset."""
    import tarfile
    import zipfile
    
    print(f"Downloading: {asset['name']}")
    
    temp_path = dest_dir / asset["name"]
    req = Request(asset["browser_download_url"])
    
    with urlopen(req, timeout=300) as response:
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = response.read(8192 * 16)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\rProgress: {downloaded/total*100:.1f}%", end="")
    print()
    
    extract_dir = dest_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    if asset["name"].endswith(".zip"):
        with zipfile.ZipFile(temp_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        with tarfile.open(temp_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    
    return extract_dir


def build_wheel(platform, arch, backend, output_dir):
    """Build a wheel with bundled binaries."""
    print(f"\nBuilding wheel for {platform}-{arch} ({backend})")
    
    release = get_latest_release()
    print(f"Using llama.cpp release: {release['tag_name']}")
    
    asset = find_asset(release, platform, arch, backend)
    if not asset:
        print(f"ERROR: No binary found for {platform}-{arch} ({backend})")
        return None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        extract_dir = download_and_extract(asset, tmpdir)
        
        bin_dest = PROJECT_ROOT / "moxing" / "bin" / platform
        bin_dest.mkdir(parents=True, exist_ok=True)
        
        for item in extract_dir.rglob("*"):
            if item.is_file():
                name = item.name
                if name.startswith("llama-") or name.endswith((".dylib", ".so", ".dll")):
                    dest = bin_dest / name
                    shutil.copy2(item, dest)
                    if not name.endswith((".dylib", ".so", ".dll")):
                        os.chmod(dest, 0o755)
                    print(f"  Bundled: {name}")
        
        for link in extract_dir.rglob("*"):
            if link.is_symlink():
                dest = bin_dest / link.name
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                os.symlink(os.readlink(link), dest)
                print(f"  Linked: {link.name}")
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return None
        
        for f in Path(output_dir).glob("*.whl"):
            print(f"Created: {f}")
            return f
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Build moxing with bundled binaries")
    parser.add_argument("--backend", "-b", choices=["metal", "cuda", "vulkan", "cpu", "auto"], 
                       default="auto", help="GPU backend")
    parser.add_argument("--platform", "-p", choices=["darwin", "linux", "windows"],
                       default=None, help="Target platform (default: current)")
    parser.add_argument("--arch", "-a", choices=["arm64", "x64"],
                       default=None, help="Target architecture (default: current)")
    parser.add_argument("--output", "-o", default="dist",
                       help="Output directory for wheels")
    parser.add_argument("--all", action="store_true",
                       help="Build wheels for all platforms")
    
    args = parser.parse_args()
    
    import platform
    current_platform = args.platform or ("darwin" if sys.platform == "darwin" else 
                                          "windows" if sys.platform == "win32" else "linux")
    current_arch = args.arch or ("arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64")
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.all:
        configs = [
            ("darwin", "arm64", "metal"),
            ("darwin", "x64", "metal"),
            ("linux", "x64", "cuda"),
            ("linux", "x64", "vulkan"),
            ("linux", "arm64", "cpu"),
            ("windows", "x64", "cuda"),
            ("windows", "x64", "vulkan"),
            ("windows", "x64", "cpu"),
        ]
        
        for p, a, b in configs:
            build_wheel(p, a, b, output_dir)
    else:
        build_wheel(current_platform, current_arch, args.backend, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()