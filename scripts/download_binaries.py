#!/usr/bin/env python3
"""
Download pre-built llama.cpp binaries from GitHub releases.
This script downloads binaries for all platforms (Windows, Linux, macOS).
"""

import os
import sys
import json
import argparse
import tempfile
import tarfile
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
BINARY_DIR = Path(__file__).parent.parent / "moxing" / "bin"

ESSENTIAL_BINARIES = [
    "llama-server",
    "llama-cli",
    "llama-bench",
    "llama-quantize",
]


def get_latest_release():
    """Get the latest release info from GitHub API."""
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    
    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data
    except (URLError, HTTPError) as e:
        print(f"Error fetching release info: {e}")
        return None


def get_release_by_tag(tag):
    """Get release info by tag name."""
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/tags/{tag}"
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    
    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data
    except (URLError, HTTPError) as e:
        print(f"Error fetching release {tag}: {e}")
        return None


def find_asset_for_platform(assets, platform, gpu_backend="cpu"):
    """Find the appropriate asset for a platform."""
    platform_patterns = {
        "windows": ["win", "windows", "msvc"],
        "linux": ["linux", "ubuntu"],
        "darwin": ["macos", "darwin", "osx"],
    }
    
    backend_patterns = {
        "cpu": ["cpu", "noavx"],
        "cuda": ["cuda", "cu", "gpu"],
        "vulkan": ["vulkan"],
        "metal": ["metal", "m1", "m2", "m3", "apple"],
    }
    
    patterns = platform_patterns.get(platform, [])
    backend_pats = backend_patterns.get(gpu_backend, [])
    
    for asset in assets:
        name = asset["name"].lower()
        
        if not any(p in name for p in patterns):
            continue
        
        if gpu_backend != "auto" and backend_pats:
            if not any(p in name for p in backend_pats):
                continue
        
        if name.endswith((".zip", ".tar.gz", ".tgz")):
            return asset
    
    return None


def download_file(url, dest):
    """Download a file with progress."""
    print(f"Downloading: {url}")
    
    req = Request(url)
    req.add_header("Accept", "application/octet-stream")
    
    try:
        with urlopen(req, timeout=300) as response:
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192 * 16
            
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = (downloaded / total) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")
            
            print()
            return True
    except (URLError, HTTPError) as e:
        print(f"Error downloading: {e}")
        return False


def extract_binaries(archive_path, dest_dir, platform):
    """Extract binaries from archive."""
    print(f"Extracting: {archive_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(tmpdir)
        elif archive_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(tmpdir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if platform == "windows":
                    if f.endswith(".exe"):
                        src = Path(root) / f
                        dst = dest_dir / f
                        shutil.copy2(src, dst)
                        print(f"  Extracted: {f}")
                        
                        dll_files = list(Path(root).glob("*.dll"))
                        for dll in dll_files:
                            shutil.copy2(dll, dest_dir / dll.name)
                            print(f"  Extracted: {dll.name}")
                else:
                    if f.startswith("llama-") or f == "main":
                        src = Path(root) / f
                        if os.access(src, os.X_OK) or f.startswith("llama-"):
                            dst = dest_dir / f
                            shutil.copy2(src, dst)
                            os.chmod(dst, 0o755)
                            print(f"  Extracted: {f}")
        
        return True


def download_for_platform(platform, gpu_backend, release, dest_dir):
    """Download binaries for a specific platform."""
    asset = find_asset_for_platform(release["assets"], platform, gpu_backend)
    
    if not asset:
        print(f"No asset found for {platform} ({gpu_backend})")
        return False
    
    print(f"Found asset: {asset['name']}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / asset["name"]
        
        if not download_file(asset["browser_download_url"], archive_path):
            return False
        
        return extract_binaries(str(archive_path), dest_dir, platform)


def main():
    parser = argparse.ArgumentParser(description="Download llama.cpp binaries")
    parser.add_argument(
        "--platform", "-p",
        choices=["windows", "linux", "darwin", "all"],
        default="all",
        help="Platform to download for (default: all)"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["cpu", "cuda", "vulkan", "metal", "auto"],
        default="auto",
        help="GPU backend (default: auto - prefer GPU if available)"
    )
    parser.add_argument(
        "--tag", "-t",
        default=None,
        help="Release tag to download (default: latest)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: moxing/bin/<platform>)"
    )
    
    args = parser.parse_args()
    
    print("Fetching release info...")
    if args.tag:
        release = get_release_by_tag(args.tag)
    else:
        release = get_latest_release()
    
    if not release:
        print("Failed to get release info")
        return 1
    
    print(f"Release: {release['tag_name']}")
    print(f"Published: {release['published_at']}")
    
    platforms = ["windows", "linux", "darwin"] if args.platform == "all" else [args.platform]
    
    backends = {
        "windows": ["vulkan", "cuda", "cpu"],
        "linux": ["vulkan", "cuda", "cpu"],
        "darwin": ["metal", "cpu"],
    }
    
    success = True
    for platform in platforms:
        print(f"\n{'='*50}")
        print(f"Downloading for {platform}")
        print('='*50)
        
        dest_dir = Path(args.output) if args.output else BINARY_DIR / platform
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        backend_order = backends.get(platform, ["cpu"])
        if args.backend != "auto":
            backend_order = [args.backend]
        
        downloaded = False
        for backend in backend_order:
            print(f"\nTrying backend: {backend}")
            if download_for_platform(platform, backend, release, dest_dir):
                downloaded = True
                break
        
        if not downloaded:
            print(f"Failed to download binaries for {platform}")
            success = False
    
    if success:
        print("\n" + "="*50)
        print("All binaries downloaded successfully!")
        print("="*50)
        return 0
    else:
        print("\nSome downloads failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())