#!/usr/bin/env python3
"""
Bundle ALL llama.cpp binaries for ALL platforms into moxing/bin/.

Usage:
    python scripts/bundle_all_binaries.py              # Download all
    python scripts/bundle_all_binaries.py --dry-run    # Preview
    python scripts/bundle_all_binaries.py --platform linux-x64-cpu  # Specific
"""

import os
import sys
import json
import argparse
import tarfile
import zipfile
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import HTTPError


REPO_ROOT = Path(__file__).parent.parent
LLAMA_CPP_REPO = "ggml-org/llama.cpp"
BIN_DIR = REPO_ROOT / "moxing" / "bin"

ESSENTIAL_BINARIES = ["llama-server", "llama-cli", "llama-bench", "llama-quantize", "llama-gguf-split"]


@dataclass
class PlatformConfig:
    name: str
    os: str
    arch: str
    backend: str
    asset_name: str
    description: str


ALL_PLATFORMS = [
    PlatformConfig(
        name="linux-x64-cpu",
        os="linux", arch="x64", backend="cpu",
        asset_name="bin-ubuntu-x64.tar.gz",
        description="Linux x64 CPU"
    ),
    PlatformConfig(
        name="linux-x64-vulkan",
        os="linux", arch="x64", backend="vulkan",
        asset_name="bin-ubuntu-vulkan-x64.tar.gz",
        description="Linux x64 Vulkan"
    ),
    PlatformConfig(
        name="linux-x64-rocm",
        os="linux", arch="x64", backend="rocm",
        asset_name="bin-ubuntu-rocm-7.2-x64.tar.gz",
        description="Linux x64 ROCm (AMD GPU)"
    ),
    PlatformConfig(
        name="windows-x64-cpu",
        os="windows", arch="x64", backend="cpu",
        asset_name="bin-win-cpu-x64.zip",
        description="Windows x64 CPU"
    ),
    PlatformConfig(
        name="windows-x64-cuda",
        os="windows", arch="x64", backend="cuda",
        asset_name="cudart-llama-bin-win-cuda-12.4-x64.zip",
        description="Windows x64 CUDA 12.4 (bundled runtime)"
    ),
    PlatformConfig(
        name="windows-x64-vulkan",
        os="windows", arch="x64", backend="vulkan",
        asset_name="bin-win-vulkan-x64.zip",
        description="Windows x64 Vulkan"
    ),
    PlatformConfig(
        name="darwin-arm64-metal",
        os="darwin", arch="arm64", backend="metal",
        asset_name="bin-macos-arm64.tar.gz",
        description="macOS ARM64 Metal"
    ),
    PlatformConfig(
        name="darwin-x64-metal",
        os="darwin", arch="x64", backend="metal",
        asset_name="bin-macos-x64.tar.gz",
        description="macOS x64 Metal"
    ),
]


def github_api(url: str, max_retries: int = 3) -> dict:
    try:
        result = subprocess.run(["gh", "api", url], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "moxing"})
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 403 and attempt < max_retries - 1:
                time.sleep(60 * (attempt + 1))
            else:
                raise
    raise Exception("Max retries exceeded")


def get_latest_version() -> str:
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
    return github_api(url)["tag_name"]


def find_asset(assets: List[dict], pattern: str) -> Optional[dict]:
    pattern_lower = pattern.lower()
    for asset in assets:
        if pattern_lower in asset["name"].lower():
            return asset
    return None


def download_file(url: str, dest: Path) -> bool:
    print(f"  Downloading: {url}")
    req = Request(url, headers={"User-Agent": "moxing"})
    try:
        with urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                while True:
                    chunk = resp.read(8192 * 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r  Progress: {downloaded*100//total}%", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def extract_all(archive: Path, dest: Path, config: PlatformConfig) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    extracted = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(tmpdir)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                try:
                    tf.extractall(tmpdir, filter='data')
                except TypeError:
                    tf.extractall(tmpdir)
        
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                src = Path(root) / f
                is_binary = any(f.startswith(b) for b in ESSENTIAL_BINARIES)
                is_lib = (
                    f.endswith((".dylib", ".so", ".dll")) or
                    ".so." in f or
                    ".dylib." in f
                )
                
                if is_binary or is_lib:
                    dst = dest / f
                    shutil.copy2(src, dst)
                    if is_binary and config.os != "windows":
                        dst.chmod(0o755)
                    extracted.append(f)
        
        for root, dirs, files in os.walk(tmpdir):
            for item in os.listdir(root):
                src = Path(root) / item
                if src.is_symlink():
                    dst = dest / item
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    os.symlink(os.readlink(src), dst)
                    extracted.append(f"{item} -> {os.readlink(src)}")
    
    return extracted


def bundle_platform(config: PlatformConfig, version: str, assets: List[dict]) -> bool:
    print(f"\n{'='*60}")
    print(f"Platform: {config.name}")
    print(f"Backend: {config.backend}")
    print(f"Description: {config.description}")
    print('='*60)
    
    asset = find_asset(assets, config.asset_name)
    if not asset:
        print(f"  No matching asset found for: {config.asset_name}")
        return False
    
    print(f"  Asset: {asset['name']}")
    
    dest_dir = BIN_DIR / config.name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / asset["name"]
        url = f"https://github.com/{LLAMA_CPP_REPO}/releases/download/{version}/{asset['name']}"
        
        if not download_file(url, archive):
            return False
        
        print("  Extracting...")
        extracted = extract_all(archive, dest_dir, config)
        for f in extracted[:15]:
            print(f"    {f}")
        if len(extracted) > 15:
            print(f"    ... and {len(extracted) - 15} more files")
    
    (dest_dir / "VERSION").write_text(f"{version}\n{config.backend}\n")
    print(f"  Done: {dest_dir}")
    return True


def bundle_all(version: str, platforms: Optional[List[str]] = None, dry_run: bool = False) -> Tuple[int, int]:
    configs = ALL_PLATFORMS
    if platforms:
        configs = [c for c in configs if c.name in platforms]
    
    print(f"\nBundling {len(configs)} platform(s)")
    print(f"Version: {version}")
    print(f"Target: {BIN_DIR}")
    
    if dry_run:
        print("\nWould download:")
        for c in configs:
            print(f"  - {c.name}: {c.description}")
            print(f"    Asset: {c.asset_name}")
        return 0, 0
    
    release_url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/tags/{version}"
    release = github_api(release_url)
    assets = release.get("assets", [])
    
    success = 0
    failed = 0
    
    for config in configs:
        try:
            if bundle_platform(config, version, assets):
                success += 1
            else:
                failed += 1
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success} succeeded, {failed} failed")
    
    if BIN_DIR.exists():
        total_size = sum(f.stat().st_size for f in BIN_DIR.rglob("*") if f.is_file())
        print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    
    return success, failed


def list_platforms():
    print("\nAvailable platforms:")
    for c in ALL_PLATFORMS:
        print(f"  {c.name:<25} {c.description}")
    print("\nNote: Linux CUDA binaries are NOT bundled (requires system CUDA)")
    print("      Users with NVIDIA GPUs on Linux should have CUDA installed.")


def main():
    parser = argparse.ArgumentParser(description="Bundle all llama.cpp binaries")
    parser.add_argument("--platform", "-p", default=None, help="Specific platform(s)")
    parser.add_argument("--version", "-v", default=None, help="llama.cpp version")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--list", "-l", action="store_true", help="List platforms")
    
    args = parser.parse_args()
    
    if args.list:
        list_platforms()
        return
    
    version = args.version or get_latest_version()
    platforms = args.platform.split(",") if args.platform else None
    
    bundle_all(version, platforms, args.dry_run)


if __name__ == "__main__":
    main()