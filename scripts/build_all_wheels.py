#!/usr/bin/env python3
"""
Build MoXing wheels for all platforms with all backends.

This script downloads llama.cpp binaries (including CUDA runtime) and builds
platform-specific wheels that include ALL dependencies for offline installation.

Users can simply: pip install moxing[cuda] and run immediately!

Usage:
    python scripts/build_all_wheels.py                    # Build all
    python scripts/build_all_wheels.py --platform darwin-arm64-metal
    python scripts/build_all_wheels.py --list             # List configs
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
from typing import List, Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError


REPO_ROOT = Path(__file__).parent.parent
LLAMA_CPP_REPO = "ggml-org/llama.cpp"


@dataclass
class PlatformConfig:
    """Platform and backend configuration."""
    name: str
    os: str
    arch: str
    backend: str
    asset_pattern: str
    wheel_tag: str
    binary_ext: str
    lib_ext: str
    description: str
    extra_name: str
    includes_runtime: bool = False


PLATFORM_CONFIGS = [
    # macOS - Metal (无需额外依赖，macOS 自带)
    PlatformConfig(
        name="darwin-arm64",
        os="darwin", arch="arm64", backend="metal",
        asset_pattern="bin-macos-arm64.tar.gz",
        wheel_tag="macosx_11_0_arm64",
        binary_ext="", lib_ext=".dylib",
        description="macOS Apple Silicon (Metal)",
        extra_name="metal",
        includes_runtime=True
    ),
    PlatformConfig(
        name="darwin-x64",
        os="darwin", arch="x64", backend="metal",
        asset_pattern="bin-macos-x64.tar.gz",
        wheel_tag="macosx_10_9_x86_64",
        binary_ext="", lib_ext=".dylib",
        description="macOS Intel (Metal)",
        extra_name="metal",
        includes_runtime=True
    ),
    
    # Windows - CUDA (包含 CUDA runtime，无需安装 CUDA)
    # Use cudart-llama-bin-* which includes CUDA runtime DLLs
    PlatformConfig(
        name="windows-x64-cuda",
        os="windows", arch="x64", backend="cuda",
        asset_pattern="cudart-llama-bin-win-cuda-12.4-x64.zip",
        wheel_tag="win_amd64",
        binary_ext=".exe", lib_ext=".dll",
        description="Windows x64 (CUDA 12.4, bundled)",
        extra_name="cuda",
        includes_runtime=True
    ),
    # Windows - Vulkan
    PlatformConfig(
        name="windows-x64-vulkan",
        os="windows", arch="x64", backend="vulkan",
        asset_pattern="bin-win-vulkan-x64.zip",
        wheel_tag="win_amd64",
        binary_ext=".exe", lib_ext=".dll",
        description="Windows x64 (Vulkan)",
        extra_name="vulkan",
        includes_runtime=False
    ),
    # Windows - CPU
    PlatformConfig(
        name="windows-x64-cpu",
        os="windows", arch="x64", backend="cpu",
        asset_pattern="bin-win-cpu-x64.zip",
        wheel_tag="win_amd64",
        binary_ext=".exe", lib_ext=".dll",
        description="Windows x64 (CPU)",
        extra_name="cpu",
        includes_runtime=True
    ),
    
    # Linux - CPU
    PlatformConfig(
        name="linux-x64-cpu",
        os="linux", arch="x64", backend="cpu",
        asset_pattern="bin-ubuntu-x64.tar.gz",
        wheel_tag="manylinux_2_17_x86_64",
        binary_ext="", lib_ext=".so",
        description="Linux x64 (CPU)",
        extra_name="cpu",
        includes_runtime=True
    ),
    # Linux - Vulkan
    PlatformConfig(
        name="linux-x64-vulkan",
        os="linux", arch="x64", backend="vulkan",
        asset_pattern="bin-ubuntu-vulkan-x64.tar.gz",
        wheel_tag="manylinux_2_17_x86_64",
        binary_ext="", lib_ext=".so",
        description="Linux x64 (Vulkan)",
        extra_name="vulkan",
        includes_runtime=False
    ),
    # Linux - ROCm (AMD GPU)
    PlatformConfig(
        name="linux-x64-rocm",
        os="linux", arch="x64", backend="rocm",
        asset_pattern="bin-ubuntu-rocm-7.2-x64.tar.gz",
        wheel_tag="manylinux_2_17_x86_64",
        binary_ext="", lib_ext=".so",
        description="Linux x64 (ROCm/AMD)",
        extra_name="rocm",
        includes_runtime=False
    ),
]

BINARIES = ["llama-server", "llama-cli", "llama-bench", "llama-quantize", "llama-gguf-split"]


def github_api_request(url: str, max_retries: int = 3) -> dict:
    """Make GitHub API request with retry logic. Uses gh CLI if available."""
    api_path = url.replace("https://api.github.com/", "")
    
    try:
        result = subprocess.run(
            ["gh", "api", api_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "moxing-build"})
    
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            if e.code == 403 and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")


def get_latest_llama_version() -> str:
    """Get latest llama.cpp release version."""
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
    data = github_api_request(url)
    return data["tag_name"]


def find_release_asset(version: str, pattern: str) -> Optional[str]:
    """Find matching asset in release."""
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/tags/{version}"
    data = github_api_request(url)
    
    pattern_lower = pattern.lower()
    
    for asset in data.get("assets", []):
        name = asset["name"].lower()
        
        if pattern_lower in name:
            return asset["name"]
    
    pattern_without_ext = pattern_lower.replace(".tar.gz", "").replace(".zip", "")
    for asset in data.get("assets", []):
        name = asset["name"].lower()
        if pattern_without_ext in name:
            return asset["name"]
    
    return None


def download_asset(version: str, asset_name: str, dest: Path, max_retries: int = 3) -> Path:
    """Download release asset with progress and retry."""
    url = f"https://github.com/{LLAMA_CPP_REPO}/releases/download/{version}/{asset_name}"
    
    print(f"  Downloading: {asset_name}")
    
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "moxing-build"})
            
            with urlopen(req, timeout=600) as response:
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                
                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(8192 * 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\r  Progress: {downloaded/total*100:.1f}% ({downloaded/1024/1024:.1f} MB)", end="")
            
            print()
            return dest
        except HTTPError as e:
            if e.code == 403 and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"\n  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max download retries exceeded")


def extract_binaries(archive: Path, dest: Path, config: PlatformConfig) -> List[str]:
    """Extract ALL files from archive (including DLLs, .so, .dylib)."""
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
                
                is_binary = any(f.startswith(b) for b in BINARIES)
                is_lib = f.endswith((".dylib", ".so", ".dll"))
                
                if is_binary or is_lib:
                    dst = dest / f
                    shutil.copy2(src, dst)
                    
                    if is_binary and config.os != "windows":
                        dst.chmod(0o755)
                    
                    extracted.append(f)
                    print(f"    {f}")
        
        for root, dirs, files in os.walk(tmpdir):
            for item in os.listdir(root):
                src = Path(root) / item
                if src.is_symlink():
                    dst = dest / item
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    os.symlink(os.readlink(src), dst)
                    print(f"    {item} -> {os.readlink(src)}")
    
    return extracted


def build_wheel(config: PlatformConfig, version: str, output_dir: Path) -> Optional[Path]:
    """Build wheel for specific platform."""
    print(f"\n{'='*70}")
    print(f"Building: {config.name}")
    print(f"  {config.description}")
    print(f"  Includes runtime: {'YES' if config.includes_runtime else 'NO'}")
    print('='*70)
    
    asset_name = find_release_asset(version, config.asset_pattern)
    if not asset_name:
        print(f"  WARNING: No asset found matching {config.asset_pattern}")
        return None
    
    print(f"  Asset: {asset_name}")
    
    bin_dir = REPO_ROOT / "moxing" / "bin" / config.name
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / asset_name
        download_asset(version, asset_name, archive)
        
        print("  Extracting files:")
        extract_binaries(archive, bin_dir, config)
    
    (bin_dir / "VERSION").write_text(f"{version}\n{config.backend}\n")
    (bin_dir / "BACKEND").write_text(config.backend)
    
    env = os.environ.copy()
    env["MOXING_PLATFORM"] = config.name
    env["MOXING_BACKEND"] = config.backend
    
    with tempfile.TemporaryDirectory() as build_tmpdir:
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", build_tmpdir],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env
        )
        
        shutil.rmtree(bin_dir)
        
        if result.returncode != 0:
            print(f"  ERROR: Build failed")
            print(result.stderr)
            return None
        
        wheels = list(Path(build_tmpdir).glob("*.whl"))
        if wheels:
            wheel = wheels[0]
            new_name = wheel.name.replace("-py3-none-any.whl", f"-py3-none-{config.name}.whl")
            new_path = output_dir / new_name
            shutil.copy2(wheel, new_path)
            print(f"  Created: {new_name}")
            return new_path
    
    return None


def build_all_wheels(version: str, output_dir: Path, platforms: Optional[List[str]] = None) -> List[Path]:
    """Build wheels for all or specified platforms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = PLATFORM_CONFIGS
    if platforms:
        configs = [c for c in configs if c.name in platforms or c.extra_name in platforms]
    
    print(f"\nBuilding {len(configs)} wheel(s) for llama.cpp {version}")
    print(f"Output: {output_dir}")
    
    built = []
    failed = []
    
    for config in configs:
        try:
            wheel = build_wheel(config, version, output_dir)
            if wheel:
                built.append(wheel)
            else:
                failed.append(config.name)
            time.sleep(2)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(config.name)
    
    print(f"\n{'='*70}")
    print("Build Summary")
    print('='*70)
    print(f"  Successful: {len(built)}")
    print(f"  Failed: {len(failed)}")
    
    if built:
        print("\n  Built wheels:")
        for w in built:
            size = w.stat().st_size / (1024*1024)
            print(f"    - {w.name} ({size:.1f} MB)")
    
    if failed:
        print("\n  Failed platforms:")
        for f in failed:
            print(f"    - {f}")
    
    return built


def list_platforms():
    """List all available platform configurations."""
    print("\nAvailable platform configurations:")
    print("-" * 80)
    print(f"{'Name':<25} {'Description':<35} {'Runtime':<8} {'Extra':<10}")
    print("-" * 80)
    
    for config in PLATFORM_CONFIGS:
        runtime = "YES" if config.includes_runtime else "NO"
        print(f"{config.name:<25} {config.description:<35} {runtime:<8} [{config.extra_name}]")
    
    print("\nInstall examples:")
    print("  pip install moxing                    # Auto-detect (CPU/Metal/Vulkan)")
    print("  pip install moxing[cuda]              # NVIDIA GPU (includes CUDA runtime)")
    print("  pip install moxing[vulkan]            # Vulkan (cross-platform)")
    print("  pip install moxing[rocm]              # AMD GPU (Linux)")
    print("  pip install moxing[metal]             # Apple Metal (macOS)")


def main():
    parser = argparse.ArgumentParser(description="Build MoXing wheels for all platforms")
    parser.add_argument("--platform", "-p", default=None,
                       help="Platform(s) to build (comma-separated)")
    parser.add_argument("--version", "-v", default=None,
                       help="llama.cpp version (default: latest)")
    parser.add_argument("--output", "-o", default="dist",
                       help="Output directory for wheels")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available platforms")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be built")
    
    args = parser.parse_args()
    
    if args.list:
        list_platforms()
        return
    
    version = args.version or get_latest_llama_version()
    print(f"llama.cpp version: {version}")
    
    platforms = None
    if args.platform:
        platforms = [p.strip() for p in args.platform.split(",")]
    
    if args.dry_run:
        configs = PLATFORM_CONFIGS
        if platforms:
            configs = [c for c in configs if c.name in platforms or c.extra_name in platforms]
        
        print(f"\nWould build {len(configs)} wheel(s):")
        for c in configs:
            print(f"  - {c.name}: {c.description}")
        return
    
    output_dir = Path(args.output)
    build_all_wheels(version, output_dir, platforms)


if __name__ == "__main__":
    main()