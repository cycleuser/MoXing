"""
Post-install hooks for backend-specific binaries.

When users install with extras like:
    pip install moxing[cuda]
    pip install moxing[vulkan]
    pip install moxing[rocm]

This module downloads the appropriate binaries after installation.
"""

import os
import sys
import json
import tarfile
import zipfile
import tempfile
import platform
import subprocess
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from urllib.request import urlopen, Request


LLAMA_CPP_REPO = "ggml-org/llama.cpp"
CACHE_DIR = Path.home() / ".cache" / "moxing" / "binaries"


@dataclass
class BackendConfig:
    name: str
    backend: str
    asset_patterns: List[str]
    description: str
    platforms: List[str]


BACKEND_CONFIGS = {
    "metal": BackendConfig(
        name="metal",
        backend="metal",
        asset_patterns=["bin-macos-arm64.tar.gz", "bin-macos-x64.tar.gz"],
        description="Apple Metal (macOS)",
        platforms=["darwin"]
    ),
    "cuda": BackendConfig(
        name="cuda",
        backend="cuda",
        asset_patterns=["bin-win-cuda-12.4-x64.zip", "bin-ubuntu-cuda-12.4-x64.tar.gz"],
        description="NVIDIA CUDA",
        platforms=["linux", "windows"]
    ),
    "vulkan": BackendConfig(
        name="vulkan",
        backend="vulkan",
        asset_patterns=["bin-win-vulkan-x64.zip", "bin-ubuntu-vulkan-x64.tar.gz"],
        description="Vulkan (cross-platform)",
        platforms=["linux", "windows"]
    ),
    "rocm": BackendConfig(
        name="rocm",
        backend="rocm",
        asset_patterns=["bin-ubuntu-rocm-7.2-x64.tar.gz"],
        description="AMD ROCm",
        platforms=["linux"]
    ),
    "cpu": BackendConfig(
        name="cpu",
        backend="cpu",
        asset_patterns=["bin-win-cpu-x64.zip", "bin-ubuntu-x64.tar.gz"],
        description="CPU only",
        platforms=["linux", "windows"]
    ),
}

BINARIES = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]


def get_platform() -> str:
    """Get current platform."""
    if sys.platform == "darwin":
        return "darwin"
    elif sys.platform == "win32":
        return "windows"
    else:
        return "linux"


def get_arch() -> str:
    """Get current architecture."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x64"


def get_latest_version() -> str:
    """Get latest llama.cpp version."""
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    
    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data["tag_name"]
    except Exception:
        return "b8420"


def find_asset(version: str, patterns: List[str], pf: str, arch: str) -> Optional[str]:
    """Find matching asset for platform."""
    url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/tags/{version}"
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    
    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            
            for asset in data.get("assets", []):
                name = asset["name"].lower()
                
                # 检查平台和架构
                if pf == "darwin":
                    if arch == "arm64" and "arm64" not in name:
                        continue
                    if arch == "x64" and ("arm64" in name or "x64" not in name):
                        continue
                    if "macos" not in name:
                        continue
                elif pf == "windows":
                    if "win" not in name:
                        continue
                elif pf == "linux":
                    if "ubuntu" not in name and "linux" not in name:
                        continue
                
                # 检查模式匹配
                for pattern in patterns:
                    pattern_clean = pattern.lower().replace(".tar.gz", "").replace(".zip", "")
                    if pattern_clean in name:
                        return asset["name"]
        
        return None
    except Exception as e:
        print(f"Error finding asset: {e}")
        return None


def download_and_extract(version: str, asset_name: str, dest: Path):
    """Download and extract binaries."""
    url = f"https://github.com/{LLAMA_CPP_REPO}/releases/download/{version}/{asset_name}"
    
    print(f"Downloading: {asset_name}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        archive = tmpdir / asset_name
        
        # 下载
        req = Request(url, headers={"Accept": "application/octet-stream"})
        with urlopen(req, timeout=300) as response:
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(archive, "wb") as f:
                while True:
                    chunk = response.read(8192 * 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r  Progress: {downloaded/total*100:.1f}%", end="")
        
        print()
        
        # 解压
        dest.mkdir(parents=True, exist_ok=True)
        
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                for member in zf.namelist():
                    filename = Path(member).name
                    if filename:
                        is_binary = any(filename.startswith(b) for b in BINARIES)
                        is_lib = filename.endswith((".dll", ".so", ".dylib"))
                        
                        if is_binary or is_lib:
                            source = zf.open(member)
                            target = dest / filename
                            with open(target, "wb") as f:
                                f.write(source.read())
                            print(f"  Extracted: {filename}")
        else:
            with tarfile.open(archive, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile() or member.issym():
                        filename = Path(member.name).name
                        is_binary = any(filename.startswith(b) for b in BINARIES)
                        is_lib = filename.endswith((".dll", ".so", ".dylib"))
                        
                        if is_binary or is_lib:
                            if member.issym():
                                target = dest / filename
                                if target.exists() or target.is_symlink():
                                    target.unlink()
                                os.symlink(member.linkname, target)
                                print(f"  Linked: {filename}")
                            else:
                                source = tf.extractfile(member)
                                target = dest / filename
                                with open(target, "wb") as f:
                                    f.write(source.read())
                                if is_binary:
                                    os.chmod(target, 0o755)
                                print(f"  Extracted: {filename}")
    
    # 保存版本信息
    (dest / "VERSION").write_text(f"{version}\n")


def install_backend(backend: str, version: str = None):
    """Install binaries for specific backend."""
    if backend not in BACKEND_CONFIGS:
        print(f"Unknown backend: {backend}")
        print(f"Available: {list(BACKEND_CONFIGS.keys())}")
        return False
    
    config = BACKEND_CONFIGS[backend]
    pf = get_platform()
    arch = get_arch()
    
    # 检查平台支持
    if pf not in config.platforms:
        print(f"Backend '{backend}' is not supported on {pf}")
        print(f"Supported platforms: {config.platforms}")
        return False
    
    print(f"\nInstalling {config.description} binaries...")
    print(f"Platform: {pf}-{arch}")
    
    # 获取版本
    if not version:
        version = get_latest_version()
    print(f"Version: {version}")
    
    # 查找资源
    asset_name = find_asset(version, config.asset_patterns, pf, arch)
    if not asset_name:
        print(f"ERROR: No matching binary found for {pf}-{arch}")
        return False
    
    print(f"Asset: {asset_name}")
    
    # 下载到缓存目录
    dest = CACHE_DIR / f"{pf}-{arch}-{backend}"
    download_and_extract(version, asset_name, dest)
    
    # 同时复制到 wheel 内置目录（如果可写）
    try:
        wheel_bin = Path(__file__).parent / "bin" / f"{pf}-{arch}-{backend}"
        if wheel_bin.parent.exists():
            if wheel_bin.exists():
                import shutil
                shutil.rmtree(wheel_bin)
            import shutil
            shutil.copytree(dest, wheel_bin)
            print(f"Also installed to: {wheel_bin}")
    except Exception:
        pass
    
    print(f"\nBinaries installed to: {dest}")
    return True


def auto_detect_backend() -> str:
    """Auto-detect best backend for current system."""
    pf = get_platform()
    
    if pf == "darwin":
        return "metal"
    
    # 尝试检测 NVIDIA GPU
    try:
        if pf == "linux":
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return "cuda"
        elif pf == "windows":
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, shell=True)
            if result.returncode == 0:
                return "cuda"
    except Exception:
        pass
    
    # 尝试检测 AMD GPU
    try:
        if pf == "linux":
            result = subprocess.run(["rocm-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return "rocm"
    except Exception:
        pass
    
    # 默认使用 Vulkan 或 CPU
    return "vulkan"


def main():
    """Main entry point for post-install."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install backend-specific binaries")
    parser.add_argument("backend", nargs="?", default=None,
                       choices=list(BACKEND_CONFIGS.keys()) + ["auto"],
                       help="Backend to install (auto=detect)")
    parser.add_argument("--version", "-v", default=None,
                       help="llama.cpp version")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available backends")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable backends:")
        print("-" * 50)
        for name, config in BACKEND_CONFIGS.items():
            print(f"  {name:<10} - {config.description}")
            print(f"             Platforms: {', '.join(config.platforms)}")
        return
    
    backend = args.backend or os.environ.get("MOXING_BACKEND", "auto")
    
    if backend == "auto":
        backend = auto_detect_backend()
        print(f"Auto-detected backend: {backend}")
    
    install_backend(backend, args.version)


if __name__ == "__main__":
    main()