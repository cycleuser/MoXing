"""
Binary management for moxing.

Supports three-tier binary loading:
1. Wheel built-in binaries (moxing/bin/{platform}/)
2. Local cache (~/.cache/moxing/binaries/{platform}/)
3. Runtime download from GitHub/mirrors

Users can install with extras for automatic backend selection:
    pip install moxing[cuda]      # NVIDIA GPU
    pip install moxing[vulkan]    # Vulkan
    pip install moxing[metal]     # Apple Metal
    pip install moxing[rocm]      # AMD ROCm
    pip install moxing[auto]      # Auto-detect
"""

import os
import sys
import json
import time
import shutil
import tarfile
import zipfile
import tempfile
import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from rich.console import Console
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, TextColumn

console = Console()


LLAMA_CPP_REPO = "ggml-org/llama.cpp"
CACHE_DIR = Path.home() / ".cache" / "moxing" / "binaries"

ESSENTIAL_BINARIES = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]


@dataclass
class BinaryInfo:
    name: str
    version: str
    platform: str
    backend: str
    path: Path


class PlatformDetector:
    """Detect current platform and recommended backend."""
    
    @staticmethod
    def get_os() -> str:
        if sys.platform == "darwin":
            return "darwin"
        elif sys.platform == "win32":
            return "windows"
        else:
            return "linux"
    
    @staticmethod
    def get_arch() -> str:
        machine = platform.machine().lower()
        if machine in ("arm64", "aarch64"):
            return "arm64"
        return "x64"
    
    @classmethod
    def get_platform_name(cls) -> str:
        return f"{cls.get_os()}-{cls.get_arch()}"
    
    @classmethod
    def detect_backend(cls) -> str:
        """Auto-detect best backend for current system."""
        os_name = cls.get_os()
        
        if os_name == "darwin":
            return "metal"
        
        if os_name == "windows":
            if cls._has_nvidia():
                return "cuda"
            return "vulkan"
        
        if os_name == "linux":
            if cls._has_nvidia():
                return "cuda"
            if cls._has_amd():
                return "rocm"
            return "vulkan"
        
        return "cpu"
    
    @staticmethod
    def _has_nvidia() -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi"] if sys.platform != "win32" else ["nvidia-smi"],
                capture_output=True,
                timeout=5,
                shell=(sys.platform == "win32")
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def _has_amd() -> bool:
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False


class BinaryManager:
    """
    Manage llama.cpp binaries with three-tier loading:
    
    1. Wheel built-in: moxing/bin/{platform}-{backend}/
    2. Local cache: ~/.cache/moxing/binaries/{platform}-{backend}/
    3. Runtime download from GitHub
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, backend: str = "auto"):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend if backend != "auto" else None
        self._version_file = self.cache_dir / "version.txt"
        self._binary_info: Optional[BinaryInfo] = None
    
    @property
    def platform_name(self) -> str:
        return PlatformDetector.get_platform_name()
    
    @property
    def backend(self) -> str:
        if self._backend:
            return self._backend
        return PlatformDetector.detect_backend()
    
    @property
    def binary_extension(self) -> str:
        return ".exe" if sys.platform == "win32" else ""
    
    @property
    def lib_extension(self) -> str:
        if sys.platform == "darwin":
            return ".dylib"
        elif sys.platform == "win32":
            return ".dll"
        else:
            return ".so"
    
    def get_platform_dir(self, backend: str = None) -> str:
        """Get platform directory name."""
        b = backend or self.backend
        pf = self.platform_name
        
        if b in ("metal", "cpu"):
            return pf
        else:
            return f"{pf}-{b}"
    
    def get_builtin_binary_dir(self, backend: str = None) -> Path:
        """Get path to wheel built-in binaries."""
        dir_name = self.get_platform_dir(backend)
        return Path(__file__).parent / "bin" / dir_name
    
    def get_cache_binary_dir(self, backend: str = None) -> Path:
        """Get path to cached binaries."""
        dir_name = self.get_platform_dir(backend)
        return self.cache_dir / dir_name
    
    def get_binary_path(self, name: str = "llama-server") -> Path:
        """Get path to a binary, checking all locations."""
        binary_name = name if name.endswith(self.binary_extension) else name + self.binary_extension
        
        # 1. Check wheel built-in
        builtin_dir = self.get_builtin_binary_dir()
        builtin_path = builtin_dir / binary_name
        if builtin_path.exists():
            return builtin_path
        
        # 2. Check cache
        cache_dir = self.get_cache_binary_dir()
        cache_path = cache_dir / binary_name
        if cache_path.exists():
            return cache_path
        
        # 3. Download
        return self._download_and_get_path(name)
    
    def has_builtin_binaries(self) -> bool:
        """Check if wheel contains built-in binaries."""
        builtin_dir = self.get_builtin_binary_dir()
        server = builtin_dir / f"llama-server{self.binary_extension}"
        return server.exists()
    
    def _download_and_get_path(self, name: str) -> Path:
        """Download binaries and return path."""
        self.download_binaries()
        
        cache_path = self.get_cache_binary_dir() / name
        if not cache_path.exists():
            cache_path = self.get_cache_binary_dir() / (name + self.binary_extension)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Binary not found after download: {name}")
        
        return cache_path
    
    def get_all_dlls(self) -> List[Path]:
        """Get all required DLLs for Windows."""
        if sys.platform != "win32":
            return []
        
        builtin_dir = self.get_builtin_binary_dir()
        cache_dir = self.get_cache_binary_dir()
        
        dlls = []
        for d in [builtin_dir, cache_dir]:
            if d.exists():
                dlls.extend(d.glob("*.dll"))
        
        return dlls
    
    def is_downloaded(self) -> bool:
        """Check if binaries are available (builtin or cached)."""
        if self.has_builtin_binaries():
            return True
        
        cache_dir = self.get_cache_binary_dir()
        server = cache_dir / f"llama-server{self.binary_extension}"
        return server.exists()
    
    def get_installed_version(self) -> Optional[str]:
        """Get installed binary version."""
        version_file = self.get_cache_binary_dir() / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip().split("\n")[0]
        
        builtin_version = self.get_builtin_binary_dir() / "VERSION"
        if builtin_version.exists():
            return builtin_version.read_text().strip().split("\n")[0]
        
        return None
    
    def get_latest_release(self) -> dict:
        """Get latest release info from GitHub API."""
        url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
        req = Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "moxing"
        })
        
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    
    def find_asset_for_platform(self, assets: List[dict]) -> Optional[dict]:
        """Find the appropriate asset for current platform and backend."""
        os_name = PlatformDetector.get_os()
        arch = PlatformDetector.get_arch()
        backend = self.backend
        
        platform_patterns = {
            "windows": ["win", "windows"],
            "linux": ["linux", "ubuntu"],
            "darwin": ["macos", "darwin"],
        }
        
        backend_patterns = {
            "cuda": ["cuda", "cu"],
            "vulkan": ["vulkan"],
            "rocm": ["rocm", "hip"],
            "metal": ["metal"],
            "cpu": ["cpu"],
        }
        
        patterns = platform_patterns.get(os_name, [])
        b_pats = backend_patterns.get(backend, [])
        
        for asset in assets:
            name = asset["name"].lower()
            
            if not any(p in name for p in patterns):
                continue
            
            if os_name == "darwin":
                if arch == "arm64" and "arm64" not in name:
                    continue
                if arch == "x64" and "arm64" in name:
                    continue
            elif backend == "cuda":
                if "cudart" in name:
                    return asset
                if any(p in name for p in b_pats):
                    pass
            elif b_pats and not any(p in name for p in b_pats):
                continue
            
            if name.endswith((".zip", ".tar.gz", ".tgz")):
                return asset
        
        return None
    
    def download_binaries(self, force: bool = False, quiet: bool = False) -> Path:
        """Download binaries from GitHub release."""
        
        if not force and self.is_downloaded():
            if not quiet:
                console.print("[green]Binaries already available[/green]")
            return self.get_cache_binary_dir()
        
        if not quiet:
            console.print("[blue]Fetching llama.cpp release info...[/blue]")
        
        try:
            release = self.get_latest_release()
            tag = release["tag_name"]
            
            if not quiet:
                console.print(f"[blue]Found release: {tag}[/blue]")
            
            asset = self.find_asset_for_platform(release["assets"])
            
            if not asset:
                raise RuntimeError(f"No binary found for {self.platform_name} ({self.backend})")
            
            if not quiet:
                console.print(f"[blue]Downloading: {asset['name']}[/blue]")
            
            cache_dir = self.get_cache_binary_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / asset["name"]
                
                self._download_file(asset["browser_download_url"], archive_path, quiet)
                
                self._extract_binaries(archive_path, cache_dir, quiet)
            
            (cache_dir / "VERSION").write_text(f"{tag}\n{self.backend}\n")
            
            if not quiet:
                console.print(f"[green]Binaries installed to: {cache_dir}[/green]")
            
            return cache_dir
            
        except Exception as e:
            raise RuntimeError(f"Failed to download binaries: {e}")
    
    def _download_file(self, url: str, dest: Path, quiet: bool = False):
        """Download a file with progress."""
        req = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "moxing"})
        
        with urlopen(req, timeout=600) as response:
            total = int(response.headers.get("content-length", 0))
            
            if quiet:
                with open(dest, "wb") as f:
                    f.write(response.read())
            else:
                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Downloading", total=total)
                    downloaded = 0
                    chunk_size = 8192 * 16
                    
                    with open(dest, "wb") as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(task, completed=downloaded)
    
    def _extract_binaries(self, archive_path: Path, dest_dir: Path, quiet: bool = False):
        """Extract binaries from archive."""
        if not quiet:
            console.print("[blue]Extracting binaries...[/blue]")
        
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.namelist():
                    filename = Path(member).name
                    if filename:
                        is_binary = any(filename.startswith(b) for b in ESSENTIAL_BINARIES)
                        is_lib = filename.endswith((".dll", ".so", ".dylib"))
                        
                        if is_binary or is_lib:
                            source = zf.open(member)
                            target = dest_dir / filename
                            with open(target, "wb") as f:
                                f.write(source.read())
                            
                            if is_binary:
                                os.chmod(target, 0o755)
                            
                            if not quiet:
                                console.print(f"  [green]{filename}[/green]")
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile() or member.issym():
                        filename = Path(member.name).name
                        is_binary = any(filename.startswith(b) for b in ESSENTIAL_BINARIES)
                        is_lib = filename.endswith((".dll", ".so", ".dylib"))
                        
                        if is_binary or is_lib:
                            if member.issym():
                                target = dest_dir / filename
                                if target.exists() or target.is_symlink():
                                    target.unlink()
                                os.symlink(member.linkname, target)
                                if not quiet:
                                    console.print(f"  [green]{filename} -> {member.linkname}[/green]")
                            else:
                                source = tf.extractfile(member)
                                target = dest_dir / filename
                                with open(target, "wb") as f:
                                    f.write(source.read())
                                
                                if is_binary:
                                    os.chmod(target, 0o755)
                                
                                if not quiet:
                                    console.print(f"  [green]{filename}[/green]")
    
    def clear_cache(self):
        """Clear binary cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            console.print(f"[green]Cleared cache: {self.cache_dir}[/green]")
    
    def list_cached_binaries(self) -> List[str]:
        """List cached binaries."""
        builtin_dir = self.get_builtin_binary_dir()
        cache_dir = self.get_cache_binary_dir()
        
        binaries = []
        
        for d in [builtin_dir, cache_dir]:
            if d.exists():
                if self.binary_extension:
                    binaries.extend([f.stem for f in d.glob(f"*{self.binary_extension}")])
                else:
                    binaries.extend([f.name for f in d.iterdir() if f.is_file() and os.access(f, os.X_OK)])
        
        return list(set(binaries))


_binary_manager: Optional[BinaryManager] = None


def get_binary_manager(backend: str = "auto") -> BinaryManager:
    """Get the global binary manager instance."""
    global _binary_manager
    if _binary_manager is None or backend != "auto":
        _binary_manager = BinaryManager(backend=backend)
    return _binary_manager


def ensure_binaries(backend: str = "auto") -> Path:
    """Ensure binaries are available, return path to binary directory."""
    manager = get_binary_manager(backend)
    if not manager.is_downloaded():
        manager.download_binaries(quiet=False)
    return manager.get_cache_binary_dir()


def get_server_binary(backend: str = "auto") -> Path:
    """Get path to llama-server binary."""
    manager = get_binary_manager(backend)
    return manager.get_binary_path("llama-server")