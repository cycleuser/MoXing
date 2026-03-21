"""
Binary management for moxing.

Binaries are downloaded on first use from GitHub releases:
1. https://github.com/cycleuser/MoXing/releases/tag/binaries (primary)
2. https://github.com/ggml-org/llama.cpp/releases (fallback)

Binary loading priority:
    1. Bundled binaries: moxing/bin/{os}-{arch}-{backend}/
    2. Cached binaries: ~/.cache/moxing/binaries/{os}-{arch}-{backend}/
    3. Runtime download

Environment variables:
    MOXING_BINARY_MIRROR    - Custom mirror URL (optional)
    MOXING_NO_UPDATE_CHECK  - Skip update check (set to "1")
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
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from rich.console import Console
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()


MOXING_REPO = "cycleuser/MoXing"
LLAMA_CPP_REPO = "ggml-org/llama.cpp"
CACHE_DIR = Path.home() / ".cache" / "moxing" / "binaries"
BIN_DIR = Path(__file__).parent / "bin"
CONFIG_DIR = Path.home() / ".config" / "moxing"
SKIP_UPDATE_FILE = CONFIG_DIR / "skip_update"

ESSENTIAL_BINARIES = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]

BACKEND_PRIORITY = {
    "linux": ["cuda", "vulkan", "rocm", "cpu"],
    "windows": ["cuda", "vulkan", "cpu"],
    "darwin": ["metal", "cpu"],
}

PLATFORM_ALIASES = {
    "linux_x86_64": "linux-x64",
    "linux_x64": "linux-x64",
    "win_amd64": "windows-x64",
    "win_x64": "windows-x64",
    "macosx_arm64": "darwin-arm64",
    "darwin_arm64": "darwin-arm64",
    "macosx_x86_64": "darwin-x64",
    "darwin_x64": "darwin-x64",
}


@dataclass
class BinaryInfo:
    name: str
    version: str
    platform: str
    backend: str
    path: Path


@dataclass
class VersionInfo:
    moxing_version: Optional[str]
    llama_cpp_version: Optional[str]
    release_url: Optional[str]


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


def detect_bundled_platform() -> Optional[str]:
    """
    Detect what platform's binaries are bundled in this wheel.
    
    Returns None for universal wheels (all platforms bundled).
    Returns platform name for platform-specific wheels.
    """
    if not BIN_DIR.exists():
        return None
    
    bundled = list(BIN_DIR.iterdir())
    
    if len(bundled) == 1 and bundled[0].is_dir():
        name = bundled[0].name
        if "-" in name:
            return name
    
    return None


def get_wheel_platform_info() -> dict:
    """
    Get information about the current wheel's platform.
    
    Returns:
        dict with keys: 'bundled_platform', 'is_universal', 'available_backends'
    """
    bundled = detect_bundled_platform()
    
    available = {}
    if BIN_DIR.exists():
        for d in BIN_DIR.iterdir():
            if d.is_dir() and "-" in d.name:
                parts = d.name.rsplit("-", 1)
                if len(parts) == 2:
                    backend = parts[1]
                    server = d / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
                    available[backend] = server.exists()
    
    return {
        "bundled_platform": bundled,
        "is_universal": bundled is None and len(available) > 1,
        "available_backends": available
    }


def get_latest_llama_cpp_version() -> Optional[str]:
    """Get the latest llama.cpp release version."""
    try:
        url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
        req = Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "moxing"
        })
        
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("tag_name")
    except Exception:
        return None


def get_moxing_binaries_release() -> Optional[VersionInfo]:
    """Get version info from moxing binaries release."""
    try:
        url = f"https://api.github.com/repos/{MOXING_REPO}/releases/tags/binaries"
        req = Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "moxing"
        })
        
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            body = data.get("body", "")
            moxing_version = None
            llama_cpp_version = None
            
            for line in body.split("\n"):
                if line.startswith("llama.cpp:"):
                    llama_cpp_version = line.split(":", 1)[1].strip()
                elif line.startswith("moxing:"):
                    moxing_version = line.split(":", 1)[1].strip()
            
            if not llama_cpp_version:
                llama_cpp_version = data.get("tag_name", "").replace("binaries-", "")
            
            return VersionInfo(
                moxing_version=moxing_version,
                llama_cpp_version=llama_cpp_version,
                release_url=f"https://github.com/{MOXING_REPO}/releases/tag/binaries"
            )
    except Exception:
        return None


def should_check_for_updates() -> bool:
    """Check if we should check for binary updates."""
    if os.environ.get("MOXING_NO_UPDATE_CHECK") == "1":
        return False
    
    if SKIP_UPDATE_FILE.exists():
        return False
    
    return True


def skip_update_forever():
    """Create a file to skip update checks forever."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SKIP_UPDATE_FILE.write_text(str(int(time.time())))


def clear_skip_update():
    """Remove the skip update file."""
    if SKIP_UPDATE_FILE.exists():
        SKIP_UPDATE_FILE.unlink()


class BinaryManager:
    """
    Manage llama.cpp binaries with bundled support.
    
    Priority:
    1. Bundled binaries in moxing/bin/{os}-{arch}-{backend}/
    2. Cached binaries in ~/.cache/moxing/binaries/{os}-{arch}-{backend}/
    3. Download from GitHub releases
    """
    
    def __init__(self, backend: str = "auto", cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._requested_backend = backend
        self._resolved_backend: Optional[str] = None
    
    @property
    def platform_name(self) -> str:
        return PlatformDetector.get_platform_name()
    
    @property
    def backend(self) -> str:
        if self._resolved_backend:
            return self._resolved_backend
        
        if self._requested_backend != "auto":
            self._resolved_backend = self._requested_backend
            return self._resolved_backend
        
        detected = PlatformDetector.detect_backend()
        
        if self._has_bundled_backend(detected):
            self._resolved_backend = detected
            return detected
        
        for fallback in BACKEND_PRIORITY.get(PlatformDetector.get_os(), ["cpu"]):
            if self._has_bundled_backend(fallback):
                self._resolved_backend = fallback
                return fallback
        
        self._resolved_backend = detected
        return detected
    
    @property
    def binary_extension(self) -> str:
        return ".exe" if sys.platform == "win32" else ""
    
    def _has_bundled_backend(self, backend: str) -> bool:
        """Check if a specific backend is bundled."""
        platform_dir = BIN_DIR / f"{self.platform_name}-{backend}"
        if platform_dir.exists():
            server = platform_dir / f"llama-server{self.binary_extension}"
            return server.exists()
        return False
    
    def list_bundled_backends(self) -> List[str]:
        """List all bundled backends for current platform."""
        backends = []
        if not BIN_DIR.exists():
            return backends
        
        prefix = f"{self.platform_name}-"
        for d in BIN_DIR.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                backend = d.name[len(prefix):]
                server = d / f"llama-server{self.binary_extension}"
                if server.exists():
                    backends.append(backend)
        
        return backends
    
    def get_binary_dir(self, backend: Optional[str] = None) -> Path:
        """Get the directory containing binaries for a backend."""
        b = backend or self.backend
        return BIN_DIR / f"{self.platform_name}-{b}"
    
    def get_cache_dir(self, backend: Optional[str] = None) -> Path:
        """Get the cache directory for a backend."""
        b = backend or self.backend
        return self.cache_dir / f"{self.platform_name}-{b}"
    
    def get_binary_path(self, name: str = "llama-server") -> Path:
        """Get path to a binary, checking bundled then cache."""
        binary_name = name if name.endswith(self.binary_extension) else name + self.binary_extension
        
        bundled_dir = self.get_binary_dir()
        bundled_path = bundled_dir / binary_name
        if bundled_path.exists():
            return bundled_path
        
        cache_dir = self.get_cache_dir()
        cache_path = cache_dir / binary_name
        if cache_path.exists():
            return cache_path
        
        return self._download_and_get_path(name)
    
    def has_binaries(self) -> bool:
        """Check if binaries are available (bundled or cached)."""
        for dir_path in [self.get_binary_dir(), self.get_cache_dir()]:
            if not dir_path.exists():
                continue
            
            server = dir_path / f"llama-server{self.binary_extension}"
            if not server.exists():
                continue
            
            if sys.platform == "win32":
                libs = list(dir_path.glob("*.dll"))
            elif sys.platform == "darwin":
                libs = list(dir_path.glob("*.dylib"))
            else:
                libs = list(dir_path.glob("*.so*"))
            
            if len(libs) >= 3:
                return True
        
        return False
    
    def _download_and_get_path(self, name: str) -> Path:
        """Download binaries and return path."""
        self.download_binaries()
        
        cache_path = self.get_cache_dir() / name
        if not cache_path.exists():
            cache_path = self.get_cache_dir() / (name + self.binary_extension)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Binary not found after download: {name}")
        
        return cache_path
    
    def get_all_libs(self) -> List[Path]:
        """Get all required shared libraries."""
        libs = []
        
        for dir_path in [self.get_binary_dir(), self.get_cache_dir()]:
            if dir_path.exists():
                if sys.platform == "win32":
                    libs.extend(dir_path.glob("*.dll"))
                elif sys.platform == "darwin":
                    libs.extend(dir_path.glob("*.dylib"))
                else:
                    libs.extend(dir_path.glob("*.so*"))
        
        return libs
    
    def get_installed_version(self) -> Optional[str]:
        """Get installed binary version."""
        for version_file in [
            self.get_binary_dir() / "VERSION",
            self.get_cache_dir() / "VERSION"
        ]:
            if version_file.exists():
                content = version_file.read_text().strip()
                lines = content.split("\n")
                if lines:
                    return lines[0]
        return None
    
    def get_installed_version_info(self) -> Dict[str, Optional[str]]:
        """Get detailed installed version info."""
        llama_cpp_version: Optional[str] = None
        backend: Optional[str] = None
        source: Optional[str] = None
        
        for version_file in [
            self.get_cache_dir() / "VERSION",
            self.get_binary_dir() / "VERSION"
        ]:
            if version_file.exists():
                content = version_file.read_text().strip()
                lines = content.split("\n")
                if len(lines) >= 1:
                    llama_cpp_version = lines[0]
                if len(lines) >= 2:
                    backend = lines[1]
                source = "cache" if version_file.parent == self.get_cache_dir() else "bundled"
                break
        
        return {
            "llama_cpp_version": llama_cpp_version,
            "backend": backend,
            "source": source
        }
    
    def get_release(self, repo: str, tag: str = "latest") -> dict:
        """Get release info from GitHub API."""
        if tag == "latest":
            url = f"https://api.github.com/repos/{repo}/releases/latest"
        else:
            url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        req = Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "moxing"
        })
        
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    
    def find_asset_for_platform(self, assets: List[dict]) -> Optional[dict]:
        """Find the appropriate asset for current platform and backend."""
        platform_backend = f"{self.platform_name}-{self.backend}"
        
        for asset in assets:
            name = asset["name"].lower()
            if platform_backend.replace("-", "_") in name or platform_backend in name:
                if name.endswith((".zip", ".tar.gz", ".tgz")):
                    return asset
        
        return None
    
    def find_llama_cpp_asset(self, assets: List[dict]) -> Optional[dict]:
        """Find asset from llama.cpp releases."""
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
            elif b_pats and not any(p in name for p in b_pats):
                continue
            
            if name.endswith((".zip", ".tar.gz", ".tgz")):
                return asset
        
        return None
    
    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if binary updates are available.
        
        Returns:
            (has_update, current_version, latest_version)
        """
        current = self.get_installed_version()
        if not current:
            return False, None, None
        
        moxing_info = get_moxing_binaries_release()
        if moxing_info and moxing_info.llama_cpp_version:
            latest = moxing_info.llama_cpp_version
            if current != latest:
                return True, current, latest
        
        latest = get_latest_llama_cpp_version()
        if latest and current != latest:
            return True, current, latest
        
        return False, current, latest
    
    def prompt_for_update(self, current: str, latest: str) -> Optional[str]:
        """
        Prompt user for update decision.
        
        Returns:
            "update", "skip_once", "skip_forever", or None
        """
        console.print(f"\n[yellow]Binary update available![/yellow]")
        console.print(f"  Current: {current}")
        console.print(f"  Latest:  {latest}")
        console.print()
        console.print("Options:")
        console.print("  [1] Update now")
        console.print("  [2] Skip this update")
        console.print("  [3] Skip all future updates")
        console.print("  [4] Continue without updating")
        
        try:
            choice = Prompt.ask("\nYour choice", choices=["1", "2", "3", "4"], default="1")
            
            if choice == "1":
                return "update"
            elif choice == "2":
                return "skip_once"
            elif choice == "3":
                skip_update_forever()
                return "skip_forever"
            else:
                return None
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Skipping update...[/yellow]")
            return "skip_once"
    
    def download_binaries(self, force: bool = False, quiet: bool = False, check_updates: bool = True) -> Path:
        """Download binaries from moxing releases, fallback to llama.cpp."""
        
        if not force and self.has_binaries():
            if check_updates and should_check_for_updates():
                has_update, current, latest = self.check_for_updates()
                if has_update and current and latest:
                    decision = self.prompt_for_update(current, latest)
                    if decision == "update":
                        force = True
                    elif decision == "skip_once":
                        pass
            
            if not force:
                if not quiet:
                    console.print("[green]Binaries already available[/green]")
                return self.get_cache_dir()
        
        custom_mirror = os.environ.get("MOXING_BINARY_MIRROR", "")
        
        if custom_mirror:
            download_url = f"{custom_mirror}/{self.platform_name}-{self.backend}.tar.gz"
            asset_name = f"{self.platform_name}-{self.backend}.tar.gz"
            tag = "custom"
            
            if not quiet:
                console.print(f"[blue]Downloading from custom mirror...[/blue]")
            
            return self._do_download(download_url, asset_name, tag, quiet)
        
        if not quiet:
            console.print("[blue]Downloading from moxing releases...[/blue]")
        
        try:
            release = self.get_release(MOXING_REPO, "binaries")
            tag = release.get("tag_name", "binaries")
            
            body = release.get("body", "")
            llama_version = None
            for line in body.split("\n"):
                if line.startswith("llama.cpp:"):
                    llama_version = line.split(":", 1)[1].strip()
                    break
            
            if not llama_version:
                llama_version = tag.replace("binaries-", "") if tag.startswith("binaries-") else tag
            
            asset = self.find_asset_for_platform(release["assets"])
            
            if asset:
                download_url = asset["browser_download_url"]
                asset_name = asset["name"]
                
                if not quiet:
                    console.print(f"[blue]Found: {asset_name}[/blue]")
                    console.print(f"[blue]llama.cpp version: {llama_version}[/blue]")
                
                return self._do_download(download_url, asset_name, llama_version, quiet)
        except Exception as e:
            if not quiet:
                console.print(f"[yellow]Moxing release not found: {e}[/yellow]")
        
        if not quiet:
            console.print("[blue]Falling back to llama.cpp releases...[/blue]")
        
        release = self.get_release(LLAMA_CPP_REPO, "latest")
        tag = release["tag_name"]
        asset = self.find_llama_cpp_asset(release["assets"])
        
        if not asset:
            raise RuntimeError(f"No binary found for {self.platform_name} ({self.backend})")
        
        download_url = asset["browser_download_url"]
        asset_name = asset["name"]
        
        if not quiet:
            console.print(f"[blue]Found: {asset_name}[/blue]")
            console.print(f"[blue]llama.cpp version: {tag}[/blue]")
        
        return self._do_download(download_url, asset_name, tag, quiet)
    
    def _do_download(self, download_url: str, asset_name: str, tag: str, quiet: bool = False) -> Path:
        """Perform the actual download and extraction."""
        if not quiet:
            console.print(f"[blue]Release: {tag}[/blue]")
            console.print(f"[blue]Backend: {self.backend}[/blue]")
            console.print(f"[blue]Downloading: {asset_name}[/blue]")
        
        cache_dir = self.get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / asset_name
            self._download_file(download_url, archive_path, quiet)
            self._extract_binaries(archive_path, cache_dir, quiet)
        
        (cache_dir / "VERSION").write_text(f"{tag}\n{self.backend}\n")
        
        if not quiet:
            console.print(f"[green]Binaries installed to: {cache_dir}[/green]")
        
        return cache_dir
    
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
                        is_lib = (
                            filename.endswith((".dll", ".so", ".dylib")) or
                            ".so." in filename or
                            ".dylib." in filename
                        )
                        
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
                        is_lib = (
                            filename.endswith((".dll", ".so", ".dylib")) or
                            ".so." in filename or
                            ".dylib." in filename
                        )
                        
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
                                if source:
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
        binaries = []
        
        for dir_path in [self.get_binary_dir(), self.get_cache_dir()]:
            if dir_path.exists():
                if self.binary_extension:
                    binaries.extend([f.stem for f in dir_path.glob(f"*{self.binary_extension}")])
                else:
                    binaries.extend([f.name for f in dir_path.iterdir() if f.is_file() and os.access(f, os.X_OK)])
        
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
    if not manager.has_binaries():
        manager.download_binaries(quiet=False)
    return manager.get_cache_dir()


def get_server_binary(backend: str = "auto") -> Path:
    """Get path to llama-server binary."""
    manager = get_binary_manager(backend)
    return manager.get_binary_path("llama-server")


def list_available_backends() -> Dict[str, bool]:
    """List all available backends and their status."""
    manager = BinaryManager()
    os_name = PlatformDetector.get_os()
    
    backends = BACKEND_PRIORITY.get(os_name, ["cpu"])
    if os_name == "darwin":
        backends = ["metal", "cpu"]
    
    result = {}
    for backend in backends:
        result[backend] = manager._has_bundled_backend(backend)
    
    return result


def check_binary_version() -> Dict[str, Optional[str]]:
    """
    Check the installed binary version.
    
    Returns:
        dict with keys: 'llama_cpp_version', 'latest_llama_cpp', 'has_update'
    """
    manager = BinaryManager()
    info = manager.get_installed_version_info()
    latest = get_latest_llama_cpp_version()
    current = info.get("llama_cpp_version")
    
    has_update: Optional[str] = None
    if current and latest:
        has_update = "yes" if current != latest else "no"
    
    return {
        "llama_cpp_version": current,
        "latest_llama_cpp": latest,
        "has_update": has_update
    }