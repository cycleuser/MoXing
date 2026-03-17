"""
Binary management for pyllm.
Downloads pre-built binaries from GitHub releases on first use.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from rich.console import Console
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, TextColumn

console = Console()


LLAMA_CPP_REPO = "ggml-org/llama.cpp"
BINARY_CACHE_DIR = Path.home() / ".cache" / "pyllm" / "binaries"

ESSENTIAL_BINARIES = [
    "llama-server",
    "llama-cli",
    "llama-bench",
    "llama-quantize",
]


@dataclass
class BinaryInfo:
    name: str
    version: str
    platform: str
    path: Path
    dlls: List[str]


class BinaryManager:
    """
    Manage llama.cpp binaries.
    
    Downloads pre-built binaries from GitHub releases on first use.
    Caches them in ~/.cache/pyllm/binaries/
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or BINARY_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._version_file = self.cache_dir / "version.txt"
        self._binaries: Optional[BinaryInfo] = None
    
    @property
    def platform(self) -> str:
        """Get current platform."""
        if sys.platform == "win32":
            return "windows"
        elif sys.platform == "darwin":
            return "darwin"
        else:
            return "linux"
    
    @property
    def binary_extension(self) -> str:
        """Get binary extension for platform."""
        return ".exe" if sys.platform == "win32" else ""
    
    def get_binary_path(self, name: str = "llama-server") -> Path:
        """Get path to a binary, downloading if necessary."""
        binary_name = name if name.endswith(self.binary_extension) else name + self.binary_extension
        
        if self._binaries:
            return self._binaries.path / binary_name
        
        platform_dir = self.cache_dir / self.platform
        binary_path = platform_dir / binary_name
        
        if binary_path.exists():
            return binary_path
        
        self.download_binaries()
        
        if not binary_path.exists():
            raise FileNotFoundError(
                f"Binary not found: {binary_path}\n"
                f"Please run 'pyllm download-binaries' to download binaries."
            )
        
        return binary_path
    
    def get_all_dlls(self) -> List[Path]:
        """Get all required DLLs for Windows."""
        if sys.platform != "win32":
            return []
        
        if self._binaries:
            return [self._binaries.path / dll for dll in self._binaries.dlls]
        
        platform_dir = self.cache_dir / self.platform
        return list(platform_dir.glob("*.dll"))
    
    def is_downloaded(self) -> bool:
        """Check if binaries are downloaded."""
        platform_dir = self.cache_dir / self.platform
        server_path = platform_dir / f"llama-server{self.binary_extension}"
        return server_path.exists()
    
    def get_installed_version(self) -> Optional[str]:
        """Get installed binary version."""
        if self._version_file.exists():
            return self._version_file.read_text().strip()
        return None
    
    def get_latest_release(self) -> dict:
        """Get latest release info from GitHub API."""
        url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
        req = Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "pyllm-server"
        })
        
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    
    def find_asset_for_platform(self, assets: List[dict], backend: str = "auto") -> Optional[dict]:
        """Find the appropriate asset for current platform."""
        platform_patterns = {
            "windows": ["win", "windows", "msvc", "mingw"],
            "linux": ["linux", "ubuntu"],
            "darwin": ["macos", "darwin", "osx"],
        }
        
        backend_patterns = {
            "cpu": ["cpu", "noavx"],
            "cuda": ["cuda", "cu", "gpu"],
            "vulkan": ["vulkan"],
            "metal": ["metal", "apple"],
        }
        
        patterns = platform_patterns.get(self.platform, [])
        
        backend_order = []
        if backend == "auto":
            if self.platform == "darwin":
                backend_order = ["metal", "cpu"]
            else:
                backend_order = ["vulkan", "cuda", "cpu"]
        else:
            backend_order = [backend]
        
        for b in backend_order:
            b_pats = backend_patterns.get(b, [])
            
            for asset in assets:
                name = asset["name"].lower()
                
                if not any(p in name for p in patterns):
                    continue
                
                if b_pats and not any(p in name for p in b_pats):
                    continue
                
                if name.endswith((".zip", ".tar.gz", ".tgz")):
                    return asset
        
        return None
    
    def download_binaries(
        self,
        version: str = "latest",
        backend: str = "auto",
        force: bool = False,
        quiet: bool = False
    ) -> Path:
        """Download binaries from GitHub release."""
        
        if not force and self.is_downloaded():
            if not quiet:
                console.print("[green]Binaries already downloaded[/green]")
            return self.cache_dir / self.platform
        
        if not quiet:
            console.print("[blue]Fetching release info...[/blue]")
        
        try:
            release = self.get_latest_release()
            tag = release["tag_name"]
            
            if not quiet:
                console.print(f"[blue]Found release: {tag}[/blue]")
            
            asset = self.find_asset_for_platform(release["assets"], backend)
            
            if not asset:
                raise RuntimeError(f"No binary release found for {self.platform}")
            
            if not quiet:
                console.print(f"[blue]Downloading: {asset['name']}[/blue]")
            
            platform_dir = self.cache_dir / self.platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / asset["name"]
                
                self._download_file(asset["browser_download_url"], archive_path, quiet)
                
                self._extract_binaries(archive_path, platform_dir, quiet)
            
            self._version_file.write_text(tag)
            
            if not quiet:
                console.print(f"[green]Binaries installed to: {platform_dir}[/green]")
            
            return platform_dir
            
        except Exception as e:
            raise RuntimeError(f"Failed to download binaries: {e}")
    
    def _download_file(self, url: str, dest: Path, quiet: bool = False):
        """Download a file with progress."""
        req = Request(url, headers={"Accept": "application/octet-stream"})
        
        try:
            with urlopen(req, timeout=300) as response:
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
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Download failed: {e}")
    
    def _extract_binaries(self, archive_path: Path, dest_dir: Path, quiet: bool = False):
        """Extract binaries from archive."""
        if not quiet:
            console.print("[blue]Extracting binaries...[/blue]")
        
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.namelist():
                    filename = Path(member).name
                    if filename:
                        if filename.endswith(self.binary_extension) or filename.endswith(".dll"):
                            source = zf.open(member)
                            target = dest_dir / filename
                            with open(target, "wb") as f:
                                f.write(source.read())
                            if not filename.endswith(".dll"):
                                os.chmod(target, 0o755)
                            if not quiet:
                                console.print(f"  [green]Extracted: {filename}[/green]")
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        filename = Path(member.name).name
                        if filename.startswith("llama-") or filename in ["main"]:
                            source = tf.extractfile(member)
                            target = dest_dir / filename
                            with open(target, "wb") as f:
                                f.write(source.read())
                            os.chmod(target, 0o755)
                            if not quiet:
                                console.print(f"  [green]Extracted: {filename}[/green]")
    
    def clear_cache(self):
        """Clear binary cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            console.print(f"[green]Cleared cache: {self.cache_dir}[/green]")
    
    def list_cached_binaries(self) -> List[str]:
        """List cached binaries."""
        platform_dir = self.cache_dir / self.platform
        if not platform_dir.exists():
            return []
        
        if self.binary_extension:
            return [f.stem for f in platform_dir.glob(f"*{self.binary_extension}")]
        else:
            return [f.name for f in platform_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)]


_binary_manager: Optional[BinaryManager] = None


def get_binary_manager() -> BinaryManager:
    """Get the global binary manager instance."""
    global _binary_manager
    if _binary_manager is None:
        _binary_manager = BinaryManager()
    return _binary_manager


def ensure_binaries() -> Path:
    """Ensure binaries are downloaded, return path to binary directory."""
    manager = get_binary_manager()
    if not manager.is_downloaded():
        manager.download_binaries(quiet=False)
    return manager.cache_dir / manager.platform


def get_server_binary() -> Path:
    """Get path to llama-server binary."""
    manager = get_binary_manager()
    return manager.get_binary_path("llama-server")