"""
Server management for llama.cpp
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import psutil

import httpx
from rich.console import Console

console = Console()


@dataclass
class GPUInfo:
    name: str
    backend: str
    memory: int
    index: int = 0


@dataclass
class ServerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 4096
    n_gpu_layers: int = -1
    n_threads: int = -1
    batch_size: int = 512
    flash_attn: bool = True
    device: str = "auto"
    verbose: bool = False
    gpu_backend: str = "auto"


def _find_binary() -> Path:
    """Find llama-server binary using BinaryManager."""
    from moxing.binaries import get_binary_manager
    
    manager = get_binary_manager()
    if not manager.has_binaries():
        console.print("[blue]Downloading llama.cpp binaries...[/blue]")
        manager.download_binaries()
    
    return manager.get_binary_path("llama-server")


class LlamaServer:
    """
    Manage llama.cpp server instance.
    
    Usage:
        server = LlamaServer(model="model.gguf")
        server.start()
        
        # Or use as context manager
        with LlamaServer(model="model.gguf") as s:
            # Server is running
            response = s.chat("Hello!")
    """
    
    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        ctx_size: int = 4096,
        n_gpu_layers: int = -1,
        device: str = "auto",
        gpu_backend: str = "auto",
        **kwargs
    ):
        model_path = Path(model)
        
        # Handle compressed/split GGUF files transparently
        if model_path.exists():
            from moxing.gguf_compress import resolve_model_path, is_gguf_compressed
            if is_gguf_compressed(model_path):
                console.print("[blue]Detected compressed GGUF, decompressing...[/blue]")
            resolved_path = resolve_model_path(model_path)
            self.model = resolved_path.resolve()
        else:
            self.model = model_path.resolve()
        
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.device = device
        self.gpu_backend = gpu_backend
        self.extra_args = kwargs
        
        self._process: Optional[subprocess.Popen] = None
        self._server_thread: Optional[threading.Thread] = None
        self._base_url = f"http://{host}:{port}"
        
    @staticmethod
    def get_binary_path() -> Path:
        """Get the path to the llama-server binary for current platform."""
        return _find_binary()
    
    @staticmethod
    def detect_gpus() -> List[GPUInfo]:
        """Detect available GPUs."""
        gpus = []
        
        try:
            binary = LlamaServer.get_binary_path()
            result = subprocess.run(
                [str(binary), "--list-devices"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(binary.parent)
            )
            
            import re
            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if ":" in line and "MiB" in line:
                    match = re.match(r"(\w+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB", line)
                    if match:
                        backend = match.group(1).lower()
                        idx = int(match.group(2))
                        name = match.group(3).strip()
                        memory = int(match.group(4))
                        gpus.append(GPUInfo(name=name, backend=backend, memory=memory, index=idx))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not detect GPUs: {e}[/yellow]")
            
        return gpus
    
    def _build_args(self) -> List[str]:
        """Build command line arguments."""
        args = [
            str(self.get_binary_path()),
            "-m", str(self.model),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-ngl", str(self.n_gpu_layers) if self.n_gpu_layers >= 0 else "all",
        ]
        
        if self.device != "auto":
            args.extend(["-dev", self.device])
            
        if self.gpu_backend != "auto":
            os.environ["GGML_BACKEND"] = self.gpu_backend
            
        for key, value in self.extra_args.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])
                
        return args
    
    def start(self, wait: bool = True, timeout: int = 60) -> "LlamaServer":
        """Start the server."""
        if self._process is not None:
            raise RuntimeError("Server is already running")
            
        args = self._build_args()
        console.print(f"[blue]Starting llama-server...[/blue]")
        console.print(f"[dim]Command: {' '.join(args[:6])}...[/dim]")
        
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(self.get_binary_path().parent)
        )
        
        if wait:
            self._wait_for_server(timeout)
        else:
            time.sleep(1.0)
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                console.print(f"\n[red bold]Server failed to start![/red bold]")
                if stderr:
                    for line in stderr.strip().split("\n")[-15:]:
                        if any(kw in line.lower() for kw in ["error", "failed", "warning"]):
                            console.print(f"[red]{line}[/red]")
                        else:
                            console.print(f"[dim]{line}[/dim]")
                raise RuntimeError("Server failed to start")
            
            self._start_monitor()
            
        return self
    
    def _start_monitor(self):
        """Start background thread to monitor server process."""
        def monitor():
            while self._process and self._process.poll() is None:
                time.sleep(0.5)
            
            if self._process and self._process.poll() is not None:
                try:
                    stdout, stderr = self._process.communicate(timeout=1)
                    if stderr:
                        console.print(f"\n[red bold]Server crashed:[/red bold]")
                        for line in stderr.strip().split("\n")[-10:]:
                            console.print(f"[red]{line}[/red]")
                except:
                    pass
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _wait_for_server(self, timeout: int = 60):
        """Wait for server to be ready."""
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=2)
                if resp.status_code == 200:
                    try:
                        props = httpx.get(f"{self._base_url}/props", timeout=2)
                        if props.status_code == 200:
                            data = props.json()
                            if data.get("total_slots", 0) > 0:
                                console.print(f"[green]Server ready at {self._base_url}[/green]")
                                return
                    except:
                        pass
            except:
                pass
            
            if self._process is not None and self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                raise RuntimeError(f"Server failed to start:\n{stderr}")
                
            time.sleep(0.5)
            
        raise TimeoutError(f"Server did not start within {timeout} seconds")
    
    def stop(self):
        """Stop the server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            console.print("[yellow]Server stopped[/yellow]")
    
    def is_running(self) -> bool:
        """Check if server is running."""
        if self._process is None:
            return False
        return self._process.poll() is None
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def base_url(self) -> str:
        return self._base_url


def main():
    """CLI entry point."""
    import typer
    from moxing.cli import app
    app()


if __name__ == "__main__":
    main()