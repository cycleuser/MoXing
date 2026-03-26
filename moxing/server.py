"""
Server management for llama.cpp
"""

import os
import sys
import json
import time
import signal
import socket
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import psutil

import httpx
from rich.console import Console

console = Console()


def find_available_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try
    
    Returns:
        Available port number
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def is_port_in_use(port: int, host: str = '127.0.0.1') -> bool:
    """Check if a port is already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind((host, port))
            return False
    except OSError:
        return True


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
    auto_ctx: bool = True


def _find_binary(backend: str = "auto") -> Path:
    """Find llama-server binary using BinaryManager.
    
    Args:
        backend: Backend type (auto, vulkan, cuda, rocm, metal, cpu)
    """
    from moxing.binaries import get_binary_manager
    
    manager = get_binary_manager(backend)
    if not manager.has_binaries():
        console.print(f"[blue]Downloading llama.cpp binaries for {manager.backend}...[/blue]")
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
        ctx_size: int = 0,
        n_gpu_layers: int = -1,
        device: str = "auto",
        gpu_backend: str = "auto",
        auto_ctx: bool = True,
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
        self._requested_ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.device = device
        self.gpu_backend = gpu_backend
        self.auto_ctx = auto_ctx
        self.extra_args = kwargs
        
        self.ctx_size = ctx_size if ctx_size > 0 else 4096
        
        if auto_ctx and ctx_size == 0:
            self._auto_detect_context()
        
        self._process: Optional[subprocess.Popen] = None
        self._server_thread: Optional[threading.Thread] = None
        self._base_url = f"http://{host}:{port}"
    
    def _auto_detect_context(self):
        """Auto-detect optimal context size based on available VRAM."""
        from moxing.device import DeviceDetector, estimate_model_size_gb, calculate_optimal_context
        
        try:
            detector = DeviceDetector()
            devices = detector.detect()
            
            model_size_gb = estimate_model_size_gb(str(self.model))
            
            gpu_devices = [d for d in devices if d.backend.value != "cpu"]
            if not gpu_devices:
                self.ctx_size = 4096
                return
            
            best_device = max(gpu_devices, key=lambda d: d.free_memory_mb)
            available_vram_gb = best_device.free_memory_gb
            
            if available_vram_gb <= 0:
                available_vram_gb = best_device.memory_gb * 0.8
            
            ctx_size, n_gpu_layers, notes = calculate_optimal_context(
                model_size_gb=model_size_gb,
                available_vram_gb=available_vram_gb,
                ctx_size_requested=0,
            )
            
            self.ctx_size = ctx_size
            if n_gpu_layers != -1:
                self.n_gpu_layers = n_gpu_layers
            
            console.print(f"[dim]Auto-detected context size: {ctx_size} ({notes})[/dim]")
            
        except Exception as e:
            console.print(f"[yellow]Could not auto-detect context: {e}[/yellow]")
            self.ctx_size = 4096
        
    @staticmethod
    def get_binary_path(backend: str = "auto") -> Path:
        """Get the path to the llama-server binary for current platform.
        
        Args:
            backend: Backend type (auto, vulkan, cuda, rocm, metal, cpu)
        """
        return _find_binary(backend)
    
    def _get_binary_for_backend(self) -> Path:
        """Get the correct binary for the configured backend."""
        return _find_binary(self.gpu_backend)
    
    @staticmethod
    def detect_gpus() -> List[GPUInfo]:
        """Detect available GPUs."""
        gpus = []
        
        try:
            binary = LlamaServer.get_binary_path()
            result = subprocess.run(
                [str(binary), "--list-devices"],
                capture_output=True,
                encoding='utf-8',
                errors='replace',
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
            str(self._get_binary_for_backend()),
            "-m", str(self.model),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-ngl", str(self.n_gpu_layers) if self.n_gpu_layers >= 0 else "all",
        ]
        
        if self.device != "auto" and self.device != "CPU" and self.n_gpu_layers != 0:
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
        
        binary_path = self._get_binary_for_backend()
        
        if not binary_path.exists():
            raise RuntimeError(f"Binary not found: {binary_path}")
        
        args = self._build_args()
        console.print(f"[blue]Starting llama-server...[/blue]")
        console.print(f"[dim]Binary: {binary_path}[/dim]")
        console.print(f"[dim]Command: {' '.join(args)}[/dim]")
        console.print(f"[dim]Working dir: {binary_path.parent}[/dim]")
        console.print(f"[dim]Model: {self.model}[/dim]")
        console.print(f"[dim]Backend: {self.gpu_backend}[/dim]")
        console.print(f"[dim]GPU layers: {self.n_gpu_layers}[/dim]")
        console.print(f"[dim]Context: {self.ctx_size}[/dim]")
        
        try:
            self._process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace',
                cwd=str(binary_path.parent)
            )
        except Exception as e:
            console.print(f"[red]Failed to start process: {e}[/red]")
            raise RuntimeError(f"Failed to start server: {e}")
        
        if wait:
            self._wait_for_server(timeout)
        else:
            time.sleep(2.0)
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                console.print(f"\n[red bold]Server failed to start![/red bold]")
                console.print(f"[dim]Exit code: {self._process.returncode}[/dim]")
                if stdout:
                    console.print(f"[dim]stdout:[/dim]")
                    for line in stdout.strip().split("\n")[-20:]:
                        console.print(f"[dim]  {line}[/dim]")
                if stderr:
                    console.print(f"[red]stderr:[/red]")
                    for line in stderr.strip().split("\n")[-20:]:
                        console.print(f"[red]  {line}[/red]")
                raise RuntimeError("Server failed to start")
            
            self._start_monitor()
            
        return self
    
    def _start_monitor(self):
        """Start background thread to monitor server process and consume output."""
        def drain_stream(stream, name):
            try:
                for line in stream:
                    pass
            except:
                pass
        
        if self._process:
            if self._process.stdout:
                stdout_thread = threading.Thread(
                    target=drain_stream, 
                    args=(self._process.stdout, "stdout"),
                    daemon=True
                )
                stdout_thread.start()
            if self._process.stderr:
                stderr_thread = threading.Thread(
                    target=drain_stream,
                    args=(self._process.stderr, "stderr"),
                    daemon=True
                )
                stderr_thread.start()
        
        def monitor():
            while self._process and self._process.poll() is None:
                time.sleep(0.5)
            
            if self._process and self._process.poll() is not None:
                exit_code = self._process.returncode
                console.print(f"\n[red bold]Server crashed! (exit code: {exit_code})[/red bold]")
                console.print(f"[dim]Check the model path and binary compatibility.[/dim]")
        
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