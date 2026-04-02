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
from rich.prompt import Confirm

console = Console()


def find_available_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
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
    n_cpu_layers: int = 0
    n_threads: int = -1
    batch_size: int = 512
    flash_attn: bool = True
    device: str = "auto"
    verbose: bool = False
    gpu_backend: str = "auto"
    auto_ctx: bool = True
    kv_cache_quant: str = "auto"
    cpu_offload: bool = False
    cpu_offload_layers: int = 0


def _find_binary(backend: str = "auto") -> Path:
    """Find llama-server binary using BinaryManager."""
    from moxing.binaries import get_binary_manager
    
    manager = get_binary_manager(backend)
    if not manager.has_binaries():
        console.print(f"[blue]Downloading llama.cpp binaries for {manager.backend}...[/blue]")
        manager.download_binaries()
    
    return manager.get_binary_path("llama-server")


class LlamaServer:
    """Manage llama.cpp server instance."""
    
    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        ctx_size: int = 0,
        n_gpu_layers: int = -1,
        n_cpu_layers: int = 0,
        device: str = "auto",
        gpu_backend: str = "auto",
        auto_ctx: bool = True,
        kv_cache_quant: str = "auto",
        cpu_offload: bool = False,
        cpu_offload_layers: int = 0,
        prompt_offload: bool = False,
        **kwargs
    ):
        model_path = Path(model)
        
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
        self.n_cpu_layers = n_cpu_layers
        self.device = device
        self.gpu_backend = gpu_backend
        self.auto_ctx = auto_ctx
        self.kv_cache_quant = kv_cache_quant
        self.cpu_offload = cpu_offload
        self.cpu_offload_layers = cpu_offload_layers
        self.prompt_offload = prompt_offload
        self.extra_args = kwargs
        
        self.ctx_size = ctx_size if ctx_size > 0 else 4096
        
        if auto_ctx and ctx_size == 0:
            self._auto_detect_context()
        
        self._process: Optional[subprocess.Popen] = None
        self._server_thread: Optional[threading.Thread] = None
        self._base_url = f"http://{host}:{port}"
        self._offload_plan = None
    
    def _auto_detect_context(self):
        """Auto-detect optimal context size based on available VRAM."""
        from moxing.device import DeviceDetector, estimate_model_size_gb, calculate_optimal_context
        
        try:
            detector = DeviceDetector()
            devices = detector.detect()
            
            model_size_gb = estimate_model_size_gb(str(self.model))
            
            gpu_devices = [d for d in devices if d.backend.is_gpu()]
            if not gpu_devices:
                self.ctx_size = 4096
                return
            
            best_device = max(gpu_devices, key=lambda d: d.free_memory_mb)
            available_vram_gb = best_device.free_memory_gb
            
            if available_vram_gb <= 0:
                available_vram_gb = best_device.memory_gb * 0.8
            
            offload_plan = detector.calculate_offload_plan(best_device, model_size_gb)
            self._offload_plan = offload_plan
            
            if offload_plan.needs_offload and self.n_gpu_layers == -1:
                if self.cpu_offload_layers > 0:
                    pass
                elif self.prompt_offload or self.cpu_offload:
                    console.print(f"\n[yellow]Warning: Model ({model_size_gb:.1f}GB) may not fit in GPU VRAM ({available_vram_gb:.1f}GB)[/yellow]")
                    console.print(f"[yellow]Suggested CPU offload: {offload_plan.suggested_cpu_layers} layers[/yellow]")
                    
                    if sys.stdin.isatty():
                        try:
                            if Confirm.ask(f"Enable CPU offload for {offload_plan.suggested_cpu_layers} layers?", default=True):
                                self.cpu_offload_layers = offload_plan.suggested_cpu_layers
                                self.n_gpu_layers = offload_plan.gpu_layers
                        except:
                            pass
            
            ctx_size, n_gpu_layers, n_cpu_layers, notes = calculate_optimal_context(
                model_size_gb=model_size_gb,
                available_vram_gb=available_vram_gb,
                ctx_size_requested=0,
            )
            
            self.ctx_size = ctx_size
            
            console.print(f"[dim]Auto-detected context size: {ctx_size} ({notes})[/dim]")
            
        except Exception as e:
            console.print(f"[yellow]Could not auto-detect context: {e}[/yellow]")
            self.ctx_size = 4096
    
    @staticmethod
    def get_binary_path(backend: str = "auto") -> Path:
        """Get the path to the llama-server binary for current platform."""
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
        ]
        
        if self.cpu_offload_layers > 0 and self.n_gpu_layers != 0:
            if self.n_gpu_layers > 0:
                gpu_layers = self.n_gpu_layers
            else:
                gpu_layers = 99
            args.extend(["-ngl", str(gpu_layers)])
            args.extend(["-ts", f"{gpu_layers},{self.cpu_offload_layers}"])
        else:
            ngl_value = str(self.n_gpu_layers) if self.n_gpu_layers >= 0 else "all"
            args.extend(["-ngl", ngl_value])
        
        if self.device != "auto" and self.device != "CPU" and self.n_gpu_layers != 0:
            args.extend(["-dev", self.device])
        
        if self.gpu_backend not in ["auto", "cpu"]:
            os.environ["GGML_BACKEND"] = self.gpu_backend
        
        kv_cache_args = self._get_kv_cache_args()
        args.extend(kv_cache_args)
        
        for key, value in self.extra_args.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])
        
        return args
    
    def _get_kv_cache_args(self) -> List[str]:
        """Get KV cache optimization arguments."""
        args = []
        
        quant_map = {
            "q8_0": "q8_0",
            "q5_0": "q5_0",
            "q4_0": "q4_0",
            "q4_1": "q4_1",
            "q5_1": "q5_1",
            "iq4_nl": "iq4_nl",
            "iq3_s": "q4_0",
            "q3_k": "q4_0",
            "q2_k": "q4_0",
            "tq4": "iq4_nl",
            "tq3.5": "q4_0",
            "tq3": "q4_0",
            "tq2.5": "q4_0",
            "tq2": "q4_0",
        }
        
        if self.kv_cache_quant == "auto":
            from moxing.kv_cache import recommend_cache_config, estimate_model_size_gb
            
            try:
                model_size_gb = estimate_model_size_gb(str(self.model))
                config = recommend_cache_config(
                    model_size_gb=model_size_gb,
                    available_vram_gb=8.0,
                    desired_ctx_size=self.ctx_size,
                )
                
                from moxing.kv_cache import get_llama_cpp_cache_args
                args.extend(get_llama_cpp_cache_args(config))
                
                self._kv_cache_config = config
            except Exception:
                pass
        elif self.kv_cache_quant in quant_map:
            cache_type = quant_map[self.kv_cache_quant]
            args.extend(["-ctk", cache_type])
            args.extend(["-ctv", cache_type])
        
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
        
        if self.cpu_offload_layers > 0:
            console.print(f"[dim]GPU layers: {self.n_gpu_layers if self.n_gpu_layers > 0 else 'all'}, CPU offload: {self.cpu_offload_layers}[/dim]")
        else:
            gpu_layers_str = str(self.n_gpu_layers) if self.n_gpu_layers >= 0 else "all"
            console.print(f"[dim]GPU layers: {gpu_layers_str}[/dim]")
        
        console.print(f"[dim]Context: {self.ctx_size}[/dim]")
        
        env = os.environ.copy()
        
        from moxing.device import DeviceDetector, BackendType
        detector = DeviceDetector()
        
        try:
            backend_type = BackendType(self.gpu_backend) if self.gpu_backend != "auto" else BackendType.CPU
            device_obj = detector.get_device_by_name(self.device) if self.device != "auto" else None
            backend_env = detector.get_backend_env(backend_type, device_obj)
            env.update(backend_env)
        except Exception:
            pass
        
        if self.gpu_backend == "rocm":
            rocm_paths = [
                "/opt/rocm/lib",
                "/opt/rocm/core/lib",
            ]
            
            import glob
            rocm_core_paths = glob.glob("/opt/rocm/core-*/lib")
            rocm_paths.extend(rocm_core_paths)
            
            for path in rocm_paths:
                if Path(path).exists():
                    if "LD_LIBRARY_PATH" in env:
                        env["LD_LIBRARY_PATH"] = f"{path}:{env['LD_LIBRARY_PATH']}"
                    else:
                        env["LD_LIBRARY_PATH"] = path
                    break
            
            if "HIP_VISIBLE_DEVICES" not in env:
                env["HIP_VISIBLE_DEVICES"] = "0"
        
        env["LD_LIBRARY_PATH"] = f"{binary_path.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        
        try:
            self._process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace',
                cwd=str(binary_path.parent),
                env=env
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