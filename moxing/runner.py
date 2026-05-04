"""
Unified runner for automatic model loading and optimal configuration.

Supports:
- llama_cpp: llama.cpp server (GGUF models)
- ollama: Ollama runner (Ollama model format)
- vllm: vLLM engine (HuggingFace/GGUF models)
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from moxing.device import (
    DeviceDetector, Device, DeviceConfig, BackendType,
    get_device_config, estimate_model_size_gb
)
from moxing.models import (
    ModelDownloader, ModelInfo, ModelRegistry, download_model
)
from moxing.runners.base import (
    BaseRunner, RunnerConfig, create_runner, detect_best_runner
)

RunConfig = RunnerConfig
from moxing.runners.llama_cpp import LlamaCppRunner
from moxing.runners.vllm import VLLMRunner
from moxing.runners.ollama import OllamaRunner

console = Console()


class AutoRunner:
    """
    Automatic model runner with optimal configuration.
    
    Usage:
        runner = AutoRunner()
        
        # Download and run a model
        runner.run("llama-3.2-3b")
        
        # Or with specific model
        runner.run_model("path/to/model.gguf")
        
        # Or start server
        with runner.server("llama-3.2-3b") as server:
            # Use server...
            pass
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        auto_detect_device: bool = True,
        prefer_backend: Optional[str] = None,
        runner_type: str = "auto"
    ):
        self.model_dir = model_dir or ModelDownloader().cache_dir
        self.auto_detect_device = auto_detect_device
        self.prefer_backend = prefer_backend
        self.runner_type = runner_type
        
        self._detector = DeviceDetector() if auto_detect_device else None
        self._downloader = ModelDownloader(model_dir)
        self._current_server: Optional[BaseRunner] = None
    
    def detect_config(
        self,
        model_path: Union[str, Path],
        ctx_size: int = 4096
    ) -> RunnerConfig:
        """Detect optimal configuration for a model."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_size_gb = estimate_model_size_gb(str(model_path))
        
        runner = self.runner_type
        if runner == "auto":
            runner = detect_best_runner(model_path=model_path)
        
        device_config = None
        if self.auto_detect_device:
            device_config = self._detector.get_best_device(model_size_gb)
        else:
            device_config = DeviceConfig(
                backend=BackendType.CPU,
                device=Device(index=0, name="CPU", backend=BackendType.CPU),
                n_gpu_layers=0
            )
        
        if ctx_size == 0 and device_config.recommended_ctx > 0:
            ctx_size = device_config.recommended_ctx
        
        return RunnerConfig(
            model=str(model_path),
            runner_type=runner,
            ctx_size=ctx_size,
            backend=device_config.backend.value,
        )
    
    def download(
        self,
        model: str,
        quant: str = "Q4_K_M",
        source: str = "auto"
    ) -> Path:
        """Download a model by name or repo."""
        registry_info = ModelRegistry.get_model_info(model, source)
        
        if registry_info:
            console.print(f"[blue]Found in registry: {registry_info['description']}[/blue]")
            repo = registry_info["repo"]
            
            files = self._downloader.list_files(repo, source)
            if not files:
                raise FileNotFoundError(f"No GGUF files found in {repo}")
            
            filename = None
            for f, _ in files:
                if quant.lower() in f.lower():
                    filename = f
                    break
            
            if not filename:
                filename = files[0][0]
                console.print(f"[yellow]Quantization {quant} not found, using {filename}[/yellow]")
        else:
            if "/" in model:
                repo = model
            else:
                raise ValueError(f"Model not found: {model}")
            
            files = self._downloader.list_files(repo, source)
            if not files:
                raise FileNotFoundError(f"No GGUF files found in {repo}")
            
            filename = None
            if quant:
                for f, _ in files:
                    if quant.lower() in f.lower():
                        filename = f
                        break
            
            if not filename:
                filename = files[0][0]
        
        console.print(f"[green]Downloading: {repo}/{filename}[/green]")
        return self._downloader.download(repo, filename, source)
    
    def run(
        self,
        model: str,
        prompt: str = "Hello!",
        n_tokens: int = 128,
        quant: str = "Q4_K_M",
        source: str = "auto",
        ctx_size: int = 4096,
        chat: bool = True,
        runner_type: str = "auto"
    ) -> str:
        """Download (if needed) and run a model."""
        model_path = self.resolve_model(model, quant, source)
        config = self.detect_config(model_path, ctx_size)
        config.runner_type = runner_type if runner_type != "auto" else config.runner_type
        
        self._print_config(config)
        
        from moxing.client import Client
        
        runner = create_runner(config)
        with runner:
            time.sleep(2)
            
            client = Client(runner.base_url)
            
            if chat:
                response = client.chat.completions.create(
                    model="llama",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=n_tokens
                )
                
                if response.choices:
                    return response.choices[0].get("message", {}).get("content", "")
                return ""
            else:
                resp = client.post(
                    "/completion",
                    json={"prompt": prompt, "n_predict": n_tokens}
                )
                return resp.json().get("content", "")
    
    def resolve_model(
        self,
        model: str,
        quant: str = "Q4_K_M",
        source: str = "auto"
    ) -> Path:
        """Resolve model to local path, downloading if necessary."""
        if Path(model).exists():
            return Path(model)
        
        for local_model in self._downloader.get_local_models():
            if model.lower() in local_model.name.lower() or model.lower() in local_model.repo.lower():
                if quant.lower() in local_model.quantization.lower() or not quant:
                    return local_model.local_path
        
        return self.download(model, quant, source)
    
    def server(
        self,
        model: str,
        quant: str = "Q4_K_M",
        source: str = "auto",
        ctx_size: int = 4096,
        port: int = 8080,
        runner_type: str = "auto",
        **kwargs
    ) -> BaseRunner:
        """Create a server instance for a model."""
        model_path = self.resolve_model(model, quant, source)
        config = self.detect_config(model_path, ctx_size)
        config.port = port
        config.runner_type = runner_type if runner_type != "auto" else config.runner_type
        config.extra_args.update(kwargs)
        
        self._print_config(config)
        
        runner = create_runner(config)
        return runner
    
    def quick_chat(
        self,
        model: str,
        prompt: str,
        quant: str = "Q4_K_M",
        **kwargs
    ) -> str:
        """Quick chat with a model (auto-downloads if needed)."""
        return self.run(model, prompt, quant=quant, **kwargs)
    
    def list_local_models(self) -> List[ModelInfo]:
        """List locally cached models."""
        return self._downloader.get_local_models()
    
    def list_available_models(self) -> None:
        """Print table of popular models."""
        table = Table(title="Popular Models")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Quants", style="yellow")
        
        for name, info in ModelRegistry.list_models().items():
            table.add_row(name, info["description"], ", ".join(info["sizes"]))
        
        console.print(table)
    
    def _print_config(self, config: RunnerConfig):
        """Print the run configuration."""
        panel = Panel(
            f"[green]Model:[/green] {Path(config.model).name}\n"
            f"[blue]Runner:[/blue] {config.runner_type}\n"
            f"[yellow]Backend:[/yellow] {config.backend}\n"
            f"[magenta]Context:[/magenta] {config.ctx_size}\n"
            f"[cyan]Port:[/cyan] {config.port}",
            title="Configuration"
        )
        console.print(panel)
    
    def print_device_info(self):
        """Print device information."""
        self._detector.detect()
        self._detector.list_devices()


def quick_run(
    model: str,
    prompt: str = "Hello!",
    quant: str = "Q4_K_M",
    runner_type: str = "auto",
    **kwargs
) -> str:
    """Quick run function for convenience."""
    runner = AutoRunner(runner_type=runner_type)
    return runner.run(model, prompt, quant=quant, runner_type=runner_type, **kwargs)


def quick_server(
    model: str,
    quant: str = "Q4_K_M",
    port: int = 8080,
    runner_type: str = "auto",
    **kwargs
) -> BaseRunner:
    """Quick server function for convenience."""
    runner = AutoRunner(runner_type=runner_type)
    return runner.server(model, quant=quant, port=port, runner_type=runner_type, **kwargs)
