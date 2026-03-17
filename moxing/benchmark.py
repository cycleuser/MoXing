"""
Benchmark and performance testing for GGUF models.
Measures tokens/second, memory usage, and other performance metrics.
"""

import os
import sys
import time
import json
import psutil
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_time_sec: float = 0.0
    completion_time_sec: float = 0.0
    total_time_sec: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    gpu_layers: int = -1
    device: str = ""
    backend: str = ""
    ctx_size: int = 4096
    batch_size: int = 512
    peak_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    model_size_gb: float = 0.0
    
    @property
    def prompt_speed(self) -> float:
        if self.prompt_time_sec > 0:
            return self.prompt_tokens / self.prompt_time_sec
        return 0.0
    
    @property
    def generation_speed(self) -> float:
        if self.completion_time_sec > 0:
            return self.completion_tokens / self.completion_time_sec
        return 0.0


@dataclass
class SystemStats:
    """System resource usage during benchmark."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0


class BenchmarkRunner:
    """
    Run benchmarks on GGUF models with detailed performance metrics.
    
    Usage:
        from moxing.benchmark import BenchmarkRunner
        
        runner = BenchmarkRunner()
        result = runner.run("model.gguf", prompt="Hello", n_tokens=100)
        print(f"Speed: {result.tokens_per_second:.2f} tokens/s")
    """
    
    # Standard benchmark prompts
    BENCHMARK_PROMPTS = {
        "quick": "Write a short poem about AI.",
        "standard": "Write a detailed explanation of how neural networks work, including the concepts of weights, biases, activation functions, and backpropagation.",
        "code": "Write a Python function that implements a binary search tree with insert, delete, and search operations.",
        "creative": "Write a short story about a robot discovering emotions for the first time.",
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._process = None
        self._server = None
    
    def run(
        self,
        model: str,
        prompt: str = "standard",
        n_tokens: int = 128,
        n_runs: int = 1,
        warmup: bool = True,
        ctx_size: int = 4096,
        n_gpu_layers: int = -1,
        device: str = "auto",
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark on the specified model.
        
        Args:
            model: Path to the GGUF model file
            prompt: Benchmark prompt name or custom prompt text
            n_tokens: Number of tokens to generate
            n_runs: Number of benchmark runs (returns average)
            warmup: Run a warmup iteration before measuring
            ctx_size: Context size
            n_gpu_layers: GPU layers (-1 for all)
            device: Device to use
        
        Returns:
            BenchmarkResult with performance metrics
        """
        model_path = Path(model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model}")
        
        if prompt in self.BENCHMARK_PROMPTS:
            prompt_text = self.BENCHMARK_PROMPTS[prompt]
        else:
            prompt_text = prompt
        
        results = []
        
        for run_idx in range(n_runs + (1 if warmup else 0)):
            is_warmup = warmup and run_idx == 0
            
            if is_warmup:
                console.print(f"[dim]Running warmup...[/dim]")
            else:
                console.print(f"[blue]Running benchmark {run_idx + 1 - (1 if warmup else 0)}/{n_runs}...[/blue]")
            
            result = self._run_single(
                model_path=model_path,
                prompt=prompt_text,
                n_tokens=n_tokens,
                ctx_size=ctx_size,
                n_gpu_layers=n_gpu_layers,
                device=device,
                **kwargs
            )
            
            if not is_warmup:
                results.append(result)
        
        return self._average_results(results)
    
    def _run_single(
        self,
        model_path: Path,
        prompt: str,
        n_tokens: int,
        ctx_size: int,
        n_gpu_layers: int,
        device: str,
        **kwargs
    ) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        from moxing.server import LlamaServer
        from moxing.client import Client
        
        model_size_gb = model_path.stat().st_size / (1024 ** 3)
        
        port = 8080 + hash(str(model_path)) % 1000
        while self._is_port_in_use(port):
            port += 1
        
        server = LlamaServer(
            model=str(model_path),
            port=port,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            device=device,
            verbose=self.verbose
        )
        
        result = BenchmarkResult(
            model=model_path.name,
            ctx_size=ctx_size,
            model_size_gb=model_size_gb
        )
        
        start_time = time.time()
        
        try:
            server.start(timeout=120)
            
            client = Client(server.base_url)
            
            props = client.props()
            result.gpu_layers = props.get("default_generation_settings", {}).get("n_gpu_layers", -1)
            
            prompt_start = time.time()
            
            response = client.chat.completions.create(
                model="llama",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=n_tokens,
                stream=False
            )
            
            prompt_end = time.time()
            
            if response.choices:
                message = response.choices[0].get("message", {})
                content = message.get("content", "")
                
                usage = response.usage
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)
                result.total_tokens = usage.get("total_tokens", 0)
            
            result.completion_time_sec = prompt_end - prompt_start
            result.total_time_sec = time.time() - start_time
            
            if result.completion_tokens > 0 and result.completion_time_sec > 0:
                result.tokens_per_second = result.completion_tokens / result.completion_time_sec
            
            if result.prompt_tokens > 0 and result.completion_time_sec > 0:
                result.prompt_tokens_per_second = result.prompt_tokens / result.completion_time_sec
            
            result.peak_memory_mb = self._get_memory_usage()
            
        finally:
            server.stop()
        
        return result
    
    def _is_port_in_use(self, port: int) -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Average multiple benchmark results."""
        if not results:
            return BenchmarkResult(model="unknown")
        
        if len(results) == 1:
            return results[0]
        
        avg = BenchmarkResult(model=results[0].model)
        
        for field_name in ["prompt_tokens", "completion_tokens", "total_tokens",
                           "prompt_time_sec", "completion_time_sec", "total_time_sec",
                           "tokens_per_second", "prompt_tokens_per_second",
                           "peak_memory_mb", "gpu_memory_mb", "model_size_gb"]:
            values = [getattr(r, field_name) for r in results]
            setattr(avg, field_name, sum(values) / len(values))
        
        avg.ctx_size = results[0].ctx_size
        avg.gpu_layers = results[0].gpu_layers
        avg.device = results[0].device
        avg.backend = results[0].backend
        
        return avg
    
    def compare_models(
        self,
        models: List[str],
        prompt: str = "standard",
        n_tokens: int = 128,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Benchmark multiple models and compare results."""
        results = []
        
        for model in models:
            console.print(f"\n[cyan]Benchmarking: {Path(model).name}[/cyan]")
            try:
                result = self.run(model, prompt=prompt, n_tokens=n_tokens, **kwargs)
                results.append(result)
            except Exception as e:
                console.print(f"[red]Error benchmarking {model}: {e}[/red]")
        
        return results
    
    def print_results(self, result: BenchmarkResult):
        """Print benchmark results in a formatted table."""
        console.print(Panel(
            f"[green]Model:[/green] {result.model}\n"
            f"[green]Size:[/green] {result.model_size_gb:.2f} GB\n"
            f"[green]GPU Layers:[/green] {'all' if result.gpu_layers < 0 else result.gpu_layers}",
            title="Benchmark Results"
        ))
        
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        
        table.add_row("Prompt Tokens", f"{result.prompt_tokens}", "tokens")
        table.add_row("Completion Tokens", f"{result.completion_tokens}", "tokens")
        table.add_row("Total Tokens", f"{result.total_tokens}", "tokens")
        table.add_row("Generation Speed", f"{result.tokens_per_second:.2f}", "tokens/s")
        table.add_row("Prompt Speed", f"{result.prompt_tokens_per_second:.2f}", "tokens/s")
        table.add_row("Total Time", f"{result.total_time_sec:.2f}", "seconds")
        table.add_row("Generation Time", f"{result.completion_time_sec:.2f}", "seconds")
        table.add_row("Memory Used", f"{result.peak_memory_mb:.0f}", "MB")
        
        console.print(table)
    
    def print_comparison(self, results: List[BenchmarkResult]):
        """Print a comparison table of multiple benchmark results."""
        if not results:
            console.print("[yellow]No results to compare[/yellow]")
            return
        
        table = Table(title="Model Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="blue")
        table.add_column("Speed", style="green")
        table.add_column("Prompt", style="yellow")
        table.add_column("Total", style="magenta")
        table.add_column("Memory", style="white")
        
        for r in results:
            table.add_row(
                r.model[:30],
                f"{r.model_size_gb:.1f}GB",
                f"{r.tokens_per_second:.1f} t/s",
                f"{r.prompt_tokens_per_second:.1f} t/s",
                f"{r.total_tokens}",
                f"{r.peak_memory_mb:.0f}MB"
            )
        
        console.print(table)


def benchmark_model(
    model: str,
    prompt: str = "standard",
    n_tokens: int = 128,
    **kwargs
) -> BenchmarkResult:
    """Convenience function to benchmark a model."""
    runner = BenchmarkRunner()
    return runner.run(model, prompt=prompt, n_tokens=n_tokens, **kwargs)


def estimate_speed(model_size_gb: float, gpu_memory_gb: float, backend: str = "vulkan") -> Dict[str, float]:
    """
    Estimate inference speed based on model and hardware.
    
    Returns estimated tokens/second for different scenarios.
    """
    estimates = {
        "vulkan": {
            "full_gpu": 25,
            "partial_gpu": 15,
            "cpu": 3,
        },
        "cuda": {
            "full_gpu": 35,
            "partial_gpu": 20,
            "cpu": 3,
        },
        "metal": {
            "full_gpu": 30,
            "partial_gpu": 18,
            "cpu": 4,
        },
        "rocm": {
            "full_gpu": 28,
            "partial_gpu": 16,
            "cpu": 3,
        },
        "cpu": {
            "cpu": 2,
        }
    }
    
    backend_estimates = estimates.get(backend, estimates["cpu"])
    
    if gpu_memory_gb >= model_size_gb * 1.2:
        mode = "full_gpu"
    elif gpu_memory_gb >= model_size_gb * 0.5:
        mode = "partial_gpu"
    else:
        mode = "cpu"
    
    base_speed = backend_estimates.get(mode, backend_estimates.get("cpu", 2))
    
    if mode == "full_gpu":
        speed = base_speed * (gpu_memory_gb / 8)
    elif mode == "partial_gpu":
        speed = base_speed * (gpu_memory_gb / 8) * 0.8
    else:
        speed = base_speed
    
    return {
        "estimated_tokens_per_second": round(speed, 1),
        "mode": mode,
        "confidence": "high" if mode != "partial_gpu" else "medium"
    }