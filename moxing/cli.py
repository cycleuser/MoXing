"""
CLI interface for pyllm
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

console = Console()
app = typer.Typer(name="pyllm", help="Python wrapper for llama.cpp server")


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model name or path to GGUF file"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization type"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source (huggingface/modelscope/auto)"),
    auto: bool = typer.Option(True, "--auto/--no-auto", help="Auto-detect best device"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """Start the llama.cpp server with automatic configuration."""
    from moxing.runner import AutoRunner
    
    runner = AutoRunner(auto_detect_device=auto)
    
    try:
        server = runner.server(
            model=model,
            quant=quant,
            source=source,
            ctx_size=ctx_size,
            port=port,
            verbose=verbose
        )
        
        console.print(Panel(
            f"[green]Server running at:[/green] http://{host}:{port}\n"
            f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
            f"[yellow]Press Ctrl+C to stop[/yellow]",
            title="pyllm server"
        ))
        
        server.start(wait=False)
        while server.is_running():
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if runner._current_server:
            runner._current_server.stop()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or path"),
    prompt: str = typer.Option("Hello!", "-p", "--prompt", help="Prompt to send"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization"),
    tokens: int = typer.Option(256, "-n", "--tokens", help="Max tokens to generate"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source"),
    chat: bool = typer.Option(True, "--chat/--completion", help="Chat or completion mode"),
):
    """Run inference with a model (auto-downloads if needed)."""
    from moxing.runner import AutoRunner
    
    runner = AutoRunner()
    
    try:
        result = runner.run(
            model=model,
            prompt=prompt,
            quant=quant,
            source=source,
            ctx_size=ctx_size,
            n_tokens=tokens,
            chat=chat
        )
        console.print(result)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("chat")
def chat_cmd(
    model: str = typer.Argument(..., help="Model name or path"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source"),
):
    """Interactive chat with a model."""
    from moxing.runner import AutoRunner
    from moxing.client import Client
    
    runner = AutoRunner()
    
    try:
        server = runner.server(model=model, quant=quant, source=source, ctx_size=ctx_size)
        server.start()
        
        console.print("[green]Chat ready! Type 'exit' or 'quit' to end.[/green]\n")
        
        messages = []
        
        while True:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if user_input.lower() in ("exit", "quit", "q"):
                break
            
            messages.append({"role": "user", "content": user_input})
            
            client = Client(server.base_url)
            response = client.chat.completions.create(
                model="llama",
                messages=messages,
                max_tokens=512
            )
            
            if response.choices:
                assistant_msg = response.choices[0].get("message", {}).get("content", "")
                console.print(f"[bold green]Assistant[/bold green]: {assistant_msg}")
                messages.append({"role": "assistant", "content": assistant_msg})
        
        server.stop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def download(
    model: str = typer.Argument(..., help="Model name (e.g., llama-3.2-3b) or repo (user/model)"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization type"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source (huggingface/modelscope/auto)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
    list_files: bool = typer.Option(False, "-l", "--list", help="List available files"),
):
    """Download a model from HuggingFace or ModelScope."""
    from moxing.models import ModelDownloader, ModelRegistry
    
    downloader = ModelDownloader(output)
    
    registry_info = ModelRegistry.get_model_info(model, source)
    
    if registry_info:
        repo = registry_info["repo"]
        console.print(f"[blue]Found in registry: {registry_info['description']}[/blue]")
    else:
        repo = model
    
    if list_files:
        files = downloader.list_files(repo, source)
        if not files:
            console.print("[red]No GGUF files found[/red]")
            return
        
        table = Table(title=f"Available files in {repo}")
        table.add_column("Filename", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("Quant", style="green")
        
        for filename, size in files:
            size_str = f"{size / (1024**3):.2f} GB" if size > 0 else "unknown"
            quant_str = downloader._extract_quantization(filename)
            table.add_row(filename, size_str, quant_str)
        
        console.print(table)
        return
    
    try:
        path = downloader.download(repo, None if quant == "auto" else f"*{quant}*", source, output)
        console.print(f"[green]Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def models(
    local: bool = typer.Option(False, "-l", "--local", help="Show local models only"),
    search: Optional[str] = typer.Option(None, "--search", help="Search for models"),
):
    """List available models."""
    from moxing.models import ModelDownloader, ModelRegistry
    from moxing.runner import AutoRunner
    
    if local:
        runner = AutoRunner()
        models_list = runner.list_local_models()
        
        if not models_list:
            console.print("[yellow]No local models found[/yellow]")
            return
        
        table = Table(title="Local Models")
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Quant", style="magenta")
        
        for m in models_list:
            table.add_row(
                m.name,
                str(m.local_path.parent),
                f"{m.size_gb:.2f} GB",
                m.quantization
            )
        
        console.print(table)
    elif search:
        downloader = ModelDownloader()
        results = downloader.search(search)
        
        if not results:
            console.print(f"[yellow]No models found for '{search}'[/yellow]")
            return
        
        table = Table(title=f"Search Results for '{search}'")
        table.add_column("Repo", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Source", style="magenta")
        
        for m in results[:20]:
            table.add_row(
                m.repo,
                m.filename[:50] + "..." if len(m.filename) > 50 else m.filename,
                f"{m.size_gb:.2f} GB",
                m.source
            )
        
        console.print(table)
    else:
        runner = AutoRunner()
        runner.list_available_models()


@app.command()
def devices():
    """List available GPU devices and their capabilities."""
    from moxing.device import DeviceDetector
    
    detector = DeviceDetector()
    detector.detect()
    detector.list_devices()


@app.command()
def config(
    model: str = typer.Argument(..., help="Model path to analyze"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Desired context size"),
):
    """Show optimal configuration for a model."""
    from moxing.runner import AutoRunner
    
    if not Path(model).exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)
    
    runner = AutoRunner()
    config = runner.detect_config(model, ctx_size)
    
    console.print(Panel(
        f"[green]Model:[/green] {config.model_path}\n"
        f"[blue]Backend:[/blue] {config.device_config.backend.value}\n"
        f"[yellow]Device:[/yellow] {config.device_config.device}\n"
        f"[magenta]GPU Layers:[/magenta] {config.device_config.n_gpu_layers if config.device_config.n_gpu_layers >= 0 else 'all'}\n"
        f"[cyan]Recommended Context:[/cyan] {config.device_config.recommended_ctx}\n"
        f"[dim]{config.device_config.notes}[/dim]",
        title="Recommended Configuration"
    ))


@app.command("build")
def build_binary(
    backend: str = typer.Option("vulkan", "-b", "--backend", help="GPU backend (vulkan, cuda, rocm, cpu)"),
    jobs: int = typer.Option(8, "-j", "--jobs", help="Parallel jobs"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory for binaries"),
):
    """Build llama.cpp binaries from source."""
    console.print(f"[blue]Building llama.cpp with {backend} backend...[/blue]")
    
    llama_cpp_dir = Path(__file__).parent.parent.parent
    build_dir = llama_cpp_dir / "build"
    
    cmake_args = [
        "cmake", "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    
    if backend == "vulkan":
        cmake_args.append("-DGGML_VULKAN=ON")
    elif backend == "cuda":
        cmake_args.append("-DGGML_CUDA=ON")
    elif backend == "rocm":
        cmake_args.append("-DGGML_HIP=ON")
    
    console.print(f"[dim]Running: {' '.join(cmake_args)}[/dim]")
    subprocess.run(cmake_args, cwd=llama_cpp_dir, check=True)
    
    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(jobs)]
    subprocess.run(build_cmd, cwd=llama_cpp_dir, check=True)
    
    if output:
        import shutil
        output.mkdir(parents=True, exist_ok=True)
        for exe in (build_dir / "bin").glob("llama-*.exe" if sys.platform == "win32" else "llama-*"):
            shutil.copy2(exe, output / exe.name)
        console.print(f"[green]Binaries copied to: {output}[/green]")
    else:
        console.print(f"[green]Build complete! Binaries at: {build_dir / 'bin'}[/green]")


@app.command("download-binaries")
def download_binaries(
    backend: str = typer.Option("auto", "-b", "--backend", help="GPU backend (auto, vulkan, cuda, metal, cpu)"),
    force: bool = typer.Option(False, "-f", "--force", help="Force re-download"),
):
    """Download pre-built llama.cpp binaries."""
    from moxing.binaries import get_binary_manager
    
    manager = get_binary_manager()
    
    console.print(f"[blue]Downloading binaries for {manager.platform}...[/blue]")
    
    try:
        manager.download_binaries(backend=backend, force=force)
        
        binaries = manager.list_cached_binaries()
        console.print(f"\n[green]Installed binaries:[/green]")
        for b in binaries[:10]:
            console.print(f"  - {b}")
        if len(binaries) > 10:
            console.print(f"  ... and {len(binaries) - 10} more")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("clear-cache")
def clear_cache(
    model: Optional[str] = typer.Argument(None, help="Specific model to remove"),
    binaries: bool = typer.Option(False, "--binaries", help="Clear binary cache"),
):
    """Clear model and/or binary cache."""
    if binaries or model is None:
        from moxing.binaries import get_binary_manager
        manager = get_binary_manager()
        manager.clear_cache()
        console.print("[green]Binary cache cleared[/green]")
    
    if model or not binaries:
        from moxing.models import ModelDownloader
        downloader = ModelDownloader()
        downloader.clear_cache(model)
        console.print("[green]Model cache cleared[/green]")
    console.print("[green]Cache cleared[/green]")


@app.command("diagnose")
def diagnose(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    install: bool = typer.Option(False, "--install", "-i", help="Auto-install after diagnosis"),
):
    """Diagnose system and show installation recommendations."""
    import subprocess
    
    script_path = Path(__file__).parent.parent / "scripts" / "detect_and_install.py"
    
    if not script_path.exists():
        console.print("[yellow]Running built-in diagnostics...[/yellow]")
        
        from moxing.device import DeviceDetector, BackendType
        
        detector = DeviceDetector()
        devices = detector.detect()
        
        if json_output:
            import json
            data = {
                "platform": sys.platform,
                "python_version": platform.python_version(),
                "devices": [
                    {
                        "index": d.index,
                        "name": d.name,
                        "backend": d.backend.value,
                        "memory_mb": d.memory_mb,
                        "vendor": d.vendor,
                    }
                    for d in devices
                ],
                "recommended_backend": min(
                    [d.backend for d in devices if d.backend != BackendType.CPU],
                    default=BackendType.CPU
                ).value,
            }
            print(json.dumps(data, indent=2))
        else:
            console.print(Panel(
                f"[cyan]Platform:[/cyan] {sys.platform}\n"
                f"[cyan]Python:[/cyan] {platform.python_version()}\n"
                f"[cyan]Devices:[/cyan] {len(devices)} found",
                title="System Diagnostics"
            ))
            
            if devices:
                table = Table(title="Detected Devices")
                table.add_column("Index", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Backend", style="magenta")
                table.add_column("Memory", style="yellow")
                
                for d in devices:
                    mem = f"{d.memory_gb:.1f}GB" if d.memory_mb > 0 else "N/A"
                    table.add_row(str(d.index), d.name, d.backend.value, mem)
                
                console.print(table)
            
            if install:
                console.print("\n[blue]Starting automatic installation...[/blue]")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "pyllm-server"])
        return
    
    cmd = [sys.executable, str(script_path)]
    if json_output:
        cmd.append("--json")
    if install:
        cmd.append("--install")
    
    subprocess.run(cmd)


@app.command("bench")
def benchmark(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    prompt: str = typer.Option("standard", "-p", "--prompt", help="Prompt type: quick, standard, code, creative, or custom text"),
    n_tokens: int = typer.Option(128, "-n", "--tokens", help="Number of tokens to generate"),
    n_runs: int = typer.Option(1, "-r", "--runs", help="Number of benchmark runs"),
    warmup: bool = typer.Option(True, "-w", "--warmup", help="Run warmup iteration"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    compare: Optional[str] = typer.Option(None, "--compare", help="Second model to compare"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Benchmark model performance (tokens/second, memory usage)."""
    from moxing.benchmark import BenchmarkRunner, estimate_speed
    from moxing.device import DeviceDetector, BackendType
    import time
    
    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)
    
    model_size_gb = model_path.stat().st_size / (1024 ** 3)
    
    console.print(Panel(
        f"[cyan]Model:[/cyan] {model_path.name}\n"
        f"[cyan]Size:[/cyan] {model_size_gb:.2f} GB\n"
        f"[cyan]Tokens:[/cyan] {n_tokens}\n"
        f"[cyan]Runs:[/cyan] {n_runs}",
        title="Benchmark Configuration"
    ))
    
    runner = BenchmarkRunner(verbose=False)
    
    models_to_bench = [model]
    if compare:
        if Path(compare).exists():
            models_to_bench.append(compare)
        else:
            console.print(f"[yellow]Warning: Compare model not found: {compare}[/yellow]")
    
    results = []
    
    for i, m in enumerate(models_to_bench):
        if len(models_to_bench) > 1:
            console.print(f"\n[bold]Benchmarking model {i+1}/{len(models_to_bench)}: {Path(m).name}[/bold]")
        
        try:
            result = runner.run(
                model=m,
                prompt=prompt,
                n_tokens=n_tokens,
                n_runs=n_runs,
                warmup=warmup,
                ctx_size=ctx_size
            )
            results.append(result)
            
            if not json_output and len(models_to_bench) == 1:
                runner.print_results(result)
        
        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            raise typer.Exit(1)
    
    if json_output:
        output = {
            "model": results[0].model,
            "model_size_gb": results[0].model_size_gb,
            "prompt_tokens": results[0].prompt_tokens,
            "completion_tokens": results[0].completion_tokens,
            "tokens_per_second": round(results[0].tokens_per_second, 2),
            "prompt_tokens_per_second": round(results[0].prompt_tokens_per_second, 2),
            "total_time_sec": round(results[0].total_time_sec, 2),
            "peak_memory_mb": round(results[0].peak_memory_mb, 2),
        }
        print(json.dumps(output, indent=2))
    
    elif len(results) > 1:
        runner.print_comparison(results)
    
    console.print()
    speed_display = results[0].tokens_per_second
    console.print(f"[bold green]Speed: {speed_display:.2f} tokens/second[/bold green]")


@app.command("speed")
def speed_test(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    prompt: str = typer.Option("Hello, how are you?", "-p", "--prompt", help="Test prompt"),
    ctx_size: int = typer.Option(2048, "-c", "--ctx-size", help="Context size"),
):
    """Quick speed test with detailed output similar to ollama."""
    from moxing import LlamaServer, Client
    from moxing.device import DeviceDetector
    import time
    
    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)
    
    detector = DeviceDetector()
    devices = detector.detect()
    device_config = detector.get_best_device(model_path.stat().st_size / (1024**3))
    
    model_size_gb = model_path.stat().st_size / (1024 ** 3)
    
    console.print()
    console.print(f"[bold cyan]Model:[/bold cyan] {model_path.name}")
    console.print(f"[bold cyan]Size:[/bold cyan] {model_size_gb:.2f} GB")
    console.print(f"[bold cyan]Device:[/bold cyan] {device_config.device.name} ({device_config.backend.value})")
    console.print()
    
    port = 8080 + hash(str(model_path)) % 1000
    
    server = LlamaServer(
        model=str(model_path),
        port=port,
        ctx_size=ctx_size,
        n_gpu_layers=device_config.n_gpu_layers,
        device=f"{device_config.backend.value.capitalize()}{device_config.device.index}"
    )
    
    try:
        console.print("[dim]Loading model...[/dim]")
        start_load = time.time()
        server.start(timeout=120)
        load_time = time.time() - start_load
        
        console.print(f"[green]Model loaded in {load_time:.2f}s[/green]")
        console.print()
        
        client = Client(server.base_url)
        
        console.print(f"[bold]Prompt:[/bold] {prompt}")
        console.print()
        
        console.print("[bold green]Generating...[/bold green]")
        
        prompt_start = time.time()
        
        response = client.chat.completions.create(
            model="llama",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            stream=True
        )
        
        generated_text = ""
        first_token_time = None
        token_count = 0
        
        for chunk in response:
            if isinstance(chunk, dict) and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    generated_text += content
                    token_count += 1
                    print(content, end="", flush=True)
        
        total_time = time.time() - prompt_start
        
        if token_count > 0 and total_time > 0:
            tokens_per_second = token_count / total_time
        else:
            tokens_per_second = 0
        
        console.print()
        console.print()
        console.print(Panel(
            f"[green]Total tokens:[/green] {token_count}\n"
            f"[green]Time:[/green] {total_time:.2f}s\n"
            f"[green]Speed:[/green] {tokens_per_second:.2f} tokens/s\n"
            f"[green]Time to first token:[/green] {first_token_time - prompt_start:.2f}s" if first_token_time else "",
            title="Performance"
        ))
        
    finally:
        server.stop()


@app.command("info")
def model_info(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
):
    """Show detailed model information and estimated performance."""
    from moxing.device import DeviceDetector
    from moxing.benchmark import estimate_speed
    import struct
    
    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)
    
    model_size_gb = model_path.stat().st_size / (1024 ** 3)
    
    detector = DeviceDetector()
    devices = detector.detect()
    device_config = detector.get_best_device(model_size_gb)
    
    console.print(Panel(
        f"[cyan]File:[/cyan] {model_path.name}\n"
        f"[cyan]Path:[/cyan] {model_path}\n"
        f"[cyan]Size:[/cyan] {model_size_gb:.2f} GB",
        title="Model Information"
    ))
    
    table = Table(title="Recommended Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Notes", style="yellow")
    
    table.add_row("Backend", device_config.backend.value, "Best available for your hardware")
    table.add_row("Device", device_config.device.name, "")
    table.add_row("GPU Layers", str(device_config.n_gpu_layers) if device_config.n_gpu_layers >= 0 else "all", "")
    table.add_row("Context Size", str(device_config.recommended_ctx), "Based on available memory")
    table.add_row("Notes", device_config.notes, "")
    
    console.print(table)
    
    if devices:
        console.print()
        gpu_table = Table(title="Available GPUs")
        gpu_table.add_column("Device", style="cyan")
        gpu_table.add_column("Memory", style="green")
        gpu_table.add_column("Est. Speed", style="yellow")
        
        for d in devices:
            if d.backend.value != "cpu":
                est = estimate_speed(model_size_gb, d.memory_gb, d.backend.value)
                gpu_table.add_row(
                    f"{d.name}",
                    f"{d.memory_gb:.1f} GB",
                    f"~{est['estimated_tokens_per_second']} t/s ({est['mode']})"
                )
        
        console.print(gpu_table)


if __name__ == "__main__":
    app()