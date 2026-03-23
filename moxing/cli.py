"""
CLI interface for moxing
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
app = typer.Typer(
    name="moxing",
    help="Python wrapper for llama.cpp server",
    add_completion=False
)


def version_callback(value: bool):
    if value:
        from moxing import __version__
        from moxing.binaries import get_binary_manager
        
        console.print(f"[bold]MoXing[/bold] version: {__version__}")
        
        try:
            manager = get_binary_manager()
            binary_version = manager.get_installed_version()
            if binary_version:
                console.print(f"[bold]Binaries:[/bold] {binary_version} ({manager.backend})")
            else:
                console.print("[bold]Binaries:[/bold] not installed")
        except Exception:
            pass
        
        console.print(f"[bold]Python:[/bold] {sys.version.split()[0]}")
        console.print(f"[bold]Platform:[/bold] {platform.system()} {platform.machine()}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version", "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True
    ),
):
    pass


@app.command()
def version_cmd():
    """Show version information."""
    from moxing import __version__
    from moxing.binaries import get_binary_manager, PlatformDetector
    
    console.print(f"\n[bold cyan]MoXing[/bold cyan] version {__version__}")
    
    console.print(f"\n[bold]Platform:[/bold]")
    console.print(f"  OS: {PlatformDetector.get_os()}")
    console.print(f"  Arch: {PlatformDetector.get_arch()}")
    console.print(f"  Detected backend: {PlatformDetector.detect_backend()}")
    
    try:
        manager = get_binary_manager()
        binary_version = manager.get_installed_version()
        console.print(f"\n[bold]Binaries:[/bold]")
        if binary_version:
            console.print(f"  Version: {binary_version}")
            console.print(f"  Backend: {manager.backend}")
            console.print(f"  Path: {manager.get_cache_dir()}")
        else:
            console.print("  Not installed (will download on first use)")
            console.print(f"  Will use: {manager.backend}")
    except Exception as e:
        console.print(f"\n[yellow]Binary info unavailable: {e}[/yellow]")
    
    console.print(f"\n[bold]Python:[/bold] {sys.version.split(0)[0]}")
    console.print(f"[bold]Python path:[/bold] {sys.executable}")


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model name, path to GGUF file, HuggingFace repo, or ollama:model"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization type"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
    ctx_size: int = typer.Option(0, "-c", "--ctx-size", help="Context size (0=auto-detect based on VRAM)"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source (huggingface/modelscope/auto)"),
    backend: str = typer.Option("auto", "-b", "--backend", help="Backend: auto, llama.cpp, mlx, ollama"),
    auto: bool = typer.Option(True, "--auto/--no-auto", help="Auto-detect best device"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    force: bool = typer.Option(False, "-f", "--force", help="Force use specified backend without compatibility check"),
):
    """Start the LLM server with automatic configuration.
    
    Backends:
    - auto: Automatically choose best backend based on model compatibility
    - llama.cpp: Use llama.cpp (default for GGUF files)
    - mlx: Use Apple MLX framework (macOS only, supports newer models)
    - ollama: Use Ollama (requires Ollama installed)
    
    Examples:
    - GGUF file: moxing serve model.gguf
    - HuggingFace: moxing serve Qwen/Qwen2.5-3B-Instruct -b mlx
    - Ollama model: moxing serve ollama:gemma3:4b
    - Ollama model: moxing serve gemma3:4b -b ollama
    """
    from moxing.runner import AutoRunner
    from moxing.mlx_server import MLXServer
    from moxing.gguf_check import diagnose_gguf, print_diagnosis, GGUFParser
    from moxing.ollama import OllamaClient, get_ollama_model
    from moxing.gguf_compress import is_gguf_compressed, resolve_model_path
    
    if model.startswith("ollama:"):
        ollama_model = model[7:]
        ollama_serve(ollama_model, port, host)
        return
    
    if backend == "ollama":
        ollama_serve(model, port, host)
        return
    
    model_file = Path(model)
    
    # Handle compressed GGUF files - decompress first
    if model_file.exists() and is_gguf_compressed(model_file):
        console.print("[blue]Detected compressed GGUF, decompressing...[/blue]")
        decompressed_path = resolve_model_path(model_file)
        model = str(decompressed_path)
        model_file = decompressed_path
        console.print(f"[green]Decompressed to cache: {decompressed_path.name}[/green]")
    
    model_path = model_file if model_file.exists() else None
    
    # Check if it's a GGUF file by extension or by trying to parse it
    is_gguf = False
    if model_path:
        if model_path.suffix == ".gguf" or str(model).endswith(".gguf"):
            is_gguf = True
        else:
            # Try to check magic bytes
            try:
                import struct
                with open(model_path, "rb") as f:
                    magic = struct.unpack("<I", f.read(4))[0]
                    if magic == 0x46554747:  # "GGUF"
                        is_gguf = True
            except:
                pass
    
    use_mlx = False
    gguf_compatible = True
    
    if backend == "mlx":
        if not MLXServer.is_available():
            console.print("[red]MLX backend is only available on macOS with Apple Silicon[/red]")
            raise typer.Exit(1)
        use_mlx = True
    elif backend == "auto":
        if is_gguf and model_path:
            if MLXServer.is_available() and not force:
                try:
                    parser = GGUFParser(model_path)
                    meta = parser.parse()
                    
                    if not meta.is_valid:
                        gguf_compatible = False
                        console.print(f"[yellow]GGUF compatibility issues detected:[/yellow]")
                        for err in meta.errors[:3]:
                            console.print(f"  [red]✗[/red] {err}")
                        
                        console.print(f"\n[blue]Switching to MLX backend for better compatibility...[/blue]")
                        console.print(f"[dim]Use --force to try llama.cpp anyway[/dim]\n")
                        use_mlx = True
                        
                except Exception as e:
                    console.print(f"[yellow]Could not check GGUF compatibility: {e}[/yellow]")
            
            if gguf_compatible and not use_mlx:
                console.print("[blue]Using llama.cpp backend for GGUF file[/blue]")
        else:
            if MLXServer.is_available():
                use_mlx = True
                console.print("[blue]Auto-detected: Using MLX backend for HuggingFace model[/blue]")
    elif backend == "llama.cpp" and is_gguf and model_path and not force:
        try:
            parser = GGUFParser(model_path)
            meta = parser.parse()
            
            if not meta.is_valid:
                console.print(f"[yellow]Warning: GGUF compatibility issues detected[/yellow]")
                for err in meta.errors[:3]:
                    console.print(f"  [red]✗[/red] {err}")
                
                if MLXServer.is_available():
                    console.print(f"\n[yellow]Consider using MLX backend: moxing serve {model} -b mlx[/yellow]")
                console.print(f"[dim]Use --force to proceed anyway[/dim]\n")
        except Exception:
            pass
    
    if use_mlx:
        try:
            server = MLXServer(model=model, host=host, port=port)
            
            console.print(Panel(
                f"[green]Server running at:[/green] http://{host}:{port}\n"
                f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
                f"[magenta]Backend:[/magenta] MLX (Apple Silicon)\n"
                f"[yellow]Press Ctrl+C to stop[/yellow]",
                title="moxing server (MLX)"
            ))
            
            server.start(wait=False)
            while server.is_running():
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            server.stop()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    else:
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
                f"[magenta]Backend:[/magenta] llama.cpp\n"
                f"[yellow]Press Ctrl+C to stop[/yellow]",
                title="moxing server"
            ))
            
            try:
                server.start(wait=False)
            except RuntimeError as e:
                console.print(f"\n[red bold]Failed to start server![/red bold]")
                console.print(f"[red]{e}[/red]")
                
                from moxing.gguf_check import diagnose_gguf, get_model_suggestions
                model_path = Path(model)
                if model_path.exists() and model.endswith(".gguf"):
                    console.print(f"\n[blue]Checking model compatibility...[/blue]")
                    try:
                        meta = diagnose_gguf(model_path)
                        if not meta.is_valid:
                            console.print(f"[yellow]Issues found:[/yellow]")
                            for err in meta.errors:
                                console.print(f"  [red]✗[/red] {err}")
                    except:
                        pass
                    
                    console.print(f"\n[blue]Suggestions:[/blue]")
                    for s in get_model_suggestions(model_path):
                        console.print(f"  • {s}")
                
                raise typer.Exit(1)
            
            while server.is_running():
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            if runner._current_server:
                runner._current_server.stop()
        except RuntimeError:
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
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
    from moxing.binaries import get_binary_manager, list_available_backends
    
    manager = get_binary_manager(backend)
    
    console.print(f"[blue]Downloading binaries for {manager.platform_name} ({manager.backend})...[/blue]")
    
    available = list_available_backends()
    console.print(f"[dim]Available bundled backends: {available}[/dim]")
    
    try:
        manager.download_binaries(force=force)
        
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
                subprocess.run([sys.executable, "-m", "pip", "install", "moxing"])
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


@app.command("check")
def check_model(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Suggest fixes for compatibility issues"),
):
    """Check GGUF model compatibility with llama.cpp."""
    from moxing.gguf_check import diagnose_gguf, print_diagnosis, get_model_suggestions
    
    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)
    
    if not model.endswith(".gguf"):
        console.print(f"[yellow]Warning: File does not have .gguf extension[/yellow]")
    
    console.print(f"[blue]Analyzing GGUF file: {model}[/blue]\n")
    
    try:
        meta = diagnose_gguf(model_path)
        print_diagnosis(meta)
        
        if fix and not meta.is_valid:
            console.print("\n[blue]Suggested fixes:[/blue]")
            for suggestion in get_model_suggestions(model_path):
                console.print(f"  • {suggestion}")
    
    except Exception as e:
        console.print(f"[red]Failed to parse GGUF file: {e}[/red]")
        raise typer.Exit(1)


ollama_app = typer.Typer(name="ollama", help="Manage Ollama models")
app.add_typer(ollama_app, name="ollama")


@ollama_app.command("list")
def ollama_list(
    show_embeddings: bool = typer.Option(True, "--embeddings/--no-embeddings", help="Show embedding models"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    select: bool = typer.Option(False, "--select", "-s", help="Interactive model selection"),
):
    """List all models from local Ollama installation."""
    from moxing.ollama import list_ollama_models, print_ollama_models, OllamaClient
    
    models = list_ollama_models()
    
    if not show_embeddings:
        client = OllamaClient()
        models = [m for m in models if not client.is_embedding_model(m.full_name)]
    
    if json_output:
        import json
        data = [
            {
                "name": m.full_name,
                "size": m.size,
                "size_gb": round(m.size_gb, 2),
                "id": m.id,
                "modified": m.modified
            }
            for m in models
        ]
        print(json.dumps(data, indent=2))
    elif select:
        if not models:
            console.print("[yellow]No Ollama models found.[/yellow]")
            return
        
        console.print("\n[bold]Select a model to serve:[/bold]\n")
        
        for i, m in enumerate(models, 1):
            size_str = f"{m.size_gb:.1f} GB" if m.size_gb >= 1 else f"{m.size / (1024**2):.0f} MB"
            console.print(f"  [cyan]{i:2d}[/cyan]. {m.full_name:<40} [dim]{size_str}[/dim]")
        
        console.print()
        selection = Prompt.ask(
            "[bold]Enter number[/bold]",
            default="1"
        )
        
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                console.print(f"\n[green]Selected: {selected.full_name}[/green]\n")
                
                action = Prompt.ask(
                    "[bold]Action[/bold]",
                    choices=["serve", "info", "run", "cancel"],
                    default="serve"
                )
                
                if action == "serve":
                    ollama_serve(selected.full_name)
                elif action == "info":
                    ollama_info(selected.full_name)
                elif action == "run":
                    ollama_run(selected.full_name)
                else:
                    console.print("[yellow]Cancelled[/yellow]")
            else:
                console.print("[red]Invalid selection[/red]")
        except ValueError:
            console.print("[red]Invalid input[/red]")
    else:
        print_ollama_models(models, show_embeddings)


@ollama_app.command("serve")
def ollama_serve(
    model: str = typer.Argument(..., help="Ollama model name (e.g., gemma3:4b)"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
):
    """Serve an Ollama model with OpenAI-compatible API.
    
    Uses moxing's llama.cpp to run the model with optimal GPU acceleration.
    If the Ollama GGUF is incompatible, downloads a compatible version from Hugging Face.
    
    Examples:
        moxing ollama serve gemma3:4b
        moxing ollama serve llama3.2:3b -c 8192
    """
    from moxing.ollama import OllamaClient, get_ollama_model
    from moxing.server import LlamaServer
    from moxing.device import DeviceDetector
    
    client = OllamaClient()
    
    ollama_model = get_ollama_model(model)
    if not ollama_model:
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("[yellow]Available models:[/yellow]")
        models = client.list_models()
        for m in models[:10]:
            console.print(f"  • {m.full_name}")
        if len(models) > 10:
            console.print(f"  ... and {len(models) - 10} more")
        console.print("\n[dim]Run 'moxing ollama list' to see all models[/dim]")
        raise typer.Exit(1)
    
    is_accessible, gguf_path, message = client.check_model_access(model)
    
    if not is_accessible:
        console.print(f"[red]Cannot access model '{model}':[/red]")
        console.print(message)
        raise typer.Exit(1)
    
    gguf_path = _get_compatible_gguf(model, gguf_path, ollama_model.size_gb)
    if not gguf_path:
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[green]Model:[/green] {ollama_model.full_name}\n"
        f"[blue]Size:[/blue] {ollama_model.size_gb:.1f} GB\n"
        f"[blue]GGUF:[/blue] {gguf_path.name[:50]}...\n"
        f"[yellow]Port:[/yellow] {port}\n"
        f"[magenta]Backend:[/magenta] llama.cpp (moxing)",
        title="Ollama Model"
    ))
    
    detector = DeviceDetector()
    devices = detector.detect()
    device_config = detector.get_best_device(ollama_model.size_gb)
    
    device_str = f"MTL{device_config.device.index}" if device_config.backend.value == "metal" else "auto"
    
    console.print(Panel(
        f"[green]Model:[/green] {ollama_model.full_name}\n"
        f"[blue]Backend:[/blue] {device_config.backend.value}\n"
        f"[yellow]Device:[/yellow] {device_config.device.name}\n"
        f"[magenta]GPU Layers:[/magenta] all\n"
        f"[cyan]Context:[/cyan] {ctx_size}",
        title="Configuration"
    ))
    
    server = None
    try:
        server = LlamaServer(
            model=str(gguf_path),
            host=host,
            port=port,
            ctx_size=ctx_size,
            n_gpu_layers=-1,
            device=device_str,
            gpu_backend=device_config.backend.value
        )
        
        console.print(Panel(
            f"[green]Server running at:[/green] http://{host}:{port}\n"
            f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
            f"[magenta]Backend:[/magenta] llama.cpp\n"
            f"[yellow]Press Ctrl+C to stop[/yellow]",
            title="moxing server"
        ))
        
        server.start(wait=False)
        
        import time
        time.sleep(2)
        
        if server._process and server._process.poll() is not None:
            stdout, stderr = server._process.communicate()
            console.print(f"\n[red bold]Server failed to start![/red bold]")
            for line in (stderr or "").strip().split("\n")[-15:]:
                if line.strip():
                    console.print(f"[red]{line}[/red]")
            raise typer.Exit(1)
        
        while server.is_running():
            time.sleep(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if server:
            server.stop()
    except RuntimeError:
        raise typer.Exit(1)


def _get_compatible_gguf(model: str, ollama_gguf: Path, size_gb: float) -> Optional[Path]:
    """Get a compatible GGUF file, downloading from Hugging Face if Ollama's is incompatible."""
    from moxing.binaries import get_binary_manager
    
    manager = get_binary_manager()
    if not manager.has_binaries():
        console.print("[blue]Downloading llama.cpp binaries...[/blue]")
        manager.download_binaries()
    
    llama_cli = manager.get_binary_path("llama-cli")
    
    console.print("[dim]Loading GGUF model...[/dim]")
    
    incompatible_patterns = [
        "error loading model",
        "wrong number of tensors",
        "wrong shape",
        "key not found in model",
        "missing tensor",
        "failed to load model",
        "done_getting_tensors",
    ]
    
    stderr = ""
    try:
        proc = subprocess.Popen(
            [str(llama_cli), "-m", str(ollama_gguf), "-n", "1", "-p", "x", "-c", "128"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            _, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
        
        if stderr:
            stderr_lower = stderr.lower()
            is_incompatible = any(p in stderr_lower for p in incompatible_patterns)
            
            if is_incompatible:
                console.print(f"[yellow]⚠ Ollama GGUF is incompatible with llama.cpp[/yellow]")
                console.print(f"[dim]Ollama uses a custom GGUF format. Downloading compatible version...[/dim]")
                
                hf_repo = _get_hf_repo_for_ollama_model(model)
                if not hf_repo:
                    console.print(f"[red]No compatible GGUF found on Hugging Face for: {model}[/red]")
                    console.print(f"[dim]Suggestion: Download manually with:[/dim]")
                    console.print(f"[cyan]  moxing download <repo>[/cyan]")
                    console.print(f"[dim]Then serve with: moxing serve <path/to/model.gguf>[/dim]")
                    return None
                
                console.print(f"[blue]Downloading from Hugging Face: {hf_repo}[/blue]")
                
                try:
                    from moxing.models import download_model
                    gguf_path = download_model(hf_repo)
                    return gguf_path
                except Exception as e:
                    console.print(f"[red]Failed to download: {e}[/red]")
                    return None
            
            console.print(f"[green]✓ GGUF is compatible[/green]")
            return ollama_gguf
            
    except Exception as e:
        console.print(f"[yellow]⚠ Compatibility check error: {e}[/yellow]")
    
    return ollama_gguf


def _get_hf_repo_for_ollama_model(model: str) -> Optional[str]:
    """Map Ollama model name to Hugging Face GGUF repository."""
    model_lower = model.lower().split(":")[0]
    
    OLLAMA_TO_HF_MAP = {
        "gemma3": "google/gemma-3-4b-it-GGUF",
        "gemma3:1b": "google/gemma-3-1b-it-GGUF",
        "gemma3:4b": "google/gemma-3-4b-it-GGUF",
        "gemma3:12b": "google/gemma-3-12b-it-GGUF",
        "gemma3:27b": "google/gemma-3-27b-it-GGUF",
        "gemma2": "google/gemma-2-9b-it-GGUF",
        "gemma2:2b": "google/gemma-2-2b-it-GGUF",
        "gemma2:9b": "google/gemma-2-9b-it-GGUF",
        "gemma2:27b": "google/gemma-2-27b-it-GGUF",
        "llama3.2": "meta-llama/Llama-3.2-3B-Instruct-GGUF",
        "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct-GGUF",
        "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct-GGUF",
        "llama3.3": "meta-llama/Llama-3.3-70B-Instruct-GGUF",
        "llama3.1": "meta-llama/Llama-3.1-8B-Instruct-GGUF",
        "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct-GGUF",
        "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct-GGUF",
        "llama3": "meta-llama/Llama-3-8B-Instruct-GGUF",
        "llama3:8b": "meta-llama/Llama-3-8B-Instruct-GGUF",
        "llama3:70b": "meta-llama/Llama-3-70B-Instruct-GGUF",
        "qwen2.5": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "qwen2.5:72b": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "qwen3": "Qwen/Qwen3-8B-GGUF",
        "qwen3:0.6b": "Qwen/Qwen3-0.6B-GGUF",
        "qwen3:1.7b": "Qwen/Qwen3-1.7B-GGUF",
        "qwen3:4b": "Qwen/Qwen3-4B-GGUF",
        "qwen3:8b": "Qwen/Qwen3-8B-GGUF",
        "qwen3:14b": "Qwen/Qwen3-14B-GGUF",
        "qwen3.5": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen3.5:0.8b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen3.5:2b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen3.5:4b": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen3.5:9b": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen3.5:27b": "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "mistral": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral:7b": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mixtral": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "mixtral:8x7b": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "mixtral:8x22b": "MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF",
        "codellama": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "codellama:7b": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "codellama:13b": "TheBloke/CodeLlama-13B-Instruct-GGUF",
        "codellama:34b": "TheBloke/CodeLlama-34B-Instruct-GGUF",
        "deepseek-coder": "TheBloke/deepseek-coder-6.7B-Instruct-GGUF",
        "deepseek-coder:6.7b": "TheBloke/deepseek-coder-6.7B-Instruct-GGUF",
        "phi3": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "phi3:3.8b": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "phi3:14b": "microsoft/Phi-3-medium-4k-instruct-gguf",
        "gemma3n": "google/gemma-3-4b-it-GGUF",
        "gemma3n:e2b": "google/gemma-3-1b-it-GGUF",
        "gemma3n:e4b": "google/gemma-3-4b-it-GGUF",
        "translategemma": "catchmeifyoucan/Qwen2.5-7B-Instruct-GGUF",
    }
    
    for key, hf_repo in OLLAMA_TO_HF_MAP.items():
        key_lower = key.lower()
        # Direct match
        if model_lower == key_lower:
            return hf_repo
        # Match with tag (e.g., "qwen3.5:4b" matches "qwen3.5")
        if model_lower.startswith(key_lower.replace(":", "-") + "-") or model_lower.startswith(key_lower + ":"):
            return hf_repo
        # Match model name without owner prefix (e.g., "huihui_ai/qwen3.5-abliterated:4b" matches "qwen3.5")
        model_without_owner = model_lower.split("/")[-1] if "/" in model_lower else model_lower
        model_base = model_without_owner.split(":")[0].split("-")[0]  # Get base model name
        if model_base == key_lower.split(":")[0]:
            return hf_repo
    
    return None


@ollama_app.command("info")
def ollama_info(
    model: str = typer.Argument(..., help="Ollama model name"),
):
    """Show detailed information about an Ollama model."""
    from moxing.ollama import OllamaClient, get_ollama_model
    from rich.panel import Panel
    
    client = OllamaClient()
    
    if not client.is_available():
        console.print("[red]Ollama is not running![/red]")
        raise typer.Exit(1)
    
    info = client.get_model_info(model)
    ollama_model = get_ollama_model(model)
    
    if not info and not ollama_model:
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)
    
    lines = []
    
    if ollama_model:
        lines.append(f"[cyan]Name:[/cyan] {ollama_model.full_name}")
        lines.append(f"[cyan]Size:[/cyan] {ollama_model.size_gb:.2f} GB")
        lines.append(f"[cyan]ID:[/cyan] {ollama_model.id}")
    
    if info:
        details = info.get("details", {})
        if details:
            lines.append(f"[cyan]Family:[/cyan] {details.get('family', 'unknown')}")
            lines.append(f"[cyan]Format:[/cyan] {details.get('format', 'unknown')}")
            lines.append(f"[cyan]Parameters:[/cyan] {details.get('parameter_size', 'unknown')}")
            lines.append(f"[cyan]Quantization:[/cyan] {details.get('quantization_level', 'unknown')}")
        
        modelfile = info.get("modelfile", "")
        if modelfile:
            lines.append(f"\n[dim]Modelfile:[/dim]")
            for line in modelfile.split("\n")[:10]:
                if line.strip():
                    lines.append(f"  [dim]{line}[/dim]")
    
    console.print(Panel("\n".join(lines), title=f"Ollama Model: {model}"))


@ollama_app.command("run")
def ollama_run(
    model: str = typer.Argument(..., help="Ollama model name"),
    prompt: str = typer.Option("Hello!", "-p", "--prompt", help="Prompt to send"),
):
    """Quick run an Ollama model."""
    from moxing.ollama import OllamaClient
    
    client = OllamaClient()
    
    if not client.is_available():
        console.print("[red]Ollama is not running![/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Running {model}...[/blue]\n")
    
    import subprocess
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=False
    )
    
    if result.returncode != 0:
        console.print(f"[red]Failed to run model[/red]")
        raise typer.Exit(1)


compress_app = typer.Typer(name="compress", help="GGUF compression commands")
app.add_typer(compress_app, name="compress")


@compress_app.command("pack")
def compress_pack(
    input_path: Path = typer.Argument(..., help="Path to GGUF file"),
    output_path: Optional[Path] = typer.Option(None, "-o", "--output", help="Output path"),
    algorithm: str = typer.Option("zstd", "-a", "--algorithm", help="Compression algorithm: zstd, lz4, xz, gzip"),
    level: int = typer.Option(0, "-l", "--level", help="Compression level (0=auto)"),
    keep_original: bool = typer.Option(True, "-k", "--keep", help="Keep original file"),
):
    """Compress a GGUF file to save disk space.
    
    Algorithms:
    - zstd: Best balance (default)
    - lz4: Fastest
    - xz: Best compression (slowest)
    - gzip: Universal
    
    Example:
        moxing compress pack model.gguf
        moxing compress pack model.gguf -a lz4
        moxing compress pack model.gguf -a xz -l 9
    """
    from moxing.gguf_compress import MultiCompressor, is_gguf_compressed
    
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    if is_gguf_compressed(input_path):
        console.print(f"[yellow]File is already compressed[/yellow]")
        raise typer.Exit(0)
    
    try:
        lvl = level if level > 0 else None
        compressor = MultiCompressor(algorithm=algorithm, level=lvl)
        info = compressor.compress(input_path, output_path, keep_original)
        
        console.print(f"\n[green]Compression complete![/green]")
        console.print(f"  Algorithm: {info.algorithm}")
        console.print(f"  Original: {info.original_size / (1024**3):.2f} GB")
        console.print(f"  Compressed: {info.compressed_size / (1024**3):.2f} GB")
        console.print(f"  Ratio: {info.compression_ratio:.1%}")
        console.print(f"  Saved: {info.savings_percent:.1f}% ({info.savings_mb:.0f} MB)")
        console.print(f"  Time: {info.compression_time:.1f}s")
        
    except Exception as e:
        console.print(f"[red]Compression failed: {e}[/red]")
        raise typer.Exit(1)


@compress_app.command("unpack")
def compress_unpack(
    input_path: Path = typer.Argument(..., help="Path to compressed file"),
    output_path: Optional[Path] = typer.Option(None, "-o", "--output", help="Output path"),
    keep_compressed: bool = typer.Option(True, "-k", "--keep", help="Keep compressed file"),
):
    """Decompress a compressed GGUF file.
    
    Example:
        moxing compress unpack model.gguf.zst
    """
    from moxing.gguf_compress import MultiCompressor, is_gguf_compressed, detect_compression_type
    
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    if not is_gguf_compressed(input_path):
        console.print(f"[yellow]File is not compressed[/yellow]")
        raise typer.Exit(0)
    
    try:
        alg = detect_compression_type(input_path) or "zstd"
        compressor = MultiCompressor(algorithm=alg)
        result, decomp_time = compressor.decompress(input_path, output_path, keep_compressed)
        
        console.print(f"[green]Decompressed to: {result}[/green]")
        console.print(f"[dim]Time: {decomp_time:.1f}s[/dim]")
        
    except Exception as e:
        console.print(f"[red]Decompression failed: {e}[/red]")
        raise typer.Exit(1)


@compress_app.command("cache")
def compress_cache(
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear decompression cache"),
    size: bool = typer.Option(False, "--size", "-s", help="Show cache size"),
    days: int = typer.Option(0, "--older-than", help="Clear files older than N days"),
):
    """Manage decompression cache.
    
    Examples:
        moxing compress cache --size
        moxing compress cache --clear
        moxing compress cache --clear --older-than 7
    """
    from moxing.gguf_compress import TransparentDecompressor
    
    decompressor = TransparentDecompressor()
    
    if size:
        cache_size = decompressor.get_cache_size()
        console.print(f"Cache size: {cache_size / (1024**3):.2f} GB ({cache_size / (1024**2):.1f} MB)")
        console.print(f"Cache location: {decompressor.cache_dir}")
    
    if clear:
        decompressor.clear_cache(older_than_days=days)
        console.print("[green]Cache cleared[/green]")


@compress_app.command("split")
def compress_split(
    input_path: Path = typer.Argument(..., help="Path to GGUF file"),
    chunk_size: int = typer.Option(1024, "-s", "--size", help="Chunk size in MB"),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
):
    """Split a GGUF file into chunks.
    
    Useful for storage on filesystems with size limits.
    
    Example:
        moxing compress split model.gguf --size 512
    """
    from moxing.gguf_compress import GGUFSplitter
    
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    try:
        splitter = GGUFSplitter(chunk_size_mb=chunk_size)
        chunks = splitter.split(input_path, output_dir)
        
        console.print(f"[green]Split into {len(chunks)} chunks[/green]")
        for chunk in chunks[:5]:
            console.print(f"  {chunk.name}")
        if len(chunks) > 5:
            console.print(f"  ... and {len(chunks) - 5} more")
        
    except Exception as e:
        console.print(f"[red]Split failed: {e}[/red]")
        raise typer.Exit(1)


@compress_app.command("merge")
def compress_merge(
    input_pattern: Path = typer.Argument(..., help="First chunk file or pattern"),
    output_path: Path = typer.Argument(..., help="Output GGUF file"),
):
    """Merge split GGUF chunks back into a single file.
    
    Example:
        moxing compress merge model.gguf-part-aa merged.gguf
    """
    from moxing.gguf_compress import GGUFSplitter, find_split_files
    
    try:
        if input_pattern.is_file():
            chunks = find_split_files(input_pattern)
        else:
            chunks = sorted(input_pattern.parent.glob(f"{input_pattern.name}*"))
        
        if not chunks:
            console.print("[red]No chunk files found[/red]")
            raise typer.Exit(1)
        
        splitter = GGUFSplitter()
        splitter.merge(chunks, output_path)
        
        console.print(f"[green]Merged to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Merge failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()