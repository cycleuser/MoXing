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


class UnsupportedArchitectureError(Exception):
    """Raised when a model architecture is not supported by llama.cpp."""
    pass


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
    port: int = typer.Option(8080, "-p", "--port", help="Server port (0 for auto)"),
    ctx_size: int = typer.Option(0, "-c", "--ctx-size", help="Context size (0=auto-detect based on VRAM)"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source (huggingface/modelscope/auto)"),
    backend: str = typer.Option("auto", "-b", "--backend", help="Backend: auto, vulkan, cuda, rocm, metal, cpu"),
    device: str = typer.Option("auto", "-d", "--device", help="Device: auto, gpu0, gpu1, cpu (use 'moxing devices' to list)"),
    auto: bool = typer.Option(True, "--auto/--no-auto", help="Auto-detect best device"),
    auto_port: bool = typer.Option(False, "-a", "--auto-port", help="Auto-find available port"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed monitoring in terminal"),
    web_monitor: bool = typer.Option(False, "-w", "--web", help="Enable web monitoring page"),
    force: bool = typer.Option(False, "-f", "--force", help="Force use specified backend without compatibility check"),
    kv_cache: str = typer.Option("auto", "--kv-cache", help="KV cache quantization: auto, f16, q8_0, q5_0, q4_0, tq4, tq3.5, tq3, tq2.5, tq2"),
    cpu_offload: int = typer.Option(0, "--cpu-offload", help="Number of layers to offload to CPU (0=auto)"),
    analyze_cache: bool = typer.Option(False, "--analyze-cache", help="Show KV cache analysis and exit"),
):
    """Start the LLM server with automatic configuration.
    
    KV Cache Quantization:
    - auto: Automatically choose based on available memory
    - f16: Full precision (16-bit)
    - q8_0: 8-bit quantization (high quality)
    - q5_0: 5-bit quantization (good quality)
    - q4_0: 4-bit quantization (balanced, recommended)
    
    TurboQuant (Google arXiv:2504.19874):
    - tq4: 4-bit TurboQuant (high quality)
    - tq3.5: 3.5-bit mixed precision (quality neutral) ⭐
    - tq3: 3-bit TurboQuant (good quality)
    - tq2.5: 2.5-bit mixed precision (slight loss) ⭐
    - tq2: 2-bit TurboQuant (maximum compression)
    
    Memory Optimization:
    - --cpu-offload N: Offload N layers to CPU RAM
    - Use with large models when GPU memory is limited
    
    Examples:
    - Auto KV cache: moxing serve model.gguf
    - 4-bit cache: moxing serve model.gguf --kv-cache q4_0
    - TurboQuant 3.5: moxing serve model.gguf --kv-cache tq3.5
    - CPU offload: moxing serve model.gguf --cpu-offload 10
    - Analyze cache: moxing serve model.gguf --analyze-cache
    """
    from moxing.runner import AutoRunner
    from moxing.mlx_server import MLXServer
    from moxing.gguf_check import diagnose_gguf, print_diagnosis, GGUFParser
    from moxing.ollama import OllamaClient, get_ollama_model
    from moxing.gguf_compress import is_gguf_compressed, resolve_model_path
    from moxing.server import find_available_port, is_port_in_use
    
    if port == 0 or auto_port or is_port_in_use(port, host):
        original_port = port if port > 0 else 8080
        port = find_available_port(original_port)
        if port != original_port:
            console.print(f"[yellow]Port {original_port} in use, using port {port}[/yellow]")
    
    if model.startswith("ollama:"):
        ollama_model = model[7:]
        ollama_serve_impl(ollama_model, port, host, ctx_size, device, backend, auto_port, verbose, web_monitor)
        return
    
    if backend == "ollama":
        ollama_serve_impl(model, port, host, ctx_size, device, backend, auto_port, verbose, web_monitor)
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
    
    if analyze_cache and model_path:
        from moxing.kv_cache import print_cache_analysis, estimate_model_size_gb
        
        model_size = model_path.stat().st_size / (1024 ** 3)
        print_cache_analysis(model_size, ctx_size if ctx_size > 0 else 4096)
        raise typer.Exit()
    
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
            
            model_short = Path(model).name[:20] if Path(model).exists() else model[:20]
            server_title = f"{model_short} | Apple Silicon | MLX"
            
            console.print(Panel(
                f"[green]Server:[/green] http://{host}:{port}\n"
                f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
                f"[magenta]Backend:[/magenta] MLX (Apple Silicon)\n"
                f"[cyan]Device:[/cyan] Apple GPU\n"
                f"[yellow]Press Ctrl+C to stop[/yellow]",
                title=server_title
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
        from moxing.device import DeviceDetector, BackendType
        from moxing.server import LlamaServer
        
        detector = DeviceDetector()
        detector.detect()
        
        model_size_gb = 0
        model_path_obj = Path(model)
        if model_path_obj.exists():
            model_size_gb = model_path_obj.stat().st_size / (1024 ** 3)
        
        if device != "auto":
            device_config = detector.get_device_config_by_name(device, backend, model_size_gb)
        elif backend != "auto":
            try:
                backend_type = BackendType(backend.lower())
                device_config = detector.get_best_device(model_size_gb)
                device_config.backend = backend_type
            except ValueError:
                console.print(f"[red]Unknown backend: {backend}[/red]")
                raise typer.Exit(1)
        else:
            device_config = detector.get_best_device(model_size_gb)
        
        if ctx_size == 0:
            ctx_size = device_config.recommended_ctx
        
        device_str = "auto"
        if device_config.device.backend != BackendType.CPU:
            has_amd_perm, _ = detector.check_amd_permission()
            
            if device_config.backend == BackendType.ROCM and not has_amd_perm:
                console.print("[yellow]Note: Using auto device selection (ROCm permission denied)[/yellow]")
                device_str = "auto"
            elif device_config.backend == BackendType.VULKAN:
                device_str = "auto"
                console.print("[dim]Using auto Vulkan device selection[/dim]")
            elif device_config.backend == BackendType.METAL:
                device_str = f"MTL{device_config.device.index}"
            elif device_config.backend == BackendType.ROCM:
                device_str = f"HIP{device_config.device.index}"
            elif device_config.backend == BackendType.CUDA:
                device_str = f"CUDA{device_config.device.index}"
            else:
                device_str = "auto"
        
        console.print(Panel(
            f"[green]Model:[/green] {model}\n"
            f"[blue]Backend:[/blue] {device_config.backend.value}\n"
            f"[yellow]Device:[/yellow] {device_config.device.name}\n"
            f"[magenta]GPU Layers:[/magenta] {device_config.n_gpu_layers if device_config.n_gpu_layers >= 0 else 'all'}\n"
            f"[cyan]Context:[/cyan] {ctx_size}\n"
            f"[dim]KV Cache: {kv_cache}[/dim]",
            title="Configuration"
        ))
        
        device_display = device_config.device.name[:30]
        backend_display = device_config.backend.value.upper()
        model_short = Path(model).name[:20] if Path(model).exists() else model[:20]
        server_title = f"{model_short} | {device_display} | {backend_display}"
        
        if kv_cache != "f16":
            server_title += f" | {kv_cache}"
        
        try:
            server = LlamaServer(
                model=model,
                host=host,
                port=port,
                ctx_size=ctx_size,
                n_gpu_layers=device_config.n_gpu_layers,
                device=device_str,
                gpu_backend=device_config.backend.value,
                kv_cache_quant=kv_cache,
                cpu_offload=cpu_offload > 0,
                cpu_offload_layers=cpu_offload,
            )
            
            cache_info = f"\n[dim]KV Cache: {kv_cache}" if kv_cache != "auto" else ""
            if cpu_offload > 0:
                cache_info += f"\n[dim]CPU Offload: {cpu_offload} layers"
            
            console.print(Panel(
                f"[green]Server:[/green] http://{host}:{port}\n"
                f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
                f"[magenta]Backend:[/magenta] {device_config.backend.value}\n"
                f"[cyan]Device:[/cyan] {device_config.device.name}"
                f"{cache_info}\n"
                f"[yellow]Press Ctrl+C to stop[/yellow]",
                title=server_title
            ))
            
            server.start(wait=False)
            
            import time
            time.sleep(3)
            
            if server._process and server._process.poll() is not None:
                stdout, stderr = server._process.communicate()
                console.print(f"\n[red bold]Server failed to start![/red bold]")
                console.print(f"[dim]Exit code: {server._process.returncode}[/dim]")
                if stdout:
                    console.print(f"[dim]stdout:[/dim]")
                    for line in stdout.strip().split("\n")[-20:]:
                        if line.strip():
                            console.print(f"[dim]  {line}[/dim]")
                if stderr:
                    console.print(f"[red]stderr:[/red]")
                    for line in stderr.strip().split("\n")[-20:]:
                        if line.strip():
                            console.print(f"[red]  {line}[/red]")
                console.print(f"\n[yellow]Troubleshooting tips:[/yellow]")
                console.print(f"  • Check if the model file is valid")
                console.print(f"  • Try with smaller context: -c 4096")
                console.print(f"  • Try different backend: -b vulkan")
                raise typer.Exit(1)
        except RuntimeError as e:
            console.print(f"\n[red bold]Failed to start server![/red bold]")
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        
        try:
            serve_with_verbose_monitor(server, verbose=verbose, web_monitor=web_monitor)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            server.stop()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(1)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or path"),
    prompt: str = typer.Option(None, "-p", "--prompt", help="Single prompt (leave empty for interactive chat)"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization"),
    tokens: int = typer.Option(256, "-n", "--tokens", help="Max tokens to generate"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed monitoring and statistics"),
    backend: str = typer.Option("auto", "-b", "--backend", help="Backend: auto, vulkan, cuda, metal, cpu"),
    kv_cache: str = typer.Option("auto", "--kv-cache", help="KV cache quantization"),
):
    """Run inference with a model (auto-downloads if needed).
    
    Examples:
        moxing run model.gguf                    # Interactive chat
        moxing run model.gguf -p "Hello"         # Single prompt
        moxing run model.gguf -v                 # Verbose monitoring
        moxing run model.gguf -p "Hello" -v      # Single prompt with stats
        moxing run model.gguf --kv-cache tq3.5   # With TurboQuant
    """
    from moxing.runner import AutoRunner
    from moxing.server import find_available_port
    
    runner = AutoRunner()
    
    try:
        port = find_available_port(8080)
        
        server = runner.server(
            model=model,
            quant=quant,
            source=source,
            ctx_size=ctx_size,
            backend=backend,
            kv_cache_quant=kv_cache,
            port=port
        )
        
        model_name = Path(model).name if Path(model).exists() else model
        
        console.print(Panel(
            f"[cyan]Model:[/cyan] {model_name}\n"
            f"[cyan]Context:[/cyan] {ctx_size}\n"
            f"[cyan]Backend:[/cyan] {backend}\n"
            f"[cyan]KV Cache:[/cyan] {kv_cache}",
            title="Starting Server"
        ))
        
        server.start(wait=True)
        
        console.print("[green]Server ready![/green]")
        
        run_with_verbose_monitor(
            server=server,
            model_name=model_name,
            prompt=prompt,
            max_tokens=tokens,
            verbose=verbose
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if server:
            server.stop()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)
    finally:
        if server:
            server.stop()


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


@app.command("cache")
def cache_analysis(
    model_size: float = typer.Argument(..., help="Model size in GB"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size to analyze"),
    vram: float = typer.Option(8.0, "--vram", help="Available VRAM in GB"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Run TurboQuant benchmark"),
):
    """Analyze KV cache memory usage and compression options.
    
    Shows memory requirements for different KV cache quantization methods including
    Google's TurboQuant for 3-bit compression.
    
    Examples:
        moxing cache 7.0
        moxing cache 7.0 -c 8192 --vram 16.0
        moxing cache 7.0 --benchmark
    """
    from moxing.kv_cache import print_cache_analysis
    
    if benchmark:
        from moxing.turboquant import benchmark_turboquant
        benchmark_turboquant()
        return
    
    print_cache_analysis(model_size, ctx_size, vram)


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
    backend: str = typer.Option("auto", "-b", "--backend", help="GPU backend (auto, vulkan, cuda, metal, cpu, all)"),
    force: bool = typer.Option(False, "-f", "--force", help="Force re-download"),
    list_available: bool = typer.Option(False, "-l", "--list", help="List available backends"),
):
    """Download pre-built llama.cpp binaries.
    
    Use --backend all to download binaries for all supported backends.
    """
    from moxing.binaries import get_binary_manager, list_available_backends
    
    if list_available:
        console.print("\n[bold]Available backends for this platform:[/bold]")
        available = list_available_backends()
        for b, has_bin in available.items():
            status = "[green]✓ installed[/green]" if has_bin else "[dim]not installed[/dim]"
            console.print(f"  {b}: {status}")
        console.print("\n[dim]Use --backend <name> to download specific backend[/dim]")
        console.print("[dim]Use --backend all to download all backends[/dim]")
        return
    
    if backend == "all":
        console.print("[blue]Downloading binaries for all supported backends...[/blue]")
        manager = get_binary_manager()
        results = manager.download_all_binaries(force=force)
        
        console.print("\n[bold]Download Results:[/bold]")
        for b, success in results.items():
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            console.print(f"  {status} {b}")
        
        console.print(f"\n[green]Binaries installed to: {manager.cache_dir}[/green]")
        return
    
    manager = get_binary_manager(backend)
    
    console.print(f"[blue]Downloading binaries for {manager.platform_name} ({manager.backend})...[/blue]")
    
    try:
        manager.download_binaries(force=force)
        
        binaries = manager.list_cached_binaries()
        console.print(f"\n[green]Installed binaries:[/green]")
        for b in binaries[:10]:
            console.print(f"  - {b}")
        if len(binaries) > 10:
            console.print(f"  ... and {len(binaries) - 10} more")
        
        console.print(f"\n[green]Binaries installed to: {manager.get_cache_dir()}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("update-binaries")
def update_binaries_cmd(
    force: bool = typer.Option(False, "-f", "--force", help="Force update even if up to date"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Auto-confirm update"),
):
    """Update llama.cpp binaries to the latest version.
    
    Checks for newer binary versions and downloads them if available.
    Supports automatic update detection and one-click updates.
    
    Examples:
        moxing update-binaries           # Check and update if needed
        moxing update-binaries -f        # Force re-download
        moxing update-binaries -f -y     # Force update without confirmation
    
    Note: Uses bundled binaries as fallback if update fails.
    """
    from moxing.binaries import get_binary_manager, clear_skip_update
    
    console.print("[blue]Checking for binary updates...[/blue]\n")
    
    manager = get_binary_manager()
    clear_skip_update()
    
    has_update, current, latest = manager.check_for_updates()
    
    if not current:
        console.print("[yellow]Could not determine current version[/yellow]")
        console.print("[blue]Proceeding with download...[/blue]\n")
        has_update = True
    
    if has_update or force:
        if latest:
            if current:
                console.print(f"[green bold]New version available![/green bold]")
                console.print(f"  Current:  {current}")
                console.print(f"  Latest:   {latest}")
            else:
                console.print(f"[green]Latest version: {latest}[/green]")
        else:
            console.print("[green]Update available[/green]")
        
        if not yes:
            from rich.prompt import Confirm
            console.print()
            if not Confirm.ask("Download update?", default=True):
                console.print("[yellow]Update cancelled[/yellow]")
                from moxing.binaries import skip_update_forever
                skip_update_forever()
                return
        
        console.print(f"\n[blue]Downloading update...[/blue]\n")
        
        try:
            manager.download_binaries(force=True, quiet=False, check_updates=False)
            console.print(f"\n[green bold]✓ Update complete![/green bold]")
            console.print(f"[dim]Installed to: {manager.cache_dir}[/dim]")
            console.print("\n[dim]Tip: Restart any running servers to use new binaries[/dim]")
        except Exception as e:
            console.print(f"[red]Update failed: {e}[/red]")
            console.print("[yellow]Falling back to bundled binaries[/yellow]")
    else:
        console.print("[green]✓ Already up to date[/green]")
        console.print(f"[dim]Current version: {current}[/dim]")
        console.print("\n[dim]Tip: Use --force to re-download binaries[/dim]")


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


def run_with_verbose_monitor(server, model_name: str, prompt: str = None, max_tokens: int = 256, verbose: bool = False):
    """Run inference with verbose monitoring display."""
    from moxing.client import Client
    import time
    import psutil
    
    client = Client(server.base_url)
    
    def get_stats():
        try:
            return {
                'cpu': psutil.cpu_percent(interval=0.1),
                'ram_gb': psutil.virtual_memory().used / (1024**3),
            }
        except:
            return {'cpu': 0, 'ram_gb': 0}
    
    if prompt:
        messages = [{"role": "user", "content": prompt}]
        
        console.print(f"\n[bold blue]Prompt:[/bold blue] {prompt}\n")
        
        if verbose:
            stats = get_stats()
            console.print(f"[dim]📊 RAM: {stats['ram_gb']:.2f} GB | CPU: {stats['cpu']:.1f}%[/dim]\n")
        
        console.print("[bold green]Response:[/bold green]")
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        try:
            response = client.chat.completions.create(
                model="llama",
                messages=messages,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if isinstance(chunk, dict) and chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta", {})
                    # Get content from delta (may be in 'content' or 'reasoning_content')
                    content = delta.get("content") or delta.get("reasoning_content") or ""
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        full_response += content
                        token_count += 1
                        print(content, end="", flush=True)
        except Exception as e:
            console.print(f"\n[red]Error during generation: {e}[/red]")
            return ""
        
        total_time = time.time() - start_time
        print()
        
        if verbose:
            stats = get_stats()
            speed = token_count / total_time if total_time > 0 else 0
            ttft = first_token_time - start_time if first_token_time else 0
            
            console.print()
            console.print(Panel(
                f"[bold cyan]📊 Performance[/bold cyan]\n\n"
                f"[green]Tokens:[/green] {token_count} | "
                f"[green]Time:[/green] {total_time:.2f}s | "
                f"[green]Speed:[/green] {speed:.1f} tok/s | "
                f"[green]TTFT:[/green] {ttft:.2f}s\n\n"
                f"[yellow]Memory:[/yellow] RAM: {stats['ram_gb']:.2f} GB\n"
                f"[blue]CPU:[/blue] {stats['cpu']:.1f}%",
                title="Summary"
            ))
        
        return full_response
    else:
        messages = []
        
        while True:
            try:
                if verbose:
                    stats = get_stats()
                    console.print(f"[dim]📊 RAM: {stats['ram_gb']:.2f} GB | CPU: {stats['cpu']:.1f}%[/dim]")
                
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                
                messages.append({"role": "user", "content": user_input})
                
                start_time = time.time()
                first_token_time = None
                token_count = 0
                
                console.print("[bold green]Assistant[/bold green]: ", end="")
                
                try:
                    response = client.chat.completions.create(
                        model="llama",
                        messages=messages,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    assistant_msg = ""
                    for chunk in response:
                        if isinstance(chunk, dict) and chunk.get("choices"):
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content") or delta.get("reasoning_content") or ""
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                assistant_msg += content
                                token_count += 1
                                print(content, end="", flush=True)
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]")
                    continue
                
                print()
                
                total_time = time.time() - start_time
                
                if verbose:
                    speed = token_count / total_time if total_time > 0 else 0
                    ttft = first_token_time - start_time if first_token_time else 0
                    
                    console.print(f"[dim]  {token_count} tokens | {total_time:.2f}s | {speed:.1f} tok/s | TTFT: {ttft:.2f}s[/dim]")
                
                messages.append({"role": "assistant", "content": assistant_msg})
                
            except KeyboardInterrupt:
                break
        
        if verbose:
            total_prompts = sum(1 for m in messages if m.get("role") == "user")
            total_responses = sum(1 for m in messages if m.get("role") == "assistant")
            
            console.print()
            console.print(Panel(
                f"[bold cyan]📊 Session Summary[/bold cyan]\n\n"
                f"[green]Messages:[/green] {total_prompts} prompts, {total_responses} responses\n\n"
                f"[magenta]Server:[/magenta] http://{server.host}:{server.port}",
                title="Chat Complete"
            ))


def serve_with_verbose_monitor(server, verbose: bool = False, web_monitor: bool = False):
    """Run server with verbose monitoring.
    
    Args:
        server: LlamaServer instance  
        verbose: Enable detailed monitoring in terminal
        web_monitor: Enable web monitoring page
    """
    from moxing.enhanced_monitor import EnhancedMonitor
    import time
    
    if not verbose and not web_monitor:
        while server.is_running():
            time.sleep(1)
        return
    
    monitor = EnhancedMonitor(server.host, server.port)
    monitor.fetch_server_info()
    monitor.start_collection(interval=1.0)
    
    if server._process:
        monitor.set_process(server._process.pid)
    
    if web_monitor:
        console.print()
        console.print(Panel(
            f"[green]Web Monitor:[/green] http://{server.host}:{server.port}\n"
            f"[blue]OpenAI API:[/blue] http://{server.host}:{server.port}/v1\n"
            f"[dim]The web page shows live metrics and charts[/dim]",
            title="Web Monitoring Enabled"
        ))
    
    if verbose:
        console.print()
        console.print("[blue]Verbose monitoring enabled (refresh: 1s)[/blue]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    try:
        while server.is_running():
            snapshot = monitor._collect_snapshot()
            monitor.history.add(snapshot)
            stats = monitor.history.get_stats(60)
            
            if verbose:
                model_name = monitor.server_info.model_name[:30] if monitor.server_info.model_name else "Unknown"
                ctx_len = monitor.server_info.context_length
                
                console.print(Panel(
                    f"[cyan]Model:[/cyan] {model_name}\n"
                    f"[cyan]Context:[/cyan] {ctx_len:,}\n\n"
                    f"[green]Tokens:[/green]\n"
                    f"  Prompt: {snapshot.prompt_tokens:,}\n"
                    f"  Generated: {snapshot.generated_tokens:,}\n"
                    f"  Total: {snapshot.total_tokens:,}\n\n"
                    f"[yellow]Speed:[/yellow]\n"
                    f"  Prompt: {snapshot.prompt_speed:.1f} tok/s\n"
                    f"  Generate: {snapshot.generate_speed:.1f} tok/s\n"
                    f"  Avg (60s): {stats.get('avg_generate_speed', 0):.1f} tok/s\n\n"
                    f"[blue]Memory:[/blue]\n"
                    f"  GPU: {snapshot.gpu_memory_mb:.0f} MB (avg: {stats.get('avg_gpu_memory', 0):.0f})\n"
                    f"  RAM: {snapshot.ram_used_mb/1024:.2f} GB\n\n"
                    f"[magenta]CPU:[/magenta] {snapshot.cpu_percent:.1f}% (avg: {stats.get('avg_cpu', 0):.1f}%)\n\n"
                    f"[dim]Requests: {snapshot.requests_processing} processing, {snapshot.requests_deferred} deferred[/dim]",
                    title=f"🚀 MoXing Monitor - {model_name}"
                ))
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    
    monitor.stop_collection()
    
    if server._process:
        monitor.set_process(server._process.pid)
    
    console.print()
    console.print("[blue]Verbose monitoring enabled (refresh: 1s)[/blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    try:
        while server.is_running():
            snapshot = monitor._collect_snapshot()
            monitor.history.add(snapshot)
            stats = monitor.history.get_stats(60)
            
            model_name = monitor.server_info.model_name[:30] if monitor.server_info.model_name else "Unknown"
            ctx_len = monitor.server_info.context_length
            
            console.print(Panel(
                f"[cyan]Model:[/cyan] {model_name}\n"
                f"[cyan]Context:[/cyan] {ctx_len:,}\n\n"
                f"[green]Tokens:[/green]\n"
                f"  Prompt: {snapshot.prompt_tokens:,}\n"
                f"  Generated: {snapshot.generated_tokens:,}\n"
                f"  Total: {snapshot.total_tokens:,}\n\n"
                f"[yellow]Speed:[/yellow]\n"
                f"  Prompt: {snapshot.prompt_speed:.1f} tok/s\n"
                f"  Generate: {snapshot.generate_speed:.1f} tok/s\n"
                f"  Avg (60s): {stats.get('avg_generate_speed', 0):.1f} tok/s\n\n"
                f"[blue]Memory:[/blue]\n"
                f"  GPU: {snapshot.gpu_memory_mb:.0f} MB (avg: {stats.get('avg_gpu_memory', 0):.0f})\n"
                f"  RAM: {snapshot.ram_used_mb/1024:.2f} GB\n\n"
                f"[magenta]CPU:[/magenta] {snapshot.cpu_percent:.1f}% (avg: {stats.get('avg_cpu', 0):.1f}%)\n\n"
                f"[dim]Requests: {snapshot.requests_processing} processing, {snapshot.requests_deferred} deferred[/dim]",
                title=f"🚀 MoXing Monitor - {model_name}"
            ))
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    
    monitor.stop_collection()


ollama_app = typer.Typer(name="ollama", help="Manage Ollama models")
app.add_typer(ollama_app, name="ollama")


@ollama_app.command("list")
def ollama_list(
    show_embeddings: bool = typer.Option(True, "--embeddings/--no-embeddings", help="Show embedding models"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    select: bool = typer.Option(False, "--select", "-s", help="Interactive model selection"),
    show_context: bool = typer.Option(True, "--context/--no-context", "-C", help="Show context length"),
):
    """List all models from local Ollama installation.
    
    Shows model name, size, context length, and type.
    
    Examples:
        moxing ollama list
        moxing ollama list --no-embeddings
        moxing ollama list --no-context
    """
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
        print_ollama_models(models, show_embeddings, show_context)


@ollama_app.command("serve")
def ollama_serve(
    model: str = typer.Argument(..., help="Ollama model name (e.g., gemma3:4b)"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port (0 for auto)"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    ctx_size: int = typer.Option(32768, "-c", "--ctx-size", help="Context size (0=auto)"),
    device: str = typer.Option("auto", "-d", "--device", help="Device: auto, gpu0, gpu1, cpu"),
    backend: str = typer.Option("auto", "-b", "--backend", help="Backend: auto, cuda, rocm, vulkan, metal, mlx, mps, cpu"),
    auto_port: bool = typer.Option(False, "-a", "--auto-port", help="Auto-find available port if default is in use"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed monitoring in terminal"),
    web_monitor: bool = typer.Option(False, "-w", "--web", help="Enable web monitoring page"),
    skip_check: bool = typer.Option(False, "--skip-check", help="Skip compatibility check"),
    kv_cache: str = typer.Option("auto", "--kv-cache", help="KV cache quantization: auto, f16, q8_0, q4_0, tq4, tq3.5, tq3, tq2.5, tq2"),
    cpu_offload: int = typer.Option(0, "--cpu-offload", help="Number of layers to offload to CPU (0=auto)"),
    prompt_offload: bool = typer.Option(False, "--prompt-offload", help="Prompt for CPU offload if needed"),
    rope_scaling: str = typer.Option("none", "--rope-scaling", help="RoPE scaling: none, linear, yarn (for extending context)"),
    rope_scale: float = typer.Option(1.0, "--rope-scale", help="RoPE context scaling factor (e.g., 2.0 for 2x context)"),
    # Performance options
    threads: int = typer.Option(0, "-t", "--threads", help="Number of threads (-1=auto)"),
    batch_size: int = typer.Option(2048, "--batch-size", help="Batch size for prompt processing"),
    ubatch_size: int = typer.Option(512, "--ubatch-size", help="Physical batch size"),
    flash_attn: bool = typer.Option(True, "--flash-attn/--no-flash-attn", help="Enable flash attention"),
):
    """Serve an Ollama model with OpenAI-compatible API.
    
    Backends:
    - cuda: NVIDIA GPUs (fastest)
    - rocm: AMD GPUs with ROCm
    - vulkan: Cross-platform GPU (works with AMD/NVIDIA/Intel)
    - metal: Apple Silicon (macOS)
    - mlx: Apple MLX backend (macOS, optimized for Apple Silicon)
    - mps: Apple Metal Performance Shaders (macOS)
    - cpu: CPU only
    
    KV Cache Quantization:
    - auto: Automatically choose based on available memory
    - q8_0: 8-bit (high quality)
    - q4_0: 4-bit (balanced)
    - tq3: TurboQuant 3-bit (recommended, 5.3x compression)
    - tq2: TurboQuant 2-bit (extreme, 8x compression)
    
    CPU Offload:
    - 0: Auto-detect if needed (default, uses full GPU if possible)
    - N: Offload N layers to CPU, rest to GPU
    - Use --prompt-offload to be asked before offloading
    
    Performance Tips:
    - Default context is 32K (optimized for speed)
    - Use --flash-attn for faster attention (enabled by default)
    - Increase --batch-size for better throughput
    - Use -c 65536 or higher for long documents
    
    Examples:
        moxing ollama serve gemma3:4b
        moxing ollama serve gemma3:4b -b cuda
        moxing ollama serve omnicoder-9b --kv-cache q4_0
        moxing ollama serve omnicoder-9b --cpu-offload 10
        moxing ollama serve model --prompt-offload
        moxing ollama serve gemma3:1b -c 65536 --kv-cache q4_0
        moxing ollama serve omnicoder-9b -v    # Verbose monitoring
        moxing ollama serve omnicoder-9b -w    # Web monitoring
    """
    ollama_serve_impl(model, port, host, ctx_size, device, backend, auto_port, verbose, web_monitor, skip_check, kv_cache, cpu_offload, prompt_offload, rope_scaling, rope_scale, threads, batch_size, ubatch_size, flash_attn)


def serve_with_ollama_backend(
    model: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    ctx_size: int = 4096,
    device: str = "auto",
    backend: str = "auto",
):
    """Serve a model using Ollama's backend (for models requiring Ollama patches)."""
    import subprocess
    import time
    import signal
    from moxing.server import find_available_port, is_port_in_use
    
    ollama_port = 11434
    
    console.print(Panel(
        f"[green]Model:[/green] {model}\n"
        f"[blue]Backend:[/blue] Ollama (patched llama.cpp)\n"
        f"[yellow]Port:[/yellow] {port}\n"
        f"[cyan]API:[/cyan] OpenAI compatible at http://{host}:{port}/v1",
        title="Ollama Backend"
    ))
    
    console.print("[blue]Starting Ollama service...[/blue]")
    
    result = subprocess.run(
        ["ollama", "serve"],
        capture_output=True,
        text=True,
        start_new_session=True
    )
    
    time.sleep(2)
    
    console.print("[blue]Loading model in Ollama...[/blue]")
    
    load_result = subprocess.run(
        ["ollama", "run", model, "--keepalive", "24h"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    console.print(Panel(
        f"[green]Ollama API:[/green] http://127.0.0.1:{ollama_port}/api\n"
        f"[green]OpenAI API:[/green] http://127.0.0.1:{ollama_port}/v1\n"
        f"[yellow]Press Ctrl+C to stop[/yellow]",
        title=f"Ollama: {model}"
    ))
    
    console.print(f"\n[dim]This model uses Ollama's patched backend with architecture support.[/dim]")
    console.print(f"[dim]Ollama is running on port {ollama_port}[/dim]")
    
    try:
        import httpx
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping Ollama service...[/yellow]")
    
    raise typer.Exit(0)


def ollama_serve_impl(
    model: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    ctx_size: int = 4096,
    device: str = "auto",
    backend: str = "auto",
    auto_port: bool = False,
    verbose: bool = False,
    web_monitor: bool = False,
    skip_check: bool = False,
    kv_cache: str = "auto",
    cpu_offload: int = 0,
    prompt_offload: bool = False,
    rope_scaling: str = "none",
    rope_scale: float = 1.0,
    threads: int = 0,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    flash_attn: bool = True,
):
    """Implementation of ollama serve with device/backend selection."""
    from moxing.ollama import OllamaClient, get_ollama_model
    from moxing.server import LlamaServer, find_available_port, is_port_in_use
    from moxing.device import DeviceDetector, BackendType
    
    if port == 0 or auto_port or is_port_in_use(port, host):
        original_port = port if port > 0 else 8080
        port = find_available_port(original_port)
        if port != original_port:
            console.print(f"[yellow]Port {original_port} in use, using port {port}[/yellow]")
    
    if verbose:
        console.print(f"[dim]Looking up model: {model}[/dim]")
        console.print(f"[dim]Using port: {port}[/dim]")
    
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
    
    if verbose:
        console.print(f"[dim]Found model: {ollama_model.full_name} ({ollama_model.size_gb:.1f} GB)[/dim]")
    
    is_accessible, gguf_path, message = client.check_model_access(model)
    
    if verbose:
        console.print(f"[dim]Model accessible: {is_accessible}[/dim]")
        if gguf_path:
            console.print(f"[dim]GGUF path: {gguf_path}[/dim]")
    
    if not is_accessible:
        console.print(f"[red]Cannot access model '{model}':[/red]")
        console.print(message)
        raise typer.Exit(1)
    
    if skip_check:
        console.print("[yellow]Skipping compatibility check (--skip-check)[/yellow]")
    else:
        if verbose:
            console.print(f"[dim]Calling _get_compatible_gguf...[/dim]")
        
        try:
            gguf_path = _get_compatible_gguf(model, gguf_path, ollama_model.size_gb, verbose)
        except UnsupportedArchitectureError as e:
            console.print(f"\n[yellow]Model architecture requires Ollama backend[/yellow]")
            console.print(f"[dim]{e}[/dim]")
            console.print(f"[blue]Switching to Ollama service...[/blue]")
            return serve_with_ollama_backend(model, port, host, ctx_size, device, backend)
        except Exception as e:
            console.print(f"[red]Error in compatibility check: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(1)
        
        if verbose:
            console.print(f"[dim]_get_compatible_gguf returned: {gguf_path}[/dim]")
        
        if not gguf_path:
            console.print(f"[red]Failed to get compatible GGUF[/red]")
            raise typer.Exit(1)
    
    detector = DeviceDetector()
    detector.detect()
    
    if device != "auto":
        device_config = detector.get_device_config_by_name(device, backend, ollama_model.size_gb)
    elif backend != "auto":
        try:
            backend_type = BackendType(backend.lower())
            device_config = detector.get_best_device(ollama_model.size_gb)
            device_config.backend = backend_type
        except ValueError:
            console.print(f"[red]Unknown backend: {backend}[/red]")
            raise typer.Exit(1)
    else:
        device_config = detector.get_best_device(ollama_model.size_gb)
    
    has_amd_perm, amd_perm_msg = detector.check_amd_permission()
    
    if device_config.backend == BackendType.ROCM and not has_amd_perm:
        console.print(f"\n[yellow bold]AMD ROCm Permission Issue[/yellow bold]")
        console.print(f"[yellow]{amd_perm_msg}[/yellow]")
        console.print(f"\n[blue]Switching to Vulkan backend for AMD GPU...[/blue]")
        console.print(f"[dim]ROCm requires: sudo usermod -aG render \"$USER\"[/dim]")
        console.print(f"[dim]After adding group, log out and log back in.[/dim]\n")
        
        device_config.backend = BackendType.VULKAN
    
    is_cpu = device_config.backend == BackendType.CPU
    
    available_vram_gb = 0
    if not is_cpu:
        if device_config.device.memory_mb > 0:
            available_vram_gb = device_config.device.free_memory_gb * 0.85
        else:
            if device_config.backend == BackendType.ROCM:
                if "7900" in device_config.device.name:
                    available_vram_gb = 24 * 0.85
                elif "610M" in device_config.device.name:
                    available_vram_gb = 0.5
                else:
                    available_vram_gb = 8.0
            elif device_config.backend == BackendType.CUDA:
                available_vram_gb = 8.0
            elif device_config.backend in [BackendType.METAL, BackendType.MLX, BackendType.MPS]:
                available_vram_gb = 12.0
            else:
                available_vram_gb = 4.0
    
    model_size_gb = ollama_model.size_gb
    
    total_layers = 40
    if model_size_gb > 30:
        total_layers = 80
    elif model_size_gb > 15:
        total_layers = 60
    elif model_size_gb > 8:
        total_layers = 40
    else:
        total_layers = 32
    
    n_gpu_layers = -1 if not is_cpu else 0
    actual_cpu_offload = cpu_offload
    
    if not is_cpu and available_vram_gb > 0:
        offload_plan = detector.calculate_offload_plan(
            device_config.device,
            model_size_gb,
            ctx_size,
            total_layers
        )
        
        if offload_plan.needs_offload:
            console.print(f"\n[yellow]Warning: Model ({model_size_gb:.1f}GB) may not fit entirely in VRAM ({available_vram_gb:.1f}GB)[/yellow]")
            
            if cpu_offload == 0:
                if prompt_offload:
                    from rich.prompt import Confirm
                    console.print(f"[yellow]Suggested: offload {offload_plan.suggested_cpu_layers} layers to CPU[/yellow]")
                    try:
                        if Confirm.ask("Enable CPU offload?", default=True):
                            actual_cpu_offload = offload_plan.suggested_cpu_layers
                            n_gpu_layers = offload_plan.gpu_layers
                            console.print(f"[blue]Will offload {actual_cpu_offload} layers to CPU[/blue]\n")
                        else:
                            console.print("[yellow]Proceeding without CPU offload (may fail if VRAM insufficient)[/yellow]\n")
                    except:
                        actual_cpu_offload = offload_plan.suggested_cpu_layers
                        n_gpu_layers = offload_plan.gpu_layers
                else:
                    actual_cpu_offload = offload_plan.suggested_cpu_layers
                    n_gpu_layers = offload_plan.gpu_layers
                    console.print(f"[blue]Auto offloading {actual_cpu_offload}/{total_layers} layers to CPU[/blue]")
                    console.print(f"[dim]Use --cpu-offload N to customize, or --prompt-offload to be asked[/dim]\n")
            else:
                actual_cpu_offload = cpu_offload
                n_gpu_layers = total_layers - cpu_offload
                if n_gpu_layers < 1:
                    n_gpu_layers = 1
                    actual_cpu_offload = total_layers - 1
        else:
            console.print(f"[green]Model fits in VRAM: {model_size_gb:.1f}GB < {available_vram_gb:.1f}GB[/green]")
    elif cpu_offload > 0 and not is_cpu:
        n_gpu_layers = total_layers - cpu_offload
        if n_gpu_layers < 0:
            n_gpu_layers = 0
        actual_cpu_offload = cpu_offload
    
    device_str = "auto"
    if not is_cpu:
        if device_config.backend == BackendType.ROCM:
            device_str = "auto"
            console.print("[dim]Using auto ROCm device selection[/dim]")
        elif device_config.backend == BackendType.VULKAN:
            device_str = "auto"
            console.print("[dim]Using auto Vulkan device selection[/dim]")
        elif device_config.backend in [BackendType.METAL, BackendType.MLX, BackendType.MPS]:
            device_str = f"MTL{device_config.device.index}"
        elif device_config.backend == BackendType.CUDA:
            device_str = f"CUDA{device_config.device.index}"
        else:
            device_str = "auto"
    
    gpu_layers_display = "0 (CPU)" if is_cpu else (f"{n_gpu_layers}" if n_gpu_layers > 0 else "all")
    if actual_cpu_offload > 0 and not is_cpu:
        gpu_layers_display = f"{n_gpu_layers}/{total_layers} (CPU: {actual_cpu_offload})"
    
    device_display = device_config.device.name
    backend_display = device_config.backend.value.upper()
    model_short = ollama_model.full_name[:20]
    server_title = f"{model_short} | {device_display[:30]} | {backend_display}"
    
    if kv_cache != "auto" and kv_cache != "f16":
        server_title += f" | {kv_cache}"
    
    console.print(Panel(
        f"[green]Model:[/green] {ollama_model.full_name}\n"
        f"[blue]Size:[/blue] {ollama_model.size_gb:.1f} GB\n"
        f"[blue]GGUF:[/blue] {gguf_path.name[:50]}...\n"
        f"[yellow]Port:[/yellow] {port}\n"
        f"[magenta]Backend:[/magenta] {device_config.backend.value}\n"
        f"[cyan]Device:[/cyan] {device_config.device.name}",
        title=f"Ollama: {model_short}"
    ))
    
    cache_info = f"\n[dim]KV Cache: {kv_cache}" if kv_cache != "auto" else ""
    if actual_cpu_offload > 0:
        cache_info += f"\n[dim]CPU Offload: {actual_cpu_offload} layers"
    
    console.print(Panel(
        f"[green]Model:[/green] {ollama_model.full_name}\n"
        f"[blue]Backend:[/blue] {device_config.backend.value}\n"
        f"[yellow]Device:[/yellow] {device_config.device.name}\n"
        f"[magenta]GPU Layers:[/magenta] {gpu_layers_display}\n"
        f"[cyan]Context:[/cyan] {ctx_size}"
        f"{cache_info}",
        title="Configuration"
    ))
    
    server = None
    try:
        extra_kwargs = {}
        if rope_scaling != "none":
            extra_kwargs['rope_scaling'] = rope_scaling
        if rope_scale != 1.0:
            extra_kwargs['rope_scale'] = rope_scale
        if threads > 0:
            extra_kwargs['n_threads'] = threads
        if batch_size > 0:
            extra_kwargs['batch_size'] = batch_size
        if ubatch_size > 0:
            extra_kwargs['ubatch_size'] = ubatch_size
        if flash_attn:
            extra_kwargs['flash_attn'] = 'auto'  # Use 'auto', 'on', or 'off'
        
        server = LlamaServer(
            model=str(gguf_path),
            host=host,
            port=port,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            device=device_str,
            gpu_backend=device_config.backend.value,
            kv_cache_quant=kv_cache,
            cpu_offload=actual_cpu_offload > 0,
            cpu_offload_layers=actual_cpu_offload,
            **extra_kwargs
        )
        
        console.print(Panel(
            f"[green]Server:[/green] http://{host}:{port}\n"
            f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
            f"[magenta]Backend:[/magenta] {device_config.backend.value}\n"
            f"[cyan]Device:[/cyan] {device_config.device.name}\n"
            f"[yellow]Press Ctrl+C to stop[/yellow]",
            title=server_title
        ))
        
        server.start(wait=False)
        
        import time
        time.sleep(3)
        
        if server._process and server._process.poll() is not None:
            stdout, stderr = server._process.communicate()
            console.print(f"\n[red bold]Server failed to start![/red bold]")
            console.print(f"[dim]Exit code: {server._process.returncode}[/dim]")
            if stdout:
                console.print(f"[dim]stdout:[/dim]")
                for line in stdout.strip().split("\n")[-20:]:
                    if line.strip():
                        console.print(f"[dim]  {line}[/dim]")
            if stderr:
                console.print(f"[red]stderr:[/red]")
                for line in stderr.strip().split("\n")[-20:]:
                    if line.strip():
                        console.print(f"[red]  {line}[/dim]")
            
            console.print(f"\n[yellow]Troubleshooting tips:[/yellow]")
            
            error_lower = (stderr or "").lower()
            
            if "unknown model architecture" in error_lower:
                import re
                arch_match = re.search(r"unknown model architecture: '([^']+)'", error_lower)
                arch_name = arch_match.group(1) if arch_match else "unknown"
                console.print(f"  [red]✗ Model architecture '{arch_name}' not supported by llama.cpp[/red]")
                console.print(f"\n  [yellow]This model requires Ollama's patched llama.cpp with custom patches.[/yellow]")
                console.print(f"  [green]✓ Options:[/green]")
                console.print(f"     1. Use Ollama directly:")
                console.print(f"        [cyan]ollama run {model}[/cyan]")
                console.print(f"     2. Find a compatible GGUF variant on HuggingFace")
            elif server._process.returncode == -6:
                console.print(f"  [red]✗ Server crashed (SIGABRT)[/red]")
                if ollama_model.size_gb > 10:
                    console.print(f"  [yellow]Model size ({ollama_model.size_gb:.1f}GB) may be too large for GPU[/yellow]")
                    console.print(f"  [green]✓ Try CPU offloading:[/green]")
                    console.print(f"      moxing ollama serve {model} -d gpu1 -b {device_config.backend.value} --cpu-offload 20")
                else:
                    console.print(f"  [yellow]This model may require special patches not in standard llama.cpp[/yellow]")
                    console.print(f"  [green]✓ Use Ollama directly:[/green] ollama run {model}")
            elif "out of memory" in error_lower or "cuda error" in error_lower or "hip error" in error_lower:
                console.print(f"  [red]✗ GPU memory insufficient for model ({ollama_model.size_gb:.1f}GB)[/red]")
                if cpu_offload == 0:
                    console.print(f"  [green]✓ Try CPU offloading:[/green]")
                    console.print(f"      moxing ollama serve {model} -d gpu1 -b {device_config.backend.value} --cpu-offload 20")
                    console.print(f"  [green]✓ Or smaller context:[/green]")
                    console.print(f"      moxing ollama serve {model} -d gpu1 -b {device_config.backend.value} -c 1024")
                else:
                    console.print(f"  [green]✓ Increase CPU offload:[/green]")
                    console.print(f"      moxing ollama serve {model} -d gpu1 -b {device_config.backend.value} --cpu-offload 30")
            else:
                console.print(f"  • The GGUF format may be incompatible with llama.cpp")
                console.print(f"  • Try: ollama run {model} (use Ollama directly)")
                console.print(f"  • Or download a compatible GGUF from Hugging Face")
            raise typer.Exit(1)
        
        serve_with_verbose_monitor(server, verbose=verbose, web_monitor=web_monitor)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if server:
            server.stop()
    except RuntimeError as e:
        console.print(f"[red]Runtime error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


def _fast_check_gguf_compatibility(gguf_path: Path):
    """Quick check GGUF compatibility by reading metadata only.
    
    Returns:
        Tuple of (is_compatible, error_message)
    """
    import struct
    
    try:
        with open(gguf_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return False, "Not a valid GGUF file"
            
            version = struct.unpack('<I', f.read(4))[0]
            if version < 2 or version > 4:
                return False, f"Unsupported GGUF version: {version}"
            
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            
            for _ in range(min(metadata_kv_count, 100)):
                try:
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    if key_len > 10000:
                        return True, None
                    key = f.read(key_len).decode('utf-8')
                    value_type = struct.unpack('<I', f.read(4))[0]
                    
                    if value_type == 0:
                        f.read(1)
                    elif value_type == 1:
                        f.read(1)
                    elif value_type == 2:
                        f.read(2)
                    elif value_type == 3:
                        f.read(2)
                    elif value_type == 4:
                        f.read(4)
                    elif value_type == 5:
                        f.read(4)
                    elif value_type == 6:
                        f.read(4)
                    elif value_type == 7:
                        f.read(1)
                    elif value_type == 8:
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        if str_len < 1000000:
                            f.read(str_len)
                        else:
                            return True, None
                    elif value_type == 9:
                        arr_type = struct.unpack('<I', f.read(4))[0]
                        arr_len = struct.unpack('<Q', f.read(8))[0]
                        if arr_len > 100000:
                            return True, None
                        for _ in range(min(arr_len, 100)):
                            if arr_type in [0, 1, 7]:
                                f.read(1)
                            elif arr_type in [2, 3]:
                                f.read(2)
                            elif arr_type in [4, 5, 6]:
                                f.read(4)
                            elif arr_type == 8:
                                slen = struct.unpack('<Q', f.read(8))[0]
                                if slen < 100000:
                                    f.read(slen)
                                else:
                                    break
                except struct.error:
                    return True, None
            
            return True, None
            
    except Exception as e:
        return False, str(e)


def _get_compatible_gguf(model: str, ollama_gguf: Path, size_gb: float, verbose: bool = False) -> Optional[Path]:
    """Get a compatible GGUF file, downloading from Hugging Face if Ollama's is incompatible."""
    from moxing.binaries import get_binary_manager
    
    if verbose:
        console.print(f"[dim]Checking GGUF compatibility for: {model}[/dim]")
        console.print(f"[dim]GGUF path: {ollama_gguf}[/dim]")
        console.print(f"[dim]GGUF size: {ollama_gguf.stat().st_size / (1024**3):.2f} GB[/dim]")
    
    console.print("[blue]Checking GGUF compatibility...[/blue]")
    
    is_compatible, error = _fast_check_gguf_compatibility(ollama_gguf)
    
    if is_compatible:
        console.print("[green]GGUF is compatible[/green]")
        return ollama_gguf
    
    if error:
        console.print(f"[yellow]Warning: {error}[/yellow]")
    
    manager = get_binary_manager()
    if not manager.has_binaries():
        console.print("[blue]Downloading llama.cpp binaries...[/blue]")
        manager.download_binaries()
    
    llama_cli = manager.get_binary_path("llama-cli")
    
    incompatible_patterns = [
        "error loading model",
        "wrong number of tensors",
        "wrong shape",
        "key not found in model",
        "missing tensor",
        "failed to load model",
        "done_getting_tensors",
    ]
    
    fatal_patterns = [
        "unknown model architecture",
        "not supported by llama.cpp",
    ]
    
    try:
        cmd = [str(llama_cli), "-m", str(ollama_gguf), "-n", "1", "-p", "x", "-c", "128", 
               "--no-display-prompt", "-ngl", "0", "--no-conversation", "-st"]
        if verbose:
            console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
            console.print(f"[dim]Running subprocess.run with timeout=120s...[/dim]")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=120,
            cwd=str(llama_cli.parent),
            stdin=subprocess.DEVNULL
        )
        
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode
        
        if verbose:
            console.print(f"[dim]Process completed with return code: {returncode}[/dim]")
            console.print(f"[dim]stdout length: {len(stdout)}[/dim]")
            console.print(f"[dim]stderr length: {len(stderr)}[/dim]")
            if stdout:
                console.print(f"[dim]stdout preview: {stdout[:500]}[/dim]")
            if stderr:
                console.print(f"[dim]stderr output (first 15 lines):[/dim]")
                for line in stderr.strip().split("\n")[:15]:
                    console.print(f"[dim]  {line}[/dim]")
            else:
                console.print(f"[dim]No stderr output[/dim]")
        
        if stderr:
            stderr_lower = stderr.lower()
            is_fatal = any(p in stderr_lower for p in fatal_patterns)
            is_incompatible = any(p in stderr_lower for p in incompatible_patterns)
            
            if is_fatal:
                console.print(f"[red]Error: Model architecture not supported by llama.cpp[/red]")
                for line in stderr.strip().split("\n"):
                    if "unknown model architecture" in line.lower():
                        arch_match = line.split("'")
                        if len(arch_match) >= 2:
                            arch_name = arch_match[1]
                            console.print(f"[red]  Architecture: {arch_name}[/red]")
                raise UnsupportedArchitectureError(
                    "Model architecture not supported by llama.cpp. "
                    "This model requires Ollama's patched llama.cpp or MLX backend."
                )
            
            if is_incompatible:
                console.print(f"[yellow]Warning: GGUF format may not be fully compatible with llama.cpp[/yellow]")
                console.print(f"[dim]The model may still work, but some features might be limited.[/dim]")
                console.print(f"[dim]Using the Ollama GGUF file directly...[/dim]")
                return ollama_gguf
            
            console.print(f"[green]GGUF is compatible[/green]")
            return ollama_gguf
        else:
            if returncode == 0:
                console.print(f"[green]GGUF is compatible[/green]")
                return ollama_gguf
            else:
                console.print(f"[yellow]Warning: llama-cli exited with code {returncode}[/yellow]")
                if stdout:
                    console.print(f"[dim]stdout: {stdout[:300]}[/dim]")
                console.print(f"[dim]Proceeding with the GGUF file anyway...[/dim]")
                return ollama_gguf
            
    except subprocess.TimeoutExpired:
        console.print(f"[yellow]Warning: Model loading timed out (120s)[/yellow]")
        console.print(f"[dim]The model might be too large or incompatible. Proceeding anyway...[/dim]")
        return ollama_gguf
    except Exception as e:
        console.print(f"[yellow]Compatibility check error: {e}[/yellow]")
        import traceback
        if verbose:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        console.print(f"[dim]Proceeding with the GGUF file anyway...[/dim]")
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
    prompt: str = typer.Option(None, "-p", "--prompt", help="Single prompt (leave empty for interactive chat)"),
    tokens: int = typer.Option(256, "-n", "--tokens", help="Max tokens to generate"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed monitoring and statistics"),
    backend: str = typer.Option("auto", "-b", "--backend", help="Backend: auto, vulkan, cuda, metal, cpu"),
    kv_cache: str = typer.Option("auto", "--kv-cache", help="KV cache quantization: auto, f16, q8_0, q4_0, tq4, tq3.5, tq3, tq2.5, tq2"),
    device: str = typer.Option("auto", "-d", "--device", help="Device: auto, gpu0, gpu1, cpu"),
):
    """Run an Ollama model with detailed monitoring support.
    
    Examples:
        moxing ollama run carstenuhlig/omnicoder-9b           # Interactive chat
        moxing ollama run carstenuhlig/omnicoder-9b -p "Hello" # Single prompt
        moxing ollama run omnicoder-9b -v                      # Verbose monitoring
        moxing ollama run omnicoder-9b -v -c 65536             # Large context + verbose
        moxing ollama run omnicoder-9b --kv-cache tq2 -v       # TurboQuant + verbose
    """
    from moxing.server import find_available_port, LlamaServer
    from moxing.ollama import get_ollama_model, OllamaClient
    from moxing.device import DeviceDetector, BackendType
    
    ollama_model = get_ollama_model(model)
    if not ollama_model:
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("[dim]Run 'moxing ollama list' to see available models[/dim]")
        raise typer.Exit(1)
    
    client = OllamaClient()
    is_accessible, gguf_path, message = client.check_model_access(model)
    
    if not is_accessible:
        console.print(f"[red]Cannot access model: {message}[/red]")
        raise typer.Exit(1)
    
    port = find_available_port(8080)
    
    detector = DeviceDetector()
    detector.detect()
    
    if device != "auto":
        device_config = detector.get_device_config_by_name(device, backend, ollama_model.size_gb)
    else:
        device_config = detector.get_best_device(ollama_model.size_gb)
    
    device_str = "auto"
    if device_config.device.backend != BackendType.CPU:
        if device_config.backend == BackendType.METAL:
            device_str = f"MTL{device_config.device.index}"
        elif device_config.backend == BackendType.CUDA:
            device_str = f"CUDA{device_config.device.index}"
        else:
            device_str = "auto"
    
    console.print(Panel(
        f"[cyan]Model:[/cyan] {ollama_model.full_name}\n"
        f"[cyan]Size:[/cyan] {ollama_model.size_gb:.1f} GB\n"
        f"[cyan]Context:[/cyan] {ctx_size}\n"
        f"[cyan]Backend:[/cyan] {device_config.backend.value}\n"
        f"[cyan]Device:[/cyan] {device_config.device.name}\n"
        f"[cyan]KV Cache:[/cyan] {kv_cache}",
        title="Configuration"
    ))
    
    server = None
    try:
        server = LlamaServer(
            model=str(gguf_path),
            host="127.0.0.1",
            port=port,
            ctx_size=ctx_size,
            n_gpu_layers=device_config.n_gpu_layers,
            device=device_str,
            gpu_backend=device_config.backend.value,
            kv_cache_quant=kv_cache,
            quiet=False
        )
        
        console.print("[blue]Loading model...[/blue]")
        server.start(wait=True)
        
        console.print("[green]Model loaded![/green]")
        console.print("[green]Interactive chat ready! Type 'exit' or 'quit' to end.[/green]")
        console.print("[dim]Ctrl+C to stop[/dim]\n")
        
        run_with_verbose_monitor(
            server=server,
            model_name=ollama_model.full_name,
            prompt=prompt,
            max_tokens=tokens,
            verbose=verbose
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)
    finally:
        if server:
            server.stop()


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


turboquant_app = typer.Typer(name="turboquant", help="TurboQuant KV cache compression commands")
app.add_typer(turboquant_app, name="turboquant")


@turboquant_app.command("info")
def turboquant_info():
    """Show TurboQuant information and recommendations.
    
    TurboQuant (arXiv:2504.19874) is a near-optimal KV cache quantization algorithm.
    """
    console.print(Panel(
        "[bold cyan]TurboQuant: Near-Optimal KV Cache Quantization[/bold cyan]\n\n"
        "Based on Google's paper: arXiv:2504.19874\n\n"
        "[bold]Key Features:[/bold]\n"
        "• Data-oblivious online quantization\n"
        "• Near-optimal distortion (within 2.7x of theoretical optimum)\n"
        "• Unbiased inner product estimation\n"
        "• Mixed precision support (3.5 bits, 2.5 bits)",
        title="TurboQuant Overview"
    ))
    
    table = Table(title="Available Quantization Types")
    table.add_column("Type", style="cyan")
    table.add_column("Bits", style="yellow")
    table.add_column("Compression", style="green")
    table.add_column("Quality", style="magenta")
    table.add_column("Use Case", style="blue")
    
    quants = [
        ("f16", "16", "1x", "Perfect", "Baseline"),
        ("q8_0", "8", "2x", "Excellent", "Quality-first"),
        ("q5_0", "5", "3.2x", "Very Good", "Good balance"),
        ("q4_0", "4", "4x", "Good", "Recommended"),
        ("tq4", "4", "4x", "High", "TurboQuant 4-bit"),
        ("tq3.5", "3.5", "4.6x", "Quality Neutral ⭐", "Best for quality"),
        ("tq3", "3", "5.3x", "Good", "TurboQuant 3-bit"),
        ("tq2.5", "2.5", "6.4x", "Slight Loss ⭐", "Best for memory"),
        ("tq2", "2", "8x", "Acceptable", "Maximum compression"),
    ]
    
    for q in quants:
        table.add_row(*q)
    
    console.print(table)
    
    console.print(Panel(
        "[bold]Recommended Commands:[/bold]\n\n"
        "[cyan]# Quality neutral (3.5-bit)[/cyan]\n"
        "moxing serve model.gguf --kv-cache tq3.5 -c 32768\n\n"
        "[cyan]# Memory efficient (2.5-bit)[/cyan]\n"
        "moxing serve model.gguf --kv-cache tq2.5 -c 65536\n\n"
        "[cyan]# Maximum compression (2-bit)[/cyan]\n"
        "moxing serve model.gguf --kv-cache tq2 -c 131072\n\n"
        "[cyan]# Ollama models[/cyan]\n"
        "moxing ollama serve llama3 --kv-cache tq3.5\n\n"
        "[cyan]# Using llama.cpp built-in q4_0 (recommended)[/cyan]\n"
        "moxing serve model.gguf --kv-cache q4_0 -c 65536",
        title="Usage Examples"
    ))


@turboquant_app.command("estimate")
def turboquant_estimate(
    model_size: float = typer.Argument(..., help="Model size in GB (e.g., 7, 9, 70)"),
    ctx_size: int = typer.Option(32768, "-c", "--ctx", help="Context size"),
    n_layers: int = typer.Option(0, "-l", "--layers", help="Number of layers (0=auto)"),
    n_heads: int = typer.Option(0, "--heads", help="Number of attention heads (0=auto)"),
    head_dim: int = typer.Option(0, "-d", "--head-dim", help="Head dimension (0=auto)"),
):
    """Estimate KV cache memory usage for different quantization types.
    
    Example:
        moxing turboquant estimate 9 -c 65536
        moxing turboquant estimate 70 -c 32768
    """
    from moxing.kv_cache import (
        KVCacheQuantType, estimate_kv_cache_size_gb, get_model_kv_params
    )
    
    if n_layers == 0 or n_heads == 0 or head_dim == 0:
        n_layers, n_heads, head_dim = get_model_kv_params(model_size)
    
    console.print(Panel(
        f"[cyan]Model Size:[/cyan] {model_size:.1f} GB\n"
        f"[cyan]Context:[/cyan] {ctx_size:,}\n"
        f"[cyan]Layers:[/cyan] {n_layers}\n"
        f"[cyan]Heads:[/cyan] {n_heads}\n"
        f"[cyan]Head Dim:[/cyan] {head_dim}",
        title="Model Configuration"
    ))
    
    table = Table(title=f"KV Cache Memory (Context: {ctx_size:,})")
    table.add_column("Type", style="cyan")
    table.add_column("Bits", style="yellow")
    table.add_column("KV Size", style="green")
    table.add_column("vs F16", style="magenta")
    table.add_column("Quality", style="blue")
    
    quants = [
        (KVCacheQuantType.F16, "Baseline"),
        (KVCacheQuantType.Q8_0, "Excellent"),
        (KVCacheQuantType.Q5_0, "Very Good"),
        (KVCacheQuantType.Q4_0, "Good"),
        (KVCacheQuantType.TURBOQUANT_4, "High"),
        (KVCacheQuantType.TURBOQUANT_35, "Quality Neutral ⭐"),
        (KVCacheQuantType.TURBOQUANT_3, "Good"),
        (KVCacheQuantType.TURBOQUANT_25, "Slight Loss ⭐"),
        (KVCacheQuantType.TURBOQUANT_2, "Acceptable"),
    ]
    
    f16_size = None
    for quant, quality in quants:
        size = estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, ctx_size, 1, quant)
        
        if f16_size is None:
            f16_size = size
            vs_f16 = "-"
        else:
            savings = (1 - size / f16_size) * 100
            vs_f16 = f"-{savings:.0f}%"
        
        table.add_row(
            quant.value,
            f"{quant.bits:.1f}",
            f"{size:.2f} GB",
            vs_f16,
            quality
        )
    
    console.print(table)
    
    console.print(Panel(
        f"[bold green]Recommended:[/bold green]\n\n"
        f"• Quality first: [cyan]--kv-cache tq3.5[/cyan]\n"
        f"• Memory efficient: [cyan]--kv-cache tq2.5[/cyan]\n"
        f"• Built-in q4_0: [cyan]--kv-cache q4_0[/cyan] (recommended)\n\n"
        f"[dim]Note: TurboQuant types (tq*) use llama.cpp's nearest equivalent.[/dim]",
        title="Recommendations"
    ))


@turboquant_app.command("test")
def turboquant_test(
    dim: int = typer.Option(128, "-d", "--dim", help="Vector dimension"),
    n_vectors: int = typer.Option(100, "-n", "--num", help="Number of test vectors"),
    bits: float = typer.Option(3.5, "-b", "--bits", help="Bits per channel"),
):
    """Run TurboQuant algorithm test.
    
    Example:
        moxing turboquant test -d 128 -n 1000 -b 3.5
    """
    import numpy as np
    from moxing.turboquant import TurboQuant, TurboQuantConfig, TurboQuantMode
    
    console.print(f"[blue]Testing TurboQuant with {bits} bits per channel...[/blue]")
    
    np.random.seed(42)
    test_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
    test_vectors = test_vectors / norms
    
    config = TurboQuantConfig(
        dim=dim,
        bits_per_channel=bits,
        mode=TurboQuantMode.INNER_PRODUCT,
        seed=42,
    )
    
    tq = TurboQuant(config)
    
    import time
    start = time.perf_counter()
    compressed = tq.quantize(test_vectors)
    quant_time = time.perf_counter() - start
    
    start = time.perf_counter()
    reconstructed = tq.dequantize(compressed)
    dequant_time = time.perf_counter() - start
    
    mse = np.mean((test_vectors - reconstructed) ** 2)
    
    query_vectors = np.random.randn(10, dim).astype(np.float32)
    biases = []
    for q in query_vectors:
        true_ips = np.dot(test_vectors, q)
        est_ips = np.dot(reconstructed, q)
        biases.extend(est_ips - true_ips)
    mean_bias = np.mean(biases)
    
    console.print(Panel(
        f"[cyan]Configuration:[/cyan]\n"
        f"  Dimension: {dim}\n"
        f"  Vectors: {n_vectors}\n"
        f"  Bits: {bits}\n\n"
        f"[green]Results:[/green]\n"
        f"  MSE: {mse:.6f}\n"
        f"  Inner Product Bias: {mean_bias:.6f} (ideal: 0)\n"
        f"  Quantize Time: {quant_time*1000:.2f} ms\n"
        f"  Dequantize Time: {dequant_time*1000:.2f} ms\n"
        f"  Compression: {16/bits:.1f}x\n\n"
        f"[magenta]Unbiased: {'✓ Yes' if abs(mean_bias) < 0.05 else '✗ No'}[/magenta]",
        title="TurboQuant Test Results"
    ))


monitor_app = typer.Typer(name="monitor", help="Real-time monitoring commands")
app.add_typer(monitor_app, name="monitor")


@monitor_app.command("start")
def monitor_start(
    host: str = typer.Option("127.0.0.1", "--host", help="Monitor server host"),
    port: int = typer.Option(9090, "-p", "--port", help="Monitor server port"),
    llama_port: int = typer.Option(8080, "-l", "--llama-port", help="llama.cpp server port"),
):
    """Start the web-based monitoring dashboard.
    
    Displays real-time metrics including:
    - GPU/CPU memory usage
    - Token generation speed
    - Request statistics
    - Slot status
    
    Example:
        # Terminal 1: Start llama.cpp server
        moxing serve model.gguf -p 8080
        
        # Terminal 2: Start monitor
        moxing monitor start --llama-port 8080
        
        # Open browser: http://127.0.0.1:9090
    """
    from moxing.monitor import start_monitor_server
    
    console.print(Panel(
        f"[bold cyan]MoXing Monitor Dashboard[/bold cyan]\n\n"
        f"[green]Monitor URL:[/green] http://{host}:{port}\n"
        f"[green]Server URL:[/green] http://127.0.0.1:{llama_port}\n\n"
        f"[yellow]Make sure llama.cpp server is running![/yellow]\n"
        f"[dim]Start with: moxing serve model.gguf -p {llama_port}[/dim]",
        title="Starting Monitor"
    ))
    
    start_monitor_server(host, port, llama_port)


@monitor_app.command("cli")
def monitor_cli(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
):
    """Show live metrics in terminal.
    
    Displays real-time statistics in the terminal.
    
    Example:
        moxing monitor cli --port 8080
    """
    from moxing.monitor import print_live_metrics
    
    print_live_metrics(host, port)


@monitor_app.command("open")
def monitor_open(
    port: int = typer.Option(8080, "-p", "--port", help="llama.cpp server port"),
):
    """Open the built-in monitoring page.
    
    Opens the monitoring page in your browser.
    Requires llama.cpp server running with --metrics enabled.
    
    Example:
        moxing monitor open --port 8080
    """
    import webbrowser
    
    url = f"http://127.0.0.1:{port}"
    
    console.print(f"[blue]Opening browser: {url}[/blue]")
    console.print("[dim]Note: llama.cpp server must be running with --metrics enabled[/dim]")
    
    webbrowser.open(url)


@monitor_app.command("stats")
def monitor_stats(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
):
    """Show current server statistics.
    
    Example:
        moxing monitor stats --port 8080
    """
    from moxing.monitor import MetricsCollector
    
    collector = MetricsCollector(host, port)
    metrics = collector.fetch_metrics()
    slots = collector.fetch_slots()
    props = collector.fetch_props()
    
    if props:
        model_name = Path(props.get("model_path", "Unknown")).name
        console.print(Panel(
            f"[cyan]Model:[/cyan] {model_name}\n"
            f"[cyan]Context:[/cyan] {props.get('n_ctx', '--')}\n"
            f"[cyan]Batch:[/cyan] {props.get('n_batch', '--')}",
            title="Model Info"
        ))
    
    if metrics:
        table = Table(title="Server Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Prompt Tokens", f"{metrics.prompt_tokens_total:,}")
        table.add_row("Generated Tokens", f"{metrics.tokens_predicted_total:,}")
        table.add_row("Total Tokens", f"{metrics.prompt_tokens_total + metrics.tokens_predicted_total:,}")
        table.add_row("Prompt Speed", f"{metrics.prompt_tokens_per_second:.1f} tok/s")
        table.add_row("Generate Speed", f"{metrics.predicted_tokens_per_second:.1f} tok/s")
        table.add_row("Processing", str(metrics.requests_processing))
        table.add_row("Deferred", str(metrics.requests_deferred))
        
        console.print(table)
    
    if slots:
        slots_table = Table(title="Slots")
        slots_table.add_column("ID", style="cyan")
        slots_table.add_column("Status", style="green")
        slots_table.add_column("Context", style="yellow")
        
        for slot in slots:
            status = "[green]Processing[/green]" if slot.get("is_processing") else "[dim]Idle[/dim]"
            slots_table.add_row(
                str(slot.get("id", "?")),
                status,
                str(slot.get("n_ctx", "--"))
            )
        
        console.print(slots_table)
    
    if not metrics and not slots:
        console.print(f"[red]Failed to connect to server at {host}:{port}[/red]")


if __name__ == "__main__":
    app()