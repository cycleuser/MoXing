import os
import socket
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


class UnsupportedArchitectureError(Exception):
    """Raised when a model architecture is not supported by llama.cpp."""

    pass


ollama_app = typer.Typer(name="ollama", help="Manage Ollama models")


@ollama_app.command("list")
def ollama_list(
    show_embeddings: bool = typer.Option(
        True, "--embeddings/--no-embeddings", help="Show embedding models"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    select: bool = typer.Option(False, "--select", "-s", help="Interactive model selection"),
    show_context: bool = typer.Option(
        True, "--context/--no-context", "-C", help="Show context length"
    ),
):
    """List all models from local Ollama installation.

    Shows model name, size, context length, and type.

    Examples:
        moxing ollama list
        moxing ollama list --no-embeddings
        moxing ollama list --no-context
    """
    from moxing.ollama import OllamaClient, list_ollama_models, print_ollama_models

    models_list = list_ollama_models()

    if not show_embeddings:
        client = OllamaClient()
        models_list = [m for m in models_list if not client.is_embedding_model(m.full_name)]

    if json_output:
        import json

        data = [
            {
                "name": m.full_name,
                "size": m.size,
                "size_gb": round(m.size_gb, 2),
                "id": m.id,
                "modified": m.modified,
                "family": m.family,
                "parameter_size": m.parameter_size,
                "quantization": m.quantization_level,
                "context_length": m.context_length,
            }
            for m in models_list
        ]
        print(json.dumps(data, indent=2))
    elif select:
        if not models_list:
            console.print("[yellow]No Ollama models found.[/yellow]")
            return

        console.print("\n[bold]Select a model to serve:[/bold]\n")

        for i, m in enumerate(models_list, 1):
            size_str = f"{m.size_gb:.1f} GB" if m.size_gb >= 1 else f"{m.size / (1024**2):.0f} MB"
            console.print(f"  [cyan]{i:2d}[/cyan]. {m.full_name:<40} [dim]{size_str}[/dim]")

        console.print()
        selection = Prompt.ask("[bold]Enter number[/bold]", default="1")

        try:
            idx = int(selection) - 1
            if 0 <= idx < len(models_list):
                selected = models_list[idx]
                console.print(f"\n[green]Selected: {selected.full_name}[/green]\n")

                action = Prompt.ask(
                    "[bold]Action[/bold]",
                    choices=["serve", "info", "run", "cancel"],
                    default="serve",
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
        print_ollama_models(models_list, show_embeddings, show_context)


@ollama_app.command("serve")
def ollama_serve(
    model: str = typer.Argument(..., help="Ollama model name (e.g., gemma3:4b)"),
    port: int = typer.Option(11434, "-p", "--port", help="Ollama API port (default: 11434)"),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Server host (use 0.0.0.0 for LAN access)"
    ),
    ctx_size: int = typer.Option(0, "-c", "--ctx-size", help="Context size (0=model default)"),
    device: str = typer.Option("auto", "-d", "--device", help="Device: auto, gpu0, gpu1, cpu"),
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="Backend: auto, cuda, rocm, vulkan, cpu"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show verbose output from Ollama"
    ),
):
    """Serve an Ollama model using the system Ollama daemon with OpenAI-compatible API.

    Uses Ollama's built-in runner which supports all Ollama models natively.
    The Ollama API is exposed at the specified port.

    Host Binding:
    - 127.0.0.1 (default): Local access only
    - 0.0.0.0: Allow LAN access (all network interfaces)

    Backends:
    - cuda: NVIDIA GPUs (fastest)
    - rocm: AMD GPUs with ROCm
    - vulkan: Cross-platform GPU (works with AMD/NVIDIA/Intel)
    - cpu: CPU only

    Examples:
        moxing ollama serve gemma3:4b
        moxing ollama serve gemma3:4b -b cuda
        moxing ollama serve omnicoder-9b -v
        moxing ollama serve omnicoder-9b --host 0.0.0.0
    """
    serve_with_ollama_backend(
        model, port, host, ctx_size, device, backend, verbose
    )


@ollama_app.command("info")
def ollama_info(
    model: str = typer.Argument(..., help="Ollama model name"),
):
    """Show detailed information about an Ollama model."""
    from rich.panel import Panel

    from moxing.ollama import OllamaClient, get_ollama_model

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
            lines.append(
                f"[cyan]Quantization:[/cyan] {details.get('quantization_level', 'unknown')}"
            )

        modelfile = info.get("modelfile", "")
        if modelfile:
            lines.append("\n[dim]Modelfile:[/dim]")
            for line in modelfile.split("\n")[:10]:
                if line.strip():
                    lines.append(f"  [dim]{line}[/dim]")

    console.print(Panel("\n".join(lines), title=f"Ollama Model: {model}"))


@ollama_app.command("run")
def ollama_run(
    model: str = typer.Argument(..., help="Ollama model name"),
    prompt: str = typer.Option(
        None, "-p", "--prompt", help="Single prompt (leave empty for interactive chat)"
    ),
    tokens: int = typer.Option(256, "-n", "--tokens", help="Max tokens to generate"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show detailed monitoring and statistics"
    ),
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="Backend: auto, vulkan, cuda, metal, cpu"
    ),
    kv_cache: str = typer.Option(
        "auto",
        "--kv-cache",
        help="KV cache quantization: auto, f16, q8_0, q4_0, tq4, tq3.5, tq3, tq2.5, tq2",
    ),
    device: str = typer.Option("auto", "-d", "--device", help="Device: auto, gpu0, gpu1, cpu"),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Server host (use 0.0.0.0 for LAN access)"
    ),
):
    """Run an Ollama model with detailed monitoring support.

    Host Binding:
    - 127.0.0.1 (default): Local access only
    - 0.0.0.0: Allow LAN access (all network interfaces)

    Examples:
        moxing ollama run carstenuhlig/omnicoder-9b           # Interactive chat
        moxing ollama run carstenuhlig/omnicoder-9b -p "Hello" # Single prompt
        moxing ollama run omnicoder-9b -v                      # Verbose monitoring
        moxing ollama run omnicoder-9b -v -c 65536             # Large context + verbose
        moxing ollama run omnicoder-9b --kv-cache tq2 -v       # TurboQuant + verbose
        moxing ollama run omnicoder-9b --host 0.0.0.0          # LAN access
    """
    from moxing.cli.serve import run_with_verbose_monitor
    from moxing.device import BackendType, DeviceDetector
    from moxing.ollama import OllamaClient, get_ollama_model
    from moxing.server import LlamaServer, find_available_port

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

    console.print(
        Panel(
            f"[cyan]Model:[/cyan] {ollama_model.full_name}\n"
            f"[cyan]Size:[/cyan] {ollama_model.size_gb:.1f} GB\n"
            f"[cyan]Context:[/cyan] {ctx_size}\n"
            f"[cyan]Backend:[/cyan] {device_config.backend.value}\n"
            f"[cyan]Device:[/cyan] {device_config.device.name}\n"
            f"[cyan]KV Cache:[/cyan] {kv_cache}",
            title="Configuration",
        )
    )

    server = None
    try:
        server = LlamaServer(
            model=str(gguf_path),
            host=host,
            port=port,
            ctx_size=ctx_size,
            n_gpu_layers=device_config.n_gpu_layers,
            device=device_str,
            gpu_backend=device_config.backend.value,
            kv_cache_quant=kv_cache,
            quiet=False,
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
            verbose=verbose,
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1) from e
    finally:
        if server:
            server.stop()


@ollama_app.command("tune")
def ollama_tune(
    model: str = typer.Argument(..., help="Ollama model name (e.g., gemma3:4b)"),
    force: bool = typer.Option(False, "-f", "--force", help="Force re-tuning even if cache exists"),
):
    """Run warmup benchmark to find optimal parameters for an Ollama model.

    Measures actual performance to find the best configuration:
    - Optimal context size
    - Best ubatch size
    - MoE offloading strategy
    - KV cache type selection

    Results are cached for 30 days. Second launch skips warmup (2s startup).

    Examples:
        moxing ollama tune qwen3:30b-a3b
        moxing ollama tune gemma3:4b --force
    """
    from moxing.binaries import get_binary_manager
    from moxing.device import DeviceDetector
    from moxing.gguf_metadata import extract_model_architecture
    from moxing.ollama import OllamaClient
    from moxing.warmup_benchmark import ProfileCache, WarmupBenchmark, get_hardware_fingerprint

    console.print(f"[blue]Looking up model: {model}[/blue]")

    client = OllamaClient()
    model_path = client.get_model_gguf_path(model)

    if not model_path:
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("[dim]Run 'ollama pull {model}' to download[/dim]")
        raise typer.Exit(1)

    console.print(
        f"[green]Found model: {model} ({model_path.stat().st_size / (1024**3):.1f} GB)[/green]"
    )
    console.print(f"[dim]Path: {model_path}[/dim]")

    console.print("\n[blue]Analyzing model architecture...[/blue]")

    try:
        arch = extract_model_architecture(model_path)
        console.print(f"[green]Architecture: {arch.architecture}[/green]")
        console.print(f"[green]Parameters: {arch.parameter_count_b:.1f}B[/green]")
        console.print(f"[green]Quantization: {arch.quantization}[/green]")

        if arch.is_moe:
            console.print(
                f"[yellow]MoE detected: {arch.expert_count} experts, "
                f"{arch.expert_used_count} active[/yellow]"
            )
            console.print(
                "[yellow]Recommendation: Use --cpu-moe for 7-8x speedup "
                "on constrained hardware[/yellow]"
            )
        else:
            console.print("[green]Dense model (no MoE)[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not extract model architecture: {e}[/yellow]")
        arch = None

    console.print("\n[blue]Detecting hardware...[/blue]")
    detector = DeviceDetector()
    devices = detector.detect()

    gpu_devices = [d for d in devices if d.backend.is_gpu()]
    if not gpu_devices:
        console.print("[yellow]No GPU detected, using CPU[/yellow]")
        return

    best_gpu = max(gpu_devices, key=lambda d: d.free_memory_mb)
    console.print(
        f"[green]Best GPU: {best_gpu.name} ({best_gpu.free_memory_gb:.1f}GB free)[/green]"
    )

    hardware_fp = get_hardware_fingerprint(
        gpu_backend=best_gpu.backend.value,
        gpu_name=best_gpu.name,
        gpu_vram_mb=best_gpu.free_memory_mb,
    )

    cache = ProfileCache()
    model_id = model.replace("/", "_").replace(":", "_")
    hardware_fp_str = hardware_fp.to_hash()

    if not force:
        cached = cache.load(model_id, hardware_fp_str)
        if cached is not None:
            console.print(
                f"\n[green]Using cached profile ({cached.measured_tps:.1f} tok/s)[/green]"
            )
            console.print(f"[dim]Context: {cached.ctx_size}, ubatch: {cached.ubatch_size}[/dim]")
            if cached.cpu_moe:
                console.print("[dim]MoE mode: CPU experts, GPU attention[/dim]")
            console.print(f"\n[yellow]To re-tune, use: moxing ollama tune {model} --force[/yellow]")
            return

    console.print("\n[blue]Running warmup benchmark...[/blue]")

    manager = get_binary_manager()
    if not manager.has_binaries():
        console.print("[blue]Downloading llama.cpp binaries...[/blue]")
        manager.download_binaries()

    binary_path = manager.get_binary_path("llama-server")

    bench = WarmupBenchmark(
        model_path=model_path,
        binary_path=binary_path,
        hardware_fp=hardware_fp,
        cache=cache,
    )

    profile = bench.run()

    if profile is not None:
        console.print("\n[green bold]Optimal Configuration:[/green bold]")
        console.print(f"[green]Context size: {profile.ctx_size}[/green]")
        console.print(f"[green]ubatch size: {profile.ubatch_size}[/green]")
        console.print(f"[green]Threads: {profile.n_threads}[/green]")
        console.print(f"[green]Measured performance: {profile.measured_tps:.1f} tok/s[/green]")

        if profile.cpu_moe:
            console.print("[yellow]MoE mode: CPU experts, GPU attention[/yellow]")
            console.print(f"[yellow]Use: moxing ollama serve {model} --cpu-moe[/yellow]")

        console.print("\n[dim]Profile cached for 30 days[/dim]")
    else:
        console.print("[yellow]Benchmark failed, using default configuration[/yellow]")


def serve_with_ollama_backend(
    model: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    ctx_size: int = 4096,
    device: str = "auto",
    backend: str = "auto",
    verbose: bool = False,
    gguf_path: Optional[Path] = None,
):
    """Serve a model using system Ollama with device/backend selection.

    For Ollama-specific architectures (like gemma4), uses system Ollama
    which has the native runner with custom patches.
    """
    import shutil
    import subprocess
    import time

    from moxing.device import BackendType, DeviceDetector

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        console.print("[red]Ollama not found in PATH[/red]")
        console.print("[yellow]Install Ollama: https://ollama.ai[/yellow]")
        raise typer.Exit(1)

    detector = DeviceDetector()
    detector.detect()

    env = os.environ.copy()
    backend_str = backend.upper() if backend != "auto" else "AUTO"
    device_str = device if device != "auto" else "auto"
    gpu_idx = None

    if backend == "cuda":
        env["OLLAMA_LLM_LIBRARY"] = "cuda_v12"
        backend_str = "CUDA"
    elif backend == "rocm":
        env["OLLAMA_LLM_LIBRARY"] = "rocm"
        backend_str = "ROCm"
    elif backend == "vulkan":
        env["OLLAMA_VULKAN"] = "1"
        backend_str = "Vulkan"
    elif backend == "cpu":
        env["OLLAMA_LLM_LIBRARY"] = "cpu"
        backend_str = "CPU"
    elif backend == "auto":
        if device.startswith("gpu") and len(device) > 4:
            try:
                gpu_idx = int(device[4:])
                dev_info = detector.get_device_by_name(device)
                if dev_info:
                    if dev_info.backend == BackendType.CUDA:
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                        backend_str = "CUDA"
                    elif dev_info.backend == BackendType.ROCM:
                        env["HIP_VISIBLE_DEVICES"] = str(gpu_idx)
                        env["OLLAMA_LLM_LIBRARY"] = "rocm"
                        backend_str = "ROCm"
                    elif dev_info.backend == BackendType.VULKAN:
                        env["OLLAMA_VULKAN"] = "1"
                        backend_str = "Vulkan"
                    device_str = f"GPU {gpu_idx} ({dev_info.name})"
            except ValueError:
                pass

    if device.startswith("gpu") and len(device) > 4 and gpu_idx is None:
        try:
            gpu_idx = int(device[4:])
            if backend == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            elif backend == "rocm":
                env["HIP_VISIBLE_DEVICES"] = str(gpu_idx)
            device_str = f"GPU {gpu_idx}"
        except ValueError:
            pass

    if gpu_idx is not None:
        try:
            dev_info = detector.get_device_by_name(f"gpu{gpu_idx}")
            if dev_info:
                device_str = f"GPU {gpu_idx} ({dev_info.name})"
        except:  # noqa: E722
            pass

    ollama_running = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", 11434))
        sock.close()
        ollama_running = result == 0
    except:  # noqa: E722
        pass

    ollama_process = None

    if not ollama_running:
        console.print(
            Panel(
                f"[green]Model:[/green] {model}\n"
                f"[blue]Backend:[/blue] {backend_str}\n"
                f"[cyan]Device:[/cyan] {device_str}\n"
                f"[yellow]Port:[/yellow] 11434",
                title="Ollama Backend",
            )
        )

        if verbose:
            console.print("[dim]Environment variables:[/dim]")
            for key in [
                "OLLAMA_LLM_LIBRARY",
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "OLLAMA_VULKAN",
            ]:
                if key in env:
                    console.print(f"[dim]  {key}={env[key]}[/dim]")

        console.print("[blue]Starting Ollama service...[/blue]")

        ollama_process = subprocess.Popen(
            [ollama_bin, "serve"],
            env=env,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
            start_new_session=True,
        )

        for _ in range(20):
            time.sleep(1)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", 11434))
                sock.close()
                if result == 0:
                    break
            except:  # noqa: E722
                pass
            if ollama_process.poll() is not None:
                console.print("[red]Ollama failed to start[/red]")
                raise typer.Exit(1)
    else:
        console.print(
            Panel(
                f"[green]Model:[/green] {model}\n"
                f"[blue]Backend:[/blue] {backend_str}\n"
                f"[cyan]Device:[/cyan] {device_str}\n"
                f"[yellow]Port:[/yellow] 11434\n"
                f"[dim]Using existing Ollama service[/dim]",
                title="Ollama Backend",
            )
        )

        if verbose:
            console.print("[dim]Environment variables for model loading:[/dim]")
            for key in [
                "OLLAMA_LLM_LIBRARY",
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "OLLAMA_VULKAN",
            ]:
                if key in env:
                    console.print(f"[dim]  {key}={env[key]}[/dim]")

    console.print(f"[blue]Loading model {model}...[/blue]")

    load_result = subprocess.run(
        [ollama_bin, "run", model, "--keepalive", "24h"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if load_result.returncode != 0:
        console.print("[red]Failed to load model[/red]")
        if load_result.stderr:
            console.print(f"[dim]{load_result.stderr[:500]}[/dim]")
        raise typer.Exit(1)

    console.print(
        Panel(
            "[green]Ollama API:[/green] http://127.0.0.1:11434/api\n"
            "[green]OpenAI API:[/green] http://127.0.0.1:11434/v1\n"
            "[yellow]Press Ctrl+C to stop[/yellow]",
            title=f"Running: {model}",
        )
    )

    console.print(f"[dim]Backend: {backend_str}, Device: {device_str}[/dim]")

    try:
        while True:
            time.sleep(1)
            if ollama_process and ollama_process.poll() is not None:
                console.print(f"[red]Ollama exited with code {ollama_process.returncode}[/red]")
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        if ollama_process:
            ollama_process.terminate()
            try:
                ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_process.kill()

    raise typer.Exit(0)

