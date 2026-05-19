import os
import socket
import struct
import subprocess
import time
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


@ollama_app.command("download-runner")
def ollama_download_runner(
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="Backend: auto, cuda, rocm, vulkan, cpu, all"
    ),
    version: str = typer.Option(None, "-v", "--version", help="Ollama version (default: latest)"),
    force: bool = typer.Option(False, "-f", "--force", help="Force re-download"),
    list_available: bool = typer.Option(False, "-l", "--list", help="List available backends"),
):
    """Download Ollama runners for different backends.

    Ollama provides pre-built runners that support all Ollama models.
    MoXing can use these runners without needing Ollama installed.

    Examples:
        moxing ollama download-runner --list
        moxing ollama download-runner -b cuda
        moxing ollama download-runner -b rocm
        moxing ollama download-runner -b all
    """
    from moxing.ollama_runner import OllamaRunnerDownloader

    downloader = OllamaRunnerDownloader(version)

    if list_available:
        console.print(f"\n[bold]Platform: {downloader.detect_platform()}[/bold]")
        console.print("[bold]Available backends:[/bold]")
        for b in downloader.list_available_backends():
            status = (
                "[green]✓ installed[/green]"
                if downloader.has_runner(b)
                else "[dim]not installed[/dim]"
            )
            console.print(f"  {b}: {status}")
        console.print("\n[dim]Use --backend <name> to download[/dim]")
        console.print("[dim]Use --backend all to download all[/dim]")
        return

    if backend == "auto":
        backends = downloader.list_available_backends()
        backend = backends[0] if backends else "cuda"
        console.print(f"[blue]Auto-selecting backend: {backend}[/blue]")

    if backend == "all":
        results = downloader.download_all(force=force)
        console.print("\n[bold]Download Results:[/bold]")
        for b, success in results.items():
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            console.print(f"  {status} {b}")
        return

    try:
        bin_dir = downloader.download_runner(backend, force=force)
        console.print(f"\n[green]Runner installed: {bin_dir}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1) from e


@ollama_app.command("serve")
def ollama_serve(
    model: str = typer.Argument(..., help="Ollama model name (e.g., gemma3:4b)"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port (0 for auto)"),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Server host (use 0.0.0.0 for LAN access)"
    ),
    ctx_size: int = typer.Option(32768, "-c", "--ctx-size", help="Context size (0=auto)"),
    device: str = typer.Option("auto", "-d", "--device", help="Device: auto, gpu0, gpu1, cpu"),
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="Backend: auto, cuda, rocm, vulkan, metal, mlx, mps, cpu"
    ),
    runner_type: str = typer.Option(
        "ollama", "--runner", help="Runner type: ollama (Ollama patched) or official (llama.cpp)"
    ),
    runner_verbose: bool = typer.Option(
        False, "--runner-verbose", help="Enable verbose output from llama.cpp runner"
    ),
    fit_mode: str = typer.Option(
        "auto",
        "--fit",
        help="Parameter fitting mode: auto, on, off (use 'off' to disable auto-tuning)",
    ),
    auto_port: bool = typer.Option(
        False, "-a", "--auto-port", help="Auto-find available port if default is in use"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show detailed monitoring in terminal"
    ),
    web_monitor: bool = typer.Option(False, "-w", "--web", help="Enable web monitoring page"),
    skip_check: bool = typer.Option(False, "--skip-check", help="Skip compatibility check"),
    kv_cache: str = typer.Option(
        "auto",
        "--kv-cache",
        help="KV cache quantization: auto, f16, q8_0, q4_0, tq4, tq3.5, tq3, tq2.5, tq2",
    ),
    cpu_offload: int = typer.Option(
        0, "--cpu-offload", help="Number of layers to offload to CPU (0=auto)"
    ),
    cpu_moe: bool = typer.Option(
        False,
        "--cpu-moe",
        help="Offload MoE experts to CPU, keep attention on GPU (7-8x speedup for MoE models)",
    ),
    prompt_offload: bool = typer.Option(
        False, "--prompt-offload", help="Prompt for CPU offload if needed"
    ),
    rope_scaling: str = typer.Option(
        "none", "--rope-scaling", help="RoPE scaling: none, linear, yarn (for extending context)"
    ),
    rope_scale: float = typer.Option(
        1.0, "--rope-scale", help="RoPE context scaling factor (e.g., 2.0 for 2x context)"
    ),
    threads: int = typer.Option(0, "-t", "--threads", help="Number of threads (-1=auto)"),
    batch_size: int = typer.Option(2048, "--batch-size", help="Batch size for prompt processing"),
    ubatch_size: int = typer.Option(512, "--ubatch-size", help="Physical batch size"),
    flash_attn: bool = typer.Option(
        True, "--flash-attn/--no-flash-attn", help="Enable flash attention"
    ),
    lookahead: int = typer.Option(
        0, "--lookahead", help="Lookahead decoding steps (0=disabled, 2-4 recommended)"
    ),
    cache_prompts: bool = typer.Option(
        False, "--cache-prompts", help="Enable prompt caching for repeated system prompts"
    ),
    slots: int = typer.Option(
        1, "--slots", help="Number of parallel slots for concurrent requests"
    ),
    cont_batching: bool = typer.Option(
        True, "--cont-batching/--no-cont-batching", help="Enable continuous batching"
    ),
    mlock: bool = typer.Option(False, "--mlock", help="Lock model in RAM to prevent swapping"),
    no_kv_offload: bool = typer.Option(
        False, "--no-kv-offload", help="Disable KV cache offloading to CPU (force GPU)"
    ),
    speculative_draft: Optional[str] = typer.Option(
        None, "--draft", help="Draft model path for speculative decoding"
    ),
    speculative_max: int = typer.Option(
        5, "--draft-max", help="Max draft tokens (speculative decoding)"
    ),
    speculative_pmin: float = typer.Option(
        0.75, "--draft-p-min", help="Min acceptance probability for draft tokens"
    ),
):
    """Serve an Ollama model with OpenAI-compatible API.

    Host Binding:
    - 127.0.0.1 (default): Local access only
    - 0.0.0.0: Allow LAN access (all network interfaces)

    Backends:
    - cuda: NVIDIA GPUs (fastest)
    - rocm: AMD GPUs with ROCm
    - vulkan: Cross-platform GPU (works with AMD/NVIDIA/Intel)
    - metal: Apple Silicon (macOS)
    - mlx: Apple MLX backend (macOS, optimized for Apple Silicon)
    - mps: Apple Metal Performance Shaders (macOS)
    - cpu: CPU only

    Runner Types:
    - ollama: Ollama patched llama.cpp (supports gemma4 etc.)
    - official: Official llama.cpp (may have better compatibility)

    Debug Options:
    - --runner-verbose: Enable detailed llama.cpp logs
    - --fit off: Disable parameter auto-tuning (useful for debugging)

    KV Cache Quantization:
    - auto: Automatically choose based on available memory
    - q8_0: 8-bit (high quality)
    - q4_0: 4-bit (balanced)
    - tq3: TurboQuant 3-bit (recommended, 5.3x compression)
    - tq2: TurboQuant 2-bit (extreme, 8x compression)

    CPU Offload:
    - 0: Auto-detect if needed (default, uses full GPU if possible)
    - N: Offload N layers to CPU, rest to GPU
    - --cpu-moe: Offload MoE experts to CPU, keep attention on GPU (7-8x speedup)
    - Use --prompt-offload to be asked before offloading

    Speed Optimization:
    - --draft MODEL: Speculative decoding with draft model (2-4x speedup)
    - --lookahead N: Lookahead decoding without extra model (1.5-2x speedup)
    - --cache-prompts: Cache repeated system prompts
    - --cont-batching: Enable continuous batching for concurrent requests
    - --slots N: Number of parallel request slots
    - --mlock: Lock model in RAM to prevent swapping
    - --no-kv-offload: Force KV cache to stay on GPU

    Context Extension:
    - --rope-scaling TYPE: RoPE scaling (linear/yarn)
    - --rope-scale N: Context scaling factor (2.0 = 2x context)

    Performance Tips:
    - Default context is 32K (optimized for speed)
    - Use --flash-attn for faster attention (enabled by default)
    - Increase --batch-size for better throughput
    - Use -c 65536 or higher for long documents

    Examples:
        moxing ollama serve gemma3:4b
        moxing ollama serve gemma3:4b -b cuda
        moxing ollama serve gemma4:31b --runner official
        moxing ollama serve gemma4:31b --runner-verbose --fit off  # Debug mode
        moxing ollama serve omnicoder-9b --kv-cache q4_0
        moxing ollama serve omnicoder-9b --cpu-offload 10
        moxing ollama serve qwen3:30b-a3b --cpu-moe  # MoE offloading
        moxing ollama serve model --prompt-offload
        moxing ollama serve gemma3:1b -c 65536 --kv-cache q4_0
        moxing ollama serve omnicoder-9b -v    # Verbose monitoring
        moxing ollama serve omnicoder-9b -w    # Web monitoring
        moxing ollama serve omnicoder-9b --host 0.0.0.0  # LAN access
        moxing ollama serve model --lookahead 3  # Lookahead decoding
        moxing ollama serve model --cache-prompts  # Prompt caching
        moxing ollama serve model --draft small.gguf  # Speculative decoding
    """
    ollama_serve_impl(
        model,
        port,
        host,
        ctx_size,
        device,
        backend,
        auto_port,
        verbose,
        web_monitor,
        skip_check,
        kv_cache,
        cpu_offload,
        cpu_moe,
        prompt_offload,
        rope_scaling,
        rope_scale,
        threads,
        batch_size,
        ubatch_size,
        flash_attn,
        runner_type,
        runner_verbose,
        fit_mode,
        lookahead,
        cache_prompts,
        slots,
        cont_batching,
        mlock,
        no_kv_offload,
        speculative_draft,
        speculative_max,
        speculative_pmin,
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
    from moxing.ollama_runner import OllamaModelResolver
    from moxing.warmup_benchmark import ProfileCache, WarmupBenchmark, get_hardware_fingerprint

    console.print(f"[blue]Looking up model: {model}[/blue]")

    resolver = OllamaModelResolver()
    model_path = resolver.resolve(model)

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
    cpu_moe: bool = False,
    prompt_offload: bool = False,
    rope_scaling: str = "none",
    rope_scale: float = 1.0,
    threads: int = 0,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    flash_attn: bool = True,
    runner_type: str = "ollama",
    runner_verbose: bool = False,
    fit_mode: str = "auto",
    lookahead: int = 0,
    cache_prompts: bool = False,
    slots: int = 1,
    cont_batching: bool = True,
    mlock: bool = False,
    no_kv_offload: bool = False,
    speculative_draft: Optional[str] = None,
    speculative_max: int = 5,
    speculative_pmin: float = 0.75,
):
    """Serve an Ollama model using moxing's Ollama runner.

    Uses moxing's compiled Ollama runner binaries which support:
    - Direct device selection (-d gpu0, gpu1, etc.)
    - All backends (CUDA, ROCm, Vulkan, CPU)
    - Ollama-specific models (gemma4, etc.)
    - Runner type selection (--runner ollama or official)
    - Debug options (--runner-verbose, --fit off)

    For direct GGUF files, use 'moxing serve <gguf_file>' instead.
    """
    from moxing.ollama_runner import OllamaModelResolver, serve_ollama_model

    console.print(f"[blue]Looking up model: {model}[/blue]")

    resolver = OllamaModelResolver()
    model_path = resolver.resolve(model)

    if not model_path:
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("\n[yellow]Available Ollama models:[/yellow]")
        models = resolver.list_models()
        for m in models[:10]:
            console.print(f"  • {m['full_name']} ({m['size_gb']:.1f}GB)")
        if len(models) > 10:
            console.print(f"  ... and {len(models) - 10} more")
        console.print("\n[dim]Run 'moxing ollama list' to see all models[/dim]")
        console.print("[dim]To download: ollama pull {model}[/dim]")
        raise typer.Exit(1)

    file_size_gb = model_path.stat().st_size / (1024**3)
    console.print(f"[green]Found model: {model} ({file_size_gb:.1f} GB)[/green]")
    console.print(f"[dim]Path: {model_path}[/dim]")

    if not cpu_moe:
        try:
            from moxing.gguf_metadata import extract_model_architecture

            arch = extract_model_architecture(model_path)
            if arch.is_moe and arch.expert_count > 0:
                console.print(
                    f"[yellow]MoE model detected: {arch.expert_count} experts, "
                    f"{arch.expert_used_count} active[/yellow]"
                )
                console.print(
                    "[yellow]Tip: Use --cpu-moe for 7-8x speedup on constrained hardware[/yellow]"
                )
        except Exception:
            pass

    if backend == "auto":
        from moxing.device import DeviceDetector

        detector = DeviceDetector()
        detector.detect()
        best = detector.get_best_device(file_size_gb)
        backend = best.backend.value
        console.print(f"[blue]Auto-selected backend: {backend}[/blue]")

    server = serve_ollama_model(
        model_name=model,
        backend=backend,
        device=device,
        port=port,
        host=host,
        ctx_size=ctx_size,
        verbose=verbose,
        runner_type=runner_type,
        verbose_runner=runner_verbose,
        fit_mode=fit_mode,
        n_gpu_layers=-1 if cpu_offload == 0 and not cpu_moe else 999 - cpu_offload,
        threads=threads,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        flash_attn=flash_attn,
        kv_cache=kv_cache,
        cpu_moe=cpu_moe,
        lookahead=lookahead,
        cache_prompts=cache_prompts,
        slots=slots,
        cont_batching=cont_batching,
        mlock=mlock,
        no_kv_offload=no_kv_offload,
        speculative_draft=speculative_draft,
        speculative_max=speculative_max,
        speculative_pmin=speculative_pmin,
    )

    if not server:
        console.print("[red]Failed to start server[/red]")
        raise typer.Exit(1)

    try:
        if verbose:
            console.print("\n[blue]Server running. Press Ctrl+C to stop.[/blue]")

        while server.is_running():
            time.sleep(1)

        console.print("[red]服务进程已退出[/red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping server...[/yellow]")
    finally:
        server.stop()


def _fast_check_gguf_compatibility(gguf_path: Path):
    """Quick check GGUF compatibility by reading metadata only.

    Returns:
        Tuple of (is_compatible, error_message)

    Also checks for Ollama-specific architectures that require Ollama backend.
    """

    OLLAMA_ONLY_ARCHITECTURES = ["gemma4", "gemma4.it"]

    try:
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return False, "Not a valid GGUF file"

            version = struct.unpack("<I", f.read(4))[0]
            if version < 2 or version > 4:
                return False, f"Unsupported GGUF version: {version}"

            struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            architecture = None

            for _ in range(min(metadata_kv_count, 100)):
                try:
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    if key_len > 10000:
                        return True, None
                    key = f.read(key_len).decode("utf-8")
                    value_type = struct.unpack("<I", f.read(4))[0]

                    if key == "general.architecture" and value_type == 8:
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        architecture = f.read(str_len).decode("utf-8")
                        continue

                    for ollama_arch in OLLAMA_ONLY_ARCHITECTURES:
                        if key.startswith(f"{ollama_arch}."):
                            architecture = ollama_arch

                    if value_type == 0 or value_type == 1:
                        f.read(1)
                    elif value_type == 2 or value_type == 3:
                        f.read(2)
                    elif value_type == 4 or value_type == 5 or value_type == 6:
                        f.read(4)
                    elif value_type == 7:
                        f.read(1)
                    elif value_type == 8:
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        if str_len < 1000000:
                            f.read(str_len)
                        else:
                            return True, None
                    elif value_type == 9:
                        arr_type = struct.unpack("<I", f.read(4))[0]
                        arr_len = struct.unpack("<Q", f.read(8))[0]
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
                                slen = struct.unpack("<Q", f.read(8))[0]
                                if slen < 100000:
                                    f.read(slen)
                                else:
                                    break
                except struct.error:
                    return True, None

            if architecture:
                for ollama_arch in OLLAMA_ONLY_ARCHITECTURES:
                    if architecture.startswith(ollama_arch) or architecture == ollama_arch:
                        return (
                            False,
                            f"Architecture '{architecture}' requires Ollama backend "
                            f"(llama.cpp doesn't support it)",
                        )

            return True, None

    except Exception as e:
        return False, str(e)


def _get_compatible_gguf(
    model: str, ollama_gguf: Path, size_gb: float, verbose: bool = False
) -> Optional[Path]:
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
        cmd = [
            str(llama_cli),
            "-m",
            str(ollama_gguf),
            "-n",
            "1",
            "-p",
            "x",
            "-c",
            "128",
            "--no-display-prompt",
            "-ngl",
            "0",
            "--no-conversation",
            "-st",
        ]
        if verbose:
            console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
            console.print("[dim]Running subprocess.run with timeout=120s...[/dim]")

        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            cwd=str(llama_cli.parent),
            stdin=subprocess.DEVNULL,
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
                console.print("[dim]stderr output (first 15 lines):[/dim]")
                for line in stderr.strip().split("\n")[:15]:
                    console.print(f"[dim]  {line}[/dim]")
            else:
                console.print("[dim]No stderr output[/dim]")

        if stderr:
            stderr_lower = stderr.lower()
            is_fatal = any(p in stderr_lower for p in fatal_patterns)
            is_incompatible = any(p in stderr_lower for p in incompatible_patterns)

            if is_fatal:
                console.print("[red]Error: Model architecture not supported by llama.cpp[/red]")
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
                console.print(
                    "[yellow]Warning: GGUF format may not be fully compatible "
                    "with llama.cpp[/yellow]"
                )
                console.print(
                    "[dim]The model may still work, but some features might be limited.[/dim]"
                )
                console.print("[dim]Using the Ollama GGUF file directly...[/dim]")
                return ollama_gguf

            console.print("[green]GGUF is compatible[/green]")
            return ollama_gguf
        else:
            if returncode == 0:
                console.print("[green]GGUF is compatible[/green]")
                return ollama_gguf
            else:
                console.print(f"[yellow]Warning: llama-cli exited with code {returncode}[/yellow]")
                if stdout:
                    console.print(f"[dim]stdout: {stdout[:300]}[/dim]")
                console.print("[dim]Proceeding with the GGUF file anyway...[/dim]")
                return ollama_gguf

    except subprocess.TimeoutExpired:
        console.print("[yellow]Warning: Model loading timed out (120s)[/yellow]")
        console.print(
            "[dim]The model might be too large or incompatible. Proceeding anyway...[/dim]"
        )
        return ollama_gguf
    except Exception as e:
        console.print(f"[yellow]Compatibility check error: {e}[/yellow]")
        import traceback

        if verbose:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        console.print("[dim]Proceeding with the GGUF file anyway...[/dim]")
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
        if model_lower == key_lower:
            return hf_repo
        if model_lower.startswith(key_lower.replace(":", "-") + "-") or model_lower.startswith(
            key_lower + ":"
        ):
            return hf_repo
        model_without_owner = model_lower.split("/")[-1] if "/" in model_lower else model_lower
        model_base = model_without_owner.split(":")[0].split("-")[0]
        if model_base == key_lower.split(":")[0]:
            return hf_repo

    return None
