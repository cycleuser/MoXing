import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def tune(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    force: bool = typer.Option(False, "-f", "--force", help="Force re-tuning even if cache exists"),
):
    """Run warmup benchmark to find optimal parameters for a model.

    Measures actual performance to find the best configuration:
    - Optimal context size
    - Best ubatch size
    - MoE offloading strategy
    - KV cache type selection

    Results are cached for 30 days. Second launch skips warmup (2s startup).

    Examples:
        moxing tune ./model.gguf
        moxing tune ./model.gguf --force
    """
    from pathlib import Path

    from moxing.binaries import get_binary_manager
    from moxing.device import DeviceDetector
    from moxing.gguf_metadata import extract_model_architecture
    from moxing.warmup_benchmark import ProfileCache, WarmupBenchmark, get_hardware_fingerprint

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Analyzing model: {model_path.name}[/blue]")

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
    model_id = model_path.stem
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
            console.print(f"\n[yellow]To re-tune, use: moxing tune {model} --force[/yellow]")
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
            console.print(f"[yellow]Use: moxing serve {model} --cpu-moe[/yellow]")

        console.print("\n[dim]Profile cached for 30 days[/dim]")
    else:
        console.print("[yellow]Benchmark failed, using default configuration[/yellow]")


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
        from moxing.turboquant import benchmark_turboquant_comprehensive

        benchmark_turboquant_comprehensive()
        return

    print_cache_analysis(model_size, ctx_size, vram)


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

    gpu_layers = (
        config.n_gpu_layers
        if config.n_gpu_layers >= 0
        else 'all'
    )
    console.print(
        Panel(
            f"[green]Model:[/green] {config.model}\n"
            f"[blue]Backend:[/blue] {config.backend}\n"
            f"[yellow]Device:[/yellow] {config.device}\n"
            f"[magenta]GPU Layers:[/magenta] {gpu_layers}\n"
            f"[cyan]Recommended Context:[/cyan] "
            f"{config.ctx_size}\n",
            title="Recommended Configuration",
        )
    )


def benchmark(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    prompt: str = typer.Option(
        "standard",
        "-p",
        "--prompt",
        help="Prompt type: quick, standard, code, creative, or custom text",
    ),
    n_tokens: int = typer.Option(128, "-n", "--tokens", help="Number of tokens to generate"),
    n_runs: int = typer.Option(1, "-r", "--runs", help="Number of benchmark runs"),
    warmup: bool = typer.Option(True, "-w", "--warmup", help="Run warmup iteration"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    compare: Optional[str] = typer.Option(None, "--compare", help="Second model to compare"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Benchmark model performance (tokens/second, memory usage)."""
    from moxing.benchmark import BenchmarkRunner

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    model_size_gb = model_path.stat().st_size / (1024**3)

    console.print(
        Panel(
            f"[cyan]Model:[/cyan] {model_path.name}\n"
            f"[cyan]Size:[/cyan] {model_size_gb:.2f} GB\n"
            f"[cyan]Tokens:[/cyan] {n_tokens}\n"
            f"[cyan]Runs:[/cyan] {n_runs}",
            title="Benchmark Configuration",
        )
    )

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
            console.print(
                f"\n[bold]Benchmarking model {i + 1}/{len(models_to_bench)}: {Path(m).name}[/bold]"
            )

        try:
            result = runner.run(
                model=m,
                prompt=prompt,
                n_tokens=n_tokens,
                n_runs=n_runs,
                warmup=warmup,
                ctx_size=ctx_size,
            )
            results.append(result)

            if not json_output and len(models_to_bench) == 1:
                runner.print_results(result)

        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            raise typer.Exit(1) from e

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


def speed_test(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    prompt: str = typer.Option("Hello, how are you?", "-p", "--prompt", help="Test prompt"),
    ctx_size: int = typer.Option(2048, "-c", "--ctx-size", help="Context size"),
):
    """Quick speed test with detailed output similar to ollama."""
    from moxing import Client, LlamaServer
    from moxing.device import DeviceDetector

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    detector = DeviceDetector()
    detector.detect()
    device_config = detector.get_best_device(model_path.stat().st_size / (1024**3))

    model_size_gb = model_path.stat().st_size / (1024**3)

    console.print()
    console.print(f"[bold cyan]Model:[/bold cyan] {model_path.name}")
    console.print(f"[bold cyan]Size:[/bold cyan] {model_size_gb:.2f} GB")
    backend_str = device_config.backend.value
    console.print(
        f"[bold cyan]Device:[/bold cyan] {device_config.device.name} ({backend_str})"
    )
    console.print()

    port = 8080 + hash(str(model_path)) % 1000

    server = LlamaServer(
        model=str(model_path),
        port=port,
        ctx_size=ctx_size,
        n_gpu_layers=device_config.n_gpu_layers,
        device=f"{device_config.backend.value.capitalize()}{device_config.device.index}",
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
            stream=True,
        )

        generated_text = ""
        first_token_time = None
        token_count = 0

        if hasattr(response, '__iter__'):
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

        tokens_per_second = token_count / total_time if token_count > 0 and total_time > 0 else 0

        console.print()
        console.print()
        console.print(
            Panel(
                f"[green]Total tokens:[/green] {token_count}\n"
                f"[green]Time:[/green] {total_time:.2f}s\n"
                f"[green]Speed:[/green] {tokens_per_second:.2f} tokens/s\n"
                f"[green]Time to first token:[/green] {first_token_time - prompt_start:.2f}s"
                if first_token_time
                else "",
                title="Performance",
            )
        )

    finally:
        server.stop()


def model_info(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
):
    """Show detailed model information and estimated performance."""
    from moxing.benchmark import estimate_speed
    from moxing.device import DeviceDetector

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    model_size_gb = model_path.stat().st_size / (1024**3)

    detector = DeviceDetector()
    devices = detector.detect()
    device_config = detector.get_best_device(model_size_gb)

    console.print(
        Panel(
            f"[cyan]File:[/cyan] {model_path.name}\n"
            f"[cyan]Path:[/cyan] {model_path}\n"
            f"[cyan]Size:[/cyan] {model_size_gb:.2f} GB",
            title="Model Information",
        )
    )

    table = Table(title="Recommended Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Notes", style="yellow")

    table.add_row("Backend", device_config.backend.value, "Best available for your hardware")
    table.add_row("Device", device_config.device.name, "")
    table.add_row(
        "GPU Layers",
        str(device_config.n_gpu_layers) if device_config.n_gpu_layers >= 0 else "all",
        "",
    )
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
                    f"~{est['estimated_tokens_per_second']} t/s ({est['mode']})",
                )

        console.print(gpu_table)


def check_model(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Suggest fixes for compatibility issues"),
):
    """Check GGUF model compatibility with llama.cpp."""
    from moxing.gguf_check import diagnose_gguf, get_model_suggestions, print_diagnosis

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    if not model.endswith(".gguf"):
        console.print("[yellow]Warning: File does not have .gguf extension[/yellow]")

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
        raise typer.Exit(1) from e


def register(app: typer.Typer):
    app.command()(tune)
    app.command("cache")(cache_analysis)
    app.command()(config)
    app.command("bench")(benchmark)
    app.command("speed")(speed_test)
    app.command("info")(model_info)
    app.command("check")(check_model)
