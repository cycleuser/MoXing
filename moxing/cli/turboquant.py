import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

turboquant_app = typer.Typer(name="turboquant", help="TurboQuant KV cache compression commands")


@turboquant_app.command("info")
def turboquant_info():
    """Show TurboQuant information and recommendations.

    TurboQuant (arXiv:2504.19874) is a near-optimal KV cache quantization algorithm.
    """
    console.print(
        Panel(
            "[bold cyan]TurboQuant: Near-Optimal KV Cache Quantization[/bold cyan]\n\n"
            "Based on Google's paper: arXiv:2504.19874\n\n"
            "[bold]Key Features:[/bold]\n"
            "• Data-oblivious online quantization\n"
            "• Near-optimal distortion (within 2.7x of theoretical optimum)\n"
            "• Unbiased inner product estimation\n"
            "• Mixed precision support (3.5 bits, 2.5 bits)",
            title="TurboQuant Overview",
        )
    )

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

    console.print(
        Panel(
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
            title="Usage Examples",
        )
    )


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
    from moxing.kv_cache import KVCacheQuantType, estimate_kv_cache_size_gb, get_model_kv_params

    if n_layers == 0 or n_heads == 0 or head_dim == 0:
        n_layers, n_heads, head_dim = get_model_kv_params(model_size)

    console.print(
        Panel(
            f"[cyan]Model Size:[/cyan] {model_size:.1f} GB\n"
            f"[cyan]Context:[/cyan] {ctx_size:,}\n"
            f"[cyan]Layers:[/cyan] {n_layers}\n"
            f"[cyan]Heads:[/cyan] {n_heads}\n"
            f"[cyan]Head Dim:[/cyan] {head_dim}",
            title="Model Configuration",
        )
    )

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

        table.add_row(quant.value, f"{quant.bits:.1f}", f"{size:.2f} GB", vs_f16, quality)

    console.print(table)

    console.print(
        Panel(
            "[bold green]Recommended:[/bold green]\n\n"
            "• Quality first: [cyan]--kv-cache tq3.5[/cyan]\n"
            "• Memory efficient: [cyan]--kv-cache tq2.5[/cyan]\n"
            "• Built-in q4_0: [cyan]--kv-cache q4_0[/cyan] (recommended)\n\n"
            "[dim]Note: TurboQuant types (tq*) use llama.cpp's nearest equivalent.[/dim]",
            title="Recommendations",
        )
    )


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

    console.print(
        Panel(
            f"[cyan]Configuration:[/cyan]\n"
            f"  Dimension: {dim}\n"
            f"  Vectors: {n_vectors}\n"
            f"  Bits: {bits}\n\n"
            f"[green]Results:[/green]\n"
            f"  MSE: {mse:.6f}\n"
            f"  Inner Product Bias: {mean_bias:.6f} (ideal: 0)\n"
            f"  Quantize Time: {quant_time * 1000:.2f} ms\n"
            f"  Dequantize Time: {dequant_time * 1000:.2f} ms\n"
            f"  Compression: {16 / bits:.1f}x\n\n"
            f"[magenta]Unbiased: {'✓ Yes' if abs(mean_bias) < 0.05 else '✗ No'}[/magenta]",
            title="TurboQuant Test Results",
        )
    )
