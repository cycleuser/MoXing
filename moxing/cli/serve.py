import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def serve(
    model: str = typer.Argument(
        ..., help="Model name, path to GGUF file, HuggingFace repo, or ollama:model"
    ),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization type"),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Server host (use 0.0.0.0 for LAN access)"
    ),
    port: int = typer.Option(8080, "-p", "--port", help="Server port (0 for auto)"),
    ctx_size: int = typer.Option(
        0, "-c", "--ctx-size", help="Context size (0=auto-detect based on VRAM)"
    ),
    source: str = typer.Option(
        "modelscope",
        "-s",
        "--source",
        help="Model source (huggingface/modelscope, default: modelscope)",
    ),
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="Backend: auto, vulkan, cuda, rocm, metal, cpu"
    ),
    device: str = typer.Option(
        "auto",
        "-d",
        "--device",
        help="Device: auto, gpu0, gpu1, cpu (use 'moxing devices' to list)",
    ),
    runner: str = typer.Option(
        "auto", "-r", "--runner", help="Runner engine: auto, llama_cpp, vllm, ollama"
    ),
    auto: bool = typer.Option(True, "--auto/--no-auto", help="Auto-detect best device"),
    auto_port: bool = typer.Option(False, "-a", "--auto-port", help="Auto-find available port"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show detailed monitoring in terminal"
    ),
    web_monitor: bool = typer.Option(False, "-w", "--web", help="Enable web monitoring page"),
    force: bool = typer.Option(
        False, "-f", "--force", help="Force use specified backend without compatibility check"
    ),
    kv_cache: str = typer.Option(
        "auto",
        "--kv-cache",
        help="KV cache quantization: auto, f16, q8_0, q5_0, q4_0, tq4, tq3.5, tq3, tq2.5, tq2",
    ),
    cpu_offload: int = typer.Option(
        0, "--cpu-offload", help="Number of layers to offload to CPU (0=auto)"
    ),
    ngl: int = typer.Option(
        -1, "--ngl", "--gpu-layers", help="Number of GPU layers (-1=auto, 0=CPU only, 999=all GPU)"
    ),
    cpu_moe: bool = typer.Option(
        False,
        "--cpu-moe",
        help="Offload MoE experts to CPU, keep attention on GPU (7-8x speedup for MoE models)",
    ),
    analyze_cache: bool = typer.Option(
        False, "--analyze-cache", help="Show KV cache analysis and exit"
    ),
    speculative_draft: Optional[str] = typer.Option(
        None, "--draft", help="Draft model path for speculative decoding (MTP: same model path)"
    ),
    speculative_type: str = typer.Option(
        "", "--spec-type", help="Speculative type: draft-simple, draft-eagle3, draft-mtp, ngram-*"
    ),
    speculative_max: int = typer.Option(
        5, "--draft-max", help="Max draft tokens (speculative decoding)"
    ),
    speculative_min: int = typer.Option(0, "--draft-min", help="Min draft tokens (0=auto)"),
    speculative_pmin: float = typer.Option(
        0.75, "--draft-p-min", help="Min acceptance probability for draft tokens"
    ),
    lookahead: int = typer.Option(
        0, "--lookahead", help="Lookahead decoding steps (0=disabled, 2-4 recommended)"
    ),
    cache_prompts: bool = typer.Option(
        False, "--cache-prompts", help="Enable prompt caching for repeated system prompts"
    ),
    cache_rem: str = typer.Option(
        "lru", "--cache-rem", help="Cache removal policy: lru, lru-sc, fifo"
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
    tensor_split: Optional[str] = typer.Option(
        None, "--tensor-split", help="GPU memory split ratios (e.g., '50,50' for 2 GPUs)"
    ),
    main_gpu: int = typer.Option(
        0, "--main-gpu", help="Main GPU index for tensor parallelism (0-based)"
    ),
    numa: Optional[str] = typer.Option(
        None, "--numa", help="NUMA policy: none, distribute, isolate, numactl"
    ),
    defrag_thold: float = typer.Option(
        0.1, "--defrag-thold", help="Memory defragmentation threshold (0=disabled)"
    ),
    rope_scaling: str = typer.Option(
        "none", "--rope-scaling", help="RoPE scaling: none, linear, yarn"
    ),
    rope_scale: float = typer.Option(
        1.0, "--rope-scale", help="RoPE context scaling factor (e.g., 2.0 for 2x context)"
    ),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel sequences"),
    mirostat: int = typer.Option(
        0, "--mirostat", help="Mirostat sampling: 0=disabled, 1=mirostat, 2=mirostat 2.0"
    ),
    mirostat_tau: float = typer.Option(5.0, "--mirostat-tau", help="Mirostat target perplexity"),
    mirostat_eta: float = typer.Option(0.1, "--mirostat-eta", help="Mirostat learning rate"),
    tensor_parallel_size: int = typer.Option(
        1, "--tp", "--tensor-parallel-size", help="Tensor parallelism (vLLM: number of GPUs)"
    ),
    gpu_memory_utilization: float = typer.Option(
        0.9, "--gpu-mem-util", help="GPU memory utilization ratio (vLLM: 0.0-1.0)"
    ),
    max_model_len: int = typer.Option(
        0, "--max-model-len", help="Maximum model context length (vLLM)"
    ),
    dtype: str = typer.Option(
        "auto", "--dtype", help="Data type: auto, float16, bfloat16, float32 (vLLM)"
    ),
    quantization: Optional[str] = typer.Option(
        None, "-qz", "--quantization", help="Quantization method: awq, gptq, fp8, gguf (vLLM)"
    ),
    enable_prefix_caching: bool = typer.Option(
        False, "--prefix-cache", help="Enable prefix caching (vLLM)"
    ),
    enforce_eager: bool = typer.Option(
        False, "--eager", help="Disable CUDA graph optimization (vLLM)"
    ),
    attention_backend: str = typer.Option(
        "auto",
        "--attn-backend",
        help="Attention backend: auto, flash_attn, flashinfer, triton (vLLM)",
    ),
):
    """Start the LLM server with automatic configuration.

    Runner Engines:
    - auto: Auto-detect best runner (llama_cpp for GGUF, vllm for HF repos)
    - llama_cpp: llama.cpp server (GGUF models, all backends)
    - vllm: vLLM engine (HuggingFace/GGUF, CUDA/ROCm only, higher throughput)
    - ollama: Ollama runner (Ollama model format)

    Host Binding:
    - 127.0.0.1 (default): Local access only
    - 0.0.0.0: Allow LAN access (all network interfaces)

    KV Cache Quantization (llama_cpp):
    - auto: Automatically choose based on available memory
    - f16: Full precision (16-bit)
    - q8_0: 8-bit quantization (high quality)
    - q5_0: 5-bit quantization (good quality)
    - q4_0: 4-bit quantization (balanced, recommended)

    TurboQuant (Google arXiv:2504.19874, llama_cpp):
    - tq4: 4-bit TurboQuant (high quality)
    - tq3.5: 3.5-bit mixed precision (quality neutral)
    - tq3: 3-bit TurboQuant (good quality)
    - tq2.5: 2.5-bit mixed precision (slight loss)
    - tq2: 2-bit TurboQuant (maximum compression)

    vLLM Options:
    - --tp N: Tensor parallelism across N GPUs
    - --gpu-mem-util R: GPU memory utilization (0.0-1.0)
    - --max-model-len N: Maximum context length
    - --dtype TYPE: Data type (auto/float16/bfloat16/float32)
    - --quantization METHOD: Quantization (awq/gptq/fp8/gguf)
    - --prefix-cache: Enable prefix caching
    - --attn-backend BACKEND: Attention backend

    Memory Optimization:
    - --cpu-offload N: Offload N layers to CPU RAM
    - --cpu-moe: Offload MoE experts to CPU, keep attention on GPU (7-8x speedup)
    - --mlock: Lock model in RAM to prevent swapping
    - --no-kv-offload: Force KV cache to stay on GPU

    Speed Optimization:
    - --draft MODEL: Speculative decoding with draft model (2-4x speedup)
    - --lookahead N: Lookahead decoding without extra model (1.5-2x speedup)
    - --cache-prompts: Cache repeated system prompts
    - --cont-batching: Enable continuous batching for concurrent requests
    - --slots N: Number of parallel request slots

    Multi-GPU:
    - --tensor-split R: Split model across GPUs (llama_cpp, e.g., '50,50')
    - --tp N: Tensor parallelism (vLLM, number of GPUs)
    - --main-gpu N: Set main GPU index

    Context Extension:
    - --rope-scaling TYPE: RoPE scaling (linear/yarn)
    - --rope-scale N: Context scaling factor (2.0 = 2x context)

    Examples:
    - GGUF model: moxing serve model.gguf
    - HuggingFace: moxing serve Qwen/Qwen2.5-7B-Instruct -r vllm
    - vLLM multi-GPU: moxing serve model -r vllm --tp 2
    - Auto KV cache: moxing serve model.gguf --kv-cache q4_0
    - TurboQuant 3.5: moxing serve model.gguf --kv-cache tq3.5
    - CPU offload: moxing serve model.gguf --cpu-offload 10
    - MoE offload: moxing serve model.gguf --cpu-moe
    - Speculative: moxing serve model.gguf --draft small-model.gguf
    - Lookahead: moxing serve model.gguf --lookahead 3
    - Prompt cache: moxing serve model.gguf --cache-prompts
    - Multi-slot: moxing serve model.gguf --slots 4
    - Multi-GPU: moxing serve model.gguf --tensor-split 50,50
    - Extended context: moxing serve model.gguf --rope-scaling yarn --rope-scale 4
    - LAN access: moxing serve model.gguf --host 0.0.0.0
    """
    from moxing.cli.ollama_cmds import ollama_serve_impl
    from moxing.gguf_check import GGUFParser
    from moxing.gguf_compress import is_gguf_compressed, resolve_model_path
    from moxing.mlx_server import MLXServer
    from moxing.server import find_available_port, is_port_in_use

    if port == 0 or auto_port or is_port_in_use(port, host):
        original_port = port if port > 0 else 8080
        port = find_available_port(original_port)
        if port != original_port:
            console.print(f"[yellow]Port {original_port} in use, using port {port}[/yellow]")

    if model.startswith("ollama:"):
        ollama_model = model[7:]
        ollama_serve_impl(
            ollama_model, port, host, ctx_size, device, backend, auto_port, verbose, web_monitor
        )
        return

    if backend == "ollama":
        ollama_serve_impl(
            model, port, host, ctx_size, device, backend, auto_port, verbose, web_monitor
        )
        return

    model_file = Path(model)

    if model_file.exists() and is_gguf_compressed(model_file):
        console.print("[blue]Detected compressed GGUF, decompressing...[/blue]")
        decompressed_path = resolve_model_path(model_file)
        model = str(decompressed_path)
        model_file = decompressed_path
        console.print(f"[green]Decompressed to cache: {decompressed_path.name}[/green]")

    model_path = model_file if model_file.exists() else None

    if analyze_cache and model_path:
        from moxing.kv_cache import print_cache_analysis

        model_size = model_path.stat().st_size / (1024**3)
        print_cache_analysis(model_size, ctx_size if ctx_size > 0 else 4096)
        raise typer.Exit()

    is_gguf = False
    if model_path:
        if model_path.suffix == ".gguf" or str(model).endswith(".gguf"):
            is_gguf = True
        else:
            try:
                import struct

                with open(model_path, "rb") as f:
                    magic = struct.unpack("<I", f.read(4))[0]
                    if magic == 0x46554747:
                        is_gguf = True
            except:  # noqa: E722
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
                        console.print("[yellow]GGUF compatibility issues detected:[/yellow]")
                        for err in meta.errors[:3]:
                            console.print(f"  [red]✗[/red] {err}")

                        console.print(
                            "\n[blue]Switching to MLX backend for better compatibility...[/blue]"
                        )
                        console.print("[dim]Use --force to try llama.cpp anyway[/dim]\n")
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
                console.print("[yellow]Warning: GGUF compatibility issues detected[/yellow]")
                for err in meta.errors[:3]:
                    console.print(f"  [red]✗[/red] {err}")

                if MLXServer.is_available():
                    console.print(
                        f"\n[yellow]Consider using MLX backend: "
                        f"moxing serve {model} -b mlx[/yellow]"
                    )
                console.print("[dim]Use --force to proceed anyway[/dim]\n")
        except Exception:
            pass

    if use_mlx:
        try:
            server: Any = MLXServer(model=model, host=host, port=port)

            model_short = Path(model).name[:20] if Path(model).exists() else model[:20]
            server_title = f"{model_short} | Apple Silicon | MLX"

            console.print(
                Panel(
                    f"[green]Server:[/green] http://{host}:{port}\n"
                    f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
                    f"[magenta]Backend:[/magenta] MLX (Apple Silicon)\n"
                    f"[cyan]Device:[/cyan] Apple GPU\n"
                    f"[yellow]Press Ctrl+C to stop[/yellow]",
                    title=server_title,
                )
            )

            server.start(wait=False)
            while server.is_running():
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            server.stop()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        from moxing.runners.base import RunnerConfig, create_runner, detect_best_runner
        from moxing.runners.vllm import VLLMRunner

        effective_runner = runner
        if effective_runner == "auto":
            effective_runner = detect_best_runner(
                model_path=model_path if model_path else None,
                model_name=model,
            )

        if effective_runner == "vllm" and not VLLMRunner.is_vllm_available():
            console.print("[yellow]vLLM not installed, attempting to install...[/yellow]")
            from moxing.vllm_installer import ensure_vllm

            if not ensure_vllm():
                console.print("[red]vLLM installation failed. Falling back to llama_cpp.[/red]")
                effective_runner = "llama_cpp"

        if effective_runner in ("vllm",):
            from moxing.runners.vllm import VLLMRunner

            runner_config = RunnerConfig(
                model=model,
                runner_type="vllm",
                host=host,
                port=port,
                backend=backend if backend != "auto" else "cuda",
                device=device,
                ctx_size=ctx_size if ctx_size > 0 else 0,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                quantization=quantization,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                attention_backend=attention_backend,
                speculative_draft=speculative_draft,
                speculative_type=speculative_type,
                speculative_max=speculative_max,
                verbose=verbose,
            )

            server = create_runner(runner_config)

            model_short = Path(model).name[:20] if Path(model).exists() else model[:20]
            server_title = f"{model_short} | vLLM | {backend}"

            console.print(
                Panel(
                    f"[green]Server:[/green] http://{host}:{port}\n"
                    f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
                    f"[magenta]Runner:[/magenta] vLLM\n"
                    f"[cyan]Tensor Parallel:[/cyan] {tensor_parallel_size}\n"
                    f"[yellow]Press Ctrl+C to stop[/yellow]",
                    title=server_title,
                )
            )

            server.start(wait=False)

            try:
                serve_with_verbose_monitor(server, verbose=verbose, web_monitor=web_monitor)
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down...[/yellow]")
                server.stop()
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                raise typer.Exit(1) from e

        elif effective_runner == "llama_cpp":
            runner_config = RunnerConfig(
                model=model,
                runner_type="llama_cpp",
                host=host,
                port=port,
                backend=backend,
                device=device,
                ctx_size=ctx_size if ctx_size > 0 else 0,
                n_gpu_layers=ngl,
                kv_cache_quant=kv_cache,
                cpu_offload=cpu_offload > 0,
                cpu_offload_layers=cpu_offload,
                cpu_moe=cpu_moe,
                speculative_draft=speculative_draft,
                speculative_type=speculative_type,
                speculative_max=speculative_max,
                speculative_min=speculative_min,
                speculative_pmin=speculative_pmin,
                lookahead=lookahead,
                cache_prompts=cache_prompts,
                cache_rem=cache_rem,
                slots=slots,
                cont_batching=cont_batching,
                mlock=mlock,
                no_kv_offload=no_kv_offload,
                tensor_split=tensor_split,
                main_gpu=main_gpu,
                numa=numa,
                defrag_thold=defrag_thold,
                rope_scaling=rope_scaling,
                rope_scale=rope_scale,
                parallel=parallel,
                mirostat=mirostat,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                verbose=verbose,
            )

            server = create_runner(runner_config)

            model_short = Path(model).name[:20] if Path(model).exists() else model[:20]
            server_title = f"{model_short} | llama.cpp | {backend}"

            console.print(
                Panel(
                    f"[green]Server:[/green] http://{host}:{port}\n"
                    f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
                    f"[magenta]Runner:[/magenta] llama.cpp\n"
                    f"[cyan]Device:[/cyan] {device}\n"
                    f"[yellow]Press Ctrl+C to stop[/yellow]",
                    title=server_title,
                )
            )

            server.start(wait=False)

            try:
                serve_with_verbose_monitor(server, verbose=verbose, web_monitor=web_monitor)
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down...[/yellow]")
                server.stop()
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                raise typer.Exit(1) from e


def run(
    model: str = typer.Argument(..., help="Model name or path"),
    prompt: str = typer.Option(
        None, "-p", "--prompt", help="Single prompt (leave empty for interactive chat)"
    ),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization"),
    tokens: int = typer.Option(256, "-n", "--tokens", help="Max tokens to generate"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option(
        "modelscope", "-s", "--source", help="Model source (default: modelscope)"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show detailed monitoring and statistics"
    ),
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="Backend: auto, vulkan, cuda, metal, cpu"
    ),
    kv_cache: str = typer.Option("auto", "--kv-cache", help="KV cache quantization"),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Server host (use 0.0.0.0 for LAN access)"
    ),
    lookahead: int = typer.Option(
        0, "--lookahead", help="Lookahead decoding steps (0=disabled, 2-4 recommended)"
    ),
    mlock: bool = typer.Option(False, "--mlock", help="Lock model in RAM to prevent swapping"),
    no_kv_offload: bool = typer.Option(
        False, "--no-kv-offload", help="Disable KV cache offloading to CPU"
    ),
    rope_scaling: str = typer.Option(
        "none", "--rope-scaling", help="RoPE scaling: none, linear, yarn"
    ),
    rope_scale: float = typer.Option(1.0, "--rope-scale", help="RoPE context scaling factor"),
):
    """Run inference with a model (auto-downloads if needed).

    Host Binding:
    - 127.0.0.1 (default): Local access only
    - 0.0.0.0: Allow LAN access (all network interfaces)

    Speed Optimization:
    - --lookahead N: Lookahead decoding (1.5-2x speedup, no extra model needed)
    - --mlock: Lock model in RAM to prevent swapping
    - --no-kv-offload: Force KV cache to stay on GPU

    Context Extension:
    - --rope-scaling TYPE: RoPE scaling (linear/yarn)
    - --rope-scale N: Context scaling factor (2.0 = 2x context)

    Examples:
        moxing run model.gguf                    # Interactive chat
        moxing run model.gguf -p "Hello"         # Single prompt
        moxing run model.gguf -v                 # Verbose monitoring
        moxing run model.gguf -p "Hello" -v      # Single prompt with stats
        moxing run model.gguf --kv-cache tq3.5   # With TurboQuant
        moxing run model.gguf --lookahead 3      # With lookahead decoding
        moxing run model.gguf --host 0.0.0.0     # LAN access
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
            port=port,
            host=host,
            lookahead=lookahead,
            mlock=mlock,
            no_kv_offload=no_kv_offload,
            rope_scaling=rope_scaling,
            rope_scale=rope_scale,
        )

        model_name = Path(model).name if Path(model).exists() else model

        console.print(
            Panel(
                f"[cyan]Model:[/cyan] {model_name}\n"
                f"[cyan]Context:[/cyan] {ctx_size}\n"
                f"[cyan]Backend:[/cyan] {backend}\n"
                f"[cyan]KV Cache:[/cyan] {kv_cache}",
                title="Starting Server",
            )
        )

        server.start(wait=True)

        console.print("[green]Server ready![/green]")

        run_with_verbose_monitor(
            server=server, model_name=model_name, prompt=prompt, max_tokens=tokens, verbose=verbose
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if server:
            server.stop()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1) from e
    finally:
        if server:
            server.stop()


def chat_cmd(
    model: str = typer.Argument(..., help="Model name or path"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option(
        "modelscope", "-s", "--source", help="Model source (default: modelscope)"
    ),
):
    """Interactive chat with a model."""
    from moxing.client import Client
    from moxing.runner import AutoRunner

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
                model="llama", messages=messages, max_tokens=512
            )

            if hasattr(response, 'choices') and response.choices:
                assistant_msg = response.choices[0].get("message", {}).get("content", "")
                console.print(f"[bold green]Assistant[/bold green]: {assistant_msg}")
                messages.append({"role": "assistant", "content": assistant_msg})

        server.stop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


def run_with_verbose_monitor(
    server,
    model_name: str,
    prompt: Optional[str] = None,
    max_tokens: int = 256,
    verbose: bool = False,
):
    """Run inference with verbose monitoring display."""
    import psutil

    from moxing.client import Client

    client = Client(server.base_url)

    def get_stats():
        try:
            return {
                "cpu": psutil.cpu_percent(interval=0.1),
                "ram_gb": psutil.virtual_memory().used / (1024**3),
            }
        except:  # noqa: E722
            return {"cpu": 0, "ram_gb": 0}

    if prompt:
        messages = [{"role": "user", "content": prompt}]

        console.print(f"\n[bold blue]Prompt:[/bold blue] {prompt}\n")

        if verbose:
            stats = get_stats()
            console.print(
                f"[dim]📊 RAM: {stats['ram_gb']:.2f} GB | CPU: {stats['cpu']:.1f}%[/dim]\n"
            )

        console.print("[bold green]Response:[/bold green]")

        start_time = time.time()
        first_token_time = None
        token_count = 0

        try:
            response = client.chat.completions.create(
                model="llama", messages=messages, max_tokens=max_tokens, stream=True
            )

            full_response = ""
            if hasattr(response, '__iter__'):
                for chunk in response:
                    if isinstance(chunk, dict) and chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
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
            console.print(
                Panel(
                    f"[bold cyan]📊 Performance[/bold cyan]\n\n"
                    f"[green]Tokens:[/green] {token_count} | "
                    f"[green]Time:[/green] {total_time:.2f}s | "
                    f"[green]Speed:[/green] {speed:.1f} tok/s | "
                    f"[green]TTFT:[/green] {ttft:.2f}s\n\n"
                    f"[yellow]Memory:[/yellow] RAM: {stats['ram_gb']:.2f} GB\n"
                    f"[blue]CPU:[/blue] {stats['cpu']:.1f}%",
                    title="Summary",
                )
            )

        return full_response
    else:
        messages = []

        while True:
            try:
                if verbose:
                    stats = get_stats()
                    console.print(
                        f"[dim]📊 RAM: {stats['ram_gb']:.2f} GB | CPU: {stats['cpu']:.1f}%[/dim]"
                    )

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
                        model="llama", messages=messages, max_tokens=max_tokens, stream=True
                    )

                    assistant_msg = ""
                    if hasattr(response, '__iter__'):
                        for chunk in response:
                            if isinstance(chunk, dict) and chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                content = (
                                    delta.get("content")
                                    or delta.get("reasoning_content")
                                    or ""
                                )
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

                    console.print(
                        f"[dim]  {token_count} tokens | {total_time:.2f}s | "
                        f"{speed:.1f} tok/s | TTFT: {ttft:.2f}s[/dim]"
                    )

                messages.append({"role": "assistant", "content": assistant_msg})

            except KeyboardInterrupt:
                break

        if verbose:
            total_prompts = sum(1 for m in messages if m.get("role") == "user")
            total_responses = sum(1 for m in messages if m.get("role") == "assistant")

            console.print()
            console.print(
                Panel(
                    f"[bold cyan]📊 Session Summary[/bold cyan]\n\n"
                    f"[green]Messages:[/green] {total_prompts} prompts, "
                    f"{total_responses} responses\n\n"
                    f"[magenta]Server:[/magenta] "
                    f"http://{server.host}:{server.port}",
                    title="Chat Complete",
                )
            )


def serve_with_verbose_monitor(server, verbose: bool = False, web_monitor: bool = False):
    """Run server with verbose monitoring.

    Args:
        server: LlamaServer instance
        verbose: Enable detailed monitoring in terminal
        web_monitor: Enable web monitoring page
    """
    from moxing.enhanced_monitor import EnhancedMonitor

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
        console.print(
            Panel(
                f"[green]Web Monitor:[/green] http://{server.host}:{server.port}\n"
                f"[blue]OpenAI API:[/blue] http://{server.host}:{server.port}/v1\n"
                f"[dim]The web page shows live metrics and charts[/dim]",
                title="Web Monitoring Enabled",
            )
        )

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
                model_name = (
                    monitor.server_info.model_name[:30]
                    if monitor.server_info.model_name
                    else "Unknown"
                )
                ctx_len = monitor.server_info.context_length

                avg_gpu = stats.get('avg_gpu_memory', 0)
                avg_cpu = stats.get('avg_cpu', 0)
                console.print(
                    Panel(
                        f"[cyan]Model:[/cyan] {model_name}\n"
                        f"[cyan]Context:[/cyan] {ctx_len:,}\n\n"
                        f"[green]Tokens:[/green]\n"
                        f"  Prompt: {snapshot.prompt_tokens:,}\n"
                        f"  Generated: {snapshot.generated_tokens:,}\n"
                        f"  Total: {snapshot.total_tokens:,}\n\n"
                        f"[yellow]Speed:[/yellow]\n"
                        f"  Prompt: {snapshot.prompt_speed:.1f} tok/s\n"
                        f"  Generate: {snapshot.generate_speed:.1f} tok/s\n"
                        f"  Avg (60s): "
                        f"{stats.get('avg_generate_speed', 0):.1f} tok/s\n\n"
                        f"[blue]Memory:[/blue]\n"
                        f"  GPU: {snapshot.gpu_memory_mb:.0f} MB "
                        f"(avg: {avg_gpu:.0f})\n"
                        f"  RAM: {snapshot.ram_used_mb / 1024:.2f} GB\n\n"
                        f"[magenta]CPU:[/magenta] {snapshot.cpu_percent:.1f}% "
                        f"(avg: {avg_cpu:.1f}%)\n\n"
                        f"[dim]Requests: {snapshot.requests_processing} "
                        f"processing, {snapshot.requests_deferred} "
                        f"deferred[/dim]",
                        title=f"🚀 MoXing Monitor - {model_name}",
                    )
                )

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

            model_name = (
                monitor.server_info.model_name[:30] if monitor.server_info.model_name else "Unknown"
            )
            ctx_len = monitor.server_info.context_length

            avg_gpu2 = stats.get('avg_gpu_memory', 0)
            avg_cpu2 = stats.get('avg_cpu', 0)
            console.print(
                Panel(
                    f"[cyan]Model:[/cyan] {model_name}\n"
                    f"[cyan]Context:[/cyan] {ctx_len:,}\n\n"
                    f"[green]Tokens:[/green]\n"
                    f"  Prompt: {snapshot.prompt_tokens:,}\n"
                    f"  Generated: {snapshot.generated_tokens:,}\n"
                    f"  Total: {snapshot.total_tokens:,}\n\n"
                    f"[yellow]Speed:[/yellow]\n"
                    f"  Prompt: {snapshot.prompt_speed:.1f} tok/s\n"
                    f"  Generate: {snapshot.generate_speed:.1f} tok/s\n"
                    f"  Avg (60s): "
                    f"{stats.get('avg_generate_speed', 0):.1f} tok/s\n\n"
                    f"[blue]Memory:[/blue]\n"
                    f"  GPU: {snapshot.gpu_memory_mb:.0f} MB "
                    f"(avg: {avg_gpu2:.0f})\n"
                    f"  RAM: {snapshot.ram_used_mb / 1024:.2f} GB\n\n"
                    f"[magenta]CPU:[/magenta] {snapshot.cpu_percent:.1f}% "
                    f"(avg: {avg_cpu2:.1f}%)\n\n"
                    f"[dim]Requests: {snapshot.requests_processing} "
                    f"processing, {snapshot.requests_deferred} "
                    f"deferred[/dim]",
                    title=f"🚀 MoXing Monitor - {model_name}",
                )
            )

            time.sleep(1)

    except KeyboardInterrupt:
        pass

    monitor.stop_collection()


def register(app: typer.Typer):
    app.command()(serve)
    app.command()(run)
    app.command("chat")(chat_cmd)
