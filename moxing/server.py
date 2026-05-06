"""
Server management for llama.cpp
"""

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil
from rich.console import Console
from rich.prompt import Confirm

logger = logging.getLogger(__name__)

console = Console()


def find_available_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind((host, port))
            return False
    except OSError:
        return True


@dataclass
class GPUInfo:
    name: str
    backend: str
    memory: int
    index: int = 0


@dataclass
class ServerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 4096
    n_gpu_layers: int = -1
    n_cpu_layers: int = 0
    n_threads: int = -1
    batch_size: int = 512
    ubatch_size: int = 512
    flash_attn: bool = True
    device: str = "auto"
    verbose: bool = False
    gpu_backend: str = "auto"
    auto_ctx: bool = True
    kv_cache_quant: str = "auto"
    cpu_offload: bool = False
    cpu_offload_layers: int = 0
    cpu_moe: bool = False
    speculative_draft: Optional[str] = None
    speculative_max: int = 5
    speculative_min: int = 0
    speculative_pmin: float = 0.75
    lookahead: int = 0
    cache_prompts: bool = False
    cache_rem: str = "lru"
    slots: int = 1
    cont_batching: bool = True
    mlock: bool = False
    no_kv_offload: bool = False
    tensor_split: Optional[str] = None
    main_gpu: int = 0
    numa: Optional[str] = None
    defrag_thold: float = 0.1
    rope_scaling: str = "none"
    rope_scale: float = 1.0
    parallel: int = 1
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    fit_on: bool = False
    kv_unified: bool = True
    cache_reuse: int = 0
    tune_config: Optional[Dict[str, Any]] = None


def _find_binary(backend: str = "auto", runner: str = "official") -> Path:
    """Find llama-server binary using BinaryManager."""
    from moxing.binaries import get_binary_manager

    manager = get_binary_manager(backend, runner)
    if not manager.has_binaries():
        console.print(f"[blue]Downloading llama.cpp binaries for {manager.backend}...[/blue]")
        manager.download_binaries()

    return manager.get_binary_path("llama-server")


class LlamaServer:
    """Manage llama.cpp server instance."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        ctx_size: int = 0,
        n_gpu_layers: int = -1,
        n_cpu_layers: int = 0,
        device: str = "auto",
        gpu_backend: str = "auto",
        runner: str = "official",
        auto_ctx: bool = True,
        kv_cache_quant: str = "auto",
        cpu_offload: bool = False,
        cpu_offload_layers: int = 0,
        cpu_moe: bool = False,
        prompt_offload: bool = False,
        quiet: bool = False,
        speculative_draft: Optional[str] = None,
        speculative_max: int = 5,
        speculative_min: int = 0,
        speculative_pmin: float = 0.75,
        lookahead: int = 0,
        cache_prompts: bool = False,
        cache_rem: str = "lru",
        slots: int = 1,
        cont_batching: bool = True,
        mlock: bool = False,
        no_kv_offload: bool = False,
        tensor_split: Optional[str] = None,
        main_gpu: int = 0,
        numa: Optional[str] = None,
        defrag_thold: float = 0.1,
        rope_scaling: str = "none",
        rope_scale: float = 1.0,
        parallel: int = 1,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        batch_size: int = 512,
        ubatch_size: int = 512,
        n_threads: int = -1,
        fit_on: bool = False,
        kv_unified: bool = True,
        cache_reuse: int = 0,
        tune_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        model_path = Path(model)

        if model_path.exists():
            from moxing.gguf_compress import is_gguf_compressed, resolve_model_path

            if is_gguf_compressed(model_path):
                console.print("[blue]Detected compressed GGUF, decompressing...[/blue]")
            resolved_path = resolve_model_path(model_path)
            self.model = resolved_path.resolve()
        else:
            self.model = model_path.resolve()

        self.host = host
        self.port = port
        self._requested_ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.n_cpu_layers = n_cpu_layers
        self.device = device
        self.gpu_backend = gpu_backend
        self.runner = runner
        self.auto_ctx = auto_ctx
        self.kv_cache_quant = kv_cache_quant
        self.cpu_offload = cpu_offload
        self.cpu_offload_layers = cpu_offload_layers
        self.cpu_moe = cpu_moe
        self.prompt_offload = prompt_offload
        self.quiet = quiet
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.n_threads = n_threads
        self.fit_on = fit_on
        self.kv_unified = kv_unified
        self.cache_reuse = cache_reuse
        self.tune_config = tune_config
        self.extra_args = kwargs

        self.speculative_draft = speculative_draft
        self.speculative_max = speculative_max
        self.speculative_min = speculative_min
        self.speculative_pmin = speculative_pmin
        self.lookahead = lookahead
        self.cache_prompts = cache_prompts
        self.cache_rem = cache_rem
        self.slots = slots
        self.cont_batching = cont_batching
        self.mlock = mlock
        self.no_kv_offload = no_kv_offload
        self.tensor_split = tensor_split
        self.main_gpu = main_gpu
        self.numa = numa
        self.defrag_thold = defrag_thold
        self.rope_scaling = rope_scaling
        self.rope_scale = rope_scale
        self.parallel = parallel
        self.mirostat = mirostat
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta

        self.ctx_size = ctx_size if ctx_size > 0 else 4096

        if auto_ctx and ctx_size == 0:
            self._auto_detect_context()

        self._process: Optional[subprocess.Popen] = None
        self._server_thread: Optional[threading.Thread] = None
        self._base_url = f"http://{host}:{port}"
        self._offload_plan = None

    def _auto_detect_context(self):
        """Auto-detect optimal context size based on available VRAM."""
        from moxing.device import DeviceDetector, calculate_optimal_context, estimate_model_size_gb

        try:
            detector = DeviceDetector()
            devices = detector.detect()

            model_size_gb = estimate_model_size_gb(str(self.model))

            gpu_devices = [d for d in devices if d.backend.is_gpu()]
            if not gpu_devices:
                self.ctx_size = 4096
                return

            best_device = max(gpu_devices, key=lambda d: d.free_memory_mb)
            available_vram_gb = best_device.free_memory_gb

            if available_vram_gb <= 0:
                available_vram_gb = best_device.memory_gb * 0.8

            offload_plan = detector.calculate_offload_plan(best_device, model_size_gb)
            self._offload_plan = offload_plan

            if offload_plan.needs_offload and self.n_gpu_layers == -1:
                if self.cpu_offload_layers > 0:
                    pass
                elif self.prompt_offload or self.cpu_offload:
                    console.print(
                        f"\n[yellow]Warning: Model ({model_size_gb:.1f}GB) "
                        f"may not fit in GPU VRAM "
                        f"({available_vram_gb:.1f}GB)[/yellow]"
                    )
                    suggested = offload_plan.suggested_cpu_layers
                    console.print(
                        f"[yellow]Suggested CPU offload: {suggested} layers[/yellow]"
                    )

                    if sys.stdin.isatty():
                        try:
                            suggested_layers = offload_plan.suggested_cpu_layers
                            if Confirm.ask(
                                f"Enable CPU offload for {suggested_layers} layers?",
                                default=True,
                            ):
                                self.cpu_offload_layers = offload_plan.suggested_cpu_layers
                                self.n_gpu_layers = offload_plan.gpu_layers
                        except:  # noqa: E722
                            pass

            ctx_size, n_gpu_layers, n_cpu_layers, notes = calculate_optimal_context(
                model_size_gb=model_size_gb,
                available_vram_gb=available_vram_gb,
                ctx_size_requested=0,
            )

            self.ctx_size = ctx_size

            console.print(f"[dim]Auto-detected context size: {ctx_size} ({notes})[/dim]")

        except Exception as e:
            logger.debug("Auto-detect context failed: %s", e, exc_info=True)
            console.print(f"[yellow]Could not auto-detect context: {e}[/yellow]")
            self.ctx_size = 4096

    @staticmethod
    def get_binary_path(backend: str = "auto", runner: str = "official") -> Path:
        """Get the path to the llama-server binary for current platform."""
        return _find_binary(backend, runner)

    def _get_binary_for_backend(self) -> Path:
        """Get the correct binary for the configured backend."""
        return _find_binary(self.gpu_backend, self.runner)

    @staticmethod
    def detect_gpus() -> List[GPUInfo]:
        """Detect available GPUs."""
        gpus = []

        try:
            binary = LlamaServer.get_binary_path()
            result = subprocess.run(
                [str(binary), "--list-devices"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=str(binary.parent),
            )

            import re

            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if ":" in line and "MiB" in line:
                    match = re.match(r"(\w+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB", line)
                    if match:
                        backend = match.group(1).lower()
                        idx = int(match.group(2))
                        name = match.group(3).strip()
                        memory = int(match.group(4))
                        gpus.append(GPUInfo(name=name, backend=backend, memory=memory, index=idx))
        except Exception as e:
            logger.debug("Operation failed: %s", e, exc_info=True)
            console.print(f"[yellow]Warning: Could not detect GPUs: {e}[/yellow]")

        return gpus

    def _build_args(self) -> List[str]:
        """Build command line arguments."""
        args = [
            str(self._get_binary_for_backend()),
            "-m",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.ctx_size),
            "--metrics",
        ]

        if self.cpu_moe:
            args.extend(["-ngl", "999"])
        elif self.cpu_offload_layers > 0:
            if self.n_gpu_layers > 0:
                args.extend(["-ngl", str(self.n_gpu_layers)])
            else:
                args.extend(["-ngl", "auto"])
        elif self.n_gpu_layers > 0:
            args.extend(["-ngl", str(self.n_gpu_layers)])
        elif self.n_gpu_layers == 0:
            args.extend(["-ngl", "0"])
        else:
            args.extend(["-ngl", "auto"])

        if self.gpu_backend == "vulkan" and self.n_gpu_layers != 0:
            args.extend(["-dev", "Vulkan0"])
        elif self.device != "auto" and self.device != "CPU" and self.n_gpu_layers != 0:
            if self.gpu_backend in ["cuda", "rocm"]:
                from moxing.device import DeviceDetector
                detector = DeviceDetector()
                device_obj = detector.get_device_by_name(self.device, self.gpu_backend)
                if device_obj:
                    dev_idx = device_obj.backend_index if device_obj.backend_index >= 0 else device_obj.index
                    args.extend(["-dev", str(dev_idx)])
            else:
                from moxing.device import DeviceDetector
                detector = DeviceDetector()
                device_obj = detector.get_device_by_name(self.device, self.gpu_backend)
                if device_obj:
                    dev_idx = device_obj.backend_index if device_obj.backend_index >= 0 else device_obj.index
                    if self.gpu_backend == "metal":
                        device_arg = f"MTL{dev_idx}"
                    else:
                        device_arg = str(dev_idx)
                    args.extend(["-dev", device_arg])

        if self.gpu_backend not in ["auto", "cpu"]:
            os.environ["GGML_BACKEND"] = self.gpu_backend

        if self.n_threads > 0:
            args.extend(["--threads", str(self.n_threads)])

        if self.batch_size > 0:
            args.extend(["--batch-size", str(self.batch_size)])

        if self.ubatch_size > 0:
            args.extend(["--ubatch-size", str(self.ubatch_size)])

        if self.cpu_moe:
            args.append("--cpu-moe")

        if self.fit_on:
            args.extend(["--fit", "on"])

        if self.kv_unified:
            args.append("--kv-unified")

        if self.cache_reuse > 0:
            args.extend(["--cache-reuse", str(self.cache_reuse)])

        kv_cache_args = self._get_kv_cache_args()
        args.extend(kv_cache_args)

        if self.speculative_draft:
            args.extend(["--draft", self.speculative_draft])
            args.extend(["--draft-max", str(self.speculative_max)])
            if self.speculative_min > 0:
                args.extend(["--draft-min", str(self.speculative_min)])
            args.extend(["--draft-p-min", str(self.speculative_pmin)])

        if self.lookahead > 0:
            args.extend(["--lookahead", str(self.lookahead)])

        if self.cache_prompts:
            args.append("--cache-prompts")
            args.extend(["--cache-rem", self.cache_rem])

        if self.slots > 1:
            args.extend(["--slots", str(self.slots)])

        if self.cont_batching:
            args.append("--cont-batching")

        if self.mlock:
            args.append("--mlock")

        if self.no_kv_offload:
            args.append("--no-kv-offload")

        if self.tensor_split:
            args.extend(["--tensor-split", self.tensor_split])

        if self.main_gpu > 0:
            args.extend(["--main-gpu", str(self.main_gpu)])

        if self.numa:
            args.extend(["--numa", self.numa])

        if self.defrag_thold > 0:
            args.extend(["--defrag-thold", str(self.defrag_thold)])

        if self.rope_scaling != "none":
            args.extend(["--rope-scaling", self.rope_scaling])

        if self.rope_scale != 1.0:
            args.extend(["--rope-scale", str(self.rope_scale)])

        if self.parallel > 1:
            args.extend(["--parallel", str(self.parallel)])

        if self.mirostat > 0:
            args.extend(["--mirostat", str(self.mirostat)])
            args.extend(["--mirostat-tau", str(self.mirostat_tau)])
            args.extend(["--mirostat-eta", str(self.mirostat_eta)])

        for key, value in self.extra_args.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])

        return args

    def _get_kv_cache_args(self) -> List[str]:
        """Get KV cache optimization arguments."""
        args = []

        quant_map = {
            "q8_0": "q8_0",
            "q5_0": "q5_0",
            "q4_0": "q4_0",
            "q4_1": "q4_1",
            "q5_1": "q5_1",
            "iq4_nl": "iq4_nl",
            "iq3_s": "q4_0",
            "q3_k": "q4_0",
            "q2_k": "q4_0",
            "tq4": "iq4_nl",
            "tq3.5": "q4_0",
            "tq3": "q4_0",
            "tq2.5": "q4_0",
            "tq2": "q4_0",
        }

        if self.kv_cache_quant == "auto":
            from moxing.device import estimate_model_size_gb
            from moxing.kv_cache import recommend_cache_config

            try:
                model_size_gb = estimate_model_size_gb(str(self.model))
                config = recommend_cache_config(
                    model_size_gb=model_size_gb,
                    available_vram_gb=8.0,
                    desired_ctx_size=self.ctx_size,
                )

                from moxing.kv_cache import get_llama_cpp_cache_args

                args.extend(get_llama_cpp_cache_args(config))

                self._kv_cache_config = config
            except Exception as e:
                logger.debug("KV cache auto-configuration failed: %s", e, exc_info=True)
                pass
        elif self.kv_cache_quant in quant_map:
            cache_type = quant_map[self.kv_cache_quant]
            args.extend(["-ctk", cache_type])
            args.extend(["-ctv", cache_type])

        return args

    def _cleanup_old_processes(self):
        """Kill processes using the target GPU device to free memory."""

        killed_pids = set()

        target_device = self.device if self.device != "auto" else None
        target_backend = self.gpu_backend if self.gpu_backend != "auto" else None

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = proc.info["name"] or ""
                cmdline = proc.info["cmdline"] or []
                cmdline_str = " ".join(cmdline)

                is_llama = name == "llama-server" or "llama-server" in cmdline_str
                is_ollama_runner = (
                    name == "ollama" or name == "runner" or "ollama runner" in cmdline_str
                )

                if not (is_llama or is_ollama_runner):
                    continue

                if target_device and target_backend:
                    device_match = False

                    if target_backend in ["rocm", "hip"] and (
                        "HIP_VISIBLE_DEVICES" in cmdline_str
                        or f"-d {target_device}" in cmdline_str
                    ):
                        device_match = True

                    if not device_match:
                        proc.kill()
                        killed_pids.add(proc.info["pid"])
                else:
                    proc.kill()
                    killed_pids.add(proc.info["pid"])

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if target_backend in ["rocm", "hip"] and sys.platform != "win32":
            try:
                result = subprocess.run(
                    ["amd-smi", "list", "--gpum"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.decode().split("\n")
                    for line in lines:
                        if "PID" in line and "llama" in line.lower():
                            parts = line.split()
                            for part in parts:
                                if part.isdigit():
                                    pid = int(part)
                                    if pid not in killed_pids:
                                        try:
                                            psutil.Process(pid).kill()
                                            killed_pids.add(pid)
                                        except:  # noqa: E722
                                            pass
            except:  # noqa: E722
                pass

        if target_backend == "cuda" and sys.platform != "win32":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.decode().strip().split("\n"):
                        if line.strip() and line.strip().isdigit():
                            pid = int(line.strip())
                            try:
                                proc = psutil.Process(pid)
                                if (
                                    "llama" in proc.name().lower()
                                    or "llama" in " ".join(proc.cmdline()).lower()
                                ) and pid not in killed_pids:
                                    proc.kill()
                                    killed_pids.add(pid)
                            except:  # noqa: E722
                                pass
            except:  # noqa: E722
                pass

        if killed_pids and not self.quiet:
            console.print(
                f"[yellow]Killed {len(killed_pids)} process(es) to free GPU memory[/yellow]"
            )
            time.sleep(3)

    def start(self, wait: bool = True, timeout: int = 60) -> "LlamaServer":
        """Start the server."""
        if self._process is not None:
            raise RuntimeError("Server is already running")

        self._cleanup_old_processes()

        binary_path = self._get_binary_for_backend()

        if not binary_path.exists():
            raise RuntimeError(f"Binary not found: {binary_path}")

        args = self._build_args()

        if not self.quiet:
            console.print("[blue]Starting llama-server...[/blue]")
            console.print(f"[dim]Binary: {binary_path}[/dim]")
            console.print(f"[dim]Command: {' '.join(args)}[/dim]")
            console.print(f"[dim]Working dir: {binary_path.parent}[/dim]")
            console.print(f"[dim]Model: {self.model}[/dim]")
            console.print(f"[dim]Backend: {self.gpu_backend}[/dim]")

            if self.cpu_offload_layers > 0:
                gpu_str = (
                    str(self.n_gpu_layers)
                    if self.n_gpu_layers > 0
                    else 'all'
                )
                console.print(
                    f"[dim]GPU layers: {gpu_str}, "
                    f"CPU offload: {self.cpu_offload_layers}[/dim]"
                )
            else:
                gpu_layers_str = str(self.n_gpu_layers) if self.n_gpu_layers >= 0 else "all"
                console.print(f"[dim]GPU layers: {gpu_layers_str}[/dim]")

            console.print(f"[dim]Context: {self.ctx_size}[/dim]")

        env = os.environ.copy()

        from moxing.device import BackendType, DeviceDetector

        detector = DeviceDetector()

        try:
            backend_type = (
                BackendType(self.gpu_backend) if self.gpu_backend != "auto" else BackendType.CPU
            )

            device_name = self.device
            if (
                device_name.startswith("ROCm")
                or device_name.startswith("CUDA")
                or device_name.startswith("MTL")
            ):
                device_name = "auto"

            device_obj = (
                detector.get_device_by_name(device_name, self.gpu_backend)
                if device_name != "auto"
                else None
            )
            backend_env = detector.get_backend_env(backend_type, device_obj)
            env.update(backend_env)
        except Exception as e:
            logger.debug("Backend environment setup failed: %s", e, exc_info=True)
            pass

        if self.gpu_backend == "rocm":
            rocm_paths = [
                "/opt/rocm/lib",
                "/opt/rocm/core/lib",
            ]

            import glob

            rocm_core_paths = glob.glob("/opt/rocm/core-*/lib")
            rocm_paths.extend(rocm_core_paths)

            for path in rocm_paths:
                if Path(path).exists():
                    if "LD_LIBRARY_PATH" in env:
                        env["LD_LIBRARY_PATH"] = f"{path}:{env['LD_LIBRARY_PATH']}"
                    else:
                        env["LD_LIBRARY_PATH"] = path
                    break

            if "HIP_VISIBLE_DEVICES" not in env:
                env["HIP_VISIBLE_DEVICES"] = "0"

            if "HSA_OVERRIDE_GFX_VERSION" not in env:
                env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

        if self.gpu_backend == "cuda" and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = "0"

        if self.gpu_backend == "vulkan" and "GGML_VK_VISIBLE_DEVICES" not in env:
            env["GGML_VK_VISIBLE_DEVICES"] = "0"

        env["LD_LIBRARY_PATH"] = f"{binary_path.parent}:{env.get('LD_LIBRARY_PATH', '')}"

        try:
            self._process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                cwd=str(binary_path.parent),
                env=env,
            )
        except Exception as e:
            logger.debug("Version info check failed: %s", e, exc_info=True)
            console.print(f"[red]Failed to start process: {e}[/red]")
            raise RuntimeError(f"Failed to start server: {e}") from e

        if wait:
            self._wait_for_server(timeout)
        else:
            time.sleep(2.0)
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                console.print("\n[red bold]Server failed to start![/red bold]")
                console.print(f"[dim]Exit code: {self._process.returncode}[/dim]")
                if stdout:
                    console.print("[dim]stdout:[/dim]")
                    for line in stdout.strip().split("\n")[-20:]:
                        console.print(f"[dim]  {line}[/dim]")
                if stderr:
                    console.print("[red]stderr:[/red]")
                    for line in stderr.strip().split("\n")[-20:]:
                        console.print(f"[red]  {line}[/red]")
                raise RuntimeError("Server failed to start")

            self._start_monitor()

        return self

    def _start_monitor(self):
        """Start background thread to monitor server process and consume output."""

        def drain_stream(stream, name):
            try:
                for _line in stream:
                    pass
            except:  # noqa: E722
                pass

        if self._process:
            if self._process.stdout:
                stdout_thread = threading.Thread(
                    target=drain_stream, args=(self._process.stdout, "stdout"), daemon=True
                )
                stdout_thread.start()
            if self._process.stderr:
                stderr_thread = threading.Thread(
                    target=drain_stream, args=(self._process.stderr, "stderr"), daemon=True
                )
                stderr_thread.start()

        def monitor():
            while self._process and self._process.poll() is None:
                time.sleep(0.5)

            if self._process and self._process.poll() is not None:
                exit_code = self._process.returncode
                console.print(f"\n[red bold]Server crashed! (exit code: {exit_code})[/red bold]")
                console.print("[dim]Check the model path and binary compatibility.[/dim]")

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def _wait_for_server(self, timeout: int = 120):
        """Wait for server to be ready."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=5, follow_redirects=True)
                if resp.status_code == 200:
                    try:
                        props = httpx.get(f"{self._base_url}/props", timeout=5)
                        if props.status_code == 200:
                            data = props.json()
                            if data.get("total_slots", 0) > 0:
                                if not self.quiet:
                                    console.print(
                                        f"[green]Server ready at {self._base_url}[/green]"
                                    )
                                return
                    except Exception as e:
                        logger.debug("Server health check failed: %s", e, exc_info=True)
                        if not self.quiet:
                            console.print(f"[dim]Waiting for props... {e}[/dim]")
                        pass
            except Exception as e:
                logger.debug("Server health check failed: %s", e, exc_info=True)
                if not self.quiet:
                    console.print(f"[dim]Waiting for health... {e}[/dim]")
                pass

            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.poll()
                if exit_code != 0:
                    stdout, stderr = self._process.communicate()
                    raise RuntimeError(f"Server failed to start (exit code {exit_code}):\n{stderr}")

            time.sleep(1)

        raise TimeoutError(f"Server did not start within {timeout} seconds")

    def stop(self):
        """Stop the server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            console.print("[yellow]Server stopped[/yellow]")

    def is_running(self) -> bool:
        """Check if server is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def base_url(self) -> str:
        return self._base_url


def main():
    """CLI entry point."""
    from moxing.cli import app

    app()


if __name__ == "__main__":
    main()
