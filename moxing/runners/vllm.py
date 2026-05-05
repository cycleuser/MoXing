"""
vLLM runner - wraps vLLM engine for serving models.

Supports:
- HuggingFace models (safetensors, pt)
- GGUF models (via vLLM's GGUF loader, ModelScope download)
- Multi-GPU tensor parallelism
- PagedAttention, continuous batching
- Prefix caching, speculative decoding
"""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from moxing.runners.base import BaseRunner, RunnerConfig
from moxing.vllm_installer import ensure_vllm, is_vllm_installed

logger = logging.getLogger(__name__)

console = Console()


class VLLMRunner(BaseRunner):
    """Runner for vLLM engine. Uses `python -m vllm.entrypoints.openai.api_server`."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._base_url = f"http://{config.host}:{config.port}"

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def supported_backends(self) -> List[str]:
        return ["cuda", "rocm", "cpu"]

    @property
    def supported_formats(self) -> List[str]:
        return ["safetensors", "pt", "gguf", "hf"]

    @staticmethod
    def is_vllm_available() -> bool:
        return is_vllm_installed()

    def is_available(self) -> bool:
        return self.is_vllm_available()

    def start(self, wait: bool = True, timeout: int = 180) -> "VLLMRunner":
        if not self.is_vllm_available():
            console.print("[yellow]vLLM not installed, attempting to install...[/yellow]")
            if not ensure_vllm():
                raise RuntimeError("vLLM is not installed. Install: pip install vllm")

        args = self._build_args()
        env = self._prepare_env()

        model = self.config.model

        if self.config.verbose:
            console.print("[blue]Starting vLLM server...[/blue]")
            console.print(f"[dim]Model: {model}[/dim]")
            console.print(f"[dim]Args: {' '.join(args)}[/dim]")

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
        ] + args

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not self.config.verbose else None,
                stderr=subprocess.STDOUT if not self.config.verbose else None,
                env=env,
            )
        except Exception as e:
            logger.debug("Subprocess operation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to start vLLM server: {e}") from e

        if wait:
            self._wait_for_server(timeout)
        else:
            time.sleep(5)
            if self._process.poll() is not None:
                stdout, _ = self._process.communicate()
                console.print("\n[red bold]vLLM server failed to start![/red bold]")
                console.print(f"[dim]Exit code: {self._process.returncode}[/dim]")
                if stdout:
                    for line in stdout.decode("utf-8", errors="replace").strip().split("\n")[-20:]:
                        console.print(f"[dim]  {line}[/dim]")
                raise RuntimeError("vLLM server failed to start")
            self._start_monitor()

        return self

    def stop(self):
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            console.print("[yellow]vLLM server stopped[/yellow]")

    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def base_url(self) -> str:
        return self._base_url

    def _build_args(self) -> List[str]:
        args = []
        args.extend(["--host", self.config.host])
        args.extend(["--port", str(self.config.port)])

        model_path = Path(self.config.model)
        if model_path.exists() and model_path.suffix == ".gguf":
            args.extend(["--load-format", "gguf"])
            args.extend(["--served-model-name", model_path.stem])
        elif self.config.load_format != "auto":
            args.extend(["--load-format", self.config.load_format])

        if self.config.tensor_parallel_size > 1:
            args.extend(["--tensor-parallel-size", str(self.config.tensor_parallel_size)])

        if self.config.pipeline_parallel_size > 1:
            args.extend(["--pipeline-parallel-size", str(self.config.pipeline_parallel_size)])

        if self.config.data_parallel_size > 1:
            args.extend(["--data-parallel-size", str(self.config.data_parallel_size)])

        if self.config.dtype != "auto":
            args.extend(["--dtype", self.config.dtype])

        if self.config.quantization:
            args.extend(["--quantization", self.config.quantization])

        if self.config.gpu_memory_utilization != 0.9:
            args.extend(["--gpu-memory-utilization", str(self.config.gpu_memory_utilization)])

        max_len = (
            self.config.max_model_len
            if self.config.max_model_len > 0
            else (self.config.ctx_size if self.config.ctx_size > 0 else 0)
        )
        if max_len > 0:
            args.extend(["--max-model-len", str(max_len)])

        if self.config.max_num_batched_tokens > 0:
            args.extend(["--max-num-batched-tokens", str(self.config.max_num_batched_tokens)])

        if self.config.max_num_seqs != 256:
            args.extend(["--max-num-seqs", str(self.config.max_num_seqs)])

        if self.config.block_size > 0:
            args.extend(["--block-size", str(self.config.block_size)])

        if self.config.enable_prefix_caching:
            args.append("--enable-prefix-caching")

        if self.config.enforce_eager:
            args.append("--enforce-eager")

        if self.config.attention_backend != "auto":
            args.extend(["--attention-backend", self.config.attention_backend])

        if self.config.distributed_executor_backend != "auto":
            args.extend(
                ["--distributed-executor-backend", self.config.distributed_executor_backend]
            )

        if self.config.speculative_draft:
            args.extend(["--speculative-model", self.config.speculative_draft])
            args.extend(["--num-speculative-tokens", str(self.config.speculative_max)])

        for key, value in self.config.extra_args.items():
            k = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{k}")
            else:
                args.extend([f"--{k}", str(value)])

        return args

    def _prepare_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.config.backend == "rocm":
            env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
            if "HSA_OVERRIDE_GFX_VERSION" not in env:
                env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
            import glob

            rocm_paths = ["/opt/rocm/lib", "/opt/rocm/core/lib"]
            rocm_paths.extend(glob.glob("/opt/rocm/core-*/lib"))
            ld_path = env.get("LD_LIBRARY_PATH", "")
            for path in rocm_paths:
                if Path(path).exists():
                    ld_path = f"{path}:{ld_path}"
                    break
            env["LD_LIBRARY_PATH"] = ld_path

        if self.config.device.startswith("gpu"):
            try:
                gpu_id = self.config.device.replace("gpu", "")
                key = (
                    "HIP_VISIBLE_DEVICES"
                    if self.config.backend == "rocm"
                    else "CUDA_VISIBLE_DEVICES"
                )
                env[key] = gpu_id
            except ValueError:
                pass

        return env

    def _wait_for_server(self, timeout: int = 180):
        start = time.time()
        while time.time() - start < timeout:
            try:
                import httpx

                resp = httpx.get(f"{self._base_url}/health", timeout=5)
                if resp.status_code == 200:
                    console.print(f"[green]vLLM server ready at {self._base_url}[/green]")
                    self._start_monitor()
                    return
            except Exception as e:
                logger.debug("Server health check failed: %s", e, exc_info=True)
                pass

            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.poll()
                stdout, _ = self._process.communicate()
                error_msg = stdout.decode("utf-8", errors="replace") if stdout else "unknown error"
                raise RuntimeError(f"vLLM server exited ({exit_code}):\n{error_msg[-2000:]}")

            time.sleep(2)

        raise TimeoutError(f"vLLM server did not start within {timeout}s")

    def _start_monitor(self):
        if not self._process:
            return

        def drain_stream(stream):
            try:
                for _ in stream:
                    pass
            except Exception as e:
                logger.debug("Stream drain failed: %s", e, exc_info=True)
                pass

        if self._process.stdout:
            threading.Thread(target=drain_stream, args=(self._process.stdout,), daemon=True).start()

        def monitor():
            while self._process and self._process.poll() is None:
                time.sleep(0.5)
            if self._process and self._process.poll() is not None:
                rc = self._process.returncode
                console.print(
                    f"\n[red bold]vLLM server crashed! (exit: {rc})[/red bold]"
                )

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
