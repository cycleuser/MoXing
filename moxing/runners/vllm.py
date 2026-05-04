"""
vLLM runner - wraps vLLM engine for serving models.

Supports:
- HuggingFace models (safetensors, pt)
- GGUF models (via vLLM's GGUF loader)
- Multi-GPU tensor parallelism
- PagedAttention, continuous batching
- Prefix caching, speculative decoding
"""

import os
import sys
import time
import socket
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console

from moxing.runners.base import BaseRunner, RunnerConfig
from moxing.vllm_installer import is_vllm_installed, ensure_vllm

console = Console()


class VLLMRunner(BaseRunner):
    """Runner for vLLM engine."""

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
        """Check if vLLM is available."""
        return is_vllm_installed()

    def is_available(self) -> bool:
        return self.is_vllm_available()

    def start(self, wait: bool = True, timeout: int = 120) -> "VLLMRunner":
        if not self.is_vllm_available():
            console.print("[yellow]vLLM not installed, attempting to install...[/yellow]")
            if not ensure_vllm():
                raise RuntimeError(
                    "vLLM is not installed and could not be installed automatically. "
                    "Install manually: pip install vllm"
                )

        args = self._build_args()
        env = self._prepare_env()

        binary_path = self._find_vllm_binary()

        if not self.config.verbose:
            console.print(f"[blue]Starting vLLM server...[/blue]")
            console.print(f"[dim]Model: {self.config.model}[/dim]")
            console.print(f"[dim]Command: vllm serve {' '.join(args)}[/dim]")

        try:
            self._process = subprocess.Popen(
                [str(binary_path), "serve", self.config.model] + args,
                stdout=subprocess.PIPE if not self.config.verbose else None,
                stderr=subprocess.PIPE if not self.config.verbose else None,
                env=env,
                cwd=str(Path(binary_path).parent),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start vLLM server: {e}")

        if wait:
            self._wait_for_server(timeout)
        else:
            time.sleep(3)
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                console.print(f"\n[red bold]vLLM server failed to start![/red bold]")
                console.print(f"[dim]Exit code: {self._process.returncode}[/dim]")
                if stderr:
                    for line in stderr.decode("utf-8", errors="replace").strip().split("\n")[-20:]:
                        console.print(f"[red]  {line}[/red]")
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
        """Build vLLM serve command arguments."""
        args = []

        args.extend(["--host", self.config.host])
        args.extend(["--port", str(self.config.port)])

        if self.config.tensor_parallel_size > 1:
            args.extend(["--tensor-parallel-size", str(self.config.tensor_parallel_size)])

        if self.config.data_parallel_size > 1:
            args.extend(["--data-parallel-size", str(self.config.data_parallel_size)])

        if self.config.pipeline_parallel_size > 1:
            args.extend(["--pipeline-parallel-size", str(self.config.pipeline_parallel_size)])

        if self.config.dtype != "auto":
            args.extend(["--dtype", self.config.dtype])

        if self.config.quantization:
            args.extend(["--quantization", self.config.quantization])

        model_path = Path(self.config.model)
        if model_path.exists() and model_path.suffix == ".gguf":
            args.extend(["--load-format", "gguf"])
        elif self.config.load_format != "auto":
            args.extend(["--load-format", self.config.load_format])

        if self.config.gpu_memory_utilization != 0.9:
            args.extend(["--gpu-memory-utilization", str(self.config.gpu_memory_utilization)])

        if self.config.max_model_len > 0:
            args.extend(["--max-model-len", str(self.config.max_model_len)])
        elif self.config.ctx_size > 0:
            args.extend(["--max-model-len", str(self.config.ctx_size)])

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

        if self.config.optimization_level != "O2":
            args.extend(["--optimization-level", self.config.optimization_level])

        if self.config.distributed_executor_backend != "auto":
            args.extend(["--distributed-executor-backend", self.config.distributed_executor_backend])

        if self.config.speculative_draft:
            args.extend(["--speculative-model", self.config.speculative_draft])
            args.extend(["--num-speculative-tokens", str(self.config.speculative_max)])

        download_dir = os.environ.get("VLLM_DOWNLOAD_DIR", "")
        if download_dir:
            args.extend(["--download-dir", download_dir])

        for key, value in self.config.extra_args.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])

        return args

    def _prepare_env(self) -> Dict[str, str]:
        """Prepare environment variables for vLLM."""
        env = os.environ.copy()

        if self.config.backend == "rocm":
            env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
            if "HSA_OVERRIDE_GFX_VERSION" not in env:
                env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

            rocm_paths = ["/opt/rocm/lib", "/opt/rocm/core/lib"]
            import glob
            rocm_paths.extend(glob.glob("/opt/rocm/core-*/lib"))

            ld_path = env.get("LD_LIBRARY_PATH", "")
            for path in rocm_paths:
                if Path(path).exists():
                    ld_path = f"{path}:{ld_path}"
                    break
            env["LD_LIBRARY_PATH"] = ld_path

        elif self.config.backend == "cuda":
            if self.config.device.startswith("gpu"):
                try:
                    gpu_id = self.config.device.replace("gpu", "")
                    env["CUDA_VISIBLE_DEVICES"] = gpu_id
                except ValueError:
                    pass

        if self.config.device.startswith("gpu"):
            try:
                gpu_id = self.config.device.replace("gpu", "")
                if self.config.backend == "rocm":
                    env["HIP_VISIBLE_DEVICES"] = gpu_id
                elif self.config.backend == "cuda":
                    env["CUDA_VISIBLE_DEVICES"] = gpu_id
            except ValueError:
                pass

        return env

    def _find_vllm_binary(self) -> Path:
        """Find vllm CLI binary."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "vllm"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Location:"):
                        location = line.split(":", 1)[1].strip()
                        vllm_cli = Path(location) / "vllm" / "entrypoints" / "cli" / "main.py"
                        if vllm_cli.exists():
                            return Path(sys.executable)

                        bin_path = Path(location) / ".." / ".." / "bin" / "vllm"
                        if bin_path.exists():
                            return bin_path
        except Exception:
            pass

        import shutil
        vllm_path = shutil.which("vllm")
        if vllm_path:
            return Path(vllm_path)

        return Path(sys.executable)

    def _wait_for_server(self, timeout: int = 120):
        """Wait for vLLM server to be ready."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                import httpx
                resp = httpx.get(f"{self._base_url}/health", timeout=5)
                if resp.status_code == 200:
                    console.print(f"[green]vLLM server ready at {self._base_url}[/green]")
                    return
            except Exception:
                pass

            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.poll()
                if exit_code != 0:
                    stdout, stderr = self._process.communicate()
                    error_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
                    raise RuntimeError(f"vLLM server failed to start (exit {exit_code}):\n{error_msg}")

            time.sleep(1)

        raise TimeoutError(f"vLLM server did not start within {timeout} seconds")

    def _start_monitor(self):
        """Start background thread to monitor vLLM process."""
        def drain_stream(stream, name):
            try:
                for line in stream:
                    pass
            except Exception:
                pass

        if self._process:
            if self._process.stdout:
                t = threading.Thread(target=drain_stream, args=(self._process.stdout, "stdout"), daemon=True)
                t.start()
            if self._process.stderr:
                t = threading.Thread(target=drain_stream, args=(self._process.stderr, "stderr"), daemon=True)
                t.start()

        def monitor():
            while self._process and self._process.poll() is None:
                time.sleep(0.5)
            if self._process and self._process.poll() is not None:
                console.print(f"\n[red bold]vLLM server crashed! (exit code: {self._process.returncode})[/red bold]")

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
