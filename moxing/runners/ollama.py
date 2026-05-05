"""
Ollama runner - wraps the existing OllamaRunnerServer.
"""

import logging
from typing import Any, List, Optional

from moxing.device import DeviceDetector
from moxing.runners.base import BaseRunner, RunnerConfig

logger = logging.getLogger(__name__)


class OllamaRunner(BaseRunner):
    """Runner for Ollama models."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._server: Optional[Any] = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def supported_backends(self) -> List[str]:
        return ["cuda", "rocm", "vulkan", "cpu", "metal", "mlx"]

    @property
    def supported_formats(self) -> List[str]:
        return ["ollama", "gguf"]

    def is_available(self) -> bool:
        try:
            from moxing.ollama import OllamaClient

            client = OllamaClient()
            return client.is_available()
        except Exception as e:
            logger.debug("Service availability check failed: %s", e, exc_info=True)
            return False

    def start(self, wait: bool = True, timeout: int = 120) -> "OllamaRunner":
        model_name = self.config.model
        if model_name.startswith("ollama:"):
            model_name = model_name[7:]

        from moxing.ollama_runner import serve_ollama_model

        backend = self.config.backend
        if backend == "auto":
            detector = DeviceDetector()
            devices = detector.detect()
            gpu_devices = [d for d in devices if d.backend.is_gpu()]
            if gpu_devices:
                backend = min(gpu_devices, key=lambda d: d.backend).backend.value
            else:
                backend = "cpu"

        device = self.config.device
        if device == "auto":
            device = "gpu0"

        self._server = serve_ollama_model(
            model_name=model_name,
            backend=backend,
            device=device,
            port=self.config.port,
            host=self.config.host,
            ctx_size=self.config.ctx_size if self.config.ctx_size > 0 else 32768,
            verbose=self.config.verbose,
            runner_type="ollama",
            lookahead=self.config.lookahead,
            cache_prompts=self.config.cache_prompts,
            slots=self.config.slots,
            cont_batching=self.config.cont_batching,
            mlock=self.config.mlock,
            no_kv_offload=self.config.no_kv_offload,
            rope_scaling=self.config.rope_scaling,
            rope_scale=self.config.rope_scale,
            speculative_draft=self.config.speculative_draft,
            speculative_max=self.config.speculative_max,
            speculative_pmin=self.config.speculative_pmin,
            cpu_moe=self.config.cpu_moe,
            n_threads=self.config.n_threads if self.config.n_threads > 0 else 0,
            batch_size=self.config.batch_size,
            ubatch_size=self.config.ubatch_size,
            flash_attn=True,
            kv_cache=self.config.kv_cache_quant if self.config.kv_cache_quant != "auto" else "f16",
            n_gpu_layers=self.config.n_gpu_layers,
            **self.config.extra_args,
        )

        if self._server is None:
            raise RuntimeError(f"Failed to start Ollama server for model: {model_name}")

        return self

    def stop(self):
        if self._server:
            self._server.stop()
            self._server = None

    def is_running(self) -> bool:
        if self._server:
            return self._server.is_running()
        return False

    @property
    def base_url(self) -> str:
        if self._server:
            return self._server.base_url
        return f"http://{self.config.host}:{self.config.port}"

    def _build_args(self) -> List[str]:
        return []
