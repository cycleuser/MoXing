"""
Llama.cpp runner - wraps the existing LlamaServer.
"""

import contextlib
import logging
from pathlib import Path
from typing import List, Optional

from moxing.device import BackendType, DeviceDetector
from moxing.runners.base import BaseRunner, RunnerConfig
from moxing.server import LlamaServer

logger = logging.getLogger(__name__)


class LlamaCppRunner(BaseRunner):
    """Runner for llama.cpp server."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._server: Optional[LlamaServer] = None

    @property
    def name(self) -> str:
        return "llama_cpp"

    @property
    def supported_backends(self) -> List[str]:
        return ["cuda", "rocm", "vulkan", "metal", "mlx", "mps", "cpu"]

    @property
    def supported_formats(self) -> List[str]:
        return ["gguf", "gguf-compressed"]

    def is_available(self) -> bool:
        try:
            from moxing.binaries import get_binary_manager

            manager = get_binary_manager(self.config.backend)
            return manager.has_binaries() or True
        except Exception as e:
            logger.debug("Binary manager lookup failed: %s", e, exc_info=True)
            return True

    def start(self, wait: bool = True, timeout: int = 120) -> "LlamaCppRunner":
        model_path = Path(self.config.model)

        if model_path.exists():
            from moxing.gguf_compress import is_gguf_compressed, resolve_model_path

            if is_gguf_compressed(model_path):
                pass
            resolved_path = resolve_model_path(model_path)
            model_str = str(resolved_path.resolve())
        else:
            model_str = self.config.model

        device_config = self._resolve_device_config()

        self._server = LlamaServer(
            model=model_str,
            host=self.config.host,
            port=self.config.port,
            ctx_size=self.config.ctx_size if self.config.ctx_size > 0 else 4096,
            n_gpu_layers=device_config.n_gpu_layers,
            device=device_config.device_str,
            gpu_backend=device_config.backend,
            kv_cache_quant=self.config.kv_cache_quant,
            cpu_offload=self.config.cpu_offload,
            cpu_offload_layers=self.config.cpu_offload_layers,
            cpu_moe=self.config.cpu_moe,
            speculative_draft=self.config.speculative_draft,
            speculative_type=self.config.speculative_type,
            speculative_max=self.config.speculative_max,
            speculative_min=self.config.speculative_min,
            speculative_pmin=self.config.speculative_pmin,
            lookahead=self.config.lookahead,
            cache_prompts=self.config.cache_prompts,
            cache_rem=self.config.cache_rem,
            slots=self.config.slots,
            cont_batching=self.config.cont_batching,
            mlock=self.config.mlock,
            no_kv_offload=self.config.no_kv_offload,
            tensor_split=self.config.tensor_split,
            main_gpu=self.config.main_gpu,
            numa=self.config.numa,
            defrag_thold=self.config.defrag_thold,
            rope_scaling=self.config.rope_scaling,
            rope_scale=self.config.rope_scale,
            parallel=self.config.parallel,
            mirostat=self.config.mirostat,
            mirostat_tau=self.config.mirostat_tau,
            mirostat_eta=self.config.mirostat_eta,
            batch_size=self.config.batch_size,
            ubatch_size=self.config.ubatch_size,
            n_threads=self.config.n_threads,
            fit_on=self.config.fit_on,
            kv_unified=self.config.kv_unified,
            cache_reuse=self.config.cache_reuse,
            tune_config=self.config.tune_config,
            verbose=self.config.verbose,
            **self.config.extra_args,
        )

        self._server.start(wait=wait, timeout=timeout)
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
        if self._server:
            return self._server._build_args()
        return []

    def _resolve_device_config(self) -> "DeviceConfig":
        detector = DeviceDetector()
        devices = detector.detect()

        model_size_gb = 0.0
        model_path = Path(self.config.model)
        if model_path.exists():
            model_size_gb = model_path.stat().st_size / (1024**3)

        backend = self.config.backend
        if backend == "auto":
            gpu_devices = [d for d in devices if d.backend.is_gpu()]
            if gpu_devices:
                backend = min(gpu_devices, key=lambda d: d.backend).backend.value
            else:
                backend = "cpu"

        if backend == "cpu":
            return DeviceConfig(
                backend="cpu",
                device_str="auto",
                n_gpu_layers=0,
            )

        if self.config.device != "auto":
            device_config = detector.get_device_config_by_name(
                self.config.device, backend, model_size_gb
            )
        else:
            device_config = detector.get_best_device(model_size_gb)
            if backend != "auto":
                with contextlib.suppress(ValueError):
                    device_config.backend = BackendType(backend.lower())

        n_gpu_layers = self.config.n_gpu_layers
        if n_gpu_layers == -1:
            n_gpu_layers = device_config.n_gpu_layers

        device_str = "auto"
        if device_config.device.backend != BackendType.CPU and self.config.device != "auto":
            device_str = self.config.device

        return DeviceConfig(
            backend=device_config.backend.value,
            device_str=device_str,
            n_gpu_layers=n_gpu_layers,
        )


class DeviceConfig:
    def __init__(self, backend: str, device_str: str, n_gpu_layers: int):
        self.backend = backend
        self.device_str = device_str
        self.n_gpu_layers = n_gpu_layers
