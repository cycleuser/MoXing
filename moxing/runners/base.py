"""
Unified runner abstraction for all LLM backends.

Supported runners:
- llama_cpp: llama.cpp server (GGUF models)
- ollama: Ollama runner (Ollama model format)
- vllm: vLLM engine (HuggingFace/GGUF models)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from moxing.device import BackendType


@dataclass
class RunnerConfig:
    """Unified configuration for all runners."""
    model: str
    runner_type: str = "llama_cpp"
    host: str = "127.0.0.1"
    port: int = 8080
    backend: str = "auto"
    device: str = "auto"
    ctx_size: int = 4096
    n_gpu_layers: int = -1
    verbose: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)
    kv_cache_quant: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 0
    dtype: str = "auto"
    quantization: Optional[str] = None
    load_format: str = "auto"
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 0
    block_size: int = 0
    attention_backend: str = "auto"
    optimization_level: str = "O2"
    distributed_executor_backend: str = "auto"
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
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
    batch_size: int = 512
    ubatch_size: int = 512
    n_threads: int = -1
    fit_on: bool = False
    kv_unified: bool = True
    cache_reuse: int = 0
    tune_config: Optional[Dict[str, Any]] = None


class BaseRunner(ABC):
    """Abstract base class for all LLM runners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Runner name (e.g., 'llama_cpp', 'vllm', 'ollama')."""
        pass

    @property
    @abstractmethod
    def supported_backends(self) -> List[str]:
        """List of supported hardware backends."""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported model formats (e.g., 'gguf', 'safetensors', 'hf')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this runner is available on the current system."""
        pass

    @abstractmethod
    def start(self, wait: bool = True, timeout: int = 120) -> "BaseRunner":
        """Start the runner server."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the runner server."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the runner is currently running."""
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Get the base URL of the running server."""
        pass

    @abstractmethod
    def _build_args(self) -> List[str]:
        """Build command line arguments or configuration."""
        pass

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def get_runner_class(runner_type: str) -> type:
    """Get the runner class for a given runner type."""
    if runner_type == "llama_cpp":
        from moxing.runners.llama_cpp import LlamaCppRunner
        return LlamaCppRunner
    elif runner_type == "vllm":
        from moxing.runners.vllm import VLLMRunner
        return VLLMRunner
    elif runner_type == "ollama":
        from moxing.runners.ollama import OllamaRunner
        return OllamaRunner
    else:
        raise ValueError(f"Unknown runner type: {runner_type}. Supported: llama_cpp, vllm, ollama")


def create_runner(
    config: RunnerConfig,
) -> BaseRunner:
    """Factory function to create a runner based on configuration."""
    runner_class = get_runner_class(config.runner_type)
    return runner_class(config)


def detect_best_runner(
    model_path: Optional[Path] = None,
    model_name: str = "",
    prefer_backend: Optional[str] = None,
) -> str:
    """Detect the best runner for the given model.

    Priority:
    1. If model is GGUF file -> llama_cpp
    2. If model is ollama: prefix or Ollama model -> ollama
    3. If model is HuggingFace repo -> vllm (if available) else llama_cpp
    4. Default -> llama_cpp
    """
    if model_name.startswith("ollama:"):
        return "ollama"

    if model_path and model_path.exists():
        if model_path.suffix == ".gguf" or str(model_path).endswith(".gguf"):
            return "llama_cpp"
        if model_path.is_dir():
            from moxing.runners.vllm import VLLMRunner
            if VLLMRunner.is_vllm_available():
                return "vllm"
            return "llama_cpp"

    if "/" in model_name and not model_name.endswith(".gguf"):
        from moxing.runners.vllm import VLLMRunner
        if VLLMRunner.is_vllm_available():
            return "vllm"
        return "llama_cpp"

    return "llama_cpp"
