"""
moxing (模型) - Python wrapper for llama.cpp

Provides an OpenAI API compatible interface for running GGUF models
with GPU acceleration (Vulkan, CUDA, ROCm, Metal).

Features:
- Auto-detect best GPU device and backend
- Download models from HuggingFace and ModelScope
- OpenAI API compatible server
- Function calling / tool support
- Multimodal support
- GGUF compression and transparent decompression
- TurboQuant KV cache compression (arXiv:2504.19874)

Quick start:
    from moxing import quick_run, quick_server

    # Quick inference
    result = quick_run("llama-3.2-3b", "Write a haiku")

    # Start server
    with quick_server("llama-3.2-3b") as server:
        # Use OpenAI API at http://localhost:8080/v1
        pass

    # TurboQuant for KV cache compression
    from moxing import TurboQuant, TurboQuantConfig
    tq = TurboQuant(TurboQuantConfig(bits_per_channel=3.5))
"""

__version__ = "0.1.38"

from moxing.binaries import (
    BinaryManager,
    check_binary_version,
    clear_skip_update,
    ensure_binaries,
    get_binary_manager,
    get_latest_llama_cpp_version,
    get_server_binary,
    skip_update_forever,
)
from moxing.client import ChatCompletion, Client, Message
from moxing.device import (
    BackendType,
    Device,
    DeviceConfig,
    DeviceDetector,
    detect_best_backend,
    get_device_config,
)
from moxing.gguf_compress import (
    GGUFSplitter,
    MultiCompressor,
    TransparentDecompressor,
    compress_model,
    resolve_model_path,
)
from moxing.gguf_metadata import (
    ModelArchitecture,
    extract_model_architecture,
    should_use_cpu_moe,
)
from moxing.kv_cache import (
    KVCacheConfig,
    KVCacheQuantType,
    estimate_kv_cache_size,
    estimate_kv_cache_size_gb,
    get_llama_cpp_cache_args,
    recommend_cache_config,
)
from moxing.kv_cache_selector import (
    KVCacheSelection,
    calculate_kv_cache_vram_mb,
    estimate_context_from_vram,
    select_kv_cache_type,
)
from moxing.models import ModelDownloader, ModelInfo, ModelRegistry, download_model
from moxing.runner import AutoRunner, RunConfig, quick_run, quick_server
from moxing.server import GPUInfo, LlamaServer, ServerConfig
from moxing.turboquant import (
    LloydMaxQuantizer,
    QJLQuantizer,
    TurboQuant,
    TurboQuantConfig,
    TurboQuantMixedPrecision,
    TurboQuantMode,
)
from moxing.warmup_benchmark import (
    HardwareFingerprint,
    ProfileCache,
    TunedProfile,
    WarmupBenchmark,
    get_hardware_fingerprint,
)

GGUFCompressor = MultiCompressor  # Alias for backward compatibility

__all__ = [
    # Client
    "Client",
    "ChatCompletion",
    "Message",
    # Server
    "LlamaServer",
    "ServerConfig",
    "GPUInfo",
    # Device
    "Device",
    "DeviceConfig",
    "DeviceDetector",
    "BackendType",
    "detect_best_backend",
    "get_device_config",
    # Models
    "ModelDownloader",
    "ModelInfo",
    "ModelRegistry",
    "download_model",
    # Runner
    "AutoRunner",
    "RunConfig",
    "quick_run",
    "quick_server",
    # Binaries
    "BinaryManager",
    "get_binary_manager",
    "ensure_binaries",
    "get_server_binary",
    "check_binary_version",
    "get_latest_llama_cpp_version",
    "skip_update_forever",
    "clear_skip_update",
    # Compression
    "GGUFCompressor",
    "MultiCompressor",
    "TransparentDecompressor",
    "GGUFSplitter",
    "compress_model",
    "resolve_model_path",
    # TurboQuant
    "TurboQuant",
    "TurboQuantConfig",
    "TurboQuantMode",
    "TurboQuantMixedPrecision",
    "LloydMaxQuantizer",
    "QJLQuantizer",
    # KV Cache
    "KVCacheQuantType",
    "KVCacheConfig",
    "estimate_kv_cache_size",
    "estimate_kv_cache_size_gb",
    "recommend_cache_config",
    "get_llama_cpp_cache_args",
    # GGUF Metadata
    "ModelArchitecture",
    "extract_model_architecture",
    "should_use_cpu_moe",
    # KV Cache Selector
    "KVCacheSelection",
    "select_kv_cache_type",
    "calculate_kv_cache_vram_mb",
    "estimate_context_from_vram",
    # Warmup Benchmark
    "WarmupBenchmark",
    "ProfileCache",
    "HardwareFingerprint",
    "TunedProfile",
    "get_hardware_fingerprint",
]
