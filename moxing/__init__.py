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

__version__ = '0.1.26'

from moxing.client import Client, ChatCompletion, Message
from moxing.server import LlamaServer, ServerConfig, GPUInfo
from moxing.device import (
    Device, DeviceConfig, DeviceDetector, BackendType,
    detect_best_backend, get_device_config
)
from moxing.models import (
    ModelDownloader, ModelInfo, ModelRegistry, download_model
)
from moxing.runner import (
    AutoRunner, RunConfig, quick_run, quick_server
)
from moxing.binaries import (
    BinaryManager, get_binary_manager, ensure_binaries, get_server_binary,
    check_binary_version, get_latest_llama_cpp_version, skip_update_forever,
    clear_skip_update
)
from moxing.gguf_compress import (
    MultiCompressor, TransparentDecompressor, GGUFSplitter,
    compress_model, resolve_model_path
)
from moxing.turboquant import (
    TurboQuant, TurboQuantConfig, TurboQuantMode,
    TurboQuantMixedPrecision, LloydMaxQuantizer, QJLQuantizer,
)
from moxing.kv_cache import (
    KVCacheQuantType, KVCacheConfig,
    estimate_kv_cache_size, estimate_kv_cache_size_gb,
    recommend_cache_config, get_llama_cpp_cache_args,
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
]