# MoXing

[![PyPI version](https://img.shields.io/pypi/v/moxing.svg)](https://pypi.org/project/moxing/)
[![Python Version](https://img.shields.io/pypi/pyversions/moxing.svg)](https://pypi.org/project/moxing/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

A Python wrapper for llama.cpp that provides an OpenAI API compatible LLM backend with automatic GPU detection, model downloading, and advanced performance optimizations.

**MoXing is faster than Ollama** - it runs llama.cpp directly without the abstraction layer overhead, delivering 30-50% better performance on the same hardware.

## Key Features

- **OpenAI API Compatible** - Drop-in replacement for OpenAI API, works with any OpenAI SDK client
- **Auto GPU Detection** - Automatically detects and configures CUDA, Vulkan, ROCm, Metal backends
- **Multiple Model Sources** - Download from HuggingFace, ModelScope (China mirror), or use Ollama models
- **GGUF Compression** - Save disk space with zstd compression and transparent decompression
- **TurboQuant KV Cache** - Google's KV cache compression (arXiv:2504.19874) for larger context
- **Speculative Decoding** - 2-4x speedup with draft models or lookahead decoding
- **Multi-GPU Support** - Tensor parallelism across multiple GPUs
- **Web Monitoring** - Real-time performance dashboard with charts
- **vLLM Engine** - Optional vLLM runner for higher throughput on CUDA/ROCm
- **MLX Backend** - Native Apple Silicon support via MLX framework

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [OpenAI API Compatible Server](#openai-api-compatible-server)
- [Integration with OpenCode and Other Clients](#integration-with-opencode-and-other-clients)
- [CLI Commands](#cli-commands)
- [Model Sources](#model-sources)
- [GPU Backends](#gpu-backends)
- [Device Selection](#device-selection)
- [Performance Optimization](#performance-optimization)
- [KV Cache Management](#kv-cache-management)
- [GGUF Compression](#gguf-compression)
- [TurboQuant](#turboquant)
- [Monitoring](#monitoring)
- [Python API](#python-api)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Building from Source](#building-from-source)
- [License](#license)

## Installation

```bash
pip install moxing
```

Binaries are downloaded automatically on first use (~60 MB). No manual setup required.

### Optional Dependencies

```bash
# Install with all optional dependencies
pip install moxing[all]

# Install with specific features
pip install moxing[openai]    # OpenAI SDK client
pip install moxing[hf]        # HuggingFace Hub integration
pip install moxing[modelscope] # ModelScope integration
pip install moxing[dev]       # Development tools (pytest, ruff, mypy)
```

## Quick Start

```bash
# Serve an Ollama model (auto-downloads if needed)
moxing ollama serve llama3.2

# Serve a HuggingFace GGUF model
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF

# Serve a local GGUF file
moxing serve ./model.gguf

# List available GPU devices
moxing devices
```

After starting the server, the OpenAI API is available at `http://localhost:8080/v1`.

## OpenAI API Compatible Server

MoXing provides a fully OpenAI API compatible server that works as a drop-in replacement for OpenAI's API. This means you can use any OpenAI SDK client, web UI, or IDE integration tool with MoXing.

### Starting the Server

```bash
# Basic usage - serves model on port 8080
moxing serve model.gguf

# Specify port and host
moxing serve model.gguf -p 8080 --host 0.0.0.0

# Use specific GPU and backend
moxing serve model.gguf -d gpu0 -b cuda

# Auto-find available port
moxing serve model.gguf --auto-port

# Enable verbose monitoring
moxing serve model.gguf -v

# Enable web monitoring dashboard
moxing serve model.gguf -w
```

### Server Endpoints

Once running, MoXing exposes these OpenAI-compatible endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (streaming supported) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings (if model supports) |

### Using with Python

```python
from openai import OpenAI

# Connect to MoXing server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Any value works
)

# Chat completion
response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=512,
    stream=False
)

print(response.choices[0].message.content)

# Streaming response
stream = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Write a poem about AI."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Using with curl

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Tell me a story."}],
    "stream": true
  }'
```

### Using with JavaScript/Node.js

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "not-needed"
});

const response = await client.chat.completions.create({
  model: "local",
  messages: [
    { role: "user", content: "What is machine learning?" }
  ]
});

console.log(response.choices[0].message.content);
```

## Integration with OpenCode and Other Clients

MoXing works seamlessly with any OpenAI-compatible client. Here's how to configure popular tools:

### OpenCode

[OpenCode](https://opencode.ai) is an AI coding assistant that supports custom LLM backends. To use MoXing with OpenCode:

1. Start MoXing server:
```bash
moxing serve your-model.gguf -p 8080
```

2. Configure OpenCode to use the local endpoint:
```json
{
  "provider": "openai",
  "baseUrl": "http://localhost:8080/v1",
  "apiKey": "not-needed",
  "model": "local"
}
```

### Continue (VS Code Extension)

[Continue](https://continue.dev) is an open-source autopilot for VS Code:

1. Start MoXing server
2. Edit `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "MoXing Local Model",
      "provider": "openai",
      "model": "local",
      "apiBase": "http://localhost:8080/v1"
    }
  ]
}
```

### Cursor

Cursor IDE supports custom OpenAI endpoints:

1. Start MoXing server
2. Go to Cursor Settings > AI > Custom API
3. Set:
   - API Base URL: `http://localhost:8080/v1`
   - API Key: `any-value`
   - Model: `local`

### Roo Code / Cline

For Roo Code or Cline extensions:

1. Start MoXing server
2. In extension settings, configure:
   - API Provider: OpenAI Compatible
   - Base URL: `http://localhost:8080/v1`
   - API Key: `not-needed`
   - Model ID: `local`

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="local",
    temperature=0.7
)

response = llm.invoke("Explain the theory of relativity.")
print(response.content)
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="http://localhost:8080/v1",
    api_key="not-needed",
    model="local"
)

response = llm.complete("What is Python?")
print(response)
```

### Web UIs

**Open WebUI:**
1. Start MoXing server
2. In Open WebUI admin panel, add custom OpenAI endpoint:
   - URL: `http://localhost:8080/v1`
   - API Key: `not-needed`

**LibreChat:**
Configure in `.env`:
```
OPENAI_REVERSE_PROXY=http://localhost:8080/v1/chat/completions
PROXY_URL=http://localhost:8080
```

## CLI Commands

### Model Serving

| Command | Description |
|---------|-------------|
| `moxing serve <model>` | Start OpenAI API server with model |
| `moxing run <model> -p "prompt"` | Quick single inference |
| `moxing chat <model>` | Interactive chat mode |

### Model Management

| Command | Description |
|---------|-------------|
| `moxing download <repo>` | Download model from HuggingFace/ModelScope |
| `moxing models` | List downloaded models |

### Ollama Integration

| Command | Description |
|---------|-------------|
| `moxing ollama list` | List installed Ollama models |
| `moxing ollama serve <model>` | Serve Ollama model with OpenAI API |
| `moxing ollama info <model>` | Show Ollama model details |
| `moxing ollama serve --select` | Interactive model selection |

### GGUF Compression

| Command | Description |
|---------|-------------|
| `moxing compress pack <file>` | Compress GGUF file with zstd |
| `moxing compress unpack <file>` | Decompress file |
| `moxing compress split <file>` | Split into chunks |
| `moxing compress merge <pattern>` | Merge chunks |
| `moxing compress cache --size` | Check cache size |
| `moxing compress cache --clear` | Clear decompression cache |

### TurboQuant

| Command | Description |
|---------|-------------|
| `moxing turboquant analyze <file>` | Analyze model for TurboQuant |
| `moxing turboquant calibrate <file>` | Calibrate TurboQuant parameters |

### System & Diagnostics

| Command | Description |
|---------|-------------|
| `moxing devices` | List GPU devices and backends |
| `moxing diagnose` | Full system diagnostics |
| `moxing bench <model>` | Benchmark performance |
| `moxing speed <model>` | Quick speed test |
| `moxing info` | Show system information |
| `moxing check <file>` | Check GGUF compatibility |
| `moxing tune <model>` | Auto-tune parameters |
| `moxing config` | Show current configuration |
| `moxing --version` | Show version info |

### Monitoring

| Command | Description |
|---------|-------------|
| `moxing monitor` | Start web monitoring dashboard |

### Binary Management

| Command | Description |
|---------|-------------|
| `moxing download-binaries --list` | List available backends |
| `moxing download-binaries --backend <name>` | Download specific backend |
| `moxing download-binaries --backend all` | Download all backends |
| `moxing download-binaries --force` | Force re-download |
| `moxing clear-cache --all` | Clear all caches |

## Model Sources

### Ollama Models

MoXing can directly serve Ollama models without conversion:

```bash
# List installed Ollama models
moxing ollama list

# Serve an Ollama model (auto-downloads if needed)
moxing ollama serve llama3.2

# Serve with specific device and backend
moxing ollama serve llama3.2 -d gpu0 -b vulkan

# Interactive model selection
moxing ollama serve --select

# Skip compatibility check for custom models
moxing ollama serve my-custom-model --skip-check
```

### HuggingFace

```bash
# Download a GGUF model
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF

# Download with specific quantization
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF -q Q4_K_M

# Download and serve in one command
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF
```

### ModelScope (China Mirror)

For faster downloads in China:

```bash
# Download from ModelScope
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF --source modelscope

# Set as default source
export MOXING_DEFAULT_SOURCE=modelscope
```

### Local Files

```bash
# Serve a local GGUF file
moxing serve ./path/to/model.gguf

# Serve a compressed GGUF file (auto-decompresses)
moxing serve ./model.gguf.zst
```

## GPU Backends

MoXing automatically detects and uses the best available backend for your hardware:

| Platform | CPU | CUDA | Vulkan | ROCm | Metal | MLX |
|----------|-----|------|--------|------|-------|-----|
| Linux x64 | ✅ | ✅ | ✅ | ✅ | - | - |
| Windows x64 | ✅ | ✅ | ✅ | - | - | - |
| macOS ARM64 | ✅ | - | ✅ | - | ✅ | ✅ |

### Backend Selection

Force a specific backend:

```bash
# Use CUDA (NVIDIA GPUs)
moxing serve model.gguf -b cuda

# Use Vulkan (cross-platform: AMD, Intel, NVIDIA)
moxing serve model.gguf -b vulkan

# Use ROCm (AMD GPUs on Linux)
moxing serve model.gguf -b rocm

# Use Metal (Apple GPUs)
moxing serve model.gguf -b metal

# Use MLX (Apple Silicon, better compatibility)
moxing serve model.gguf -b mlx

# Use CPU only
moxing serve model.gguf -b cpu

# Auto-detect (default)
moxing serve model.gguf -b auto
```

### Installing Backend-Specific Dependencies

```bash
pip install moxing[cuda]   # NVIDIA GPU support
pip install moxing[vulkan] # Cross-platform GPU support
pip install moxing[metal]  # Apple Metal support
pip install moxing[rocm]   # AMD ROCm support
pip install moxing[cpu]    # CPU only
```

Note: Backend packages are lightweight - the actual binaries are downloaded automatically on first use.

## Device Selection

### Listing Devices

```bash
moxing devices
```

Example output:
```
Available Devices:
  cpu       - CPU (Apple M4)
  gpu0      - Apple GPU (Metal)
```

### Selecting a Device

```bash
# Use GPU 0 with Vulkan backend
moxing serve model.gguf -d gpu0 -b vulkan

# Use GPU 1 with CUDA backend
moxing serve model.gguf -d gpu1 -b cuda

# Use CPU only
moxing serve model.gguf -d cpu

# Auto-select best device (default)
moxing serve model.gguf -d auto
```

### Running Multiple Instances

Run multiple models simultaneously on different devices:

```bash
# Auto-find available ports
moxing serve model1.gguf -d gpu0 --auto-port &
moxing serve model2.gguf -d gpu1 --auto-port &
moxing serve model3.gguf -d cpu --auto-port &

# Or specify ports manually
moxing serve model1.gguf -d gpu0 -p 8080 &
moxing serve model2.gguf -d gpu1 -p 8081 &
moxing serve model3.gguf -d cpu -p 8082 &
```

### Device Options

| Option | Description |
|--------|-------------|
| `-d gpu0`, `-d gpu1`, ... | Select GPU by index |
| `-d cpu` | Use CPU only |
| `-d auto` | Auto-select best device (default) |

### Port Options

| Option | Description |
|--------|-------------|
| `-p 8080` | Use specific port |
| `-p 0` or `--auto-port` | Auto-find available port |

### Backend Options

| Option | Description |
|--------|-------------|
| `-b vulkan` | Cross-platform GPU (AMD, Intel, NVIDIA) |
| `-b cuda` | NVIDIA GPU |
| `-b rocm` | AMD GPU (Linux) |
| `-b metal` | Apple GPU (macOS) |
| `-b mlx` | MLX framework (Apple Silicon) |
| `-b cpu` | CPU only |
| `-b auto` | Auto-detect (default) |

### Downloading Multiple Backend Binaries

Download binaries for all supported backends to enable device switching:

```bash
# List available backends
moxing download-binaries --list

# Download specific backend
moxing download-binaries --backend vulkan

# Download all backends for multi-device support
moxing download-binaries --backend all
```

## Performance Optimization

MoXing provides multiple optimization techniques to maximize inference speed and minimize memory usage.

### Speculative Decoding

Speculative decoding uses a smaller draft model to predict tokens, then verifies with the main model. This can achieve 2-4x speedup.

```bash
# Using a smaller draft model
moxing serve main-model.gguf --draft small-model.gguf

# Using the same model for Multi-Token Prediction (MTP)
moxing serve model.gguf --draft model.gguf

# Configure draft parameters
moxing serve model.gguf --draft small-model.gguf \
  --draft-max 5 \      # Max draft tokens
  --draft-p-min 0.75   # Min acceptance probability
```

### Lookahead Decoding

Lookahead decoding achieves 1.5-2x speedup without requiring a separate draft model:

```bash
# Enable lookahead decoding (2-4 steps recommended)
moxing serve model.gguf --lookahead 3
```

### Prompt Caching

Cache repeated system prompts to avoid re-computation:

```bash
# Enable prompt caching
moxing serve model.gguf --cache-prompts

# Set cache removal policy
moxing serve model.gguf --cache-prompts --cache-rem lru   # Least Recently Used
moxing serve model.gguf --cache-prompts --cache-rem fifo   # First In First Out
```

### Continuous Batching

Handle multiple concurrent requests efficiently:

```bash
# Enable continuous batching (default)
moxing serve model.gguf --cont-batching

# Set number of parallel slots
moxing serve model.gguf --slots 4
```

### Context Extension

Extend context length beyond model's native limit using RoPE scaling:

```bash
# 2x context with linear scaling
moxing serve model.gguf --rope-scaling linear --rope-scale 2

# 4x context with YaRN scaling (better quality)
moxing serve model.gguf --rope-scaling yarn --rope-scale 4
```

### Memory Optimization

```bash
# Offload layers to CPU RAM (useful when model doesn't fit in VRAM)
moxing serve model.gguf --cpu-offload 10

# Offload MoE experts to CPU, keep attention on GPU (7-8x speedup for MoE models)
moxing serve model.gguf --cpu-moe

# Lock model in RAM to prevent swapping
moxing serve model.gguf --mlock

# Force KV cache to stay on GPU (faster but uses more VRAM)
moxing serve model.gguf --no-kv-offload
```

### Multi-GPU Support

```bash
# Split model across GPUs (llama.cpp)
moxing serve model.gguf --tensor-split 50,50  # Equal split between 2 GPUs
moxing serve model.gguf --tensor-split 70,30  # 70% on GPU0, 30% on GPU1

# Set main GPU for tensor parallelism
moxing serve model.gguf --main-gpu 0

# NUMA policy for multi-CPU systems
moxing serve model.gguf --numa distribute
```

### vLLM Engine

For higher throughput on CUDA/ROCm systems, use the vLLM engine:

```bash
# Use vLLM engine
moxing serve model -r vllm

# Tensor parallelism across multiple GPUs
moxing serve model -r vllm --tp 2

# Configure GPU memory utilization
moxing serve model -r vllm --gpu-mem-util 0.95

# Set maximum context length
moxing serve model -r vllm --max-model-len 8192

# Enable prefix caching
moxing serve model -r vllm --prefix-cache

# Specify attention backend
moxing serve model -r vllm --attn-backend flash_attn
```

## KV Cache Management

The KV cache stores attention key-value pairs for faster generation. MoXing provides automatic KV cache management and compression.

### Automatic KV Cache Selection

```bash
# Auto-select best KV cache type
moxing serve model.gguf --kv-cache auto

# Analyze KV cache requirements
moxing serve model.gguf --analyze-cache
```

### KV Cache Quantization Types

| Type | Description | Quality | Memory |
|------|-------------|---------|--------|
| `f16` | Full precision (16-bit) | Best | Highest |
| `q8_0` | 8-bit quantization | High | Medium |
| `q5_0` | 5-bit quantization | Good | Low |
| `q4_0` | 4-bit quantization | Balanced | Lowest |

```bash
# Use 8-bit KV cache
moxing serve model.gguf --kv-cache q8_0

# Use 4-bit KV cache (recommended for most cases)
moxing serve model.gguf --kv-cache q4_0
```

### Estimating KV Cache Size

```python
from moxing import estimate_kv_cache_size_gb, recommend_cache_config

# Estimate KV cache size for a 7B model with 8192 context
size_gb = estimate_kv_cache_size_gb(model_size_gb=4.5, context=8192)
print(f"Estimated KV cache: {size_gb:.2f} GB")

# Get recommended cache configuration
config = recommend_cache_config(model_size_gb=4.5, available_vram_gb=12)
print(f"Recommended: {config}")
```

## GGUF Compression

Save disk space by compressing GGUF files with zstd. Compressed files are transparently decompressed when served.

### Compressing Models

```bash
# Compress a GGUF file
moxing compress pack model.gguf
# Creates: model.gguf.zst

# Serve compressed file (auto-decompresses on first use)
moxing serve model.gguf.zst
```

Typical compression ratios:
- Q4_K_M models: ~3-5% size reduction
- Larger models: up to 10% size reduction

### Splitting Large Files

```bash
# Split into 512 MB chunks
moxing compress split model.gguf --size 512
# Creates: model.gguf-part-00, model.gguf-part-01, ...

# Merge chunks back
moxing compress merge "model.gguf-part-*" merged.gguf
```

### Cache Management

```bash
# Check decompression cache size
moxing compress cache --size

# Clear decompression cache
moxing compress cache --clear
```

## TurboQuant

TurboQuant is Google's KV cache compression technique (arXiv:2504.19874) that achieves higher compression ratios than standard quantization with minimal quality loss.

### TurboQuant Modes

| Mode | Bits | Quality | Use Case |
|------|------|---------|----------|
| `tq4` | 4-bit | High quality | General use |
| `tq3.5` | 3.5-bit | Quality neutral | Recommended |
| `tq3` | 3-bit | Good quality | Memory constrained |
| `tq2.5` | 2.5-bit | Slight loss | Maximum compression |
| `tq2` | 2-bit | Noticeable loss | Extreme compression |

### Using TurboQuant

```bash
# Use TurboQuant 3.5-bit (recommended)
moxing serve model.gguf --kv-cache tq3.5

# Use TurboQuant 4-bit
moxing serve model.gguf --kv-cache tq4

# Use TurboQuant 3-bit
moxing serve model.gguf --kv-cache tq3

# Analyze model for TurboQuant
moxing turboquant analyze model.gguf

# Calibrate TurboQuant parameters
moxing turboquant calibrate model.gguf
```

### Python API

```python
from moxing import TurboQuant, TurboQuantConfig

# Create TurboQuant instance
tq = TurboQuant(TurboQuantConfig(bits_per_channel=3.5))

# Quantize KV cache
quantized_cache = tq.quantize(kv_cache)
```

## Monitoring

MoXing provides real-time monitoring of server performance.

### Terminal Monitoring

```bash
# Enable verbose monitoring in terminal
moxing serve model.gguf -v
```

This shows:
- Token generation speed
- Prompt processing speed
- GPU and CPU utilization
- Memory usage
- Request queue status

### Web Monitoring Dashboard

```bash
# Enable web monitoring dashboard
moxing serve model.gguf -w
```

Access the dashboard at `http://localhost:8080` to see:
- Live performance charts
- Token throughput graphs
- Memory usage over time
- Request latency metrics

### Dedicated Monitor Command

```bash
# Start monitoring dashboard separately
moxing monitor
```

## Python API

MoXing provides a comprehensive Python API for programmatic access.

### Quick Start

```python
from moxing import quick_run, quick_server

# Quick inference (downloads model if needed)
result = quick_run("llama3.2", "Write a haiku about coding")
print(result)

# Start server as context manager
with quick_server("llama3.2", port=8080) as server:
    # Server is running at http://localhost:8080/v1
    # Use any OpenAI SDK client
    pass
```

### Full Server Control

```python
from moxing import LlamaServer, ServerConfig

# Create server with configuration
config = ServerConfig(
    model="model.gguf",
    backend="cuda",
    device="gpu0",
    ctx_size=8192,
    gpu_layers=99,
    kv_cache_quant="q4_0"
)

server = LlamaServer(config)
server.start()

try:
    # Server is running...
    import time
    time.sleep(60)
finally:
    server.stop()
```

### Using the Client

```python
from moxing import Client

# Connect to server
client = Client("http://localhost:8080")

# Chat completion
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Device Detection

```python
from moxing import DeviceDetector, detect_best_backend

# Detect available devices
detector = DeviceDetector()
devices = detector.detect()

for device in devices:
    print(f"{device.name}: {device.backend} ({device.memory_mb} MB)")

# Detect best backend for current system
backend = detect_best_backend()
print(f"Best backend: {backend}")
```

### Model Downloading

```python
from moxing import ModelDownloader, download_model

# Download from HuggingFace
downloader = ModelDownloader()
path = downloader.download("Qwen/Qwen2.5-7B-Instruct-GGUF", quant="Q4_K_M")
print(f"Downloaded to: {path}")

# Download from ModelScope
path = downloader.download(
    "Qwen/Qwen2.5-7B-Instruct-GGUF",
    source="modelscope"
)

# Simple download function
path = download_model("Qwen/Qwen2.5-7B-Instruct-GGUF")
```

### GGUF Compression

```python
from moxing import MultiCompressor, TransparentDecompressor

# Compress a GGUF file
compressor = MultiCompressor()
compressed_path = compressor.compress("model.gguf")
print(f"Compressed to: {compressed_path}")

# Transparent decompression (auto-caches)
decompressor = TransparentDecompressor()
resolved_path = decompressor.resolve("model.gguf.zst")
print(f"Resolved to: {resolved_path}")
```

### KV Cache Utilities

```python
from moxing import (
    estimate_kv_cache_size,
    estimate_kv_cache_size_gb,
    recommend_cache_config,
    get_llama_cpp_cache_args
)

# Estimate KV cache size
size_bytes = estimate_kv_cache_size(
    model_size_gb=4.5,
    context=8192,
    num_layers=32
)

size_gb = estimate_kv_cache_size_gb(model_size_gb=4.5, context=8192)

# Get recommended configuration
config = recommend_cache_config(
    model_size_gb=4.5,
    available_vram_gb=12,
    target_context=8192
)

# Get llama.cpp cache arguments
args = get_llama_cpp_cache_args(quant_type="q4_0", context=8192)
```

### Binary Management

```python
from moxing import (
    get_binary_manager,
    ensure_binaries,
    get_server_binary,
    check_binary_version,
    get_latest_llama_cpp_version
)

# Get binary manager
manager = get_binary_manager()

# Ensure binaries are downloaded
ensure_binaries()

# Get server binary path
binary_path = get_server_binary()

# Check installed version
installed = check_binary_version()
print(f"Installed: {installed}")

# Check latest available version
latest = get_latest_llama_cpp_version()
print(f"Latest: {latest}")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MOXING_BINARY_SOURCE` | Binary source: `github`, `modelscope`, `auto` | `auto` |
| `MOXING_BINARY_MIRROR` | Custom binary mirror URL | - |
| `MOXING_NO_UPDATE_CHECK` | Skip binary update check | `0` |
| `MOXING_DEFAULT_SOURCE` | Default model source: `huggingface`, `modelscope` | `huggingface` |
| `HF_TOKEN` | HuggingFace API token for gated models | - |
| `MODELSCOPE_TOKEN` | ModelScope API token | - |

### Example Usage

```bash
# Use ModelScope for binary downloads (faster in China)
export MOXING_BINARY_SOURCE=modelscope

# Use custom mirror
export MOXING_BINARY_MIRROR=https://my-mirror.example.com/binaries

# Skip update check for offline use
export MOXING_NO_UPDATE_CHECK=1

# Set default model source to ModelScope
export MOXING_DEFAULT_SOURCE=modelscope
```

## Troubleshooting

### Common Issues

**Binary download fails:**
```bash
# Retry download
moxing download-binaries --force

# Use alternative source
export MOXING_BINARY_SOURCE=modelscope
moxing download-binaries
```

**Model not found:**
```bash
# Check if model exists
moxing models

# Download model first
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF
```

**GPU not detected:**
```bash
# Run diagnostics
moxing diagnose

# Check available devices
moxing devices

# Force CPU mode
moxing serve model.gguf -b cpu
```

**Out of memory:**
```bash
# Use KV cache quantization
moxing serve model.gguf --kv-cache q4_0

# Offload layers to CPU
moxing serve model.gguf --cpu-offload 10

# Use TurboQuant for more compression
moxing serve model.gguf --kv-cache tq3.5
```

**Slow performance:**
```bash
# Benchmark your setup
moxing bench model.gguf

# Auto-tune parameters
moxing tune model.gguf

# Check if using GPU
moxing serve model.gguf -v  # Verbose mode shows backend
```

### Diagnostic Commands

```bash
# Full system diagnostics
moxing diagnose

# Check binary version
moxing --version

# Check GGUF compatibility
moxing check model.gguf

# Show system information
moxing info

# Benchmark performance
moxing bench model.gguf

# Quick speed test
moxing speed model.gguf
```

### Clearing Caches

```bash
# Clear all caches
moxing clear-cache --all

# Clear decompression cache only
moxing compress cache --clear
```

## Building from Source

### Prerequisites

- Python 3.8+
- Git
- CMake (for building llama.cpp)

### Build Wheel

```bash
# Clone repository
git clone https://github.com/cycleuser/MoXing.git
cd MoXing

# Install in development mode
pip install -e ".[dev]"

# Build wheel
python -m build
```

### Build Binaries

```bash
# Build all available backends
./scripts/build_platform_wheels.py

# Build for specific platform
# (Requires appropriate build environment)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_device.py

# Run with coverage
pytest --cov=moxing
```

### Code Quality

```bash
# Lint check
ruff check moxing/

# Auto-fix lint issues
ruff check --fix moxing/

# Format code
ruff format moxing/

# Type check
mypy moxing/
```

### Upload to PyPI

```bash
# Build distribution
python -m build

# Upload to PyPI
twine upload dist/*

# Or use the provided script
./upload_pypi.sh
```

## How It Works

### Architecture

```
User Request → MoXing CLI → AutoRunner → LlamaServer → llama.cpp (GPU accelerated)
                    ↓            ↓            ↓
              Download      Configure    OpenAI API
              Model if      Device &     Response
              needed        Backend
```

### Transparent Decompression

```
model.gguf.zst → ~/.cache/moxing/decompressed/model.gguf → llama.cpp
```

Compressed files are automatically decompressed and cached. The cache is reused on subsequent runs.

### Device Selection Flow

```
1. Detect available GPUs
2. Check installed backends
3. Match backend to GPU type
4. Select best device
5. Start server with configuration
```

## Compatibility

### Tested Models

- Qwen2.5 series (0.5B to 72B)
- Llama 3.x series (8B to 405B)
- Mistral series (7B to 24B)
- Phi-3 series (3B to 14B)
- DeepSeek series (various sizes)
- carstenuhlig/omnicoder-9b

**If it works with llama.cpp, it works with MoXing.**

### Supported Platforms

| OS | Architecture | Supported Backends |
|----|--------------|-------------------|
| Linux | x86_64 | CPU, CUDA, Vulkan, ROCm |
| Windows | x86_64 | CPU, CUDA, Vulkan |
| macOS | ARM64 | CPU, Metal, Vulkan, MLX |

### Python Versions

- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13

## Performance

Benchmarks on Apple M4 with `carstenuhlig/omnicoder-9b` (Q4_K_M):

| Framework | Speed | Notes |
|-----------|-------|-------|
| Ollama | ~10 tok/s | With abstraction overhead |
| MoXing | ~15 tok/s | Direct llama.cpp execution |

Results vary by model and hardware. MoXing removes Ollama's abstraction layer for direct llama.cpp execution, achieving 30-50% better performance.

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `ruff check moxing/ && ruff format moxing/`
6. Commit your changes: `git commit -m "Add my feature"`
7. Push to the branch: `git push origin feature/my-feature`
8. Open a Pull Request

### Development Setup

```bash
# Clone and install
git clone https://github.com/cycleuser/MoXing.git
cd MoXing
pip install -e ".[dev,all]"

# Run tests
pytest

# Check code quality
ruff check moxing/
mypy moxing/
```

## License

GPL-3.0 License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/cycleuser/MoXing)
- [PyPI Package](https://pypi.org/project/moxing/)
- [Issue Tracker](https://github.com/cycleuser/MoXing/issues)
- [Chinese Documentation (中文文档)](README_CN.md)

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The underlying LLM inference engine
- [OpenAI](https://openai.com) - For the API specification
- [HuggingFace](https://huggingface.co) - For model hosting
- [ModelScope](https://modelscope.cn) - For China mirror support
- [Google](https://arxiv.org/abs/2504.19874) - For TurboQuant research