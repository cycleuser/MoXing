# moxing (模型)

Python wrapper for llama.cpp - OpenAI API compatible LLM backend with automatic GPU detection and model downloading.

**moxing** (模型) means "model" in Chinese. A simple, unified interface for running LLMs locally.

## Features

- **Auto GPU Detection**: Automatically detects and configures the best GPU backend (Vulkan, CUDA, ROCm, Metal)
- **Model Downloading**: Download GGUF models from HuggingFace and ModelScope
- **OpenAI API Compatible**: Drop-in replacement for OpenAI API
- **Function Calling**: Support for tools and function calling
- **Pre-built Binaries**: Automatically downloads pre-built llama.cpp binaries
- **Benchmark**: Measure tokens/second performance like ollama

## Installation

```bash
pip install moxing
```

## Quick Start

### Step 1: Download a Model

Using ModelScope CLI (recommended for users in China):

```bash
# Install modelscope
pip install modelscope

# Download OmniCoder-9B GGUF model
modelscope download --model Tesslate/OmniCoder-9B-GGUF \
    omnicoder-9b-q4_k_m.gguf \
    --local_dir ./models
```

Or using moxing's built-in downloader:

```bash
moxing download Tesslate/OmniCoder-9B-GGUF -q Q4_K_M
```

### Step 2: List GPU Devices

```bash
moxing devices
```

Output:
```
Available Devices
+----------------------------------------------------------------+
| #   | Name                 | Backend | Memory | Free  | Vendor |
|-----+----------------------+---------+--------+-------+--------|
| 0   | AMD Radeon RX590 GME | vulkan  | 8.0GB  | 7.2GB | amd    |
+----------------------------------------------------------------+
```

### Step 3: Run Inference

```bash
# Quick speed test
moxing speed ./models/omnicoder-9b-q4_k_m.gguf

# Benchmark performance
moxing bench ./models/omnicoder-9b-q4_k_m.gguf

# Interactive chat
moxing chat ./models/omnicoder-9b-q4_k_m.gguf
```

### Step 4: Start OpenAI-Compatible Server

```bash
moxing serve ./models/omnicoder-9b-q4_k_m.gguf -p 8080
```

Now you can use OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}]
)
print(response.choices[0].message.content)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `moxing serve` | Start OpenAI-compatible server |
| `moxing run` | Run inference with a model |
| `moxing chat` | Interactive chat with a model |
| `moxing bench` | Benchmark model performance |
| `moxing speed` | Quick speed test |
| `moxing info` | Show model info and estimates |
| `moxing download` | Download a model |
| `moxing models` | List available models |
| `moxing devices` | List GPU devices |
| `moxing diagnose` | Diagnose system setup |

## Python API

### Quick Inference

```python
from moxing import quick_run, quick_server, Client

# Quick inference
result = quick_run("./models/omnicoder-9b-q4_k_m.gguf", "Write a haiku about coding")
print(result)
```

### Server with Auto-Configuration

```python
from moxing import quick_server, Client

with quick_server("./models/omnicoder-9b-q4_k_m.gguf") as server:
    client = Client(server.base_url)
    
    response = client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0]["message"]["content"])
```

### Auto GPU Detection

```python
from moxing import DeviceDetector

# Detect available GPUs
detector = DeviceDetector()
devices = detector.detect()
for device in devices:
    print(f"{device.name} ({device.backend.value}, {device.memory_gb:.1f}GB)")

# Get optimal configuration
config = detector.get_best_device(model_size_gb=5.0)
print(f"Best device: {config.device.name}")
print(f"Recommended GPU layers: {config.n_gpu_layers}")
```

### Model Download

```python
from moxing import ModelDownloader

downloader = ModelDownloader()

# Download from HuggingFace
path = downloader.download("Qwen/Qwen2.5-7B-Instruct-GGUF", "Q4_K_M.gguf")

# Download from ModelScope
path = downloader.download(
    "Tesslate/OmniCoder-9B-GGUF",
    "omnicoder-9b-q4_k_m.gguf",
    source="modelscope"
)
```

## Popular Models

| Name | Description | Sizes |
|------|-------------|-------|
| OmniCoder-9B | Code generation model | Q4_K_M, Q5_K_M, Q8_0 |
| llama-3.2-3b | Llama 3.2 3B | Q4_K_M, Q5_K_M, Q8_0 |
| qwen2.5-7b | Qwen 2.5 7B | Q4_K_M, Q5_K_M, Q8_0 |
| deepseek-coder-6.7b | DeepSeek Coder | Q4_K_M, Q5_K_M, Q8_0 |

## GPU Backends

| Backend | Platforms | Description |
|---------|-----------|-------------|
| Vulkan | Windows, Linux | Cross-platform GPU API, works on AMD, Intel, NVIDIA |
| CUDA | Windows, Linux | NVIDIA GPUs |
| ROCm | Linux | AMD GPUs |
| Metal | macOS | Apple Silicon |
| CPU | All | Fallback, no GPU required |

## Function Calling

```python
from moxing import Client, LlamaServer

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

with LlamaServer("./models/omnicoder-9b-q4_k_m.gguf") as server:
    client = Client(server.base_url)
    
    response = client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools
    )
    
    if response.choices[0]["message"].get("tool_calls"):
        print("Model wants to call:", response.choices[0]["message"]["tool_calls"])
```

## Performance Example

On AMD Radeon RX590 GME (8GB VRAM) with Vulkan backend:

| Model | Size | Speed |
|-------|------|-------|
| TinyLLama Q4_K_M | 0.62 GB | ~90 t/s |
| OmniCoder-9B Q4_K_M | 5.34 GB | ~18 t/s |

## Requirements

- Python 3.8+
- Vulkan SDK (for Vulkan backend)
- CUDA Toolkit (for CUDA backend)
- ROCm (for ROCm backend)

## License

MIT License - same as llama.cpp

## Links

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [OmniCoder-9B](https://modelscope.cn/models/Tesslate/OmniCoder-9B-GGUF)
- [Issues](https://github.com/ggml-org/llama.cpp/issues)