# MoXing

A Python wrapper for llama.cpp that provides an OpenAI API compatible LLM backend with automatic GPU detection and model downloading.

**Key Features:**
- 🚀 **Faster than Ollama** - Direct llama.cpp execution without overhead
- 🔧 **OpenAI Compatible** - Drop-in replacement for OpenAI API
- 🎮 **Multi-GPU Support** - CUDA, Vulkan, ROCm, Metal backends
- 📦 **Auto Download** - Models from HuggingFace, ModelScope, or Ollama
- 💾 **GGUF Compression** - Save disk space with transparent decompression

## Installation

```bash
pip install moxing
```

Binaries are downloaded automatically on first use (~60 MB). No manual setup required.

## Quick Start

```bash
# Serve an Ollama model
moxing ollama serve llama3.2

# Serve from HuggingFace
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF

# Serve a local GGUF file
moxing serve ./model.gguf --port 8080
```

Then use OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## CLI Commands

### Model Management

| Command | Description |
|---------|-------------|
| `moxing serve <model>` | Start server with a model |
| `moxing run <model> "prompt"` | Quick inference |
| `moxing chat <model>` | Interactive chat |
| `moxing download <repo>` | Download model from HF/ModelScope |
| `moxing models` | List available models |

### Ollama Integration

| Command | Description |
|---------|-------------|
| `moxing ollama list` | List Ollama models |
| `moxing ollama serve <model>` | Serve Ollama model |
| `moxing ollama info <model>` | Show model details |

### GGUF Compression

| Command | Description |
|---------|-------------|
| `moxing compress pack <file>` | Compress GGUF file |
| `moxing compress unpack <file>` | Decompress file |
| `moxing compress split <file>` | Split into chunks |
| `moxing compress merge <pattern>` | Merge chunks |

### System & Diagnostics

| Command | Description |
|---------|-------------|
| `moxing devices` | List GPU devices |
| `moxing diagnose` | System diagnostics |
| `moxing bench <model>` | Benchmark performance |
| `moxing --version` | Show version info |

## GPU Backends

MoXing automatically detects and uses the best available backend:

| Platform | CPU | CUDA | Vulkan | ROCm | Metal |
|----------|-----|------|--------|------|-------|
| Linux x64 | ✅ | ✅ | ✅ | ✅ | - |
| Windows x64 | ✅ | ✅ | ✅ | - | - |
| macOS ARM64 | ✅ | - | - | - | ✅ |

Force a specific backend:

```bash
pip install moxing[cuda]   # NVIDIA GPU
pip install moxing[vulkan] # Cross-platform GPU
pip install moxing[metal]  # Apple Silicon
pip install moxing[rocm]   # AMD GPU
pip install moxing[cpu]    # CPU only
```

## Model Sources

### Ollama Models

```bash
moxing ollama list                  # List installed models
moxing ollama serve llama3.2        # Serve with OpenAI API
moxing ollama serve --select        # Interactive selection
```

### HuggingFace

```bash
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF -q Q4_K_M
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF
```

### ModelScope (China Mirror)

```bash
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF --source modelscope
```

## Python API

```python
from moxing import quick_run, quick_server, LlamaServer

# Quick inference
result = quick_run("llama3.2", "Write a haiku about coding")
print(result)

# Start server
with quick_server("llama3.2", port=8080) as server:
    # Use OpenAI API at http://localhost:8080/v1
    pass

# Full control
server = LlamaServer(
    model="model.gguf",
    backend="cuda",
    ctx_size=8192,
    gpu_layers=99
)
server.start()
```

## GGUF Compression

Save disk space by compressing GGUF files:

```bash
# Compress (typically 3-5% smaller)
moxing compress pack model.gguf
# Creates: model.gguf.zst

# Serve compressed file (auto-decompresses)
moxing serve model.gguf.zst

# Split large files
moxing compress split model.gguf --size 512  # 512 MB chunks

# Merge back
moxing compress merge "model.gguf-part-*" merged.gguf

# Manage cache
moxing compress cache --size
moxing compress cache --clear
```

## Performance

On Apple M4 with `carstenuhlig/omnicoder-9b`:

| Framework | Speed |
|-----------|-------|
| Ollama | ~10 tokens/s |
| MoXing | ~15 tokens/s |

Results vary by model and hardware. MoXing removes Ollama's abstraction layer for direct llama.cpp execution.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MOXING_BINARY_SOURCE` | Binary source: `github`, `modelscope`, `auto` |
| `MOXING_BINARY_MIRROR` | Custom binary mirror URL |
| `MOXING_NO_UPDATE_CHECK` | Skip binary update check (set to `1`) |

## Building from Source

### Build Wheel

```bash
./generate_wheel.sh --version 0.1.9
```

### Build Binaries

```bash
# Build all available backends
./generate_binaries.sh

# Build specific backend
./generate_binaries.sh --backend cuda

# Build specific llama.cpp version
./generate_binaries.sh --version b8468
```

### Upload

```bash
./upload_binaries.sh  # Upload to GitHub
./upload_pypi.sh      # Upload to PyPI
```

## How It Works

```
User Request → MoXing → llama.cpp (GPU accelerated) → OpenAI API Response
                  ↓
           Auto-download model if needed
                  ↓
           Auto-download binaries if needed
```

Compressed files are transparently decompressed:

```
model.gguf.zst → ~/.cache/moxing/decompressed/model.gguf → llama.cpp
```

## Compatibility

**Tested Models:**
- Qwen2.5 series
- Llama 3.x series
- Mistral series
- Phi-3 series
- carstenuhlig/omnicoder-9b

**Try your model directly - if it works with llama.cpp, it works with MoXing.**

## Troubleshooting

```bash
# Check system status
moxing diagnose

# Check binary version
moxing --version

# Re-download binaries
moxing download-binaries --force

# Clear all caches
moxing clear-cache --all
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Links

- [GitHub Repository](https://github.com/cycleuser/MoXing)
- [PyPI Package](https://pypi.org/project/moxing/)
- [Issue Tracker](https://github.com/cycleuser/MoXing/issues)