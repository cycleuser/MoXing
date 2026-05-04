# MoXing Architecture

## Overview

MoXing is a multi-backend LLM inference platform that provides:
1. GGUF file execution via llama.cpp
2. HuggingFace model support via MLX (macOS)
3. Ollama model discovery and execution

## Architecture Diagram

```
+---------------------------------------------------------------------+
|                           CLI Layer                                  |
|  (cli.py - typer commands: serve, ollama, bench, check, etc.)       |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                        Backend Router                                |
|  - Parse model string (GGUF path / HuggingFace / ollama:xxx)        |
|  - Check GGUF compatibility                                          |
|  - Select appropriate backend                                        |
+---------------------------------------------------------------------+
                                  |
          +-----------------------+-----------------------+
          v                       v                       v
+-----------------+     +-----------------+     +-----------------+
|   llama.cpp     |     |      MLX        |     |    Ollama API   |
|   Backend       |     |    Backend      |     |    (fallback)   |
+-----------------+     +-----------------+     +-----------------+
| LlamaServer     |     | MLXServer       |     | HTTP client     |
| (server.py)     |     | (mlx_server.py) |     │ to port 11434   |
+-----------------+     +-----------------+     +-----------------+
          |                       |                       |
          +-----------------------+-----------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                        Model Sources                                 |
+---------------------------------------------------------------------+
|  Local GGUF files                                                    |
|  ├── /path/to/model.gguf                                            |
|  └── ~/.cache/moxing/models/                                         |
|                                                                      |
|  HuggingFace                                                         |
|  ├── Download via huggingface_hub                                   |
|  └── Cache in ~/.cache/huggingface/                                  |
|                                                                      |
|  Ollama                                                              |
|  ├── Manifests: ~/.ollama/models/manifests/                         |
|  └── Blobs (GGUF): ~/.ollama/models/blobs/                          |
+---------------------------------------------------------------------+
```

## Component Details

### 1. CLI Layer (cli.py)

Entry point using typer. Main commands:

- `serve`: Start OpenAI-compatible server
- `ollama list/serve/info/run`: Ollama integration
- `check`: GGUF compatibility check
- `bench/speed`: Performance testing

### 2. Backend Router

Logic for backend selection:

```python
if model.startswith("ollama:"):
    # Find GGUF in ~/.ollama/models/blobs/
    # Run with llama.cpp
elif model.endswith(".gguf"):
    # Check compatibility
    # If incompatible + MLX available -> suggest MLX
    # Else -> llama.cpp
elif "/" in model:  # HuggingFace
    # macOS + MLX available -> MLX
    # Else -> download GGUF or error
```

### 3. llama.cpp Backend (server.py)

```python
class LlamaServer:
    def __init__(self, model, port, ...):
        self.model = model  # GGUF path
        self._process = None
    
    def start(self):
        binary = get_binary_path()  # From BinaryManager
        args = [binary, "-m", self.model, "--port", ...]
        self._process = subprocess.Popen(args, ...)
```

**Binary Management (binaries.py)**:
- Check platform: darwin-arm64, windows-x64-cuda, linux-x64-vulkan
- Download from llama.cpp releases if needed
- Cache in `~/.cache/moxing/binaries/`

### 4. MLX Backend (mlx_server.py)

```python
class MLXServer:
    def __init__(self, model, ...):
        self.model = model  # HuggingFace model name
    
    def start(self):
        # Generate embedded server script
        # Launch with Python + mlx_lm
        # Provides OpenAI-compatible HTTP API
```

**Server Script** (embedded in mlx_server.py):
- HTTP server on specified port
- `/health`, `/v1/models`, `/v1/chat/completions`
- Uses `mlx_lm.load()` and `mlx_lm.generate()`

### 5. Ollama Integration (ollama.py)

```python
class OllamaClient:
    def list_models(self):
        # Try Ollama API first (localhost:11434)
        # Fallback: parse ~/.ollama/models/manifests/
    
    def get_model_gguf_path(self, name):
        # 1. Find manifest: ~/.ollama/models/manifests/.../name/tag
        # 2. Parse JSON for digest
        # 3. Return blob path: ~/.ollama/models/blobs/sha256-xxx
```

**Manifest Example**:
```json
{
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256:aeda25e63ebd...",
      "size": 3338792448
    }
  ]
}
```

### 6. GGUF Checker (gguf_check.py)

```python
class GGUFParser:
    def parse(self):
        # Read GGUF header (magic, version)
        # Parse key-value pairs
        # Check for required keys per architecture
        
CRITICAL_KEYS = {
    "gemma3": ["gemma3.attention.layer_norm_rms_epsilon"],
    "qwen35": ["qwen35.attention.layer_norm_rms_epsilon"],
    ...
}
```

## Data Flow: `moxing ollama serve gemma3:4b`

```
1. cli.py: ollama_serve("gemma3:4b")
   |
2. ollama.py: OllamaClient.get_model_gguf_path("gemma3:4b")
   |   ├── Find manifest: ~/.ollama/models/manifests/.../gemma3/4b
   |   ├── Parse digest: "sha256:7cd4618c..."
   |   └── Return: ~/.ollama/models/blobs/sha256-7cd4618c...
   |
3. gguf_check.py: diagnose_gguf(blob_path)
   |   ├── Parse GGUF header
   |   ├── Check architecture: "gemma3"
   |   ├── Check for critical keys
   |   └── Return: compatibility report
   |
4. cli.py: Display compatibility warnings (if any)
   |
5. device.py: DeviceDetector.get_best_device(model_size)
   |   ├── Detect GPUs (Metal/CUDA/Vulkan)
   |   └── Return: optimal config
   |
6. server.py: LlamaServer(gguf_path, ...).start()
   |   ├── Get binary: ~/.cache/moxing/binaries/darwin-arm64/llama-server
   |   ├── Build args: [-m, gguf_path, --port, 8080, ...]
   |   └── subprocess.Popen(...)
   |
7. HTTP server running on port 8080
```

## File Structure

```
moxing/
├── __init__.py          # Public API exports
├── cli.py               # Typer CLI commands
├── server.py            # LlamaServer class
├── mlx_server.py        # MLXServer class  
├── binaries.py          # Binary download/management
├── device.py            # GPU detection
├── models.py            # HuggingFace/ModelScope download
├── ollama.py            # Ollama integration
├── gguf_check.py        # GGUF compatibility check
├── runner.py            # AutoRunner helper
├── client.py            # OpenAI client wrapper
└── benchmark.py         # Performance testing
```

## Key Design Decisions

### 1. Why Multiple Backends?

- **llama.cpp**: Best performance for GGUF, cross-platform
- **MLX**: Supports latest models that llama.cpp doesn't yet support
- **Ollama API**: Fallback for when neither works

### 2. Why Read Ollama Blobs Directly?

- Ollama's runtime has overhead
- llama.cpp is often faster
- User already has the GGUF files

### 3. Why Embed MLX Server Script?

- Avoid external dependencies
- Simple HTTP server is enough
- Easy to customize

### 4. Why GGUF Compatibility Check?

- llama.cpp updates frequently
- New model architectures appear
- Give users helpful error messages

## Limitations

1. Not all GGUF files work: Depends on llama.cpp version
2. MLX only on macOS: Apple Silicon specific
3. No streaming in MLX backend: Basic implementation
4. Ollama must be installed: For listing models from API

## Future Improvements

1. Auto-download compatible GGUF when incompatible
2. Support more backends (ONNX Runtime, TensorRT)
3. Better streaming support
4. Windows/Linux MLX alternatives