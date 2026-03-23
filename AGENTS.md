# AGENTS.md - Coding Agent Guidelines for MoXing

## Project Overview

MoXing (模型) is a Python wrapper for llama.cpp that provides an OpenAI API compatible LLM backend with automatic GPU detection and model downloading. It supports CPU, Vulkan, CUDA, ROCm, and Metal backends. Features include GGUF compression with transparent decompression, Ollama integration, and MLX backend support for macOS.

## Build/Lint/Test Commands

### Installation

```bash
pip install -e ".[dev]"        # Install with dev dependencies
pip install -e ".[all]"        # Install with all optional dependencies
```

### Running Tests

```bash
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest tests/test_device.py     # Run specific file
pytest tests/test_device.py::TestDeviceDetector::test_detect  # Run single test
pytest -x --tb=short            # Stop on first failure, short traceback
```

### Building the Package

```bash
python scripts/build_platform_wheels.py  # Build wheel (~54 KB)
twine upload dist/*.whl                   # Upload to PyPI
```

### CLI Commands (for testing)

```bash
moxing devices                              # List GPU devices
moxing download Tesslate/OmniCoder-9B-GGUF  # Download model
moxing serve ./model.gguf -p 8080           # Start server
moxing bench ./model.gguf                   # Benchmark model
moxing ollama list                          # List Ollama models
moxing compress pack model.gguf -a zstd     # Compress GGUF
moxing compress cache --size                # Check cache size
```

## Code Style Guidelines

### Imports

Group imports in order, separated by blank lines:
1. Standard library (alphabetically)
2. Third-party packages (alphabetically)
3. Local imports (from moxing.*)

```python
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.table import Table

from moxing.server import LlamaServer
from moxing.device import DeviceDetector
```

### Type Hints

Use type hints for all function signatures. Use `Optional[T]` for optional parameters (Python 3.8 compatibility), not `T | None`.

```python
def download(
    self,
    repo: str,
    filename: Optional[str] = None,
    source: str = "auto",
) -> Path:
```

Use `from typing import ...` for all typing imports.

### Error Handling

Raise specific exceptions with helpful messages. Use try/except with specific exception types.

```python
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

try:
    result = subprocess.run(cmd, capture_output=True, timeout=30)
except subprocess.TimeoutExpired:
    raise RuntimeError(f"Command timed out: {' '.join(cmd)}")
```

### Console Output

Use `rich.Console` for all user-facing output:

```python
from rich.console import Console
console = Console()
console.print("[green]Success![/green]")
console.print("[red]Error: ...[/red]")
console.print("[blue]Processing...[/blue]")
console.print("[yellow]Warning: ...[/yellow]")
```

Use `rich.table.Table` for tabular output.

### Naming Conventions

- **Classes**: PascalCase (`LlamaServer`, `DeviceDetector`, `MultiCompressor`)
- **Functions/Methods**: snake_case (`get_best_device`, `download_model`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL_DIR`, `GGUF_MAGIC`)
- **Private methods**: prefix with underscore (`_build_args`, `_search_hf`)
- **Module-level "constants" that are objects**: UPPER or lower based on usage

### CLI Commands

Use `typer` for CLI commands. Group related commands with `typer.Typer`:

```python
import typer

app = typer.Typer()
compress_app = typer.Typer(name="compress")
app.add_typer(compress_app, name="compress")

@app.command()
def serve(
    model: str = typer.Argument(..., help="Model path"),
    port: int = typer.Option(8080, "-p", "--port"),
):
    """Start the llama.cpp server."""
```

### Dataclasses

Use `@dataclass` for data containers:

```python
from dataclasses import dataclass

@dataclass
class Device:
    index: int
    name: str
    backend: BackendType
    memory_mb: int = 0
    
    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024
```

## Project Structure

```
moxing/
    __init__.py          # Public API exports
    cli.py               # Typer CLI commands
    client.py            # OpenAI-compatible client
    server.py            # LlamaServer management
    device.py            # GPU detection and configuration
    models.py            # Model downloading (HF, ModelScope)
    runner.py            # AutoRunner for easy usage
    benchmark.py         # Performance benchmarking
    binaries.py          # Binary management (multi-source download)
    gguf_compress.py     # GGUF compression/transparency
    gguf_check.py        # GGUF compatibility checking
    ollama.py            # Ollama integration
    mlx_server.py        # MLX backend (macOS)
    bin/                 # Pre-built binaries (development only)

scripts/
    build_platform_wheels.py
    upload_to_modelscope.py
```

## Key Patterns

### Transparent Decompression

When serving compressed files, decompress transparently:

```python
from moxing.gguf_compress import resolve_model_path, is_gguf_compressed

if model_path.exists() and is_gguf_compressed(model_path):
    resolved_path = resolve_model_path(model_path)
    # Use resolved_path for serving
```

### Binary Management

Binaries download on first use from ModelScope (China mirror) or GitHub:

```python
from moxing.binaries import get_binary_manager

manager = get_binary_manager()
if not manager.has_binaries():
    manager.download_binaries()
binary_path = manager.get_binary_path("llama-server")
```

## Important Notes

- Python 3.8+ compatibility required - use `Optional[T]` not `T | None`
- All external dependencies must be in `pyproject.toml`
- Wheel is small (~54 KB), binaries download on first use
- ModelScope mirror for fast downloads in China
- Linux CUDA binaries require system CUDA installation
- **NO COMMENTS in code** unless explicitly requested