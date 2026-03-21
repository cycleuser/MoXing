# AGENTS.md - Coding Agent Guidelines for MoXing

## Project Overview

MoXing (模型) is a Python wrapper for llama.cpp that provides an OpenAI API compatible LLM backend with automatic GPU detection and model downloading. It supports CPU, Vulkan, CUDA, ROCm, and Metal backends.

## Build/Lint/Test Commands

### Installation

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
pytest -v
pytest tests/test_device.py
pytest tests/test_device.py::TestDeviceDetector::test_detect
pytest -x --tb=short
```

### Building the Package

```bash
# Build wheel for PyPI (small, ~54 KB, downloads binaries on first use)
python scripts/build_platform_wheels.py

# Upload to PyPI
twine upload dist/*.whl
```

### CLI Commands (for testing)

```bash
moxing devices
moxing download Tesslate/OmniCoder-9B-GGUF -q Q4_K_M
moxing serve ./model.gguf -p 8080
moxing bench ./model.gguf
```

## Distribution

MoXing uses a **small wheel** (~54 KB) that downloads binaries on first use:

```bash
pip install moxing
moxing serve model.gguf  # Downloads binaries automatically
```

### Binary Download Sources

Binaries are downloaded from multiple sources (tried in order):
1. **ModelScope** (China mirror, fast in mainland China)
2. **GitHub** (global)

**Environment variables**:
```bash
MOXING_BINARY_SOURCE=github     # Force GitHub only
MOXING_BINARY_SOURCE=modelscope # Force ModelScope only
MOXING_BINARY_SOURCE=auto       # Try ModelScope first, then GitHub (default)
MOXING_BINARY_MIRROR=URL        # Custom mirror URL
```

### Backend Availability

| Platform | CPU | CUDA | Vulkan | ROCm | Metal |
|----------|-----|------|--------|------|-------|
| Linux x64 | ✅ | ✅ | ✅ | ✅ | - |
| Windows x64 | ✅ | ✅ | ✅ | - | - |
| macOS ARM64 | ✅ | - | - | - | ✅ |

### Upload Binaries to ModelScope

```bash
pip install modelscope
python scripts/upload_to_modelscope.py --token YOUR_TOKEN
```

## Code Style Guidelines

### Imports

Group imports in this order, separated by blank lines:
1. Standard library (alphabetically)
2. Third-party packages (alphabetically)
3. Local imports (from moxing.*)

```python
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx
from rich.console import Console

from moxing.server import LlamaServer
from moxing.device import DeviceDetector
```

### Type Hints

Use type hints for all function signatures:

```python
def download(
    self,
    repo: str,
    filename: Optional[str] = None,
    source: str = "auto",
) -> Path:
```

Use `Optional[T]` for optional parameters, not `T | None` (Python 3.8 compatibility).

### Error Handling

- Raise specific exceptions with helpful messages
- Use try/except with specific exception types

```python
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")
```

### Console Output

Use `rich.Console` for all user-facing output:

```python
from rich.console import Console
console = Console()
console.print("[green]Success![/green]")
console.print("[red]Error: ...[/red]")
```

### Naming Conventions

- **Classes**: PascalCase (`LlamaServer`, `DeviceDetector`)
- **Functions/Methods**: snake_case (`get_best_device`, `download_model`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL_DIR`)
- **Private methods**: prefix with underscore (`_build_args`)

### CLI Commands

Use `typer` for CLI commands:

```python
import typer

@app.command()
def serve(
    model: str = typer.Argument(..., help="Model path"),
    port: int = typer.Option(8080, "-p", "--port"),
):
    """Start the llama.cpp server."""
```

## Project Structure

```
moxing/
    __init__.py      # Public API exports
    cli.py           # Typer CLI commands
    client.py        # OpenAI-compatible client
    server.py        # LlamaServer management
    device.py        # GPU detection and configuration
    models.py        # Model downloading (HF, ModelScope)
    runner.py        # AutoRunner for easy usage
    benchmark.py     # Performance benchmarking
    binaries.py      # Binary management (multi-source download)
    bin/             # Pre-built binaries (for development)
        linux-x64-cpu/
        linux-x64-cuda/
        linux-x64-vulkan/
        linux-x64-rocm/
        windows-x64-cpu/
        windows-x64-cuda/
        windows-x64-vulkan/
        darwin-arm64-metal/

scripts/
    build_platform_wheels.py  # Build wheel for PyPI
    upload_to_modelscope.py   # Upload binaries to ModelScope
```

## Important Notes

- Python 3.8+ compatibility required
- Use `Optional[T]` instead of `T | None` for type hints
- All external dependencies must be in `pyproject.toml`
- Wheel is small (~54 KB), binaries download on first use
- ModelScope mirror for fast downloads in China
- Linux CUDA binaries require system CUDA installation