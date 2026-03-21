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
# Build all platform-backend wheels
python scripts/build_platform_wheels.py

# Build specific wheel(s)
python scripts/build_platform_wheels.py --platform linux-x64-vulkan
python scripts/build_platform_wheels.py --platform windows-x64-cuda,darwin-arm64-metal

# Download binaries first, then build
python scripts/build_platform_wheels.py --download

# List available wheels
python scripts/build_platform_wheels.py --list
```

### Wheel Sizes

| Wheel | Size |
|-------|------|
| linux-x64-cpu | 18 MB |
| linux-x64-vulkan | 34 MB |
| linux-x64-rocm | 133 MB |
| windows-x64-cpu | 17 MB |
| windows-x64-cuda | 379 MB |
| windows-x64-vulkan | 33 MB |
| darwin-arm64-metal | 14 MB |

### CLI Commands (for testing)

```bash
moxing devices
moxing download Tesslate/OmniCoder-9B-GGUF -q Q4_K_M
moxing serve ./model.gguf -p 8080
moxing bench ./model.gguf
moxing speed ./model.gguf
```

## Distribution

MoXing uses **one wheel per platform-backend** for minimal download sizes:

```bash
# Install specific wheel
pip install ./moxing-0.1.7-py3-none-linux_x64_vulkan.whl
pip install ./moxing-0.1.7-py3-none-windows_x64_cuda.whl
pip install ./moxing-0.1.7-py3-none-darwin_arm64_metal.whl
```

**Backend Availability**:
| Platform | CPU | CUDA | Vulkan | ROCm | Metal |
|----------|-----|------|--------|------|-------|
| Linux x64 | ✅ | ❌ system | ✅ | ✅ | - |
| Windows x64 | ✅ | ✅ bundled | ✅ | - | - |
| macOS ARM64 | ✅ | - | - | - | ✅ |

**Notes**:
- Linux CUDA requires system CUDA installation (not bundled)
- Windows CUDA includes runtime (~379MB)
- Each wheel contains only ONE backend's binaries

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

### Module Structure

Each module should start with a docstring:

```python
"""
Brief description of the module.

More detailed explanation if needed.
"""
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

### Data Classes

Use `@dataclass` for data structures:

```python
from dataclasses import dataclass, field

@dataclass
class ModelInfo:
    name: str
    repo: str
    filename: str
    size_bytes: int = 0
    local_path: Optional[Path] = None
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)
```

### Enums

Use `Enum` for fixed sets of values:

```python
from enum import Enum

class BackendType(Enum):
    CUDA = "cuda"
    VULKAN = "vulkan"
    ROCM = "rocm"
    METAL = "metal"
    CPU = "cpu"
```

### Error Handling

- Raise specific exceptions with helpful messages
- Use try/except with specific exception types
- Log warnings for non-critical issues

```python
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

try:
    result = subprocess.run([str(binary), "--list-devices"], ...)
except Exception as e:
    console.print(f"[yellow]Warning: Device detection failed: {e}[/yellow]")
```

### Console Output

Use `rich.Console` for all user-facing output:

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

console.print("[green]Success![/green]")
console.print("[red]Error: ...[/red]")
console.print("[yellow]Warning: ...[/yellow]")
console.print("[blue]Info: ...[/blue]")
```

### Naming Conventions

- **Classes**: PascalCase (`LlamaServer`, `DeviceDetector`)
- **Functions/Methods**: snake_case (`get_best_device`, `download_model`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL_DIR`, `HF_API`)
- **Private methods**: prefix with underscore (`_build_args`, `_detect_source`)
- **Module-level singletons**: prefix with underscore (`_binary_manager`)

### File Operations

Use `pathlib.Path` for all file operations:

```python
from pathlib import Path

model_path = Path(model).resolve()
if not model_path.exists():
    raise FileNotFoundError(...)

output.parent.mkdir(parents=True, exist_ok=True)
```

### Context Managers

Implement `__enter__` and `__exit__` for resources:

```python
class LlamaServer:
    def __enter__(self):
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
```

### CLI Commands

Use `typer` for CLI commands with rich formatting:

```python
import typer

@app.command()
def serve(
    model: str = typer.Argument(..., help="Model path"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
):
    """Start the llama.cpp server."""
```

### Docstrings

Use descriptive docstrings for public APIs:

```python
def get_best_device(self, model_size_gb: float = 0) -> DeviceConfig:
    """Get the best device configuration for the given model size.
    
    Args:
        model_size_gb: Size of the model in gigabytes.
        
    Returns:
        DeviceConfig with optimal settings.
    """
```

### Testing Patterns

When adding tests, follow pytest conventions:

```python
import pytest
from moxing.device import DeviceDetector, BackendType

class TestDeviceDetector:
    def test_detect_returns_list(self):
        detector = DeviceDetector()
        devices = detector.detect()
        assert isinstance(devices, list)
    
    def test_cpu_fallback(self, mocker):
        mocker.patch('subprocess.run', side_effect=Exception("error"))
        detector = DeviceDetector()
        devices = detector.detect()
        assert devices[0].backend == BackendType.CPU
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
    binaries.py      # Binary management
    bin/             # Pre-built binaries (one subdir per platform-backend)
        linux-x64-cpu/
        linux-x64-vulkan/
        linux-x64-rocm/
        windows-x64-cpu/
        windows-x64-cuda/
        windows-x64-vulkan/
        darwin-arm64-metal/

scripts/
    build_platform_wheels.py  # Build platform-specific wheels
    bundle_all_binaries.py    # Download all binaries
    download_binaries.sh      # Shell script for offline download
```

## Important Notes

- Python 3.8+ compatibility required
- Use `Optional[T]` instead of `T | None` for type hints
- All external dependencies must be in `pyproject.toml`
- Handle cross-platform differences (Windows, Linux, macOS)
- Use `sys.platform` checks for platform-specific code
- Each wheel contains ONE platform-backend combination
- Windows CUDA wheel includes CUDA runtime (~379MB)
- Linux CUDA requires system CUDA installation