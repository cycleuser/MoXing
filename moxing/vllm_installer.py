"""
vLLM installation and dependency management.

Handles installing vLLM with proper CUDA/ROCm support.
"""

import importlib.util
import logging
import subprocess
import sys
from typing import Optional, Tuple

from rich.console import Console

logger = logging.getLogger(__name__)

console = Console()


VLLM_MIN_VERSION = "0.8.0"


def is_vllm_installed() -> bool:
    """Check if vLLM is installed."""
    return importlib.util.find_spec("vllm") is not None


def get_vllm_version() -> Optional[str]:
    """Get installed vLLM version."""
    try:
        import vllm

        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return None


def check_vllm_compatibility() -> Tuple[bool, str]:
    """Check if vLLM is compatible with current system."""
    if not is_vllm_installed():
        return False, "vLLM is not installed"

    import torch

    if not torch.cuda.is_available():
        if sys.platform == "darwin":
            return False, "vLLM does not support macOS"
        return False, "No CUDA GPU detected (vLLM requires CUDA or ROCm)"

    cuda_version = torch.version.cuda
    if cuda_version:
        major_minor = ".".join(cuda_version.split(".")[:2])
        cuda_ver = float(major_minor)
        if cuda_ver < 12.0:
            return False, f"CUDA {cuda_ver} detected, vLLM requires CUDA 12.0+"

    return True, "vLLM is compatible"


def detect_cuda_version() -> Optional[str]:
    """Detect CUDA version from system."""
    try:
        import torch

        if torch.version.cuda:
            return torch.version.cuda
    except ImportError:
        pass

    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")
                    if len(parts) > 1:
                        return parts[1].strip().split(",")[0].strip()
    except Exception as e:
        logger.debug("CUDA version detection failed: %s", e, exc_info=True)
        pass

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        return parts[1].strip().split(" ")[0].strip()
    except Exception as e:
        logger.debug("GPU detection via nvidia-smi failed: %s", e, exc_info=True)
        pass

    return None


def detect_rocm_version() -> Optional[str]:
    """Detect ROCm version from system."""
    for cmd in ["rocminfo", "rocm-smi"]:
        try:
            result = subprocess.run(
                [cmd, "--version"] if cmd == "rocm-smi" else [cmd],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "ROCm" in line or "Version" in line:
                        return line.strip()
        except Exception as e:
            logger.debug("ROCm GPU detection via rocm-smi failed: %s", e, exc_info=True)
            pass
    return None


def install_vllm(
    cuda_version: Optional[str] = None,
    extra_index: Optional[str] = None,
    quiet: bool = False,
) -> bool:
    """Install vLLM with appropriate CUDA support.

    Args:
        cuda_version: CUDA version to use (auto-detected if None)
        extra_index: Extra pip index URL for CUDA-specific wheels
        quiet: Suppress output

    Returns:
        True if installation succeeded
    """
    if not quiet:
        console.print("[blue]Installing vLLM...[/blue]")

    cmd = [sys.executable, "-m", "pip", "install", "vllm"]

    if extra_index:
        cmd.extend(["--extra-index-url", extra_index])

    if not quiet:
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, capture_output=quiet, text=True, timeout=600)

        if result.returncode == 0:
            if not quiet:
                console.print("[green]vLLM installed successfully[/green]")
            return True
        else:
            if not quiet:
                console.print("[red]vLLM installation failed:[/red]")
                if result.stderr:
                    console.print(f"[dim]{result.stderr[-500:]}[/dim]")
            return False
    except subprocess.TimeoutExpired:
        if not quiet:
            console.print("[red]vLLM installation timed out (10 min)[/red]")
        return False
    except Exception as e:
        logger.debug("Operation failed: %s", e, exc_info=True)
        if not quiet:
            console.print(f"[red]vLLM installation error: {e}[/red]")
        return False


def install_vllm_rocm(quiet: bool = False) -> bool:
    """Install vLLM with ROCm support."""
    if not quiet:
        console.print("[blue]Installing vLLM with ROCm support...[/blue]")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "vllm",
        "--extra-index-url",
        "https://download.pytorch.org/whl/rocm6.2",
    ]

    try:
        result = subprocess.run(cmd, capture_output=quiet, text=True, timeout=600)

        if result.returncode == 0:
            if not quiet:
                console.print("[green]vLLM ROCm installed successfully[/green]")
            return True
        else:
            if not quiet:
                console.print("[red]vLLM ROCm installation failed[/red]")
            return False
    except Exception as e:
        logger.debug("Operation failed: %s", e, exc_info=True)
        if not quiet:
            console.print(f"[red]vLLM ROCm installation error: {e}[/red]")
        return False


def get_install_recommendation() -> str:
    """Get installation recommendation based on system."""
    has_cuda = detect_cuda_version() is not None
    has_rocm = detect_rocm_version() is not None

    if sys.platform == "darwin":
        return "vLLM does not support macOS. Use llama.cpp or MLX backend instead."

    if has_cuda:
        return "Install vLLM with CUDA: pip install vllm"

    if has_rocm:
        return "Install vLLM with ROCm: pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2"

    return "No GPU detected. vLLM requires CUDA or ROCm. Use llama.cpp CPU backend instead."


def ensure_vllm(quiet: bool = False) -> bool:
    """Ensure vLLM is installed, install if needed."""
    if is_vllm_installed():
        compatible, msg = check_vllm_compatibility()
        if compatible:
            if not quiet:
                version = get_vllm_version()
                console.print(f"[green]vLLM {version} is available[/green]")
            return True
        else:
            if not quiet:
                console.print(f"[yellow]vLLM installed but incompatible: {msg}[/yellow]")
            return False

    recommendation = get_install_recommendation()
    if not quiet:
        console.print(f"[yellow]vLLM not installed. {recommendation}[/yellow]")

    if "does not support" in recommendation or "No GPU detected" in recommendation:
        return False

    return install_vllm(quiet=quiet)
