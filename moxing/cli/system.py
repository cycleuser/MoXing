import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_OPT_OUTPUT_BIN = typer.Option(None, "-o", "--output", help="Output directory for binaries")

console = Console()

BACKEND_DEPENDENCIES = {
    "cpu": {
        "apt": [
            "build-essential",
            "cmake",
            "git",
            "libssl-dev",
        ],
        "description": "CPU only (no GPU)",
    },
    "cuda": {
        "apt": [
            "build-essential",
            "cmake",
            "git",
            "libssl-dev",
        ],
        "extra": "NVIDIA CUDA Toolkit (nvidia-cuda-toolkit or from https://developer.nvidia.com/cuda-downloads)",
        "description": "NVIDIA GPU via CUDA",
    },
    "vulkan": {
        "apt": [
            "build-essential",
            "cmake",
            "git",
            "libssl-dev",
            "libvulkan-dev",
            "glslang-dev",
            "glslang-tools",
            "libshaderc-dev",
            "pkg-config",
        ],
        "description": "Cross-vendor GPU via Vulkan",
    },
    "rocm": {
        "apt": [
            "build-essential",
            "cmake",
            "git",
            "libssl-dev",
        ],
        "extra": "AMD ROCm (from https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)",
        "description": "AMD GPU via ROCm/HIP",
    },
    "metal": {
        "apt": [],
        "brew": ["cmake", "git"],
        "description": "Apple Metal (macOS only)",
    },
    "sycl": {
        "apt": [
            "build-essential",
            "cmake",
            "git",
            "libssl-dev",
        ],
        "extra": "Intel oneAPI Base Toolkit (from https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)",
        "description": "Intel GPU via SYCL",
    },
}

BACKEND_CMAKE_FLAGS = {
    "cpu": [],
    "cuda": ["-DGGML_CUDA=ON"],
    "vulkan": ["-DGGML_VULKAN=ON"],
    "rocm": ["-DGGML_HIP=ON"],
    "metal": ["-DGGML_METAL=ON"],
    "sycl": ["-DGGML_SYCL=ON", "-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"],
}

LINUX_AMDGPU_TARGETS = (
    "gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1103"
)

ESSENTIAL_BINARIES_BUILD = [
    "llama-server",
    "llama-cli",
    "llama-mtmd-cli",
    "llama-bench",
    "llama-quantize",
]


def _check_command_exists(cmd: str) -> bool:
    try:
        result = subprocess.run(["which", cmd], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def _check_apt_package_installed(pkg: str) -> bool:
    try:
        result = subprocess.run(["dpkg", "-s", pkg], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def _check_cuda_available() -> bool:
    return _check_command_exists("nvcc") or _check_command_exists("nvidia-smi")


def _check_vulkan_available() -> bool:
    if _check_command_exists("vulkaninfo"):
        return True
    vulkan_icd = Path("/usr/share/vulkan/icd.d")
    return vulkan_icd.exists() and any(vulkan_icd.iterdir())


def _check_rocm_available() -> bool:
    for cmd in ["rocm-smi", "rocminfo", "hipconfig"]:
        if _check_command_exists(cmd):
            return True
    rocm_path = Path("/opt/rocm")
    return rocm_path.exists()


def _detect_gpu_arch(backend: str) -> Optional[str]:
    if backend == "cuda":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                caps = result.stdout.decode().strip().split("\n")
                archs = set()
                for cap in caps:
                    cap = cap.strip().replace(".", "")
                    if cap.isdigit() and len(cap) >= 2:
                        archs.add(cap[:3])
                if archs:
                    return ";".join(sorted(archs))
        except Exception:
            pass
        return None

    if backend == "rocm":
        try:
            result = subprocess.run(["rocminfo"], capture_output=True, timeout=10)
            if result.returncode == 0:
                archs = set()
                for line in result.stdout.decode().split("\n"):
                    m = re.search(r"gfx(\d+)", line)
                    if m:
                        name = m.group()
                        if "generic" not in line.lower():
                            archs.add(name)
                if archs:
                    return ";".join(sorted(archs))
        except Exception:
            pass
        return LINUX_AMDGPU_TARGETS

    return None


def _check_build_prerequisites(backend: str) -> List[str]:
    missing: List[str] = []
    os_name = sys.platform

    if os_name == "darwin":
        if not _check_command_exists("cmake"):
            missing.append("cmake (install via: brew install cmake)")
        if backend != "metal" and backend != "cpu":
            missing.append(f"Backend '{backend}' not supported on macOS")
        return missing

    if not _check_command_exists("cmake"):
        missing.append("cmake (install via: sudo apt install cmake)")

    if (
        not _check_command_exists("g++")
        and not _check_command_exists("cc1plus")
        and not _check_command_exists("c++")
    ):
        missing.append("g++ (install via: sudo apt install build-essential)")

    if not _check_command_exists("git"):
        missing.append("git (install via: sudo apt install git)")

    if backend == "cuda":
        if not _check_cuda_available():
            missing.append(
                "NVIDIA CUDA Toolkit not found. Install via:\n"
                "  sudo apt install nvidia-cuda-toolkit\n"
                "  Or download from: https://developer.nvidia.com/cuda-downloads"
            )

    elif backend == "vulkan":
        for pkg in ["libvulkan-dev", "libspirv-dev", "glslang-tools", "shaderc", "pkg-config"]:
            if not _check_apt_package_installed(pkg):
                missing.append(f"{pkg} (install via: sudo apt install {pkg})")

    elif backend == "rocm":
        if not _check_rocm_available():
            missing.append(
                "AMD ROCm not found. Install from:\n"
                "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/\n"
                "  Or: sudo amdgpu-install --usecase=rocm"
            )

    elif backend == "sycl" and not _check_command_exists("icx"):
        missing.append(
            "Intel oneAPI not found. Install from:\n"
            "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html\n"
            "  Then: source /opt/intel/oneapi/setvars.sh"
        )

    return missing


def _get_install_hint(backend: str) -> str:
    os_name = sys.platform

    if os_name == "darwin":
        return "brew install cmake git"

    if backend == "cpu":
        return "sudo apt install build-essential cmake git libssl-dev"

    if backend == "cuda":
        return (
            "sudo apt install build-essential cmake git libssl-dev\n"
            "# Then install CUDA Toolkit:\n"
            "sudo apt install nvidia-cuda-toolkit\n"
            "# OR download from: https://developer.nvidia.com/cuda-downloads"
        )

    if backend == "vulkan":
        return (
            "sudo apt install build-essential cmake git libssl-dev \\\n"
            "  libvulkan-dev libspirv-dev glslang-tools shaderc pkg-config"
        )

    if backend == "rocm":
        return (
            "sudo apt install build-essential cmake git libssl-dev\n"
            "# Then install ROCm:\n"
            "sudo amdgpu-install --usecase=rocm\n"
            "# OR download from: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
        )

    if backend == "sycl":
        return (
            "sudo apt install build-essential cmake git libssl-dev\n"
            "# Then install Intel oneAPI Base Toolkit:\n"
            "# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html\n"
            "source /opt/intel/oneapi/setvars.sh"
        )

    return "sudo apt install build-essential cmake git libssl-dev"


def _build_cmake_args(
    build_dir: Path,
    backend: str,
    gpu_arch: Optional[str] = None,
) -> List[str]:
    args = [
        "cmake",
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=OFF",
    ]

    args.extend(BACKEND_CMAKE_FLAGS.get(backend, []))

    if backend == "rocm":
        arch = gpu_arch or LINUX_AMDGPU_TARGETS
        args.append(f"-DAMDGPU_TARGETS={arch}")
        args.append(f"-DCMAKE_HIP_ARCHITECTURES={arch}")
        rocm_path = Path("/opt/rocm")
        if rocm_path.exists():
            args.append(f"-DCMAKE_PREFIX_PATH={rocm_path}")
        hipconfig = shutil.which("hipconfig")
        if hipconfig:
            hip_path = subprocess.run([hipconfig, "-R"], capture_output=True, text=True, timeout=10)
            if hip_path.returncode == 0:
                hip_dir = hip_path.stdout.strip()
                if hip_dir:
                    args.append(f"-DHIP_PATH={hip_dir}")

    if backend == "cuda" and gpu_arch:
        args.append(f"-DCMAKE_CUDA_ARCHITECTURES={gpu_arch.replace(';', ';')}")

    return args


def _copy_build_outputs(build_dir: Path, output_dir: Path, backend: str) -> List[str]:
    copied: List[str] = []
    bin_dir = build_dir / "bin"
    strip = shutil.which("strip")

    if bin_dir.exists():
        for exe in bin_dir.iterdir():
            if exe.is_file():
                name = exe.name
                if any(name.startswith(b) for b in ESSENTIAL_BINARIES_BUILD):
                    shutil.copy2(exe, output_dir / name)
                    if strip:
                        subprocess.run([strip, str(output_dir / name)], capture_output=True)
                    os.chmod(output_dir / name, 0o755)
                    copied.append(name)

    share_libs: List[Path] = []
    if backend in ("cuda", "rocm") and sys.platform == "linux":
        share_libs = list(build_dir.glob("*.so*")) + list(build_dir.glob("lib/*.so*"))
    elif backend == "vulkan" and sys.platform == "linux":
        share_libs = list(build_dir.glob("*.so*"))
    elif sys.platform == "darwin":
        share_libs = list(build_dir.glob("*.dylib"))
    elif sys.platform == "win32":
        share_libs = list(build_dir.glob("*.dll"))

    for lib in share_libs:
        if lib.is_file():
            shutil.copy2(lib, output_dir / lib.name)
            copied.append(lib.name)

    return sorted(set(copied))


def build_binary(
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="GPU backend: cpu, cuda, vulkan, rocm, metal, sycl"
    ),
    jobs: int = typer.Option(
        lambda: os.cpu_count() or 8, "-j", "--jobs", help="Parallel build jobs"
    ),
    output: Optional[Path] = _OPT_OUTPUT_BIN,
    repo: Optional[str] = typer.Option(
        None, "--repo", help="llama.cpp repo URL (default: official)"
    ),
    branch: Optional[str] = typer.Option(None, "--branch", help="llama.cpp branch or tag"),
    skip_checks: bool = typer.Option(False, "--skip-checks", help="Skip dependency checks"),
):
    """Build llama.cpp binaries from source.

    Automatically detects GPU backend if not specified. Checks build
    dependencies and provides installation hints for each backend.

    Each backend requires different system packages:

    \b
    CPU:
        sudo apt install build-essential cmake git libssl-dev

    \b
    CUDA (NVIDIA GPU):
        sudo apt install build-essential cmake git libssl-dev nvidia-cuda-toolkit

    \b
    Vulkan (cross-vendor GPU):
        sudo apt install build-essential cmake git libssl-dev \\
            libvulkan-dev libspirv-dev glslang-tools shaderc pkg-config

    \b
    ROCm (AMD GPU):
        sudo apt install build-essential cmake git libssl-dev
        # Plus AMD ROCm from https://rocm.docs.amd.com

    \b
    Metal (macOS, enabled by default):
        brew install cmake git

    Examples:
        moxing build                          # Auto-detect backend
        moxing build -b cuda                 # Build for CUDA
        moxing build -b vulkan               # Build for Vulkan
        moxing build -b rocm -j 16           # Build for ROCm with 16 jobs
        moxing build -b cuda -o ./bins       # Build and copy to output dir
        moxing build --skip-checks -b cpu   # Skip dependency checks
    """
    if backend == "auto":
        if sys.platform == "darwin":
            backend = "metal"
        elif _check_cuda_available():
            backend = "cuda"
        elif _check_rocm_available():
            backend = "rocm"
        elif _check_vulkan_available():
            backend = "vulkan"
        else:
            backend = "cpu"
        console.print(f"[blue]Auto-detected backend: {backend}[/blue]")

    if backend == "metal" and sys.platform != "darwin":
        console.print("[red]Metal backend is only available on macOS[/red]")
        raise typer.Exit(1)

    if backend in BACKEND_DEPENDENCIES:
        dep_info = BACKEND_DEPENDENCIES[backend]
        console.print(f"\n[bold]Building for {dep_info['description']} ({backend})[/bold]\n")

    if not skip_checks:
        missing = _check_build_prerequisites(backend)
        if missing:
            console.print("[red]Missing build dependencies:[/red]\n")
            for m in missing:
                console.print(f"  [red]✗[/red] {m}")
            console.print("\n[yellow]Install with:[/yellow]")
            console.print(f"  {_get_install_hint(backend)}")
            console.print("\n[dim]Use --skip-checks to bypass dependency checks[/dim]")
            raise typer.Exit(1)
        else:
            console.print("[green]✓ All build dependencies found[/green]")

    gpu_arch = _detect_gpu_arch(backend)
    if gpu_arch and backend in ("cuda", "rocm"):
        console.print(f"[dim]Detected GPU architectures: {gpu_arch}[/dim]")

    llama_cpp_dir = Path(__file__).parent.parent.parent / "llama.cpp"

    if not (llama_cpp_dir / "CMakeLists.txt").exists():
        console.print("[yellow]llama.cpp source not found, cloning...[/yellow]")
        clone_url = repo or "https://github.com/ggml-org/llama.cpp.git"
        clone_cmd = ["git", "clone", "--depth", "1"]
        if branch:
            clone_cmd.extend(["--branch", branch])
        clone_cmd.extend([clone_url, str(llama_cpp_dir)])
        subprocess.run(clone_cmd, check=True)
        console.print("[green]llama.cpp cloned successfully[/green]")
    else:
        console.print(f"[dim]Using existing source: {llama_cpp_dir}[/dim]")

    build_dir = llama_cpp_dir / "build"
    cmake_args = _build_cmake_args(build_dir, backend, gpu_arch)

    console.print(f"\n[bold]CMake configure ({backend}):[/bold]")
    for arg in cmake_args[2:]:
        console.print(f"  [dim]{arg}[/dim]")
    console.print()

    result = subprocess.run(cmake_args, cwd=llama_cpp_dir)
    if result.returncode != 0:
        if backend == "cuda":
            console.print("\n[yellow]CUDA build failed. Common fixes:[/yellow]")
            console.print("  1. Install CUDA Toolkit: sudo apt install nvidia-cuda-toolkit")
            console.print("  2. Or download from: https://developer.nvidia.com/cuda-downloads")
            console.print("  3. Ensure nvcc is in PATH: which nvcc")
        elif backend == "vulkan":
            console.print("\n[yellow]Vulkan build failed. Common fixes:[/yellow]")
            console.print(
                "  sudo apt install libvulkan-dev libspirv-dev glslang-tools shaderc pkg-config"
            )
        elif backend == "rocm":
            console.print("\n[yellow]ROCm build failed. Common fixes:[/yellow]")
            console.print(
                "  1. Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
            )
            console.print("  2. Ensure hipconfig is in PATH: which hipconfig")
            console.print("  3. Try specifying GPU arch: moxing build -b rocm")
        raise typer.Exit(1)

    actual_jobs = jobs if isinstance(jobs, int) else (os.cpu_count() or 8)
    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(actual_jobs)]

    console.print(f"\n[bold]Building with {actual_jobs} jobs...[/bold]")
    result = subprocess.run(build_cmd, cwd=llama_cpp_dir)
    if result.returncode != 0:
        console.print("[red]Build failed[/red]")
        raise typer.Exit(1)

    console.print("\n[green bold]✓ Build complete![/green bold]")

    bin_dir = build_dir / "bin"
    if bin_dir.exists():
        binaries = list(bin_dir.glob("llama-*"))
        console.print(f"\n[bold]Built binaries ({len(binaries)}):[/bold]")
        for b in sorted(binaries):
            size_mb = b.stat().st_size / (1024 * 1024)
            console.print(f"  [green]✓[/green] {b.name} ({size_mb:.1f} MB)")

    if output:
        output.mkdir(parents=True, exist_ok=True)
        copied = _copy_build_outputs(build_dir, output, backend)
        if copied:
            console.print(f"\n[green]Copied {len(copied)} files to: {output}[/green]")
            for name in copied:
                console.print(f"  [dim]{name}[/dim]")
        else:
            console.print("[yellow]No binaries found to copy[/yellow]")
    else:
        from moxing.binaries import CACHE_DIR, PlatformDetector

        platform_name = PlatformDetector.get_platform_name()
        cache_dir = CACHE_DIR / f"{platform_name}-{backend}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_build_outputs(build_dir, cache_dir, backend)
        (cache_dir / "VERSION").write_text(f"local-build-{backend}\n{backend}\n")
        if copied:
            console.print(f"\n[green]Installed {len(copied)} files to cache:[/green]")
            console.print(f"  [dim]{cache_dir}[/dim]")
            for name in copied:
                console.print(f"  [green]✓[/green] {name}")
        else:
            console.print("[yellow]No binaries found to copy[/yellow]")


def download_binaries(
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="GPU backend (auto, vulkan, cuda, metal, cpu, all)"
    ),
    force: bool = typer.Option(False, "-f", "--force", help="Force re-download"),
    list_available: bool = typer.Option(False, "-l", "--list", help="List available backends"),
):
    """Download pre-built llama.cpp binaries.

    Use --backend all to download binaries for all supported backends.
    """
    from moxing.binaries import get_binary_manager, list_available_backends

    if list_available:
        console.print("\n[bold]Available backends for this platform:[/bold]")
        available = list_available_backends()
        for b, has_bin in available.items():
            status = "[green]✓ installed[/green]" if has_bin else "[dim]not installed[/dim]"
            console.print(f"  {b}: {status}")
        console.print("\n[dim]Use --backend <name> to download specific backend[/dim]")
        console.print("[dim]Use --backend all to download all backends[/dim]")
        return

    if backend == "all":
        console.print("[blue]Downloading binaries for all supported backends...[/blue]")
        manager = get_binary_manager()
        results = manager.download_all_binaries(force=force)

        console.print("\n[bold]Download Results:[/bold]")
        for b, success in results.items():
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            console.print(f"  {status} {b}")

        console.print(f"\n[green]Binaries installed to: {manager.cache_dir}[/green]")
        return

    manager = get_binary_manager(backend)

    console.print(
        f"[blue]Downloading binaries for {manager.platform_name} ({manager.backend})...[/blue]"
    )

    try:
        manager.download_binaries(force=force)

        binaries = manager.list_cached_binaries()
        console.print("\n[green]Installed binaries:[/green]")
        for b in binaries[:10]:
            console.print(f"  - {b}")
        if len(binaries) > 10:
            console.print(f"  ... and {len(binaries) - 10} more")

        console.print(f"\n[green]Binaries installed to: {manager.get_cache_dir()}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1) from e


def update_binaries_cmd(
    force: bool = typer.Option(False, "-f", "--force", help="Force update even if up to date"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Auto-confirm update"),
    backend: str = typer.Option(
        "auto", "-b", "--backend", help="Backend to download: auto/cuda/rocm/vulkan/cpu/metal"
    ),
    all_backends: bool = typer.Option(
        False, "--all", help="Download binaries for all supported backends"
    ),
):
    """Update llama.cpp binaries to the latest version.

    Checks for newer binary versions and downloads them if available.
    Supports automatic update detection and one-click updates.

    Examples:
        moxing update-binaries                  # Check and update auto-detected backend
        moxing update-binaries -b rocm          # Update ROCm binaries specifically
        moxing update-binaries --all            # Download all backend binaries
        moxing update-binaries -f               # Force re-download
        moxing update-binaries -f -y -b vulkan  # Force update vulkan without confirmation

    Note: Uses bundled binaries as fallback if update fails.
    """
    from moxing.binaries import (
        BACKEND_PRIORITY,
        PlatformDetector,
        clear_skip_update,
        get_binary_manager,
    )

    console.print("[blue]Checking for binary updates...[/blue]\n")

    if all_backends:
        os_name = PlatformDetector.get_os()
        supported = BACKEND_PRIORITY.get(os_name, ["cpu"])
        console.print(f"[blue]Downloading all backends: {', '.join(supported)}[/blue]\n")

        results = {}
        for b in supported:
            console.print(f"[blue]--- Backend: {b} ---[/blue]")
            try:
                manager = get_binary_manager(backend=b)
                clear_skip_update()
                if force:
                    manager.download_binaries(force=True, quiet=False, check_updates=False)
                else:
                    manager.download_binaries(force=False, quiet=False, check_updates=True)
                results[b] = "OK"
                console.print(f"[green]✓ {b} complete[/green]\n")
            except Exception as e:
                results[b] = f"FAILED: {e}"
                console.print(f"[red]✗ {b} failed: {e}[/red]\n")

        console.print("\n[bold]Summary:[/bold]")
        for b, status in results.items():
            color = "green" if status == "OK" else "red"
            console.print(f"  [{color}]{b}: {status}[/{color}]")
        return

    clear_skip_update()

    manager = get_binary_manager(backend=backend)
    has_update, current, latest = manager.check_for_updates()

    if not current:
        console.print("[yellow]Could not determine current version[/yellow]")
        console.print("[blue]Proceeding with download...[/blue]\n")
        has_update = True

    if has_update or force:
        if latest:
            if current:
                console.print("[green bold]New version available![/green bold]")
                console.print(f"  Current:  {current}")
                console.print(f"  Latest:   {latest}")
            else:
                console.print(f"[green]Latest version: {latest}[/green]")
        else:
            console.print("[green]Update available[/green]")

        if not yes:
            from rich.prompt import Confirm

            console.print()
            if not Confirm.ask("Download update?", default=True):
                console.print("[yellow]Update cancelled[/yellow]")
                from moxing.binaries import skip_update_forever

                skip_update_forever()
                return

        console.print("\n[blue]Downloading update...[/blue]\n")

        try:
            manager.download_binaries(force=True, quiet=False, check_updates=False)
            console.print("\n[green bold]✓ Update complete![/green bold]")
            console.print(f"[dim]Installed to: {manager.cache_dir}[/dim]")
            console.print("\n[dim]Tip: Restart any running servers to use new binaries[/dim]")
        except Exception as e:
            console.print(f"[red]Update failed: {e}[/red]")
            console.print("[yellow]Falling back to bundled binaries[/yellow]")
    else:
        console.print("[green]✓ Already up to date[/green]")
        console.print(f"[dim]Current version: {current}[/dim]")
        console.print("\n[dim]Tip: Use --force to re-download binaries[/dim]")


def clear_cache(
    model: Optional[str] = typer.Argument(None, help="Specific model to remove"),
    binaries: bool = typer.Option(False, "--binaries", help="Clear binary cache"),
):
    """Clear model and/or binary cache."""
    if binaries or model is None:
        from moxing.binaries import get_binary_manager

        manager = get_binary_manager()
        manager.clear_cache()
        console.print("[green]Binary cache cleared[/green]")

    if model or not binaries:
        from moxing.models import ModelDownloader

        downloader = ModelDownloader()
        downloader.clear_cache(model)
        console.print("[green]Model cache cleared[/green]")
    console.print("[green]Cache cleared[/green]")


def diagnose(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    install: bool = typer.Option(False, "--install", "-i", help="Auto-install after diagnosis"),
):
    """Diagnose system and show installation recommendations."""
    import subprocess

    script_path = Path(__file__).parent.parent.parent / "scripts" / "detect_and_install.py"

    if not script_path.exists():
        console.print("[yellow]Running built-in diagnostics...[/yellow]")

        from moxing.device import BackendType, DeviceDetector

        detector = DeviceDetector()
        devices = detector.detect()

        if json_output:
            import json

            data = {
                "platform": sys.platform,
                "python_version": platform.python_version(),
                "devices": [
                    {
                        "index": d.index,
                        "name": d.name,
                        "backend": d.backend.value,
                        "memory_mb": d.memory_mb,
                        "vendor": d.vendor,
                    }
                    for d in devices
                ],
                "recommended_backend": min(
                    [d.backend for d in devices if d.backend != BackendType.CPU],
                    default=BackendType.CPU,
                ).value,
            }
            print(json.dumps(data, indent=2))
        else:
            console.print(
                Panel(
                    f"[cyan]Platform:[/cyan] {sys.platform}\n"
                    f"[cyan]Python:[/cyan] {platform.python_version()}\n"
                    f"[cyan]Devices:[/cyan] {len(devices)} found",
                    title="System Diagnostics",
                )
            )

            if devices:
                table = Table(title="Detected Devices")
                table.add_column("Index", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Backend", style="magenta")
                table.add_column("Memory", style="yellow")

                for d in devices:
                    mem = f"{d.memory_gb:.1f}GB" if d.memory_mb > 0 else "N/A"
                    table.add_row(str(d.index), d.name, d.backend.value, mem)

                console.print(table)

            if install:
                console.print("\n[blue]Starting automatic installation...[/blue]")
                import subprocess

                subprocess.run([sys.executable, "-m", "pip", "install", "moxing"])
        return

    cmd = [sys.executable, str(script_path)]
    if json_output:
        cmd.append("--json")
    if install:
        cmd.append("--install")

    subprocess.run(cmd)


def extract_mmproj_cmd(
    model: str = typer.Argument(..., help="Ollama model name or GGUF file path"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output mmproj file path"),
    model_type: Optional[str] = typer.Option(
        None, "-m", "--model-type", help="Model type (auto-detect if not specified)"
    ),
):
    """Extract multimodal projector from Ollama model."""

    from moxing.gguf_tools.extract_mmproj import extract_mmproj

    if Path(model).exists():
        model_path = model
    else:
        console.print(f"[blue]Finding Ollama model {model}...[/blue]")
        blob_dir = Path.home() / ".ollama" / "models" / "blobs"
        ollama_models = Path("/usr/share/ollama/.ollama/models/blobs")

        model_path = None
        for search_dir in [ollama_models, blob_dir]:
            matches = (
                list(search_dir.glob(f"sha256-{model}*"))
                if "-" in model
                else list(search_dir.glob("*"))
            )
            if matches:
                for match in matches:
                    if match.stat().st_size > 1e9:
                        model_path = str(match)
                        break

        if not model_path:
            console.print(f"[red]Error: Model not found: {model}[/red]")
            console.print("[yellow]Tip: Use 'ollama list' to see installed models[/yellow]")
            raise typer.Exit(1)

    if not output:
        output = f"mmproj-{Path(model).name}.gguf"

    console.print(f"[blue]Extracting mmproj from: {model_path}[/blue]")
    console.print(f"[blue]Output: {output}[/blue]\n")

    try:
        success = extract_mmproj(str(model_path), output, model_type)

        if success:
            console.print(f"\n[green]✓ Successfully extracted mmproj to {output}[/green]")
            console.print("\n[yellow]Usage example:[/yellow]")
            console.print(f"  llama-server -m {model_path} --mmproj {output} --port 8080")
        else:
            console.print("\n[yellow]Model may not have multimodal support[/yellow]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Failed to extract mmproj: {e}[/red]")
        raise typer.Exit(1) from e


def register(app: typer.Typer):
    app.command("build")(build_binary)
    app.command("download-binaries")(download_binaries)
    app.command("update-binaries")(update_binaries_cmd)
    app.command("clear-cache")(clear_cache)
    app.command("diagnose")(diagnose)
    app.command("extract-mmproj")(extract_mmproj_cmd)
