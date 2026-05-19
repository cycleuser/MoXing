import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_OPT_OUTPUT_BIN = typer.Option(None, "-o", "--output", help="Output directory for binaries")

console = Console()


def build_binary(
    backend: str = typer.Option(
        "vulkan", "-b", "--backend", help="GPU backend (vulkan, cuda, rocm, cpu)"
    ),
    jobs: int = typer.Option(8, "-j", "--jobs", help="Parallel jobs"),
    output: Optional[Path] = _OPT_OUTPUT_BIN,
    repo: Optional[str] = typer.Option(
        None, "--repo", help="llama.cpp repo URL (default: official)"
    ),
    branch: Optional[str] = typer.Option(None, "--branch", help="llama.cpp branch or tag"),
):
    """Build llama.cpp binaries from source.

    Clones llama.cpp automatically if not present. Supports all GPU backends.
    Uses the official llama.cpp repo by default.

    Examples:
        moxing build -b rocm -j 16                      # Build for ROCm
        moxing build -b vulkan                           # Build for Vulkan
        moxing build -b cuda                             # Build for CUDA
    """
    console.print(f"[blue]Building llama.cpp with {backend} backend...[/blue]")

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

    build_dir = llama_cpp_dir / "build"

    cmake_args = [
        "cmake",
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=OFF",
    ]

    if backend == "vulkan":
        cmake_args.append("-DGGML_VULKAN=ON")
    elif backend == "cuda":
        cmake_args.append("-DGGML_CUDA=ON")
    elif backend == "rocm":
        cmake_args.append("-DGGML_HIP=ON")
        cmake_args.append("-DAMDGPU_TARGETS=gfx1100;gfx1030;gfx900")
    elif backend == "metal":
        cmake_args.append("-DGGML_METAL=ON")

    console.print(f"[dim]Running: {' '.join(cmake_args)}[/dim]")
    subprocess.run(cmake_args, cwd=llama_cpp_dir, check=True)

    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(jobs)]
    console.print(f"[dim]Building with {jobs} jobs...[/dim]")
    subprocess.run(build_cmd, cwd=llama_cpp_dir, check=True)

    if output:
        output.mkdir(parents=True, exist_ok=True)
        bin_dir = build_dir / "bin"
        if bin_dir.exists():
            for exe in bin_dir.glob("llama-*.exe" if sys.platform == "win32" else "llama-*"):
                shutil.copy2(exe, output / exe.name)
        if sys.platform != "win32":
            for lib in build_dir.glob("*.so*"):
                shutil.copy2(lib, output / lib.name)
        console.print(f"[green]Binaries copied to: {output}[/green]")
    else:
        console.print(f"[green]Build complete! Binaries at: {build_dir / 'bin'}[/green]")


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
