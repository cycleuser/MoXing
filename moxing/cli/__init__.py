"""
CLI interface for moxing
"""

import platform
import sys

import typer
from rich.console import Console

console = Console()
app = typer.Typer(name="moxing", help="Python wrapper for llama.cpp server", add_completion=False)


def version_callback(value: bool):
    if value:
        from moxing import __version__
        from moxing.binaries import get_binary_manager

        console.print(f"[bold]MoXing[/bold] version: {__version__}")

        try:
            manager = get_binary_manager()
            binary_version = manager.get_installed_version()
            if binary_version:
                console.print(f"[bold]Binaries:[/bold] {binary_version} ({manager.backend})")
            else:
                console.print("[bold]Binaries:[/bold] not installed")
        except Exception:
            pass

        console.print(f"[bold]Python:[/bold] {sys.version.split()[0]}")
        console.print(f"[bold]Platform:[/bold] {platform.system()} {platform.machine()}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    pass


@app.command()
def version_cmd():
    """Show version information."""
    from moxing import __version__
    from moxing.binaries import PlatformDetector, get_binary_manager

    console.print(f"\n[bold cyan]MoXing[/bold cyan] version {__version__}")

    console.print("\n[bold]Platform:[/bold]")
    console.print(f"  OS: {PlatformDetector.get_os()}")
    console.print(f"  Arch: {PlatformDetector.get_arch()}")
    console.print(f"  Detected backend: {PlatformDetector.detect_backend()}")

    try:
        manager = get_binary_manager()
        binary_version = manager.get_installed_version()
        console.print("\n[bold]Binaries:[/bold]")
        if binary_version:
            console.print(f"  Version: {binary_version}")
            console.print(f"  Backend: {manager.backend}")
            console.print(f"  Path: {manager.get_cache_dir()}")
        else:
            console.print("  Not installed (will download on first use)")
            console.print(f"  Will use: {manager.backend}")
    except Exception as e:
        console.print(f"\n[yellow]Binary info unavailable: {e}[/yellow]")

    console.print(f"\n[bold]Python:[/bold] {sys.version.split(0)[0]}")
    console.print(f"[bold]Python path:[/bold] {sys.executable}")


from moxing.cli.benchmark import register as _register_benchmark  # noqa: E402
from moxing.cli.compress import compress_app  # noqa: E402
from moxing.cli.devices import register as _register_devices  # noqa: E402
from moxing.cli.download import register as _register_download  # noqa: E402
from moxing.cli.monitor import monitor_app  # noqa: E402
from moxing.cli.ollama_cmds import ollama_app  # noqa: E402
from moxing.cli.serve import register as _register_serve  # noqa: E402
from moxing.cli.system import register as _register_system  # noqa: E402
from moxing.cli.turboquant import turboquant_app  # noqa: E402

_register_serve(app)
_register_download(app)
_register_devices(app)
_register_benchmark(app)
_register_system(app)

app.add_typer(ollama_app, name="ollama")
app.add_typer(compress_app, name="compress")
app.add_typer(turboquant_app, name="turboquant")
app.add_typer(monitor_app, name="monitor")


if __name__ == "__main__":
    app()
