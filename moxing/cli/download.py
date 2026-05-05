from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

_OPT_OUTPUT_DIR = typer.Option(None, "-o", "--output", help="Output directory")

console = Console()


def download(
    model: str = typer.Argument(..., help="Model name (e.g., llama-3.2-3b) or repo (user/model)"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization type"),
    source: str = typer.Option(
        "modelscope",
        "-s",
        "--source",
        help="Model source (huggingface/modelscope/auto, default: modelscope)",
    ),
    output: Optional[Path] = _OPT_OUTPUT_DIR,
    list_files: bool = typer.Option(False, "-l", "--list", help="List available files"),
):
    """Download a model from ModelScope (default) or HuggingFace.

    Use --source huggingface to download from HuggingFace instead.
    """
    from moxing.models import ModelDownloader, ModelRegistry

    downloader = ModelDownloader(output)

    registry_info = ModelRegistry.get_model_info(model, source)

    if registry_info:
        repo = registry_info["repo"]
        console.print(f"[blue]Found in registry: {registry_info['description']}[/blue]")
    else:
        repo = model

    if list_files:
        files = downloader.list_files(repo, source)
        if not files:
            console.print("[red]No GGUF files found[/red]")
            return

        table = Table(title=f"Available files in {repo}")
        table.add_column("Filename", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("Quant", style="green")

        for filename, size in files:
            size_str = f"{size / (1024**3):.2f} GB" if size > 0 else "unknown"
            quant_str = downloader._extract_quantization(filename)
            table.add_row(filename, size_str, quant_str)

        console.print(table)
        return

    try:
        path = downloader.download(repo, None if quant == "auto" else f"*{quant}*", source, output)
        console.print(f"[green]Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1) from e


def models(
    local: bool = typer.Option(False, "-l", "--local", help="Show local models only"),
    search: Optional[str] = typer.Option(None, "--search", help="Search for models"),
):
    """List available models."""
    from moxing.models import ModelDownloader
    from moxing.runner import AutoRunner

    if local:
        runner = AutoRunner()
        models_list = runner.list_local_models()

        if not models_list:
            console.print("[yellow]No local models found[/yellow]")
            return

        table = Table(title="Local Models")
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Quant", style="magenta")

        for m in models_list:
            local_path_str = str(m.local_path.parent) if m.local_path else "unknown"
            table.add_row(m.name, local_path_str, f"{m.size_gb:.2f} GB", m.quantization)

        console.print(table)
    elif search:
        downloader = ModelDownloader()
        results = downloader.search(search)

        if not results:
            console.print(f"[yellow]No models found for '{search}'[/yellow]")
            return

        table = Table(title=f"Search Results for '{search}'")
        table.add_column("Repo", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Source", style="magenta")

        for m in results[:20]:
            table.add_row(
                m.repo,
                m.filename[:50] + "..." if len(m.filename) > 50 else m.filename,
                f"{m.size_gb:.2f} GB",
                m.source,
            )

        console.print(table)
    else:
        runner = AutoRunner()
        runner.list_available_models()


def register(app: typer.Typer):
    app.command()(download)
    app.command()(models)
