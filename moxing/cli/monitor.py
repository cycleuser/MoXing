import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

monitor_app = typer.Typer(name="monitor", help="Real-time monitoring commands")


@monitor_app.command("start")
def monitor_start(
    host: str = typer.Option("127.0.0.1", "--host", help="Monitor server host"),
    port: int = typer.Option(9090, "-p", "--port", help="Monitor server port"),
    llama_port: int = typer.Option(8080, "-l", "--llama-port", help="llama.cpp server port"),
):
    """Start the web-based monitoring dashboard.

    Displays real-time metrics including:
    - GPU/CPU memory usage
    - Token generation speed
    - Request statistics
    - Slot status

    Example:
        # Terminal 1: Start llama.cpp server
        moxing serve model.gguf -p 8080

        # Terminal 2: Start monitor
        moxing monitor start --llama-port 8080

        # Open browser: http://127.0.0.1:9090
    """
    from moxing.monitor import start_monitor_server

    console.print(
        Panel(
            f"[bold cyan]MoXing Monitor Dashboard[/bold cyan]\n\n"
            f"[green]Monitor URL:[/green] http://{host}:{port}\n"
            f"[green]Server URL:[/green] http://127.0.0.1:{llama_port}\n\n"
            f"[yellow]Make sure llama.cpp server is running![/yellow]\n"
            f"[dim]Start with: moxing serve model.gguf -p {llama_port}[/dim]",
            title="Starting Monitor",
        )
    )

    start_monitor_server(host, port, llama_port)


@monitor_app.command("cli")
def monitor_cli(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
):
    """Show live metrics in terminal.

    Displays real-time statistics in the terminal.

    Example:
        moxing monitor cli --port 8080
    """
    from moxing.monitor import print_live_metrics

    print_live_metrics(host, port)


@monitor_app.command("open")
def monitor_open(
    port: int = typer.Option(8080, "-p", "--port", help="llama.cpp server port"),
):
    """Open the built-in monitoring page.

    Opens the monitoring page in your browser.
    Requires llama.cpp server running with --metrics enabled.

    Example:
        moxing monitor open --port 8080
    """

    url = f"http://127.0.0.1:{port}"

    console.print(f"[blue]Opening browser: {url}[/blue]")
    console.print("[dim]Note: llama.cpp server must be running with --metrics enabled[/dim]")

    webbrowser.open(url)


@monitor_app.command("stats")
def monitor_stats(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
):
    """Show current server statistics.

    Example:
        moxing monitor stats --port 8080
    """
    from moxing.monitor import MetricsCollector

    collector = MetricsCollector(host, port)
    metrics = collector.fetch_metrics()
    slots = collector.fetch_slots()
    props = collector.fetch_props()

    if props:
        model_name = Path(props.get("model_path", "Unknown")).name
        console.print(
            Panel(
                f"[cyan]Model:[/cyan] {model_name}\n"
                f"[cyan]Context:[/cyan] {props.get('n_ctx', '--')}\n"
                f"[cyan]Batch:[/cyan] {props.get('n_batch', '--')}",
                title="Model Info",
            )
        )

    if metrics:
        table = Table(title="Server Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Prompt Tokens", f"{metrics.prompt_tokens_total:,}")
        table.add_row("Generated Tokens", f"{metrics.tokens_predicted_total:,}")
        table.add_row(
            "Total Tokens", f"{metrics.prompt_tokens_total + metrics.tokens_predicted_total:,}"
        )
        table.add_row("Prompt Speed", f"{metrics.prompt_tokens_per_second:.1f} tok/s")
        table.add_row("Generate Speed", f"{metrics.predicted_tokens_per_second:.1f} tok/s")
        table.add_row("Processing", str(metrics.requests_processing))
        table.add_row("Deferred", str(metrics.requests_deferred))

        console.print(table)

    if slots:
        slots_table = Table(title="Slots")
        slots_table.add_column("ID", style="cyan")
        slots_table.add_column("Status", style="green")
        slots_table.add_column("Context", style="yellow")

        for slot in slots:
            status = "[green]Processing[/green]" if slot.get("is_processing") else "[dim]Idle[/dim]"
            slots_table.add_row(str(slot.get("id", "?")), status, str(slot.get("n_ctx", "--")))

        console.print(slots_table)

    if not metrics and not slots:
        console.print(f"[red]Failed to connect to server at {host}:{port}[/red]")
