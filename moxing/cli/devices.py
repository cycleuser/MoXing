import typer
from rich.console import Console

console = Console()


def devices():
    """List available GPU devices and their capabilities."""
    from moxing.device import DeviceDetector

    detector = DeviceDetector()
    detector.detect()
    detector.list_devices()


def register(app: typer.Typer):
    app.command()(devices)
