from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()

_ARG_INPUT_GGUF = typer.Argument(..., help="Path to GGUF file")
_ARG_INPUT_COMPRESSED = typer.Argument(..., help="Path to compressed file")
_ARG_INPUT_CHUNK = typer.Argument(..., help="First chunk file or pattern")
_ARG_OUTPUT_GGUF = typer.Argument(..., help="Output GGUF file")
_OPT_OUTPUT_FILE = typer.Option(None, "-o", "--output", help="Output path")
_OPT_OUTPUT_DIR = typer.Option(None, "-o", "--output", help="Output directory")

compress_app = typer.Typer(name="compress", help="GGUF compression commands")


@compress_app.command("pack")
def compress_pack(
    input_path: Path = _ARG_INPUT_GGUF,
    output_path: Optional[Path] = _OPT_OUTPUT_FILE,
    algorithm: str = typer.Option(
        "zstd", "-a", "--algorithm", help="Compression algorithm: zstd, lz4, xz, gzip"
    ),
    level: int = typer.Option(0, "-l", "--level", help="Compression level (0=auto)"),
    keep_original: bool = typer.Option(True, "-k", "--keep", help="Keep original file"),
):
    """Compress a GGUF file to save disk space.

    Algorithms:
    - zstd: Best balance (default)
    - lz4: Fastest
    - xz: Best compression (slowest)
    - gzip: Universal

    Example:
        moxing compress pack model.gguf
        moxing compress pack model.gguf -a lz4
        moxing compress pack model.gguf -a xz -l 9
    """
    from moxing.gguf_compress import MultiCompressor, is_gguf_compressed

    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)

    if is_gguf_compressed(input_path):
        console.print("[yellow]File is already compressed[/yellow]")
        raise typer.Exit(0)

    try:
        lvl = level if level > 0 else None
        compressor = MultiCompressor(algorithm=algorithm, level=lvl)
        info = compressor.compress(input_path, output_path, keep_original)

        console.print("\n[green]Compression complete![/green]")
        console.print(f"  Algorithm: {info.algorithm}")
        console.print(f"  Original: {info.original_size / (1024**3):.2f} GB")
        console.print(f"  Compressed: {info.compressed_size / (1024**3):.2f} GB")
        console.print(f"  Ratio: {info.compression_ratio:.1%}")
        console.print(f"  Saved: {info.savings_percent:.1f}% ({info.savings_mb:.0f} MB)")
        console.print(f"  Time: {info.compression_time:.1f}s")

    except Exception as e:
        console.print(f"[red]Compression failed: {e}[/red]")
        raise typer.Exit(1) from e


@compress_app.command("unpack")
def compress_unpack(
    input_path: Path = _ARG_INPUT_COMPRESSED,
    output_path: Optional[Path] = _OPT_OUTPUT_FILE,
    keep_compressed: bool = typer.Option(True, "-k", "--keep", help="Keep compressed file"),
):
    """Decompress a compressed GGUF file.

    Example:
        moxing compress unpack model.gguf.zst
    """
    from moxing.gguf_compress import MultiCompressor, detect_compression_type, is_gguf_compressed

    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)

    if not is_gguf_compressed(input_path):
        console.print("[yellow]File is not compressed[/yellow]")
        raise typer.Exit(0)

    try:
        alg = detect_compression_type(input_path) or "zstd"
        compressor = MultiCompressor(algorithm=alg)
        result, decomp_time = compressor.decompress(input_path, output_path, keep_compressed)

        console.print(f"[green]Decompressed to: {result}[/green]")
        console.print(f"[dim]Time: {decomp_time:.1f}s[/dim]")

    except Exception as e:
        console.print(f"[red]Decompression failed: {e}[/red]")
        raise typer.Exit(1) from e


@compress_app.command("cache")
def compress_cache(
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear decompression cache"),
    size: bool = typer.Option(False, "--size", "-s", help="Show cache size"),
    days: int = typer.Option(0, "--older-than", help="Clear files older than N days"),
):
    """Manage decompression cache.

    Examples:
        moxing compress cache --size
        moxing compress cache --clear
        moxing compress cache --clear --older-than 7
    """
    from moxing.gguf_compress import TransparentDecompressor

    decompressor = TransparentDecompressor()

    if size:
        cache_size = decompressor.get_cache_size()
        console.print(
            f"Cache size: {cache_size / (1024**3):.2f} GB ({cache_size / (1024**2):.1f} MB)"
        )
        console.print(f"Cache location: {decompressor.cache_dir}")

    if clear:
        decompressor.clear_cache(older_than_days=days)
        console.print("[green]Cache cleared[/green]")


@compress_app.command("split")
def compress_split(
    input_path: Path = _ARG_INPUT_GGUF,
    chunk_size: int = typer.Option(1024, "-s", "--size", help="Chunk size in MB"),
    output_dir: Optional[Path] = _OPT_OUTPUT_DIR,
):
    """Split a GGUF file into chunks.

    Useful for storage on filesystems with size limits.

    Example:
        moxing compress split model.gguf --size 512
    """
    from moxing.gguf_compress import GGUFSplitter

    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)

    try:
        splitter = GGUFSplitter(chunk_size_mb=chunk_size)
        chunks = splitter.split(input_path, output_dir)

        console.print(f"[green]Split into {len(chunks)} chunks[/green]")
        for chunk in chunks[:5]:
            console.print(f"  {chunk.name}")
        if len(chunks) > 5:
            console.print(f"  ... and {len(chunks) - 5} more")

    except Exception as e:
        console.print(f"[red]Split failed: {e}[/red]")
        raise typer.Exit(1) from e


@compress_app.command("merge")
def compress_merge(
    input_pattern: Path = _ARG_INPUT_CHUNK,
    output_path: Path = _ARG_OUTPUT_GGUF,
):
    """Merge split GGUF chunks back into a single file.

    Example:
        moxing compress merge model.gguf-part-aa merged.gguf
    """
    from moxing.gguf_compress import GGUFSplitter, find_split_files

    try:
        if input_pattern.is_file():
            chunks = find_split_files(input_pattern)
        else:
            chunks = sorted(input_pattern.parent.glob(f"{input_pattern.name}*"))

        if not chunks:
            console.print("[red]No chunk files found[/red]")
            raise typer.Exit(1)

        splitter = GGUFSplitter()
        splitter.merge(chunks, output_path)

        console.print(f"[green]Merged to: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Merge failed: {e}[/red]")
        raise typer.Exit(1) from e
