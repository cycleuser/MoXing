"""
GGUF compression and transparent decompression support.

Supports multiple compression algorithms:
- zstd: Best balance of speed and compression ratio
- lz4: Fastest compression/decompression
- xz: Best compression ratio (slowest)
- gzip: Universal compatibility

GGUF files are already quantized, so compression ratio is limited (~3-5%).
"""

import os
import sys
import struct
import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import threading
import shutil

from rich.console import Console
from rich.table import Table

console = Console()

GGUF_MAGIC = 0x46554747

COMPRESSION_EXTENSIONS = {
    ".zst": "zstd",
    ".zstd": "zstd",
    ".lz4": "lz4",
    ".xz": "xz",
    ".gz": "gzip",
    ".gzip": "gzip",
    ".bz2": "bzip2",
}

COMPRESSION_LEVELS = {
    "zstd": (1, 22, 19),
    "lz4": (1, 12, 9),
    "xz": (0, 9, 6),
    "gzip": (1, 9, 9),
    "bzip2": (1, 9, 9),
}


@dataclass
class CompressedGGUFInfo:
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: str
    compression_time: float
    decompression_time: float = 0.0
    original_path: Optional[Path] = None
    
    @property
    def savings_mb(self) -> float:
        return (self.original_size - self.compressed_size) / (1024 * 1024)
    
    @property
    def savings_percent(self) -> float:
        return (1 - self.compression_ratio) * 100


def check_tool_available(tool: str) -> bool:
    try:
        result = subprocess.run([tool, "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def get_available_compressors() -> Dict[str, bool]:
    return {
        "zstd": check_tool_available("zstd"),
        "lz4": check_tool_available("lz4"),
        "xz": check_tool_available("xz"),
        "gzip": check_tool_available("gzip"),
    }


def is_gguf_compressed(path: Path) -> bool:
    if not path.exists():
        return False
    
    suffix = path.suffix.lower()
    if suffix in COMPRESSION_EXTENSIONS:
        return True
    
    if suffix == ".gguf":
        return False
    
    return False


def detect_compression_type(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    
    suffix = path.suffix.lower()
    return COMPRESSION_EXTENSIONS.get(suffix)


def is_gguf_split(path: Path) -> bool:
    return "-part-" in path.name


def find_split_files(base_path: Path) -> List[Path]:
    parent = base_path.parent
    name = base_path.name
    
    if "-part-" in name:
        base_name = name.rsplit("-part-", 1)[0]
    else:
        base_name = name.replace(".gguf", "").replace(".zst", "")
    
    parts = sorted(parent.glob(f"{base_name}-part-*"))
    return list(parts)


class MultiCompressor:
    """
    Multi-algorithm compressor for GGUF files.
    
    Supports: zstd, lz4, xz, gzip
    """
    
    def __init__(self, algorithm: str = "zstd", level: Optional[int] = None):
        self.algorithm = algorithm
        self.level = level
        self._tools = get_available_compressors()
        
        if algorithm not in self._tools:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        if not self._tools[algorithm]:
            raise RuntimeError(f"{algorithm} not installed")
        
        min_lvl, max_lvl, default_lvl = COMPRESSION_LEVELS[algorithm]
        if level is None:
            self.level = default_lvl
        elif level < min_lvl or level > max_lvl:
            raise ValueError(f"Level must be {min_lvl}-{max_lvl} for {algorithm}")
    
    def compress(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        keep_original: bool = True
    ) -> CompressedGGUFInfo:
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        ext = f".{self.algorithm}" if self.algorithm != "gzip" else ".gz"
        if output_path is None:
            output_path = Path(str(input_path) + ext)
        
        original_size = input_path.stat().st_size
        
        start_time = time.time()
        
        if self.algorithm == "zstd":
            result = subprocess.run(
                ["zstd", f"-{self.level}", "-f", str(input_path), "-o", str(output_path)],
                capture_output=True, text=True
            )
        elif self.algorithm == "lz4":
            result = subprocess.run(
                ["lz4", f"-{self.level}", "-f", str(input_path), str(output_path)],
                capture_output=True, text=True
            )
        elif self.algorithm == "xz":
            result = subprocess.run(
                ["xz", f"-{self.level}", "-k", "-f", str(input_path)],
                capture_output=True, text=True
            )
            output_path = Path(str(input_path) + ".xz")
        elif self.algorithm == "gzip":
            result = subprocess.run(
                ["gzip", f"-{self.level}", "-k", "-f", str(input_path)],
                capture_output=True, text=True
            )
            output_path = Path(str(input_path) + ".gz")
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        if result.returncode != 0:
            raise RuntimeError(f"Compression failed: {result.stderr}")
        
        compression_time = time.time() - start_time
        compressed_size = output_path.stat().st_size
        ratio = compressed_size / original_size if original_size > 0 else 0
        
        if not keep_original:
            input_path.unlink()
        
        return CompressedGGUFInfo(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            algorithm=self.algorithm,
            compression_time=compression_time,
            original_path=input_path if keep_original else None
        )
    
    def decompress(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        keep_compressed: bool = True
    ) -> Tuple[Path, float]:
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        alg = detect_compression_type(input_path)
        if alg is None:
            alg = self.algorithm
        
        if output_path is None:
            name = input_path.name
            for ext in COMPRESSION_EXTENSIONS.keys():
                if name.lower().endswith(ext):
                    name = name[:-len(ext)]
                    break
            output_path = input_path.parent / name
        
        start_time = time.time()
        
        if alg == "zstd":
            result = subprocess.run(
                ["zstd", "-d", "-f", str(input_path), "-o", str(output_path)],
                capture_output=True, text=True
            )
        elif alg == "lz4":
            result = subprocess.run(
                ["lz4", "-d", "-f", str(input_path), str(output_path)],
                capture_output=True, text=True
            )
        elif alg == "xz":
            result = subprocess.run(
                ["xz", "-d", "-k", "-f", str(input_path)],
                capture_output=True, text=True
            )
            output_path = input_path.with_suffix("")
        elif alg == "gzip":
            result = subprocess.run(
                ["gzip", "-d", "-k", "-f", str(input_path)],
                capture_output=True, text=True
            )
            output_path = input_path.with_suffix("")
        else:
            raise ValueError(f"Unknown algorithm: {alg}")
        
        if result.returncode != 0:
            raise RuntimeError(f"Decompression failed: {result.stderr}")
        
        decompression_time = time.time() - start_time
        
        if not keep_compressed:
            input_path.unlink()
        
        return output_path, decompression_time


class TransparentDecompressor:
    """
    Transparent decompression with intelligent caching.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "moxing" / "decompressed"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._compressor = MultiCompressor()
    
    def get_decompressed_path(self, compressed_path: Path) -> Path:
        if not is_gguf_compressed(compressed_path):
            return compressed_path
        
        stem = compressed_path.stem
        if stem.endswith(".gguf"):
            cache_name = stem
        else:
            cache_name = f"{stem}.gguf"
        
        cache_path = self.cache_dir / cache_name
        
        with self._lock:
            if cache_path.exists():
                console.print(f"[green]Using cached: {cache_path.name}[/green]")
                return cache_path
            
            console.print(f"[blue]Decompressing {compressed_path.name}...[/blue]")
            result_path, _ = self._compressor.decompress(compressed_path, cache_path, keep_compressed=True)
            return result_path
    
    def clear_cache(self, older_than_days: int = 0):
        import time as t
        cutoff = t.time() - (older_than_days * 24 * 60 * 60)
        
        for f in self.cache_dir.iterdir():
            if older_than_days == 0 or f.stat().st_mtime < cutoff:
                f.unlink()
    
    def get_cache_size(self) -> int:
        total = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                total += f.stat().st_size
        return total


class GGUFSplitter:
    def __init__(self, chunk_size_mb: int = 1024):
        self.chunk_size = chunk_size_mb * 1024 * 1024
    
    def split(self, input_path: Path, output_dir: Optional[Path] = None) -> List[Path]:
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        if output_dir is None:
            output_dir = input_path.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = input_path.name
        chunk_paths = []
        
        console.print(f"[blue]Splitting into {self.chunk_size // (1024*1024)}MB chunks...[/blue]")
        
        with open(input_path, "rb") as f:
            chunk_index = 0
            while True:
                chunk_data = f.read(self.chunk_size)
                if not chunk_data:
                    break
                
                chunk_suffix = chr(ord('a') + chunk_index // 26) + chr(ord('a') + chunk_index % 26)
                chunk_name = f"{base_name}-part-{chunk_suffix}"
                chunk_path = output_dir / chunk_name
                
                with open(chunk_path, "wb") as chunk_file:
                    chunk_file.write(chunk_data)
                
                chunk_paths.append(chunk_path)
                chunk_index += 1
        
        console.print(f"[green]Created {len(chunk_paths)} chunks[/green]")
        return chunk_paths
    
    def merge(self, chunk_paths: List[Path], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[blue]Merging {len(chunk_paths)} chunks...[/blue]")
        
        with open(output_path, "wb") as f_out:
            for chunk_path in chunk_paths:
                with open(chunk_path, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
        
        console.print(f"[green]Merged to {output_path}[/green]")
        return output_path


def benchmark_compression(
    input_path: Path,
    algorithms: Optional[List[str]] = None
) -> List[CompressedGGUFInfo]:
    """
    Benchmark different compression algorithms on a GGUF file.
    """
    if algorithms is None:
        algorithms = ["zstd", "lz4", "xz", "gzip"]
    
    tools = get_available_compressors()
    results = []
    
    for alg in algorithms:
        if not tools.get(alg):
            console.print(f"[yellow]{alg}: not available[/yellow]")
            continue
        
        try:
            compressor = MultiCompressor(alg)
            ext = f".{alg}" if alg != "gzip" else ".gz"
            output_path = Path(f"/tmp/bench_{alg}{ext}")
            
            console.print(f"[blue]Testing {alg}...[/blue]")
            info = compressor.compress(input_path, output_path)
            results.append(info)
            
            # Test decompression speed
            start = time.time()
            compressor.decompress(output_path, Path(f"/tmp/bench_{alg}_out.gguf"))
            info.decompression_time = time.time() - start
            
            # Cleanup
            output_path.unlink(missing_ok=True)
            Path(f"/tmp/bench_{alg}_out.gguf").unlink(missing_ok=True)
            
        except Exception as e:
            console.print(f"[red]{alg}: {e}[/red]")
    
    return results


def print_benchmark_results(results: List[CompressedGGUFInfo]):
    """Print benchmark results in a table."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return
    
    table = Table(title="Compression Benchmark Results")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Original", style="blue")
    table.add_column("Compressed", style="green")
    table.add_column("Ratio", style="yellow")
    table.add_column("Saved", style="magenta")
    table.add_column("Comp.Time", style="dim")
    table.add_column("Decomp.Time", style="dim")
    
    for r in results:
        table.add_row(
            r.algorithm,
            f"{r.original_size / (1024**3):.2f} GB",
            f"{r.compressed_size / (1024**3):.2f} GB",
            f"{r.compression_ratio:.1%}",
            f"{r.savings_percent:.1f}%",
            f"{r.compression_time:.1f}s",
            f"{r.decompression_time:.1f}s"
        )
    
    console.print(table)
    
    # Find best
    if len(results) > 1:
        best_ratio = min(results, key=lambda x: x.compression_ratio)
        best_speed = min(results, key=lambda x: x.compression_time)
        
        console.print(f"\n[green]Best compression: {best_ratio.algorithm} ({best_ratio.savings_percent:.1f}% saved)[/green]")
        console.print(f"[green]Fastest: {best_speed.algorithm} ({best_speed.compression_time:.1f}s)[/green]")


def resolve_model_path(model_path: Path) -> Path:
    if is_gguf_compressed(model_path):
        decompressor = TransparentDecompressor()
        return decompressor.get_decompressed_path(model_path)
    
    if is_gguf_split(model_path):
        parts = find_split_files(model_path)
        if parts:
            output_path = model_path.parent / f"{model_path.name}.merged"
            if not output_path.exists():
                splitter = GGUFSplitter()
                splitter.merge(parts, output_path)
            return output_path
    
    return model_path


def compress_model(
    input_path: Path,
    output_path: Optional[Path] = None,
    algorithm: str = "zstd",
    level: Optional[int] = None,
    keep_original: bool = True
) -> CompressedGGUFInfo:
    compressor = MultiCompressor(algorithm, level)
    return compressor.compress(input_path, output_path, keep_original)


def get_compression_stats(model_dir: Path) -> Dict[str, Any]:
    stats = {
        "total_original": 0,
        "total_compressed": 0,
        "compressed_files": 0,
        "uncompressed_files": 0,
    }
    
    for f in model_dir.glob("*.gguf*"):
        ext = f.suffix.lower()
        if ext in COMPRESSION_EXTENSIONS:
            stats["compressed_files"] += 1
            stats["total_compressed"] += f.stat().st_size
        elif ext == ".gguf":
            stats["uncompressed_files"] += 1
            stats["total_original"] += f.stat().st_size
    
    stats["savings"] = stats["total_original"] - stats["total_compressed"]
    stats["savings_gb"] = stats["savings"] / (1024**3)
    
    return stats