"""
GGUF model diagnostics and compatibility checking.
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from rich.console import Console

console = Console()


GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class GGUFMetadata:
    """GGUF model metadata."""
    version: int
    architecture: str
    name: str
    quantization: str
    context_length: int
    parameter_count: int
    file_size: int
    metadata: Dict[str, Any]
    required_keys: List[str]
    missing_keys: List[str]
    warnings: List[str]
    errors: List[str]
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    @property
    def parameter_count_b(self) -> float:
        return self.parameter_count / 1e9


class GGUFParser:
    """Parse GGUF files to extract metadata and check compatibility."""
    
    KNOWN_ARCHITECTURES = {
        "llama", "mistral", "mixtral", "qwen2", "qwen3", "qwen35", "gemma", "gemma2", "gemma3",
        "phi3", "starcoder2", "mpt", "falcon", "baichuan", "chatglm", "internlm2",
        "deepseek", "deepseek2", "command-r", "dbrx", "granite", "gpt-2", "gpt-j",
        "gpt-neox", "grok-1", "jais", "llava", "minicpm", "opt", "orca", "persimmon",
        "plamo", "refact", "solar", "t5", "vikhr", "lfm", "lfm2"
    }
    
    CRITICAL_KEYS = {
        "llama": ["llama.attention.layer_norm_rms_epsilon"],
        "mistral": ["llama.attention.layer_norm_rms_epsilon"],
        "gemma": ["gemma.attention.layer_norm_rms_epsilon"],
        "gemma2": ["gemma2.attention.layer_norm_rms_epsilon"],
        "gemma3": ["gemma3.attention.layer_norm_rms_epsilon", "gemma3.attention.head_count"],
        "qwen2": ["qwen2.attention.layer_norm_rms_epsilon"],
        "qwen3": ["qwen3.attention.layer_norm_rms_epsilon"],
        "qwen35": ["qwen35.attention.layer_norm_rms_epsilon"],
    }
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fp = None
        self._metadata: Dict[str, Any] = {}
        
    def parse(self) -> GGUFMetadata:
        """Parse the GGUF file and return metadata."""
        with open(self.path, "rb") as f:
            self._fp = f
            
            magic = self._read_uint32()
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
            
            version = self._read_uint32()
            tensor_count = self._read_uint64()
            metadata_kv_count = self._read_uint64()
            
            for _ in range(metadata_kv_count):
                key = self._read_string()
                value = self._read_value()
                self._metadata[key] = value
        
        return self._build_metadata(version)
    
    def _read_uint32(self) -> int:
        return struct.unpack("<I", self._fp.read(4))[0]
    
    def _read_uint64(self) -> int:
        return struct.unpack("<Q", self._fp.read(8))[0]
    
    def _read_int32(self) -> int:
        return struct.unpack("<i", self._fp.read(4))[0]
    
    def _read_float32(self) -> float:
        return struct.unpack("<f", self._fp.read(4))[0]
    
    def _read_string(self) -> str:
        length = self._read_uint64()
        return self._fp.read(length).decode("utf-8", errors="replace")
    
    def _read_value(self) -> Any:
        value_type = self._read_uint32()
        
        if value_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", self._fp.read(1))[0]
        elif value_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", self._fp.read(1))[0]
        elif value_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", self._fp.read(2))[0]
        elif value_type == GGUF_TYPE_INT16:
            return struct.unpack("<h", self._fp.read(2))[0]
        elif value_type == GGUF_TYPE_UINT32:
            return self._read_uint32()
        elif value_type == GGUF_TYPE_INT32:
            return self._read_int32()
        elif value_type == GGUF_TYPE_FLOAT32:
            return self._read_float32()
        elif value_type == GGUF_TYPE_BOOL:
            return struct.unpack("<?", self._fp.read(1))[0]
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string()
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = self._read_uint32()
            array_length = self._read_uint64()
            return [self._read_value_by_type(array_type) for _ in range(array_length)]
        elif value_type == GGUF_TYPE_UINT64:
            return self._read_uint64()
        elif value_type == GGUF_TYPE_INT64:
            return struct.unpack("<q", self._fp.read(8))[0]
        elif value_type == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", self._fp.read(8))[0]
        else:
            return None
    
    def _read_value_by_type(self, value_type: int) -> Any:
        if value_type == GGUF_TYPE_UINT32:
            return self._read_uint32()
        elif value_type == GGUF_TYPE_INT32:
            return self._read_int32()
        elif value_type == GGUF_TYPE_FLOAT32:
            return self._read_float32()
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string()
        else:
            return self._read_value()
    
    def _build_metadata(self, version: int) -> GGUFMetadata:
        arch = self._metadata.get("general.architecture", "unknown")
        name = self._metadata.get("general.name", self.path.stem)
        quant = self._get_quantization()
        ctx_len = self._get_context_length(arch)
        param_count = self._metadata.get("general.parameter_count", 0)
        
        errors = []
        warnings = []
        missing_keys = []
        required_keys = []
        
        if arch in self.CRITICAL_KEYS:
            required_keys = self.CRITICAL_KEYS[arch]
            for key in required_keys:
                if key not in self._metadata:
                    missing_keys.append(key)
                    errors.append(f"Missing required key: {key}")
        
        if arch not in self.KNOWN_ARCHITECTURES:
            warnings.append(f"Unknown architecture: {arch}. May not be supported.")
        
        if version > GGUF_VERSION:
            warnings.append(f"GGUF version {version} is newer than supported ({GGUF_VERSION}). Update llama.cpp.")
        
        return GGUFMetadata(
            version=version,
            architecture=arch,
            name=name,
            quantization=quant,
            context_length=ctx_len,
            parameter_count=param_count,
            file_size=self.path.stat().st_size,
            metadata=self._metadata,
            required_keys=required_keys,
            missing_keys=missing_keys,
            warnings=warnings,
            errors=errors
        )
    
    def _get_quantization(self) -> str:
        file_type = self._metadata.get("general.file_type", 0)
        quant_map = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
            6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
            10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
            14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
            18: "Q6_K", 19: "Q8_K", 20: "IQ2_XXS", 21: "IQ2_XS",
            22: "IQ3_XXS", 23: "IQ1_S", 24: "IQ4_NL", 25: "IQ3_S",
            26: "IQ2_S", 27: "IQ4_XS", 28: "I8", 29: "I16", 30: "I32"
        }
        return quant_map.get(file_type, f"Unknown({file_type})")
    
    def _get_context_length(self, arch: str) -> int:
        keys = [
            f"{arch}.context_length",
            f"{arch}.context_length_training",
            "llama.context_length",
        ]
        for key in keys:
            if key in self._metadata:
                return self._metadata[key]
        return 4096


def diagnose_gguf(path: Path) -> GGUFMetadata:
    """Diagnose a GGUF file and return metadata with compatibility info."""
    parser = GGUFParser(path)
    return parser.parse()


def print_diagnosis(meta: GGUFMetadata):
    """Print a formatted diagnosis report."""
    from rich.table import Table
    from rich.panel import Panel
    
    console.print(Panel(
        f"[cyan]File:[/cyan] {meta.name}\n"
        f"[cyan]Architecture:[/cyan] {meta.architecture}\n"
        f"[cyan]Parameters:[/cyan] {meta.parameter_count_b:.1f}B\n"
        f"[cyan]Quantization:[/cyan] {meta.quantization}\n"
        f"[cyan]Context Length:[/cyan] {meta.context_length:,}\n"
        f"[cyan]File Size:[/cyan] {meta.file_size / (1024**3):.2f} GB\n"
        f"[cyan]GGUF Version:[/cyan] {meta.version}",
        title="GGUF Model Info"
    ))
    
    if meta.errors:
        console.print("\n[red bold]Errors:[/red bold]")
        for err in meta.errors:
            console.print(f"  [red]✗[/red] {err}")
    
    if meta.warnings:
        console.print("\n[yellow bold]Warnings:[/yellow bold]")
        for warn in meta.warnings:
            console.print(f"  [yellow]![/yellow] {warn}")
    
    if not meta.is_valid:
        console.print("\n[red]This model may not work with the current llama.cpp version.[/red]")
        console.print("\n[yellow]Suggested solutions:[/yellow]")
        console.print("  1. Use MLX backend: moxing serve <model> -b mlx")
        console.print("  2. Download a compatible GGUF from HuggingFace")
        console.print("  3. Re-convert the model with the latest llama.cpp")
    else:
        console.print("\n[green]✓ Model appears compatible with llama.cpp[/green]")


def check_compatibility(path: Path) -> Tuple[bool, List[str]]:
    """
    Check if a GGUF file is compatible with the current llama.cpp.
    
    Returns:
        Tuple of (is_compatible, list of issues)
    """
    try:
        meta = diagnose_gguf(path)
        issues = meta.errors + meta.warnings
        return meta.is_valid, issues
    except Exception as e:
        return False, [f"Failed to parse GGUF: {e}"]


def get_model_suggestions(path: Path) -> List[str]:
    """Get suggestions for fixing compatibility issues."""
    suggestions = []
    
    try:
        meta = diagnose_gguf(path)
        
        if not meta.is_valid:
            suggestions.append(f"Try MLX backend: moxing serve {path} -b mlx")
            
            if meta.architecture in ("gemma3", "qwen3", "qwen35"):
                suggestions.append(f"This is a {meta.architecture.upper()} model which requires the latest llama.cpp or MLX")
            
            suggestions.append("Download a pre-converted GGUF from HuggingFace GGUF repos")
            suggestions.append("Update llama.cpp binaries: moxing download-binaries --force")
    
    except Exception:
        suggestions.append("File may not be a valid GGUF file")
    
    return suggestions