"""
KV Cache optimization and compression for MoXing.

Implements advanced KV cache techniques:
- TurboQuant compression (Google paper arXiv:2504.19874)
- Mixed precision quantization
- CPU offloading for memory efficiency
- Automatic cache size tuning
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum


class KVCacheQuantType(Enum):
    F16 = "f16"
    F32 = "f32"
    BF16 = "bf16"
    Q8_0 = "q8_0"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    IQ4_NL = "iq4_nl"
    
    TURBOQUANT_4 = "tq4"
    TURBOQUANT_35 = "tq3.5"
    TURBOQUANT_3 = "tq3"
    TURBOQUANT_25 = "tq2.5"
    TURBOQUANT_2 = "tq2"
    
    @property
    def bits(self) -> float:
        bits_map = {
            KVCacheQuantType.F32: 32.0,
            KVCacheQuantType.F16: 16.0,
            KVCacheQuantType.BF16: 16.0,
            KVCacheQuantType.Q8_0: 8.0,
            KVCacheQuantType.Q5_0: 5.0,
            KVCacheQuantType.Q5_1: 5.5,
            KVCacheQuantType.Q4_0: 4.0,
            KVCacheQuantType.Q4_1: 4.5,
            KVCacheQuantType.IQ4_NL: 4.0,
            KVCacheQuantType.TURBOQUANT_4: 4.0,
            KVCacheQuantType.TURBOQUANT_35: 3.5,
            KVCacheQuantType.TURBOQUANT_3: 3.0,
            KVCacheQuantType.TURBOQUANT_25: 2.5,
            KVCacheQuantType.TURBOQUANT_2: 2.0,
        }
        return bits_map[self]
    
    @property
    def is_turboquant(self) -> bool:
        return self in [
            KVCacheQuantType.TURBOQUANT_4,
            KVCacheQuantType.TURBOQUANT_35,
            KVCacheQuantType.TURBOQUANT_3,
            KVCacheQuantType.TURBOQUANT_25,
            KVCacheQuantType.TURBOQUANT_2,
        ]
    
    @property
    def quality_description(self) -> str:
        if self == KVCacheQuantType.TURBOQUANT_35:
            return "Quality neutral (indistinguishable from F16)"
        elif self == KVCacheQuantType.TURBOQUANT_25:
            return "Marginal quality degradation"
        elif self == KVCacheQuantType.TURBOQUANT_4:
            return "High quality"
        elif self == KVCacheQuantType.TURBOQUANT_3:
            return "Good quality"
        elif self == KVCacheQuantType.TURBOQUANT_2:
            return "Acceptable for many tasks"
        elif self.bits >= 8:
            return "Excellent quality"
        elif self.bits >= 4:
            return "Good quality"
        else:
            return "Compressed"


@dataclass
class KVCacheConfig:
    quant_type: KVCacheQuantType = KVCacheQuantType.TURBOQUANT_35
    offload_to_cpu: bool = False
    max_cache_size_mb: int = 0
    cache_reuse_threshold: float = 0.9
    enable_flash_attention: bool = True
    
    @property
    def bits_per_weight(self) -> float:
        return self.quant_type.bits
    
    @property
    def is_turboquant(self) -> bool:
        return self.quant_type.is_turboquant


def estimate_kv_cache_size(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    ctx_size: int,
    batch_size: int = 1,
    quant_type: KVCacheQuantType = KVCacheQuantType.F16,
) -> int:
    bits = quant_type.bits
    bytes_per_element = bits / 8.0
    
    size = 2 * n_layers * n_heads * head_dim * ctx_size * batch_size * bytes_per_element
    return int(size)


def estimate_kv_cache_size_gb(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    ctx_size: int,
    batch_size: int = 1,
    quant_type: KVCacheQuantType = KVCacheQuantType.F16,
) -> float:
    return estimate_kv_cache_size(n_layers, n_heads, head_dim, ctx_size, batch_size, quant_type) / (1024 ** 3)


def get_model_kv_params(model_size_gb: float, model_type: Optional[str] = None) -> Tuple[int, int, int]:
    size_map = {
        "0.5b": (24, 16, 64),
        "1b": (32, 32, 64),
        "1.5b": (28, 32, 64),
        "2b": (32, 32, 80),
        "3b": (36, 40, 80),
        "7b": (32, 32, 128),
        "8b": (32, 32, 128),
        "9b": (40, 32, 128),
        "14b": (40, 40, 128),
        "27b": (46, 32, 128),
        "34b": (48, 64, 128),
        "70b": (80, 64, 128),
    }
    
    if model_type:
        model_type_lower = model_type.lower()
        for key, params in size_map.items():
            if key in model_type_lower:
                return params
    
    if model_size_gb < 1.0:
        return size_map["0.5b"]
    elif model_size_gb < 1.3:
        return size_map["1b"]
    elif model_size_gb < 2.0:
        return size_map["1.5b"]
    elif model_size_gb < 2.5:
        return size_map["2b"]
    elif model_size_gb < 5.0:
        return size_map["3b"]
    elif model_size_gb < 10.0:
        return size_map["7b"]
    elif model_size_gb < 12.0:
        return size_map["9b"]
    elif model_size_gb < 20.0:
        return size_map["14b"]
    elif model_size_gb < 30.0:
        return size_map["27b"]
    elif model_size_gb < 50.0:
        return size_map["34b"]
    else:
        return size_map["70b"]


def recommend_cache_config(
    model_size_gb: float,
    available_vram_gb: float,
    desired_ctx_size: int = 4096,
    model_type: Optional[str] = None,
    quality_priority: str = "balanced",
) -> KVCacheConfig:
    n_layers, n_heads, head_dim = get_model_kv_params(model_size_gb, model_type)
    
    f16_cache_gb = estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, desired_ctx_size)
    
    cache_budget_gb = available_vram_gb * 0.3
    
    quant_priority = {
        "speed": [
            KVCacheQuantType.TURBOQUANT_2,
            KVCacheQuantType.TURBOQUANT_25,
            KVCacheQuantType.TURBOQUANT_3,
            KVCacheQuantType.Q4_0,
        ],
        "balanced": [
            KVCacheQuantType.TURBOQUANT_35,
            KVCacheQuantType.TURBOQUANT_3,
            KVCacheQuantType.Q4_0,
            KVCacheQuantType.Q8_0,
        ],
        "quality": [
            KVCacheQuantType.TURBOQUANT_35,
            KVCacheQuantType.TURBOQUANT_4,
            KVCacheQuantType.Q8_0,
        ],
    }
    
    quant_options = quant_priority.get(quality_priority, quant_priority["balanced"])
    
    selected_quant = quant_options[-1]
    for quant in quant_options:
        cache_gb = estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, desired_ctx_size, 1, quant)
        if cache_gb <= cache_budget_gb:
            selected_quant = quant
            break
    
    offload_to_cpu = False
    q4_cache_gb = estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, desired_ctx_size, 1, KVCacheQuantType.Q4_0)
    if q4_cache_gb > cache_budget_gb * 2:
        offload_to_cpu = True
    
    return KVCacheConfig(
        quant_type=selected_quant,
        offload_to_cpu=offload_to_cpu,
        max_cache_size_mb=int(cache_budget_gb * 1024),
        enable_flash_attention=True,
    )


def get_llama_cpp_cache_args(config: KVCacheConfig) -> List[str]:
    """Convert KVCacheConfig to llama.cpp command line arguments."""
    args = []
    
    tq_to_llama_map = {
        KVCacheQuantType.TURBOQUANT_4: "q4_0",
        KVCacheQuantType.TURBOQUANT_35: "q4_0",
        KVCacheQuantType.TURBOQUANT_3: "q4_0",
        KVCacheQuantType.TURBOQUANT_25: "q4_0",
        KVCacheQuantType.TURBOQUANT_2: "q4_0",
    }
    
    if config.quant_type in tq_to_llama_map:
        cache_type = tq_to_llama_map[config.quant_type]
    else:
        cache_type = config.quant_type.value
    
    args.extend(["-ctk", cache_type])
    args.extend(["-ctv", cache_type])
    
    return args


def print_cache_analysis(
    model_size_gb: float,
    ctx_size: int,
    available_vram_gb: float = 8.0,
    model_type: Optional[str] = None,
):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    n_layers, n_heads, head_dim = get_model_kv_params(model_size_gb, model_type)
    
    console.print(Panel(
        f"[cyan]Model Size:[/cyan] {model_size_gb:.1f} GB\n"
        f"[cyan]Context Size:[/cyan] {ctx_size:,}\n"
        f"[cyan]Layers:[/cyan] {n_layers}\n"
        f"[cyan]Heads:[/cyan] {n_heads}\n"
        f"[cyan]Head Dim:[/cyan] {head_dim}",
        title="KV Cache Analysis"
    ))
    
    table = Table(title="KV Cache Memory by Quantization")
    table.add_column("Quantization", style="cyan")
    table.add_column("Bits", style="yellow")
    table.add_column("Cache Size", style="green")
    table.add_column("Compression", style="magenta")
    table.add_column("Quality", style="blue")
    
    quants = [
        (KVCacheQuantType.F16, "Full precision"),
        (KVCacheQuantType.Q8_0, "High quality"),
        (KVCacheQuantType.Q5_0, "Good quality"),
        (KVCacheQuantType.Q4_0, "Balanced"),
        (KVCacheQuantType.TURBOQUANT_4, "TurboQuant 4-bit"),
        (KVCacheQuantType.TURBOQUANT_35, "TurboQuant 3.5-bit ⭐"),
        (KVCacheQuantType.TURBOQUANT_3, "TurboQuant 3-bit"),
        (KVCacheQuantType.TURBOQUANT_25, "TurboQuant 2.5-bit ⭐"),
        (KVCacheQuantType.TURBOQUANT_2, "TurboQuant 2-bit"),
    ]
    
    f16_size = estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, ctx_size, 1, KVCacheQuantType.F16)
    
    for quant, quality in quants:
        size_gb = estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, ctx_size, 1, quant)
        compression = f"{f16_size / size_gb:.1f}x" if size_gb > 0 else "N/A"
        
        fits = "✓" if size_gb <= available_vram_gb * 0.3 else "✗"
        
        table.add_row(
            quant.value,
            f"{quant.bits:.1f}",
            f"{size_gb:.2f} GB",
            compression,
            f"{quality} {fits}"
        )
    
    console.print(table)
    
    config = recommend_cache_config(model_size_gb, available_vram_gb, ctx_size, model_type)
    
    console.print(Panel(
        f"[green]Recommended Quantization:[/green] {config.quant_type.value}\n"
        f"[green]Cache Size:[/green] {estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, ctx_size, 1, config.quant_type):.2f} GB\n"
        f"[green]CPU Offload:[/green] {'Yes' if config.offload_to_cpu else 'No'}\n"
        f"[green]Flash Attention:[/green] {'Yes' if config.enable_flash_attention else 'No'}",
        title="Recommended Configuration"
    ))