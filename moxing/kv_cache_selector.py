"""
KV cache type auto-selection based on VRAM calculation.
Based on kaiwu project patterns for optimal performance.
"""

from dataclasses import dataclass


@dataclass
class KVCacheSelection:
    """KV cache type selection result."""

    k_type: str
    v_type: str
    reason: str
    estimated_vram_mb: float


def calculate_kv_cache_vram_mb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    ctx_size: int,
    bytes_per_element: float = 2.0,
) -> float:
    """Calculate KV cache VRAM usage in MB.

    Formula: n_layers * n_kv_heads * head_dim * ctx_size * bytes_per_element
    """
    total_elements = n_layers * n_kv_heads * head_dim * ctx_size
    total_bytes = total_elements * bytes_per_element
    return total_bytes / (1024 * 1024)


def select_kv_cache_type(
    free_vram_mb: float,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    ctx_size: int,
    supports_iso3: bool = False,
) -> KVCacheSelection:
    """Select optimal KV cache type based on available VRAM.

    Strategy (from kaiwu):
    1. f16 (fastest) -- if it fits
    2. q8_0 K + q4_0 V (balanced) -- if f16 doesn't fit
    3. iso3 (most compressed, ~3 bits/element) -- if turboquant binary available
    4. q4_0 + q4_0 (minimum) -- last resort
    """
    f16_vram_mb = calculate_kv_cache_vram_mb(n_layers, n_kv_heads, head_dim, ctx_size, 2.0)
    q8_q4_vram_mb = calculate_kv_cache_vram_mb(n_layers, n_kv_heads, head_dim, ctx_size, 1.25)
    q4_q4_vram_mb = calculate_kv_cache_vram_mb(n_layers, n_kv_heads, head_dim, ctx_size, 1.0)

    if f16_vram_mb <= free_vram_mb * 0.7:
        return KVCacheSelection(
            k_type="f16",
            v_type="f16",
            reason=f"f16 fits ({f16_vram_mb:.0f}MB < {free_vram_mb * 0.7:.0f}MB)",
            estimated_vram_mb=f16_vram_mb,
        )

    if supports_iso3 and q8_q4_vram_mb > free_vram_mb * 0.8:
        iso3_vram_mb = calculate_kv_cache_vram_mb(n_layers, n_kv_heads, head_dim, ctx_size, 0.375)
        if iso3_vram_mb <= free_vram_mb * 0.8:
            return KVCacheSelection(
                k_type="iq3_s",
                v_type="iq3_s",
                reason=f"iso3 compression needed ({iso3_vram_mb:.0f}MB)",
                estimated_vram_mb=iso3_vram_mb,
            )

    if q8_q4_vram_mb <= free_vram_mb * 0.8:
        return KVCacheSelection(
            k_type="q8_0",
            v_type="q4_0",
            reason=f"q8_0+q4_0 balanced ({q8_q4_vram_mb:.0f}MB)",
            estimated_vram_mb=q8_q4_vram_mb,
        )

    return KVCacheSelection(
        k_type="q4_0",
        v_type="q4_0",
        reason=f"q4_0 minimum ({q4_q4_vram_mb:.0f}MB)",
        estimated_vram_mb=q4_q4_vram_mb,
    )


def get_kv_cache_args(selection: KVCacheSelection) -> list:
    """Get llama.cpp command line arguments for KV cache type."""
    if selection.k_type == selection.v_type:
        if selection.k_type == "f16":
            return []
        return ["-ctk", selection.k_type, "-ctv", selection.v_type]
    return ["-ctk", selection.k_type, "-ctv", selection.v_type]


def estimate_context_from_vram(
    free_vram_mb: float,
    model_size_mb: float,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    overhead_mb: float = 1516.0,
    target_usage_pct: float = 0.85,
) -> int:
    """Estimate maximum context size from available VRAM using oobabooga formula.

    Based on kaiwu's implementation with 19,517 measurements across 60 models.
    """
    available_for_kv = free_vram_mb * target_usage_pct - model_size_mb * 1.1 - overhead_mb

    if available_for_kv <= 0:
        return 512

    bytes_per_element = 2.0
    kv_per_token = n_layers * n_kv_heads * head_dim * bytes_per_element

    max_ctx = int(available_for_kv * 1024 * 1024 / kv_per_token)

    max_ctx = max(512, min(max_ctx, 524288))

    ctx = 512
    while ctx * 2 <= max_ctx:
        ctx *= 2

    return ctx
