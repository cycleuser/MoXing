"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Based on Google's TurboQuant paper (arXiv:2504.19874):
- Data-oblivious online quantization
- Near-optimal distortion (within 2.7x of theoretical lower bound)
- MSE and Inner-product optimal variants
- 3.5 bits per channel for quality neutrality
- 2.5 bits per channel for marginal quality degradation

Reference: https://arxiv.org/html/2504.19874v1
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Literal
from dataclasses import dataclass
from enum import Enum


class TurboQuantMode(Enum):
    MSE = "mse"
    INNER_PRODUCT = "inner_product"


@dataclass
class TurboQuantConfig:
    dim: int = 128
    bits_per_channel: float = 3.5
    mode: TurboQuantMode = TurboQuantMode.INNER_PRODUCT
    use_rotation: bool = True
    seed: Optional[int] = None


class LloydMaxQuantizer:
    """Optimal scalar quantizer for Beta/Normal distribution using Lloyd-Max algorithm.
    
    For high-dimensional vectors after random rotation, each coordinate follows
    a Beta distribution that converges to N(0, 1/d). This class computes and
    stores optimal scalar quantizers for this distribution.
    """
    
    _codebooks_cache: dict = {}
    
    @staticmethod
    def get_codebook(dim: int, bits: int) -> np.ndarray:
        """Get optimal codebook centroids for given dimension and bit-width.
        
        For Beta distribution converging to N(0, 1/d), the optimal centroids are:
        - b=1: {±sqrt(2/(pi*d))}
        - b=2: {±0.453/sqrt(d), ±1.51/sqrt(d)}
        - b>2: computed via Lloyd-Max iterations
        """
        cache_key = (dim, bits)
        if cache_key in LloydMaxQuantizer._codebooks_cache:
            return LloydMaxQuantizer._codebooks_cache[cache_key]
        
        n_levels = 2 ** bits
        scale = 1.0 / math.sqrt(dim)
        
        if bits == 1:
            centroids = np.array([-scale * math.sqrt(2 / math.pi), 
                                  scale * math.sqrt(2 / math.pi)])
        elif bits == 2:
            centroids = np.array([-scale * 1.51, -scale * 0.453,
                                   scale * 0.453, scale * 1.51])
        else:
            centroids = LloydMaxQuantizer._compute_optimal_centroids(
                dim, bits, scale
            )
        
        LloydMaxQuantizer._codebooks_cache[cache_key] = centroids
        return centroids
    
    @staticmethod
    def _compute_optimal_centroids(dim: int, bits: int, scale: float, 
                                   n_iter: int = 50) -> np.ndarray:
        """Compute optimal centroids using Lloyd-Max algorithm.
        
        For high-dim Beta distribution, we use Normal N(0, 1/d) approximation
        for which optimal Lloyd-Max centroids are well known.
        """
        n_levels = 2 ** bits
        
        std = scale * math.sqrt(dim)
        initial_centroids = np.linspace(-3 * std, 3 * std, n_levels) / math.sqrt(dim)
        
        for _ in range(n_iter):
            boundaries = np.zeros(n_levels + 1)
            boundaries[0] = -np.inf
            boundaries[-1] = np.inf
            
            for i in range(1, n_levels):
                boundaries[i] = (initial_centroids[i-1] + initial_centroids[i]) / 2
            
            new_centroids = np.zeros(n_levels)
            x_vals = np.linspace(-4 * scale, 4 * scale, 10000)
            
            for i in range(n_levels):
                mask = (x_vals >= boundaries[i]) & (x_vals < boundaries[i+1])
                if np.any(mask):
                    new_centroids[i] = np.mean(x_vals[mask])
                else:
                    new_centroids[i] = initial_centroids[i]
            
            if np.max(np.abs(new_centroids - initial_centroids)) < 1e-6:
                break
            initial_centroids = new_centroids
        
        return initial_centroids
    
    @staticmethod
    def quantize_scalar(x: float, codebook: np.ndarray) -> Tuple[int, float]:
        """Quantize a scalar value to nearest codebook entry."""
        distances = np.abs(codebook - x)
        idx = np.argmin(distances)
        return idx, codebook[idx]
    
    @staticmethod
    def quantize_array(arr: np.ndarray, codebook: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize numpy array using codebook."""
        indices = np.zeros(len(arr), dtype=np.int32)
        reconstructed = np.zeros(len(arr), dtype=np.float32)
        
        for i, x in enumerate(arr):
            distances = np.abs(codebook - x)
            idx = np.argmin(distances)
            indices[i] = idx
            reconstructed[i] = codebook[idx]
        
        return indices, reconstructed


class QJLQuantizer:
    """Quantized Johnson-Lindenstrauss transform for 1-bit inner product quantization.
    
    Provides unbiased inner product estimation with minimal distortion.
    QJL map: Q_qjl(x) = sign(S @ x)
    QJL inverse: Q_qjl^{-1}(z) = sqrt(pi/2)/d * S^T @ z
    """
    
    def __init__(self, dim: int, seed: Optional[int] = None):
        self.dim = dim
        self.seed = seed or np.random.randint(0, 2**31)
        
        rng = np.random.RandomState(self.seed)
        self.S = rng.randn(dim, dim).astype(np.float32)
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Apply QJL quantization: sign(S @ x)."""
        return np.sign(self.S @ x).astype(np.int8)
    
    def dequantize(self, z: np.ndarray) -> np.ndarray:
        """Apply QJL dequantization: sqrt(pi/2)/d * S^T @ z."""
        scale = math.sqrt(math.pi / 2) / self.dim
        return scale * (self.S.T @ z)


class TurboQuant:
    """
    TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
    
    Two operating modes:
    1. MSE mode: Minimizes reconstruction MSE
    2. Inner-product mode: Provides unbiased inner product estimates (for attention)
    
    Key features:
    - Random rotation to induce Beta distribution on coordinates
    - Optimal scalar quantization per coordinate (Lloyd-Max)
    - Two-stage design for inner-product mode (MSE + QJL residual)
    - 3.5 bits per channel achieves quality neutrality
    - 2.5 bits per channel for marginal quality degradation
    
    Paper: arXiv:2504.19874v1
    """
    
    def __init__(self, config: Optional[TurboQuantConfig] = None):
        self.config = config or TurboQuantConfig()
        self.dim = self.config.dim
        self.bits = self._compute_bits()
        
        self.rotation_matrix: Optional[np.ndarray] = None
        self.codebook: Optional[np.ndarray] = None
        self.qjl: Optional[QJLQuantizer] = None
        
        self._initialized = False
    
    def _compute_bits(self) -> int:
        """Convert bits per channel to integer bit-width for quantization."""
        bpc = self.config.bits_per_channel
        if bpc <= 1.5:
            return 1
        elif bpc <= 2.5:
            return 2
        elif bpc <= 3.5:
            return 3
        elif bpc <= 4.5:
            return 4
        else:
            return int(math.ceil(bpc))
    
    def _generate_rotation_matrix(self) -> np.ndarray:
        """Generate random orthogonal matrix via QR decomposition."""
        seed = self.config.seed or np.random.randint(0, 2**31)
        rng = np.random.RandomState(seed)
        
        random_matrix = rng.randn(self.dim, self.dim).astype(np.float32)
        q, r = np.linalg.qr(random_matrix)
        
        diag_signs = np.diag(r)
        if diag_signs.size > 0:
            flip = np.where(diag_signs < 0)
            if len(flip) > 0 and len(flip[0]) > 0:
                q[:, flip[0]] *= -1
        
        return q
    
    def _apply_rotation(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation transform."""
        if self.rotation_matrix is None:
            return x
        return x @ self.rotation_matrix.T
    
    def _apply_inverse_rotation(self, x: np.ndarray) -> np.ndarray:
        """Apply inverse rotation transform."""
        if self.rotation_matrix is None:
            return x
        return x @ self.rotation_matrix
    
    def fit(self):
        """Initialize TurboQuant with precomputed codebooks."""
        if self.config.use_rotation:
            self.rotation_matrix = self._generate_rotation_matrix()
        
        mse_bits = self._compute_mse_bits()
        self.codebook = LloydMaxQuantizer.get_codebook(self.dim, mse_bits)
        
        if self.config.mode == TurboQuantMode.INNER_PRODUCT:
            self.qjl = QJLQuantizer(self.dim, self.config.seed)
        
        self._initialized = True
    
    def _compute_mse_bits(self) -> int:
        """Compute MSE bit-width for inner-product mode.
        
        Inner-product mode uses (b-1) bits for MSE quantizer and 1 bit for QJL.
        """
        if self.config.mode == TurboQuantMode.INNER_PRODUCT:
            return max(1, self.bits - 1)
        return self.bits
    
    def quantize(self, x: np.ndarray, norms: Optional[np.ndarray] = None) -> dict:
        """
        Quantize vectors using TurboQuant.
        
        Args:
            x: Input array of shape (n_vectors, dim) or (dim,)
            norms: L2 norms for rescaling (for non-unit vectors)
        
        Returns:
            Dictionary containing quantized representation
        """
        if not self._initialized:
            self.fit()
        
        original_shape = x.shape
        is_1d = x.ndim == 1
        
        if is_1d:
            x = x.reshape(1, -1)
        
        n_vectors = x.shape[0]
        
        if norms is None:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
        
        x_normalized = x / (norms + 1e-8)
        
        if self.config.use_rotation:
            x_rotated = self._apply_rotation(x_normalized)
        else:
            x_rotated = x_normalized
        
        if self.config.mode == TurboQuantMode.MSE:
            return self._quantize_mse(x_rotated, norms, original_shape, is_1d)
        else:
            return self._quantize_inner_product(x_rotated, norms, original_shape, is_1d)
    
    def _quantize_mse(self, x_rotated: np.ndarray, norms: np.ndarray,
                      original_shape: Tuple, is_1d: bool) -> dict:
        """Quantize using MSE-optimal TurboQuant."""
        indices = np.zeros((x_rotated.shape[0], self.dim), dtype=np.int8)
        reconstructed = np.zeros_like(x_rotated)
        
        for i in range(x_rotated.shape[0]):
            for j in range(self.dim):
                idx, val = LloydMaxQuantizer.quantize_scalar(x_rotated[i, j], self.codebook)
                indices[i, j] = idx
                reconstructed[i, j] = val
        
        reconstructed = reconstructed * norms
        
        if self.config.use_rotation:
            reconstructed = self._apply_inverse_rotation(reconstructed)
        
        if is_1d:
            reconstructed = reconstructed[0]
            indices = indices[0]
            norms = norms[0] if norms.ndim > 1 else norms[0, 0]
        
        return {
            'indices': indices,
            'norms': norms,
            'mode': 'mse',
            'original_shape': original_shape,
            'dim': self.dim,
            'bits': self.bits,
            'use_rotation': self.config.use_rotation,
        }
    
    def _quantize_inner_product(self, x_rotated: np.ndarray, norms: np.ndarray,
                                 original_shape: Tuple, is_1d: bool) -> dict:
        """Quantize using Inner-product optimal TurboQuant (MSE + QJL residual)."""
        indices = np.zeros((x_rotated.shape[0], self.dim), dtype=np.int8)
        qjl_signs = np.zeros((x_rotated.shape[0], self.dim), dtype=np.int8)
        residuals_norm = np.zeros(x_rotated.shape[0], dtype=np.float32)
        
        for i in range(x_rotated.shape[0]):
            residual = x_rotated[i].copy()
            
            for j in range(self.dim):
                idx, val = LloydMaxQuantizer.quantize_scalar(residual[j], self.codebook)
                indices[i, j] = idx
                residual[j] -= val
            
            residual_norm = np.linalg.norm(residual)
            residuals_norm[i] = residual_norm
            
            if residual_norm > 1e-8:
                residual_normalized = residual / residual_norm
                qjl_signs[i] = self.qjl.quantize(residual_normalized)
        
        result = {
            'indices': indices,
            'qjl_signs': qjl_signs,
            'norms': norms,
            'residuals_norm': residuals_norm,
            'mode': 'inner_product',
            'original_shape': original_shape,
            'dim': self.dim,
            'bits': self.bits,
            'use_rotation': self.config.use_rotation,
        }
        
        if is_1d:
            for key in ['indices', 'qjl_signs', 'residuals_norm']:
                if result[key].ndim > 1:
                    result[key] = result[key][0]
            if result['norms'].ndim > 1:
                result['norms'] = result['norms'][0]
        
        return result
    
    def dequantize(self, compressed: dict) -> np.ndarray:
        """
        Dequantize vectors back to original space.
        
        Args:
            compressed: Dictionary from quantize()
        
        Returns:
            Reconstructed vectors
        """
        mode = compressed.get('mode', 'mse')
        
        if mode == 'mse':
            return self._dequantize_mse(compressed)
        else:
            return self._dequantize_inner_product(compressed)
    
    def _dequantize_mse(self, compressed: dict) -> np.ndarray:
        """Dequantize MSE-compressed vectors."""
        indices = compressed['indices']
        norms = compressed['norms']
        original_shape = compressed['original_shape']
        
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
            norms = norms.reshape(1, -1) if np.isscalar(norms) or norms.ndim == 1 else norms
        
        reconstructed = np.zeros((indices.shape[0], self.dim), dtype=np.float32)
        
        for i in range(indices.shape[0]):
            for j in range(self.dim):
                idx = indices[i, j]
                reconstructed[i, j] = self.codebook[idx]
        
        reconstructed = reconstructed * norms
        
        if compressed.get('use_rotation', False):
            reconstructed = self._apply_inverse_rotation(reconstructed)
        
        if len(original_shape) == 1:
            reconstructed = reconstructed[0]
        
        return reconstructed
    
    def _dequantize_inner_product(self, compressed: dict) -> np.ndarray:
        """Dequantize inner-product-compressed vectors."""
        indices = compressed['indices']
        qjl_signs = compressed['qjl_signs']
        norms = compressed['norms']
        residuals_norm = compressed['residuals_norm']
        original_shape = compressed['original_shape']
        
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
            qjl_signs = qjl_signs.reshape(1, -1)
            residuals_norm = residuals_norm.reshape(1, -1)
            norms = norms.reshape(1, -1) if np.isscalar(norms) or norms.ndim == 1 else norms
        
        reconstructed = np.zeros((indices.shape[0], self.dim), dtype=np.float32)
        
        for i in range(indices.shape[0]):
            for j in range(self.dim):
                idx = indices[i, j]
                reconstructed[i, j] = self.codebook[idx]
            
            if residuals_norm[i] > 1e-8:
                qjl_contribution = self.qjl.dequantize(qjl_signs[i])
                reconstructed[i] += residuals_norm[i] * qjl_contribution
        
        reconstructed = reconstructed * norms
        
        if compressed.get('use_rotation', False):
            reconstructed = self._apply_inverse_rotation(reconstructed)
        
        if len(original_shape) == 1:
            reconstructed = reconstructed[0]
        
        return reconstructed
    
    @staticmethod
    def estimate_compression_ratio(bits_per_channel: float = 3.5) -> float:
        """Estimate compression ratio for given bits per channel."""
        original_bits = 16
        return original_bits / bits_per_channel
    
    @staticmethod
    def estimate_memory_saving(
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ctx_size: int,
        bits_per_channel: float = 3.5,
    ) -> dict:
        """Estimate memory saving from TurboQuant compression."""
        f16_size = 2 * n_layers * n_heads * head_dim * ctx_size * 2
        compressed_size = 2 * n_layers * n_heads * head_dim * ctx_size * bits_per_channel / 8
        
        return {
            'f16_bytes': f16_size,
            'compressed_bytes': compressed_size,
            'savings_percent': (1 - compressed_size / f16_size) * 100,
            'compression_ratio': f16_size / compressed_size,
        }


class TurboQuantKVCache:
    """
    TurboQuant for KV Cache compression.
    
    Provides extreme KV cache compression with minimal quality loss:
    - 3.5 bits/channel: Quality neutrality
    - 2.5 bits/channel: Marginal quality degradation
    - 5x+ overall compression
    """
    
    def __init__(self, head_dim: int = 128, bits_per_channel: float = 3.5):
        self.head_dim = head_dim
        self.bits_per_channel = bits_per_channel
        
        config = TurboQuantConfig(
            dim=head_dim,
            bits_per_channel=bits_per_channel,
            mode=TurboQuantMode.INNER_PRODUCT,
        )
        self.turboquant = TurboQuant(config)
    
    def compress_kv(self, k_cache: np.ndarray, v_cache: np.ndarray) -> Tuple[dict, dict]:
        """
        Compress KV cache tensors.
        
        Args:
            k_cache: Key cache of shape (batch, heads, seq, head_dim)
            v_cache: Value cache of shape (batch, heads, seq, head_dim)
        
        Returns:
            Tuple of (compressed_k, compressed_v)
        """
        self.turboquant.fit()
        
        original_shape = k_cache.shape
        
        k_flat = k_cache.reshape(-1, self.head_dim)
        v_flat = v_cache.reshape(-1, self.head_dim)
        
        k_norms = np.linalg.norm(k_flat, axis=1, keepdims=True)
        v_norms = np.linalg.norm(v_flat, axis=1, keepdims=True)
        
        k_compressed = self.turboquant.quantize(k_flat, k_norms)
        v_compressed = self.turboquant.quantize(v_flat, v_norms)
        
        k_compressed['original_shape'] = original_shape
        v_compressed['original_shape'] = original_shape
        
        return k_compressed, v_compressed
    
    def decompress_kv(self, k_compressed: dict, v_compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompress KV cache tensors.
        
        Args:
            k_compressed: Compressed K cache
            v_compressed: Compressed V cache
        
        Returns:
            Tuple of (k_cache, v_cache) with original shape
        """
        k_reconstructed = self.turboquant.dequantize(k_compressed)
        v_reconstructed = self.turboquant.dequantize(v_compressed)
        
        original_shape = k_compressed['original_shape']
        
        return (k_reconstructed.reshape(original_shape),
                v_reconstructed.reshape(original_shape))


def benchmark_turboquant():
    """Benchmark TurboQuant compression."""
    print("=" * 70)
    print("TurboQuant KV Cache Compression Benchmark")
    print("=" * 70)
    
    configs = [
        ("F16 (baseline)", 16.0),
        ("Q8", 8.0),
        ("Q4", 4.0),
        ("TurboQuant-3.5", 3.5),
        ("TurboQuant-3", 3.0),
        ("TurboQuant-2.5", 2.5),
        ("TurboQuant-2", 2.0),
    ]
    
    n_layers = 32
    n_heads = 32
    head_dim = 128
    ctx_sizes = [4096, 8192, 16384]
    
    print("\nKV Cache Size Comparison:")
    print("-" * 70)
    
    for ctx in ctx_sizes:
        print(f"\nContext Size: {ctx:,}")
        print(f"{'Config':<20} {'Size (GB)':<15} {'Compression':<15} {'Saving':<10} {'Quality'}")
        print("-" * 70)
        
        f16_size = 2 * n_layers * n_heads * head_dim * ctx * 2 / (1024**3)
        
        for name, bits in configs:
            size = 2 * n_layers * n_heads * head_dim * ctx * bits / 8 / (1024**3)
            compression = f"{f16_size / size:.1f}x"
            saving = f"{(1 - size/f16_size)*100:.1f}%"
            
            if bits == 3.5:
                quality = "Quality neutral"
            elif bits == 2.5:
                quality = "Marginal loss"
            elif bits == 2.0:
                quality = "Some loss"
            elif bits >= 4.0:
                quality = "High quality"
            else:
                quality = ""
            
            print(f"{name:<20} {size:<15.3f} {compression:<15} {saving:<10} {quality}")
    
    print("\n" + "=" * 70)
    print("TurboQuant Key Insights from Paper:")
    print("- 3.5 bits/channel: Achieves quality neutrality (indistinguishable from F16)")
    print("- 2.5 bits/channel: Marginal quality degradation")
    print("- 5x+ compression with near-identical output quality")
    print("- Random rotation + optimal scalar quantization per coordinate")
    print("=" * 70)
    
    print("\nTheoretical Distortion Bounds (per paper):")
    print("-" * 50)
    print(f"{'Bits (b)':<10} {'MSE D_mse':<15} {'Inner-product D_prod':<20}")
    print("-" * 50)
    
    for b in [1, 2, 3, 4]:
        mse_distortion = [0.36, 0.117, 0.03, 0.009][b-1]
        prod_distortion = [1.57, 0.56, 0.18, 0.047][b-1]
        print(f"{b:<10} {mse_distortion:<15.3f} ~{prod_distortion}/d (inner-product)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_turboquant()