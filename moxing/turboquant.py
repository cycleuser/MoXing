"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Based on Google's TurboQuant paper (arXiv:2504.19874v1):
https://arxiv.org/html/2504.19874v1

Key Features:
- Data-oblivious online quantization
- Near-optimal distortion (within 2.7x of theoretical lower bound)
- Two modes: MSE-optimal and Inner-product-optimal (unbiased)
- Mixed precision: 3.5 bits (quality neutral), 2.5 bits (marginal loss)
- 5x+ compression ratio

Algorithm Overview:
1. TurboQuant_MSE (optimize for MSE):
   - Random rotation to induce Beta distribution on coordinates
   - Optimal scalar quantization per coordinate (Lloyd-Max)
   - Distortion: D_mse ≈ 0.36, 0.117, 0.03, 0.009 for b=1,2,3,4

2. TurboQuant_PROD (optimize for inner product, UNBIASED):
   - Two-stage: (b-1) bits MSE quantizer + 1 bit QJL on residual
   - Provides unbiased inner product estimation
   - Distortion: D_prod ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d for b=1,2,3,4

3. Mixed Precision:
   - 3.5 bits: 32 outlier channels @ 4bits + 96 normal @ 3bits
   - 2.5 bits: 32 outlier channels @ 3bits + 96 normal @ 2bits
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.special import gamma as gamma_func
from scipy.integrate import quad


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
    n_outlier_channels: int = 32
    outlier_threshold: float = 0.99


class BetaDistribution:
    """
    Beta distribution for coordinates after random rotation.
    
    After random rotation, each coordinate follows:
    f_X(x) = Γ(d/2) / (sqrt(π) * Γ((d-1)/2)) * (1 - x²)^((d-3)/2)
    
    In high dimensions, this converges to N(0, 1/d).
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        self._normal_approx = dim >= 64
        
    def pdf(self, x: float) -> float:
        """Probability density function."""
        if abs(x) > 1:
            return 0.0
        
        if self._normal_approx:
            return (1.0 / (self.scale * math.sqrt(2 * math.pi))) * \
                   math.exp(-0.5 * (x / self.scale) ** 2)
        
        d = self.dim
        coef = gamma_func(d / 2) / (math.sqrt(math.pi) * gamma_func((d - 1) / 2))
        return coef * ((1 - x * x) ** ((d - 3) / 2))
    
    def sample(self, n: int) -> np.ndarray:
        """Sample from the distribution."""
        if self._normal_approx:
            return np.random.randn(n) * self.scale
        else:
            theta = np.random.uniform(0, 2 * math.pi, n)
            phi = np.random.uniform(0, math.pi, n)
            x = np.sin(phi) * np.cos(theta)
            return x


class LloydMaxQuantizer:
    """
    Optimal scalar quantizer for Beta distribution using Lloyd-Max algorithm.
    
    Solves the continuous k-means problem:
    min_{c_1,...,c_{2^b}} sum_i ∫_{boundary} |x - c_i|² * f_X(x) dx
    
    Precomputed optimal centroids for b=1,2,3,4 based on Normal approximation.
    """
    
    _codebook_cache: Dict[Tuple[int, int], np.ndarray] = {}
    _boundary_cache: Dict[Tuple[int, int], np.ndarray] = {}
    
    OPTIMAL_CENTROIDS = {
        1: [0.7979],  # sqrt(2/π)
        2: [0.453, 1.51],
        3: [0.267, 0.803, 1.49, 2.32],
        4: [0.153, 0.453, 0.803, 1.22, 1.67, 2.17, 2.75, 3.44],
    }
    
    MSE_DISTORTION = {
        1: 0.36,
        2: 0.117,
        3: 0.03,
        4: 0.009,
    }
    
    PROD_DISTORTION = {
        1: 1.57,
        2: 0.56,
        3: 0.18,
        4: 0.047,
    }
    
    @classmethod
    def get_codebook(cls, dim: int, bits: int) -> np.ndarray:
        """Get optimal codebook for given dimension and bit-width."""
        cache_key = (dim, bits)
        
        if cache_key in cls._codebook_cache:
            return cls._codebook_cache[cache_key]
        
        scale = 1.0 / math.sqrt(dim)
        
        if bits in cls.OPTIMAL_CENTROIDS:
            centroids = cls.OPTIMAL_CENTROIDS[bits]
            codebook = []
            for c in centroids:
                codebook.extend([-c * scale, c * scale])
            codebook = np.array(sorted(codebook), dtype=np.float32)
        else:
            codebook = cls._compute_codebook_lloyd_max(dim, bits)
        
        cls._codebook_cache[cache_key] = codebook
        return codebook
    
    @classmethod
    def get_boundaries(cls, dim: int, bits: int) -> np.ndarray:
        """Get quantization boundaries for given dimension and bit-width."""
        cache_key = (dim, bits)
        
        if cache_key in cls._boundary_cache:
            return cls._boundary_cache[cache_key]
        
        codebook = cls.get_codebook(dim, bits)
        n_levels = len(codebook)
        
        boundaries = np.zeros(n_levels + 1, dtype=np.float32)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        
        for i in range(1, n_levels):
            boundaries[i] = (codebook[i-1] + codebook[i]) / 2
        
        cls._boundary_cache[cache_key] = boundaries
        return boundaries
    
    @classmethod
    def _compute_codebook_lloyd_max(cls, dim: int, bits: int, 
                                     n_iter: int = 100) -> np.ndarray:
        """Compute optimal codebook using Lloyd-Max iterations."""
        n_levels = 2 ** bits
        scale = 1.0 / math.sqrt(dim)
        std = 1.0
        
        centroids = np.linspace(-3 * std, 3 * std, n_levels) * scale
        
        beta_dist = BetaDistribution(dim)
        
        for _ in range(n_iter):
            boundaries = np.zeros(n_levels + 1)
            boundaries[0] = -np.inf
            boundaries[-1] = np.inf
            
            for i in range(1, n_levels):
                boundaries[i] = (centroids[i-1] + centroids[i]) / 2
            
            new_centroids = np.zeros(n_levels)
            
            for i in range(n_levels):
                lower = boundaries[i]
                upper = boundaries[i + 1]
                
                def weighted_x(x):
                    return x * beta_dist.pdf(x)
                
                def weight(x):
                    return beta_dist.pdf(x)
                
                numerator, _ = quad(weighted_x, lower, upper, limit=100)
                denominator, _ = quad(weight, lower, upper, limit=100)
                
                if denominator > 1e-10:
                    new_centroids[i] = numerator / denominator
                else:
                    new_centroids[i] = centroids[i]
            
            if np.max(np.abs(new_centroids - centroids)) < 1e-8:
                break
            
            centroids = new_centroids
        
        return centroids.astype(np.float32)
    
    @classmethod
    def quantize(cls, x: np.ndarray, codebook: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize array to nearest codebook entries."""
        x_flat = x.flatten()
        indices = np.zeros(len(x_flat), dtype=np.int32)
        reconstructed = np.zeros(len(x_flat), dtype=np.float32)
        
        for i, val in enumerate(x_flat):
            distances = np.abs(codebook - val)
            idx = np.argmin(distances)
            indices[i] = idx
            reconstructed[i] = codebook[idx]
        
        return indices.reshape(x.shape), reconstructed.reshape(x.shape)
    
    @classmethod
    def dequantize(cls, indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Dequantize indices back to values."""
        return codebook[indices]


class QJLQuantizer:
    """
    Quantized Johnson-Lindenstrauss transform for 1-bit inner product quantization.
    
    Provides UNBIASED inner product estimation:
    E[⟨y, QJL⁻¹(QJL(x))⟩] = ⟨y, x⟩
    
    Algorithm:
    - Quantize: Q_qjl(x) = sign(S @ x)  where S ~ N(0, I)
    - Dequantize: Q_qjl⁻¹(z) = sqrt(π/2)/d * Sᵀ @ z
    
    Variance bound: Var ≤ π/(2d) * ||y||²
    """
    
    def __init__(self, dim: int, seed: Optional[int] = None):
        self.dim = dim
        self.seed = seed or np.random.randint(0, 2**31)
        
        rng = np.random.RandomState(self.seed)
        self.S = rng.randn(dim, dim).astype(np.float32) / np.sqrt(dim)
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize to binary signs: sign(S @ x)."""
        if x.ndim == 1:
            projected = self.S @ x
        else:
            projected = x @ self.S.T
        
        return np.sign(projected).astype(np.int8)
    
    def dequantize(self, z: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Dequantize: sqrt(π/2)/d * scale * Sᵀ @ z."""
        deq_scale = math.sqrt(math.pi / 2) / self.dim * scale
        
        if z.ndim == 1:
            return deq_scale * (self.S.T @ z)
        else:
            return deq_scale * (z @ self.S)
    
    def inner_product(self, z1: np.ndarray, z2: np.ndarray, 
                      scale1: float = 1.0, scale2: float = 1.0) -> float:
        """Estimate inner product from quantized vectors."""
        if z1.ndim == 1:
            hamming = np.sum(z1 != z2)
            cos_theta = 1 - 2 * hamming / self.dim
        else:
            hamming = np.sum(z1 != z2, axis=1)
            cos_theta = 1 - 2 * hamming / self.dim
        
        return scale1 * scale2 * cos_theta * math.pi / 2


class TurboQuant:
    """
    TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
    
    Two operating modes:
    1. MSE mode: Minimizes reconstruction MSE
       - Random rotation + Lloyd-Max scalar quantization
       - D_mse ≤ sqrt(3)π/2 * 4^(-b)
    
    2. Inner-product mode: Unbiased inner product estimation
       - Two-stage: (b-1) bits MSE + 1 bit QJL residual
       - E[⟨y, x̂⟩] = ⟨y, x⟩ (unbiased!)
       - D_prod ≤ sqrt(3)π²/d * ||y||² * 4^(-b)
    
    Mixed precision support:
    - 3.5 bits: 32 outlier channels @ 4bits + 96 normal @ 3bits (quality neutral)
    - 2.5 bits: 32 outlier channels @ 3bits + 96 normal @ 2bits (marginal loss)
    """
    
    def __init__(self, config: Optional[TurboQuantConfig] = None):
        self.config = config or TurboQuantConfig()
        self.dim = self.config.dim
        self.bits = self.config.bits_per_channel
        
        self.rotation_matrix: Optional[np.ndarray] = None
        self.codebook_mse: Optional[np.ndarray] = None
        self.codebook_residual: Optional[np.ndarray] = None
        self.qjl: Optional[QJLQuantizer] = None
        
        self._initialized = False
    
    def _generate_rotation_matrix(self) -> np.ndarray:
        """Generate random orthogonal matrix via QR decomposition."""
        seed = self.config.seed or np.random.randint(0, 2**31)
        rng = np.random.RandomState(seed)
        
        random_matrix = rng.randn(self.dim, self.dim).astype(np.float32)
        q, r = np.linalg.qr(random_matrix)
        
        d = np.diag(r)
        q[:, d < 0] *= -1
        
        return q
    
    def _get_mse_bits(self, total_bits: float) -> int:
        """Get MSE bit-width for inner-product mode (b-1 bits)."""
        if self.config.mode == TurboQuantMode.INNER_PRODUCT:
            return max(1, int(total_bits) - 1)
        return int(total_bits)
    
    def fit(self):
        """Initialize TurboQuant with rotation matrix and codebooks."""
        if self.config.use_rotation:
            self.rotation_matrix = self._generate_rotation_matrix()
        
        if self.config.mode == TurboQuantMode.INNER_PRODUCT:
            mse_bits = self._get_mse_bits(self.bits)
            self.codebook_mse = LloydMaxQuantizer.get_codebook(self.dim, mse_bits)
            self.qjl = QJLQuantizer(self.dim, self.config.seed)
        else:
            self.codebook_mse = LloydMaxQuantizer.get_codebook(self.dim, int(self.bits))
        
        self._initialized = True
    
    def _rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply random rotation."""
        if self.rotation_matrix is None:
            return x
        if x.ndim == 1:
            return self.rotation_matrix @ x
        return x @ self.rotation_matrix.T
    
    def _inverse_rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply inverse rotation."""
        if self.rotation_matrix is None:
            return x
        if x.ndim == 1:
            return self.rotation_matrix.T @ x
        return x @ self.rotation_matrix
    
    def quantize(self, x: np.ndarray, 
                 norms: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Quantize vectors using TurboQuant.
        
        Args:
            x: Input vectors, shape (n, dim) or (dim,)
            norms: Optional L2 norms for rescaling
        
        Returns:
            Dictionary with quantized representation
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
        elif norms.ndim == 1:
            norms = norms.reshape(-1, 1)
        
        x_normalized = x / (norms + 1e-10)
        
        if self.config.use_rotation:
            x_rotated = self._rotate(x_normalized)
        else:
            x_rotated = x_normalized.copy()
        
        if self.config.mode == TurboQuantMode.MSE:
            result = self._quantize_mse(x_rotated, norms)
        else:
            result = self._quantize_inner_product(x_rotated, norms)
        
        result['original_shape'] = original_shape
        result['is_1d'] = is_1d
        
        return result
    
    def _quantize_mse(self, x_rotated: np.ndarray, 
                      norms: np.ndarray) -> Dict[str, Any]:
        """MSE-optimal quantization."""
        indices, reconstructed = LloydMaxQuantizer.quantize(
            x_rotated, self.codebook_mse
        )
        
        return {
            'indices': indices,
            'norms': norms.flatten(),
            'mode': 'mse',
            'bits': int(self.bits),
            'use_rotation': self.config.use_rotation,
        }
    
    def _quantize_inner_product(self, x_rotated: np.ndarray,
                                 norms: np.ndarray) -> Dict[str, Any]:
        """Inner-product-optimal quantization (UNBIASED)."""
        n_vectors = x_rotated.shape[0]
        
        mse_bits = self._get_mse_bits(self.bits)
        n_levels = 2 ** mse_bits
        
        indices = np.zeros((n_vectors, self.dim), dtype=np.int32)
        qjl_signs = np.zeros((n_vectors, self.dim), dtype=np.int8)
        residual_norms = np.zeros(n_vectors, dtype=np.float32)
        
        for i in range(n_vectors):
            vec = x_rotated[i]
            idx, reconstructed = LloydMaxQuantizer.quantize(vec, self.codebook_mse)
            indices[i] = idx
            
            residual = vec - reconstructed
            residual_norm = np.linalg.norm(residual)
            residual_norms[i] = residual_norm
            
            if residual_norm > 1e-8:
                residual_normalized = residual / residual_norm
                qjl_signs[i] = self.qjl.quantize(residual_normalized)
        
        return {
            'indices': indices,
            'qjl_signs': qjl_signs,
            'residual_norms': residual_norms,
            'norms': norms.flatten(),
            'mode': 'inner_product',
            'bits': int(self.bits),
            'mse_bits': mse_bits,
            'use_rotation': self.config.use_rotation,
        }
    
    def dequantize(self, compressed: Dict[str, Any]) -> np.ndarray:
        """Dequantize vectors back to original space."""
        mode = compressed.get('mode', 'mse')
        
        if mode == 'mse':
            return self._dequantize_mse(compressed)
        else:
            return self._dequantize_inner_product(compressed)
    
    def _dequantize_mse(self, compressed: Dict[str, Any]) -> np.ndarray:
        """Dequantize MSE-compressed vectors."""
        indices = compressed['indices']
        norms = compressed['norms']
        
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
        
        reconstructed = LloydMaxQuantizer.dequantize(indices, self.codebook_mse)
        
        reconstructed = reconstructed * norms.reshape(-1, 1)
        
        if compressed.get('use_rotation', False):
            reconstructed = self._inverse_rotate(reconstructed)
        
        if compressed.get('is_1d', False):
            reconstructed = reconstructed[0]
        
        return reconstructed.astype(np.float32)
    
    def _dequantize_inner_product(self, compressed: Dict[str, Any]) -> np.ndarray:
        """Dequantize inner-product-compressed vectors (UNBIASED)."""
        indices = compressed['indices']
        qjl_signs = compressed['qjl_signs']
        residual_norms = compressed['residual_norms']
        norms = compressed['norms']
        
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
            qjl_signs = qjl_signs.reshape(1, -1)
        
        n_vectors = indices.shape[0]
        reconstructed = np.zeros((n_vectors, self.dim), dtype=np.float32)
        
        for i in range(n_vectors):
            reconstructed[i] = LloydMaxQuantizer.dequantize(
                indices[i], self.codebook_mse
            )
            
            if residual_norms[i] > 1e-8:
                qjl_contrib = self.qjl.dequantize(qjl_signs[i], residual_norms[i])
                reconstructed[i] += qjl_contrib
        
        reconstructed = reconstructed * norms.reshape(-1, 1)
        
        if compressed.get('use_rotation', False):
            reconstructed = self._inverse_rotate(reconstructed)
        
        if compressed.get('is_1d', False):
            reconstructed = reconstructed[0]
        
        return reconstructed.astype(np.float32)
    
    @staticmethod
    def get_distortion_bounds(bits: int, dim: int = 1) -> Dict[str, float]:
        """Get theoretical distortion bounds for given bit-width."""
        mse = LloydMaxQuantizer.MSE_DISTORTION.get(bits, 0.36 / (4 ** (bits - 1)))
        prod = LloydMaxQuantizer.PROD_DISTORTION.get(bits, 1.57 / (4 ** (bits - 1)))
        
        return {
            'mse_upper': mse,
            'mse_lower': 1.0 / (4 ** bits),
            'prod_upper': prod / dim if dim > 0 else prod,
            'prod_lower': 1.0 / (dim * 4 ** bits) if dim > 0 else 1.0 / (4 ** bits),
        }


class TurboQuantMixedPrecision:
    """
    Mixed-precision TurboQuant for KV cache compression.
    
    Implements the paper's mixed-precision strategy:
    - 3.5 bits: 32 outlier channels @ 4bits + 96 normal @ 3bits
    - 2.5 bits: 32 outlier channels @ 3bits + 96 normal @ 2bits
    
    Achieves quality neutrality at 3.5 bits, marginal loss at 2.5 bits.
    """
    
    def __init__(self, dim: int = 128, bits_per_channel: float = 3.5,
                 n_outliers: int = 32, seed: Optional[int] = None):
        self.dim = dim
        self.bits = bits_per_channel
        self.n_outliers = n_outliers
        self.seed = seed
        
        self._setup_quantizers()
    
    def _setup_quantizers(self):
        """Setup separate quantizers for outlier and normal channels."""
        if self.bits >= 3.5:
            outlier_bits = 4
            normal_bits = 3
        elif self.bits >= 2.5:
            outlier_bits = 3
            normal_bits = 2
        else:
            outlier_bits = max(1, int(self.bits))
            normal_bits = outlier_bits
        
        self.outlier_quantizer = TurboQuant(TurboQuantConfig(
            dim=self.n_outliers,
            bits_per_channel=outlier_bits,
            mode=TurboQuantMode.INNER_PRODUCT,
            seed=self.seed,
        ))
        
        self.normal_quantizer = TurboQuant(TurboQuantConfig(
            dim=self.dim - self.n_outliers,
            bits_per_channel=normal_bits,
            mode=TurboQuantMode.INNER_PRODUCT,
            seed=self.seed + 1 if self.seed else None,
        ))
        
        self.outlier_indices: Optional[np.ndarray] = None
    
    def identify_outliers(self, x: np.ndarray) -> np.ndarray:
        """Identify outlier channels based on magnitude variance."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        channel_vars = np.var(np.abs(x), axis=0)
        
        threshold = np.percentile(channel_vars, 
                                   100 * (1 - self.n_outliers / self.dim))
        
        outlier_mask = channel_vars >= threshold
        return outlier_mask
    
    def quantize(self, x: np.ndarray, 
                 norms: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Quantize with mixed precision."""
        if self.outlier_indices is None:
            self.outlier_indices = self.identify_outliers(x)
        
        if norms is None:
            norms = np.linalg.norm(x, axis=1 if x.ndim > 1 else 0, keepdims=True)
        
        normal_mask = ~self.outlier_indices
        
        x_outliers = x[..., self.outlier_indices]
        x_normal = x[..., normal_mask]
        
        outlier_compressed = self.outlier_quantizer.quantize(x_outliers)
        normal_compressed = self.normal_quantizer.quantize(x_normal)
        
        return {
            'outlier_compressed': outlier_compressed,
            'normal_compressed': normal_compressed,
            'outlier_indices': self.outlier_indices,
            'norms': norms.flatten() if norms.ndim > 1 else norms,
            'bits': self.bits,
            'original_shape': x.shape,
        }
    
    def dequantize(self, compressed: Dict[str, Any]) -> np.ndarray:
        """Dequantize mixed-precision compressed vectors."""
        outlier_reconstructed = self.outlier_quantizer.dequantize(
            compressed['outlier_compressed']
        )
        normal_reconstructed = self.normal_quantizer.dequantize(
            compressed['normal_compressed']
        )
        
        original_shape = compressed['original_shape']
        result = np.zeros(original_shape, dtype=np.float32)
        
        outlier_indices = compressed['outlier_indices']
        normal_indices = ~outlier_indices
        
        if result.ndim == 1:
            result[outlier_indices] = outlier_reconstructed
            result[normal_indices] = normal_reconstructed
        else:
            result[:, outlier_indices] = outlier_reconstructed
            result[:, normal_indices] = normal_reconstructed
        
        return result


def benchmark_turboquant_comprehensive():
    """Comprehensive TurboQuant benchmark."""
    print("=" * 80)
    print("TurboQuant: Complete KV Cache Quantization Analysis")
    print("=" * 80)
    
    print("\n1. Theoretical Distortion Bounds (from paper)")
    print("-" * 60)
    print(f"{'Bits':<8} {'MSE Upper':<15} {'MSE Lower':<15} {'Ratio':<10}")
    print("-" * 60)
    
    for b in [1, 2, 3, 4]:
        bounds = TurboQuant.get_distortion_bounds(b)
        ratio = bounds['mse_upper'] / bounds['mse_lower']
        print(f"{b:<8} {bounds['mse_upper']:<15.4f} {bounds['mse_lower']:<15.4f} {ratio:<10.2f}x")
    
    print("\nTurboQuant achieves distortion within ~2.7x of theoretical optimum!")
    print("At b=1, it's within 1.45x - very close to optimal.")
    
    print("\n" + "=" * 80)
    print("2. KV Cache Memory Estimation")
    print("-" * 80)
    
    configs = [
        ("F16 (baseline)", 16.0),
        ("Q8", 8.0),
        ("Q4", 4.0),
        ("TurboQuant-4", 4.0),
        ("TurboQuant-3.5", 3.5),
        ("TurboQuant-3", 3.0),
        ("TurboQuant-2.5", 2.5),
        ("TurboQuant-2", 2.0),
    ]
    
    n_layers = 32
    n_heads = 32
    head_dim = 128
    ctx_sizes = [4096, 8192, 16384, 32768, 65536]
    
    print(f"\nModel: {n_layers} layers, {n_heads} heads, {head_dim} dim")
    print(f"\n{'Config':<20} {'4K':<10} {'16K':<10} {'64K':<10} {'Compression':<12} {'Quality'}")
    print("-" * 80)
    
    for name, bits in configs:
        sizes = []
        for ctx in ctx_sizes:
            size_gb = 2 * n_layers * n_heads * head_dim * ctx * bits / 8 / (1024**3)
            sizes.append(size_gb)
        
        compression = 16.0 / bits
        quality = "Quality neutral" if bits == 3.5 else \
                  "Marginal loss" if bits == 2.5 else \
                  "High quality" if bits >= 4 else \
                  "Some loss"
        
        print(f"{name:<20} {sizes[0]:<10.2f} {sizes[2]:<10.2f} {sizes[4]:<10.2f} {compression:<12.1f}x {quality}")
    
    print("\n" + "=" * 80)
    print("3. Practical Recommendations")
    print("-" * 80)
    print("""
• For quality-critical applications:
  Use TurboQuant-3.5 (mixed precision: 4-bit outliers + 3-bit normal)
  - Achieves quality neutrality (indistinguishable from F16)
  - 4.5x compression
  
• For memory-constrained scenarios:
  Use TurboQuant-2.5 (mixed precision: 3-bit outliers + 2-bit normal)
  - Marginal quality degradation
  - 6.4x compression
  
• For extreme compression:
  Use TurboQuant-2 (2-bit uniform)
  - Acceptable quality loss for many tasks
  - 8x compression
""")
    
    print("=" * 80)


if __name__ == "__main__":
    benchmark_turboquant_comprehensive()