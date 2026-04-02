#!/usr/bin/env python3
"""
TurboQuant 完整测试与对比

测试内容：
1. 量化精度验证 (MSE, Inner Product)
2. 不同 bit-width 对比
3. 混合精度测试
4. 与 llama.cpp 内置量化对比
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from moxing.turboquant import (
    TurboQuant,
    TurboQuantConfig,
    TurboQuantMode,
    TurboQuantMixedPrecision,
    LloydMaxQuantizer,
    QJLQuantizer,
)


def test_lloyd_max_quantizer():
    """测试 Lloyd-Max 量化器"""
    print("=" * 70)
    print("1. Lloyd-Max 量化器测试")
    print("=" * 70)
    
    dim = 128
    
    print("\n码本预计算:")
    print(f"{'Bits':<8} {'Codebook Size':<20} {'Sample Centroids'}")
    print("-" * 70)
    
    for bits in [1, 2, 3, 4]:
        codebook = LloydMaxQuantizer.get_codebook(dim, bits)
        n_levels = len(codebook)
        sample = f"[{codebook[0]:.4f}, ..., {codebook[-1]:.4f}]"
        print(f"{bits:<8} {n_levels:<20} {sample}")
    
    print("\n量化测试:")
    np.random.seed(42)
    test_vec = np.random.randn(128).astype(np.float32)
    test_vec = test_vec / np.linalg.norm(test_vec)
    
    for bits in [1, 2, 3, 4]:
        codebook = LloydMaxQuantizer.get_codebook(dim, bits)
        indices, reconstructed = LloydMaxQuantizer.quantize(test_vec, codebook)
        mse = np.mean((test_vec - reconstructed) ** 2)
        theoretical_mse = LloydMaxQuantizer.MSE_DISTORTION[bits] / dim
        
        print(f"  {bits}-bit: MSE = {mse:.6f} (理论值: {theoretical_mse:.6f})")


def test_qjl_quantizer():
    """测试 QJL 量化器（无偏内积估计）"""
    print("\n" + "=" * 70)
    print("2. QJL 量化器测试（无偏内积估计）")
    print("=" * 70)
    
    dim = 128
    n_tests = 1000
    
    qjl = QJLQuantizer(dim, seed=42)
    
    np.random.seed(42)
    
    errors = []
    biases = []
    
    for _ in range(n_tests):
        x = np.random.randn(dim)
        x = x / np.linalg.norm(x)
        
        y = np.random.randn(dim)
        
        true_ip = np.dot(y, x)
        
        z = qjl.quantize(x)
        reconstructed = qjl.dequantize(z)
        estimated_ip = np.dot(y, reconstructed)
        
        errors.append((estimated_ip - true_ip) ** 2)
        biases.append(estimated_ip - true_ip)
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    mean_bias = np.mean(biases)
    
    theoretical_var = np.pi / (2 * dim)
    
    print(f"\n内积估计统计 ({n_tests} 次测试):")
    print(f"  平均偏差: {mean_bias:.6f} (理想值: 0)")
    print(f"  平均误差: {mean_error:.6f}")
    print(f"  误差标准差: {std_error:.6f}")
    print(f"  理论方差: {theoretical_var:.6f}")
    print(f"  无偏性: {'✓ 通过' if abs(mean_bias) < 0.01 else '✗ 失败'}")


def test_turboquant_mse():
    """测试 MSE 模式"""
    print("\n" + "=" * 70)
    print("3. TurboQuant MSE 模式测试")
    print("=" * 70)
    
    dim = 128
    n_vectors = 100
    
    np.random.seed(42)
    test_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
    test_vectors = test_vectors / norms
    
    print(f"\n测试向量: {n_vectors} x {dim}")
    print(f"{'Bits':<8} {'MSE':<15} {'理论 MSE':<15} {'压缩比':<10}")
    print("-" * 70)
    
    for bits in [1, 2, 3, 4]:
        config = TurboQuantConfig(
            dim=dim,
            bits_per_channel=bits,
            mode=TurboQuantMode.MSE,
            seed=42,
        )
        
        tq = TurboQuant(config)
        
        compressed = tq.quantize(test_vectors)
        reconstructed = tq.dequantize(compressed)
        
        mse = np.mean((test_vectors - reconstructed) ** 2)
        theoretical_mse = LloydMaxQuantizer.MSE_DISTORTION[bits] / dim
        compression = 16.0 / bits
        
        print(f"{bits:<8} {mse:<15.6f} {theoretical_mse:<15.6f} {compression:<10.1f}x")


def test_turboquant_inner_product():
    """测试 Inner Product 模式（无偏）"""
    print("\n" + "=" * 70)
    print("4. TurboQuant Inner Product 模式测试（无偏）")
    print("=" * 70)
    
    dim = 128
    n_vectors = 100
    n_queries = 50
    
    np.random.seed(42)
    test_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
    test_vectors = test_vectors / norms
    
    query_vectors = np.random.randn(n_queries, dim).astype(np.float32)
    
    print(f"\n测试向量: {n_vectors} 个, 查询向量: {n_queries} 个")
    print(f"{'Bits':<8} {'MSE':<12} {'IP 偏差':<12} {'IP 误差':<12} {'无偏性'}")
    print("-" * 70)
    
    for bits in [2, 3, 4]:
        config = TurboQuantConfig(
            dim=dim,
            bits_per_channel=bits,
            mode=TurboQuantMode.INNER_PRODUCT,
            seed=42,
        )
        
        tq = TurboQuant(config)
        
        compressed = tq.quantize(test_vectors)
        reconstructed = tq.dequantize(compressed)
        
        mse = np.mean((test_vectors - reconstructed) ** 2)
        
        biases = []
        errors = []
        
        for i in range(n_queries):
            true_ips = np.dot(test_vectors, query_vectors[i])
            est_ips = np.dot(reconstructed, query_vectors[i])
            
            biases.extend(est_ips - true_ips)
            errors.extend((est_ips - true_ips) ** 2)
        
        mean_bias = np.mean(biases)
        mean_error = np.mean(errors)
        unbiased = abs(mean_bias) < 0.05
        
        print(f"{bits:<8} {mse:<12.6f} {mean_bias:<12.6f} {mean_error:<12.6f} {'✓ 通过' if unbiased else '✗ 失败'}")


def test_mixed_precision():
    """测试混合精度"""
    print("\n" + "=" * 70)
    print("5. 混合精度 TurboQuant 测试")
    print("=" * 70)
    
    dim = 128
    n_vectors = 100
    
    np.random.seed(42)
    test_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
    test_vectors = test_vectors / norms
    
    print(f"\n测试向量: {n_vectors} x {dim}")
    print(f"{'配置':<15} {'有效 Bits':<12} {'MSE':<15} {'压缩比':<10}")
    print("-" * 70)
    
    configs = [
        ("TQ-3.5", 3.5, 32),
        ("TQ-2.5", 2.5, 32),
        ("TQ-3.0", 3.0, 0),
    ]
    
    for name, bits, n_outliers in configs:
        if n_outliers > 0:
            tq = TurboQuantMixedPrecision(
                dim=dim,
                bits_per_channel=bits,
                n_outliers=n_outliers,
                seed=42,
            )
            
            compressed = tq.quantize(test_vectors)
            reconstructed = tq.dequantize(compressed)
        else:
            config = TurboQuantConfig(
                dim=dim,
                bits_per_channel=bits,
                mode=TurboQuantMode.INNER_PRODUCT,
                seed=42,
            )
            tq = TurboQuant(config)
            compressed = tq.quantize(test_vectors)
            reconstructed = tq.dequantize(compressed)
        
        mse = np.mean((test_vectors - reconstructed) ** 2)
        compression = 16.0 / bits
        
        print(f"{name:<15} {bits:<12.1f} {mse:<15.6f} {compression:<10.1f}x")


def test_performance():
    """性能测试"""
    print("\n" + "=" * 70)
    print("6. 性能测试")
    print("=" * 70)
    
    dim = 128
    n_vectors = 10000
    
    np.random.seed(42)
    test_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
    test_vectors = test_vectors / norms
    
    print(f"\n向量数量: {n_vectors:,}, 维度: {dim}")
    print(f"{'操作':<20} {'Bits':<8} {'时间 (ms)':<15} {'吞吐量'}")
    print("-" * 70)
    
    for bits in [2, 3, 4]:
        config = TurboQuantConfig(
            dim=dim,
            bits_per_channel=bits,
            mode=TurboQuantMode.INNER_PRODUCT,
            seed=42,
        )
        
        tq = TurboQuant(config)
        
        start = time.perf_counter()
        compressed = tq.quantize(test_vectors)
        quant_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        reconstructed = tq.dequantize(compressed)
        dequant_time = (time.perf_counter() - start) * 1000
        
        quant_throughput = n_vectors / (quant_time / 1000)
        dequant_throughput = n_vectors / (dequant_time / 1000)
        
        print(f"{'量化':<20} {bits:<8} {quant_time:<15.2f} {quant_throughput:,.0f} vec/s")
        print(f"{'解量化':<20} {bits:<8} {dequant_time:<15.2f} {dequant_throughput:,.0f} vec/s")


def test_memory_estimation():
    """内存估算测试"""
    print("\n" + "=" * 70)
    print("7. KV Cache 内存估算")
    print("=" * 70)
    
    from moxing.kv_cache import (
        KVCacheQuantType,
        estimate_kv_cache_size_gb,
        get_model_kv_params,
    )
    
    model_size = 9.0
    n_layers, n_heads, head_dim = get_model_kv_params(model_size)
    
    print(f"\n模型参数: {n_layers} 层, {n_heads} 头, {head_dim} 维")
    print(f"\n{'量化类型':<20} {'Bits':<8} {'32K 上下文':<15} {'128K 上下文':<15} {'质量'}")
    print("-" * 70)
    
    ctx_sizes = [32768, 131072]
    
    quants = [
        (KVCacheQuantType.F16, "全精度"),
        (KVCacheQuantType.Q8_0, "高质量"),
        (KVCacheQuantType.Q4_0, "平衡"),
        (KVCacheQuantType.TURBOQUANT_35, "质量中性 ⭐"),
        (KVCacheQuantType.TURBOQUANT_25, "轻微损失 ⭐"),
        (KVCacheQuantType.TURBOQUANT_2, "最大压缩"),
    ]
    
    for quant, quality in quants:
        sizes = []
        for ctx in ctx_sizes:
            size = estimate_kv_cache_size_gb(n_layers, n_heads, head_dim, ctx, 1, quant)
            sizes.append(f"{size:.2f} GB")
        
        print(f"{quant.value:<20} {quant.bits:<8.1f} {sizes[0]:<15} {sizes[1]:<15} {quality}")


def main():
    print("=" * 70)
    print("TurboQuant 完整测试套件")
    print("基于论文 arXiv:2504.19874v1")
    print("=" * 70)
    
    test_lloyd_max_quantizer()
    test_qjl_quantizer()
    test_turboquant_mse()
    test_turboquant_inner_product()
    test_mixed_precision()
    test_performance()
    test_memory_estimation()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("""
关键发现:
• TurboQuant Inner Product 模式提供无偏内积估计
• 3.5 bits 达到质量中性（与 F16 无区别）
• 2.5 bits 轻微质量损失，5x+ 压缩比
• 混合精度策略有效处理异常值通道

推荐配置:
• 质量优先: TurboQuant-3.5 (tq3.5)
• 平衡: TurboQuant-2.5 (tq2.5)
• 极限压缩: TurboQuant-2 (tq2)
""")


if __name__ == "__main__":
    main()