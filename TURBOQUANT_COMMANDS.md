# TurboQuant 命令示例

## 快速开始

### 查看帮助信息

```bash
moxing turboquant info
```

输出：
```
Available Quantization Types:
┏━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Type  ┃ Bits ┃ Compression ┃ Quality            ┃ Use Case            ┃
┡━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ f16   │ 16   │ 1x          │ Perfect            │ Baseline            │
│ q8_0  │ 8    │ 2x          │ Excellent          │ Quality-first       │
│ q5_0  │ 5    │ 3.2x        │ Very Good          │ Good balance        │
│ q4_0  │ 4    │ 4x          │ Good               │ Recommended         │
│ tq4   │ 4    │ 4x          │ High               │ TurboQuant 4-bit    │
│ tq3.5 │ 3.5  │ 4.6x        │ Quality Neutral ⭐ │ Best for quality    │
│ tq3   │ 3    │ 5.3x        │ Good               │ TurboQuant 3-bit    │
│ tq2.5 │ 2.5  │ 6.4x        │ Slight Loss ⭐     │ Best for memory     │
│ tq2   │ 2    │ 8x          │ Acceptable         │ Maximum compression │
└───────┴──────┴─────────────┴────────────────────┴─────────────────────┘
```

---

## 启动服务器

### 1. 质量 neutral (推荐)

3.5-bit 混合精度，与 F16 无区别：

```bash
moxing serve model.gguf --kv-cache tq3.5 -c 32768
```

### 2. 内存优化

2.5-bit 混合精度，轻微质量损失：

```bash
moxing serve model.gguf --kv-cache tq2.5 -c 65536
```

### 3. 最大压缩

2-bit 均匀量化：

```bash
moxing serve model.gguf --kv-cache tq2 -c 131072
```

### 4. 使用 llama.cpp 内置 q4_0 (推荐)

```bash
moxing serve model.gguf --kv-cache q4_0 -c 65536
```

---

## Ollama 模型

### 服务 Ollama 模型

```bash
# 质量 neutral
moxing ollama serve llama3 --kv-cache tq3.5

# 内存优化
moxing ollama serve gemma3:1b --kv-cache tq2.5 -c 65536

# 最大上下文 (256K)
moxing ollama serve carstenuhlig/omnicoder-9b --kv-cache q4_0 -c 262144
```

---

## 内存估算

### 估算 KV Cache 内存

```bash
# 9B 模型，65K 上下文
moxing turboquant estimate 9 -c 65536

# 70B 模型，32K 上下文
moxing turboquant estimate 70 -c 32768
```

输出：
```
            KV Cache Memory (Context: 65,536)            
┏━━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Type  ┃ Bits ┃ KV Size  ┃ vs F16 ┃ Quality            ┃
┡━━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ f16   │ 16.0 │ 32.00 GB │ -      │ Baseline           │
│ q8_0  │ 8.0  │ 16.00 GB │ -50%   │ Excellent          │
│ q5_0  │ 5.0  │ 10.00 GB │ -69%   │ Very Good          │
│ q4_0  │ 4.0  │ 8.00 GB  │ -75%   │ Good               │
│ tq3.5 │ 3.5  │ 7.00 GB  │ -78%   │ Quality Neutral ⭐ │
│ tq2.5 │ 2.5  │ 5.00 GB  │ -84%   │ Slight Loss ⭐     │
│ tq2   │ 2.0  │ 4.00 GB  │ -88%   │ Acceptable         │
└───────┴──────┴──────────┴────────┴────────────────────┘
```

---

## 算法测试

### 测试 TurboQuant 算法

```bash
# 3.5-bit 测试
moxing turboquant test -b 3.5 -n 100

# 2.5-bit 测试
moxing turboquant test -b 2.5 -n 1000 -d 256
```

输出：
```
╭────────────────────────── TurboQuant Test Results ───────────────────────────╮
│ Configuration:                                                               │
│   Dimension: 128                                                             │
│   Vectors: 100                                                               │
│   Bits: 3.5                                                                  │
│                                                                              │
│ Results:                                                                     │
│   MSE: 0.002443                                                              │
│   Inner Product Bias: 0.007275 (ideal: 0)                                    │
│   Quantize Time: 29.64 ms                                                    │
│   Dequantize Time: 1.25 ms                                                   │
│   Compression: 4.6x                                                          │
│                                                                              │
│ Unbiased: ✓ Yes                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Python API

### 使用 TurboQuant

```python
from moxing import TurboQuant, TurboQuantConfig, TurboQuantMode
import numpy as np

# 配置
config = TurboQuantConfig(
    dim=128,                    # KV head 维度
    bits_per_channel=3.5,       # 3.5 bits = 质量中性
    mode=TurboQuantMode.INNER_PRODUCT,  # 无偏内积模式
)
tq = TurboQuant(config)

# 生成测试数据
test_vectors = np.random.randn(100, 128).astype(np.float32)

# 量化
compressed = tq.quantize(test_vectors)

# 解量化
reconstructed = tq.dequantize(compressed)

# 检查质量
mse = np.mean((test_vectors - reconstructed) ** 2)
print(f"MSE: {mse:.6f}")
```

### 混合精度

```python
from moxing import TurboQuantMixedPrecision

# 3.5-bit 混合精度
tq_mp = TurboQuantMixedPrecision(
    dim=128,
    bits_per_channel=3.5,
    n_outliers=32,  # 32 个异常值通道
)

compressed = tq_mp.quantize(test_vectors)
reconstructed = tq_mp.dequantize(compressed)
```

### KV Cache 配置

```python
from moxing import (
    KVCacheQuantType,
    estimate_kv_cache_size_gb,
    recommend_cache_config,
)

# 估算内存
kv_size = estimate_kv_cache_size_gb(
    n_layers=32,
    n_heads=32,
    head_dim=128,
    ctx_size=65536,
    quant_type=KVCacheQuantType.TURBOQUANT_35,
)
print(f"KV Cache: {kv_size:.2f} GB")

# 获取推荐配置
config = recommend_cache_config(
    model_size_gb=9.0,
    available_vram_gb=12.0,
    desired_ctx_size=65536,
    quality_priority="balanced",  # "speed", "balanced", "quality"
)
print(f"推荐量化: {config.quant_type.value}")
```

---

## 完整示例

### 场景 1: 长上下文代码助手

```bash
# 256K 上下文，使用 q4_0
moxing ollama serve carstenuhlig/omnicoder-9b \
    --kv-cache q4_0 \
    -c 262144 \
    -p 8080
```

### 场景 2: 内存受限设备

```bash
# 8GB 显存，运行 7B 模型
moxing serve llama-3-8b.gguf \
    --kv-cache tq2.5 \
    -c 16384 \
    --cpu-offload 10
```

### 场景 3: 高质量生成

```bash
# 质量优先
moxing serve model.gguf \
    --kv-cache q8_0 \
    -c 8192
```

---

## 量化类型对照表

| 类型 | Bits | 压缩比 | KV 内存 (65K/9B) | 质量 | 适用场景 |
|------|------|--------|-----------------|------|----------|
| f16 | 16 | 1x | 32 GB | 完美 | 基线 |
| q8_0 | 8 | 2x | 16 GB | 优秀 | 质量优先 |
| q5_0 | 5 | 3.2x | 10 GB | 很好 | 平衡 |
| **q4_0** | 4 | 4x | 8 GB | 好 | **推荐** |
| tq4 | 4 | 4x | 8 GB | 高 | TurboQuant 4-bit |
| **tq3.5** | 3.5 | 4.6x | 7 GB | 质量中性 ⭐ | **质量优先** |
| tq3 | 3 | 5.3x | 6 GB | 好 | TurboQuant 3-bit |
| **tq2.5** | 2.5 | 6.4x | 5 GB | 轻微损失 ⭐ | **内存优化** |
| tq2 | 2 | 8x | 4 GB | 可接受 | 最大压缩 |

---

## 注意事项

1. **TurboQuant 类型 (tq*)** 目前使用 llama.cpp 最接近的内置量化类型
2. **q4_0** 是当前最佳选择，提供 4x 压缩和良好质量
3. **大上下文** 需要 KV Cache 量化，否则内存不足
4. **CPU Offload** 可与 KV Cache 量化配合使用进一步节省显存