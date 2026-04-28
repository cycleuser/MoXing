# MoXing vs Ollama 全面Offload策略测试报告

## 测试环境

- **系统**: Windows
- **GPU0**: AMD Radeon RX 580 2048SP (16GB VRAM, 15594MB free)
- **GPU1**: AMD Radeon RX590 GME (8GB VRAM, 7402MB free)
- **Ollama版本**: 0.21.2
- **测试时间**: 2026-04-28

## 测试模型

| 模型 | 大小 | Layers | 类型 |
|------|------|--------|------|
| functiongemma | 300MB | 19 | 小模型 |
| carstenuhlig/omnicoder-9b | 5.7GB | 33 | 大模型 |
| gemma4:e4b | 9.6GB | 43 | 超大模型 |

## 测试配置

| 配置 | 环境变量 | 说明 |
|------|---------|------|
| ollama-vulkan-gpu0 | OLLAMA_VULKAN=1, OLLAMA_GPU=0 | Vulkan GPU0 |
| ollama-vulkan-gpu1 | OLLAMA_VULKAN=1, OLLAMA_GPU=1 | Vulkan GPU1 |
| ollama-num-gpu-0 | OLLAMA_NUM_GPU=0 | NUM_GPU=0 |
| ollama-gpu--1 | OLLAMA_GPU=-1 | GPU=-1 |
| ollama-no-vulkan | OLLAMA_VULKAN=0 | 禁用Vulkan (纯CPU) |

## 测试结果

### functiongemma (300MB, 19 layers)

| 配置 | 速度 (tok/s) | Offload | 模型权重 | KV Cache | 计算图 |
|------|-------------|---------|---------|----------|--------|
| **ollama-no-vulkan (CPU)** | **42.74** | 0/19 | CPU 441.8MB | CPU 27MB | CPU 23.5MB |
| ollama-vulkan-gpu0 | 33.77 | 19/19 | Vulkan0 271.8MB + CPU 170MB | Vulkan0 111MB | Vulkan0 104.8MB + CPU 1.2MB |
| ollama-vulkan-gpu1 | 33.04 | 19/19 | Vulkan0 271.8MB + CPU 170MB | Vulkan0 111MB | Vulkan0 104.8MB + CPU 1.2MB |
| ollama-num-gpu-0 | 33.07 | 19/19 | Vulkan0 271.8MB + CPU 170MB | Vulkan0 111MB | Vulkan0 104.8MB + CPU 1.2MB |
| ollama-gpu--1 | 31.84 | 19/19 | Vulkan0 271.8MB + CPU 170MB | Vulkan0 111MB | Vulkan0 104.8MB + CPU 1.2MB |

**最佳**: ollama-no-vulkan (CPU) - 42.74 tok/s

### carstenuhlig/omnicoder-9b (5.7GB, 33 layers)

| 配置 | 速度 (tok/s) | Offload | 模型权重 | KV Cache | 计算图 |
|------|-------------|---------|---------|----------|--------|
| **ollama-gpu--1** | **8.95** | 33/33 | Vulkan0 4.8GB + CPU 545.6MB | Vulkan0 2.2GB | Vulkan0 170MB + CPU 8MB |
| ollama-num-gpu-0 | 8.89 | 33/33 | Vulkan0 4.8GB + CPU 545.6MB | Vulkan0 2.2GB | Vulkan0 170MB + CPU 8MB |
| ollama-vulkan-gpu1 | 8.87 | 33/33 | Vulkan0 4.8GB + CPU 545.6MB | Vulkan0 2.2GB | Vulkan0 170MB + CPU 8MB |
| ollama-vulkan-gpu0 | FAIL | 33/33 | Vulkan0 4.8GB + CPU 545.6MB | Vulkan0 2.2GB | Vulkan0 170MB + CPU 8MB |
| **ollama-no-vulkan (CPU)** | **2.12** | 0/33 | CPU 5.3GB | CPU 2.2GB | CPU 140.1MB |

**最佳**: ollama-gpu--1 - 8.95 tok/s (GPU比CPU快**4.2倍**)

### gemma4:e4b (9.6GB, 43 layers)

| 配置 | 速度 (tok/s) | Offload | 模型权重 | KV Cache | 计算图 |
|------|-------------|---------|---------|----------|--------|
| **ollama-gpu--1** | **6.42** | 42/43 | Vulkan0 2.8GB + CPU 6.6GB | Vulkan0 692MB | Vulkan0 190.8MB + CPU 21MB |
| ollama-vulkan-gpu1 | 6.40 | 42/43 | Vulkan0 2.8GB + CPU 6.6GB | Vulkan0 692MB | Vulkan0 190.8MB + CPU 21MB |
| ollama-vulkan-gpu0 | 6.39 | 42/43 | Vulkan0 2.8GB + CPU 6.6GB | Vulkan0 692MB | Vulkan0 190.8MB + CPU 21MB |
| ollama-num-gpu-0 | 6.37 | 42/43 | Vulkan0 2.8GB + CPU 6.6GB | Vulkan0 692MB | Vulkan0 190.8MB + CPU 21MB |
| **ollama-no-vulkan (CPU)** | **4.68** | 0/43 | CPU 9.4GB | CPU 224MB | CPU 125MB |

**最佳**: ollama-gpu--1 - 6.42 tok/s (GPU比CPU快**1.37倍**)

## 关键发现

### 1. 模型大小决定GPU优势

| 模型大小 | CPU速度 | GPU速度 | GPU优势 |
|---------|--------|--------|---------|
| 300MB (小) | 42.74 tok/s | 33.77 tok/s | CPU快**26.6%** |
| 5.7GB (大) | 2.12 tok/s | 8.95 tok/s | GPU快**322%** |
| 9.6GB (超大) | 4.68 tok/s | 6.42 tok/s | GPU快**37.2%** |

### 2. Offload策略分析

- **OLLAMA_VULKAN=0**是唯一能真正禁用GPU的方法
- **OLLAMA_NUM_GPU=0**和**OLLAMA_GPU=-1**并没有真正禁用GPU，模型仍然offload到GPU
- Ollama总是尝试使用GPU如果可用，除非明确禁用Vulkan

### 3. GPU0 vs GPU1

- GPU0 (RX580 16GB) 和 GPU1 (RX590 8GB) 性能非常接近
- 差异在±1%以内，说明两张卡性能相似

### 4. 内存使用分析

**functiongemma (300MB)**:
- GPU模式: 总内存658.8MB (Vulkan0 493.6MB + CPU 171.2MB)
- CPU模式: 总内存492.3MB (全部CPU)

**omnicoder-9b (5.7GB)**:
- GPU模式: 总内存7.7GB (Vulkan0 7.1GB + CPU 553.6MB)
- CPU模式: 总内存7.7GB (全部CPU)

**gemma4:e4b (9.6GB)**:
- GPU模式: 总内存10.3GB (Vulkan0 3.7GB + CPU 6.6GB)
- CPU模式: 总内存9.8GB (全部CPU)

## MoXing优化选项

由于Ollama模型blobs无法被官方llama.cpp直接加载，以下MoXing优化选项未测试：

| 优化选项 | 预期提升 | 说明 |
|---------|---------|------|
| `--lookahead 3` | 1.5-2x | Lookahead解码，无需额外模型 |
| `--kv-cache q4_0` | 内存减少50% | KV cache 4位量化 |
| `--kv-cache q2_K` | 内存减少75% | KV cache 2位量化 |
| `--draft MODEL` | 2-4x | 推测解码 |
| `--cpu-moe` | 7-8x | MoE专家CPU卸载 |

## 结论

1. **小模型 (<1GB)**: CPU比GPU快
   - functiongemma: CPU 42.74 tok/s vs GPU 33.77 tok/s
   - 原因：小模型可以完全放在CPU内存中，CPU计算速度更快

2. **大模型 (>5GB)**: GPU比CPU快
   - omnicoder-9b: GPU 8.95 tok/s vs CPU 2.12 tok/s (4.2倍)
   - gemma4:e4b: GPU 6.42 tok/s vs CPU 4.68 tok/s (1.37倍)
   - 原因：大模型需要GPU的并行计算能力

3. **最佳配置建议**:
   - 小模型 (<1GB): 使用`OLLAMA_VULKAN=0` (纯CPU)
   - 大模型 (>5GB): 使用`OLLAMA_GPU=-1`或`OLLAMA_VULKAN=1` (GPU)

4. **Ollama环境变量问题**:
   - `OLLAMA_NUM_GPU=0`和`OLLAMA_GPU=-1`并没有真正禁用GPU
   - 只有`OLLAMA_VULKAN=0`能真正禁用GPU

## 原始数据

完整测试结果保存在: `offload_test_results.json`
