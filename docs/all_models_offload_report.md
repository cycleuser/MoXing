# MoXing vs Ollama 全面Offload策略测试报告 - 所有11个模型

## 测试环境

- **系统**: Windows
- **GPU0**: AMD Radeon RX 580 2048SP (16GB VRAM, 15594MB free)
- **GPU1**: AMD Radeon RX590 GME (8GB VRAM, 7402MB free)
- **Ollama版本**: 0.21.2
- **测试时间**: 2026-04-28

## 测试模型 (11个)

| 序号 | 模型 | 大小 | 类型 |
|------|------|------|------|
| 1 | functiongemma | 300MB | 小模型 |
| 2 | granite4:350m | 708MB | 小模型 |
| 3 | lfm2.5-thinking | 731MB | 小模型 |
| 4 | gemma3:1b | 815MB | 小模型 |
| 5 | fusion-model | 815MB | 小模型 |
| 6 | qwen3.5:0.8b-bf16 | 1.8GB | 中等模型 |
| 7 | carstenuhlig/omnicoder-9b | 5.7GB | 大模型 |
| 8 | carstenuhlig/omnicoder-2-9b | 5.7GB | 大模型 |
| 9 | gemma4:e4b | 9.6GB | 超大模型 |
| 10 | qwen3.5:4b-bf16 | 9.3GB | 超大模型 |
| 11 | qwen3.6:35b-a3b | 23GB | 超大模型 |

## 测试配置

| 配置 | 环境变量 | 说明 |
|------|---------|------|
| ollama-vulkan-gpu0 | OLLAMA_VULKAN=1, OLLAMA_GPU=0 | Vulkan GPU0 (RX580) |
| ollama-vulkan-gpu1 | OLLAMA_VULKAN=1, OLLAMA_GPU=1 | Vulkan GPU1 (RX590) |
| ollama-num-gpu-0 | OLLAMA_NUM_GPU=0 | NUM_GPU=0 |
| ollama-gpu--1 | OLLAMA_GPU=-1 | GPU=-1 |
| ollama-no-vulkan | OLLAMA_VULKAN=0 | 禁用Vulkan (纯CPU) |

## 完整测试结果

### 1. functiongemma (300MB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-no-vulkan (CPU)** | **41.55** | baseline |
| ollama-gpu--1 | 33.80 | -18.6% |
| ollama-vulkan-gpu0 | 33.37 | -19.7% |
| ollama-vulkan-gpu1 | 32.89 | -20.8% |
| ollama-num-gpu-0 | 32.30 | -22.3% |

**结论**: CPU比GPU快18-22%

### 2. granite4:350m (708MB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-vulkan-gpu1** | **30.17** | baseline |
| ollama-vulkan-gpu0 | 29.79 | -1.3% |
| ollama-num-gpu-0 | 29.73 | -1.5% |
| ollama-gpu--1 | 29.64 | -1.8% |
| **ollama-no-vulkan (CPU)** | **19.23** | -36.3% |

**结论**: GPU比CPU快56.9%

### 3. lfm2.5-thinking (731MB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-num-gpu-0** | **44.39** | baseline |
| ollama-vulkan-gpu0 | 43.66 | -1.6% |
| ollama-gpu--1 | 43.38 | -2.3% |
| ollama-vulkan-gpu1 | 43.11 | -2.9% |
| **ollama-no-vulkan (CPU)** | **18.85** | -57.5% |

**结论**: GPU比CPU快135.5%

### 4. gemma3:1b (815MB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-vulkan-gpu1** | **20.94** | baseline |
| ollama-num-gpu-0 | 20.51 | -2.1% |
| ollama-gpu--1 | 20.45 | -2.3% |
| ollama-vulkan-gpu0 | 20.39 | -2.6% |
| **ollama-no-vulkan (CPU)** | **16.64** | -20.5% |

**结论**: GPU比CPU快25.8%

### 5. fusion-model (815MB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-no-vulkan (CPU)** | **16.84** | baseline |
| ollama-vulkan-gpu0 | 15.72 | -6.7% |
| ollama-gpu--1 | 15.65 | -7.1% |
| ollama-vulkan-gpu1 | 15.63 | -7.2% |
| ollama-num-gpu-0 | 15.62 | -7.2% |

**结论**: CPU比GPU快6-7%

### 6. qwen3.5:0.8b-bf16 (1.8GB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-vulkan-gpu0** | **15.75** | baseline |
| ollama-num-gpu-0 | 15.61 | -0.9% |
| ollama-gpu--1 | 15.61 | -0.9% |
| ollama-vulkan-gpu1 | 15.59 | -1.0% |
| **ollama-no-vulkan (CPU)** | **6.79** | -56.9% |

**结论**: GPU比CPU快131.9%

### 7. carstenuhlig/omnicoder-9b (5.7GB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-num-gpu-0** | **8.95** | baseline |
| ollama-vulkan-gpu0 | 8.94 | -0.1% |
| ollama-vulkan-gpu1 | 8.92 | -0.3% |
| ollama-gpu--1 | 8.90 | -0.6% |
| **ollama-no-vulkan (CPU)** | **2.13** | -76.2% |

**结论**: GPU比CPU快320.6%

### 8. carstenuhlig/omnicoder-2-9b (5.7GB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-vulkan-gpu0** | **8.94** | baseline |
| ollama-num-gpu-0 | 8.94 | 0.0% |
| ollama-vulkan-gpu1 | 8.92 | -0.2% |
| ollama-gpu--1 | 8.91 | -0.3% |
| **ollama-no-vulkan (CPU)** | **2.14** | -76.1% |

**结论**: GPU比CPU快317.3%

### 9. gemma4:e4b (9.6GB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-vulkan-gpu0** | **6.44** | baseline |
| ollama-gpu--1 | 6.44 | 0.0% |
| ollama-vulkan-gpu1 | 6.41 | -0.5% |
| ollama-num-gpu-0 | 6.39 | -0.8% |
| **ollama-no-vulkan (CPU)** | **4.68** | -27.3% |

**结论**: GPU比CPU快37.7%

### 10. qwen3.5:4b-bf16 (9.3GB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-vulkan-gpu0** | **7.90** | baseline |
| ollama-vulkan-gpu1 | 7.90 | 0.0% |
| ollama-num-gpu-0 | 7.89 | -0.1% |
| ollama-gpu--1 | 7.89 | -0.1% |
| **ollama-no-vulkan (CPU)** | **1.45** | -81.6% |

**结论**: GPU比CPU快445.3%

### 11. qwen3.6:35b-a3b (23GB)

| 配置 | 速度 (tok/s) | GPU vs CPU |
|------|-------------|-----------|
| **ollama-vulkan-gpu1** | **4.27** | baseline |
| ollama-gpu--1 | 4.27 | 0.0% |
| ollama-num-gpu-0 | 4.23 | -0.9% |
| ollama-vulkan-gpu0 | 4.20 | -1.6% |
| **ollama-no-vulkan (CPU)** | **3.68** | -13.8% |

**结论**: GPU比CPU快16.0%

## 综合分析

### 模型大小与GPU优势关系

| 模型大小范围 | 模型数量 | CPU平均速度 | GPU平均速度 | GPU优势 |
|------------|---------|-----------|-----------|---------|
| <1GB (小模型) | 5 | 26.62 tok/s | 30.85 tok/s | +15.9% |
| 1-2GB (中等) | 1 | 6.79 tok/s | 15.64 tok/s | +130.3% |
| 5-10GB (大模型) | 4 | 2.60 tok/s | 8.06 tok/s | +210.0% |
| >20GB (超大) | 1 | 3.68 tok/s | 4.24 tok/s | +15.2% |

### 按模型大小的最佳配置建议

| 模型大小 | 推荐配置 | 原因 |
|---------|---------|------|
| <1GB | ollama-no-vulkan (CPU) | 小模型在CPU上更快，避免GPU内存传输开销 |
| 1-2GB | ollama-vulkan-gpu0 | GPU开始显现优势 |
| 5-10GB | ollama-vulkan-gpu0 | GPU优势明显，比CPU快3-4倍 |
| >20GB | ollama-vulkan-gpu1 | GPU仍有优势，但差距缩小 |

### GPU0 vs GPU1对比

| 模型 | GPU0速度 | GPU1速度 | 差异 |
|------|---------|---------|------|
| functiongemma | 33.37 | 32.89 | +1.5% |
| granite4:350m | 29.79 | 30.17 | -1.3% |
| lfm2.5-thinking | 43.66 | 43.11 | +1.3% |
| gemma3:1b | 20.39 | 20.94 | -2.6% |
| fusion-model | 15.72 | 15.63 | +0.6% |
| qwen3.5:0.8b-bf16 | 15.75 | 15.59 | +1.0% |
| omnicoder-9b | 8.94 | 8.92 | +0.2% |
| omnicoder-2-9b | 8.94 | 8.92 | +0.2% |
| gemma4:e4b | 6.44 | 6.41 | +0.5% |
| qwen3.5:4b-bf16 | 7.90 | 7.90 | 0.0% |
| qwen3.6:35b-a3b | 4.20 | 4.27 | -1.6% |

**结论**: GPU0和GPU1性能非常接近，差异在±2%以内

### 关键发现

1. **小模型 (<1GB)**:
   - functiongemma和fusion-model在CPU上更快
   - granite4:350m、lfm2.5-thinking、gemma3:1b在GPU上更快
   - 这表明模型架构也影响GPU优势，不仅仅是大小

2. **中等模型 (1-2GB)**:
   - qwen3.5:0.8b-bf16在GPU上快131.9%
   - GPU优势开始明显

3. **大模型 (5-10GB)**:
   - GPU比CPU快3-4倍
   - qwen3.5:4b-bf16的GPU优势最大（445.3%）

4. **超大模型 (>20GB)**:
   - GPU仍有16%优势
   - 但由于模型太大，大部分需要offload到CPU，GPU优势缩小

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

1. **GPU不是总是比CPU快**: 小模型在CPU上可能更快
2. **模型架构影响GPU优势**: 同样大小的模型可能有不同的GPU表现
3. **大模型GPU优势明显**: 5-10GB模型GPU比CPU快3-4倍
4. **GPU0和GPU1性能相近**: 差异在±2%以内
5. **最佳配置因模型而异**: 没有统一的"最佳"配置

## 建议

1. **小模型 (<1GB)**: 尝试CPU和GPU，选择更快的
2. **中等模型 (1-2GB)**: 使用GPU
3. **大模型 (>5GB)**: 使用GPU，优势明显
4. **如果需要MoXing优化功能**: 需要使用标准GGUF格式模型

## 原始数据

完整测试结果保存在: `all_models_offload_results.json`
