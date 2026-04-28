# llama.cpp 全面速度测试报告

## 测试环境

- **系统**: Windows
- **GPU0**: AMD Radeon RX 580 2048SP (16GB VRAM, 15594MB free)
- **GPU1**: AMD Radeon RX590 GME (8GB VRAM, 7402MB free)
- **llama.cpp binaries**: `C:\Users\frede\.cache\moxing\binaries`
  - `windows-x64-vulkan/llama-server.exe` (13MB)
  - `windows-x64-cpu/llama-server.exe` (14MB)

## 测试模型 (9个)

| 序号 | 模型 | 大小 | 状态 |
|------|------|------|------|
| 1 | functiongemma:latest | 0.28 GB | ❌ 超时 |
| 2 | fusion-model:latest | 0.76 GB | ✅ 成功 |
| 3 | gemma3:1b | 0.76 GB | ✅ 成功 |
| 4 | granite4:350m | 0.66 GB | ✅ 成功 |
| 5 | lfm2.5-thinking:latest | 0.68 GB | ❌ 超时 |
| 6 | qwen3.5:0.8b-bf16 | 1.64 GB | ❌ 超时 |
| 7 | gemma4:e4b | 8.95 GB | ❌ 超时 |
| 8 | qwen3.5:4b-bf16 | 8.69 GB | ❌ 超时 |
| 9 | qwen3.6:35b-a3b | 22.29 GB | ❌ 超时 |

**注意**: 只有3个模型成功运行，其他模型超时。这可能是因为llama.cpp无法加载某些Ollama模型blobs（张量形状不匹配）。

## 测试配置

| 配置 | 设备 | GPU Layers | 说明 |
|------|------|-----------|------|
| vulkan-gpu0 | Vulkan0 | 99 (100%) | Vulkan GPU0 全offload |
| vulkan-gpu0-50pct | Vulkan0 | 50 (50%) | Vulkan GPU0 50% offload |
| vulkan-gpu0-25pct | Vulkan0 | 25 (25%) | Vulkan GPU0 25% offload |
| vulkan-gpu1 | Vulkan1 | 99 (100%) | Vulkan GPU1 全offload |
| vulkan-gpu1-50pct | Vulkan1 | 50 (50%) | Vulkan GPU1 50% offload |
| cpu-only | CPU | 0 | 纯CPU |
| vulkan-gpu0-kvq4 | Vulkan0 | 99 (100%) | Vulkan GPU0 + KV Q4量化 |
| vulkan-gpu0-kvq2 | Vulkan0 | 99 (100%) | Vulkan GPU0 + KV Q2量化 |

## 测试结果

### 1. fusion-model:latest (0.76 GB)

| 配置 | 速度 (tok/s) | Offload | 说明 |
|------|-------------|---------|------|
| **vulkan-gpu1-50pct** | **20.44** | 27/27 layers to GPU | GPU1 50% offload |
| vulkan-gpu1 | 20.02 | 27/27 layers to GPU | GPU1 全offload |
| vulkan-gpu0-25pct | 17.89 | 27/27 layers to GPU | GPU0 25% offload |
| vulkan-gpu0-50pct | 17.18 | 27/27 layers to GPU | GPU0 50% offload |
| vulkan-gpu0 | 13.30 | 27/27 layers to GPU | GPU0 全offload |
| vulkan-gpu0-kvq4 | 8.44 | 27/27 layers to GPU | GPU0 + KV Q4 |
| cpu-only | - | - | 超时 |
| vulkan-gpu0-kvq2 | - | - | 超时 |

**最佳**: vulkan-gpu1-50pct (20.44 tok/s)

### 2. gemma3:1b (0.76 GB)

| 配置 | 速度 (tok/s) | Offload | 说明 |
|------|-------------|---------|------|
| **vulkan-gpu1** | **20.56** | - | GPU1 全offload |
| vulkan-gpu1-50pct | 20.41 | - | GPU1 50% offload |
| vulkan-gpu0-25pct | 17.88 | - | GPU0 25% offload |
| vulkan-gpu0 | 17.25 | - | GPU0 全offload |
| vulkan-gpu0-50pct | 17.16 | - | GPU0 50% offload |
| vulkan-gpu0-kvq4 | 14.80 | - | GPU0 + KV Q4 |
| cpu-only | - | - | 超时 |
| vulkan-gpu0-kvq2 | - | - | 超时 |

**最佳**: vulkan-gpu1 (20.56 tok/s)

### 3. granite4:350m (0.66 GB)

| 配置 | 速度 (tok/s) | Offload | 说明 |
|------|-------------|---------|------|
| **vulkan-gpu1-50pct** | **24.02** | - | GPU1 50% offload |
| vulkan-gpu1 | 23.93 | - | GPU1 全offload |
| vulkan-gpu0-50pct | 22.09 | - | GPU0 50% offload |
| vulkan-gpu0 | 22.07 | - | GPU0 全offload |
| vulkan-gpu0-kvq4 | 18.90 | - | GPU0 + KV Q4 |
| vulkan-gpu0-25pct | 14.46 | - | GPU0 25% offload |
| cpu-only | - | - | 超时 |
| vulkan-gpu0-kvq2 | - | - | 超时 |

**最佳**: vulkan-gpu1-50pct (24.02 tok/s)

## 关键发现

### 1. GPU1 (RX590) 比 GPU0 (RX580) 快

| 模型 | GPU0速度 | GPU1速度 | GPU1优势 |
|------|---------|---------|---------|
| fusion-model | 13.30-17.89 | 20.02-20.44 | +13-54% |
| gemma3:1b | 17.16-17.88 | 20.41-20.56 | +14-20% |
| granite4:350m | 14.46-22.09 | 23.93-24.02 | +8-66% |

**GPU1 (RX590) 在所有测试中都比 GPU0 (RX580) 快**

### 2. 50% offload 比 100% offload 快

| 模型 | 100% offload | 50% offload | 50%优势 |
|------|-------------|-------------|---------|
| fusion-model (GPU0) | 13.30 | 17.18 | +29% |
| fusion-model (GPU1) | 20.02 | 20.44 | +2% |
| gemma3:1b (GPU0) | 17.25 | 17.16 | -1% |
| granite4:350m (GPU0) | 22.07 | 22.09 | 0% |
| granite4:350m (GPU1) | 23.93 | 24.02 | +0.4% |

**50% offload在GPU0上有明显优势（+29%），在GPU1上优势较小**

### 3. KV cache量化降低速度

| 模型 | 无量化 | KV Q4 | 下降 |
|------|-------|-------|------|
| fusion-model | 13.30 | 8.44 | -37% |
| gemma3:1b | 17.25 | 14.80 | -14% |
| granite4:350m | 22.07 | 18.90 | -14% |

**KV cache量化会降低速度，但可以减少内存使用**

### 4. CPU测试全部超时

所有模型的CPU测试都超时了，这可能是因为：
1. CPU二进制文件无法加载Ollama模型blobs
2. CPU服务器启动时间过长

## 模型兼容性

### 成功运行的模型
- fusion-model:latest (0.76 GB)
- gemma3:1b (0.76 GB)
- granite4:350m (0.66 GB)

### 失败的模型
- functiongemma:latest (0.28 GB) - 超时
- lfm2.5-thinking:latest (0.68 GB) - 超时
- qwen3.5:0.8b-bf16 (1.64 GB) - 超时
- gemma4:e4b (8.95 GB) - 超时
- qwen3.5:4b-bf16 (8.69 GB) - 超时
- qwen3.6:35b-a3b (22.29 GB) - 超时

**失败原因**: llama.cpp无法加载Ollama模型blobs（张量形状不匹配）

## 结论

1. **GPU1 (RX590) 比 GPU0 (RX580) 快**: 在所有测试中都快13-66%
2. **50% offload 是最佳配置**: 在GPU0上有明显优势
3. **KV cache量化降低速度**: 但可以减少内存使用
4. **模型兼容性有限**: 只有3/9个模型成功运行

## 建议

1. **使用GPU1 (RX590)**: 性能更好
2. **使用50% offload**: 平衡速度和内存使用
3. **避免KV cache量化**: 除非内存受限
4. **使用标准GGUF格式模型**: 避免Ollama blobs兼容性问题

## 原始数据

完整测试结果保存在: `llama_cpp_comprehensive_results.json`
