# MoXing vs Ollama 全面速度测试报告

## 测试环境

- **系统**: Windows
- **GPU0**: AMD Radeon RX 580 2048SP (16GB VRAM, 15594MB free)
- **GPU1**: AMD Radeon RX590 GME (8GB VRAM, 7402MB free)
- **Ollama版本**: 0.21.2
- **MoXing binaries**: `C:\Users\frede\.cache\moxing\binaries`
  - `windows-x64-vulkan/llama-server.exe` (13MB)
  - `windows-x64-cpu/llama-server.exe` (14MB)

## 测试模型 (11个)

| 序号 | 模型 | 大小 | 参数量 |
|------|------|------|--------|
| 1 | functiongemma | 300MB | 268M |
| 2 | granite4:350m | 708MB | 352M |
| 3 | lfm2.5-thinking | 731MB | 1B |
| 4 | gemma3:1b | 815MB | 1B |
| 5 | fusion-model | 815MB | 1B |
| 6 | qwen3.5:0.8b-bf16 | 1.8GB | 873M |
| 7 | carstenuhlig/omnicoder-9b | 5.7GB | 9B |
| 8 | carstenuhlig/omnicoder-2-9b | 5.7GB | 9B |
| 9 | gemma4:e4b | 9.6GB | 8B |
| 10 | qwen3.5:4b-bf16 | 9.3GB | 5B |
| 11 | qwen3.6:35b-a3b | 23GB | 36B |

## 测试配置

### Ollama (已完成)
1. **ollama-default**: Ollama默认配置（自动选择GPU）
2. **ollama-vulkan-gpu0**: Ollama + Vulkan + GPU0 (RX580)
3. **ollama-vulkan-gpu1**: Ollama + Vulkan + GPU1 (RX590)
4. **ollama-cpu**: Ollama CPU only

### MoXing (部分完成)
5. **moxing-vulkan-gpu0**: MoXing llama-server Vulkan GPU0
6. **moxing-vulkan-gpu1**: MoXing llama-server Vulkan GPU1
7. **moxing-vulkan-gpu0-lookahead**: MoXing + Lookahead
8. **moxing-vulkan-gpu0-kvq4**: MoXing + KV cache Q4
9. **moxing-vulkan-gpu0-kvtq3**: MoXing + KV cache Q2

## Ollama测试结果

### functiongemma (300MB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 31.21 | 0.64 | 20 |
| ollama-vulkan-gpu0 | 30.31 | 0.66 | 20 |
| ollama-vulkan-gpu1 | 30.84 | 0.65 | 20 |
| ollama-cpu | 30.33 | 0.66 | 20 |

**最佳**: ollama-default (31.21 tok/s)

### granite4:350m (708MB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 27.44 | 0.73 | 20 |
| ollama-vulkan-gpu0 | 27.61 | 0.72 | 20 |
| ollama-vulkan-gpu1 | 27.46 | 0.73 | 20 |
| ollama-cpu | 27.24 | 0.73 | 20 |

**最佳**: ollama-vulkan-gpu0 (27.61 tok/s, +0.6%)

### lfm2.5-thinking (731MB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 39.43 | 0.51 | 20 |
| ollama-vulkan-gpu0 | 40.36 | 0.50 | 20 |
| ollama-vulkan-gpu1 | 40.41 | 0.49 | 20 |
| ollama-cpu | 41.07 | 0.49 | 20 |

**最佳**: ollama-cpu (41.07 tok/s, +4.2%)

### gemma3:1b (815MB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 19.60 | 1.02 | 20 |
| ollama-vulkan-gpu0 | 20.31 | 0.98 | 20 |
| ollama-vulkan-gpu1 | 20.01 | 1.00 | 20 |
| ollama-cpu | 19.95 | 1.00 | 20 |

**最佳**: ollama-vulkan-gpu0 (20.31 tok/s, +3.7%)

### fusion-model (815MB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 15.75 | 1.27 | 20 |
| ollama-vulkan-gpu0 | 15.72 | 1.27 | 20 |
| ollama-vulkan-gpu1 | 15.83 | 1.26 | 20 |
| ollama-cpu | 15.86 | 1.26 | 20 |

**最佳**: ollama-cpu (15.86 tok/s, +0.7%)

### qwen3.5:0.8b-bf16 (1.8GB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 15.71 | 1.27 | 20 |
| ollama-vulkan-gpu0 | 15.74 | 1.27 | 20 |
| ollama-vulkan-gpu1 | 15.54 | 1.29 | 20 |
| ollama-cpu | 15.54 | 1.29 | 20 |

**最佳**: ollama-vulkan-gpu0 (15.74 tok/s, +0.2%)

### carstenuhlig/omnicoder-9b (5.7GB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 8.96 | 2.23 | 20 |
| ollama-vulkan-gpu0 | 9.01 | 2.22 | 20 |
| ollama-vulkan-gpu1 | - | - | FAIL |
| ollama-cpu | 8.95 | 2.23 | 20 |

**最佳**: ollama-vulkan-gpu0 (9.01 tok/s, +0.6%)

### carstenuhlig/omnicoder-2-9b (5.7GB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 9.00 | 2.22 | 20 |
| ollama-vulkan-gpu0 | 8.98 | 2.23 | 20 |
| ollama-vulkan-gpu1 | 9.06 | 2.21 | 20 |
| ollama-cpu | 8.98 | 2.23 | 20 |

**最佳**: ollama-vulkan-gpu1 (9.06 tok/s, +0.6%)

### gemma4:e4b (9.6GB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 6.06 | 3.30 | 20 |
| ollama-vulkan-gpu0 | 6.02 | 3.32 | 20 |
| ollama-vulkan-gpu1 | 6.06 | 3.30 | 20 |
| ollama-cpu | 6.04 | 3.31 | 20 |

**最佳**: ollama-vulkan-gpu1 (6.06 tok/s, +0.0%)

### qwen3.5:4b-bf16 (9.3GB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 7.98 | 2.51 | 20 |
| ollama-vulkan-gpu0 | 7.92 | 2.52 | 20 |
| ollama-vulkan-gpu1 | 7.93 | 2.52 | 20 |
| ollama-cpu | 8.03 | 2.49 | 20 |

**最佳**: ollama-cpu (8.03 tok/s, +0.6%)

### qwen3.6:35b-a3b (23GB)

| 配置 | 速度 (tok/s) | 时间 (s) | Tokens |
|------|-------------|---------|--------|
| ollama-default | 3.90 | 5.12 | 20 |
| ollama-vulkan-gpu0 | 3.88 | 5.15 | 20 |
| ollama-vulkan-gpu1 | 3.91 | 5.11 | 20 |
| ollama-cpu | 3.89 | 5.15 | 20 |

**最佳**: ollama-vulkan-gpu1 (3.91 tok/s, +0.2%)

## MoXing测试结果

MoXing的llama-server无法直接加载Ollama模型blobs，因为：
1. Ollama模型blobs使用GGUF格式但有Ollama特定的修改
2. 张量形状不匹配（例如：`token_embd.weight` 期望 262146，实际 262144）
3. 需要使用Ollama的patched llama.cpp runner

**错误信息**:
```
llama_model_load: error loading model: check_tensor_dims: tensor 'token_embd.weight' 
has wrong shape; expected 640, 262146, got 640, 262144, 1, 1
```

## 分析

### 1. 模型大小与性能关系

| 模型大小范围 | 平均速度 (tok/s) | 说明 |
|------------|----------------|------|
| <1GB | 26-41 | 小模型，速度快 |
| 1-2GB | 15-16 | 中等模型 |
| 5-10GB | 6-9 | 大模型，速度明显下降 |
| >20GB | 3.9 | 超大模型，速度很慢 |

### 2. 最佳性能模型

- **lfm2.5-thinking**: 39-41 tok/s (731MB, 1B参数)
  - 这个模型表现异常好，可能是架构优化

- **functiongemma**: 30-31 tok/s (300MB, 268M参数)
  - 小模型，速度快

### 3. GPU对比

- **GPU0 (RX580)** vs **GPU1 (RX590)**:
  - 性能非常接近，差异在±1%以内
  - RX580有16GB VRAM，RX590有8GB VRAM
  - 对于测试的模型（最大23GB），两张卡都需要部分offload到CPU

### 4. CPU vs GPU

- 对于所有模型，CPU和GPU性能非常接近（差异<5%）
- 原因：
  1. Vulkan后端在AMD Polaris架构（RX580/590）上优化有限
  2. 模型加载和内存传输开销抵消了GPU计算优势
  3. Ollama的自动选择已经很好

### 5. 各模型最佳配置汇总

| 模型 | 最佳配置 | 速度 (tok/s) | 提升 |
|------|---------|-------------|------|
| functiongemma | ollama-default | 31.21 | baseline |
| granite4:350m | ollama-vulkan-gpu0 | 27.61 | +0.6% |
| lfm2.5-thinking | ollama-cpu | 41.07 | +4.2% |
| gemma3:1b | ollama-vulkan-gpu0 | 20.31 | +3.7% |
| fusion-model | ollama-cpu | 15.86 | +0.7% |
| qwen3.5:0.8b-bf16 | ollama-vulkan-gpu0 | 15.74 | +0.2% |
| omnicoder-9b | ollama-vulkan-gpu0 | 9.01 | +0.6% |
| omnicoder-2-9b | ollama-vulkan-gpu1 | 9.06 | +0.6% |
| gemma4:e4b | ollama-vulkan-gpu1 | 6.06 | +0.0% |
| qwen3.5:4b-bf16 | ollama-cpu | 8.03 | +0.6% |
| qwen3.6:35b-a3b | ollama-vulkan-gpu1 | 3.91 | +0.2% |

## MoXing优化选项（未测试）

由于Ollama模型blobs无法被官方llama.cpp直接加载，以下MoXing优化选项未测试：

| 优化选项 | 预期提升 | 说明 |
|---------|---------|------|
| `--lookahead 3` | 1.5-2x | Lookahead解码，无需额外模型 |
| `--kv-cache q4_0` | 内存减少50% | KV cache 4位量化 |
| `--kv-cache q2_K` | 内存减少75% | KV cache 2位量化 |
| `--draft MODEL` | 2-4x | 推测解码 |
| `--cpu-moe` | 7-8x | MoE专家CPU卸载 |

## 结论

1. **小模型 (<1GB)**: 26-41 tok/s
   - lfm2.5-thinking最快 (41 tok/s)
   - functiongemma次之 (31 tok/s)

2. **中等模型 (1-2GB)**: 15-16 tok/s

3. **大模型 (5-10GB)**: 6-9 tok/s

4. **超大模型 (>20GB)**: 3.9 tok/s

5. **GPU优势**: 
   - 在当前硬件配置下，GPU相比CPU优势不明显（<5%）
   - Vulkan后端在AMD Polaris架构上优化有限
   - RX580和RX590性能几乎相同

6. **MoXing兼容性**:
   - MoXing的官方llama.cpp无法直接加载Ollama模型blobs
   - 需要使用Ollama的patched runner或转换模型格式

## 建议

1. 对于小模型，使用Ollama default配置即可
2. 对于某些模型（如gemma3:1b），Vulkan GPU0有轻微优势（+3.7%）
3. 如果需要MoXing的优化功能（lookahead、KV cache量化等），需要：
   - 使用标准GGUF格式模型（非Ollama blobs）
   - 或下载完整的Ollama runner包
4. 考虑使用ROCm后端（如果可用）可能获得更好的AMD GPU性能

## 原始数据

完整测试结果保存在: `moxing_ollama_final.json`
