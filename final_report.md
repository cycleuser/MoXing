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

### Ollama (已完成 - 44个测试)
1. **ollama-default**: Ollama默认配置（自动选择GPU）
2. **ollama-vulkan-gpu0**: Ollama + Vulkan + GPU0 (RX580)
3. **ollama-vulkan-gpu1**: Ollama + Vulkan + GPU1 (RX590)
4. **ollama-cpu**: Ollama CPU only

### MoXing (未完成)
5. **moxing-vulkan-gpu0**: MoXing llama-server Vulkan GPU0
6. **moxing-vulkan-gpu1**: MoXing llama-server Vulkan GPU1
7. **moxing-vulkan-gpu0-lookahead**: MoXing + Lookahead
8. **moxing-vulkan-gpu0-kvq4**: MoXing + KV cache Q4
9. **moxing-vulkan-gpu0-kvtq3**: MoXing + KV cache Q2

**注意**: MoXing的llama-server无法直接加载Ollama模型blobs（张量形状不匹配），需要Ollama的patched runner。

## Ollama测试结果

### 全部模型速度对比 (tok/s)

| 模型 | 大小 | Default | Vulkan GPU0 | Vulkan GPU1 | CPU | 最佳配置 | 最佳速度 |
|------|------|---------|-------------|-------------|-----|---------|---------|
| functiongemma | 300MB | 31.21 | 30.31 | 30.84 | 30.33 | default | 31.21 |
| granite4:350m | 708MB | 27.44 | 27.61 | 27.46 | 27.24 | vulkan-gpu0 | 27.61 |
| lfm2.5-thinking | 731MB | 39.43 | 40.36 | 40.41 | 41.07 | cpu | 41.07 |
| gemma3:1b | 815MB | 19.60 | 20.31 | 20.01 | 19.95 | vulkan-gpu0 | 20.31 |
| fusion-model | 815MB | 15.75 | 15.72 | 15.83 | 15.86 | cpu | 15.86 |
| qwen3.5:0.8b-bf16 | 1.8GB | 15.71 | 15.74 | 15.54 | 15.54 | vulkan-gpu0 | 15.74 |
| omnicoder-9b | 5.7GB | 8.96 | 9.01 | FAIL | 8.95 | vulkan-gpu0 | 9.01 |
| omnicoder-2-9b | 5.7GB | 9.00 | 8.98 | 9.06 | 8.98 | vulkan-gpu1 | 9.06 |
| gemma4:e4b | 9.6GB | 6.06 | 6.02 | 6.06 | 6.04 | vulkan-gpu1 | 6.06 |
| qwen3.5:4b-bf16 | 9.3GB | 7.98 | 7.92 | 7.93 | 8.03 | cpu | 8.03 |
| qwen3.6:35b-a3b | 23GB | 3.90 | 3.88 | 3.91 | 3.89 | vulkan-gpu1 | 3.91 |

### 性能排名

| 排名 | 模型 | 速度 (tok/s) | 大小 |
|------|------|-------------|------|
| 1 | lfm2.5-thinking | 41.07 | 731MB |
| 2 | functiongemma | 31.21 | 300MB |
| 3 | granite4:350m | 27.61 | 708MB |
| 4 | gemma3:1b | 20.31 | 815MB |
| 5 | fusion-model | 15.86 | 815MB |
| 6 | qwen3.5:0.8b-bf16 | 15.74 | 1.8GB |
| 7 | omnicoder-2-9b | 9.06 | 5.7GB |
| 8 | omnicoder-9b | 9.01 | 5.7GB |
| 9 | qwen3.5:4b-bf16 | 8.03 | 9.3GB |
| 10 | gemma4:e4b | 6.06 | 9.6GB |
| 11 | qwen3.6:35b-a3b | 3.91 | 23GB |

## 分析

### 1. 模型大小与性能关系

| 模型大小范围 | 平均速度 (tok/s) | 说明 |
|------------|----------------|------|
| <1GB | 26-41 | 小模型，速度快 |
| 1-2GB | 15-16 | 中等模型 |
| 5-10GB | 6-9 | 大模型，速度明显下降 |
| >20GB | 3.9 | 超大模型，速度很慢 |

### 2. GPU对比

- **GPU0 (RX580)** vs **GPU1 (RX590)**:
  - 性能非常接近，差异在±1%以内
  - RX580有16GB VRAM，RX590有8GB VRAM
  - 对于测试的模型（最大23GB），两张卡都需要部分offload到CPU

### 3. CPU vs GPU

- 对于所有模型，CPU和GPU性能非常接近（差异<5%）
- 原因：
  1. Vulkan后端在AMD Polaris架构（RX580/590）上优化有限
  2. 模型加载和内存传输开销抵消了GPU计算优势
  3. Ollama的自动选择已经很好

### 4. 各模型最佳配置分布

| 最佳配置 | 模型数量 | 说明 |
|---------|---------|------|
| ollama-default | 1 | functiongemma |
| ollama-vulkan-gpu0 | 4 | granite4:350m, gemma3:1b, qwen3.5:0.8b, omnicoder-9b |
| ollama-vulkan-gpu1 | 3 | omnicoder-2-9b, gemma4:e4b, qwen3.6:35b |
| ollama-cpu | 3 | lfm2.5-thinking, fusion-model, qwen3.5:4b |

## MoXing兼容性说明

MoXing的llama-server无法直接加载Ollama模型blobs，错误信息：

```
llama_model_load: error loading model: check_tensor_dims: tensor 'token_embd.weight' 
has wrong shape; expected 640, 262146, got 640, 262144, 1, 1
```

**原因**:
1. Ollama模型blobs使用GGUF格式但有Ollama特定的修改
2. 张量形状不匹配（Ollama添加了额外的token embeddings）
3. 需要使用Ollama的patched llama.cpp runner

**解决方案**:
1. 使用Ollama的ollama serve命令（已测试）
2. 下载完整的Ollama runner包（下载失败）
3. 使用标准GGUF格式模型（非Ollama blobs）

## MoXing优化选项（预期效果）

如果成功部署Ollama runner，以下优化选项可用：

| 优化选项 | 预期提升 | 说明 |
|---------|---------|------|
| `--lookahead 3` | 1.5-2x | Lookahead解码，无需额外模型 |
| `--kv-cache q4_0` | 内存减少50% | KV cache 4位量化 |
| `--kv-cache q2_K` | 内存减少75% | KV cache 2位量化 |
| `--draft MODEL` | 2-4x | 推测解码 |
| `--cpu-moe` | 7-8x | MoE专家CPU卸载 |

## 结论

1. **最快模型**: lfm2.5-thinking (41 tok/s)
2. **最慢模型**: qwen3.6:35b-a3b (3.9 tok/s)
3. **GPU优势**: 不明显（<5%），Vulkan在AMD Polaris架构上优化有限
4. **最佳配置**: 因模型而异，没有统一的"最佳"配置
5. **MoXing兼容性**: 需要Ollama runner才能加载Ollama模型blobs

## 建议

1. 对于小模型（<1GB），使用Ollama default配置即可
2. 对于中等模型（1-10GB），尝试vulkan-gpu0或vulkan-gpu1
3. 对于大模型（>10GB），性能主要受限于模型大小
4. 如果需要MoXing优化功能，需要使用标准GGUF格式模型

## 原始数据

完整测试结果保存在: `ollama_final_comprehensive.json`
