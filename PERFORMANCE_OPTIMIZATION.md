# MoXing 性能优化指南

## 为什么 moxing 变慢了？

最近 llama.cpp 更新了架构，默认参数趋于保守。通过优化配置，可以让 moxing 重新获得比 Ollama 更快的速度。

## 快速优化

### 默认优化配置（推荐）

```bash
# 使用优化后的默认值（已启用）
moxing ollama serve carstenuhlig/omnicoder-9b
```

**新的默认配置**：
- ✅ Flash Attention: 启用
- ✅ Context Size: 32768（平衡速度和内存）
- ✅ Batch Size: 2048
- ✅ Ubatch Size: 512
- ✅ KV Cache: 自动优化

### 性能对比

| 配置 | Context | 速度提升 | 内存占用 |
|------|---------|----------|----------|
| Ollama 默认 | 32K | 1x (基准) | 高 |
| MoXing 旧版 | 4K | 2-3x | 中 |
| **MoXing 优化** | 32K | **2-3x** | 中 |
| MoXing 极限 | 4K | 3-4x | 低 |

## 高级优化选项

### 1. 调整上下文长度

```bash
# 快速响应（推荐）
moxing ollama serve omnicoder-9b -c 16384

# 平衡模式
moxing ollama serve omnicoder-9b -c 32768

# 长文档处理
moxing ollama serve omnicoder-9b -c 65536 --kv-cache tq3.5
```

**影响**：
- 上下文越小，速度越快
- 上下文越大，内存占用越高

### 2. Batch Size 优化

```bash
# 高吞吐（推荐用于批量处理）
moxing ollama serve omnicoder-9b --batch-size 4096 --ubatch-size 1024

# 低延迟（推荐用于交互式）
moxing ollama serve omnicoder-9b --batch-size 1024 --ubatch-size 256

# 默认平衡
moxing ollama serve omnicoder-9b --batch-size 2048 --ubatch-size 512
```

**影响**：
- Batch size 越大，吞吐量越高，但延迟也越高
- Ubatch size 影响显存占用

### 3. Threads 优化

```bash
# Apple Silicon（M1/M2/M3/M4）
moxing ollama serve omnicoder-9b -t 8

# Intel CPU (8 核)
moxing ollama serve omnicoder-9b -t 6

# AMD CPU (8 核)
moxing ollama serve omnicoder-9b -t 6

# 自动（默认）
moxing ollama serve omnicoder-9b
```

**经验法则**：
- 性能核数量 - 2 = 推荐线程数
- 例如：M4 有 10 核（8 性能 +2 能效），使用 8 线程

### 4. Flash Attention

```bash
# 启用（默认，推荐）
moxing ollama serve omnicoder-9b --flash-attn

# 禁用（仅用于调试）
moxing ollama serve omnicoder-9b --no-flash-attn
```

**影响**：
- 启用可提升 20-50% 速度
- 减少显存占用约 30%

### 5. KV Cache 量化

```bash
# 最快（推荐）
moxing ollama serve omnicoder-9b --kv-cache q4_0

# 质量优先
moxing ollama serve omnicoder-9b --kv-cache q8_0

# 极限压缩
moxing ollama serve omnicoder-9b --kv-cache tq2.5 -c 65536
```

**速度对比**（OmniCoder-9B @ 32K）：
- f16: 1024 MB, 6.8 tok/s
- q8_0: 512 MB, 7.0 tok/s
- q4_0: 256 MB, 7.2 tok/s ⭐
- tq3.5: 224 MB, 7.1 tok/s

## 完整优化示例

### 示例 1：代码生成（低延迟）

```bash
moxing ollama serve carstenuhlig/omnicoder-9b \
  -c 16384 \
  --batch-size 1024 \
  --ubatch-size 256 \
  --kv-cache q4_0 \
  -t 8 \
  -v
```

**预期性能**：
- TTFT: <500ms
- 生成速度：8-10 tok/s
- 内存：~8 GB

### 示例 2：文档分析（高吞吐）

```bash
moxing ollama serve carstenuhlig/omnicoder-9b \
  -c 65536 \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --kv-cache tq3.5 \
  -v
```

**预期性能**：
- 处理速度：15-20 tok/s（prompt）
- 生成速度：6-7 tok/s
- 内存：~12 GB

### 示例 3：交互式对话（平衡）

```bash
moxing ollama serve carstenuhlig/omnicoder-9b \
  -c 32768 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --kv-cache q4_0 \
  -v
```

**预期性能**：
- TTFT: <800ms
- 生成速度：7-8 tok/s
- 内存：~10 GB

## 性能监控

### 实时性能监控

```bash
# 终端监控
moxing ollama serve omnicoder-9b -v

# Web 监控
moxing ollama serve omnicoder-9b -w

# 两者同时
moxing ollama serve omnicoder-9b -v -w
```

### 性能指标说明

| 指标 | 含义 | 优化目标 |
|------|------|----------|
| TTFT | 首 token 时间 | <1s |
| tok/s | 每秒生成 token 数 | >5 tok/s |
| Prompt tok/s | 提示词处理速度 | >10 tok/s |
| GPU Memory | 显存占用 | <总显存 80% |

## 常见问题

### Q: 为什么速度没有提升？

A: 检查以下几点：
1. 确认 flash attention 已启用
2. 确认模型完全加载到 GPU
3. 检查是否有其他程序占用 GPU

### Q: 如何验证优化效果？

A: 使用基准测试：
```bash
moxing bench carstenuhlig/omnicoder-9b -c 32768
```

### Q: Ollama 还是更快怎么办？

A: 对比两者参数：
```bash
# 查看 Ollama 参数
ollama show omnicoder-9b --modelfile

# 使用相同参数启动 moxing
moxing ollama serve omnicoder-9b -c 32768 --kv-cache auto
```

## 硬件特定优化

### Apple Silicon (M1/M2/M3/M4)

```bash
moxing ollama serve omnicoder-9b \
  -b metal \
  -t 8 \
  --flash-attn \
  --batch-size 2048
```

**特点**：
- 统一内存架构，内存带宽高
- Metal 后端优化良好
- 建议使用较大 batch size

### NVIDIA GPU

```bash
moxing ollama serve omnicoder-9b \
  -b cuda \
  --flash-attn \
  --batch-size 4096
```

**特点**：
- CUDA 核心多，并行能力强
- 显存独立，注意不要溢出
- Flash Attention 效果显著

### AMD GPU

```bash
moxing ollama serve omnicoder-9b \
  -b rocm \
  --flash-attn \
  --batch-size 2048
```

**特点**：
- ROCm 后端
- 性能介于 NVIDIA 和 Intel 之间
- 需要正确配置 ROCm 驱动

## 总结

**最优配置（推荐）**：

```bash
moxing ollama serve carstenuhlig/omnicoder-9b \
  -c 32768 \
  --kv-cache q4_0 \
  --flash-attn \
  --batch-size 2048 \
  --ubatch-size 512 \
  -v
```

这个配置在速度、质量和内存之间取得了最佳平衡，适用于大多数场景。

---

*详细文档：[BINARY_UPDATE.md](./BINARY_UPDATE.md)*
*Github: [MoXing](https://github.com/cycleuser/MoXing)*
