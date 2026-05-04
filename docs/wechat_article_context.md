# 上下文长度对推理速度的影响：MoXing vs Ollama

之前的测试发现 MoXing 和 Ollama 在 qwen3 系列模型上速度差不多，但在 omnicoder-9b 上差距很大。我深入调查了一下，发现关键在于上下文长度设置。

## 问题发现

Ollama 默认为 omnicoder-9b 使用 32768（32k）的上下文长度。查看日志：

```
KvSize:32768
GPULayers:32 (offloaded 32/33 layers to GPU)
model weights device=CUDA0 size="4.0 GiB"
model weights device=CPU size="1.3 GiB"
kv cache device=CUDA0 size="2.2 GiB"
total memory size="7.8 GiB"
```

可以看到：
1. 模型权重 5.3GB，但只放了 4GB 到 GPU
2. KV Cache 占了 2.2GB
3. 总共用了 7.8GB，几乎耗尽了 8GB 显存
4. 输出层被放到 CPU 上

这就是速度慢的原因：CPU 和 GPU 之间的数据传输成为了瓶颈。

## 测试方法

我控制变量，测试了不同上下文长度下的推理速度：

测试模型：carstenuhlig/omnicoder-9b（5.3GB）
测试问题：Write a Python function to check if a string is a palindrome

上下文长度：2048、4096、8192、16384、24576、32768

## 测试结果

### MoXing 生成速度（tokens/秒）

| 上下文 | 生成速度 | Prompt 处理速度 |
|--------|---------|----------------|
| 2048 | 39.8 | 306.2 |
| 4096 | 40.0 | 285.0 |
| 8192 | 40.3 | 309.7 |
| 16384 | 40.4 | 307.5 |
| 24576 | 40.6 | 310.1 |
| 32768 | 40.0 | 307.7 |

### Ollama 生成速度（tokens/秒）

| 上下文 | 生成速度 | Prompt 处理速度 |
|--------|---------|----------------|
| 默认(32k) | 15.6 | 615.4 |
| 2048 | 15.4 | 623.3 |
| 4096 | 15.6 | 629.6 |
| 8192 | 15.5 | 623.5 |
| 16384 | 15.6 | 622.6 |
| 24576 | 15.6 | 618.6 |

## 关键发现

**发现一：上下文长度对生成速度影响很小**

MoXing 在 2k-32k 上下文范围内，生成速度稳定在 40 t/s 左右。Ollama 也稳定在 15.5 t/s 左右。

这说明一旦模型加载完成，上下文长度对每 token 的生成时间影响很小。更大的上下文主要影响：
1. 显存占用（KV Cache）
2. 加载时间
3. 首 token 延迟

**发现二：Ollama 速度慢是因为 CPU 卸载**

Ollama 默认的 32k 上下文需要 2.2GB KV Cache，加上模型权重 5.3GB，总共需要 7.5GB。但 8GB 显存还要留一些给系统和计算图，所以 Ollama 把 1.3GB 模型权重放到了 CPU。

这就是 Ollama 只有 15.5 t/s 的原因——部分计算在 CPU 上进行。

**发现三：MoXing 能充分利用 GPU**

MoXing 在 32k 上下文时仍然保持 40 t/s，说明所有计算都在 GPU 上。这可能是因为：
1. MoXing 用 `-ngl 99` 强制所有层都在 GPU 上
2. llama.cpp 的内存管理更高效

**发现四：Ollama 的 Prompt 处理更快**

Ollama 在 Prompt 处理上快了 2 倍（620 t/s vs 300 t/s）。这可能是因为 Ollama 的批处理优化更好。

## 速度差异对比

| 指标 | MoXing | Ollama | 差异 |
|------|--------|--------|------|
| 生成速度 | 40 t/s | 15.5 t/s | MoXing 快 2.6x |
| Prompt 处理 | 300 t/s | 620 t/s | Ollama 快 2x |
| 32k 上下文显存 | 全 GPU | 部分卸载到 CPU | MoXing 更高效 |

## 为什么 qwen3:8b 没有这个问题？

之前测试 qwen3:8b 时，MoXing 和 Ollama 速度差不多。原因是：

1. qwen3:8b 只有 5.2GB，比 omnicoder-9b 的 5.3GB 略小
2. Ollama 为 qwen3:8b 默认使用 4096 上下文（4k），不是 32k
3. 4k 上下文只需要 ~300MB KV Cache，完全可以放进显存

查看 Ollama 日志：

```
qwen3:8b: KvSize:4096, GPULayers:37 (all on GPU)
omnicoder-9b: KvSize:32768, GPULayers:32 (1.3GB on CPU)
```

## MoXing 新功能：自动上下文检测

MoXing 0.1.10 新增了自动上下文检测功能：

```bash
# 自动检测最优上下文（默认）
moxing serve model.gguf

# 手动指定上下文
moxing serve model.gguf -c 16384
```

自动检测逻辑：
1. 检测可用显存
2. 计算模型大小
3. 留出 15% 显存余量
4. 计算最大可用上下文

例如，8GB 显存运行 5GB 模型：
- 剩余显存：8 * 0.85 - 5 = 1.8GB
- 可用于 KV Cache：~1.8GB
- 每 1k 上下文约需 2MB
- 最大上下文：~900k（实际限制在 16k-32k）

## 实用建议

**如果你有 8GB 显存：**

1. 7B-9B 模型：上下文可以设置到 16k-32k
2. 13B+ 模型：需要部分 CPU 卸载，速度会下降

**设置合理的上下文：**

```bash
# 代码任务：4k 足够
moxing serve model.gguf -c 4096

# 长文档处理：16k-32k
moxing serve model.gguf -c 32768

# 让 MoXing 自动检测
moxing serve model.gguf
```

**Ollama 用户：**

Ollama 默认的 32k 上下文可能导致大模型速度下降。可以通过环境变量调整：

```bash
# 设置 4k 上下文
OLLAMA_CONTEXT_LENGTH=4096 ollama run model

# 全局设置
export OLLAMA_CONTEXT_LENGTH=16384
```

## 总结

这次测试揭示了上下文长度对推理速度的影响：

1. 上下文本身不影响生成速度，但影响显存占用
2. 显存不足会导致 CPU 卸载，速度下降 2-3 倍
3. MoXing 的内存管理更高效，能充分利用 GPU
4. Ollama 默认配置对大模型可能不是最优的

选择合适的上下文长度，可以在性能和功能之间取得平衡。