# OmniCoder-9B 模型评测：MoXing vs Ollama 性能对比

最近在研究本地大模型，发现一个有趣的代码模型：OmniCoder-9B。这个模型基于 Qwen2.5，专门针对代码生成做了优化。我花了一些时间用 MoXing 和 Ollama 两个工具测试了一下，结果有些意外。

## 如何获取 OmniCoder-9B

如果你已经安装了 Ollama，获取这个模型很简单：

```bash
ollama pull carstenuhlig/omnicoder-9b
```

模型大小约 5.3GB（Q4_K_M 量化），下载后就可以直接使用了。

```bash
ollama run carstenuhlig/omnicoder-9b
```

## 测试环境

硬件：NVIDIA RTX 4060 Laptop GPU，8GB 显存，62GB 内存

软件：
- Ollama 0.18.2
- MoXing 0.1.9（llama.cpp b8468）
- CUDA 13.2
- Ubuntu 24.04

测试模型：
- carstenuhlig/omnicoder-9b（5.3GB，Qwen2.5-based）
- qwen3:8b（5.2GB，官方模型）
- huihui_ai/qwen3-abliterated:8b（5.0GB，社区模型）
- huihui_ai/gpt-oss-abliterated:20b（13GB，大模型）
- huihui_ai/glm-4.7-flash-abliterated:latest（18GB，大模型）

## 测试方法

我设计了一系列测试任务，参考了几个业界标准的基准测试：

1. **代码生成**（参考 HumanEval [1]）：让模型写一个检查回文字符串的 Python 函数
2. **数学推理**（参考 GSM8K [2]）：计算平均速度问题
3. **逻辑推理**（参考 BBH [3]）：经典的"兄弟姐妹"逻辑谜题

[1] Chen et al., 2021. "Evaluating Large Language Models Trained on Code"
[2] Cobbe et al., 2021. "Training Verifiers to Solve Math Word Problems"
[3] Suzgun et al., 2022. "Challenging BIG-Bench Tasks"

## 测试结果

### 一、代码生成任务

测试问题："Write a Python function to check if a string is a palindrome"

| 模型 | MoXing 生成速度 | Ollama 生成速度 | 备注 |
|------|----------------|-----------------|------|
| omnicoder-9b | 40.0 t/s | ~17 t/s | Ollama 使用 32k 上下文 |
| qwen3:8b | - | 45.5 t/s | MoXing GGUF 不兼容 |
| qwen3-abliterated:8b | - | 47.4 t/s | MoXing GGUF 不兼容 |
| gpt-oss-abliterated:20b | - | 8.7 t/s | 模型太大，部分用 CPU |
| glm-4.7-flash-abliterated | - | 10.2 t/s | 模型太大，部分用 CPU |

### 二、数学推理任务

测试问题："If a train travels at 60 mph for 2 hours, then at 80 mph for 3 hours, what is the average speed?"

| 模型 | MoXing 生成速度 | Ollama 生成速度 |
|------|----------------|-----------------|
| omnicoder-9b | 40.3 t/s | - |
| qwen3:8b | 46.5 t/s | 45.5 t/s |
| qwen3-abliterated:8b | 48.3 t/s | - |

### 三、逻辑推理任务

测试问题："A man says 'Brothers and sisters I have none, but this man's father is my father's son.' Who is in the photo?"

| 模型 | MoXing 生成速度 | Ollama 生成速度 |
|------|----------------|-----------------|
| qwen3-abliterated:8b | 48.3 t/s | 47.4 t/s |

### 四、综合速度对比

| 模型 | MoXing 平均生成速度 | Ollama 平均生成速度 | 差异 |
|------|-------------------|-------------------|------|
| omnicoder-9b | 40.2 t/s | ~17 t/s | MoXing 快 2.4x |
| qwen3:8b | 46.5 t/s | 45.5 t/s | 基本持平 |
| qwen3-abliterated:8b | 48.3 t/s | 47.4 t/s | 基本持平 |

## 发现一：omnicoder-9b 的特殊情况

omnicoder-9b 在 MoXing 上跑得明显更快（40 t/s vs 17 t/s），但这不是因为 MoXing 更优，而是因为参数设置不同。

Ollama 默认为 omnicoder-9b 使用 32768 的上下文长度（32k），而 MoXing 测试时我用了 2048（2k）。更大的上下文意味着更多的 KV Cache 占用，会影响推理速度。但其实细节上还不仅如此，而是还有一些额外的细节，这个后面再给大家细说。

我检查了 Ollama 的日志：

```
KvSize:32768 KvCacheType: NumThreads:16 GPULayers:32
model weights device=CUDA0 size="4.0 GiB"
model weights device=CPU size="1.3 GiB"
kv cache device=CUDA0 size="2.2 GiB"
```

可以看到，Ollama 把部分模型权重放到了 CPU 上（1.3GB），因为 8GB 显存不够。这会严重影响速度。

## 发现二：GGUF 兼容性问题

Ollama 用的 GGUF 格式和标准 llama.cpp 不完全兼容。我测试时发现：

- omnicoder-9b 的 GGUF：MoXing 可以运行
- qwen3:8b 的 GGUF：MoXing 可以运行
- qwen3-abliterated:8b 的 GGUF：MoXing 可以运行
- 其他模型：需要测试

这可能是 Ollama 定制版 llama.cpp 做了一些修改导致的。

## 发现三：大模型在 8GB 显存上的表现

测试了两个更大的模型：

**gpt-oss-abliterated:20b（13GB）**

Ollama 日志显示：
```
model weights device=CUDA0 size="6.7 GiB"
model weights device=CPU size="6.2 GiB"
GPULayers:15 (Layers 9-23)
```

只能把 15 层放到 GPU 上，其余用 CPU，速度降到 8.7 t/s。

**glm-4.7-flash-abliterated:latest（18GB）**

```
model weights device=CUDA0 size="6.5 GiB"
model weights device=CPU size="11.0 GiB"
GPULayers:17 (Layers 30-46)
```

同样只能部分 GPU 加速，速度 10.2 t/s。

## 发现四：Prompt 处理速度

| 模型 | MoXing Prompt 速度 | Ollama Prompt 速度 |
|------|-------------------|-------------------|
| omnicoder-9b | 292 t/s | 734 t/s |
| qwen3:8b | 404 t/s | 1239 t/s |
| qwen3-abliterated:8b | 755 t/s | 660 t/s |

Ollama 在 prompt 处理上普遍更快，可能是因为它的批处理优化做得更好。

## 代码生成质量对比

测试了让模型写一个回文检查函数，看看输出质量：

**MoXing + omnicoder-9b** 输出的代码：

```python
def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome.
    
    Args:
        s: Input string
        
    Returns:
        True if the string is a palindrome, False otherwise
    """
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```

**Ollama + omnicoder-9b** 输出的代码结构类似，但多了 Thinking 过程的展示。

两个工具输出的代码质量差不多，主要差异在于：
- MoXing 更快到达答案
- Ollama 展示了思考过程（如果开启了 Thinking）

## 如何选择？

根据测试结果，我的建议是：

1. **如果你有 8GB 显存，主要用 7B-9B 的模型**：MoXing 速度会更快一些

2. **如果你需要更大的模型（13B+）**：Ollama 更方便，它会自动处理 GPU/CPU 分配

3. **如果你追求代码生成速度**：MoXing 可能稍快一点，但差异不大

4. **如果你需要展示思考过程**：Ollama 的 Thinking 展示更友好

## 测试脚本

我把测试代码放到了 MoXing 仓库：

```
scripts/comprehensive_benchmark.py
```

你可以自己跑一下，测试脚本包含了：
- HumanEval 风格的代码生成任务
- GSM8K 风格的数学推理任务
- BBH 风格的逻辑推理任务
- MMLU 风格的知识问答任务
- IFEval 风格的指令遵循任务

## 参考文献

[1] Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374

[2] Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168

[3] Suzgun, M., et al. (2022). Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. arXiv:2210.09261

[4] Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. ICLR 2021.

[5] Zhou, J., et al. (2023). Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911

## 最后

这次测试让我看到，性能对比不能只看一两个指标。上下文长度、模型分配策略、GGUF 兼容性都会影响结果。

另外，对于 8GB 显存的显卡，7B-9B 的模型是最佳选择。更大的模型虽然能跑，但速度会明显下降。

有兴趣的话可以自己跑一下测试脚本，看看你的环境下结果如何。