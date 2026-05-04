MoXing 和 Ollama 有什么区别？

最近在用本地大模型的时候，发现同样的模型在不同工具上跑起来速度差很多。我花了一些时间研究了一下 MoXing 和 Ollama 的区别，顺便做了个简单的性能测试。

## 先说说版本差异

这两个工具都是基于 llama.cpp 的，但用的是不同版本。

Ollama 0.18.2 用的是一个定制版的 llama.cpp，发布于 2026 年 2 月初。Ollama 团队在这个基础上打了 34 个补丁，加了一些自己的功能。

MoXing 0.1.9 用的是 llama.cpp 的官方最新版 b8468，发布于 2026 年 3 月下旬。

简单算一下，MoXing 的 llama.cpp 比 Ollama 新了大概 6 周。这 6 周里 llama.cpp 可能有一些性能优化和新功能。

## 测试环境

我用自己的笔记本做了个简单的测试。

硬件版本：
- AMD 7945HX CPU
- 64G DDR5 5600MHz RAM
- NVIDIA GeForce RTX 4060 Laptop GPU，8GB 显存

软件版本：
- Ollama 0.18.2（llama.cpp 定制版，2026 年 2 月）
- MoXing 0.1.9（llama.cpp 官方 b8468，2026 年 3 月）
- CUDA 13.2
- Ubuntu 24.04

先用最近比较出彩的omnicoder-9b的q4版本来试一下。

### 测试一：简单任务

让模型从 1 数到 10，限制最多输出 100 个 token。

| 工具 | Thinking 模式 | 输出 Token 数 | 耗时 | 推理速度 |
|------|---------------|---------------|------|----------|
| MoXing | 关闭 | 31 | 0.83s | 37.3 t/s |
| MoXing | 开启（默认） | 100 | 2.52s | 39.7 t/s |
| Ollama | 开启 | 113 | 6.68s | 16.9 t/s |

### 测试二：代码生成

让模型写一个计算阶乘的 Python 函数，限制最多输出 500 个 token。

| 工具 | Thinking 模式 | 输出 Token 数 | 耗时 | 推理速度 |
|------|---------------|---------------|------|----------|
| MoXing | 关闭 | 500 | 12.37s | 40.4 t/s |
| MoXing | 开启（默认） | 500 | 12.41s | 40.3 t/s |
| Ollama | 开启 | 717 | 42.08s | 17.0 t/s |

### 初步体验

从上面的数据可以看出，MoXing 的推理速度大约是 Ollama 的 2.3 倍，这个差距还是挺明显的。有意思的是，thinking 模式对 MoXing 的速度几乎没影响，开不开启都差不多。

不过 Ollama 的加载速度更快，大概 2.3 秒就能加载好模型，而 MoXing 需要 4.7 秒左右。

| 指标 | MoXing | Ollama |
|------|--------|--------|
| 推理速度 | 约 39 t/s | 约 17 t/s |
| 首 Token 延迟 | 约 90ms | 约 90ms |
| 模型加载时间 | 4.7s | 2.3s |

### 关于 Thinking 模式

这里需要澄清一下，我之前记得 MoXing 默认不开启 thinking，后来才发现不是这样。

llama.cpp 最新版有个参数叫 --reasoning，有三个选项：on、off 和 auto。默认是 auto，会自动检测模型是否支持 thinking。OmniCoder-9B 这个模型是支持 thinking 的，所以 MoXing 默认就会启用。

关键的区别在于处理方式：MoXing 用的 llama.cpp b8468 会把 thinking 内容单独放到一个叫 reasoning_content 的字段里，和最终的回答分开。Ollama 则是把 thinking 内容直接放在 content 里，在终端里显示为 "Thinking... ...done thinking."

所以 Ollama 输出的 token 数量更多，因为 thinking 内容也算进去了。但这不是造成速度差异的主要原因，因为 MoXing 开启 thinking 后速度也没变慢。



## 更多模型对比

如果换乘其他模型，会什么样呢？

接下来选了三个模型来测试：
- qwen3:8b（官方模型，5.2GB，Qwen3）
- huihui_ai/qwen3-abliterated:8b（社区模型，5.0GB，基于 Qwen3）
- carstenuhlig/omnicoder-9b（社区模型，5.3GB，基于 Qwen3.5）


### huihui_ai/qwen3-abliterated:8b

测试问题："is it possible to mixuse rust and python"

| 工具 | Prompt 处理速度 | 生成速度 |
|------|-----------------|---------|
| MoXing | 410.4 t/s | 48.9 t/s |
| Ollama | 660.4 t/s | 47.4 t/s |

### qwen3:8b

测试问题："how about use python instead of C to manage memory?"

| 工具 | Prompt 处理速度 | 生成速度 |
|------|-----------------|---------|
| MoXing | 220.1 t/s | 46.9 t/s |
| Ollama | 683.1 t/s | 45.9 t/s |

### carstenuhlig/omnicoder-9b

测试问题："is it possible to mixuse rust and python"

| 工具 | Prompt 处理速度 | 生成速度 |
|------|-----------------|---------|
| MoXing | 261.9 t/s | 41.1 t/s |
| Ollama | 733.6 t/s | 17.1 t/s |

## 结果分析

从测试数据来看，情况比较复杂，不能简单地说谁更快。

**生成速度（Generation Speed）**

对于 qwen3 系列模型，MoXing 稍快一点，差距在 1-3 t/s 左右，实际使用中几乎感觉不出来。

对于 omnicoder-9b，MoXing 明显更快，41.1 t/s 对比 Ollama 的 17.1 t/s。不过这个差距可能跟上下文长度设置有关，Ollama 默认用了更大的上下文（32k vs 2k），会影响速度。这是个需要注意的点：测试时要确保两边参数一致。

**Prompt 处理速度**

这方面 Ollama 明显更快，大概快 50-70%。不过 prompt 处理通常不是瓶颈，几秒钟的差距对整体体验影响不大。

## 一些发现

测试过程中发现一个有趣的事：Ollama 和 MoXing 用的 GGUF 文件格式不完全兼容。

Ollama 版的 GGUF 格式，有些模型的 GGUF 在 MoXing 上会报错，比如：
- "wrong number of tensors"
- "key not found in model"
- "tensor has wrong shape"

不过 qwen3 系列的 GGUF 两个工具都能用。如果遇到不兼容的情况，可以从 Hugging Face 或者 ModelScope 下载标准的 GGUF 文件。

## 测试脚本

我把测试代码放在 MoXing 仓库里了：

```
scripts/benchmark_moxing_vs_ollama.py
```

感兴趣的朋友可以自己跑一下，看看在你的环境下结果如何。测试脚本会：
1. 先用 MoXing 跑一遍
2. 再用 Ollama 跑一遍
3. 输出对比结果

## 怎么选？

根据测试结果，我的建议是，如果追求最大兼容性，两个都装。Ollama 用来管理模型，MoXing 用来跑推理，毕竟MoXing其实就是llama.cpp的一个Python封装，本身也不占用太多空间。

另外，速度只是选择工具的一个因素。Ollama 的模型管理、API 兼容性、社区支持都是优点。MoXing 更轻量，用的是官方 llama.cpp，版本更新。

有兴趣的话可以自己跑一下测试脚本，看看你的环境下结果如何。