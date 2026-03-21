# MoXing：让部分 Ollama 模型提速 50% 的工具

同样的硬件，同样的模型文件，MoXing 对某些模型能提速 50%。

## 前言

作为一个本地大模型爱好者，我一直使用 Ollama 来运行各种模型。有一天我想：能不能让这些模型跑得更快？

答案就是 MoXing（模型）。

## 实测性能对比

我们在 Apple M4（11.8GB 统一内存）上测试：

### 成功案例：OmniCoder-9B

`carstenuhlig/omnicoder-9b` 模型：

| 指标 | Ollama 原生 | MoXing | 提升 |
|------|------------|--------|------|
| 生成速度 | ~10 tokens/s | ~15 tokens/s | +50% |

### 不成功案例：LFM2.5

`lfm2.5-thinking` 模型：

```
$ moxing ollama serve lfm2.5-thinking

Server failed to start!
llama_model_load: error loading model: missing tensor 'token_embd_norm.weight'
```

**结论：不是所有模型都能用 MoXing 运行，但能运行的模型会更快。**

## MoXing 是什么？

MoXing 直接读取 Ollama 的模型文件（GGUF 格式），用 llama.cpp 运行。

工作流程：
1. 读取 Ollama manifest 文件
2. 找到 GGUF blob 文件
3. 用 llama.cpp 启动服务

## 使用方法

### 安装

```bash
pip install moxing
```

### 查看 Ollama 模型

```bash
moxing ollama list
```

### 运行模型

```bash
# 尝试运行
moxing ollama serve carstenuhlig/omnicoder-9b

# 或者交互式选择
moxing ollama list --select
```

## 哪些模型能工作？

### 已验证成功

- carstenuhlig/omnicoder-9b（+50% 速度）
- Qwen2.5 系列
- Llama 3.x 系列
- Mistral 系列

### 已知不成功

- lfm2.5-thinking

**建议：直接尝试你的模型，能跑就跑，不能跑就用回 Ollama。**

## 总结

MoXing 的价值：

- 对兼容的模型：更快（实测 +50%）
- 直接使用 Ollama 已下载的模型
- OpenAI API 兼容
- 不是所有模型都能工作

**MoXing 不是要取代 Ollama，而是提供一个更快的选择。**

## 链接

- GitHub: https://github.com/cycleuser/MoXing
- PyPI: https://pypi.org/project/moxing/