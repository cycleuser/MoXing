# MoXing Ollama 模型上下文长度说明

## 问题：为什么上下文被限制？

当使用 `moxing ollama serve gemma3:1b -c 65536` 时，实际上下文可能被限制到模型的训练上下文长度。

### 原因分析

从服务器日志可以看到：

```
llama_model_loader: - kv   7: gemma3.context_length u32 = 32768
...
srv load_model: the slot context (65536) exceeds the training context of the model (32768) - capping
slot load_model: id  0 | task -1 | new slot, n_ctx = 32768
```

**gemma3:1b 模型的训练上下文只有 32768 tokens**，llama.cpp 会自动将请求的上下文限制到模型的最大训练上下文。

这是**模型本身的限制**，不是 moxing 或 llama.cpp 的 bug。

---

## 查看模型上下文长度

使用 `moxing ollama list` 命令查看所有模型的上下文长度：

```bash
# 显示上下文长度（默认开启，但可能较慢）
moxing ollama list --no-embeddings

# 不显示上下文（快速）
moxing ollama list --no-embeddings --no-context

# 短参数形式
moxing ollama list --no-embeddings -C
```

### 示例输出

```
                      Ollama Models (8 total)                      
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━┓
┃ Name                               ┃ Size   ┃ Context┃ Type ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━┩
│ carstenuhlig/omnicoder-9b          │ 5.3 GB │ 256K   │ llm  │
│ translategemma:4b                  │ 3.1 GB │ 128K   │ llm  │
│ gemma3:1b                          │ 778 MB │ 32K    │ llm  │
│ qwen3.5:0.8B                       │ 988 MB │ -      │ llm  │
└────────────────────────────────────┴────────┴────────┴──────┘
```

---

## 解决方案

### 方案 1: 使用支持更长上下文的模型

某些模型支持更长的上下文：

| 模型 | 上下文长度 |
|------|------------|
| carstenuhlig/omnicoder-9b | 256K (262,144) |
| translategemma:4b | 128K (131,072) |
| gemma3:1b | 32K (32,768) |
| llama3.1:8b | 128K |
| qwen2.5:7b | 128K |

### 方案 2: 使用 RoPE Scaling 扩展上下文（有质量风险）

使用 `--rope-scaling` 和 `--rope-scale` 参数尝试扩展上下文：

```bash
# 线性扩展到 2 倍上下文
moxing ollama serve gemma3:1b -c 65536 --kv-cache q4_0 --rope-scaling linear --rope-scale 2

# 使用 YaRN 扩展
moxing ollama serve gemma3:1b -c 65536 --kv-cache q4_0 --rope-scaling yarn --rope-scale 2
```

**注意**: 扩展上下文可能导致质量下降，因为模型没有在扩展后的上下文长度上训练。

### 方案 3: 结合 TurboQuant 和长上下文模型

使用支持长上下文的模型 + TurboQuant：

```bash
# 使用 256K 上下文的模型 + TurboQuant
moxing ollama serve carstenuhlig/omnicoder-9b --kv-cache q4_0 -c 262144
```

---

## 命令参数说明

### moxing ollama list

```bash
moxing ollama list [OPTIONS]

选项:
  --embeddings / --no-embeddings  显示嵌入模型 [默认: 显示]
  --json, -j                       JSON 格式输出
  --select, -s                     交互式模型选择
  --context / --no-context, -C    显示上下文长度 [默认: 显示]
```

### moxing ollama serve

```bash
moxing ollama serve MODEL [OPTIONS]

选项:
  -c, --ctx-size INTEGER          上下文大小 [默认: 4096]
  --kv-cache TEXT                 KV cache 量化: auto, f16, q8_0, q4_0, tq3, tq2
  --rope-scaling TEXT             RoPE 扩展: none, linear, yarn
  --rope-scale FLOAT              RoPE 扩展因子 [默认: 1.0]
```

---

## 最佳实践

### 推荐配置

| 场景 | 推荐命令 |
|------|----------|
| **长文档处理** | `moxing ollama serve omnicoder-9b --kv-cache q4_0 -c 256000` |
| **多轮对话** | `moxing ollama serve gemma3:1b --kv-cache q4_0` |
| **内存受限** | `moxing ollama serve gemma3:1b --kv-cache q4_0 -c 4096` |
| **质量优先** | `moxing ollama serve gemma3:1b --kv-cache f16` |

### TurboQuant + 长上下文模型

这是最佳组合，可以在相同内存下支持**更长的上下文**：

```bash
# OmniCoder-9B: 256K 上下文 + TurboQuant
moxing ollama serve omnicoder-9b --kv-cache q4_0 -c 262144

# TranslateGemma-4B: 128K 上下文 + TurboQuant
moxing ollama serve translategemma:4b --kv-cache q4_0 -c 131072
```

---

## 常见问题

### Q: 为什么 `-c 65536` 实际只有 32768？

A: 模型的训练上下文长度限制了最大可用上下文。gemma3:1b 只训练了 32K 上下文，所以无法使用更大的上下文。

### Q: 如何知道模型的上下文长度？

A: 使用 `moxing ollama list` 查看 Context 列。

### Q: `-C` 参数是什么意思？

A: `-C` 是 `--no-context` 的短参数，用于跳过读取 GGUF 文件元数据，加快列表显示速度。

### Q: RoPE Scaling 可靠吗？

A: 扩展到 2 倍以内通常问题不大，但扩展更多可能导致质量下降。建议使用原生支持长上下文的模型。

---

*MoXing v0.1.24*