# MoXing (模型)

直接运行 Ollama 模型，有时更快。支持 GGUF 压缩节省硬盘空间。

## 安装

```bash
pip install moxing
```

## 使用

```bash
# 查看 Ollama 模型
moxing ollama list

# 运行模型
moxing ollama serve carstenuhlig/omnicoder-9b

# 交互式选择
moxing ollama list --select
```

## 性能对比

Apple M4 上测试 `carstenuhlig/omnicoder-9b`：

| 框架 | 速度 |
|------|------|
| Ollama | ~10 tokens/s |
| MoXing | ~15 tokens/s |

## GGUF 压缩

压缩 GGUF 文件以节省硬盘空间。虽然 GGUF 已量化，压缩率约 3-5%，但多模型时节省可观。

```bash
# 压缩 GGUF 文件
moxing compress pack model.gguf

# 运行压缩后的文件（自动解压）
moxing serve model.gguf.zst

# 查看缓存大小
moxing compress cache --size

# 清理缓存
moxing compress cache --clear
```

### 分割大文件

```bash
# 分割成 512MB 的块
moxing compress split model.gguf --size 512

# 合并回单文件
moxing compress merge model.gguf-part-aa merged.gguf
```

## 工作原理

MoXing 读取 Ollama 的 GGUF 文件，用 llama.cpp 运行。

```
Ollama manifest -> GGUF blob -> llama.cpp -> OpenAI API
```

压缩文件会自动解压到缓存：

```
model.gguf.zst -> ~/.cache/moxing/decompressed/model.gguf -> llama.cpp
```

## 兼容性

### 已验证成功

- carstenuhlig/omnicoder-9b
- Qwen2.5 系列
- Llama 3.x 系列
- Mistral 系列

### 已知不成功

- lfm2.5-thinking

**直接尝试你的模型，能跑就跑，不能跑就用 Ollama。**

## CLI 命令

### Ollama 集成

| 命令 | 说明 |
|------|------|
| `moxing ollama list` | 列出 Ollama 模型 |
| `moxing ollama serve <model>` | 运行模型 |
| `moxing ollama info <model>` | 查看模型详情 |
| `moxing serve <model.gguf>` | 运行 GGUF 文件 |

### 压缩命令

| 命令 | 说明 |
|------|------|
| `moxing compress pack <file>` | 压缩 GGUF 文件 |
| `moxing compress unpack <file>` | 解压文件 |
| `moxing compress cache --size` | 查看缓存大小 |
| `moxing compress cache --clear` | 清理缓存 |
| `moxing compress split <file>` | 分割文件 |
| `moxing compress merge <pattern> <output>` | 合并文件 |

### 其他命令

| 命令 | 说明 |
|------|------|
| `moxing bench <model>` | 性能测试 |
| `moxing check <model>` | 检查兼容性 |
| `moxing devices` | 查看可用 GPU |

## License

MIT