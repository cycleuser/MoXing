# MoXing (模型)

[![PyPI 版本](https://img.shields.io/pypi/v/moxing.svg)](https://pypi.org/project/moxing/)
[![Python 版本](https://img.shields.io/pypi/pyversions/moxing.svg)](https://pypi.org/project/moxing/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

llama.cpp 的 Python 封装，提供 OpenAI API 兼容的 LLM 后端，支持自动 GPU 检测、模型下载和高级性能优化。

**MoXing 比 Ollama 更快** - 直接运行 llama.cpp，无需抽象层开销，在相同硬件上性能提升 30-50%。

## 核心特性

- **OpenAI API 兼容** - 无缝替换 OpenAI API，支持所有 OpenAI SDK 客户端
- **自动 GPU 检测** - 自动检测并配置 CUDA、Vulkan、ROCm、Metal 后端
- **多模型来源** - 支持 HuggingFace、ModelScope（国内镜像）、Ollama 模型
- **GGUF 压缩** - zstd 压缩节省硬盘空间，透明解压
- **TurboQuant KV 缓存** - Google 的 KV 缓存压缩技术 (arXiv:2504.19874)
- **投机解码** - 使用草稿模型或前瞻解码实现 2-4 倍加速
- **多 GPU 支持** - 跨多个 GPU 的张量并行
- **Web 监控** - 实时性能仪表板，带图表
- **vLLM 引擎** - 可选 vLLM 运行器，CUDA/ROCm 下更高吞吐量
- **MLX 后端** - 通过 MLX 框架原生支持 Apple Silicon

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [OpenAI API 兼容服务](#openai-api-兼容服务)
- [与 OpenCode 及其他客户端集成](#与-opencode-及其他客户端集成)
- [CLI 命令](#cli-命令)
- [模型来源](#模型来源)
- [GPU 后端](#gpu-后端)
- [设备选择](#设备选择)
- [性能优化](#性能优化)
- [KV 缓存管理](#kv-缓存管理)
- [GGUF 压缩](#gguf-压缩)
- [TurboQuant](#turboquant)
- [监控](#监控)
- [Python API](#python-api)
- [环境变量](#环境变量)
- [问题排查](#问题排查)
- [从源码构建](#从源码构建)
- [许可证](#许可证)

## 安装

```bash
pip install moxing
```

首次使用时自动下载二进制文件（约 60 MB），无需手动配置。

### 可选依赖

```bash
# 安装所有可选依赖
pip install moxing[all]

# 安装特定功能
pip install moxing[openai]    # OpenAI SDK 客户端
pip install moxing[hf]        # HuggingFace Hub 集成
pip install moxing[modelscope] # ModelScope 集成
pip install moxing[dev]       # 开发工具 (pytest, ruff, mypy)
```

## 快速开始

```bash
# 运行 Ollama 模型（如需要自动下载）
moxing ollama serve llama3.2

# 运行 HuggingFace GGUF 模型
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF

# 运行本地 GGUF 文件
moxing serve ./model.gguf

# 列出可用 GPU 设备
moxing devices
```

启动服务器后，OpenAI API 可在 `http://localhost:8080/v1` 访问。

## OpenAI API 兼容服务

MoXing 提供完全兼容 OpenAI API 的服务器，可作为 OpenAI API 的直接替代品。这意味着你可以使用任何 OpenAI SDK 客户端、Web UI 或 IDE 集成工具与 MoXing 配合使用。

### 启动服务器

```bash
# 基本用法 - 在 8080 端口提供服务
moxing serve model.gguf

# 指定端口和主机
moxing serve model.gguf -p 8080 --host 0.0.0.0

# 使用特定 GPU 和后端
moxing serve model.gguf -d gpu0 -b cuda

# 自动寻找可用端口
moxing serve model.gguf --auto-port

# 启用详细监控
moxing serve model.gguf -v

# 启用 Web 监控仪表板
moxing serve model.gguf -w
```

### 服务器端点

运行后，MoXing 暴露以下 OpenAI 兼容端点：

| 端点 | 描述 |
|------|------|
| `GET /v1/models` | 列出可用模型 |
| `POST /v1/chat/completions` | 聊天补全（支持流式） |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/embeddings` | 嵌入（如模型支持） |

### 使用 Python

```python
from openai import OpenAI

# 连接到 MoXing 服务器
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # 任意值均可
)

# 聊天补全
response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "用简单的语言解释量子计算。"}
    ],
    temperature=0.7,
    max_tokens=512,
    stream=False
)

print(response.choices[0].message.content)

# 流式响应
stream = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "写一首关于 AI 的诗。"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 使用 curl

```bash
# 聊天补全
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "法国的首都是哪里？"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# 流式
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "给我讲个故事。"}],
    "stream": true
  }'
```

### 使用 JavaScript/Node.js

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "not-needed"
});

const response = await client.chat.completions.create({
  model: "local",
  messages: [
    { role: "user", content: "什么是机器学习？" }
  ]
});

console.log(response.choices[0].message.content);
```

## 与 OpenCode 及其他客户端集成

MoXing 可与任何 OpenAI 兼容的客户端无缝配合使用。以下是如何配置流行工具的方法：

### OpenCode

[OpenCode](https://opencode.ai) 是一个支持自定义 LLM 后端的 AI 编程助手。要在 OpenCode 中使用 MoXing：

1. 启动 MoXing 服务器：
```bash
moxing serve your-model.gguf -p 8080
```

2. 配置 OpenCode 使用本地端点：
```json
{
  "provider": "openai",
  "baseUrl": "http://localhost:8080/v1",
  "apiKey": "not-needed",
  "model": "local"
}
```

### Continue (VS Code 扩展)

[Continue](https://continue.dev) 是 VS Code 的开源自动驾驶仪：

1. 启动 MoXing 服务器
2. 编辑 `~/.continue/config.json`：
```json
{
  "models": [
    {
      "title": "MoXing 本地模型",
      "provider": "openai",
      "model": "local",
      "apiBase": "http://localhost:8080/v1"
    }
  ]
}
```

### Cursor

Cursor IDE 支持自定义 OpenAI 端点：

1. 启动 MoXing 服务器
2. 进入 Cursor 设置 > AI > 自定义 API
3. 设置：
   - API 基础 URL：`http://localhost:8080/v1`
   - API 密钥：`any-value`
   - 模型：`local`

### Roo Code / Cline

对于 Roo Code 或 Cline 扩展：

1. 启动 MoXing 服务器
2. 在扩展设置中配置：
   - API 提供商：OpenAI Compatible
   - 基础 URL：`http://localhost:8080/v1`
   - API 密钥：`not-needed`
   - 模型 ID：`local`

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="local",
    temperature=0.7
)

response = llm.invoke("解释相对论。")
print(response.content)
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="http://localhost:8080/v1",
    api_key="not-needed",
    model="local"
)

response = llm.complete("什么是 Python？")
print(response)
```

### Web UI

**Open WebUI：**
1. 启动 MoXing 服务器
2. 在 Open WebUI 管理面板中，添加自定义 OpenAI 端点：
   - URL：`http://localhost:8080/v1`
   - API 密钥：`not-needed`

**LibreChat：**
在 `.env` 中配置：
```
OPENAI_REVERSE_PROXY=http://localhost:8080/v1/chat/completions
PROXY_URL=http://localhost:8080
```

## CLI 命令

### 模型服务

| 命令 | 描述 |
|------|------|
| `moxing serve <model>` | 启动 OpenAI API 服务器 |
| `moxing run <model> -p "提示词"` | 快速单次推理 |
| `moxing chat <model>` | 交互式聊天模式 |

### 模型管理

| 命令 | 描述 |
|------|------|
| `moxing download <repo>` | 从 HuggingFace/ModelScope 下载模型 |
| `moxing models` | 列出已下载的模型 |

### Ollama 集成

| 命令 | 描述 |
|------|------|
| `moxing ollama list` | 列出已安装的 Ollama 模型 |
| `moxing ollama serve <model>` | 使用 OpenAI API 运行 Ollama 模型 |
| `moxing ollama info <model>` | 显示 Ollama 模型详情 |
| `moxing ollama serve --select` | 交互式模型选择 |

### GGUF 压缩

| 命令 | 描述 |
|------|------|
| `moxing compress pack <file>` | 使用 zstd 压缩 GGUF 文件 |
| `moxing compress unpack <file>` | 解压文件 |
| `moxing compress split <file>` | 分割为多个块 |
| `moxing compress merge <pattern>` | 合并块 |
| `moxing compress cache --size` | 检查缓存大小 |
| `moxing compress cache --clear` | 清除解压缓存 |

### TurboQuant

| 命令 | 描述 |
|------|------|
| `moxing turboquant analyze <file>` | 分析模型以使用 TurboQuant |
| `moxing turboquant calibrate <file>` | 校准 TurboQuant 参数 |

### 系统与诊断

| 命令 | 描述 |
|------|------|
| `moxing devices` | 列出 GPU 设备和后端 |
| `moxing diagnose` | 完整系统诊断 |
| `moxing bench <model>` | 性能基准测试 |
| `moxing speed <model>` | 快速速度测试 |
| `moxing info` | 显示系统信息 |
| `moxing check <file>` | 检查 GGUF 兼容性 |
| `moxing tune <model>` | 自动调优参数 |
| `moxing config` | 显示当前配置 |
| `moxing --version` | 显示版本信息 |

### 监控

| 命令 | 描述 |
|------|------|
| `moxing monitor` | 启动 Web 监控仪表板 |

### 二进制文件管理

| 命令 | 描述 |
|------|------|
| `moxing download-binaries --list` | 列出可用后端 |
| `moxing download-binaries --backend <name>` | 下载特定后端 |
| `moxing download-binaries --backend all` | 下载所有后端 |
| `moxing download-binaries --force` | 强制重新下载 |
| `moxing clear-cache --all` | 清除所有缓存 |

## 模型来源

### Ollama 模型

MoXing 可以直接运行 Ollama 模型，无需转换：

```bash
# 列出已安装的 Ollama 模型
moxing ollama list

# 运行 Ollama 模型（如需要自动下载）
moxing ollama serve llama3.2

# 使用特定设备和后端运行
moxing ollama serve llama3.2 -d gpu0 -b vulkan

# 交互式模型选择
moxing ollama serve --select

# 跳过自定义模型的兼容性检查
moxing ollama serve my-custom-model --skip-check
```

### HuggingFace

```bash
# 下载 GGUF 模型
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF

# 下载特定量化版本
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF -q Q4_K_M

# 下载并运行（一条命令）
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF
```

### ModelScope（国内镜像）

在中国大陆使用可更快下载：

```bash
# 从 ModelScope 下载
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF --source modelscope

# 设置为默认源
export MOXING_DEFAULT_SOURCE=modelscope
```

### 本地文件

```bash
# 运行本地 GGUF 文件
moxing serve ./path/to/model.gguf

# 运行压缩的 GGUF 文件（自动解压）
moxing serve ./model.gguf.zst
```

## GPU 后端

MoXing 自动检测并使用适合你硬件的最佳后端：

| 平台 | CPU | CUDA | Vulkan | ROCm | Metal | MLX |
|------|-----|------|--------|------|-------|-----|
| Linux x64 | ✅ | ✅ | ✅ | ✅ | - | - |
| Windows x64 | ✅ | ✅ | ✅ | - | - | - |
| macOS ARM64 | ✅ | - | ✅ | - | ✅ | ✅ |

### 后端选择

强制使用特定后端：

```bash
# 使用 CUDA（NVIDIA GPU）
moxing serve model.gguf -b cuda

# 使用 Vulkan（跨平台：AMD、Intel、NVIDIA）
moxing serve model.gguf -b vulkan

# 使用 ROCm（Linux 上的 AMD GPU）
moxing serve model.gguf -b rocm

# 使用 Metal（Apple GPU）
moxing serve model.gguf -b metal

# 使用 MLX（Apple Silicon，更好的兼容性）
moxing serve model.gguf -b mlx

# 仅使用 CPU
moxing serve model.gguf -b cpu

# 自动检测（默认）
moxing serve model.gguf -b auto
```

### 安装后端特定依赖

```bash
pip install moxing[cuda]   # NVIDIA GPU 支持
pip install moxing[vulkan] # 跨平台 GPU 支持
pip install moxing[metal]  # Apple Metal 支持
pip install moxing[rocm]   # AMD ROCm 支持
pip install moxing[cpu]    # 仅 CPU
```

注意：后端包很轻量 - 实际二进制文件在首次使用时自动下载。

## 设备选择

### 列出设备

```bash
moxing devices
```

示例输出：
```
可用设备：
  cpu       - CPU (Apple M4)
  gpu0      - Apple GPU (Metal)
```

### 选择设备

```bash
# 使用 GPU 0 和 Vulkan 后端
moxing serve model.gguf -d gpu0 -b vulkan

# 使用 GPU 1 和 CUDA 后端
moxing serve model.gguf -d gpu1 -b cuda

# 仅使用 CPU
moxing serve model.gguf -d cpu

# 自动选择最佳设备（默认）
moxing serve model.gguf -d auto
```

### 运行多个实例

在不同设备上同时运行多个模型：

```bash
# 自动寻找可用端口
moxing serve model1.gguf -d gpu0 --auto-port &
moxing serve model2.gguf -d gpu1 --auto-port &
moxing serve model3.gguf -d cpu --auto-port &

# 或手动指定端口
moxing serve model1.gguf -d gpu0 -p 8080 &
moxing serve model2.gguf -d gpu1 -p 8081 &
moxing serve model3.gguf -d cpu -p 8082 &
```

### 设备选项

| 选项 | 描述 |
|------|------|
| `-d gpu0`、`-d gpu1`、... | 按索引选择 GPU |
| `-d cpu` | 仅使用 CPU |
| `-d auto` | 自动选择最佳设备（默认） |

### 端口选项

| 选项 | 描述 |
|------|------|
| `-p 8080` | 使用指定端口 |
| `-p 0` 或 `--auto-port` | 自动寻找可用端口 |

### 后端选项

| 选项 | 描述 |
|------|------|
| `-b vulkan` | 跨平台 GPU（AMD、Intel、NVIDIA） |
| `-b cuda` | NVIDIA GPU |
| `-b rocm` | AMD GPU（Linux） |
| `-b metal` | Apple GPU（macOS） |
| `-b mlx` | MLX 框架（Apple Silicon） |
| `-b cpu` | 仅 CPU |
| `-b auto` | 自动检测（默认） |

### 下载多个后端二进制文件

下载所有支持的后端二进制文件以支持设备切换：

```bash
# 列出可用后端
moxing download-binaries --list

# 下载特定后端
moxing download-binaries --backend vulkan

# 下载所有后端以支持多设备
moxing download-binaries --backend all
```

## 性能优化

MoXing 提供多种优化技术以最大化推理速度并最小化内存使用。

### 投机解码

投机解码使用较小的草稿模型预测 token，然后用主模型验证。可实现 2-4 倍加速。

```bash
# 使用较小的草稿模型
moxing serve main-model.gguf --draft small-model.gguf

# 使用相同模型进行多 Token 预测（MTP）
moxing serve model.gguf --draft model.gguf

# 配置草稿参数
moxing serve model.gguf --draft small-model.gguf \
  --draft-max 5 \      # 最大草稿 token 数
  --draft-p-min 0.75   # 最小接受概率
```

### 前瞻解码

前瞻解码无需单独的草稿模型即可实现 1.5-2 倍加速：

```bash
# 启用前瞻解码（推荐 2-4 步）
moxing serve model.gguf --lookahead 3
```

### 提示词缓存

缓存重复的系统提示词以避免重新计算：

```bash
# 启用提示词缓存
moxing serve model.gguf --cache-prompts

# 设置缓存移除策略
moxing serve model.gguf --cache-prompts --cache-rem lru   # 最近最少使用
moxing serve model.gguf --cache-prompts --cache-rem fifo   # 先进先出
```

### 连续批处理

高效处理多个并发请求：

```bash
# 启用连续批处理（默认）
moxing serve model.gguf --cont-batching

# 设置并行槽位数
moxing serve model.gguf --slots 4
```

### 上下文扩展

使用 RoPE 缩放扩展上下文长度超出模型原生限制：

```bash
# 使用线性缩放 2 倍上下文
moxing serve model.gguf --rope-scaling linear --rope-scale 2

# 使用 YaRN 缩放 4 倍上下文（更好的质量）
moxing serve model.gguf --rope-scaling yarn --rope-scale 4
```

### 内存优化

```bash
# 将层卸载到 CPU RAM（当模型不适合 VRAM 时有用）
moxing serve model.gguf --cpu-offload 10

# 将 MoE 专家卸载到 CPU，保持注意力在 GPU 上（MoE 模型 7-8 倍加速）
moxing serve model.gguf --cpu-moe

# 将模型锁定在 RAM 中以防止交换
moxing serve model.gguf --mlock

# 强制 KV 缓存保留在 GPU 上（更快但使用更多 VRAM）
moxing serve model.gguf --no-kv-offload
```

### 多 GPU 支持

```bash
# 跨 GPU 分割模型（llama.cpp）
moxing serve model.gguf --tensor-split 50,50  # 2 个 GPU 均分
moxing serve model.gguf --tensor-split 70,30  # GPU0 占 70%，GPU1 占 30%

# 设置张量并行的主 GPU
moxing serve model.gguf --main-gpu 0

# 多 CPU 系统的 NUMA 策略
moxing serve model.gguf --numa distribute
```

### vLLM 引擎

在 CUDA/ROCm 系统上使用 vLLM 引擎获得更高吞吐量：

```bash
# 使用 vLLM 引擎
moxing serve model -r vllm

# 跨多个 GPU 的张量并行
moxing serve model -r vllm --tp 2

# 配置 GPU 内存利用率
moxing serve model -r vllm --gpu-mem-util 0.95

# 设置最大上下文长度
moxing serve model -r vllm --max-model-len 8192

# 启用前缀缓存
moxing serve model -r vllm --prefix-cache

# 指定注意力后端
moxing serve model -r vllm --attn-backend flash_attn
```

## KV 缓存管理

KV 缓存存储注意力键值对以加快生成速度。MoXing 提供自动 KV 缓存管理和压缩。

### 自动 KV 缓存选择

```bash
# 自动选择最佳 KV 缓存类型
moxing serve model.gguf --kv-cache auto

# 分析 KV 缓存需求
moxing serve model.gguf --analyze-cache
```

### KV 缓存量化类型

| 类型 | 描述 | 质量 | 内存 |
|------|------|------|------|
| `f16` | 全精度（16 位） | 最佳 | 最高 |
| `q8_0` | 8 位量化 | 高 | 中等 |
| `q5_0` | 5 位量化 | 良好 | 低 |
| `q4_0` | 4 位量化 | 平衡 | 最低 |

```bash
# 使用 8 位 KV 缓存
moxing serve model.gguf --kv-cache q8_0

# 使用 4 位 KV 缓存（大多数情况推荐）
moxing serve model.gguf --kv-cache q4_0
```

### 估算 KV 缓存大小

```python
from moxing import estimate_kv_cache_size_gb, recommend_cache_config

# 估算 7B 模型在 8192 上下文下的 KV 缓存大小
size_gb = estimate_kv_cache_size_gb(model_size_gb=4.5, context=8192)
print(f"估算 KV 缓存：{size_gb:.2f} GB")

# 获取推荐的缓存配置
config = recommend_cache_config(model_size_gb=4.5, available_vram_gb=12)
print(f"推荐：{config}")
```

## GGUF 压缩

使用 zstd 压缩 GGUF 文件节省硬盘空间。压缩文件在服务时透明解压。

### 压缩模型

```bash
# 压缩 GGUF 文件
moxing compress pack model.gguf
# 生成：model.gguf.zst

# 运行压缩文件（首次使用自动解压）
moxing serve model.gguf.zst
```

典型压缩率：
- Q4_K_M 模型：约 3-5% 大小减少
- 更大模型：高达 10% 大小减少

### 分割大文件

```bash
# 分割为 512 MB 块
moxing compress split model.gguf --size 512
# 生成：model.gguf-part-00, model.gguf-part-01, ...

# 将块合并回
moxing compress merge "model.gguf-part-*" merged.gguf
```

### 缓存管理

```bash
# 检查解压缓存大小
moxing compress cache --size

# 清除解压缓存
moxing compress cache --clear
```

## TurboQuant

TurboQuant 是 Google 的 KV 缓存压缩技术 (arXiv:2504.19874)，相比标准量化实现更高压缩率，质量损失极小。

### TurboQuant 模式

| 模式 | 位数 | 质量 | 使用场景 |
|------|------|------|----------|
| `tq4` | 4 位 | 高质量 | 通用 |
| `tq3.5` | 3.5 位 | 质量无损 | 推荐 |
| `tq3` | 3 位 | 良好质量 | 内存受限 |
| `tq2.5` | 2.5 位 | 轻微损失 | 最大压缩 |
| `tq2` | 2 位 | 明显损失 | 极限压缩 |

### 使用 TurboQuant

```bash
# 使用 TurboQuant 3.5 位（推荐）
moxing serve model.gguf --kv-cache tq3.5

# 使用 TurboQuant 4 位
moxing serve model.gguf --kv-cache tq4

# 使用 TurboQuant 3 位
moxing serve model.gguf --kv-cache tq3

# 分析模型以使用 TurboQuant
moxing turboquant analyze model.gguf

# 校准 TurboQuant 参数
moxing turboquant calibrate model.gguf
```

### Python API

```python
from moxing import TurboQuant, TurboQuantConfig

# 创建 TurboQuant 实例
tq = TurboQuant(TurboQuantConfig(bits_per_channel=3.5))

# 量化 KV 缓存
quantized_cache = tq.quantize(kv_cache)
```

## 监控

MoXing 提供服务器性能的实时监控。

### 终端监控

```bash
# 启用终端详细监控
moxing serve model.gguf -v
```

显示：
- Token 生成速度
- 提示词处理速度
- GPU 和 CPU 利用率
- 内存使用
- 请求队列状态

### Web 监控仪表板

```bash
# 启用 Web 监控仪表板
moxing serve model.gguf -w
```

在 `http://localhost:8080` 访问仪表板查看：
- 实时性能图表
- Token 吞吐量图
- 内存使用随时间变化
- 请求延迟指标

### 独立监控命令

```bash
# 单独启动监控仪表板
moxing monitor
```

## Python API

MoXing 提供全面的 Python API 用于编程访问。

### 快速开始

```python
from moxing import quick_run, quick_server

# 快速推理（如需要自动下载模型）
result = quick_run("llama3.2", "写一首关于编程的俳句")
print(result)

# 作为上下文管理器启动服务器
with quick_server("llama3.2", port=8080) as server:
    # 服务器运行在 http://localhost:8080/v1
    # 使用任何 OpenAI SDK 客户端
    pass
```

### 完整服务器控制

```python
from moxing import LlamaServer, ServerConfig

# 创建带配置的服务器
config = ServerConfig(
    model="model.gguf",
    backend="cuda",
    device="gpu0",
    ctx_size=8192,
    gpu_layers=99,
    kv_cache_quant="q4_0"
)

server = LlamaServer(config)
server.start()

try:
    # 服务器正在运行...
    import time
    time.sleep(60)
finally:
    server.stop()
```

### 使用客户端

```python
from moxing import Client

# 连接到服务器
client = Client("http://localhost:8080")

# 聊天补全
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "你好！"}]
)
print(response.choices[0].message.content)
```

### 设备检测

```python
from moxing import DeviceDetector, detect_best_backend

# 检测可用设备
detector = DeviceDetector()
devices = detector.detect()

for device in devices:
    print(f"{device.name}: {device.backend} ({device.memory_mb} MB)")

# 检测当前系统的最佳后端
backend = detect_best_backend()
print(f"最佳后端：{backend}")
```

### 模型下载

```python
from moxing import ModelDownloader, download_model

# 从 HuggingFace 下载
downloader = ModelDownloader()
path = downloader.download("Qwen/Qwen2.5-7B-Instruct-GGUF", quant="Q4_K_M")
print(f"下载到：{path}")

# 从 ModelScope 下载
path = downloader.download(
    "Qwen/Qwen2.5-7B-Instruct-GGUF",
    source="modelscope"
)

# 简单下载函数
path = download_model("Qwen/Qwen2.5-7B-Instruct-GGUF")
```

### GGUF 压缩

```python
from moxing import MultiCompressor, TransparentDecompressor

# 压缩 GGUF 文件
compressor = MultiCompressor()
compressed_path = compressor.compress("model.gguf")
print(f"压缩到：{compressed_path}")

# 透明解压（自动缓存）
decompressor = TransparentDecompressor()
resolved_path = decompressor.resolve("model.gguf.zst")
print(f"解析到：{resolved_path}")
```

### KV 缓存工具

```python
from moxing import (
    estimate_kv_cache_size,
    estimate_kv_cache_size_gb,
    recommend_cache_config,
    get_llama_cpp_cache_args
)

# 估算 KV 缓存大小
size_bytes = estimate_kv_cache_size(
    model_size_gb=4.5,
    context=8192,
    num_layers=32
)

size_gb = estimate_kv_cache_size_gb(model_size_gb=4.5, context=8192)

# 获取推荐配置
config = recommend_cache_config(
    model_size_gb=4.5,
    available_vram_gb=12,
    target_context=8192
)

# 获取 llama.cpp 缓存参数
args = get_llama_cpp_cache_args(quant_type="q4_0", context=8192)
```

### 二进制文件管理

```python
from moxing import (
    get_binary_manager,
    ensure_binaries,
    get_server_binary,
    check_binary_version,
    get_latest_llama_cpp_version
)

# 获取二进制管理器
manager = get_binary_manager()

# 确保二进制文件已下载
ensure_binaries()

# 获取服务器二进制路径
binary_path = get_server_binary()

# 检查已安装版本
installed = check_binary_version()
print(f"已安装：{installed}")

# 检查最新可用版本
latest = get_latest_llama_cpp_version()
print(f"最新版：{latest}")
```

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `MOXING_BINARY_SOURCE` | 二进制来源：`github`、`modelscope`、`auto` | `auto` |
| `MOXING_BINARY_MIRROR` | 自定义二进制镜像 URL | - |
| `MOXING_NO_UPDATE_CHECK` | 跳过二进制更新检查 | `0` |
| `MOXING_DEFAULT_SOURCE` | 默认模型来源：`huggingface`、`modelscope` | `huggingface` |
| `HF_TOKEN` | HuggingFace API 令牌（用于受限模型） | - |
| `MODELSCOPE_TOKEN` | ModelScope API 令牌 | - |

### 使用示例

```bash
# 使用 ModelScope 下载二进制文件（在中国大陆更快）
export MOXING_BINARY_SOURCE=modelscope

# 使用自定义镜像
export MOXING_BINARY_MIRROR=https://my-mirror.example.com/binaries

# 离线使用时跳过更新检查
export MOXING_NO_UPDATE_CHECK=1

# 设置默认模型来源为 ModelScope
export MOXING_DEFAULT_SOURCE=modelscope
```

## 问题排查

### 常见问题

**二进制文件下载失败：**
```bash
# 重试下载
moxing download-binaries --force

# 使用替代来源
export MOXING_BINARY_SOURCE=modelscope
moxing download-binaries
```

**找不到模型：**
```bash
# 检查模型是否存在
moxing models

# 先下载模型
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF
```

**未检测到 GPU：**
```bash
# 运行诊断
moxing diagnose

# 检查可用设备
moxing devices

# 强制使用 CPU 模式
moxing serve model.gguf -b cpu
```

**内存不足：**
```bash
# 使用 KV 缓存量化
moxing serve model.gguf --kv-cache q4_0

# 将层卸载到 CPU
moxing serve model.gguf --cpu-offload 10

# 使用 TurboQuant 获得更多压缩
moxing serve model.gguf --kv-cache tq3.5
```

**性能慢：**
```bash
# 基准测试你的设置
moxing bench model.gguf

# 自动调优参数
moxing tune model.gguf

# 检查是否使用 GPU
moxing serve model.gguf -v  # 详细模式显示后端
```

### 诊断命令

```bash
# 完整系统诊断
moxing diagnose

# 检查二进制版本
moxing --version

# 检查 GGUF 兼容性
moxing check model.gguf

# 显示系统信息
moxing info

# 性能基准测试
moxing bench model.gguf

# 快速速度测试
moxing speed model.gguf
```

### 清除缓存

```bash
# 清除所有缓存
moxing clear-cache --all

# 仅清除解压缓存
moxing compress cache --clear
```

## 从源码构建

### 前置条件

- Python 3.8+
- Git
- CMake（用于构建 llama.cpp）

### 构建 Wheel

```bash
# 克隆仓库
git clone https://github.com/cycleuser/MoXing.git
cd MoXing

# 以开发模式安装
pip install -e ".[dev]"

# 构建 wheel
python -m build
```

### 构建二进制文件

```bash
# 构建所有可用后端
./scripts/build_platform_wheels.py

# 为特定平台构建
#（需要相应的构建环境）
```

### 运行测试

```bash
# 运行所有测试
pytest

# 详细输出运行测试
pytest -v

# 运行特定测试文件
pytest tests/test_device.py

# 带覆盖率运行测试
pytest --cov=moxing
```

### 代码质量

```bash
# 代码检查
ruff check moxing/

# 自动修复代码问题
ruff check --fix moxing/

# 格式化代码
ruff format moxing/

# 类型检查
mypy moxing/
```

### 上传到 PyPI

```bash
# 构建分发包
python -m build

# 上传到 PyPI
twine upload dist/*

# 或使用提供的脚本
./upload_pypi.sh
```

## 工作原理

### 架构

```
用户请求 → MoXing CLI → AutoRunner → LlamaServer → llama.cpp (GPU 加速)
                ↓            ↓            ↓
          下载模型      配置设备      OpenAI API
          （如需要）     和后端        响应
```

### 透明解压

```
model.gguf.zst → ~/.cache/moxing/decompressed/model.gguf → llama.cpp
```

压缩文件自动解压并缓存。后续运行重复使用缓存。

### 设备选择流程

```
1. 检测可用 GPU
2. 检查已安装的后端
3. 将后端与 GPU 类型匹配
4. 选择最佳设备
5. 使用配置启动服务器
```

## 兼容性

### 已测试模型

- Qwen2.5 系列（0.5B 到 72B）
- Llama 3.x 系列（8B 到 405B）
- Mistral 系列（7B 到 24B）
- Phi-3 系列（3B 到 14B）
- DeepSeek 系列（多种尺寸）
- carstenuhlig/omnicoder-9b

**如果能在 llama.cpp 运行，就能在 MoXing 运行。**

### 支持的平台

| 操作系统 | 架构 | 支持的后端 |
|----------|------|-----------|
| Linux | x86_64 | CPU、CUDA、Vulkan、ROCm |
| Windows | x86_64 | CPU、CUDA、Vulkan |
| macOS | ARM64 | CPU、Metal、Vulkan、MLX |

### Python 版本

- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13

## 性能

在 Apple M4 上测试 `carstenuhlig/omnicoder-9b`（Q4_K_M）：

| 框架 | 速度 | 备注 |
|------|------|------|
| Ollama | ~10 tok/s | 有抽象层开销 |
| MoXing | ~15 tok/s | 直接 llama.cpp 执行 |

结果因模型和硬件而异。MoXing 移除了 Ollama 的抽象层，直接执行 llama.cpp，实现 30-50% 的性能提升。

## 贡献

欢迎贡献！以下是入门方法：

1. Fork 仓库
2. 创建特性分支：`git checkout -b feature/my-feature`
3. 进行更改
4. 运行测试：`pytest`
5. 运行代码检查：`ruff check moxing/ && ruff format moxing/`
6. 提交更改：`git commit -m "添加我的特性"`
7. 推送到分支：`git push origin feature/my-feature`
8. 打开 Pull Request

### 开发设置

```bash
# 克隆并安装
git clone https://github.com/cycleuser/MoXing.git
cd MoXing
pip install -e ".[dev,all]"

# 运行测试
pytest

# 检查代码质量
ruff check moxing/
mypy moxing/
```

## 许可证

GPL-3.0 许可证 - 详见 [LICENSE](LICENSE)。

## 链接

- [GitHub 仓库](https://github.com/cycleuser/MoXing)
- [PyPI 包](https://pypi.org/project/moxing/)
- [问题追踪](https://github.com/cycleuser/MoXing/issues)
- [English Documentation](README.md)

## 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 底层 LLM 推理引擎
- [OpenAI](https://openai.com) - API 规范
- [HuggingFace](https://huggingface.co) - 模型托管
- [ModelScope](https://modelscope.cn) - 国内镜像支持
- [Google](https://arxiv.org/abs/2504.19874) - TurboQuant 研究
