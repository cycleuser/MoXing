# MoXing (模型)

llama.cpp 的 Python 封装，提供 OpenAI API 兼容的 LLM 后端，支持自动 GPU 检测和模型下载。

**核心特性：**
- 🚀 **比 Ollama 更快** - 直接执行 llama.cpp，无额外开销
- 🔧 **OpenAI 兼容** - 无缝替换 OpenAI API
- 🎮 **多 GPU 支持** - CUDA、Vulkan、ROCm、Metal 后端
- 📦 **自动下载** - 支持 HuggingFace、ModelScope、Ollama 模型
- 💾 **GGUF 压缩** - 透明解压，节省硬盘空间

## 安装

```bash
pip install moxing
```

首次使用时自动下载二进制文件（约 60 MB），无需手动配置。

## 快速开始

```bash
# 运行 Ollama 模型
moxing ollama serve llama3.2

# 运行 HuggingFace 模型
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF

# 运行本地 GGUF 文件
moxing serve ./model.gguf --port 8080
```

然后使用 OpenAI API：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "你好！"}]
)
print(response.choices[0].message.content)
```

## CLI 命令

### 模型管理

| 命令 | 说明 |
|------|------|
| `moxing serve <model>` | 启动模型服务 |
| `moxing run <model> "提示词"` | 快速推理 |
| `moxing chat <model>` | 交互式聊天 |
| `moxing download <repo>` | 从 HF/ModelScope 下载模型 |
| `moxing models` | 列出可用模型 |

### Ollama 集成

| 命令 | 说明 |
|------|------|
| `moxing ollama list` | 列出 Ollama 模型 |
| `moxing ollama serve <model>` | 运行 Ollama 模型 |
| `moxing ollama info <model>` | 显示模型详情 |

### GGUF 压缩

| 命令 | 说明 |
|------|------|
| `moxing compress pack <file>` | 压缩 GGUF 文件 |
| `moxing compress unpack <file>` | 解压文件 |
| `moxing compress split <file>` | 分割文件 |
| `moxing compress merge <pattern>` | 合并文件 |

### 系统诊断

| 命令 | 说明 |
|------|------|
| `moxing devices` | 列出 GPU 设备 |
| `moxing diagnose` | 系统诊断 |
| `moxing bench <model>` | 性能测试 |
| `moxing --version` | 显示版本信息 |

## GPU 后端

MoXing 自动检测并使用最佳可用后端：

| 平台 | CPU | CUDA | Vulkan | ROCm | Metal |
|------|-----|------|--------|------|-------|
| Linux x64 | ✅ | ✅ | ✅ | ✅ | - |
| Windows x64 | ✅ | ✅ | ✅ | - | - |
| macOS ARM64 | ✅ | - | - | - | ✅ |

指定后端安装：

```bash
pip install moxing[cuda]   # NVIDIA GPU
pip install moxing[vulkan] # 跨平台 GPU
pip install moxing[metal]  # Apple Silicon
pip install moxing[rocm]   # AMD GPU
pip install moxing[cpu]    # 仅 CPU
```

## 模型来源

### Ollama 模型

```bash
moxing ollama list                  # 列出已安装模型
moxing ollama serve llama3.2        # 运行模型
moxing ollama serve --select        # 交互式选择
```

### HuggingFace

```bash
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF -q Q4_K_M
moxing serve Qwen/Qwen2.5-7B-Instruct-GGUF
```

### ModelScope（国内镜像）

```bash
moxing download Qwen/Qwen2.5-7B-Instruct-GGUF --source modelscope
```

## Python API

```python
from moxing import quick_run, quick_server, LlamaServer

# 快速推理
result = quick_run("llama3.2", "写一首关于编程的俳句")
print(result)

# 启动服务器
with quick_server("llama3.2", port=8080) as server:
    # 在 http://localhost:8080/v1 使用 OpenAI API
    pass

# 完整控制
server = LlamaServer(
    model="model.gguf",
    backend="cuda",
    ctx_size=8192,
    gpu_layers=99
)
server.start()
```

## GGUF 压缩

压缩 GGUF 文件节省硬盘空间：

```bash
# 压缩（通常节省 3-5%）
moxing compress pack model.gguf
# 生成：model.gguf.zst

# 运行压缩文件（自动解压）
moxing serve model.gguf.zst

# 分割大文件
moxing compress split model.gguf --size 512  # 512 MB 分块

# 合并
moxing compress merge "model.gguf-part-*" merged.gguf

# 管理缓存
moxing compress cache --size
moxing compress cache --clear
```

## 性能对比

Apple M4 上测试 `carstenuhlig/omnicoder-9b`：

| 框架 | 速度 |
|------|------|
| Ollama | ~10 tokens/s |
| MoXing | ~15 tokens/s |

结果因模型和硬件而异。MoXing 移除了 Ollama 的抽象层，直接执行 llama.cpp。

## 环境变量

| 变量 | 说明 |
|------|------|
| `MOXING_BINARY_SOURCE` | 二进制来源：`github`、`modelscope`、`auto` |
| `MOXING_BINARY_MIRROR` | 自定义二进制镜像 URL |
| `MOXING_NO_UPDATE_CHECK` | 跳过二进制更新检查（设为 `1`） |

## 从源码构建

### 构建 Wheel

```bash
./generate_wheel.sh --version 0.1.9
```

### 构建二进制文件

```bash
# 构建所有可用后端
./generate_binaries.sh

# 构建特定后端
./generate_binaries.sh --backend cuda

# 构建特定 llama.cpp 版本
./generate_binaries.sh --version b8468
```

### 上传

```bash
./upload_binaries.sh  # 上传到 GitHub
./upload_pypi.sh      # 上传到 PyPI
```

## 工作原理

```
用户请求 → MoXing → llama.cpp (GPU 加速) → OpenAI API 响应
               ↓
        自动下载模型（如需要）
               ↓
        自动下载二进制文件（如需要）
```

压缩文件透明解压：

```
model.gguf.zst → ~/.cache/moxing/decompressed/model.gguf → llama.cpp
```

## 兼容性

**已测试模型：**
- Qwen2.5 系列
- Llama 3.x 系列
- Mistral 系列
- Phi-3 系列
- carstenuhlig/omnicoder-9b

**直接尝试你的模型 - 如果能在 llama.cpp 运行，就能在 MoXing 运行。**

## 问题排查

```bash
# 检查系统状态
moxing diagnose

# 检查二进制版本
moxing --version

# 重新下载二进制文件
moxing download-binaries --force

# 清除所有缓存
moxing clear-cache --all
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)。

## 贡献

欢迎贡献！请在 GitHub 提交 issue 或 PR。

## 链接

- [GitHub 仓库](https://github.com/cycleuser/MoXing)
- [PyPI 包](https://pypi.org/project/moxing/)
- [问题追踪](https://github.com/cycleuser/MoXing/issues)