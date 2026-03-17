# moxing (模型)

llama.cpp 的 Python 封装 —— OpenAI API 兼容的 LLM 后端，支持自动 GPU 检测和模型下载。

**moxing**（模型）在中文中意为 "model"。为本地运行 LLM 提供简单统一的接口。

## 功能特性

- **自动 GPU 检测**：自动检测并配置最佳 GPU 后端（Vulkan、CUDA、ROCm、Metal）
- **模型下载**：从 HuggingFace 和 ModelScope 下载 GGUF 模型
- **OpenAI API 兼容**：可直接替换 OpenAI API 使用
- **函数调用**：支持工具和函数调用
- **预编译二进制**：自动下载预编译的 llama.cpp 二进制文件
- **性能测试**：类似 ollama 的 tokens/秒 性能测量

## 安装

```bash
pip install moxing
```

## 快速开始

### 第一步：下载模型

使用 ModelScope CLI（推荐国内用户）：

```bash
# 安装 modelscope
pip install modelscope

# 下载 OmniCoder-9B GGUF 模型
modelscope download --model Tesslate/OmniCoder-9B-GGUF \
    omnicoder-9b-q4_k_m.gguf \
    --local_dir ./models
```

或使用 moxing 内置下载器：

```bash
moxing download Tesslate/OmniCoder-9B-GGUF -q Q4_K_M
```

### 第二步：列出 GPU 设备

```bash
moxing devices
```

输出：
```
Available Devices
+----------------------------------------------------------------+
| #   | Name                 | Backend | Memory | Free  | Vendor |
|-----+----------------------+---------+--------+-------+--------|
| 0   | AMD Radeon RX590 GME | vulkan  | 8.0GB  | 7.2GB | amd    |
+----------------------------------------------------------------+
```

### 第三步：运行推理

```bash
# 快速速度测试
moxing speed ./models/omnicoder-9b-q4_k_m.gguf

# 性能基准测试
moxing bench ./models/omnicoder-9b-q4_k_m.gguf

# 交互式聊天
moxing chat ./models/omnicoder-9b-q4_k_m.gguf
```

### 第四步：启动 OpenAI 兼容服务器

```bash
moxing serve ./models/omnicoder-9b-q4_k_m.gguf -p 8080
```

现在可以使用 OpenAI API：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "写一个 Python 排序函数"}]
)
print(response.choices[0].message.content)
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `moxing serve` | 启动 OpenAI 兼容服务器 |
| `moxing run` | 使用模型运行推理 |
| `moxing chat` | 与模型交互聊天 |
| `moxing bench` | 模型性能基准测试 |
| `moxing speed` | 快速速度测试 |
| `moxing info` | 显示模型信息和预估 |
| `moxing download` | 下载模型 |
| `moxing models` | 列出可用模型 |
| `moxing devices` | 列出 GPU 设备 |
| `moxing diagnose` | 诊断系统配置 |

## Python API

### 快速推理

```python
from moxing import quick_run, quick_server, Client

# 快速推理
result = quick_run("./models/omnicoder-9b-q4_k_m.gguf", "写一首关于编程的俳句")
print(result)
```

### 自动配置服务器

```python
from moxing import quick_server, Client

with quick_server("./models/omnicoder-9b-q4_k_m.gguf") as server:
    client = Client(server.base_url)
    
    response = client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "你好！"}]
    )
    print(response.choices[0]["message"]["content"])
```

### 自动 GPU 检测

```python
from moxing import DeviceDetector

# 检测可用 GPU
detector = DeviceDetector()
devices = detector.detect()
for device in devices:
    print(f"{device.name} ({device.backend.value}, {device.memory_gb:.1f}GB)")

# 获取最佳配置
config = detector.get_best_device(model_size_gb=5.0)
print(f"最佳设备: {config.device.name}")
print(f"推荐 GPU 层数: {config.n_gpu_layers}")
```

### 模型下载

```python
from moxing import ModelDownloader

downloader = ModelDownloader()

# 从 HuggingFace 下载
path = downloader.download("Qwen/Qwen2.5-7B-Instruct-GGUF", "Q4_K_M.gguf")

# 从 ModelScope 下载
path = downloader.download(
    "Tesslate/OmniCoder-9B-GGUF",
    "omnicoder-9b-q4_k_m.gguf",
    source="modelscope"
)
```

## 热门模型

| 名称 | 说明 | 可用规格 |
|------|------|----------|
| OmniCoder-9B | 代码生成模型 | Q4_K_M, Q5_K_M, Q8_0 |
| llama-3.2-3b | Llama 3.2 3B | Q4_K_M, Q5_K_M, Q8_0 |
| qwen2.5-7b | Qwen 2.5 7B | Q4_K_M, Q5_K_M, Q8_0 |
| deepseek-coder-6.7b | DeepSeek Coder | Q4_K_M, Q5_K_M, Q8_0 |

## GPU 后端

| 后端 | 平台 | 说明 |
|------|------|------|
| Vulkan | Windows, Linux | 跨平台 GPU API，支持 AMD、Intel、NVIDIA |
| CUDA | Windows, Linux | NVIDIA GPU |
| ROCm | Linux | AMD GPU |
| Metal | macOS | Apple Silicon |
| CPU | 所有平台 | 回退方案，无需 GPU |

## 函数调用

```python
from moxing import Client, LlamaServer

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取某地的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

with LlamaServer("./models/omnicoder-9b-q4_k_m.gguf") as server:
    client = Client(server.base_url)
    
    response = client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "东京天气怎么样？"}],
        tools=tools
    )
    
    if response.choices[0]["message"].get("tool_calls"):
        print("模型想调用:", response.choices[0]["message"]["tool_calls"])
```

## 性能示例

在 AMD Radeon RX590 GME（8GB 显存）上使用 Vulkan 后端：

| 模型 | 大小 | 速度 |
|------|------|------|
| TinyLLama Q4_K_M | 0.62 GB | ~90 t/s |
| OmniCoder-9B Q4_K_M | 5.34 GB | ~18 t/s |

## 系统要求

- Python 3.8+
- Vulkan SDK（Vulkan 后端）
- CUDA Toolkit（CUDA 后端）
- ROCm（ROCm 后端）

## 许可证

MIT License —— 与 llama.cpp 相同

## 链接

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [OmniCoder-9B](https://modelscope.cn/models/Tesslate/OmniCoder-9B-GGUF)
- [问题反馈](https://github.com/cycleuser/MoXing/issues)