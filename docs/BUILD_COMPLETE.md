# Ollama Runner 完整构建完成

## 构建结果

所有四个后端已成功构建：

| 后端 | 大小 | 位置 | 状态 |
|------|------|------|------|
| CPU | 39M | `build/ollama-runner-cpu/` | ✅ 完成 |
| CUDA | 342M | `build/ollama-runner-cuda/` | ✅ 完成 |
| ROCm | 222M | `build/ollama-runner-rocm/` | ✅ 完成 |
| Vulkan | 165M | `build/ollama-runner-vulkan/` | ✅ 完成 |

## 文件结构

```
MoXing/
├── build_all_ollama_runners_complete.sh  # 完整构建脚本
├── build/
│   ├── ollama-runner-cpu/
│   │   ├── ollama-runner-cpu      (11MB)
│   │   ├── ollama-cli-cpu         (5.7MB)
│   │   ├── ollama-bench-cpu       (1.9MB)
│   │   ├── libggml-cpu.so         (1.3MB)
│   │   ├── libllama.so            (3.7MB)
│   │   └── run.sh                 # 启动脚本
│   ├── ollama-runner-cuda/
│   │   ├── ollama-runner-cuda     (11MB)
│   │   ├── libggml-cuda.so        (101MB)
│   │   └── ...
│   ├── ollama-runner-rocm/
│   │   ├── ollama-runner-rocm     (11MB)
│   │   ├── libggml-hip.so         (64MB)
│   │   └── ...
│   └── ollama-runner-vulkan/
│       ├── ollama-runner-vulkan    (11MB)
│       ├── libggml-vulkan.so       (42MB)
│       └── ...
└── moxing/bin/
    ├── ollama-linux-x64-cpu/
    ├── ollama-linux-x64-cuda/
    ├── ollama-linux-x64-rocm/
    └── ollama-linux-x64-vulkan/
```

## 完整构建脚本

**脚本:** `build_all_ollama_runners_complete.sh`

功能：
1. 检测可用后端（CPU/CUDA/ROCm/Vulkan）
2. 从 Ollama vendor 复制或从源码构建
3. 创建启动脚本
4. 复制到 moxing/bin
5. 测试所有 runner

用法：
```bash
# 构建所有
./build_all_ollama_runners_complete.sh

# 清理并重新构建
./build_all_ollama_runners_complete.sh --clean
```

## 使用方法

### 启动服务

```bash
# CPU
moxing ollama serve gemma4:31b -b cpu

# CUDA GPU 0
moxing ollama serve gemma4:31b -b cuda -d gpu0

# CUDA GPU 1
moxing ollama serve gemma4:31b -b cuda -d gpu1

# ROCm GPU 0
moxing ollama serve gemma4:31b -b rocm -d gpu0

# Vulkan (自动选择设备)
moxing ollama serve gemma4:31b -b vulkan
```

### 参数选项

```bash
-b, --backend {cpu,cuda,rocm,vulkan,auto}  # 后端选择
d, --device {auto,gpu0,gpu1,...}           # 设备选择
-p, --port PORT                            # 服务端口
-c, --ctx-size SIZE                        # 上下文大小 (默认: 32768)
-v, --verbose                              # 详细输出
```

### API 调用

```bash
# 服务启动后，使用 OpenAI 兼容 API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 核心创新点

1. **直接使用 Ollama 的 patched llama.cpp**
   - 支持 gemma4 等 Ollama 特定架构
   - 无需系统 Ollama 安装

2. **灵活设备选择**
   - `-d gpu0`, `-d gpu1` 直接选择设备
   - 自动设置 CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES

3. **所有后端统一支持**
   - CPU: 39M (纯 CPU)
   - CUDA: 342M (含 100MB CUDA 库)
   - ROCm: 222M (含 64MB HIP 库)
   - Vulkan: 165M (含 42MB Vulkan 库)

4. **比系统 Ollama 更灵活**
   - 可设置上下文大小
   - 可选择批大小
   - 可开关 flash attention
   - 完全控制所有参数

## 测试

运行测试脚本：
```bash
./test_gemma4.sh
```

这将测试 gemma4:31b 和 gemma4:e4b 在所有可用后端上的表现。

## 下一步

1. ✅ 测试 `moxing ollama serve gemma4:31b -b cuda -d gpu0`
2. ✅ 测试 `moxing ollama serve gemma4:31b -b rocm -d gpu1`
3. ✅ 测试 `moxing ollama serve gemma4:e4b -b vulkan`
4. 更新 README.md 文档
