
# Ollama 补丁版 llama.cpp 编译完成报告

## 编译状态

| 平台 | 后端 | 状态 | 二进制大小 | 路径 |
|------|------|------|------------|------|
| macOS | Metal | ✅ 已完成 | 11 MB | moxing/bin/darwin-arm64-metal/ |
| macOS | CPU | ✅ 已完成 | ~ | moxing/bin/darwin-arm64-cpu/ |
| Linux | CUDA | 📝 需 Linux 系统 | - | moxing/bin/linux-x64-cuda/ |
| Linux | Vulkan | 📝 需 Linux 系统 | - | moxing/bin/linux-x64-vulkan/ |
| Linux | ROCm | 📝 需 Linux 系统 | - | moxing/bin/linux-x64-rocm/ |
| Windows | CUDA | 📝 需 Windows 系统 | - | moxing/bin/windows-x64-cuda/ |
| Windows | Vulkan | 📝 需 Windows 系统 | - | moxing/bin/windows-x64-vulkan/ |

## 源码信息

- **源码目录**: /Users/fred/Documents/GitHub/Others/ollama/llama/llama.cpp
- **补丁数量**: 35 个 Ollama 补丁
- **编译日期**: Fri Apr  3 10:46:50 UTC 2026
- **编译平台**: macOS ARM64 (Apple M4)

## 已完成工作

### 1. macOS 编译 ✅



### 2. 跨平台编译脚本 ✅

已创建自动编译脚本：

| 脚本 | 平台 | 用途 |
|------|------|------|
| build-all-platforms.sh | macOS | 编译 macOS 版本 + 创建其他平台脚本 |
| build-linux-cuda.sh | Linux | 编译 Linux CUDA 版本 |
| build-linux-vulkan.sh | Linux | 编译 Linux Vulkan 版本 |
| build-linux-rocm.sh | Linux | 编译 Linux ROCm 版本 |
| build-windows-cuda.ps1 | Windows | 编译 Windows CUDA 版本 |
| build-windows-vulkan.ps1 | Windows | 编译 Windows Vulkan 版本 |

### 3. 文档 ✅

- CROSS_PLATFORM_BUILD.md - 全平台编译指南
- OLLAMA_BUILD_INFO.txt - 编译信息

## 其他平台编译方法

### Linux (需要 Linux 系统)



### Windows (需要 Windows 系统)



## 使用说明

### macOS

Port 8080 in use, using port 8082
Checking GGUF compatibility...
GGUF is compatible
Model fits in VRAM: 5.3GB < 10.9GB
╭──────────────────────── Ollama: carstenuhlig/omnicod ────────────────────────╮
│ Model: carstenuhlig/omnicoder-9b                                             │
│ Size: 5.3 GB                                                                 │
│ GGUF: sha256-550e8f7253c8e07997fbce2570d37259b69b0d21faf...                  │
│ Port: 8082                                                                   │
│ Backend: metal                                                               │
│ Device: Apple M4                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ Model: carstenuhlig/omnicoder-9b                                             │
│ Backend: metal                                                               │
│ Device: Apple M4                                                             │
│ GPU Layers: all                                                              │
│ Context: 32768                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────── carstenuhlig/omnicod | Apple M4 | METAL ───────────────────╮
│ Server: http://127.0.0.1:8082                                                │
│ OpenAI API: http://127.0.0.1:8082/v1                                         │
│ Backend: metal                                                               │
│ Device: Apple M4                                                             │
│ Press Ctrl+C to stop                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
Starting llama-server...
Binary: 
/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/darwin-arm64-metal/llam
a-server
Command: 
/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/darwin-arm64-metal/llam
a-server -m 
/Users/fred/.ollama/models/blobs/sha256-550e8f7253c8e07997fbce2570d37259b69b0d21
faf77e5ed518d4ee4c73d8b3 --host 127.0.0.1 --port 8082 -c 32768 --metrics -ngl 
999 -dev MTL0 --batch-size 2048 --ubatch-size 512 --flash-attn auto
Working dir: 
/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/darwin-arm64-metal
Model: 
/Users/fred/.ollama/models/blobs/sha256-550e8f7253c8e07997fbce2570d37259b69b0d21
faf77e5ed518d4ee4c73d8b3
Backend: metal
GPU layers: all
Context: 32768

Server failed to start!
Exit code: -9
Runtime error: Server failed to start
Traceback (most recent call last):
  File "/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/cli.py", line 2006,
in ollama_serve_impl
    server.start(wait=False)
    ~~~~~~~~~~~~^^^^^^^^^^^^
  File "/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/server.py", line 
418, in start
    raise RuntimeError("Server failed to start")
RuntimeError: Server failed to start

### Linux



### Windows



## 注意事项

1. **跨平台限制**: macOS 上只能编译 macOS 版本，Linux/Windows 版本需要在对应系统上编译
2. **依赖项**: 各平台需要安装相应的依赖（CUDA、Vulkan、ROCm 等）
3. **补丁应用**: 编译前需要确保 Ollama 补丁已正确应用

## 相关文件

| 文件 | 位置 |
|------|------|
| 编译脚本 | scripts/build-all-platforms.sh |
| Linux CUDA 脚本 | moxing/bin/build-linux-cuda.sh |
| Linux Vulkan 脚本 | moxing/bin/build-linux-vulkan.sh |
| Linux ROCm 脚本 | moxing/bin/build-linux-rocm.sh |
| Windows CUDA 脚本 | moxing/bin/build-windows-cuda.ps1 |
| Windows Vulkan 脚本 | moxing/bin/build-windows-vulkan.ps1 |
| 编译文档 | CROSS_PLATFORM_BUILD.md |
| 版本信息 | moxing/bin/OLLAMA_BUILD_INFO.txt |

---

*报告生成时间：Fri Apr  3 10:46:54 UTC 2026*
*编译源：Ollama llama/llama.cpp (35 patches)*

