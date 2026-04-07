# Ollama 补丁版 llama.cpp 全平台编译指南

## 概述

本文档说明如何从 Ollama 的 `llama/llama.cpp` 目录编译所有平台的二进制文件。

## 目录结构

```
/Users/fred/Documents/GitHub/Others/ollama/llama/
├── llama.cpp/          # llama.cpp 源码 (已应用 Ollama 补丁)
├── patches/            # Ollama 35 个补丁
├── llama.go            # Go 绑定
└── ...
```

## macOS 编译 (已完成)

### 已完成编译

| 平台 | 后端 | 状态 | 路径 |
|------|------|------|------|
| macOS | Metal | ✅ | `moxing/bin/darwin-arm64-metal/` |
| macOS | CPU | ✅ | `moxing/bin/darwin-arm64-cpu/` |

### 使用方法

```bash
cd /Users/fred/Documents/GitHub/cycleuser/MoXing
./scripts/build-all-platforms.sh
```

## Linux 编译

### 前置条件

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl

# CUDA (可选)
sudo apt-get install -y nvidia-cuda-toolkit

# Vulkan (可选)
sudo apt-get install -y libvulkan-dev vulkan-tools

# ROCm (可选，AMD GPU)
sudo apt-get install -y rocm-dev
```

### 编译步骤

```bash
# 1. 准备源码
cd /tmp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# 2. 应用 Ollama 补丁
wget https://raw.githubusercontent.com/ollama/ollama/main/llama/patches/0001-ggml-backend-malloc-and-free-using-the-same-compiler.patch
# ... 应用所有 35 个补丁

# 3. 设置 CMake
curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CMakeLists.txt" -o CMakeLists.txt
mkdir -p cmake
curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/cmake/common.cmake" -o cmake/common.cmake
curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/cmake/license.cmake" -o cmake/license.cmake

# 4. 编译 Linux CUDA
mkdir build-linux-cuda && cd build-linux-cuda
cmake .. -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_HIPBLAS=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build . --config Release -j$(nproc)

# 5. 复制二进制
mkdir -p /path/to/MoXing/moxing/bin/linux-x64-cuda
cp bin/llama-server /path/to/MoXing/moxing/bin/linux-x64-cuda/
cp bin/llama-cli /path/to/MoXing/moxing/bin/linux-x64-cuda/
cp bin/llama-bench /path/to/MoXing/moxing/bin/linux-x64-cuda/
cp bin/llama-quantize /path/to/MoXing/moxing/bin/linux-x64-cuda/
```

### 使用自动脚本

```bash
# 在 Linux 系统上运行
./scripts/build-linux-cuda.sh /path/to/output /path/to/llama.cpp
./scripts/build-linux-vulkan.sh /path/to/output /path/to/llama.cpp
./scripts/build-linux-rocm.sh /path/to/output /path/to/llama.cpp
```

## Windows 编译

### 前置条件

1. **Visual Studio 2022** (带 C++ 桌面开发)
2. **CMake** 3.20+
3. **Git**

可选：
- **CUDA Toolkit** (NVIDIA GPU)
- **Vulkan SDK** (任何 GPU)

### 编译步骤 (PowerShell)

```powershell
# 1. 准备源码
cd C:\tmp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# 2. 应用 Ollama 补丁 (手动或使用 patch 命令)
# ...

# 3. 设置 CMake
curl https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CMakeLists.txt -o CMakeLists.txt
mkdir cmake
curl https://raw.githubusercontent.com/ggml-org/llama.cpp/master/cmake/common.cmake -o cmake\common.cmake
curl https://raw.githubusercontent.com/ggml-org/llama.cpp/master/cmake/license.cmake -o cmake\license.cmake

# 4. 编译 Windows CUDA
mkdir build-windows-cuda
cd build-windows-cuda
cmake .. `
    -DGGML_CUDA=ON `
    -DGGML_VULKAN=OFF `
    -DGGML_HIPBLAS=OFF `
    -DGGML_METAL=OFF `
    -DLLAMA_BUILD_SERVER=ON `
    -DCMAKE_GENERATOR="Visual Studio 17 2022"
cmake --build . --config Release -j $env:NUMBER_OF_PROCESSORS

# 5. 复制二进制
New-Item -ItemType Directory -Force -Path "C:\MoXing\moxing\bin\windows-x64-cuda"
Copy-Item "bin\Release\llama-server.exe" "C:\MoXing\moxing\bin\windows-x64-cuda\"
Copy-Item "bin\Release\llama-cli.exe" "C:\MoXing\moxing\bin\windows-x64-cuda\"
Copy-Item "bin\Release\llama-bench.exe" "C:\MoXing\moxing\bin\windows-x64-cuda\"
Copy-Item "bin\Release\llama-quantize.exe" "C:\MoXing\moxing\bin\windows-x64-cuda\"
```

### 使用自动脚本

```powershell
# 在 Windows 系统上运行
powershell .\scripts\build-windows-cuda.ps1 C:\output C:\llama.cpp
powershell .\scripts\build-windows-vulkan.ps1 C:\output C:\llama.cpp
```

## 编译配置对比

| 平台 | CMake 参数 | 后端 | 输出 |
|------|-----------|------|------|
| macOS Metal | `-DGGML_METAL=ON` | Metal | darwin-arm64-metal |
| macOS CPU | `-DGGML_METAL=OFF` | CPU | darwin-arm64-cpu |
| Linux CUDA | `-DGGML_CUDA=ON` | CUDA | linux-x64-cuda |
| Linux Vulkan | `-DGGML_VULKAN=ON` | Vulkan | linux-x64-vulkan |
| Linux ROCm | `-DGGML_HIPBLAS=ON` | ROCm | linux-x64-rocm |
| Windows CUDA | `-DGGML_CUDA=ON` | CUDA | windows-x64-cuda |
| Windows Vulkan | `-DGGML_VULKAN=ON` | Vulkan | windows-x64-vulkan |
| Windows CPU | 默认 | CPU | windows-x64-cpu |

## 测试编译结果

### macOS

```bash
# 测试服务器
./moxing/bin/darwin-arm64-metal/llama-server --version

# 运行模型
moxing ollama serve carstenuhlig/omnicoder-9b
```

### Linux

```bash
# 测试服务器
./moxing/bin/linux-x64-cuda/llama-server --version

# 运行模型
moxing ollama serve carstenuhlig/omnicoder-9b -b cuda
```

### Windows

```powershell
# 测试服务器
.\moxing\bin\windows-x64-cuda\llama-server.exe --version

# 运行模型
moxing ollama serve carstenuhlig/omnicoder-9b -b cuda
```

## 常见问题

### Q: 编译失败，提示找不到 CMake 文件

A: 确保已下载完整的 CMake 构建系统：
```bash
curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CMakeLists.txt" -o CMakeLists.txt
mkdir -p cmake ggml/cmake
# 下载所有 cmake 文件
```

### Q: CUDA 编译失败

A: 确保已安装 CUDA Toolkit 并且版本兼容：
```bash
nvcc --version  # 检查 CUDA 版本
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
```

### Q: ROCm 编译失败

A: 确保已安装 ROCm 并且 GPU 受支持：
```bash
rocm-smi  # 检查 ROCm 状态
cmake .. -DGGML_HIPBLAS=ON -DCMAKE_HIP_ARCHITECTURES=gfx900;gfx906;gfx908;gfx942
```

## 性能优化建议

### macOS (Metal)

```bash
cmake .. -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
```

### Linux (CUDA)

```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
```

### Windows

```powershell
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_GENERATOR="Visual Studio 17 2022"
```

## 文件清单

编译完成后，每个平台应包含：

| 文件 | 说明 |
|------|------|
| llama-server | 主服务器程序 |
| llama-cli | 命令行推理工具 |
| llama-bench | 性能基准测试 |
| llama-quantize | 模型量化工具 |

## 参考资料

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [MoXing GitHub](https://github.com/cycleuser/MoXing)

---

*最后更新：2026 年 4 月 3 日*