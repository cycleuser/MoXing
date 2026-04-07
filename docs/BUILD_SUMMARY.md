# Ollama 补丁版 llama.cpp 编译总结

## 完成状态

### ✅ macOS (Metal) - 已完成

**源码**: `/Users/fred/Documents/GitHub/Others/ollama/llama/llama.cpp`  
**补丁**: 35 个 Ollama 补丁已应用  
**编译**: 成功编译静态链接版本  

**输出文件**:
| 文件 | 大小 | 路径 |
|------|------|------|
| llama-server | 15 MB | moxing/bin/darwin-arm64-metal/ |
| llama-cli | 10 MB | moxing/bin/darwin-arm64-metal/ |
| llama-bench | 6.1 MB | moxing/bin/darwin-arm64-metal/ |
| llama-quantize | 4.6 MB | moxing/bin/darwin-arm64-metal/ |

**注意**: 需要代码签名才能在 macOS 上运行
```bash
codesign --force --deep --sign - moxing/bin/darwin-arm64-metal/llama-server
```

### 📝 Linux - 需要 Linux 系统编译

**编译脚本已创建**:
- `moxing/bin/build-linux-cuda.sh`
- `moxing/bin/build-linux-vulkan.sh`
- `moxing/bin/build-linux-rocm.sh`

**在 Linux 系统上运行**:
```bash
cd /path/to/MoXing
./moxing/bin/build-linux-cuda.sh /output/path /path/to/ollama/llama/llama.cpp
```

### 📝 Windows - 需要 Windows 系统编译

**编译脚本已创建**:
- `moxing/bin/build-windows-cuda.ps1`
- `moxing/bin/build-windows-vulkan.ps1`

**在 Windows 系统上运行 (PowerShell)**:
```powershell
cd C:\MoXing
.\moxing\bin\build-windows-cuda.ps1 C:\output C:\ollama\llama\llama.cpp
```

## 跨平台编译脚本

### 主脚本
`scripts/build-all-platforms.sh` - 在 macOS 上运行，编译 macOS 版本并创建其他平台脚本

### 平台专用脚本

| 脚本 | 平台 | 后端 | 需要 |
|------|------|------|------|
| build-linux-cuda.sh | Linux | CUDA | NVIDIA GPU + CUDA Toolkit |
| build-linux-vulkan.sh | Linux | Vulkan | Vulkan SDK |
| build-linux-rocm.sh | Linux | ROCm | AMD GPU + ROCm |
| build-windows-cuda.ps1 | Windows | CUDA | Visual Studio + CUDA |
| build-windows-vulkan.ps1 | Windows | Vulkan | Visual Studio + Vulkan SDK |

## 使用方法

### macOS (当前平台)

```bash
# 1. 签名二进制文件（必需）
codesign --force --deep --sign - /Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/darwin-arm64-metal/llama-server

# 2. 运行
moxing ollama serve carstenuhlig/omnicoder-9b
```

### Linux

```bash
# 1. 准备环境
sudo apt-get install -y build-essential cmake git curl
sudo apt-get install -y nvidia-cuda-toolkit  # 如需 CUDA

# 2. 编译
cd /path/to/MoXing
./moxing/bin/build-linux-cuda.sh ./moxing/bin/linux-x64-cuda /path/to/ollama/llama/llama.cpp

# 3. 运行
moxing ollama serve carstenuhlig/omnicoder-9b -b cuda
```

### Windows

```powershell
# 1. 安装 Visual Studio 2022 + CUDA (可选)

# 2. 编译
cd C:\MoXing
powershell .\moxing\bin\build-windows-cuda.ps1 .\moxing\bin\windows-x64-cuda C:\ollama\llama\llama.cpp

# 3. 运行
moxing ollama serve carstenuhlig/omnicoder-9b -b cuda
```

## 相关文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 主编译脚本 | scripts/build-all-platforms.sh | 全平台编译入口 |
| Linux CUDA | moxing/bin/build-linux-cuda.sh | Linux CUDA 编译 |
| Linux Vulkan | moxing/bin/build-linux-vulkan.sh | Linux Vulkan 编译 |
| Linux ROCm | moxing/bin/build-linux-rocm.sh | Linux ROCm 编译 |
| Windows CUDA | moxing/bin/build-windows-cuda.ps1 | Windows CUDA 编译 |
| Windows Vulkan | moxing/bin/build-windows-vulkan.ps1 | Windows Vulkan 编译 |
| 编译指南 | CROSS_PLATFORM_BUILD.md | 详细编译文档 |
| 编译报告 | OLLAMA_COMPILATION_REPORT.md | 编译总结 |

## 源码信息

- **Ollama 源码**: `/Users/fred/Documents/GitHub/Others/ollama/llama/`
- **llama.cpp**: `/Users/fred/Documents/GitHub/Others/ollama/llama/llama.cpp/`
- **补丁目录**: `/Users/fred/Documents/GitHub/Others/ollama/llama/patches/` (35 个补丁)

## 注意事项

1. **macOS 签名**: 新编译的二进制文件需要代码签名才能运行
2. **跨平台限制**: 每个平台需要在对应系统上编译
3. **依赖项**: 各后端需要相应的 SDK（CUDA、Vulkan、ROCm 等）
4. **补丁应用**: 确保 Ollama 补丁已正确应用到 llama.cpp 源码

## 下一步

1. ✅ macOS 版本已编译完成
2. ⏳ 在 Linux 系统上运行 Linux 编译脚本
3. ⏳ 在 Windows 系统上运行 Windows 编译脚本

---

*报告生成时间：2026 年 4 月 3 日*  
*编译源：Ollama llama/llama.cpp (35 patches)*
