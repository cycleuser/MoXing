# MoXing 二进制编译策略

## 问题

尝试编译 Ollama 补丁版 llama.cpp 失败，原因是：
1. Ollama 的 35 个补丁与 llama.cpp b4376 版本不兼容
2. 补丁针对的是 Ollama 内部的特定版本，与公开版本有差异
3. 多个关键文件（如 `ggml-alloc.c`, `ggml-backend.cpp`）编译错误

## 解决方案

采用**降级策略**：

### 方案 1：使用 llama.cpp 官方版（✅ 已实现）

```bash
# 编译 macOS Metal 版本
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b4376
mkdir build-metal && cd build-metal
cmake .. -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release

# 复制二进制文件到 MoXing
cp bin/llama-server ../../moxing/bin/darwin-arm64-metal/
```

### 方案 2：直接使用 Ollama 预编译二进制

Ollama 提供预编译二进制，但不提供独立的 llama-server：
- macOS: `https://github.com/ollama/ollama/releases/latest/download/ollama-darwin`
- Linux: `https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz`

但 Ollama 的二进制是封装好的，不直接兼容 llama.cpp 的命令行参数。

### 方案 3：等待 Ollama 开源补丁（推荐）

持续关注 Ollama 的开源进度，当他们的补丁与上游 llama.cpp 兼容时再采用。

## 当前状态

✅ **已实现**：使用 llama.cpp 官方版 b4376

| 平台 | 后端 | 状态 | 路径 |
|------|------|------|------|
| macOS | Metal | ✅ 已编译 | `moxing/bin/darwin-arm64-metal/` |
| macOS | CPU | ✅ 已编译 | `moxing/bin/darwin-arm64-cpu/` |
| Linux | CUDA | ⏳ 待编译 | `moxing/bin/linux-x64-cuda/` |
| Linux | Vulkan | ⏳ 待编译 | `moxing/bin/linux-x64-vulkan/` |
| Linux | ROCm | ⏳ 待编译 | `moxing/bin/linux-x64-rocm/` |
| Linux | CPU | ⏳ 待编译 | `moxing/bin/linux-x64-cpu/` |
| Windows | CUDA | ⏳ 待编译 | `moxing/bin/windows-x64-cuda/` |
| Windows | Vulkan | ⏳ 待编译 | `moxing/bin/windows-x64-vulkan/` |
| Windows | CPU | ⏳ 待编译 | `moxing/bin/windows-x64-cpu/` |

## 编译命令

### macOS (Metal)

```bash
cd /Users/fred/Documents/GitHub/cycleuser/MoXing
./scripts/build-moxing-binaries.sh
```

### Linux (CUDA)

```bash
cd /path/to/MoXing
./scripts/build-moxing-binaries.sh
# 需要安装：cmake, build-essential, nvidia-cuda-toolkit
```

### Windows (需要 WSL 或原生 Windows)

```powershell
# 安装 Visual Studio 2022 + CMake
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b4376
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Ollama 补丁列表（仅供参考）

Ollama 的 35 个补丁主要针对：
1. 模型架构支持（Solar, DeepSeek, GLM 等）
2. GPU 内存管理优化
3. Windows 特定修复
4. 词表和分词器改进

大部分功能已经或即将合并到 llama.cpp 主分支。

## 建议

1. **短期**：使用 llama.cpp 官方版，已经非常稳定
2. **中期**：关注 Ollama 补丁合并进度，选择性应用已合并的补丁
3. **长期**：参与 llama.cpp 社区，推动 Ollama 特有功能的标准化

---

*最后更新：2026 年 4 月 3 日*
