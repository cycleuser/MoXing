# MoXing Ollama Runner 架构设计

## 项目目标

实现一个完整的 Ollama Runner 二进制文件系统，支持所有 GPU 后端（CUDA、ROCm、Vulkan）和 CPU，使 moxing 能够直接运行 Ollama 已下载的模型。

## 当前状态分析

### 已有组件

1. **llama.cpp 二进制文件** (`moxing/bin/`)
   - 支持平台：linux-x64 (cuda/rocm/vulkan/cpu), darwin-arm64-metal, windows-x64
   - 用途：运行标准 GGUF 文件
   - 位置：`moxing/bin/{platform}-{backend}/`

2. **Ollama 源码** (`ollama/`)
   - 完整的 Ollama Go 项目
   - 包含 runner：`ollama/runner/llamarunner/`
   - 使用 llama.cpp 的 vendor 版本：`ollama/llama/vendor/`

3. **现有 Ollama 集成** (`moxing/ollama.py`, `moxing/cli.py`)
   - `moxing ollama list` - 列出模型
   - `moxing ollama serve <model>` - 使用系统 Ollama 运行
   - 支持从 Ollama 目录读取 GGUF 文件

### 问题识别

1. **缺失 Ollama 特定的 runner 二进制文件**
   - 某些模型架构（如 gemma4）需要 Ollama 的 patched llama.cpp
   - 当前使用系统 `ollama` 命令，不支持设备选择 (`-d gpu1`)

2. **设备选择限制**
   - 系统 Ollama 使用环境变量控制设备，不够灵活
   - 需要支持类似 `moxing ollama run gemma4:31b -d gpu1 -b rocm` 的用法

## 目标架构

### 1. Ollama Runner 二进制文件

```
moxing/bin/
├── linux-x64-cpu/
│   ├── llama-server (标准 llama.cpp)
│   ├── ollama-runner (Ollama 特定的 runner)
│   └── ...
├── linux-x64-cuda/
│   ├── llama-server
│   ├── ollama-runner
│   └── libggml-cuda.so
├── linux-x64-rocm/
│   ├── llama-server
│   ├── ollama-runner
│   └── libggml-hip.so
├── linux-x64-vulkan/
│   ├── llama-server
│   ├── ollama-runner
│   └── libggml-vulkan.so
└── ... (darwin, windows)
```

### 2. 构建系统

```
scripts/
├── build_ollama_runner.py          # 主构建脚本
├── build_ollama_runner.sh          # Linux/macOS 构建入口
├── build_ollama_runner.bat         # Windows 构建入口
└── ollama-runner-config/
    ├── CMakeLists.patch            # Ollama vendor 补丁
    └── build_linux.sh              # Linux 平台脚本
```

### 3. 代码集成

#### moxing/ollama_runner.py (新文件)
```python
"""
Ollama Runner 管理器
支持直接运行 Ollama 模型，支持多 GPU 选择
"""

class OllamaRunner:
    """
    管理 Ollama runner 二进制文件的下载和执行
    """
    def __init__(self, backend: str = "auto", device: str = "auto"):
        self.backend = backend
        self.device = device
        self.runner_path = self._get_runner_path()
    
    def serve(self, model_name: str, port: int, host: str, **kwargs):
        """
        使用 Ollama runner 启动模型服务
        
        Args:
            model_name: Ollama 模型名称 (如 "gemma4:31b")
            port: 服务端口
            host: 服务主机
            device: 设备选择 (如 "gpu0", "gpu1", "cpu")
            backend: 后端类型 ("cuda", "rocm", "vulkan", "cpu")
        """
        pass
    
    def _get_runner_path(self) -> Path:
        """获取对应后端的 runner 路径"""
        pass
```

#### moxing/cli.py 修改
```python
# 在 ollama_serve_impl 中
if use_ollama_runner:
    from moxing.ollama_runner import OllamaRunner
    runner = OllamaRunner(backend=backend, device=device)
    runner.serve(model, port, host, ctx_size=ctx_size)
else:
    # 现有 llama.cpp 路径
    serve_with_ollama_backend(...)
```

### 4. 使用方式

```bash
# 列出现有 Ollama 模型
moxing ollama list

# 使用特定 GPU 运行 (推荐)
moxing ollama serve gemma4:31b -d gpu1 -b rocm
moxing ollama serve gemma4:26b -d gpu0 -b cuda
moxing ollama serve llama3:8b -d gpu0 -b vulkan

# 自动选择设备
moxing ollama serve gemma4:31b

# 交互式运行
moxing ollama run gemma4:31b -d gpu1 -b rocm
```

## 实现计划

### Phase 1: 构建系统 (优先级: 高)

1. **创建构建脚本**
   - `scripts/build_ollama_runner.sh` - 主构建脚本
   - 从 `ollama/llama/vendor/` 编译
   - 支持 CPU/CUDA/ROCm/Vulkan 后端

2. **CMake 配置**
   - 使用 Ollama 的 vendor 目录
   - 应用必要的补丁
   - 静态/动态链接配置

3. **输出目录结构**
   ```
   moxing/bin/linux-x64-{backend}/
   ├── ollama-runner
   ├── libllama.so
   ├── libggml.so
   └── libggml-{backend}.so
   ```

### Phase 2: 集成代码 (优先级: 高)

1. **创建 ollama_runner.py**
   - BinaryManager 扩展，支持 Ollama runner
   - 设备选择逻辑
   - 模型路径解析

2. **修改 cli.py**
   - 更新 `ollama_serve` 命令
   - 支持 `-d/--device` 参数
   - 自动检测模型架构

3. **修改 ollama.py**
   - 添加模型架构检测
   - 区分 llama.cpp 兼容和 Ollama-only 模型

### Phase 3: 测试 (优先级: 高)

1. **功能测试**
   - 所有后端编译
   - 模型加载测试
   - 设备选择测试

2. **性能基准**
   - 对比系统 Ollama
   - 对比 llama.cpp 直接运行

3. **集成测试**
   - CLI 命令测试
   - API 兼容性测试

## 技术细节

### Ollama Runner 编译

Ollama 的 runner 是一个 Go 程序，链接 llama.cpp 库：

```bash
# 1. 编译 llama.cpp 库 (从 ollama/llama/vendor/)
cd ollama/llama/vendor
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON ...
make -j

# 2. 编译 Go runner
cd ollama/
go build -o runner ./cmd/runner/

# 3. 或者编译完整的 ollama
go build -o ollama .
```

### 设备选择实现

Ollama runner 通过环境变量控制设备：

```python
# 在 moxing/ollama_runner.py 中
def _set_device_env(self, device: str, backend: str):
    """设置设备选择环境变量"""
    if backend == "cuda":
        if device.startswith("gpu"):
            gpu_id = int(device.replace("gpu", ""))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif backend == "rocm":
        if device.startswith("gpu"):
            gpu_id = int(device.replace("gpu", ""))
            os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    elif backend == "vulkan":
        # Vulkan 使用 llama.cpp 的设备索引
        pass
```

### 模型路径解析

```python
def _get_ollama_model_path(self, model_name: str) -> Path:
    """
    从 Ollama 目录结构解析模型路径
    
    Ollama 目录结构:
    ~/.ollama/models/
    ├── manifests/
    │   └── registry.ollama.ai/
    │       └── library/
    │           └── gemma4/
    │               └── 31b  -> 指向 blobs 的 manifest
    └── blobs/
        └── sha256-xxx  -> 实际的 GGUF 文件
    """
    pass
```

## 依赖关系

```
MoXing CLI (moxing/cli.py)
    ↓ 调用
OllamaRunner (moxing/ollama_runner.py)
    ↓ 执行
ollama-runner 二进制 (moxing/bin/.../ollama-runner)
    ↓ 链接
llama.cpp 库 (libllama.so, libggml-*.so)
    ↓ 运行
Ollama 模型 (~/.ollama/models/)
```

## 风险评估

### 技术风险

1. **Ollama 版本兼容性**
   - Ollama 更新可能导致 runner API 变化
   - 缓解：锁定 Ollama 版本，定期测试

2. **二进制文件大小**
   - 每个后端的 runner 约 50-100MB
   - 缓解：按需下载，GitHub Release 分发

3. **设备选择限制**
   - 某些后端可能不支持细粒度设备选择
   - 缓解：优雅降级到自动选择

### 实现风险

1. **构建复杂度**
   - 需要 Go + C++ 混合构建
   - 缓解：提供预编译二进制

2. **平台差异**
   - Linux/macOS/Windows 构建差异
   - 缓解：分平台脚本，CI 构建

## 成功标准

1. ✅ 所有后端成功编译 (CPU/CUDA/ROCm/Vulkan)
2. ✅ 支持 `moxing ollama serve/run` 命令
3. ✅ 支持 `-d/--device` 设备选择
4. ✅ 支持 gemma4 等 Ollama-specific 模型
5. ✅ 性能不低于系统 Ollama

## 下一步行动

1. **立即**: 创建 `scripts/build_ollama_runner.sh` 构建脚本
2. **接下来**: 编译测试所有后端
3. **然后**: 创建 `moxing/ollama_runner.py` 模块
4. **最后**: 更新 CLI 并集成测试
