# MoXing Binary 更新机制

## 问题背景

某些新模型架构（如 `gemma4`）需要最新版本的 llama.cpp 才能运行：

```
llama_model_loader: error loading model: unknown model architecture: 'gemma4'
```

这是因为 bundled binaries 中的 llama.cpp 版本较旧，不支持新架构。

## 解决方案

MoXing 现已支持自动 binary 更新机制，可以轻松获取最新版本的 llama.cpp。

## 使用方法

### 1. 自动更新（推荐）

```bash
moxing update-binaries
```

这会：
- 检查 GitHub 上的最新版本
- 如果有更新，提示确认
- 下载并安装最新 binaries

### 2. 强制更新

```bash
moxing update-binaries --force
```

即使已经是最新版本也会重新下载。

### 3. 免确认更新

```bash
moxing update-binaries -f -y
```

适用于脚本自动化。

### 4. 下载 binaries

```bash
# 下载当前平台默认后端
moxing download-binaries

# 下载特定后端
moxing download-binaries -b metal      # macOS Metal
moxing download-binaries -b cuda       # NVIDIA GPU
moxing download-binaries -b vulkan     # AMD/Intel GPU
moxing download-binaries -b cpu        # CPU only

# 下载所有后端
moxing download-binaries --backend all
```

## 工作原理

### 更新检查流程

1. **版本检测**
   - 读取当前安装的 binaries 版本
   - 查询 GitHub releases 获取最新版本

2. **更新提示**
   - 如果有新版本，显示版本对比
   - 用户确认后开始下载

3. **下载来源**
   - 主源：MoXing GitHub releases
   - 备用源：llama.cpp GitHub releases

4. **Fallback 机制**
   - 如果更新失败，自动使用 bundled binaries
   - 确保始终有可用的 binaries

### Binary 存储位置

```bash
# 缓存目录（新下载的 binaries）
~/.cache/moxing/binaries/{os}-{arch}-{backend}/

# Bundled 目录（随包安装的 binaries）
moxing/bin/{os}-{arch}-{backend}/

# 配置目录
~/.config/moxing/
  - skip_update      # 跳过更新检查标志
```

## 完整示例：运行 gemma4

```bash
# 1. 更新 binaries 到最新版本
moxing update-binaries -f -y

# 2. 拉取模型
ollama pull gemma4

# 3. 运行模型
moxing ollama serve gemma4
```

如果更新后仍然不支持，可能需要等待：
1. llama.cpp 官方支持该架构
2. Ollama 发布对应的 patches
3. MoXing 更新 binaries 源

## 自动化脚本

### 定期更新（每周）

```bash
#!/bin/bash
# 添加到 crontab: 0 3 * * 0 /path/to/update_moxing.sh

moxing update-binaries -y 2>&1 | logger -t moxing-update
```

### 启动前检查

```bash
#!/bin/bash
# 在启动重要服务前检查更新

if ! moxing ollama serve gemma4 2>&1 | grep -q "unknown model architecture"; then
    echo "Model loaded successfully"
else
    echo "Updating binaries..."
    moxing update-binaries -f -y
    moxing ollama serve gemma4
fi
```

## 版本历史

Binaries 版本格式：`b{build_number}-{commit_hash}`

例如：`b1234-abc1234`

- `b1234`: llama.cpp build number
- `abc1234`: git commit hash

## 故障排除

### Q: 更新失败怎么办？

A: 会自动 fallback 到 bundled binaries。可以手动重试：
```bash
moxing download-binaries --force
```

### Q: 如何跳过更新检查？

A: 设置环境变量或创建标记文件：
```bash
export MOXING_NO_UPDATE_CHECK=1
# 或
moxing update-binaries --yes  # 然后手动取消
```

### Q: 如何查看当前版本？

A: 
```bash
moxing --version
# 输出包含 binaries 版本信息
```

### Q: 多个后端如何管理？

A: 每个后端独立管理：
```bash
# 只更新 Metal 后端
moxing download-binaries -b metal --force

# 检查所有后端
moxing download-binaries --list
```

## 注意事项

1. **网络要求**: 更新需要从 GitHub 下载，确保网络畅通
2. **磁盘空间**: 每个后端约 50-100 MB
3. **权限问题**: macOS 可能需要授权
4. **兼容性**: 新版 binaries 可能不兼容旧模型

## 相关命令

```bash
# 查看帮助
moxing update-binaries --help
moxing download-binaries --help

# 清除缓存
moxing clear-cache --binaries

# 查看版本
moxing --version
```

## 参考资料

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [MoXing GitHub](https://github.com/cycleuser/MoXing)
- [Ollama GitHub](https://github.com/ollama/ollama)
