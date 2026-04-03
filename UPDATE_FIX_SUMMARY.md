# Binary 更新机制实现总结

## 问题

用户尝试运行 `gemma4` 模型时遇到错误：

```
llama_model_loaded: error loading model: unknown model architecture: 'gemma4'
```

原因：bundled binaries 中的 llama.cpp 版本过旧，不支持新的模型架构。

## 解决方案

实现了完整的 binary 更新机制，包括：

### 1. 新增命令

#### `moxing update-binaries` - 更新 binaries

```bash
# 检查并更新
moxing update-binaries

# 强制更新
moxing update-binaries --force

# 免确认更新
moxing update-binaries -f -y
```

功能：
- ✅ 自动检查 GitHub 最新版本
- ✅ 数值化版本比较（b8642 > b8475）
- ✅ 用户确认后下载
- ✅ 更新失败自动 fallback 到 bundled binaries

#### `moxing download-binaries` - 下载 binaries

```bash
# 下载当前平台默认后端
moxing download-binaries

# 下载特定后端
moxing download-binaries -b metal
moxing download-binaries -b cuda

# 列出可用后端
moxing download-binaries --list
```

### 2. 修复的问题

1. **版本比较逻辑**
   - 旧代码：字符串比较 `b8475 != b8461` → 误报更新
   - 新代码：数值比较 `8475 < 8642` → 正确判断

2. **导入错误修复**
   - 修复 `server.py` 中的 `estimate_model_size_gb` 导入错误
   - 正确函数名：`estimate_kv_cache_size_gb`

### 3. 完整工作流程

```bash
# 步骤 1: 更新 binaries
moxing update-binaries -f -y

# 步骤 2: 拉取模型
ollama pull gemma4

# 步骤 3: 运行模型
moxing ollama serve gemma4

# 或者一步到位
moxing ollama serve gemma4
# 如果 binaries 过旧，会自动提示更新
```

## 测试验证

### 版本检查测试

```bash
$ moxing update-binaries
Checking for binary updates...

New version available!
  Current:  b8475
  Latest:   b8642

Download update? [y/n] (y): y

Downloading update...

✓ Update complete!
Installed to: /Users/fred/.cache/moxing/binaries
```

### 已经最新测试

```bash
$ moxing update-binaries
Checking for binary updates...

✓ Already up to date
Current version: b8642

Tip: Use --force to re-download binaries
```

### 帮助信息

```bash
$ moxing update-binaries --help
Update llama.cpp binaries to the latest version.

Checks for newer binary versions and downloads them if available.
Supports automatic update detection and one-click updates.

Examples:
    moxing update-binaries           # Check and update if needed
    moxing update-binaries -f        # Force re-download
    moxing update-binaries -f -y     # Force update without confirmation

Note: Uses bundled binaries as fallback if update fails.
```

## 文件变更

### 修改的文件

1. **moxing/cli.py**
   - 添加 `update_binaries_cmd()` 函数
   - 实现版本检查、用户确认、下载流程

2. **moxing/binaries.py**
   - 修复 `check_for_updates()` 版本比较逻辑
   - 数值化比较 build number

3. **moxing/server.py**
   - 修复导入错误 `estimate_model_size_gb` → `estimate_kv_cache_size_gb`

### 新增的文件

1. **BINARY_UPDATE.md** - 完整使用文档
2. **UPDATE_FIX_SUMMARY.md** - 本修复说明

## 未来改进建议

1. **自动更新检查**
   - 启动时自动检查（每周最多一次）
   - 发现新模型架构不支持时自动提示

2. **模型架构检测**
   - 尝试加载模型前检测架构支持
   - 不支持时自动建议更新

3. **增量更新**
   - 只下载差异部分
   - 减少下载时间和带宽

4. **多版本并存**
   - 支持同时安装多个版本
   - 可按模型选择版本

## 使用方法总结

| 场景 | 命令 |
|------|------|
| 日常更新 | `moxing update-binaries` |
| 强制更新 | `moxing update-binaries -f` |
| 脚本自动化 | `moxing update-binaries -f -y` |
| 下载特定后端 | `moxing download-binaries -b metal` |
| 查看帮助 | `moxing update-binaries --help` |
| 清除缓存 | `moxing clear-cache --binaries` |

## 注意事项

1. **网络要求**: 需要从 GitHub 下载（约 50-100 MB）
2. **磁盘空间**: 每个后端约 50-100 MB
3. **更新频率**: 建议每周或遇到新模型时更新
4. **Fallback 机制**: 更新失败自动使用 bundled binaries

## 参考资料

- [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
- [MoXing Binaries](https://github.com/cycleuser/MoXing/releases)
- [BINARY_UPDATE.md](./BINARY_UPDATE.md) - 详细文档
