# MoXing 最终修复版本

## 修复的问题

### 1. ✅ 移除密码提示
- 问题：`_get_macos_gpu_info()` 使用 `sudo powermetrics` 导致密码提示
- 解决：改用 `system_profiler SPDisplaysDataType`（无需 sudo）

### 2. ✅ 简化 run 命令输出
- 问题：`run` 命令显示大量服务器启动信息（Binary, Command, Working dir等）
- 解决：
  - 在 `LlamaServer` 添加 `quiet` 参数
  - `run` 命令传入 `quiet=True`
  - 只显示 "Loading model..." 和 "Model loaded!"

### 3. ✅ 实时监控显示
- 问题：监控信息没有在生成过程中实时更新
- 解决：
  - 简化监控实现，移除后台线程
  - 按需收集系统统计信息
  - 在输入前和回复后显示资源和性能

## 命令行为

### moxing ollama run（交互式聊天）

```bash
moxing ollama run carstenuhlig/omnicoder-9b -v --kv-cache tq2 -c 65536
```

输出：
```
╭──────────────── Configuration ────────────────╮
│ Model: carstenuhlig/omnicoder-9b              │
│ Size: 5.3 GB                                  │
│ Context: 65536                                │
│ Backend: metal                                │
│ Device: Apple M4                              │
│ KV Cache: tq2                                 │
╰────────────────────────────────────────────────╯

Loading model...
Model loaded!
Interactive chat ready! Type 'exit' or 'quit' to end.
Ctrl+C to stop

📊 GPU: 6352 MB | RAM: 10.65 GB | CPU: 0.0%

You: 你会做什么
Assistant: 我是一个AI助手，可以帮助你回答问题、编写代码、翻译文本...
  125 tokens | 10.23s | 12.2 tok/s | TTFT: 0.85s

📊 GPU: 6308 MB | RAM: 10.82 GB | CPU: 15.5%

You: [继续输入...]
```

### moxing ollama serve（服务器模式）

```bash
# 终端监控
moxing ollama serve omnicoder-9b -v

# Web 监控
moxing ollama serve omnicoder-9b -w

# 同时启用
moxing ollama serve omnicoder-9b -v -w
```

## 关键修改文件

### moxing/server.py
- 添加 `quiet` 参数到 `__init__`
- 在 `start()` 方法中根据 `quiet` 参数控制输出

### moxing/cli.py
- 修改 `run_with_verbose_monitor()` 函数
- 简化监控逻辑，移除后台线程
- 添加 `_collect_system_stats()` 辅助函数
- 修改 `ollama_run()` 传入 `quiet=True`

### moxing/enhanced_monitor.py
- 修改 `_get_macos_gpu_info()` 移除 sudo

## 测试命令

```bash
# 安装
pip install -e ./

# 测试交互式聊天
moxing ollama run carstenuhlig/omnicoder-9b -v --kv-cache tq2 -c 65536

# 测试服务器 + 终端监控
moxing ollama serve omnicoder-9b -v --kv-cache tq2 -c 65536

# 测试服务器 + Web 监控
moxing ollama serve omnicoder-9b -w --kv-cache tq2 -c 65536

# 测试单次提问
moxing ollama run omnicoder-9b -p "Hello" -v
```

## 功能对比

| 功能 | run 命令 | serve 命令 |
|------|---------|-----------|
| 用途 | 交互式对话 | 后台服务 |
| 输出 | 简洁（模型名、大小、配置） | 详细（Binary, Command等） |
| 监控 | 终端实时显示 | 可选终端(-v) 或 Web(-w) |
| 交互 | 流式输出，实时统计 | 纯服务器，无交互 |
| Web | 不启用 | 可选启用(-w) |

## 已知限制

1. macOS 上 GPU 内存显示为 0（因为无法访问进程 GPU 内存）
2. 监控信息在输入前和回复后显示，不在生成过程中刷新
3. Web 监控需要服务器启动后手动打开浏览器

## 下一步改进建议

1. 实现生成过程中的实时监控刷新（使用 Live Display）
2. 添加更详细的 GPU 内存监控（支持 Metal Performance API）
3. 支持历史会话保存和加载
4. 添加导出监控数据功能（CSV/JSON）