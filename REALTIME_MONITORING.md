# MoXing 实时监控功能更新

## 更新概述

所有 `serve` 和 `run` 命令现在支持实时监控显示：

### 关键改进

1. **实时显示** - 监控信息在运行时实时更新，不再等待结束
2. **终端 + Web 双模式** - 可同时启用终端监控和 Web 监控页面
3. **流式输出** - 对话时直接流式输出，不阻塞监控显示

## 命令使用

### 1. moxing serve

```bash
# 基本服务（无监控）
moxing serve model.gguf

# 终端实时监控（每秒刷新）
moxing serve model.gguf -v

# Web 监控页面
moxing serve model.gguf -w

# 同时启用终端和 Web 监控
moxing serve model.gguf -v -w

# TurboQuant + 终端监控
moxing serve model.gguf -v --kv-cache tq3.5 -c 65536
```

**终端监控显示**（每秒刷新）：
```
┌─ 🚀 MoXing Monitor - omnicoder-9b.gguf ─────────────┐
│ Model: omnicoder-9b.gguf                            │
│ Context: 65,536                                     │
│                                                      │
│ Tokens:                                              │
│   Prompt: 1,234                                      │
│   Generated: 5,678                                   │
│   Total: 6,912                                       │
│                                                      │
│ Speed:                                               │
│   Prompt: 45.2 tok/s                                 │
│   Generate: 12.3 tok/s                               │
│   Avg (60s): 13.5 tok/s                              │
│                                                      │
│ Memory:                                              │
│   GPU: 2,340 MB (avg: 2,100)                         │
│   RAM: 8.5 GB                                        │
│                                                      │
│ CPU: 15.2% (avg: 12.5%)                              │
│                                                      │
│ Requests: 1 processing, 0 deferred                   │
└──────────────────────────────────────────────────────┘
```

**Web 监控**：
- 访问 `http://127.0.0.1:8080`
- 包含历史数据图表
- 实时更新（1秒刷新）

### 2. moxing run

```bash
# 交互式聊天
moxing run model.gguf

# 单次提问
moxing run model.gguf -p "Hello"

# 交互式聊天 + 实时监控
moxing run model.gguf -v

# 单次提问 + 性能统计
moxing run model.gguf -p "What is Python?" -v
```

**交互式聊天监控**：
- 每次输入前显示实时资源使用
- 每次回复后显示性能指标
- 退出时显示完整会话统计

**实时资源监控**（输入前）：
```
┌─ 📊 Live Monitor ────────────────────────────────────┐
│ GPU: 2,340 MB (avg: 2,100) | RAM: 8.5 GB | CPU: 15.2%│
└──────────────────────────────────────────────────────┘

You: What is Python?
Assistant: Python is a high-level programming language...
  125 tokens | 10.23s | 12.2 tok/s | TTFT: 0.85s
```

**会话统计**（退出时）：
```
┌─ 📊 Session Summary ───────────────────────────────┐
│ Messages: 5 prompts, 5 responses                    │
│                                                      │
│ Memory:                                              │
│   Avg GPU: 2,100 MB                                  │
│   Max GPU: 2,500 MB                                  │
│                                                      │
│ Performance:                                         │
│   Avg Speed: 13.5 tok/s                              │
│   Avg CPU: 12.5%                                     │
│                                                      │
│ Server: http://127.0.0.1:8081                        │
└──────────────────────────────────────────────────────┘
```

### 3. moxing ollama serve

```bash
# 基本服务
moxing ollama serve carstenuhlig/omnicoder-9b

# 终端实时监控
moxing ollama serve omnicoder-9b -v

# Web 监控页面
moxing ollama serve omnicoder-9b -w

# 同时启用终端和 Web
moxing ollama serve omnicoder-9b -v -w

# TurboQuant + 监控
moxing ollama serve omnicoder-9b -v --kv-cache tq2 -c 65536
```

### 4. moxing ollama run

```bash
# 交互式聊天
moxing ollama run carstenuhlig/omnicoder-9b

# 单次提问
moxing ollama run omnicoder-9b -p "Hello"

# 交互式聊天 + 实时监控
moxing ollama run omnicoder-9b -v

# 单次提问 + 性能统计
moxing ollama run omnicoder-9b -p "What is Python?" -v

# TurboQuant + 监控
moxing ollama run omnicoder-9b -v --kv-cache tq2 -c 65536
```

## 监控指标说明

### 实时指标
- **GPU Memory**: 进程使用的 GPU 内存（macOS 上显示为进程内存）
- **RAM Used**: 系统已使用的 RAM
- **CPU**: CPU 使用百分比
- **Tokens**: 提示词、生成词、总词数
- **Speed**: 提示词速度、生成速度、平均速度
- **Requests**: 正在处理的请求数、延迟请求数

### 统计指标（60秒窗口）
- **Avg GPU Memory**: 平均 GPU 内存使用
- **Max GPU Memory**: 最大 GPU 内存使用
- **Avg CPU**: 平均 CPU 使用率
- **Avg Speed**: 平均生成速度

### 性能指标
- **TTFT** (Time To First Token): 首个词生成时间
- **Tokens/s**: 每秒生成词数
- **Total Time**: 总耗时

## TurboQuant KV Cache

结合 TurboQuant 使用可大幅减少内存占用：

```bash
# 65K 上下文示例
moxing ollama run omnicoder-9b -v --kv-cache tq2 -c 65536
```

内存节省对比（32K 上下文）：
- f16: ~1024 MB KV
- q4_0: ~288 MB KV（节省 72%）
- tq2: ~128 MB KV（节省 87.5%）

## 实现细节

### 关键函数

1. **run_with_verbose_monitor()**
   - 处理 run 命令的监控
   - 支持单次提示和交互式聊天
   - 实时显示资源和性能指标

2. **serve_with_verbose_monitor()**
   - 处理 serve 命令的监控
   - 支持终端监控（-v）
   - 支持 Web 监控（-w）

3. **EnhancedMonitor**
   - 后台线程收集指标
   - 维护历史数据（最长1小时）
   - 计算统计信息

### 数据收集

- 从 `/metrics` 端点获取 token 统计
- 从 `/slots` 端点获取上下文使用
- 从 `/props` 端点获取模型信息
- 使用 `psutil` 获取系统资源
- 使用 `system_profiler` 获取 GPU 信息（macOS，无需 sudo）

### 刷新频率

- 监控面板：每 1 秒刷新
- 历史数据：最长保留 3600 秒（1小时）
- 统计窗口：60 秒

## 修复的问题

1. ✅ 移除 `sudo powermetrics`（导致密码提示）
2. ✅ 使用 `system_profiler` 替代（无需 sudo）
3. ✅ 实时显示监控信息（不再等待结束）
4. ✅ 支持终端和 Web 双模式
5. ✅ 流式输出不阻塞监控

## 测试命令

```bash
# 测试终端监控
moxing ollama serve carstenuhlig/omnicoder-9b -v

# 测试 Web 监控
moxing ollama serve omnicoder-9b -w

# 测试交互式聊天监控
moxing ollama run omnicoder-9b -v

# 测试单次提问 + 统计
moxing ollama run omnicoder-9b -p "Hello" -v

# 测试 TurboQuant
moxing ollama run omnicoder-9b -v --kv-cache tq2 -c 65536
```