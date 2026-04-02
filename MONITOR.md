# MoXing 监控功能

## 概述

MoXing 提供实时监控功能，可以在页面上显示：
- GPU/CPU 内存使用
- 显卡算力
- Tokens 消耗统计
- 请求处理状态
- Slots 状态

## 使用方法

### 方法 1: 启动独立监控服务器

```bash
# 终端 1: 启动 llama.cpp 服务器
moxing serve model.gguf -p 8080

# 终端 2: 启动监控服务器
moxing monitor start --llama-port 8080

# 打开浏览器: http://127.0.0.1:9090
```

### 方法 2: 使用内置监控页面

服务器默认启用 `--metrics`，可以直接访问监控页面：

```bash
# 启动服务器
moxing serve model.gguf -p 8080

# 在浏览器中打开
open http://127.0.0.1:8080
```

### 方法 3: 终端实时监控

```bash
moxing monitor cli --port 8080
```

### 方法 4: 查看当前统计

```bash
moxing monitor stats --port 8080
```

输出示例：
```
╭───────────────────────────────── Model Info ─────────────────────────────────╮
│ Model: sha256-550e8f7253c8e07997fbce2570d37259b69b0d21faf77e5ed518d4ee4c73d8b3 │
│ Context: 8192                                                               │
│ Batch: 2048                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
       Server Statistics        
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric           ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Prompt Tokens    │ 11        │
│ Generated Tokens │ 10        │
│ Total Tokens     │ 21        │
│ Prompt Speed     │ 9.2 tok/s │
│ Generate Speed   │ 6.2 tok/s │
│ Processing       │ 0         │
│ Deferred         │ 0         │
└──────────────────┴───────────┘
          Slots          
┏━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ ID ┃ Status ┃ Context ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ 0  │ Idle   │ 8192    │
│ 1  │ Idle   │ 8192    │
│ 2  │ Idle   │ 8192    │
│ 3  │ Idle   │ 8192    │
└────┴────────┴─────────┘
```

## Web 监控页面

访问监控页面可以看到：

1. **内存使用**
   - GPU Memory 使用量和百分比
   - RAM 使用量和百分比
   - KV Cache 大小

2. **GPU 状态**
   - 设备名称
   - 利用率
   - 温度
   - 功耗

3. **性能指标**
   - Prompt 处理速度 (tok/s)
   - Token 生成速度 (tok/s)
   - 平均速度
   - Decode 调用次数

4. **Tokens 统计**
   - Prompt Tokens 总量
   - Generated Tokens 总量
   - Total Tokens 总量
   - 最大批次大小

5. **Slots 状态**
   - 每个 Slot 的 ID
   - 当前状态
   - 已生成 tokens

## API 端点

llama.cpp 服务器提供以下监控 API：

| 端点 | 描述 |
|------|------|
| `/metrics` | Prometheus 格式指标 |
| `/slots` | Slots 状态 |
| `/props` | 模型属性 |
| `/health` | 健康检查 |

### 示例

```bash
# 获取指标
curl http://127.0.0.1:8080/metrics

# 获取 slots
curl http://127.0.0.1:8080/slots

# 获取模型属性
curl http://127.0.0.1:8080/props

# 健康检查
curl http://127.0.0.1:8080/health
```

## 指标说明

### Tokens 相关

- `prompt_tokens_total`: 处理的 prompt tokens 总数
- `tokens_predicted_total`: 生成的 tokens 总数
- `prompt_tokens_seconds`: Prompt 处理速度
- `predicted_tokens_seconds`: Token 生成速度

### 请求相关

- `requests_processing`: 正在处理的请求数
- `requests_deferred`: 延迟的请求数

### 性能相关

- `n_decode_total`: llama_decode() 调用次数
- `n_tokens_max`: 最大批次 tokens 数
- `n_busy_slots_per_decode`: 平均忙碌 slots 数

## Python API

```python
from moxing.monitor import MetricsCollector, MemoryInfo, GPUInfo

# 创建收集器
collector = MetricsCollector("127.0.0.1", 8080)

# 获取指标
metrics = collector.fetch_metrics()
print(f"Prompt tokens: {metrics.prompt_tokens_total}")
print(f"Generated tokens: {metrics.tokens_predicted_total}")
print(f"Speed: {metrics.predicted_tokens_per_second:.1f} tok/s")

# 获取 slots
slots = collector.fetch_slots()
for slot in slots:
    print(f"Slot {slot['id']}: {'Processing' if slot['is_processing'] else 'Idle'}")

# 获取模型属性
props = collector.fetch_props()
print(f"Model: {props.get('model_path')}")
```

## 注意事项

1. 服务器必须启动时包含 `--metrics` 参数才能使用监控功能
2. MoXing 默认启用 `--metrics`
3. 监控页面每秒自动刷新
4. 大量请求时建议使用独立的监控服务器端口