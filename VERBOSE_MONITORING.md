# MoXing Verbose Monitoring Feature

## Overview

All serve and run commands now support `-v/--verbose` for detailed monitoring:

- Memory/VRAM usage (real-time and historical)
- CPU utilization
- Token statistics (prompt, generated, total)
- Speed metrics (tokens/s, time to first token)
- Server performance summary

## Commands

### 1. moxing serve

Start server with verbose monitoring:

```bash
moxing serve model.gguf -v                     # Verbose monitoring
moxing serve model.gguf -v --kv-cache tq3.5    # TurboQuant + verbose
moxing serve model.gguf -v -c 65536            # Large context + verbose
```

### 2. moxing run

Run model with detailed statistics:

```bash
moxing run model.gguf                          # Interactive chat
moxing run model.gguf -p "Hello"               # Single prompt
moxing run model.gguf -v                       # Interactive chat + verbose
moxing run model.gguf -p "Hello" -v            # Single prompt + stats
moxing run model.gguf --kv-cache tq2 -v        # TurboQuant + verbose
```

### 3. moxing ollama serve

Serve Ollama model with verbose monitoring:

```bash
moxing ollama serve carstenuhlig/omnicoder-9b -v
moxing ollama serve omnicoder-9b -v -c 65536
moxing ollama serve omnicoder-9b -v --kv-cache tq2
```

### 4. moxing ollama run

Run Ollama model with detailed monitoring:

```bash
moxing ollama run carstenuhlig/omnicoder-9b              # Interactive chat
moxing ollama run omnicoder-9b -p "Hello"                # Single prompt
moxing ollama run omnicoder-9b -v                        # Verbose monitoring
moxing ollama run omnicoder-9b -p "What is Python?" -v   # Single prompt + stats
moxing ollama run omnicoder-9b -v -c 65536               # Large context + verbose
moxing ollama run omnicoder-9b --kv-cache tq2 -v         # TurboQuant + verbose
```

## Verbose Output Features

### For serve commands:

When `-v` is enabled, the server displays a real-time monitoring panel (refresh: 1s):

```
🚀 MoXing Monitor - omnicoder-9b.gguf
┌─────────────────────────────────────────┐
│ Model: omnicoder-9b.gguf                │
│ Context: 32,768                         │
│                                         │
│ Tokens:                                 │
│   Prompt: 1,234                         │
│   Generated: 5,678                      │
│   Total: 6,912                          │
│                                         │
│ Speed:                                  │
│   Prompt: 45.2 tok/s                    │
│   Generate: 12.3 tok/s                  │
│   Avg (60s): 13.5 tok/s                 │
│                                         │
│ Memory:                                 │
│   GPU: 2,340 MB (avg: 2,100)            │
│   RAM: 8.5 GB                           │
│                                         │
│ CPU: 15.2% (avg: 12.5%)                 │
│                                         │
│ Requests: 1 processing, 0 deferred      │
└─────────────────────────────────────────┘
```

### For run commands:

When `-v` is enabled, after completion shows:

```
┌─ Session Complete ──────────────────────┐
│ 📊 Performance Summary                  │
│                                         │
│ Tokens: 125 generated                   │
│ Time: 10.23s                            │
│ Speed: 12.2 tok/s                       │
│ Time to first token: 0.85s              │
│                                         │
│ Memory:                                 │
│   GPU/Process: 2,340 MB                 │
│   RAM Used: 8.5 GB                      │
│                                         │
│ Statistics (60s):                       │
│   Avg Speed: 13.5 tok/s                 │
│   Max GPU Memory: 2,500 MB              │
│   Avg CPU: 12.5%                        │
│                                         │
│ Server: http://127.0.0.1:8080           │
│ Model: omnicoder-9b                     │
└─────────────────────────────────────────┘
```

For interactive chat, each response shows:

```
You: What is Python?
Assistant: Python is a high-level programming language...
  125 tokens in 10.23s (12.2 tok/s, TTFT: 0.85s)
```

Session summary at exit:

```
┌─ Chat Session Complete ─────────────────┐
│ 📊 Session Summary                      │
│                                         │
│ Messages: 5 prompts, 5 responses        │
│                                         │
│ Memory:                                 │
│   Avg GPU: 2,100 MB                     │
│   Max GPU: 2,500 MB                     │
│                                         │
│ Performance:                            │
│   Avg Speed: 13.5 tok/s                 │
│   Avg CPU: 12.5%                        │
│                                         │
│ Server: http://127.0.0.1:8080           │
└─────────────────────────────────────────┘
```

## TurboQuant KV Cache

When using TurboQuant, the monitoring shows cache type:

```bash
moxing ollama run omnicoder-9b --kv-cache tq2 -v -c 65536
```

Memory savings with TurboQuant:
- f16: ~1024 MB KV for 32K context
- q4_0: ~288 MB KV (72% savings)
- tq2: ~128 MB KV (87.5% savings)

## Implementation Details

### Key Functions

1. `run_with_verbose_monitor()` - Handles run commands with monitoring
2. `serve_with_verbose_monitor()` - Handles serve commands with monitoring
3. `EnhancedMonitor` - Collects metrics and maintains history

### Metrics Collected

- **Tokens**: prompt_tokens, generated_tokens, total_tokens
- **Speed**: prompt_speed, generate_speed (tok/s)
- **Memory**: gpu_memory_mb, ram_used_mb, process_memory_mb
- **CPU**: cpu_percent
- **Requests**: processing, deferred

### History Tracking

- Max 3600 seconds of history (1 hour)
- Statistics calculated over last 60 seconds
- Real-time updates every 1 second

## Testing

Test commands:

```bash
# Test serve with verbose
moxing ollama serve carstenuhlig/omnicoder-9b -v

# Test single prompt with stats
moxing ollama run omnicoder-9b -p "What is Python?" -v

# Test interactive chat with verbose
moxing ollama run omnicoder-9b -v

# Test TurboQuant
moxing ollama run omnicoder-9b --kv-cache tq2 -v -c 65536
```

## Notes

1. Verbose monitoring requires server to be running with `--metrics` enabled
2. MoXing servers automatically enable `--metrics`
3. For interactive chat, type 'exit' or 'quit' to end session
4. Ctrl+C stops the server and shows final summary