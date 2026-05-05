# OmniCoder-2-9B Benchmark Report

**Date:** 2026-05-04
**MoXing:** 0.1.29 | **llama.cpp:** d05fe1d7d | **GPU:** RTX 4060 8GB

## 1. Models

| Model | Quant | Size | Architecture |
|-------|-------|------|-------------|
| OmniCoder-2-9B Q4_K_M | Q4_K_M | 5.34GB | qwen35 (SSM+Attn) |
| OmniCoder-2-9B Q5_K_M | Q5_K_M | 6.08GB | qwen35 (SSM+Attn) |

## 2. Runner Compatibility

| Runner | Q4_K_M | Q5_K_M |
|--------|--------|--------|
| llama.cpp CUDA | PASS | PASS |
| llama.cpp Vulkan | PASS (4.3 tok/s) | PASS (4.0 tok/s) |
| llama.cpp CPU | PASS (4.5 tok/s) | PASS (3.9 tok/s) |
| vLLM 0.20.1 | FAIL | FAIL |

vLLM error: `ValueError: GGUF model with architecture qwen35 is not supported yet.`

## 3. Performance

| Model | Startup | Coding | Reasoning | Creative | Factual | **Avg** |
|-------|---------|--------|-----------|----------|---------|---------|
| OmniCoder-2-9B Q4_K_M | 4.6s | 42.0 | 41.8 | 37.8 | 41.9 | **40.9** |
| OmniCoder-2-9B Q5_K_M | 2.8s | 37.8 | 37.6 | 37.7 | 37.7 | **37.7** |

## 4. Q4 vs Q5 Comparison

| Metric | Q4_K_M | Q5_K_M | Delta |
|--------|--------|--------|-------|
| coding tok/s | 42.0 | 37.8 | -4.2 |
| reasoning tok/s | 41.8 | 37.6 | -4.2 |
| creative tok/s | 37.8 | 37.7 | -0.1 |
| factual tok/s | 41.9 | 37.7 | -4.2 |
| **Avg tok/s** | **40.9** | **37.7** | **-3.2** |
| File Size | 5.34GB | 6.08GB | +0.7GB |
| Startup | 4.6s | 2.8s | - |

## 5. Sample Outputs

### Q4_K_M Haiku
```
Silent data streams,
Learning patterns in the dark,
Mind born from the code.
```

### Q5_K_M Coding (first 400 chars)
```python
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Write a Python function.
    *   Purpose: Calculate Fibonacci numbers.
    *   Method: Recursively.
    *   Optimization: With memoization (to avoid redundant calculations).

2.  **Determine the Approach:**
    *   *Recursion:* The classic definition is $F(n) = F(n-1) + F(n-2)$ with base cases $F(0) = 0, F(1) = 1$.
    *   *Memo
```

### Q5_K_M Reasoning (train problem)
To find the meeting time, we need to determine how far apart the trains are when the second train departs and then calculate how long it takes for them to close that remaining gap.

**Step 1: Calculate the distance covered by the first train before the second train starts.**
*   The first train leaves at 3:00 PM traveling at 60 mph.
*   The second train leaves at 4:00 PM.
*   Time elapsed before the second train starts: 1 hour.
*   Distance covered by Train 1: $60 \text{ mph}

## 6. Deployment

```bash
# Best speed (~41 tok/s)
moxing serve omnicoder-2-9b-q4_k_m.gguf -b cuda

# Best quality (~37 tok/s, richer reasoning)
moxing serve omnicoder-2-9b-q5_k_m.gguf -b cuda

# Test script
bash scripts/bench_omnicoder.sh > report.md
```

## 7. Verdict

**OmniCoder-2-9B** is a capable Qwen3.5 model that fits entirely in 8GB consumer GPUs. Q5_K_M is recommended for production use — ~11% slower than Q4_K_M but produces significantly richer reasoning chains via the `<think>` tag mechanism. vLLM deployment is not yet available due to missing transformers support for qwen35 architecture.

*Report generated 2026-05-04 by MoXing 0.1.29*