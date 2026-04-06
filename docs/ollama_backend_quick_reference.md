# Ollama Backend Test - Quick Reference

## Test Results Summary

### Success Rate by Backend

| Backend | Device | Success Rate | Avg Speed | VRAM |
|---------|--------|--------------|-----------|------|
| **ROCm** | gpu1 (RX 7900 XTX) | 95% | 12-20 tok/s | 24GB |
| **CUDA** | gpu0 (RTX 4070) | 60% | 10-15 tok/s | 8GB |
| **Vulkan** | gpu0 | 50% | 7-12 tok/s | 8GB |
| **CPU** | cpu | 100% | 0.5-5 tok/s | - |

## Model Support Matrix

### Gemma4 Models

| Model | Size | CUDA (8GB) | ROCm (24GB) | Vulkan | CPU |
|-------|------|------------|-------------|--------|-----|
| gemma4:e2b | 7.2GB | ✅ | ✅ ⭐ | ✅ | ✅ |
| gemma4:e4b | 9.6GB | ⚠️ | ✅ ⭐ | ⚠️ | ✅ |
| gemma4:26b | 17GB | ❌ | ✅ ⭐ | ❌ | ⚠️ |
| gemma4:31b | 19GB | ❌ | ✅ ⭐ | ❌ | ⚠️ |
| gemma4:31b-it-q8_0 | 33GB | ❌ | ❌ | ❌ | ⚠️ |

### Qwen3.5 Models

| Model | Size | CUDA (8GB) | ROCm (24GB) | Vulkan | CPU |
|-------|------|------------|-------------|--------|-----|
| qwen3.5-abliterated:0.8B | 1.0GB | ✅ | ✅ ⭐ | ✅ | ✅ |
| qwen3.5-abliterated:2B | 1.9GB | ✅ | ✅ ⭐ | ✅ | ✅ |
| qwen3.5-abliterated:4b | 3.3GB | ✅ | ✅ ⭐ | ✅ | ✅ |
| qwen3.5-abliterated:9b | 6.6GB | ✅ | ✅ ⭐ | ✅ | ✅ |
| qwen3.5:27b | 17GB | ❌ | ✅ ⭐ | ❌ | ⚠️ |
| qwen3.5:35b | 23GB | ❌ | ⚠️ | ❌ | ⚠️ |

Legend:
- ✅ Works well
- ⚠️ Works but slow or limited
- ❌ Cannot run (insufficient VRAM)
- ⭐ Best performance

## Quick Recommendations

### By Model Size

```bash
# Small models (< 6GB) - Use CUDA or ROCm
moxing ollama serve gemma4:e2b -b cuda -d gpu0
moxing ollama serve qwen3.5-abliterated:9b -b rocm -d gpu1

# Medium models (6-10GB) - Prefer ROCm
moxing ollama serve gemma4:e4b -b rocm -d gpu1

# Large models (15-23GB) - Must use ROCm on 24GB GPU
moxing ollama serve gemma4:26b -b rocm -d gpu1
moxing ollama serve qwen3.5:35b -b rocm -d gpu1

# Testing/Development - Vulkan for compatibility
moxing ollama serve model -b vulkan

# Fallback - CPU (very slow)
moxing ollama serve model -b cpu
```

### By Use Case

**Production - Best Performance:**
- Small models: `moxing ollama serve model -b cuda -d gpu0`
- Large models: `moxing ollama serve model -b rocm -d gpu1`

**Development - Quick Testing:**
- Use small models with any backend
- `moxing ollama serve qwen3.5-abliterated:0.8B -b cuda`

**Compatibility - Cross-Platform:**
- `moxing ollama serve model -b vulkan`

**No GPU Available:**
- `moxing ollama serve small-model -b cpu`

## Performance Tips

1. **Match model size to VRAM:**
   - 8GB VRAM → models < 6GB
   - 24GB VRAM → all tested models

2. **Backend priority:**
   - ROCm > CUDA > Vulkan > CPU

3. **For large models:**
   - Use quantization (Q4_K, Q5_K)
   - Consider CPU offloading if needed

4. **Monitor VRAM:**
   - `nvidia-smi` for CUDA
   - `rocm-smi` for ROCm

## Detailed Report

See full report: [ollama_backend_test_report.md](./ollama_backend_test_report.md)