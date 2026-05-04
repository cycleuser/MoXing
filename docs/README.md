# Ollama Backend Documentation Index

## 📚 Available Documents

### 1. [Performance Comparison Matrix](./ollama_backend_comparison.md) 📊
**Quick visual reference with all performance data**

- Performance tables for all models
- Load time comparisons
- Memory usage charts
- Backend selection guide
- Quick start commands

**Best for:** Quick lookup of performance numbers and choosing the right backend

---

### 2. [Comprehensive Test Report](./ollama_backend_test_report.md) 📋
**Detailed analysis and findings**

- Complete test methodology
- Individual model performance details
- Backend recommendations
- Key findings and conclusions
- Production deployment guidance

**Best for:** Understanding the full test results and detailed recommendations

---

### 3. [Quick Reference Guide](./ollama_backend_quick_reference.md) 🚀
**Fast lookup for common tasks**

- Model support matrix
- Quick command examples
- Common use cases
- Performance tips

**Best for:** Quickly finding the right command for your use case

---

## 🎯 Which Document Should I Read?

| I want to... | Read this document |
|--------------|-------------------|
| See performance numbers at a glance | [Performance Comparison](./ollama_backend_comparison.md) |
| Understand detailed test results | [Comprehensive Report](./ollama_backend_test_report.md) |
| Find quick command examples | [Quick Reference](./ollama_backend_quick_reference.md) |
| Learn about specific models | [Comprehensive Report](./ollama_backend_test_report.md) |
| Get deployment recommendations | [Comprehensive Report](./ollama_backend_test_report.md) |
| Check if my model will work | [Quick Reference](./ollama_backend_quick_reference.md) |

---

## 🔑 Key Findings Summary

### Performance Winners

```
🥇 ROCm (AMD RX 7900 XTX - 24GB)
   95% success rate, 12-20 tok/s avg
   Can run: All models tested

🥈 CUDA (NVIDIA RTX 4070 - 8GB)  
   60% success rate, 10-15 tok/s avg
   Can run: Models < 8GB

🥉 Vulkan (Cross-platform)
   50% success rate, 7-12 tok/s avg
   Can run: Small-medium models

🏅 CPU (Fallback)
   100% success rate, 0.5-5 tok/s avg
   Can run: All models (very slow)
```

### Model Size Guidelines

| Size | Best Backend | VRAM Required |
|------|-------------|---------------|
| **< 2GB** | ROCm or CUDA | Any GPU |
| **2-6GB** | ROCm or CUDA | 8GB+ |
| **6-10GB** | ROCm | 24GB recommended |
| **10-17GB** | ROCm | 24GB required |
| **17-23GB** | ROCm | 24GB required |
| **> 23GB** | Quantize or distribute | Multiple GPUs |

---

## 🚀 Quick Start

### Most Common Commands

```bash
# Small model - CUDA (NVIDIA GPU)
moxing ollama serve gemma4:e2b -b cuda -d gpu0

# Large model - ROCm (AMD GPU)
moxing ollama serve gemma4:26b -b rocm -d gpu1

# Cross-platform - Vulkan
moxing ollama serve model -b vulkan

# Fallback - CPU
moxing ollama serve small-model -b cpu
```

---

## 📖 Model Support at a Glance

### Gemma4 Series

| Model | Size | CUDA | ROCm | Vulkan | CPU |
|-------|------|------|------|--------|-----|
| gemma4:e2b | 7.2GB | ✅ | ⭐ | ✅ | ✅ |
| gemma4:e4b | 9.6GB | ⚠️ | ⭐ | ⚠️ | ✅ |
| gemma4:26b | 17GB | ❌ | ⭐ | ❌ | ⚠️ |
| gemma4:31b | 19GB | ❌ | ⭐ | ❌ | ⚠️ |
| gemma4:31b-it-q8_0 | 33GB | ❌ | ❌ | ❌ | ⚠️ |

### Qwen3.5 Series

| Model | Size | CUDA | ROCm | Vulkan | CPU |
|-------|------|------|------|--------|-----|
| qwen3.5-abliterated:0.8B | 1.0GB | ✅ | ⭐ | ✅ | ✅ |
| qwen3.5-abliterated:2B | 1.9GB | ✅ | ⭐ | ✅ | ✅ |
| qwen3.5-abliterated:4b | 3.3GB | ✅ | ⭐ | ✅ | ✅ |
| qwen3.5-abliterated:9b | 6.6GB | ✅ | ⭐ | ✅ | ✅ |
| qwen3.5:27b | 17GB | ❌ | ⭐ | ❌ | ⚠️ |
| qwen3.5:35b | 23GB | ❌ | ⚠️ | ❌ | ⚠️ |

Legend: ✅ Works | ⚠️ Limited | ❌ Cannot run | ⭐ Best performance

---

## 📞 Getting Help

1. **Check the docs** - Start with [Quick Reference](./ollama_backend_quick_reference.md)
2. **Verify your setup** - Run `moxing devices` to see available hardware
3. **Test with small models** - Start with qwen3.5-abliterated:0.8B
4. **Monitor resources** - Use `nvidia-smi` or `rocm-smi`

---

**Last Updated:** 2026-04-06  
**Version:** 1.0  
**Author:** MoXing Test Suite