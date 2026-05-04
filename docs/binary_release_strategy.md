# MoXing Binary Release Strategy

## 📦 Binary Package Strategy

### Overview

MoXing uses a **hybrid approach** for binary distribution:

1. **llama.cpp binaries** - Bundled with MoXing (small, ~50-150MB per backend)
2. **Ollama libraries** - Use system Ollama's built-in libraries (no duplication)

### Why This Approach?

**Benefits:**
- ✅ Smaller package size
- ✅ No duplication of large CUDA libraries
- ✅ Use latest Ollama libraries automatically
- ✅ Only need to provide what's missing (ROCm)

**Size comparison:**

| Component | Bundled | System Ollama | Savings |
|-----------|---------|---------------|---------|
| CUDA v12 | 2.5GB | ✅ Use system | 2.5GB |
| CUDA v13 | - | ✅ Use system (932MB) | - |
| ROCm | 230MB | ❌ Not included | - |
| Vulkan | 101MB | ✅ Use system (55MB) | 46MB |
| CPU | 47MB | ✅ Use system | 47MB |

## 🔧 Backend Setup

### CUDA (Recommended: v13)

**No setup required** - Uses Ollama's built-in libraries

```bash
# CUDA v13 is preferred (smaller and faster)
moxing ollama serve model -b cuda -d gpu0

# Environment variables set automatically:
# OLLAMA_LLM_LIBRARY=cuda_v13
```

**Why CUDA v13?**
- 932MB total vs 2.4GB for v12
- 363MB libggml-cuda.so vs 1.6GB for v12
- 4x smaller, same performance

### ROCm (AMD GPUs)

**One-time setup required:**

```bash
# Run the setup script
./scripts/prepare_rocm_library.sh

# Or manually:
sudo mkdir -p /usr/lib/ollama/rocm
sudo cp moxing/bin/linux-x64-rocm-ollama/libggml-hip.so /usr/lib/ollama/rocm/
```

**Usage:**
```bash
moxing ollama serve model -b rocm -d gpu1

# Environment variables set automatically:
# OLLAMA_LLM_LIBRARY=rocm
# HIP_VISIBLE_DEVICES=0
```

**Why ROCm needs setup:**
- Ollama doesn't include ROCm libraries by default
- We provide the 62MB libggml-hip.so compiled from llama.cpp
- One-time setup enables all AMD GPU support

### Vulkan

**No setup required** - Uses Ollama's built-in libraries

```bash
moxing ollama serve model -b vulkan

# Environment variables set automatically:
# OLLAMA_VULKAN=1
```

### CPU

**No setup required** - Uses Ollama's built-in libraries

```bash
moxing ollama serve model -b cpu

# Environment variables set automatically:
# OLLAMA_LLM_LIBRARY=cpu
```

## 📊 Binary Sizes

### Ollama Built-in Libraries

| Library | Size | Location |
|---------|------|----------|
| CUDA v12 | 2.4GB | `/usr/lib/ollama/cuda_v12/` |
| CUDA v13 | 932MB | `/usr/lib/ollama/cuda_v13/` |
| Vulkan | 55MB | `/usr/lib/ollama/vulkan/` |
| CPU | ~6MB | `/usr/lib/ollama/` |
| ROCm | ❌ | Not included |

### MoXing-Provided Libraries

| Library | Size | Purpose |
|---------|------|---------|
| ROCm libggml-hip.so | 62MB | AMD GPU support |
| llama.cpp binaries | 50-150MB | Standard GGUF inference |

## 🚀 Release Package Contents

### For GitHub Release

**Option 1: Minimal Package (Recommended)**
```
moxing-0.3.0-py3-none-any.whl (~100KB)
├─ Python code
├─ CLI commands
└─ Documentation

+ Separate binary downloads:
├─ moxing-binaries-linux-x64.tar.gz (~200MB)
│  ├─ llama.cpp binaries (CUDA, ROCm, Vulkan, CPU)
│  └─ ROCm library for Ollama
```

**Option 2: Full Package**
```
moxing-0.3.0-linux-x64.tar.gz (~250MB)
├─ Wheel file
├─ llama.cpp binaries
└─ ROCm library
```

### What NOT to Include

❌ **Don't include:**
- CUDA libraries (use system Ollama's)
- Vulkan libraries (use system Ollama's)
- CPU libraries (use system Ollama's)

✅ **Do include:**
- llama.cpp binaries (for standard GGUF models)
- ROCm library (not included in Ollama)

## 📝 Installation Guide

### Prerequisites

1. **Install Ollama** (provides CUDA, Vulkan, CPU libraries)
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Install MoXing**
   ```bash
   pip install moxing
   ```

3. **Setup ROCm** (AMD GPUs only)
   ```bash
   moxing setup-rocm
   # or
   ./scripts/prepare_rocm_library.sh
   ```

### Verification

```bash
# Check available devices
moxing devices

# Test CUDA
moxing ollama serve gemma4:e2b -b cuda -d gpu0

# Test ROCm (AMD)
moxing ollama serve gemma4:e2b -b rocm -d gpu1

# Test Vulkan
moxing ollama serve gemma4:e2b -b vulkan
```

## 🎯 Summary

| Backend | Setup Required | Package Size | Source |
|---------|---------------|--------------|--------|
| **CUDA v13** | None | 0MB | System Ollama |
| **ROCm** | One-time | 62MB | MoXing |
| **Vulkan** | None | 0MB | System Ollama |
| **CPU** | None | 0MB | System Ollama |

**Total package size:** ~100KB (wheel) + 62MB (ROCm) = **~62MB**

vs. bundled approach: **2.5GB+**

**Savings:** 97% smaller!