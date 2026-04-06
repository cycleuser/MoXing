# 📦 Binary Release Packages

## Package Strategy

MoXing provides **optional binary packages** for enhanced GPU support:

| Package | Size | Purpose | Required? |
|---------|------|---------|-----------|
| **MoXing Wheel** | ~100KB | Core Python package | ✅ Yes |
| **CUDA v13** | 750MB | CUDA 13 libraries | ⚠️ Optional* |
| **ROCm** | ~80MB | ROCm libraries for AMD | ⚠️ Optional† |

*Most users already have CUDA v13 from Ollama installation
†Only needed for AMD GPU users

## Download Links

### From GitHub Releases

```bash
# Core package (required)
pip install moxing

# CUDA v13 (optional - for NVIDIA GPUs without Ollama)
wget https://github.com/cycleuser/MoXing/releases/download/v0.1.26/moxing-cuda-v13-0.1.26.tar.gz

# ROCm (optional - for AMD GPUs)
wget https://github.com/cycleuser/MoXing/releases/download/v0.1.26/moxing-rocm-0.1.26.tar.gz
```

## Size Comparison

### Why We Use CUDA v13 Instead of v12

| Version | Size | Components |
|---------|------|------------|
| **CUDA v13** | **932MB** | libggml-cuda.so (363MB) + cuBLAS (570MB) |
| CUDA v12 | 2.4GB | libggml-cuda.so (1.6GB) + cuBLAS (830MB) |
| **Savings** | **1.5GB (63%)** | 4x smaller! |

**Recommendation:** Use CUDA v13 for faster downloads and smaller disk footprint.

## Installation

### Option 1: Use Ollama's Built-in Libraries (Recommended)

```bash
# Install Ollama (includes CUDA v13, Vulkan, CPU)
curl -fsSL https://ollama.ai/install.sh | sh

# Install MoXing
pip install moxing

# Everything works out of the box!
moxing ollama serve model -b cuda -d gpu0
```

### Option 2: Install CUDA v13 Package

```bash
# Download package
tar -xzf moxing-cuda-v13-0.1.26.tar.gz
cd moxing-cuda-v13-0.1.26

# Install
./install.sh

# Use it
OLLAMA_LLM_LIBRARY=cuda_v13 ollama run model
```

### Option 3: Install ROCm Package (AMD GPUs)

```bash
# Download package
tar -xzf moxing-rocm-0.1.26.tar.gz
cd moxing-rocm-0.1.26

# Install
./install.sh

# Use it
OLLAMA_LLM_LIBRARY=rocm ollama run model
moxing ollama serve model -b rocm -d gpu1
```

## What's Included

### CUDA v13 Package (750MB compressed)

```
moxing-cuda-v13-0.1.26/
├── lib/
│   ├── cuda_v13/
│   │   ├── libggml-cuda.so (363MB)
│   │   ├── libcublasLt.so.13 (517MB)
│   │   ├── libcublas.so.13 (52MB)
│   │   └── libcudart.so.13 (688KB)
│   ├── libggml-base.so
│   └── libggml-cpu-*.so
├── install.sh
└── README.md
```

### ROCm Package (~80MB compressed)

```
moxing-rocm-0.1.26/
├── lib/
│   └── libggml-hip.so (62MB)
├── install.sh
└── README.md
```

## Usage Examples

### CUDA (NVIDIA)

```bash
# Default CUDA v13
moxing ollama serve gemma4:e2b -b cuda -d gpu0

# With environment variable
OLLAMA_LLM_LIBRARY=cuda_v13 moxing ollama serve model

# Specific GPU
CUDA_VISIBLE_DEVICES=0 moxing ollama serve model
```

### ROCm (AMD)

```bash
# Use ROCm backend
moxing ollama serve gemma4:e2b -b rocm -d gpu1

# With environment variable
OLLAMA_LLM_LIBRARY=rocm ollama run model

# Specific GPU
HIP_VISIBLE_DEVICES=0 ollama run model
```

### Vulkan

```bash
# Cross-platform
moxing ollama serve model -b vulkan

# Environment variable
OLLAMA_VULKAN=1 ollama run model
```

### CPU

```bash
# CPU fallback
moxing ollama serve model -b cpu

# Environment variable
OLLAMA_LLM_LIBRARY=cpu ollama run model
```

## Requirements

### For CUDA v13

- NVIDIA GPU with Compute Capability 6.0+
- CUDA 13.x runtime (included in NVIDIA driver 550+)
- Ollama installed (or use package)

### For ROCm

- AMD GPU with ROCm 6.0+ support
- ROCm runtime installed
- Ollama installed

### For Vulkan

- Any GPU with Vulkan support
- Ollama installed

### For CPU

- No special requirements
- Ollama installed

## Troubleshooting

### CUDA v12 vs v13

**Q: My system has CUDA v12, will v13 work?**

A: Yes! CUDA v13 libraries are standalone. The runtime is included in the package.

**Q: Should I use v12 or v13?**

A: Use v13. It's 4x smaller and has the same performance.

### ROCm Installation

**Q: Why does ROCm need manual setup?**

A: Ollama doesn't include ROCm libraries by default. Our package fills this gap.

**Q: Where does the library install?**

A: `/usr/lib/ollama/rocm/libggml-hip.so`

### Performance Issues

**Q: Which backend is fastest?**

A: ROCm > CUDA > Vulkan > CPU (for supported models)

**Q: Why is my model slow?**

A: Check if the model fits in VRAM. Large models may need CPU offloading.

## Building from Source

If you want to build the packages yourself:

```bash
# Clone repository
git clone https://github.com/cycleuser/MoXing.git
cd MoXing

# Build CUDA v13 package
./scripts/package_cuda_v13.sh

# Build ROCm package
./scripts/package_rocm.sh

# Build release package
./scripts/create_release_package.sh
```

## Related Documentation

- [Performance Comparison](./ollama_backend_comparison.md)
- [Test Report](./ollama_backend_test_report.md)
- [Quick Reference](./ollama_backend_quick_reference.md)

---

**Version:** 0.1.26  
**Last Updated:** 2026-04-06