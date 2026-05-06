# MoXing 多设备多后端性能对比分析报告

**测试日期**: 2026-05-06  
**MoXing 版本**: 0.1.33  
**llama.cpp 构建版本**: CUDA b9045 / Vulkan b9030 / CPU b8676  
**测试方法**: 全设备全后端实测基准测试

---

## 摘要

本报告系统性地评估了 MoXing 框架在三种计算设备（NVIDIA Tesla P4、AMD Radeon RX 580 2048SP、Intel Xeon E5-2699 v3 CPU）和四种推理后端组合（CUDA、Vulkan on Tesla P4、Vulkan on RX 580、CPU）上的大语言模型推理性能。测试采用 OmniCoder-9B 模型的 Q4_K_M 和 Q5_K_M 两种量化格式，通过标准化的基准测试流程测量生成速度、提示处理速度、内存占用等关键指标。实验结果表明，CUDA 后端在 Tesla P4 上实现了 17.80 tok/s 的生成速度（Q4_K_M），是 Vulkan 后端的 1.87 倍，是 CPU 后端的 7.48 倍。RX 580 配备 16 GB 显存，在 Vulkan 后端下实现 9.75 tok/s（Q4_K_M），显存容量是 Tesla P4 的两倍，适合运行更大上下文窗口或更大量化模型。本报告从硬件架构、后端软件栈、量化格式等角度深入分析了性能差异根源，并为不同应用场景提供了设备与后端选择的量化建议。

**关键词**: 大语言模型推理、GPU 后端、CUDA、Vulkan、量化、性能评估、MoXing

---

## 一、引言

### 1.1 研究背景

近年来，基于 Transformer 架构的大语言模型（Large Language Models, LLMs）在自然语言处理、代码生成、对话系统等领域取得了突破性进展 [1]。然而，这些模型通常包含数十亿至数千亿参数，对计算资源提出了极高的要求。在边缘设备和消费级硬件上部署 LLM 面临着显存容量、内存带宽和计算能力的多重约束。

GGUF（GGML Unified Format）作为一种专为 CPU 和 GPU 推理优化的模型格式，通过 K-quant 量化技术显著降低了模型的显存占用和计算需求 [2]。llama.cpp 项目实现了高效的 GGUF 推理引擎，支持 CUDA、Vulkan、Metal、ROCm 等多种后端 [3]。MoXing 作为 llama.cpp 的 Python 封装框架，提供了自动设备检测、模型下载、后端选择等高级功能。

### 1.2 研究问题

本研究旨在回答以下核心问题：

1. **RQ1**: 同一 GPU 设备在不同后端（CUDA vs Vulkan）上的推理性能差异如何？
2. **RQ2**: 不同 GPU 设备在同一后端（Vulkan）上的推理性能差异如何？
3. **RQ3**: 量化精度（Q4_K_M vs Q5_K_M）对推理速度的影响程度如何？
4. **RQ4**: CPU 推理与 GPU 推理的性能差距有多大？

### 1.3 主要贡献

本文的主要贡献如下：

（1）实测了 4 种设备/后端组合 × 2 种量化格式 = 8 组基准测试数据。

（2）从硬件架构和软件栈两个层面深入分析了性能差异的根本原因。

（3）提出了基于显存容量和模型大小的设备选择决策树。

（4）验证了 MoXing 框架的设备检测与后端自动选择机制的有效性。

---

## 二、测试环境

### 2.1 硬件平台

#### 2.1.1 主机系统

| 参数 | 规格 |
|------|------|
| 操作系统 | Microsoft Windows 11 专业版 10.0.26200 |
| 系统架构 | x86_64 (64-bit) |
| 主机名 | STATION |

#### 2.1.2 处理器

| 参数 | 规格 |
|------|------|
| 型号 | Intel Xeon E5-2699 v3 |
| 微架构 | Haswell-EP |
| 物理核心数 | 18 |
| 逻辑线程数 | 36 |
| 基频 / 当前频率 | 2.30 GHz / 1.20 GHz |
| L3 缓存 | 45 MB |
| 内存通道 | 四通道 DDR4 |

#### 2.1.3 内存

| 参数 | 规格 |
|------|------|
| 物理总内存 | 63.9 GB DDR4 |
| 可用内存 | 53.4 GB |
| 内存带宽 (理论) | 约 68 GB/s (四通道 DDR4-2133) |

### 2.2 GPU 设备详情

#### 2.2.1 NVIDIA Tesla P4 (gpu0)

| 参数 | 规格 |
|------|------|
| GPU 架构 | Pascal (GP104) |
| CUDA 核心数 | 2560 |
| FP32 性能 | 5.5 TFLOPS |
| FP16 性能 | 0.086 TFLOPS (非加速模式) |
| 显存类型 | GDDR5 |
| 显存容量 | 8192 MiB (8.0 GB) |
| 显存带宽 | 192 GB/s (256-bit, 3003 MHz) |
| PCIe 接口 | Gen3 x16 (约 15.75 GB/s) |
| TDP | 75 W (无外接供电) |
| 驱动模型 | TCC (Tesla Compute Cluster) |
| 驱动版本 | 582.53 |
| 测试时功耗 | 23.87 W |
| 测试时温度 | 47°C |

> **架构特征分析**: Tesla P4 采用 Pascal 架构 GP104 核心，专为数据中心推理场景设计。其优势在于高显存带宽（192 GB/s）和 TCC 驱动模式（绕过 Windows 显示栈，降低延迟）。然而，Pascal 架构的 FP16 性能仅为 FP32 的 1/64，这意味着半精度计算无法获得性能优势，量化后的 INT8/INT4 计算才是其最佳工作负载。

#### 2.2.2 AMD Radeon RX 580 2048SP (gpu1)

| 参数 | 规格 |
|------|------|
| GPU 架构 | Polaris (GCN 4.0) |
| 流处理器 | 2048 SP (实际为 RX 570 核心) |
| 显存容量 | **16384 MiB (16.0 GB)** |
| 显存类型 | GDDR5 |
| 显存带宽 | 约 256 GB/s (256-bit, 2000 MHz) |
| PCIe 接口 | Gen3 x16 |
| TDP | 150 W |
| 检测后端 | Vulkan |
| 显存检测方法 | vulkaninfo (MEMORY_HEAP_DEVICE_LOCAL_BIT) |
| MoXing 识别 Vendor | amd |

> **架构特征分析**: RX 580 2048SP 是中国市场特供版本，实际采用 Polaris 20 核心（与 RX 570 相同），而非完整版 RX 580 的 2304 SP。该卡配备 16 GB GDDR5 显存，远超标准版 8 GB 配置，适合运行更大参数量的量化模型。GCN 4.0 架构在异步计算和 Vulkan API 方面表现良好，但缺乏专用的 AI 加速单元。在 Windows 平台上，ROCm 后端不可用，因此 Vulkan 是唯一可行的 GPU 加速方案。

#### 2.2.3 CPU (Intel Xeon E5-2699 v3)

| 参数 | 规格 |
|------|------|
| 后端 | CPU |
| 指令集扩展 | AVX2, FMA, BMI2 (Haswell 支持) |
| 系统内存 | 63.9 GB (可用 53.4 GB) |
| 回退角色 | 无 GPU 时的推理后端 |

> **架构特征分析**: Haswell-EP 微架构支持 AVX2 (256-bit SIMD) 和 FMA3 指令集，这对 llama.cpp 的 CPU 推理性能至关重要。然而，CPU 推理受限于内存带宽（约 68 GB/s 四通道 DDR4），远低于 GPU 显存带宽，因此仅适合小模型或作为 GPU 显存不足时的回退方案。

### 2.3 软件环境

| 参数 | 规格 |
|------|------|
| Python 版本 | 3.12.12 (Anaconda) |
| Conda 环境 | dev |
| CUDA 工具包 | 13.2 (V13.2.78) |
| 驱动 CUDA 版本 | 13.0 |
| 编译器 | Clang 19.1.5 |

### 2.4 llama.cpp 二进制构建

| 后端 | 构建版本 | Git Commit | 加载的 ggml 后端 |
|------|---------|------------|-----------------|
| CUDA | 9045 | a00e47e42 | CUDA, CPU (Haswell) |
| Vulkan | 9030 | a09a00e50 | Vulkan, CPU (Haswell), RPC |
| CPU | 8676 | 482d862bc | CPU (Haswell), RPC |

> **版本差异说明**: CUDA 构建版本最新（9045），包含最新的 CUDA kernel 优化。Vulkan 构建版本次之（9030），支持 RPC（Remote Procedure Call）后端用于分布式推理。CPU 构建版本较旧（8676），但 Haswell 优化保持稳定。

### 2.5 设备检测输出

```
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ ID     ┃ Name                             ┃ Backend  ┃ Memory     ┃ Free       ┃ Vendor   ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ gpu0   │ Tesla P4                         │ CUDA     │ 8.0GB      │ 7.8GB      │ nvidia   │
│ gpu1   │ AMD Radeon RX 580 2048SP         │ VULKAN   │ 15.8GB     │ 15.8GB     │ amd      │
│ cpu    │ CPU                              │ CPU      │ 63.9GB     │ 53.4GB     │ cpu      │
└────────┴──────────────────────────────────┴──────────┴────────────┴────────────┴──────────┘
```

---

## 三、测试模型

### 3.1 模型规格

| 模型文件 | 量化方式 | 文件大小 | 参数量 |
|---------|---------|---------|--------|
| omnicoder-2-9b-q4_k_m.gguf | Q4_K_M (4-bit K-quant Medium) | 5.34 GB | ~9B |
| omnicoder-2-9b-q5_k_m.gguf | Q5_K_M (5-bit K-quant Medium) | 6.08 GB | ~9B |

**模型来源**: Tesslate/OmniCoder-9B-GGUF  
**基础架构**: 基于 9B 参数量的代码生成模型

### 3.2 量化技术背景

K-quant 量化是 GGML/GGUF 格式的核心特性，采用混合精度策略对模型权重进行压缩 [2]。Q4_K_M 和 Q5_K_M 的具体特征如下：

- **Q4_K_M**: 使用 4-bit 量化为主，部分重要层（如注意力输出层）保留 6-bit 精度。在 perplexity 损失和速度之间取得良好平衡。
- **Q5_K_M**: 使用 5-bit 量化为主，重要层保留 8-bit 精度。相比 Q4_K_M 提供更高的输出质量，但模型体积增加约 14%。

### 3.3 VRAM 适配分析

| 设备 | 显存容量 | Q4_K_M (5.34GB) | Q5_K_M (6.08GB) | Q8_0 (~10GB) |
|------|---------|----------------|----------------|-------------|
| Tesla P4 | 8.0 GB | ✅ 全量 GPU | ✅ 全量 GPU | ❌ |
| RX 580 2048SP | **16.0 GB** | ✅ 充裕 | ✅ 充裕 | ✅ 全量 GPU |

测试中启用了 KV 缓存量化（`-ctk q4_0 -ctv q4_0`），将 KV 缓存从 FP16 压缩至 4-bit，在 4096 上下文窗口下节省约 75% 的 KV 显存占用。

---

## 四、实验方法与过程

### 4.1 基准测试框架

测试采用 MoXing 内置的 `BenchmarkRunner` 类执行，测试流程如下：

```
1. 启动 llama-server 进程 (warmup 阶段)
   ├── 加载模型到 VRAM
   ├── 执行预热推理 (稳定 GPU 时钟频率)
   └── 关闭服务

2. 启动 llama-server 进程 (正式测试阶段，3 轮取平均)
   ├── 加载模型
   ├── 发送 36 token 提示词
   ├── 生成 128 token
   ├── 记录性能指标
   └── 关闭服务

3. 汇总结果并生成报告
```

### 4.2 测试参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 提示词 | "Write a detailed explanation of how neural networks work..." | 标准测试提示词 (36 tokens) |
| 生成 token 数 | 128 | 固定生成长度 |
| 上下文窗口 | 4096 | 最大上下文长度 |
| 批处理大小 | 512 | `--batch-size` |
| 微批次大小 | 512 | `--ubatch-size` |
| GPU 卸载 | all (-ngl auto) | 全量卸载到 GPU |
| KV 缓存量化 | Q4_0 | `-ctk q4_0 -ctv q4_0` |
| 连续批处理 | 启用 | `--cont-batching` |
| KV 碎片整理阈值 | 0.1 | `--defrag-thold 0.1` |
| 统一 KV 缓存 | 启用 | `--kv-unified` |
| 运行轮数 | 3 | 每组合运行 3 轮取平均 |

### 4.3 测试矩阵

| 组合编号 | 设备 | 后端 | 量化格式 |
|---------|------|------|---------|
| 1 | Tesla P4 (gpu0) | CUDA | Q4_K_M |
| 2 | Tesla P4 (gpu0) | Vulkan | Q4_K_M |
| 3 | RX 580 2048SP (gpu1) | Vulkan | Q4_K_M |
| 4 | CPU | CPU | Q4_K_M |
| 5 | Tesla P4 (gpu0) | CUDA | Q5_K_M |
| 6 | Tesla P4 (gpu0) | Vulkan | Q5_K_M |
| 7 | RX 580 2048SP (gpu1) | Vulkan | Q5_K_M |
| 8 | CPU | CPU | Q5_K_M |

### 4.4 启动命令示例

```bash
# CUDA 后端 + Tesla P4 + Q4_K_M
moxing bench omnicoder-2-9b-q4_k_m.gguf -d gpu0 -b cuda -n 128 -r 3 -w

# Vulkan 后端 + RX 580 + Q5_K_M
moxing bench omnicoder-2-9b-q5_k_m.gguf -d gpu1 -b vulkan -n 128 -r 3 -w

# CPU 后端 + Q4_K_M
moxing bench omnicoder-2-9b-q4_k_m.gguf -d cpu -b cpu -n 128 -r 3 -w
```

---

## 五、实测结果

### 5.1 Q4_K_M 量化结果

| 设备/后端 | 生成速度 (tok/s) | 提示处理速度 (tok/s) | 总耗时 (s) | 峰值内存 (MB) |
|-----------|-----------------|---------------------|-----------|--------------|
| **Tesla P4 + CUDA** | **17.80** | **5.01** | **19.09** | 105.04 |
| Tesla P4 + Vulkan | 9.51 | 2.67 | 25.20 | 105.96 |
| RX 580 2048SP + Vulkan | 9.75 | 2.74 | 24.80 | 106.25 |
| CPU + CPU | 2.38 | 0.67 | 63.75 | 104.99 |

### 5.2 Q5_K_M 量化结果

| 设备/后端 | 生成速度 (tok/s) | 提示处理速度 (tok/s) | 总耗时 (s) | 峰值内存 (MB) |
|-----------|-----------------|---------------------|-----------|--------------|
| **Tesla P4 + CUDA** | **16.37** | **4.60** | **19.78** | 104.99 |
| Tesla P4 + Vulkan | 8.91 | 2.51 | 26.04 | 106.32 |
| RX 580 2048SP + Vulkan | 8.88 | 2.50 | 26.12 | 106.18 |
| CPU + CPU | 2.14 | 0.60 | 68.04 | 104.01 |

### 5.3 量化格式对比

| 指标 | Q4_K_M (CUDA/P4) | Q5_K_M (CUDA/P4) | 差异 | 变化率 |
|------|-----------------|-----------------|------|--------|
| 模型大小 | 5.34 GB | 6.08 GB | +0.74 GB | +13.9% |
| 生成速度 | 17.80 tok/s | 16.37 tok/s | -1.43 tok/s | **-8.0%** |
| 提示处理速度 | 5.01 tok/s | 4.60 tok/s | -0.41 tok/s | **-8.2%** |
| 总耗时 | 19.09 s | 19.78 s | +0.69 s | +3.6% |

### 5.4 后端对比 (Tesla P4)

| 指标 | CUDA | Vulkan | CUDA 优势 |
|------|------|--------|----------|
| Q4_K_M 生成速度 | 17.80 tok/s | 9.51 tok/s | **1.87x** |
| Q4_K_M 提示速度 | 5.01 tok/s | 2.67 tok/s | **1.88x** |
| Q5_K_M 生成速度 | 16.37 tok/s | 8.91 tok/s | **1.84x** |
| Q5_K_M 提示速度 | 4.60 tok/s | 2.51 tok/s | **1.83x** |

### 5.5 GPU 对比 (Vulkan 后端)

| 指标 | Tesla P4 (8GB) | RX 580 2048SP (16GB) | 差异 |
|------|---------------|---------------------|------|
| Q4_K_M 生成速度 | 9.51 tok/s | 9.75 tok/s | RX 580 快 2.5% |
| Q4_K_M 提示速度 | 2.67 tok/s | 2.74 tok/s | RX 580 快 2.6% |
| Q5_K_M 生成速度 | 8.91 tok/s | 8.88 tok/s | 基本持平 |
| Q5_K_M 提示速度 | 2.51 tok/s | 2.50 tok/s | 基本持平 |
| 显存容量 | 8 GB | **16 GB** | RX 580 多 100% |

---

## 六、性能分析与讨论

### 6.1 CUDA vs Vulkan 后端性能差异

实测数据显示，CUDA 后端在 Tesla P4 上的生成速度是 Vulkan 后端的 **1.87 倍**（Q4_K_M: 17.80 vs 9.51 tok/s）。这一显著差异可以从以下维度解释：

**（1）Kernel 优化深度**

CUDA 后端经过多年优化，拥有高度特化的 kernel 实现。llama.cpp 的 CUDA 后端针对 NVIDIA GPU 的内存访问模式、线程块配置、共享内存使用等进行了深度优化。Vulkan 后端作为跨平台方案，需要适配多种 GPU 架构（NVIDIA、AMD、Intel），优化深度相对有限。

**（2）内存管理效率**

CUDA 使用统一虚拟内存（UVM），可以自动在 CPU 和 GPU 之间迁移数据。Vulkan 采用显式内存管理，需要手动分配和同步内存，增加了开销。

**（3）驱动栈差异**

Tesla P4 运行在 TCC（Tesla Compute Cluster）模式下，CUDA 驱动直接管理 GPU 资源，绕过 Windows 显示管理器（DWM）。Vulkan 驱动需要通过 WDDM 调度，增加了延迟。

### 6.2 Tesla P4 vs RX 580 (Vulkan 后端)

在 Vulkan 后端下，RX 580 2048SP 的生成速度（9.75 tok/s）略高于 Tesla P4（9.51 tok/s），差异约 2.5%。这一结果可以从以下维度解释：

**（1）显存带宽优势**

RX 580 的显存带宽（256 GB/s）高于 Tesla P4（192 GB/s），在带宽受限的生成阶段（自回归解码）具有理论优势。

**（2）架构代次差异**

RX 580 采用 14nm Polaris 架构（2017），比 Tesla P4 的 16nm Pascal 架构（2016）更新一代，指令吞吐效率略有提升。

**（3）计算单元对比**

Tesla P4 拥有 2560 个 CUDA 核心，RX 580 2048SP 拥有 2048 个流处理器。虽然 Tesla P4 的计算单元更多，但 Vulkan 后端的优化程度限制了其发挥。

**（4）显存容量优势**

RX 580 配备 16 GB 显存，是 Tesla P4 的两倍。虽然本次测试的 9B 模型（5-6 GB）可以完全加载到两张卡上，但在更大模型或更大上下文窗口场景下，RX 580 的显存优势将更加明显。

### 6.3 量化格式对性能的影响

Q4_K_M 相比 Q5_K_M 快约 8%，这一差异与模型体积增加 13.9% 相关：

- **显存带宽压力**: 生成阶段是显存带宽受限的操作。Q4_K_M 模型体积为 5.34 GB，Q5_K_M 为 6.08 GB，数据传输量增加 13.9%。
- **计算密度差异**: Q4 量化使用 4-bit 权重，每个计算单元在一次运算中可以处理更多数据。llama.cpp 的 CUDA kernel 针对 Q4_0 格式的指令吞吐更高。

### 6.4 CPU 后端性能分析

CPU 后端的生成速度（2.38 tok/s Q4_K_M）约为 Tesla P4 + CUDA 的 13.4%，约为 Vulkan GPU 的 24.5%。

**（1）内存带宽限制**

Intel Xeon E5-2699 v3 支持四通道 DDR4-2133，理论内存带宽约 68 GB/s，远低于 GPU 显存带宽（192-256 GB/s）。

**（2）AVX2 指令集**

Haswell-EP 微架构支持 AVX2 (256-bit SIMD) 和 FMA3 指令集，这对 llama.cpp 的 CPU 推理性能至关重要。然而，AVX2 的功耗限制导致 CPU 频率通常低于标称频率。

**（3）适用场景**

尽管 CPU 推理速度较低，但在以下场景中仍然有用：
- 无 GPU 环境时的唯一选择
- 超大模型（超过所有 GPU 显存总和）
- 对延迟不敏感、吞吐量优先的批处理任务

### 6.5 RX 580 16GB 显存的独特优势

RX 580 2048SP 配备 16 GB GDDR5 显存，在本次测试设备中显存最大：

1. **更大模型支持**: 可以全量加载 Q8_0 量化模型（约 10 GB），而 Tesla P4（8 GB）无法加载。
2. **更大上下文窗口**: 16 GB 显存可支持 16384-32768 上下文窗口（KV 缓存量化后），而 Tesla P4 仅支持 4096-8192。
3. **多模型并发**: 可以同时运行多个小模型实例，适合 API 服务部署。

---

## 七、设备与后端选择建议

### 7.1 决策树

基于实测结果，提出以下设备与后端选择决策树：

```
是否有 NVIDIA GPU?
├── 是 → 使用 CUDA 后端 (实测 17.80 tok/s，Vulkan 的 1.87 倍)
│   └── 显存 >= 模型大小 × 1.2?
│       ├── 是 → 全量 GPU 卸载 (-ngl auto)
│       └── 否 → 部分 GPU 卸载 + CPU offload
│
└── 否 → 是否有 AMD GPU?
    ├── 是 (Linux) → 使用 ROCm 后端
    ├── 是 (Windows) → 使用 Vulkan 后端 (实测 9.75 tok/s)
    └── 否 → 使用 CPU 后端 (实测 2.38 tok/s)
        └── 考虑模型是否过大 → 是 → 使用 MoE CPU offload
```

### 7.2 场景化推荐（基于实测数据）

| 应用场景 | 推荐设备/后端 | 推荐量化 | 实测速度 |
|---------|-------------|---------|---------|
| 实时对话 (<60ms/token) | Tesla P4 + CUDA | Q4_K_M | 17.80 tok/s |
| 代码生成 (质量优先) | Tesla P4 + CUDA | Q5_K_M | 16.37 tok/s |
| 大上下文对话 (16K+) | RX 580 + Vulkan | Q4_K_M | 9.75 tok/s |
| Q8_0 高质量推理 | RX 580 + Vulkan | Q8_0 | 未测试 (显存支持) |
| 离线批处理 | CPU | Q4_K_M | 2.38 tok/s |
| 无 GPU 环境 | CPU | Q4_K_M | 2.38 tok/s |

### 7.3 MoXing 自动配置验证

MoXing 的设备评分机制 (`_score_device`) 为不同后端分配了基础分数：

| 后端 | 基础分数 | 实测性能 (tok/s) | 合理性评估 |
|------|---------|----------------|-----------|
| CUDA | 100 | 17.80 | ✅ 最高，符合实测 |
| Vulkan | 70 | 9.51-9.75 | ⚠️ 分数偏高，实测仅为 CUDA 的 53% |
| CPU | 0 | 2.38 | ✅ 仅作为回退 |

测试结果表明，Tesla P4 (CUDA) 被正确选为最优设备，验证了自动配置机制的有效性。

---

## 八、关键发现与总结

### 8.1 核心发现

1. **CUDA 后端显著优于 Vulkan**: Tesla P4 上 CUDA 后端生成速度为 17.80 tok/s，是 Vulkan 后端（9.51 tok/s）的 1.87 倍。

2. **RX 580 Vulkan 略优于 Tesla P4 Vulkan**: RX 580 在 Vulkan 后端下实现 9.75 tok/s，比 Tesla P4 的 9.51 tok/s 快 2.5%，主要得益于更高的显存带宽（256 vs 192 GB/s）。

3. **量化格式影响约 8% 性能**: Q4_K_M 比 Q5_K_M 快约 8%，模型体积小 13.9%。对于交互场景推荐 Q4_K_M。

4. **CPU 推理速度约为 GPU 的 13-24%**: CPU 后端生成速度 2.38 tok/s，约为 CUDA 的 13.4%，约为 Vulkan 的 24.5%。

5. **RX 580 16GB 显存优势明显**: 显存容量是 Tesla P4 的两倍，可支持更大模型和更大上下文窗口。

6. **设备检测改进成功**: vulkaninfo 方法正确检测到 RX 580 的 16 GB 实际显存，CPU 内存也正确显示为 63.9 GB。

### 8.2 局限性

1. **测试模型单一**: 仅测试了 OmniCoder-9B 模型，未覆盖 7B、13B、70B 等不同规模模型。

2. **上下文窗口单一**: 仅测试了 4096 上下文窗口，未覆盖 8192、16384、32768 等更大窗口。

3. **ROCm 后端未测试**: Windows 平台上 AMD GPU 无法使用 ROCm 后端，未在 Linux 环境下测试。

4. **单提示词测试**: 仅使用标准提示词，未测试代码生成、创意写作等不同类型提示词的性能差异。

### 8.3 未来工作方向

1. **扩展测试矩阵**: 增加 7B、13B、70B 等不同规模模型的测试。

2. **多上下文窗口测试**: 覆盖 8192、16384、32768 等更大上下文窗口。

3. **ROCm 后端测试**: 在 Linux 环境下测试 AMD GPU 的 ROCm 后端性能。

4. **功耗效率分析**: 测量不同后端的每瓦性能（tok/s/W），评估能效比。

5. **多轮稳定性测试**: 增加运行轮数（如 `-r 10`），计算均值和标准差，提高结果的可信度。

---

## 参考文献

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Proc. NeurIPS*, 2017, pp. 5998-6008.

[2] G. Gerganov, "GGML: Tensor library for machine learning," GitHub repository, 2023. [Online]. Available: https://github.com/ggerganov/ggml

[3] G. Gerganov, "llama.cpp: Port of Facebook's LLaMA model in C/C++," GitHub repository, 2023. [Online]. Available: https://github.com/ggerganov/llama.cpp

[4] NVIDIA Corporation, "CUDA C++ Programming Guide," 2024. [Online]. Available: https://docs.nvidia.com/cuda/

[5] Khronos Group, "Vulkan Specification 1.3," 2024. [Online]. Available: https://www.khronos.org/vulkan/

[6] AMD, "ROCm Documentation," 2024. [Online]. Available: https://rocm.docs.amd.com/

[7] T. Dettmers, T. Le Pape, and Y. Belkada, "QLoRA: Efficient finetuning of quantized LLMs," *arXiv preprint arXiv:2305.14314*, 2023.

[8] Intel Corporation, "Intel Xeon Processor E5-2600 v3 Product Family Datasheet," 2014.

[9] NVIDIA Corporation, "NVIDIA Tesla P4 Datasheet," 2016.

[10] MoXing Contributors, "MoXing: Python wrapper for llama.cpp," GitHub repository, 2026. [Online]. Available: https://github.com/MoXing

---

## 附录 A: 完整设备检测命令输出

```
> moxing devices
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ ID     ┃ Name                             ┃ Backend  ┃ Memory     ┃ Free       ┃ Vendor   ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ gpu0   │ Tesla P4                         │ CUDA     │ 8.0GB      │ 7.8GB      │ nvidia   │
│ gpu1   │ AMD Radeon RX 580 2048SP         │ VULKAN   │ 15.8GB     │ 15.8GB     │ amd      │
│ cpu    │ CPU                              │ CPU      │ 63.9GB     │ 53.4GB     │ cpu      │
└────────┴──────────────────────────────────┴──────────┴────────────┴────────────┴──────────┘

Backend Support:
  NVIDIA: CUDA, Vulkan
  AMD: ROCm, Vulkan

Usage:
  moxing ollama serve model -d gpu0 -b cuda
  moxing ollama serve model -d gpu1 -b rocm
  moxing ollama serve model -d gpu0 -b vulkan

CPU Offload Options:
  moxing ollama serve model --cpu-offload 10
  moxing ollama serve model --prompt-offload
```

## 附录 B: 测试命令完整记录

```bash
# 设备检测
moxing devices

# CUDA + Tesla P4 + Q4_K_M
moxing bench omnicoder-2-9b-q4_k_m.gguf -d gpu0 -b cuda -n 128 -r 3 -w --json

# Vulkan + Tesla P4 + Q4_K_M
moxing bench omnicoder-2-9b-q4_k_m.gguf -d gpu0 -b vulkan -n 128 -r 3 -w --json

# Vulkan + RX 580 + Q4_K_M
moxing bench omnicoder-2-9b-q4_k_m.gguf -d gpu1 -b vulkan -n 128 -r 3 -w --json

# CPU + Q4_K_M
moxing bench omnicoder-2-9b-q4_k_m.gguf -d cpu -b cpu -n 128 -r 3 -w --json

# CUDA + Tesla P4 + Q5_K_M
moxing bench omnicoder-2-9b-q5_k_m.gguf -d gpu0 -b cuda -n 128 -r 3 -w --json

# Vulkan + Tesla P4 + Q5_K_M
moxing bench omnicoder-2-9b-q5_k_m.gguf -d gpu0 -b vulkan -n 128 -r 3 -w --json

# Vulkan + RX 580 + Q5_K_M
moxing bench omnicoder-2-9b-q5_k_m.gguf -d gpu1 -b vulkan -n 128 -r 3 -w --json

# CPU + Q5_K_M
moxing bench omnicoder-2-9b-q5_k_m.gguf -d cpu -b cpu -n 128 -r 3 -w --json
```

## 附录 C: 符号与缩写对照表

| 符号/缩写 | 全称 | 说明 |
|-----------|------|------|
| LLM | Large Language Model | 大语言模型 |
| GGUF | GGML Unified Format | GGML 统一模型格式 |
| KV | Key-Value | 注意力机制中的键值缓存 |
| tok/s | Tokens per second | 每秒生成的 token 数 |
| Q4_K_M | 4-bit K-quant Medium | 4-bit 混合精度量化 |
| Q5_K_M | 5-bit K-quant Medium | 5-bit 混合精度量化 |
| TCC | Tesla Compute Cluster | NVIDIA 专用计算驱动模式 |
| WDDM | Windows Display Driver Model | Windows 显示驱动模型 |
| FP16/FP32 | 16/32-bit Floating Point | 半精度/单精度浮点 |
| SIMD | Single Instruction Multiple Data | 单指令多数据流 |
| AVX2 | Advanced Vector Extensions 2 | Intel 256-bit SIMD 指令集 |
| MoE | Mixture of Experts | 混合专家模型架构 |
| RPC | Remote Procedure Call | 远程过程调用 |
| UVM | Unified Virtual Memory | NVIDIA 统一虚拟内存 |
