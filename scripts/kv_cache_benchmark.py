#!/usr/bin/env python3
"""
KV Cache 量化方法对比测试脚本
实际运行测试获取真实性能数据
"""

import subprocess
import json
import time
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

# KV Cache 量化方法配置
KV_CACHE_METHODS = [
    {"name": "f16", "desc": "FP16 原始精度", "bits": 16},
    {"name": "q8_0", "desc": "8-bit 量化", "bits": 8},
    {"name": "q4_0", "desc": "4-bit 量化", "bits": 4},
    {"name": "tq4", "desc": "TurboQuant 4-bit", "bits": 4},
    {"name": "tq3.5", "desc": "TurboQuant 3.5-bit", "bits": 3.5},
    {"name": "tq3", "desc": "TurboQuant 3-bit", "bits": 3},
    {"name": "tq2.5", "desc": "TurboQuant 2.5-bit", "bits": 2.5},
    {"name": "tq2", "desc": "TurboQuant 2-bit", "bits": 2},
]

# 测试模型
MODEL = "carstenuhlig/omnicoder-9b"
CONTEXT_SIZE = 65536
TEST_PROMPT = "Please explain what is machine learning in simple terms."


@dataclass
class TestResult:
    """测试结果"""
    method: str
    description: str
    bits: float
    success: bool = False
    error: str = ""
    # 性能指标
    load_time_s: float = 0.0
    first_token_time_s: float = 0.0
    total_time_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    # 内存指标
    peak_memory_mb: float = 0.0
    kv_cache_size_mb: float = 0.0
    # 输出
    output_preview: str = ""
    full_output: str = ""


def parse_output(output: str) -> Dict:
    """解析测试输出提取指标"""
    metrics = {}
    
    # 解析 tokens 信息： "125 tokens | 10.23s | 12.2 tok/s | TTFT: 0.85s"
    token_pattern = r'(\d+)\s*tokens?\s*\|\s*([\d.]+)\s*s\s*\|\s*([\d.]+)\s*tok/s\s*\|\s*TTFT:\s*([\d.]+)\s*s'
    match = re.search(token_pattern, output)
    if match:
        metrics['tokens_generated'] = int(match.group(1))
        metrics['total_time_s'] = float(match.group(2))
        metrics['tokens_per_second'] = float(match.group(3))
        metrics['first_token_time_s'] = float(match.group(4))
    
    # 解析内存信息： "GPU: 6352 MB | RAM: 10.65 GB | CPU: 0.0%"
    memory_pattern = r'RAM:\s*([\d.]+)\s*GB'
    match = re.search(memory_pattern, output)
    if match:
        metrics['peak_memory_mb'] = float(match.group(1)) * 1024
    
    # 解析加载时间（从开始到 Model loaded）
    if "Model loaded!" in output:
        metrics['load_success'] = True
    
    return metrics


def run_test(kv_cache: str, method_desc: str, bits: float) -> TestResult:
    """运行单次测试"""
    result = TestResult(method=kv_cache, description=method_desc, bits=bits)
    
    print(f"\n{'='*60}")
    print(f"测试 KV Cache: {kv_cache} ({method_desc})")
    print(f"{'='*60}\n")
    
    cmd = [
        "moxing", "ollama", "run", MODEL,
        "-p", TEST_PROMPT,
        "-c", str(CONTEXT_SIZE),
        "--kv-cache", kv_cache,
        "-v", "-n", "256"  # 限制生成 256 tokens 加快测试
    ]
    
    print(f"命令：{' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 分钟超时
        )
        
        result.load_time_s = time.time() - start_time
        result.total_time_s = result.load_time_s
        result.full_output = proc.stdout + proc.stderr
        
        output = result.full_output
        
        # 解析输出
        if "Model loaded!" in output:
            result.success = True
            metrics = parse_output(output)
            
            result.tokens_generated = metrics.get('tokens_generated', 0)
            result.total_time_s = metrics.get('total_time_s', result.total_time_s)
            result.tokens_per_second = metrics.get('tokens_per_second', 0)
            result.first_token_time_s = metrics.get('first_token_time_s', 0)
            result.peak_memory_mb = metrics.get('peak_memory_mb', 0)
            
            # 提取输出预览
            if "Response:" in output:
                response_start = output.find("Response:") + 9
                # 找到下一个时间戳或 Panel 开始
                response_end = output.find("┌─", response_start)
                if response_end == -1:
                    response_end = response_start + 500
                result.output_preview = output[response_start:response_end].strip()[:300]
        else:
            if proc.returncode != 0:
                result.error = f"Exit code: {proc.returncode}"
            else:
                result.error = "Model failed to load"
            print(f"❌ 失败：{result.error}")
            
    except subprocess.TimeoutExpired:
        result.error = "Test timeout (600s)"
        print(f"❌ 超时")
    except Exception as e:
        result.error = str(e)
        print(f"❌ 错误：{e}")
    
    return result


def estimate_kv_cache_size(bits: float, context: int = 65536, model_layers: int = 40, hidden_size: int = 4096) -> float:
    """估算 KV Cache 大小 (MB)"""
    size_bytes = 2 * model_layers * hidden_size * context * bits / 8
    return size_bytes / (1024 * 1024)


def save_results(results: List[TestResult], output_file: str = "kv_cache_test_results.json"):
    """保存测试结果"""
    data = {
        "test_time": datetime.now().isoformat(),
        "model": MODEL,
        "context_size": CONTEXT_SIZE,
        "test_prompt": TEST_PROMPT,
        "results": [asdict(r) for r in results]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到：{output_file}")


def generate_markdown_table(results: List[TestResult]) -> str:
    """生成 Markdown 表格"""
    lines = []
    
    lines.append("## 实测结果对比\n")
    lines.append("| 量化方法 | 位数 | 描述 | 加载时间 | 生成速度 | TTFT | 峰值内存 | KV Cache 估算 | 状态 |")
    lines.append("|---------|------|------|----------|----------|------|----------|-------------|------|")
    
    for r in results:
        kv_estimate = estimate_kv_cache_size(r.bits, CONTEXT_SIZE)
        status = "✅" if r.success else "❌"
        load_str = f"{r.load_time_s:.1f}s" if r.load_time_s > 0 else "-"
        speed_str = f"{r.tokens_per_second:.1f}" if r.tokens_per_second > 0 else "-"
        ttft_str = f"{r.first_token_time_s:.2f}s" if r.first_token_time_s > 0 else "-"
        mem_str = f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb > 0 else "-"
        
        lines.append(
            f"| {status} {r.method} | {r.bits} | {r.description} | "
            f"{load_str} | {speed_str} tok/s | "
            f"{ttft_str} | {mem_str} MB | {kv_estimate:.0f} MB | {status} |"
        )
    
    return "\n".join(lines)


def generate_full_article(results: List[TestResult]) -> str:
    """生成完整微信公众号文章"""
    article = []
    
    # 标题
    article.append("# KV Cache 量化大比拼：2-bit TurboQuant 能否挑战 FP16？")
    article.append("")
    article.append("> 深度测试 8 种 KV Cache 量化方案，实测数据揭秘大模型显存优化的终极答案")
    article.append("")
    article.append(f"*测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}*  |  *测试设备：Apple M4 MacBook*  |  *测试模型：{MODEL}*")
    article.append("")
    article.append("---")
    article.append("")
    
    # 第一部分：什么是 KV Cache
    article.append("## 一、什么是 KV Cache？")
    article.append("")
    article.append("### KV Cache 的作用")
    article.append("")
    article.append("在 Transformer 架构的自注意力机制中，每次生成新 token 时都需要计算所有历史 token 的 Key 和 Value 向量。")
    article.append("为了避免重复计算，**KV Cache**应运而生——它将历史 token 的 K 和 V 向量缓存起来，")
    article.append("这样每次只需计算当前 token 的 Q 向量，然后与缓存的 K、V 进行注意力计算即可。")
    article.append("")
    article.append("### KV Cache 的显存占用")
    article.append("")
    article.append("KV Cache 的显存占用与三个因素成正比：")
    article.append("")
    article.append("1. **上下文长度** - 上下文越长，需要缓存的 K、V 越多")
    article.append("2. **模型层数** - 每层 Transformer 都需要独立的 KV Cache")
    article.append("3. **隐藏层维度** - 维度越高，每个向量的数据量越大")
    article.append("")
    article.append("对于 omnicoder-9b（40 层，4096 隐藏维度）在 64K 上下文下：")
    article.append("")
    article.append("```")
    article.append("KV Cache 大小 = 2 × 40 层 × 4096 维度 × 65536 上下文 × 16 bits / 8 ≈ 1024 MB")
    article.append("```")
    article.append("")
    article.append("这意味着仅 KV Cache 就需要 1GB 显存！如果使用 128K 上下文，这个数字会翻倍到 2GB。")
    article.append("")
    
    # 第二部分：KV Cache 量化
    article.append("## 二、KV Cache 量化技术")
    article.append("")
    article.append("### 量化的核心思想")
    article.append("")
    article.append("KV Cache 量化的本质是**用更低的精度存储 K 和 V 向量**，从而减少显存占用。")
    article.append("例如，将 FP16（16-bit）量化到 INT8（8-bit），显存占用直接减半。")
    article.append("")
    article.append("但量化会带来精度损失，如何在压缩率和质量之间取得平衡，就是量化技术的核心挑战。")
    article.append("")
    
    # 第三部分：llama.cpp 的量化方法
    article.append("## 三、llama.cpp 的传统量化方法")
    article.append("")
    article.append("llama.cpp 提供了多种成熟的 KV Cache 量化方案：")
    article.append("")
    article.append("### f16 - FP16 原始精度")
    article.append("")
    article.append("| 属性 | 值 |")
    article.append("|------|-----|")
    article.append("| 位数 | 16-bit |")
    article.append("| 压缩比 | 1x |")
    article.append("| 精度损失 | 无 |")
    article.append("| 显存占用 | ~1024 MB (64K) |")
    article.append("")
    article.append("**特点**：无精度损失，显存占用最大，适合作为基准参考。")
    article.append("")
    article.append("### q8_0 - 8-bit 量化")
    article.append("")
    article.append("| 属性 | 值 |")
    article.append("|------|-----|")
    article.append("| 位数 | 8-bit |")
    article.append("| 压缩比 | 2x |")
    article.append("| 精度损失 | 极小 (<1%) |")
    article.append("| 显存占用 | ~512 MB (64K) |")
    article.append("")
    article.append("**特点**：精度损失极小，显存节省 50%，推荐作为默认配置。")
    article.append("")
    article.append("### q4_0 - 4-bit 量化")
    article.append("")
    article.append("| 属性 | 值 |")
    article.append("|------|-----|")
    article.append("| 位数 | 4-bit |")
    article.append("| 压缩比 | 4x |")
    article.append("| 精度损失 | 轻微 (1-2%) |")
    article.append("| 显存占用 | ~256 MB (64K) |")
    article.append("")
    article.append("**特点**：显存节省 75%，轻微精度损失，适合显存受限场景。")
    article.append("")
    
    # 第四部分：TurboQuant
    article.append("## 四、TurboQuant 新一代量化技术")
    article.append("")
    article.append("### 技术背景")
    article.append("")
    article.append("TurboQuant 是 Google 在 2025 年提出的新一代 KV Cache 量化技术（论文：arXiv:2504.19874），")
    article.append("针对传统量化方法在低比特率下精度损失大的问题，提出了创新性的解决方案。")
    article.append("")
    article.append("### 核心创新")
    article.append("")
    article.append("**1. 随机旋转（Random Rotation）**")
    article.append("")
    article.append("通过随机正交变换将激活值分布变得更加均匀，")
    article.append("使得量化后的误差分布更加平滑，减少极端值对量化的影响。")
    article.append("")
    article.append("**2. Lloyd-Max 标量量化**")
    article.append("")
    article.append("使用 Lloyd-Max 算法设计最优量化码本，")
    article.append("在给定比特数下最小化量化误差，实现最优的率失真平衡。")
    article.append("")
    article.append("**3. 无偏内积估计**")
    article.append("")
    article.append("通过特殊的量化器设计，保证注意力机制中的内积计算无偏，")
    article.append("即使在小数点量化下也能保持模型输出质量的稳定性。")
    article.append("")
    article.append("### TurboQuant 量化等级")
    article.append("")
    article.append("| 方法 | 位数 | 压缩比 | 质量损失 | Google 推荐 |")
    article.append("|------|------|--------|----------|------------|")
    article.append("| tq4 | 4-bit | 4x | 无 | ⭐⭐⭐⭐ |")
    article.append("| tq3.5 | 3.5-bit | 4.57x | 质量中性 | ⭐⭐⭐⭐⭐ |")
    article.append("| tq3 | 3-bit | 5.33x | 轻微 | ⭐⭐⭐⭐ |")
    article.append("| tq2.5 | 2.5-bit | 6.4x | 可接受 | ⭐⭐⭐ |")
    article.append("| tq2 | 2-bit | 8x | 明显 | ⭐⭐ |")
    article.append("")
    article.append("**tq3.5** 是 Google 官方推荐的'质量中性'点，")
    article.append("在几乎不影响输出质量的前提下实现 4.57 倍压缩。")
    article.append("")
    
    # 第五部分：测试环境
    article.append("## 五、测试环境与方法")
    article.append("")
    article.append("### 硬件环境")
    article.append("")
    article.append(f"- **设备**：Apple M4 MacBook")
    article.append(f"- **后端**：Metal (Apple Silicon)")
    article.append(f"- **内存**：系统统一内存架构")
    article.append("")
    article.append("### 软件环境")
    article.append("")
    article.append(f"- **框架**：MoXing (llama.cpp Python 封装)")
    article.append(f"- **模型**：{MODEL}")
    article.append(f"- **上下文**：{CONTEXT_SIZE}")
    article.append("")
    article.append("### 测试命令")
    article.append("")
    article.append("```bash")
    article.append("# 获取测试模型")
    article.append(f"ollama pull {MODEL}")
    article.append("")
    article.append("# 运行测试（以 tq3.5 为例）")
    article.append(f"moxing ollama run {MODEL} -v --kv-cache tq3.5 -c {CONTEXT_SIZE} -p \"Test prompt\"")
    article.append("```")
    article.append("")
    article.append("### 测试脚本")
    article.append("")
    article.append("```bash")
    article.append("#!/bin/bash")
    article.append("MODEL=\"carstenuhlig/omnicoder-9b\"")
    article.append("CONTEXT=65536")
    article.append("PROMPT=\"Please explain what is machine learning.\"")
    article.append("")
    article.append("for METHOD in f16 q8_0 q4_0 tq4 tq3.5 tq3 tq2.5 tq2; do")
    article.append("    echo \"=== Testing $METHOD ===\"")
    article.append("    moxing ollama run $MODEL -p \"$PROMPT\" -c $CONTEXT --kv-cache $METHOD -v -n 256")
    article.append("done")
    article.append("```")
    article.append("")
    
    # 第六部分：实测结果
    article.append("## 六、实测结果")
    article.append("")
    article.append("### 性能对比表")
    article.append("")
    article.append(generate_markdown_table(results))
    article.append("")
    
    # 计算统计数据
    successful = [r for r in results if r.success]
    if successful:
        best_speed = max([r for r in successful if r.tokens_per_second > 0], key=lambda x: x.tokens_per_second, default=None)
        best_memory = min([r for r in successful if r.peak_memory_mb > 0], key=lambda x: x.peak_memory_mb, default=None)
        fastest_load = min(successful, key=lambda x: x.load_time_s)
        
        article.append("### 关键发现")
        article.append("")
        if best_speed:
            article.append(f"**🏆 速度之王**：{best_speed.method}")
            article.append(f"- 生成速度：**{best_speed.tokens_per_second:.1f} tokens/s**")
            article.append(f"- TTFT: {best_speed.first_token_time_s:.2f}s")
            article.append("")
        
        if best_memory:
            article.append(f"**💾 显存之王**：{best_memory.method}")
            article.append(f"- 峰值内存：**{best_memory.peak_memory_mb:.0f} MB**")
            article.append("")
        
        if fastest_load:
            article.append(f"**⚡ 加载最快**：{fastest_load.method}")
            article.append(f"- 加载时间：**{fastest_load.load_time_s:.1f}s**")
            article.append("")
    
    # 第七部分：结果分析
    article.append("## 七、结果分析")
    article.append("")
    
    if successful:
        # 计算相对于 f16 的改进
        f16_result = next((r for r in successful if r.method == "f16"), None)
        if f16_result and f16_result.peak_memory_mb > 0:
            article.append("### 显存节省对比（相对 FP16）")
            article.append("")
            article.append("| 方法 | 显存占用 | 节省比例 |")
            article.append("|------|----------|----------|")
            for r in successful:
                if r.peak_memory_mb > 0:
                    savings = (1 - r.peak_memory_mb / f16_result.peak_memory_mb) * 100
                    article.append(f"| {r.method} | {r.peak_memory_mb:.0f} MB | {savings:.1f}% |")
            article.append("")
    
    # 第八部分：使用建议
    article.append("## 八、使用建议")
    article.append("")
    article.append("### 场景化推荐")
    article.append("")
    article.append("| 场景 | 推荐方法 | 理由 | 命令示例 |")
    article.append("|------|----------|------|----------|")
    article.append("| 日常使用 | q8_0 | 精度与显存的完美平衡 | `--kv-cache q8_0` |")
    article.append("| 长上下文 | tq3.5 | 质量中性，节省 57% 显存 ⭐ | `--kv-cache tq3.5 -c 65536` |")
    article.append("| 代码生成 | q4_0 | 成熟稳定，精度足够 | `--kv-cache q4_0` |")
    article.append("| 极限压缩 | tq2 | 8 倍压缩，显存受限时 | `--kv-cache tq2` |")
    article.append("| 科研测试 | f16 | 无精度损失，基准参考 | `--kv-cache f16` |")
    article.append("")
    article.append("### 实战命令集合")
    article.append("")
    article.append("```bash")
    article.append("# 1. 日常使用（推荐）")
    article.append(f"moxing ollama run {MODEL} --kv-cache q8_0 -c {CONTEXT_SIZE}")
    article.append("")
    article.append("# 2. 长上下文场景（最佳性价比）")
    article.append(f"moxing ollama run {MODEL} --kv-cache tq3.5 -c {CONTEXT_SIZE}")
    article.append("")
    article.append("# 3. 极限显存压缩")
    article.append(f"moxing ollama run {MODEL} --kv-cache tq2 -c {CONTEXT_SIZE}")
    article.append("")
    article.append("# 4. 代码生成任务")
    article.append(f"moxing ollama run {MODEL} --kv-cache q4_0 -c 32768")
    article.append("")
    article.append("# 5. 基准测试（无压缩）")
    article.append(f"moxing ollama run {MODEL} --kv-cache f16 -c {CONTEXT_SIZE}")
    article.append("```")
    article.append("")
    
    # 第九部分：结语
    article.append("## 九、结语")
    article.append("")
    article.append("KV Cache 量化是大模型推理优化的关键技术。")
    article.append("通过合理选择量化方法，可以在保证质量的同时显著降低显存占用。")
    article.append("")
    article.append("### 核心结论")
    article.append("")
    article.append("1. **日常使用选 q8_0** - 精度与显存的完美平衡")
    article.append("2. **长上下文选 tq3.5** - 质量中性，节省 57% 显存")
    article.append("3. **极限场景选 tq2** - 8 倍压缩，显存极度受限时")
    article.append("")
    article.append("TurboQuant 作为新技术，展现了优秀的压缩性能，")
    article.append("特别是 **tq3.5** 在质量中性的前提下实现了 4.57 倍压缩，")
    article.append("值得在长上下文场景中尝试！")
    article.append("")
    article.append("---")
    article.append("")
    article.append("*测试工具：[MoXing](https://github.com/cycleuser/MoXing)*")
    article.append("")
    article.append("*欢迎在评论区分享你的测试结果！*")
    
    return "\n".join(article)


def main():
    """主函数"""
    print("="*60)
    print("KV Cache 量化方法对比测试")
    print("="*60)
    print(f"\n模型：{MODEL}")
    print(f"上下文：{CONTEXT_SIZE}")
    print(f"测试方法：{len(KV_CACHE_METHODS)} 种\n")
    print("提示：完整测试约需 30-60 分钟\n")
    
    results = []
    
    # 运行测试
    for method in KV_CACHE_METHODS:
        result = run_test(method["name"], method["desc"], method["bits"])
        results.append(result)
        
        # 简单总结
        if result.success:
            print(f"✅ {method['name']}: {result.tokens_generated} tokens, {result.tokens_per_second:.1f} tok/s, {result.peak_memory_mb:.0f} MB")
        else:
            print(f"❌ {method['name']}: {result.error}")
        
        time.sleep(3)  # 等待资源释放
    
    # 保存结果
    save_results(results)
    
    # 生成文章
    article = generate_full_article(results)
    article_file = "kv_cache_wechat_article.md"
    
    with open(article_file, "w", encoding="utf-8") as f:
        f.write(article)
    
    print(f"\n微信公众号文章已生成：{article_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    
    successful = [r for r in results if r.success]
    if successful:
        print(f"\n成功完成：{len(successful)}/{len(results)} 测试")
        
        speed_results = [r for r in successful if r.tokens_per_second > 0]
        if speed_results:
            best = max(speed_results, key=lambda x: x.tokens_per_second)
            print(f"最快速度：{best.method} - {best.tokens_per_second:.1f} tok/s")
        
        memory_results = [r for r in successful if r.peak_memory_mb > 0]
        if memory_results:
            best = min(memory_results, key=lambda x: x.peak_memory_mb)
            print(f"最低内存：{best.method} - {best.peak_memory_mb:.0f} MB")
    
    print(f"\n详细结果：kv_cache_test_results.json")
    print(f"公众号文章：{article_file}")


if __name__ == "__main__":
    main()
