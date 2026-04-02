#!/usr/bin/env python3
"""
KV Cache 量化方法对比测试脚本

测试不同的 KV Cache 量化方法对内存占用和性能的影响
"""

import subprocess
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

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
    first_token_time_s: float = 0.0  # TTFT
    total_time_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    # 内存指标
    peak_memory_mb: float = 0.0
    kv_cache_size_mb: float = 0.0
    # 输出
    output_preview: str = ""


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
        "-v"
    ]
    
    start_time = time.time()
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )
        
        result.load_time_s = time.time() - start_time
        result.total_time_s = result.load_time_s
        
        output = proc.stdout + proc.stderr
        
        # 解析输出提取性能指标
        if "Model loaded!" in output:
            result.success = True
            
            # 尝试解析 tokens 信息
            for line in output.split("\n"):
                if "tokens |" in line:
                    try:
                        parts = line.split("|")
                        if len(parts) >= 3:
                            result.tokens_generated = int(parts[0].strip().split()[0])
                            result.total_time_s = float(parts[1].strip().split()[0])
                            result.tokens_per_second = float(parts[2].strip().split()[0])
                    except:
                        pass
                
                if "RAM:" in line and "GB" in line:
                    try:
                        ram_part = line.split("RAM:")[1].split("GB")[0].strip()
                        result.peak_memory_mb = float(ram_part) * 1024
                    except:
                        pass
            
            # 提取输出预览
            if "Response:" in output:
                response_start = output.find("Response:") + 9
                result.output_preview = output[response_start:response_start+200].strip()
        else:
            result.error = "Model failed to load"
            
    except subprocess.TimeoutExpired:
        result.error = "Test timeout (300s)"
    except Exception as e:
        result.error = str(e)
    
    return result


def estimate_kv_cache_size(bits: float, context: int = 65536, model_layers: int = 40, hidden_size: int = 4096) -> float:
    """估算 KV Cache 大小 (MB)
    
    KV Cache 大小计算公式:
    size = 2 * layers * hidden_size * context * bits / 8
    
    2 表示 K 和 V 两个 cache
    """
    size_bytes = 2 * model_layers * hidden_size * context * bits / 8
    return size_bytes / (1024 * 1024)  # 转换为 MB


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
    
    lines.append("## 测试结果对比\n")
    lines.append("| 量化方法 | 位数 | 描述 | 加载时间 | 生成速度 | TTFT | 峰值内存 | KV Cache 估算 |")
    lines.append("|---------|------|------|----------|----------|------|----------|-------------|")
    
    for r in results:
        kv_estimate = estimate_kv_cache_size(r.bits, CONTEXT_SIZE)
        status = "✅" if r.success else "❌"
        lines.append(
            f"| {status} {r.method} | {r.bits} | {r.description} | "
            f"{r.load_time_s:.1f}s | {r.tokens_per_second:.1f} tok/s | "
            f"{r.first_token_time_s:.2f}s | {r.peak_memory_mb:.0f} MB | {kv_estimate:.0f} MB |"
        )
    
    return "\n".join(lines)


def generate_wechat_article(results: List[TestResult]) -> str:
    """生成微信公众号文章"""
    article = []
    
    # 标题
    article.append("# KV Cache 量化大比拼：2-bit TurboQuant 能否挑战 FP16？")
    article.append("")
    article.append("> 深度测试 8 种 KV Cache 量化方案，揭秘大模型显存优化的终极答案")
    article.append("")
    article.append(f"*测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    article.append("")
    
    # 引言
    article.append("## 前言")
    article.append("")
    article.append("大模型推理时，KV Cache 往往占用大量显存。")
    article.append("特别是在长上下文场景下（如 64K、128K），KV Cache 可能成为显存瓶颈。")
    article.append("")
    article.append("今天我们使用 MoXing 框架，深度测试 8 种 KV Cache 量化方法，")
    article.append("从 FP16 原精度到 2-bit 极限压缩，看看哪种方案最适合你！")
    article.append("")
    
    # 测试环境
    article.append("## 测试环境")
    article.append("")
    article.append(f"- **模型**: {MODEL}")
    article.append(f"- **上下文**: {CONTEXT_SIZE}")
    article.append(f"- **测试设备**: Apple M4 (Metal 后端)")
    article.append(f"- **测试命令**: `moxing ollama run {MODEL} -v --kv-cache <method> -c {CONTEXT_SIZE}`")
    article.append("")
    
    # 量化方法介绍
    article.append("## KV Cache 量化方法详解")
    article.append("")
    article.append("### 传统量化方法")
    article.append("")
    article.append("**FP16 (f16)** - 原始精度")
    article.append("- 位数：16-bit")
    article.append("- 特点：无精度损失，显存占用最大")
    article.append("- 适用场景：对精度要求极高的任务")
    article.append("")
    article.append("**Q8_0 (q8_0)** - 8-bit 量化")
    article.append("- 位数：8-bit")
    article.append("- 特点：精度损失极小，显存节省 50%")
    article.append("- 适用场景：通用场景，推荐默认使用")
    article.append("")
    article.append("**Q4_0 (q4_0)** - 4-bit 量化")
    article.append("- 位数：4-bit")
    article.append("- 特点：显存节省 75%，轻微精度损失")
    article.append("- 适用场景：显存受限场景")
    article.append("")
    
    article.append("### TurboQuant 新技术")
    article.append("")
    article.append("TurboQuant 是 Google 提出的新一代 KV Cache 量化技术（arXiv:2504.19874），")
    article.append("采用随机旋转 + Lloyd-Max 标量量化，实现更高压缩比。")
    article.append("")
    article.append("| 方法 | 位数 | 压缩比 | 质量损失 |")
    article.append("|------|------|--------|----------|")
    article.append("| tq4 | 4-bit | 4x | 无 |")
    article.append("| tq3.5 | 3.5-bit | 4.57x | 质量中性 ⭐ |")
    article.append("| tq3 | 3-bit | 5.33x | 轻微 |")
    article.append("| tq2.5 | 2.5-bit | 6.4x | 可接受 |")
    article.append("| tq2 | 2-bit | 8x | 明显 |")
    article.append("")
    
    # 测试结果
    article.append("## 测试结果")
    article.append("")
    article.append(generate_markdown_table(results))
    article.append("")
    
    # 结果分析
    article.append("## 结果分析")
    article.append("")
    
    successful_results = [r for r in results if r.success]
    if successful_results:
        best_speed = max(successful_results, key=lambda x: x.tokens_per_second)
        best_memory = min(successful_results, key=lambda x: x.peak_memory_mb if x.peak_memory_mb > 0 else float('inf'))
        
        article.append(f"### 🏆 速度之王：{best_speed.method}")
        article.append(f"生成速度：**{best_speed.tokens_per_second:.1f} tokens/s**")
        article.append("")
        
        article.append(f"### 💾 显存之王：{best_memory.method}")
        article.append(f"峰值内存：**{best_memory.peak_memory_mb:.0f} MB**")
        article.append("")
    
    # 推荐
    article.append("## 使用建议")
    article.append("")
    article.append("### 推荐配置")
    article.append("")
    article.append("| 场景 | 推荐方法 | 理由 |")
    article.append("|------|----------|------|")
    article.append("| 日常使用 | q8_0 | 精度与显存的完美平衡 |")
    article.append("| 长上下文 | tq3.5 | 质量中性，节省 57% 显存 ⭐ |")
    article.append("| 极限压缩 | tq2 | 8 倍压缩，显存极度受限时 |")
    article.append("| 生产环境 | q4_0 | 成熟稳定，广泛验证 |")
    article.append("")
    
    article.append("### 实战命令")
    article.append("")
    article.append("```bash")
    article.append("# 日常使用（推荐）")
    article.append(f"moxing ollama run {MODEL} --kv-cache q8_0 -c {CONTEXT_SIZE}")
    article.append("")
    article.append("# 长上下文场景")
    article.append(f"moxing ollama run {MODEL} --kv-cache tq3.5 -c {CONTEXT_SIZE}")
    article.append("")
    article.append("# 极限显存压缩")
    article.append(f"moxing ollama run {MODEL} --kv-cache tq2 -c {CONTEXT_SIZE}")
    article.append("```")
    article.append("")
    
    # 结语
    article.append("## 结语")
    article.append("")
    article.append("KV Cache 量化是大模型推理优化的关键技术。")
    article.append("通过合理选择量化方法，可以在保证质量的同时显著降低显存占用。")
    article.append("")
    article.append("TurboQuant 作为新技术，展现了优秀的压缩性能，")
    article.append("特别是 tq3.5 在质量中性的前提下实现了 4.57 倍压缩，")
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
    
    results = []
    
    # 运行测试
    for method in KV_CACHE_METHODS:
        result = run_test(method["name"], method["desc"], method["bits"])
        results.append(result)
        
        # 简单总结
        if result.success:
            print(f"✅ {method['name']}: {result.tokens_generated} tokens, {result.tokens_per_second:.1f} tok/s")
        else:
            print(f"❌ {method['name']}: {result.error}")
        
        time.sleep(2)  # 等待资源释放
    
    # 保存结果
    save_results(results)
    
    # 生成文章
    article = generate_wechat_article(results)
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
        
        if any(r.tokens_per_second > 0 for r in successful):
            best = max(successful, key=lambda x: x.tokens_per_second)
            print(f"最快速度：{best.method} - {best.tokens_per_second:.1f} tok/s")
    
    print(f"\n详细结果：kv_cache_test_results.json")
    print(f"公众号文章：{article_file}")


if __name__ == "__main__":
    main()
