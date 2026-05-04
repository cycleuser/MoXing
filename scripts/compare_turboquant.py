#!/usr/bin/env python3
"""
TurboQuant 与 llama.cpp 对比测试

这个脚本展示：
1. TurboQuant 算法的理论优势
2. 使用 llama.cpp 内置量化的实际运行测试
3. 两种方案的对比

注意：TurboQuant 需要在 llama.cpp 层面实现才能直接使用。
目前我们使用 llama.cpp 的内置量化（q4_0 等）作为近似。
"""

import subprocess
import time
import psutil
import httpx
import json
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

BINARY = "/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/darwin-arm64-metal/llama-server"

LLAMA_CPP_KV_TYPES = [
    ("f32", "32-bit float", 32),
    ("f16", "16-bit float (默认)", 16),
    ("q8_0", "8-bit quantized", 8),
    ("q5_0", "5-bit quantized", 5),
    ("q4_0", "4-bit quantized", 4),
    ("q4_1", "4-bit with min", 4),
    ("iq4_nl", "4-bit importance", 4),
]

TURBOQUANT_TYPES = [
    ("tq4", "TurboQuant 4-bit", 4.0),
    ("tq3.5", "TurboQuant 3.5-bit (质量中性)", 3.5),
    ("tq3", "TurboQuant 3-bit", 3.0),
    ("tq2.5", "TurboQuant 2.5-bit (轻微损失)", 2.5),
    ("tq2", "TurboQuant 2-bit", 2.0),
]


@dataclass
class TestResult:
    kv_type: str
    bits: float
    memory_mb: float
    kv_buffer_mb: float
    tps: float
    ttft: float
    tokens: int
    success: bool
    error: Optional[str] = None


def get_model_path(model_name: str) -> str:
    from moxing.ollama import OllamaClient
    client = OllamaClient()
    is_accessible, gguf_path, _ = client.check_model_access(model_name)
    if not is_accessible:
        raise FileNotFoundError(f"Cannot access model: {model_name}")
    return str(gguf_path)


def get_mem(pid: int) -> float:
    try:
        p = psutil.Process(pid)
        mem = p.memory_info().rss / 1024 / 1024
        for c in p.children(recursive=True):
            try:
                mem += c.memory_info().rss / 1024 / 1024
            except:
                pass
        return mem
    except:
        return 0


def parse_kv_buffer(log_content: str) -> Optional[float]:
    import re
    match = re.search(r'llama_kv_cache:.*?KV buffer size =\s+([\d.]+)\s+MiB', log_content)
    if match:
        return float(match.group(1))
    match = re.search(r'llama_kv_cache: size =\s+([\d.]+)\s+MiB', log_content)
    if match:
        return float(match.group(1))
    return None


def test_llama_cpp_kv(model_path: str, kv_type: str, ctx: int, port: int) -> TestResult:
    """测试 llama.cpp 内置 KV Cache 量化"""
    log_file = f"/tmp/llama_{kv_type}_{port}.log"
    
    cmd = [
        BINARY, "-m", model_path,
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(ctx), "-ngl", "all",
        "-ctk", kv_type, "-ctv", kv_type, "-fa", "on",
    ]
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    try:
        for i in range(120):
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    break
            except:
                pass
            time.sleep(1)
        else:
            proc.terminate()
            proc.wait(timeout=5)
            return TestResult(kv_type, 0, 0, 0, 0, 0, 0, False, "模型加载超时")
        
        time.sleep(2)
        mem = get_mem(proc.pid)
        
        log_content = Path(log_file).read_text()
        kv_buffer = parse_kv_buffer(log_content) or 0
        
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        start = time.time()
        first_token = None
        tokens = 0
        
        try:
            with httpx.stream("POST", url, json={
                "model": "llama",
                "messages": [{"role": "user", "content": "请用中文简短介绍人工智能。"}],
                "max_tokens": 30,
                "stream": True,
            }, timeout=120) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "") or delta.get("reasoning_content", "")
                                if content:
                                    if first_token is None:
                                        first_token = time.time()
                                    tokens += 1
                        except:
                            pass
        except Exception as e:
            proc.terminate()
            proc.wait(timeout=5)
            return TestResult(kv_type, 0, mem, kv_buffer, 0, 0, 0, False, str(e))
        
        total = time.time() - start
        ttft = (first_token - start) if first_token else 0
        tps = tokens / (total - ttft) if (total - ttft) > 0 else 0
        
        return TestResult(kv_type, 0, mem, kv_buffer, tps, ttft, tokens, True)
        
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
        if os.path.exists(log_file):
            os.remove(log_file)


def estimate_turboquant_kv_size(kv_buffer_f16: float, bits: float) -> float:
    """估算 TurboQuant 的 KV 大小"""
    return kv_buffer_f16 * bits / 16.0


def print_comparison_report(results: List[TestResult], ctx: int):
    """打印对比报告"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    f16_result = next((r for r in results if r.kv_type == "f16"), None)
    
    console.print(Panel(
        f"[cyan]TurboQuant vs llama.cpp KV Cache 量化对比[/cyan]\n"
        f"上下文大小: {ctx:,}",
        title="对比报告"
    ))
    
    table = Table(title="llama.cpp 内置量化 (实际测试)")
    table.add_column("类型", style="cyan")
    table.add_column("Bits", style="yellow")
    table.add_column("KV (MB)", style="green")
    table.add_column("节省", style="magenta")
    table.add_column("速度", style="blue")
    table.add_column("状态", style="white")
    
    for r in sorted(results, key=lambda x: -x.bits):
        if r.success:
            savings = ""
            if f16_result and r.kv_type != "f16":
                s = (f16_result.kv_buffer_mb - r.kv_buffer_mb) / f16_result.kv_buffer_mb * 100
                savings = f"{s:+.1f}%"
            
            table.add_row(
                r.kv_type,
                f"{r.bits:.0f}",
                f"{r.kv_buffer_mb:.1f}",
                savings,
                f"{r.tps:.1f} tok/s",
                "✓"
            )
        else:
            table.add_row(r.kv_type, "-", "-", "-", "-", f"✗ {r.error[:20]}")
    
    console.print(table)
    
    if f16_result:
        console.print("\n")
        table2 = Table(title="TurboQuant 理论值 (基于论文)")
        table2.add_column("类型", style="cyan")
        table2.add_column("Bits", style="yellow")
        table2.add_column("KV (MB)", style="green")
        table2.add_column("节省", style="magenta")
        table2.add_column("质量", style="blue")
        
        for name, desc, bits in TURBOQUANT_TYPES:
            estimated_kv = estimate_turboquant_kv_size(f16_result.kv_buffer_mb, bits)
            savings = (1 - bits / 16) * 100
            
            quality = "质量中性" if bits == 3.5 else \
                      "轻微损失" if bits == 2.5 else \
                      "高质量" if bits >= 4 else "可接受"
            
            table2.add_row(
                name,
                f"{bits:.1f}",
                f"{estimated_kv:.1f}",
                f"{savings:.1f}%",
                quality
            )
        
        console.print(table2)
    
    console.print("\n")
    console.print(Panel(
        """
[bold]TurboQuant 优势（基于论文 arXiv:2504.19874）:[/bold]

• [green]3.5 bits[/green]: 质量中性，与 F16 无区别
• [green]2.5 bits[/green]: 轻微质量损失，5x+ 压缩
• [green]无偏内积估计[/green]: 专为注意力机制优化
• [green]在线量化[/green]: 无需预处理，即时应用

[bold]llama.cpp 内置量化:[/bold]

• 直接可用，无需修改
• q4_0 提供 4x 压缩
• 速度和内存平衡良好

[bold]推荐配置:[/bold]

• 质量优先: q8_0 或 q5_0
• 平衡: q4_0 (当前最佳选择)
• 极限压缩: q4_0 + 更大上下文
""",
        title="结论"
    ))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TurboQuant 对比测试")
    parser.add_argument("model", help="模型名称")
    parser.add_argument("-c", "--ctx", type=int, default=32768, help="上下文大小")
    parser.add_argument("-t", "--types", type=str, default="fast", 
                        help="测试类型: fast (q4_0,q8_0,f16), all, 或具体类型")
    args = parser.parse_args()
    
    print("=" * 80)
    print("TurboQuant vs llama.cpp KV Cache 量化对比测试")
    print("=" * 80)
    
    try:
        model_path = get_model_path(args.model)
        print(f"模型: {args.model}")
        print(f"上下文: {args.ctx:,}")
        print(f"路径: {model_path}")
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return
    
    if args.types == "fast":
        test_types = [("f16", "16-bit float", 16), 
                      ("q8_0", "8-bit", 8),
                      ("q4_0", "4-bit", 4)]
    elif args.types == "all":
        test_types = LLAMA_CPP_KV_TYPES
    else:
        selected = args.types.split(",")
        test_types = [t for t in LLAMA_CPP_KV_TYPES if t[0] in selected]
    
    print(f"\n测试类型: {[t[0] for t in test_types]}")
    print("=" * 80)
    
    results: List[TestResult] = []
    base_port = 9200
    
    for i, (kv_type, desc, bits) in enumerate(test_types):
        print(f"\n[{i+1}/{len(test_types)}] 测试 {kv_type} ({desc})...")
        
        result = test_llama_cpp_kv(model_path, kv_type, args.ctx, base_port + i)
        result.bits = bits
        results.append(result)
        
        if result.success:
            print(f"  KV: {result.kv_buffer_mb:.1f} MB | 内存: {result.memory_mb:.1f} MB | 速度: {result.tps:.1f} tok/s")
        else:
            print(f"  失败: {result.error}")
        
        time.sleep(2)
    
    print_comparison_report(results, args.ctx)


if __name__ == "__main__":
    main()