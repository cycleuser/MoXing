#!/usr/bin/env python3
"""
全面 KV Cache 量化性能对比测试

对比所有 llama.cpp 支持的 KV Cache 量化类型：
- f32, f16, bf16 (浮点)
- q8_0 (8 bit)
- q5_0, q5_1 (5 bit)
- q4_0, q4_1, iq4_nl (4 bit)

理论内存占用 (per token, per layer, 1024 KV dim):
- f32:  4 KB * 2 (K+V) = 8 KB
- f16:  2 KB * 2 = 4 KB
- q8_0: 1 KB * 2 = 2 KB
- q5_0: 0.625 KB * 2 = 1.25 KB
- q4_0: 0.5 KB * 2 = 1 KB
"""

import subprocess
import time
import psutil
import httpx
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

BINARY = "/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/darwin-arm64-metal/llama-server"

KV_TYPES = [
    ("f32", "32-bit float", 32),
    ("f16", "16-bit float (默认)", 16),
    ("bf16", "16-bit bfloat", 16),
    ("q8_0", "8-bit quantized", 8),
    ("q5_0", "5-bit quantized", 5),
    ("q5_1", "5-bit quantized (with min)", 5),
    ("q4_0", "4-bit quantized", 4),
    ("q4_1", "4-bit quantized (with min)", 4),
    ("iq4_nl", "4-bit importance quantized", 4),
]


@dataclass
class TestResult:
    kv_type: str
    bits: int
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
    """从日志中解析 KV buffer size"""
    import re
    match = re.search(r'llama_kv_cache:.*?KV buffer size =\s+([\d.]+)\s+MiB', log_content)
    if match:
        return float(match.group(1))
    match = re.search(r'llama_kv_cache: size =\s+([\d.]+)\s+MiB', log_content)
    if match:
        return float(match.group(1))
    return None


def test_config(model_path: str, kv_type: str, ctx: int, port: int) -> TestResult:
    """测试单个配置"""
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="全面 KV Cache 量化对比测试")
    parser.add_argument("model", help="模型名称")
    parser.add_argument("-c", "--ctx", type=int, default=32768, help="上下文大小")
    parser.add_argument("-t", "--types", type=str, default="all", help="测试类型 (all, float, quant, 或具体类型如 q4_0,q5_0)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MoXing KV Cache 全面量化对比测试")
    print("=" * 80)
    
    try:
        model_path = get_model_path(args.model)
        print(f"模型: {args.model}")
        print(f"上下文: {args.ctx:,}")
        print(f"路径: {model_path}")
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return
    
    if args.types == "all":
        test_types = KV_TYPES
    elif args.types == "float":
        test_types = [t for t in KV_TYPES if t[2] >= 16]
    elif args.types == "quant":
        test_types = [t for t in KV_TYPES if t[2] < 16]
    else:
        selected = args.types.split(",")
        test_types = [t for t in KV_TYPES if t[0] in selected]
    
    print(f"\n测试类型: {[t[0] for t in test_types]}")
    print("=" * 80)
    
    results: List[TestResult] = []
    base_port = 9100
    
    for i, (kv_type, desc, bits) in enumerate(test_types):
        print(f"\n[{i+1}/{len(test_types)}] 测试 {kv_type} ({desc})...")
        
        result = test_config(model_path, kv_type, args.ctx, base_port + i)
        result.bits = bits
        results.append(result)
        
        if result.success:
            print(f"  内存: {result.memory_mb:.1f} MB | KV: {result.kv_buffer_mb:.1f} MB | 速度: {result.tps:.1f} tok/s")
        else:
            print(f"  失败: {result.error}")
        
        time.sleep(2)
    
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    
    f16_result = next((r for r in results if r.kv_type == "f16"), None)
    
    print(f"\n{'类型':<10} {'Bits':<6} {'内存(MB)':<12} {'KV(MB)':<12} {'速度':<12} {'TTFT(s)':<10} {'状态':<8}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: -x.bits):
        if r.success:
            kv_savings = ""
            if f16_result and r.kv_type != "f16":
                savings = (f16_result.kv_buffer_mb - r.kv_buffer_mb) / f16_result.kv_buffer_mb * 100
                kv_savings = f"({savings:+.1f}%)"
            
            tps_diff = ""
            if f16_result and r.kv_type != "f16" and f16_result.tps > 0:
                diff = (r.tps - f16_result.tps) / f16_result.tps * 100
                tps_diff = f"({diff:+.1f}%)"
            
            print(f"{r.kv_type:<10} {r.bits:<6} {r.memory_mb:<12.1f} {r.kv_buffer_mb:<12.1f} {r.tps:.1f} {tps_diff:<6} {r.ttft:<10.3f} ✓")
        else:
            print(f"{r.kv_type:<10} {r.bits:<6} {'-':<12} {'-':<12} {'-':<12} {'-':<10} ✗ {r.error}")
    
    print("\n" + "=" * 80)
    print("推荐配置:")
    print("=" * 80)
    
    valid_results = [r for r in results if r.success]
    if valid_results:
        best_mem = min(valid_results, key=lambda x: x.kv_buffer_mb)
        best_speed = max(valid_results, key=lambda x: x.tps)
        
        print(f"\n最低内存: {best_mem.kv_type} ({best_mem.kv_buffer_mb:.1f} MB KV)")
        print(f"最快速度: {best_speed.kv_type} ({best_speed.tps:.1f} tok/s)")
        
        print(f"\n推荐命令:")
        print(f"  moxing ollama serve {args.model} --kv-cache {best_mem.kv_type} -c {args.ctx}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()