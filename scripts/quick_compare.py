#!/usr/bin/env python3
"""
快速性能对比测试工具

对比 F16 vs Q4_0 (TurboQuant) 的性能差异
"""

import subprocess
import time
import psutil
import httpx
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BINARY = "/Users/fred/Documents/GitHub/cycleuser/MoXing/moxing/bin/darwin-arm64-metal/llama-server"

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

def test_config(model_path: str, kv: str, ctx: int, port: int):
    """测试单个配置"""
    cmd = [
        BINARY, "-m", model_path,
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(ctx), "-ngl", "all",
        "-ctk", kv, "-ctv", kv, "-fa", "on",
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    # 等待模型加载完成
    print(f"  等待模型加载...", end="", flush=True)
    for i in range(180):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(f" ✓ ({i+1}s)")
                break
        except:
            pass
        if i % 5 == 0:
            print(".", end="", flush=True)
        time.sleep(1)
    else:
        print(" 超时！")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
        return {"mem": 0, "tps": 0, "ttft": 0, "tokens": 0, "content": ""}
    
    # 给模型额外时间稳定
    time.sleep(3)
    mem = get_mem(proc.pid)
    print(f"  内存: {mem:.1f} MB")
    
    print(f"  推理测试...")
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    start = time.time()
    first_token = None
    tokens = 0
    content_text = ""
    
    try:
        with httpx.stream("POST", url, json={
            "model": "llama",
            "messages": [{"role": "user", "content": "请用中文简短介绍人工智能。"}],
            "max_tokens": 50,
            "stream": True,
        }, timeout=180) as r:
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
                                content_text += content
                    except:
                        pass
    except Exception as e:
        print(f"  推理失败: {e}")
    
    total = time.time() - start
    ttft = (first_token - start) if first_token else 0
    tps = tokens / (total - ttft) if (total - ttft) > 0 else 0
    
    if tokens > 0:
        print(f"  速度: {tps:.1f} tok/s, TTFT: {ttft:.2f}s")
    
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()
    
    return {"mem": mem, "tps": tps, "ttft": ttft, "tokens": tokens, "content": content_text[:100]}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="快速性能对比测试")
    parser.add_argument("model", help="模型名称，如 gemma3:1b")
    parser.add_argument("-c", "--ctx", type=int, default=8192, help="上下文大小")
    parser.add_argument("-p", "--port", type=int, default=9000, help="起始端口")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MoXing KV Cache 快速性能对比")
    print("=" * 70)
    
    try:
        model_path = get_model_path(args.model)
        print(f"模型: {args.model}")
        print(f"上下文: {args.ctx:,}")
        print(f"路径: {model_path}")
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return
    
    print("\n测试 F16...")
    f16 = test_config(model_path, "f16", args.ctx, args.port)
    
    # 确保 F16 进程完全清理
    time.sleep(3)
    
    print("\n测试 Q4_0 (TurboQuant)...")
    q4 = test_config(model_path, "q4_0", args.ctx, args.port + 1)
    
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)
    
    print(f"\n{'指标':<15} {'F16':<15} {'Q4_0':<15} {'差异':<15}")
    print("-" * 60)
    
    mem_diff = f16["mem"] - q4["mem"]
    mem_pct = (mem_diff / f16["mem"] * 100) if f16["mem"] else 0
    print(f"{'内存 (MB)':<15} {f16['mem']:<15.1f} {q4['mem']:<15.1f} {mem_diff:+.1f} ({mem_pct:+.1f}%)")
    
    tps_diff = q4["tps"] - f16["tps"]
    tps_pct = (tps_diff / f16["tps"] * 100) if f16["tps"] else 0
    print(f"{'速度 (tok/s)':<15} {f16['tps']:<15.1f} {q4['tps']:<15.1f} {tps_diff:+.1f} ({tps_pct:+.1f}%)")
    
    ttft_diff = q4["ttft"] - f16["ttft"]
    print(f"{'TTFT (s)':<15} {f16['ttft']:<15.3f} {q4['ttft']:<15.3f} {ttft_diff:+.3f}")
    
    print("\n" + "=" * 70)
    print("结论:")
    if mem_diff > 0:
        print(f"✓ Q4_0 节省内存 {mem_diff:.0f} MB ({mem_pct:.1f}%)")
    if abs(tps_pct) < 10:
        print(f"✓ 速度差异很小 ({tps_pct:+.1f}%)")
    
    print("\n推荐命令:")
    print(f"  moxing ollama serve {args.model} --kv-cache q4_0 -c {args.ctx}")
    print("=" * 70)

if __name__ == "__main__":
    main()