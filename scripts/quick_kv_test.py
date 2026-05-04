#!/usr/bin/env python3
"""快速测试所有 KV Cache 方法"""

import subprocess
import json
import re
import time
from datetime import datetime

MODEL = "carstenuhlig/omnicoder-9b"
CONTEXT = 65536
PROMPT = "hi"
TOKENS = 50

methods = [
    {"name": "f16", "bits": 16, "desc": "FP16 原始精度"},
    {"name": "q8_0", "bits": 8, "desc": "8-bit 量化"},
    {"name": "q4_0", "bits": 4, "desc": "4-bit 量化"},
    {"name": "tq4", "bits": 4, "desc": "TurboQuant 4-bit"},
    {"name": "tq3.5", "bits": 3.5, "desc": "TurboQuant 3.5-bit"},
    {"name": "tq3", "bits": 3, "desc": "TurboQuant 3-bit"},
    {"name": "tq2.5", "bits": 2.5, "desc": "TurboQuant 2.5-bit"},
    {"name": "tq2", "bits": 2, "desc": "TurboQuant 2-bit"},
]

print("="*60)
print(f"KV Cache 量化测试 - {MODEL}")
print(f"上下文：{CONTEXT}, 生成：{TOKENS} tokens")
print("="*60)

results = []

for method in methods:
    print(f"\n[{methods.index(method)+1}/{len(methods)}] 测试 {method['name']}...")
    
    cmd = ["moxing", "ollama", "run", MODEL, "-p", PROMPT, "-c", str(CONTEXT), 
           "--kv-cache", method["name"], "-v", "-n", str(TOKENS)]
    
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        output = proc.stdout + proc.stderr
        
        # 解析结果
        result = {"method": method["name"], "bits": method["bits"], "desc": method["desc"], 
                  "success": False, "error": "", "load_time": elapsed}
        
        if "Model loaded!" in output:
            result["success"] = True
            # 解析性能数据 - 从 Summary Panel 中
            match = re.search(r'Tokens:\s*(\d+).*\|\s*Time:\s*([\d.]+)\s*s.*\|\s*Speed:\s*([\d.]+)\s*tok/s.*\|\s*TTFT:\s*([\d.]+)\s*s', output)
            if match:
                result["tokens"] = int(match.group(1))
                result["total_time"] = float(match.group(2))
                result["tok/s"] = float(match.group(3))
                result["TTFT"] = float(match.group(4))
            
            match = re.search(r'Memory:\s*RAM:\s*([\d.]+)\s*GB', output)
            if match:
                result["RAM_GB"] = float(match.group(1))
                result["RAM_MB"] = float(match.group(1)) * 1024
            
            print(f"  ✅ {result.get('tok/s', 0):.1f} tok/s, TTFT: {result.get('TTFT', 0):.2f}s, RAM: {result.get('RAM_MB', 0):.0f} MB")
        else:
            result["error"] = "Load failed"
            print(f"  ❌ {result['error']}")
        
        results.append(result)
        time.sleep(2)
        
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout")
        results.append({"method": method["name"], "success": False, "error": "Timeout"})
    except Exception as e:
        print(f"  ❌ {e}")
        results.append({"method": method["name"], "success": False, "error": str(e)})

# 保存结果
with open("kv_cache_results.json", "w") as f:
    json.dump({"time": datetime.now().isoformat(), "results": results}, f, indent=2)

# 打印表格
print("\n" + "="*60)
print("测试结果汇总表")
print("="*60)
print(f"{'方法':<10} {'位数':<6} {'状态':<6} {'速度 (tok/s)':<14} {'TTFT(s)':<10} {'内存 (MB)':<12}")
print("-"*60)

for r in results:
    status = "✅" if r.get("success") else "❌"
    speed = f"{r.get('tok/s', 0):.1f}" if r.get("tok/s") else "-"
    ttft = f"{r.get('TTFT', 0):.2f}" if r.get("TTFT") else "-"
    ram = f"{r.get('RAM_MB', 0):.0f}" if r.get("RAM_MB") else "-"
    print(f"{r['method']:<10} {r['bits']:<6} {status:<6} {speed:<14} {ttft:<10} {ram:<12}")

print("\n详细数据已保存到：kv_cache_results.json")
