#!/usr/bin/env python3
"""
多语言 Token 消耗对比测试脚本

测试不同语言在 LLM 中的 token 消耗差异
支持语言：中文、英文、日文、法文、俄文、德文、西班牙文、葡萄牙文、意大利文、韩文、文言文

测试模型：
1. gemma3:1b
2. carstenuhlig/omnicoder-9b:latest
"""

import subprocess
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


# 多语言测试任务
TEST_TASKS = {
    "en": {
        "name": "English",
        "prompt": "Please explain what machine learning is in 3 sentences.",
        "task_type": "explanation"
    },
    "zh": {
        "name": "中文",
        "prompt": "请用三句话解释什么是机器学习。",
        "task_type": "explanation"
    },
    "ja": {
        "name": "日本語",
        "prompt": "機械学習とは何か 3 つの文で説明してください。",
        "task_type": "explanation"
    },
    "fr": {
        "name": "Français",
        "prompt": "Expliquez ce qu'est l'apprentissage automatique en 3 phrases.",
        "task_type": "explanation"
    },
    "ru": {
        "name": "Русский",
        "prompt": "Объясните, что такое машинное обучение, в 3 предложениях.",
        "task_type": "explanation"
    },
    "de": {
        "name": "Deutsch",
        "prompt": "Erklären Sie maschinelles Lernen in 3 Sätzen.",
        "task_type": "explanation"
    },
    "es": {
        "name": "Español",
        "prompt": "Explique qué es el aprendizaje automático en 3 frases.",
        "task_type": "explanation"
    },
    "pt": {
        "name": "Português",
        "prompt": "Explique o que é aprendizado de máquina em 3 frases.",
        "task_type": "explanation"
    },
    "it": {
        "name": "Italiano",
        "prompt": "Spiega cos'è l'apprendimento automatico in 3 frasi.",
        "task_type": "explanation"
    },
    "ko": {
        "name": "한국어",
        "prompt": "기계 학습이 무엇인지 3 문장으로 설명해 주세요.",
        "task_type": "explanation"
    },
    "lzh": {
        "name": "文言文",
        "prompt": "请以文言释机器学习之义，限三句。",
        "task_type": "explanation"
    }
}


@dataclass
class TestResult:
    """单次测试结果"""
    language: str
    language_name: str
    prompt: str
    run_number: int
    
    # 性能指标
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_s: float = 0.0
    
    # 额外信息
    success: bool = False
    error: str = ""
    response_preview: str = ""


def run_moxing_test(model: str, prompt: str, max_tokens: int = 512) -> Dict:
    """使用 moxing 运行单次测试"""
    cmd = [
        "moxing", "ollama", "run", model,
        "-p", prompt,
        "-n", str(max_tokens),
        "-v"  # 启用详细输出
    ]
    
    try:
        start_time = time.time()
        first_token_time = None
        result = {
            "success": False,
            "output": "",
            "total_time": 0,
            "first_token_time": 0
        }
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        result["total_time"] = time.time() - start_time
        result["output"] = proc.stdout + proc.stderr
        
        # 解析输出
        if "Tokens:" in result["output"]:
            result["success"] = True
            # 解析 tokens 信息
            match = re.search(r'Tokens:\s*(\d+)\s*\|\s*Time:\s*([\d.]+)\s*s\s*\|\s*Speed:\s*([\d.]+)\s*tok/s', result["output"])
            if match:
                result["total_tokens"] = int(match.group(1))
                result["total_time"] = float(match.group(2))
                result["tokens_per_second"] = float(match.group(3))
            
            # 解析 TTFT
            match = re.search(r'TTFT:\s*([\d.]+)\s*s', result["output"])
            if match:
                result["first_token_time"] = float(match.group(1))
            
            # 解析 prompt/completion tokens
            match = re.search(r'Prompt Tokens:\s*(\d+)', result["output"])
            if match:
                result["prompt_tokens"] = int(match.group(1))
            
            match = re.search(r'Completion Tokens:\s*(\d+)', result["output"])
            if match:
                result["completion_tokens"] = int(match.group(1))
        
        return result
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout", "output": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "output": ""}


def run_language_test(model: str, language: str, prompt: str, run_number: int) -> TestResult:
    """运行单语言测试"""
    result = TestResult(
        language=language,
        language_name=TEST_TASKS.get(language, {}).get("name", language),
        prompt=prompt,
        run_number=run_number
    )
    
    test_result = run_moxing_test(model, prompt)
    
    if test_result["success"]:
        result.success = True
        result.total_tokens = test_result.get("total_tokens", 0)
        result.prompt_tokens = test_result.get("prompt_tokens", 0)
        result.completion_tokens = test_result.get("completion_tokens", 0)
        result.total_time_s = test_result.get("total_time", 0)
        result.tokens_per_second = test_result.get("tokens_per_second", 0)
        result.time_to_first_token_s = test_result.get("first_token_time", 0)
        result.response_preview = test_result.get("output", "")[:200]
    else:
        result.error = test_result.get("error", "Unknown error")
    
    return result


def run_full_test(model: str, repetitions: int = 3) -> Dict:
    """运行完整的多语言测试"""
    print(f"\n{'='*70}")
    print(f"多语言 Token 消耗测试 - {model}")
    print(f"重复次数：{repetitions}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for lang, task_info in TEST_TASKS.items():
        print(f"\n[{lang}] {task_info['name']}")
        print(f"Prompt: {task_info['prompt'][:50]}...")
        
        lang_results = []
        for i in range(repetitions):
            print(f"  运行 {i+1}/{repetitions}...", end=" ", flush=True)
            result = run_language_test(model, lang, task_info["prompt"], i+1)
            lang_results.append(result)
            
            if result.success:
                print(f"✅ {result.total_tokens} tokens, {result.tokens_per_second:.1f} tok/s")
            else:
                print(f"❌ {result.error}")
            
            time.sleep(1)  # 短暂等待，避免过载
        
        all_results.extend(lang_results)
        print()
    
    # 生成统计报告
    stats = generate_statistics(all_results)
    
    return {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "repetitions": repetitions,
        "results": [asdict(r) for r in all_results],
        "statistics": stats
    }


def generate_statistics(results: List[TestResult]) -> Dict:
    """生成统计数据"""
    stats = {}
    
    for lang in TEST_TASKS.keys():
        lang_results = [r for r in results if r.language == lang and r.success]
        
        if lang_results:
            stats[lang] = {
                "name": TEST_TASKS[lang]["name"],
                "runs": len(lang_results),
                "avg_tokens": sum(r.total_tokens for r in lang_results) / len(lang_results),
                "min_tokens": min(r.total_tokens for r in lang_results),
                "max_tokens": max(r.total_tokens for r in lang_results),
                "std_tokens": calculate_std([r.total_tokens for r in lang_results]),
                "avg_speed": sum(r.tokens_per_second for r in lang_results) / len(lang_results),
                "avg_ttft": sum(r.time_to_first_token_s for r in lang_results) / len(lang_results),
                "success_rate": len(lang_results) / sum(1 for r in results if r.language == lang) * 100
            }
    
    return stats


def calculate_std(values: List[float]) -> float:
    """计算标准差"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def save_results(data: Dict, filename: str):
    """保存测试结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{filename}")


def main():
    """主函数"""
    print("="*70)
    print("多语言 Token 消耗对比测试")
    print("="*70)
    print(f"\n测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n测试语言:")
    for lang, info in TEST_TASKS.items():
        print(f"  {lang}: {info['name']}")
    
    models = [
        "gemma3:1b",
        "carstenuhlig/omnicoder-9b:latest"
    ]
    
    all_test_data = []
    
    for model in models:
        print(f"\n\n{'#'*70}")
        print(f"# 测试模型：{model}")
        print(f"{'#'*70}")
        
        test_data = run_full_test(model, repetitions=3)
        all_test_data.append(test_data)
        
        # 保存单个模型结果
        filename = f"multilingual_test_{model.replace(':', '_').replace('/', '_')}.json"
        save_results(test_data, filename)
    
    # 保存汇总结果
    save_results({"tests": all_test_data}, "multilingual_test_summary.json")
    
    # 打印对比摘要
    print_comparison(all_test_data)


def print_comparison(all_test_data: List[Dict]):
    """打印对比摘要"""
    print("\n" + "="*70)
    print("测试结果对比摘要")
    print("="*70)
    
    for test_data in all_test_data:
        model = test_data["model"]
        stats = test_data["statistics"]
        
        print(f"\n{model}:")
        print(f"{'语言':<10} {'Avg Tokens':>12} {'Min':>8} {'Max':>8} {'Std':>8} {'Speed':>10} {'TTFT':>8}")
        print("-"*70)
        
        # 按平均 tokens 排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["avg_tokens"])
        
        for lang, s in sorted_stats:
            print(f"{s['name']:<10} {s['avg_tokens']:>12.1f} {s['min_tokens']:>8} {s['max_tokens']:>8} {s['std_tokens']:>8.1f} {s['avg_speed']:>10.1f} {s['avg_ttft']:>8.2f}s")


if __name__ == "__main__":
    main()
