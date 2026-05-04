#!/usr/bin/env python3
"""
Comprehensive AI Model Benchmark Script

This script benchmarks models using tasks inspired by SOTA benchmarks:
- HumanEval: Code generation
- GSM8K: Mathematical reasoning
- BBH: Complex reasoning
- MMLU: Knowledge QA
- IFEval: Instruction following

References:
- HumanEval: Chen et al., 2021. "Evaluating Large Language Models Trained on Code"
- GSM8K: Cobbe et al., 2021. "Training Verifiers to Solve Math Word Problems"
- BBH: Suzgun et al., 2022. "Challenging BIG-Bench Tasks"
- MMLU: Hendrycks et al., 2021. "Measuring Massive Multitask Language Understanding"
- IFEval: Zhou et al., 2023. "Instruction-Following Evaluation"

Usage:
    python comprehensive_benchmark.py
"""

import subprocess
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Callable
from datetime import datetime
import threading
import queue

RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results"

# ============================================================================
# BENCHMARK TASKS (Based on SOTA benchmarks)
# ============================================================================

BENCHMARK_TASKS = {
    "humaneval": {
        "name": "HumanEval (Code Generation)",
        "reference": "Chen et al., 2021. 'Evaluating Large Language Models Trained on Code'",
        "tasks": [
            {
                "id": "human_eval_1",
                "prompt": "Write a Python function that checks if a string is a valid palindrome. The function should ignore case and non-alphanumeric characters. Include docstring and type hints.",
                "category": "code_generation",
                "difficulty": "easy",
                "time_limit": 60,
            },
            {
                "id": "human_eval_2", 
                "prompt": "Write a Python function to find the longest common subsequence between two strings. Use dynamic programming. Include time complexity analysis.",
                "category": "code_generation",
                "difficulty": "medium",
                "time_limit": 90,
            },
            {
                "id": "human_eval_3",
                "prompt": "Write a Python function that implements a thread-safe LRU cache with TTL (time-to-live) support. The cache should automatically expire entries after the specified TTL.",
                "category": "code_generation",
                "difficulty": "hard",
                "time_limit": 120,
            },
        ]
    },
    "gsm8k": {
        "name": "GSM8K (Math Reasoning)",
        "reference": "Cobbe et al., 2021. 'Training Verifiers to Solve Math Word Problems'",
        "tasks": [
            {
                "id": "gsm8k_1",
                "prompt": "A store sells apples for $1.50 each and oranges for $2.00 each. If John buys 3 apples and 4 oranges, and pays with a $20 bill, how much change does he receive? Show your step-by-step calculation.",
                "category": "math",
                "difficulty": "easy",
                "expected_answer": "$7.50",
                "time_limit": 60,
            },
            {
                "id": "gsm8k_2",
                "prompt": "A train travels at 60 mph for the first 2 hours, then at 80 mph for the next 3 hours. What is the average speed of the train for the entire journey? Show your work.",
                "category": "math",
                "difficulty": "medium",
                "expected_answer": "72 mph",
                "time_limit": 60,
            },
            {
                "id": "gsm8k_3",
                "prompt": "A factory produces widgets. On Monday, it produces 150 widgets. Each day after that, it produces 10% more widgets than the previous day. How many widgets does it produce in total from Monday to Friday (5 days)? Round to the nearest integer.",
                "category": "math",
                "difficulty": "hard",
                "expected_answer": "~916 widgets",
                "time_limit": 90,
            },
        ]
    },
    "bbh": {
        "name": "BBH (Complex Reasoning)",
        "reference": "Suzgun et al., 2022. 'Challenging BIG-Bench Tasks'",
        "tasks": [
            {
                "id": "bbh_1",
                "prompt": "I have a red ball, a blue ball, and a green ball. I put the red ball in box A. I put the blue ball in box B. I move the red ball from box A to box B. I put the green ball in box A. Which balls are in box A and which are in box B?",
                "category": "tracking",
                "difficulty": "easy",
                "expected_answer": "Box A: green ball. Box B: red ball, blue ball",
                "time_limit": 60,
            },
            {
                "id": "bbh_2",
                "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain your reasoning step by step.",
                "category": "logic",
                "difficulty": "medium",
                "expected_answer": "Cannot conclude (logical fallacy)",
                "time_limit": 60,
            },
            {
                "id": "bbh_3",
                "prompt": "A man is looking at a photo. Someone asks him: 'Who is in this photo?' He answers: 'Brothers and sisters I have none, but this man's father is my father's son.' Who is in the photo? Explain step by step.",
                "category": "logic",
                "difficulty": "hard",
                "expected_answer": "His son",
                "time_limit": 90,
            },
        ]
    },
    "mmlu": {
        "name": "MMLU (Knowledge QA)",
        "reference": "Hendrycks et al., 2021. 'Measuring Massive Multitask Language Understanding'",
        "tasks": [
            {
                "id": "mmlu_1",
                "prompt": "What is the time complexity of QuickSort in the average case? A) O(n) B) O(n log n) C) O(n^2) D) O(log n). Answer with the letter and explain why.",
                "category": "computer_science",
                "difficulty": "easy",
                "expected_answer": "B",
                "time_limit": 30,
            },
            {
                "id": "mmlu_2",
                "prompt": "In Python, what is the difference between 'deepcopy' and 'copy'? When would you use each? Provide code examples.",
                "category": "programming",
                "difficulty": "medium",
                "time_limit": 60,
            },
            {
                "id": "mmlu_3",
                "prompt": "Explain the concept of 'attention mechanism' in transformer models. How does self-attention work? Use simple language but include the mathematical formula.",
                "category": "machine_learning",
                "difficulty": "hard",
                "time_limit": 90,
            },
        ]
    },
    "ifeval": {
        "name": "IFEval (Instruction Following)",
        "reference": "Zhou et al., 2023. 'Instruction-Following Evaluation'",
        "tasks": [
            {
                "id": "ifeval_1",
                "prompt": "Write a short story about a robot. Your response must: 1) Start with the word 'Once', 2) Contain exactly 3 paragraphs, 3) Include the word 'dream' at least twice, 4) End with a question.",
                "category": "format_constraint",
                "difficulty": "medium",
                "time_limit": 60,
            },
            {
                "id": "ifeval_2",
                "prompt": "Summarize the benefits of Python for data science. Your answer must be in JSON format with these keys: 'language', 'benefits' (array of strings), 'popular_libraries' (array of strings).",
                "category": "format_constraint",
                "difficulty": "medium",
                "time_limit": 60,
            },
            {
                "id": "ifeval_3",
                "prompt": "List 5 programming languages. For each, provide: name, paradigm, and one sentence description. Format as a markdown table.",
                "category": "format_constraint",
                "difficulty": "easy",
                "time_limit": 60,
            },
        ]
    },
    "practical": {
        "name": "Practical Use Cases",
        "reference": "Custom tasks based on real-world scenarios",
        "tasks": [
            {
                "id": "practical_1",
                "prompt": "Review this Python code and suggest improvements for performance and readability:\n\ndef find_duplicates(lst):\n    duplicates = []\n    for i in range(len(lst)):\n        for j in range(i+1, len(lst)):\n            if lst[i] == lst[j] and lst[i] not in duplicates:\n                duplicates.append(lst[i])\n    return duplicates",
                "category": "code_review",
                "difficulty": "medium",
                "time_limit": 90,
            },
            {
                "id": "practical_2",
                "prompt": "I need to scrape product prices from an e-commerce website that uses JavaScript rendering. Design a solution architecture that includes: 1) Technology choices, 2) Rate limiting strategy, 3) Data storage, 4) Error handling. Be specific with library names and code snippets.",
                "category": "system_design",
                "difficulty": "hard",
                "time_limit": 120,
            },
            {
                "id": "practical_3",
                "prompt": "Convert this SQL query to a pandas operation and explain the transformation:\n\nSELECT department, AVG(salary) as avg_salary, COUNT(*) as emp_count\nFROM employees\nWHERE status = 'active'\nGROUP BY department\nHAVING COUNT(*) > 5\nORDER BY avg_salary DESC;",
                "category": "data_transformation",
                "difficulty": "medium",
                "time_limit": 60,
            },
        ]
    },
}


@dataclass
class TaskResult:
    task_id: str
    benchmark: str
    model: str
    backend: str
    prompt: str
    response: str
    generation_speed: float
    prompt_speed: float
    total_time: float
    time_limit: int
    difficulty: str
    category: str
    expected_answer: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def run_ollama_task(model: str, prompt: str, timeout: int = 120) -> tuple[str, float, float, float]:
    """Run a task with Ollama and return (response, gen_speed, prompt_speed, total_time)."""
    try:
        start_time = time.time()
        result = subprocess.run(
            ["ollama", "run", model, "--verbose"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout + 30
        )
        total_time = time.time() - start_time
        
        gen_speed = 0.0
        prompt_speed = 0.0
        
        for line in result.stderr.split("\n"):
            if "prompt eval rate" in line.lower():
                match = re.search(r"(\d+\.?\d*)\s*tokens/s", line)
                if match:
                    prompt_speed = float(match.group(1))
            elif "eval rate" in line.lower() and "prompt" not in line.lower():
                match = re.search(r"(\d+\.?\d*)\s*tokens/s", line)
                if match:
                    gen_speed = float(match.group(1))
        
        return result.stdout.strip(), gen_speed, prompt_speed, total_time
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 0, 0, timeout
    except Exception as e:
        return f"ERROR: {e}", 0, 0, 0


def run_moxing_task(model: str, gguf_path: Path, prompt: str, timeout: int = 120) -> tuple[str, float, float, float]:
    """Run a task with MoXing/llama.cpp and return (response, gen_speed, prompt_speed, total_time)."""
    llama_cli = Path(__file__).parent.parent / "moxing" / "bin" / "linux-x64-cuda" / "llama-cli"
    if not llama_cli.exists():
        return "ERROR: llama-cli not found", 0, 0, 0
    
    try:
        proc = subprocess.Popen(
            [
                str(llama_cli),
                "-m", str(gguf_path),
                "-p", prompt,
                "-n", "2048",
                "--no-display-prompt",
                "-c", "4096",
                "-ngl", "99"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        start_time = time.time()
        try:
            stdout, stderr = proc.communicate(timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        total_time = time.time() - start_time
        
        gen_speed = 0.0
        prompt_speed = 0.0
        
        combined = stdout + stderr
        for line in combined.split("\n"):
            if "Prompt:" in line:
                match = re.search(r"Prompt:\s*(\d+\.?\d*)\s*t/s", line)
                if match:
                    prompt_speed = float(match.group(1))
            if "Generation:" in line:
                match = re.search(r"Generation:\s*(\d+\.?\d*)\s*t/s", line)
                if match:
                    gen_speed = float(match.group(1))
        
        return stdout.strip(), gen_speed, prompt_speed, total_time
    except Exception as e:
        return f"ERROR: {e}", 0, 0, 0


def get_ollama_gguf_path(model: str) -> Optional[Path]:
    """Get the GGUF path for an Ollama model."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from moxing.ollama import OllamaClient
        client = OllamaClient()
        return client.get_model_gguf_path(model)
    except:
        return None


def run_full_benchmark():
    """Run comprehensive benchmark across all models and tasks."""
    
    models = [
        {
            "name": "carstenuhlig/omnicoder-9b",
            "backend": "both",
            "description": "Qwen2.5-based code model",
        },
        {
            "name": "qwen3:8b",
            "backend": "both",
            "description": "Qwen3 official",
        },
        {
            "name": "huihui_ai/qwen3-abliterated:8b",
            "backend": "both",
            "description": "Qwen3 abliterated",
        },
        {
            "name": "huihui_ai/gpt-oss-abliterated:20b",
            "backend": "ollama",
            "description": "GPT-OSS abliterated (larger)",
        },
        {
            "name": "huihui_ai/glm-4.7-flash-abliterated:latest",
            "backend": "ollama",
            "description": "GLM-4.7 flash abliterated",
        },
    ]
    
    all_results: List[TaskResult] = []
    
    print("=" * 80)
    print("COMPREHENSIVE AI MODEL BENCHMARK")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    for benchmark_key, benchmark_info in BENCHMARK_TASKS.items():
        print(f"\n{'='*80}")
        print(f"Benchmark: {benchmark_info['name']}")
        print(f"Reference: {benchmark_info['reference']}")
        print("=" * 80)
        
        for task in benchmark_info["tasks"]:
            print(f"\n--- Task: {task['id']} ({task['difficulty']}) ---")
            print(f"Category: {task['category']}")
            prompt = task["prompt"]
            
            for model_info in models:
                model_name = model_info["name"]
                print(f"\n  Testing: {model_name}")
                
                if model_info["backend"] in ["both", "ollama"]:
                    response, gen_speed, prompt_speed, total_time = run_ollama_task(
                        model_name, prompt, task["time_limit"]
                    )
                    result = TaskResult(
                        task_id=task["id"],
                        benchmark=benchmark_key,
                        model=model_name,
                        backend="ollama",
                        prompt=prompt,
                        response=response,
                        generation_speed=gen_speed,
                        prompt_speed=prompt_speed,
                        total_time=total_time,
                        time_limit=task["time_limit"],
                        difficulty=task["difficulty"],
                        category=task["category"],
                        expected_answer=task.get("expected_answer"),
                    )
                    all_results.append(result)
                    print(f"    Ollama: {gen_speed:.1f} t/s (prompt: {prompt_speed:.1f} t/s, time: {total_time:.1f}s)")
                
                if model_info["backend"] in ["both", "moxing"]:
                    gguf_path = get_ollama_gguf_path(model_name)
                    if gguf_path and gguf_path.exists():
                        response, gen_speed, prompt_speed, total_time = run_moxing_task(
                            model_name, gguf_path, prompt, task["time_limit"]
                        )
                        if not response.startswith("ERROR"):
                            result = TaskResult(
                                task_id=task["id"],
                                benchmark=benchmark_key,
                                model=model_name,
                                backend="moxing",
                                prompt=prompt,
                                response=response,
                                generation_speed=gen_speed,
                                prompt_speed=prompt_speed,
                                total_time=total_time,
                                time_limit=task["time_limit"],
                                difficulty=task["difficulty"],
                                category=task["category"],
                                expected_answer=task.get("expected_answer"),
                            )
                            all_results.append(result)
                            print(f"    MoXing: {gen_speed:.1f} t/s (prompt: {prompt_speed:.1f} t/s, time: {total_time:.1f}s)")
                    else:
                        print(f"    MoXing: SKIPPED (GGUF not available)")
                
                time.sleep(2)
    
    return all_results


def save_results(results: List[TaskResult]):
    """Save benchmark results."""
    RESULTS_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = RESULTS_DIR / f"comprehensive_benchmark_{timestamp}.json"
    data = {
        "benchmark_date": datetime.now().isoformat(),
        "total_tasks": len(results),
        "benchmarks": {k: {"name": v["name"], "reference": v["reference"]} for k, v in BENCHMARK_TASKS.items()},
        "results": [asdict(r) for r in results]
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {json_file}")


def generate_summary(results: List[TaskResult]) -> str:
    """Generate summary statistics."""
    summary = []
    summary.append("\n" + "=" * 80)
    summary.append("BENCHMARK SUMMARY")
    summary.append("=" * 80)
    
    models = sorted(set(r.model for r in results))
    benchmarks = sorted(set(r.benchmark for r in results))
    
    for model in models:
        model_results = [r for r in results if r.model == model]
        avg_gen = sum(r.generation_speed for r in model_results if r.generation_speed > 0) / max(1, sum(1 for r in model_results if r.generation_speed > 0))
        avg_prompt = sum(r.prompt_speed for r in model_results if r.prompt_speed > 0) / max(1, sum(1 for r in model_results if r.prompt_speed > 0))
        avg_time = sum(r.total_time for r in model_results) / len(model_results)
        
        summary.append(f"\n{model}:")
        summary.append(f"  Avg Generation Speed: {avg_gen:.1f} t/s")
        summary.append(f"  Avg Prompt Speed: {avg_prompt:.1f} t/s")
        summary.append(f"  Avg Response Time: {avg_time:.1f}s")
        summary.append(f"  Total Tasks: {len(model_results)}")
    
    return "\n".join(summary)


def main():
    print("\n" + "="*80)
    print("Starting comprehensive benchmark...")
    print("="*80)
    
    results = run_full_benchmark()
    
    if results:
        print(generate_summary(results))
        save_results(results)
    else:
        print("\nNo results collected.")
    
    return results


if __name__ == "__main__":
    main()