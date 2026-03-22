#!/usr/bin/env python3
"""
Benchmark script: MoXing vs Ollama performance comparison.

This script compares the inference speed of:
1. Ollama (official models)
2. Ollama (community models) 
3. MoXing/llama.cpp (using Ollama GGUF files)

Usage:
    python benchmark_moxing_vs_ollama.py

Requirements:
    - Ollama installed and running
    - MoXing installed with llama.cpp binaries
    - Models already pulled via ollama pull
"""

import subprocess
import time
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results"

@dataclass
class BenchmarkResult:
    model: str
    backend: str  # "ollama" or "moxing"
    prompt_tokens_per_sec: float
    generation_tokens_per_sec: float
    total_time_sec: float
    test_prompt: str
    timestamp: str

def run_ollama_benchmark(model: str, prompt: str, timeout: int = 120) -> Optional[BenchmarkResult]:
    """Run benchmark using Ollama."""
    print(f"\n  Running Ollama benchmark: {model}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            ["ollama", "run", model, "--verbose"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        total_time = time.time() - start_time
        
        stderr = result.stderr
        
        prompt_rate = 0.0
        gen_rate = 0.0
        
        for line in stderr.split("\n"):
            if "prompt eval rate" in line.lower():
                match = re.search(r"(\d+\.?\d*)\s*tokens/s", line)
                if match:
                    prompt_rate = float(match.group(1))
            elif "eval rate" in line.lower() and "prompt" not in line.lower():
                match = re.search(r"(\d+\.?\d*)\s*tokens/s", line)
                if match:
                    gen_rate = float(match.group(1))
        
        return BenchmarkResult(
            model=model,
            backend="ollama",
            prompt_tokens_per_sec=prompt_rate,
            generation_tokens_per_sec=gen_rate,
            total_time_sec=total_time,
            test_prompt=prompt,
            timestamp=datetime.now().isoformat()
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT for {model}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def run_moxing_benchmark(model: str, gguf_path: Path, prompt: str, n_tokens: int = 200, timeout: int = 120) -> Optional[BenchmarkResult]:
    """Run benchmark using MoXing/llama.cpp."""
    print(f"\n  Running MoXing benchmark: {model}")
    
    llama_cli = Path(__file__).parent.parent / "moxing" / "bin" / "linux-x64-cuda" / "llama-cli"
    if not llama_cli.exists():
        print(f"    ERROR: llama-cli not found at {llama_cli}")
        return None
    
    try:
        proc = subprocess.Popen(
            [
                str(llama_cli),
                "-m", str(gguf_path),
                "-p", prompt,
                "-n", str(n_tokens),
                "--no-display-prompt",
                "-c", "2048",
                "-ngl", "99"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        start_time = time.time()
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        
        total_time = time.time() - start_time
        
        prompt_rate = 0.0
        gen_rate = 0.0
        
        combined = stdout + stderr
        for line in combined.split("\n"):
            if "Prompt:" in line:
                match = re.search(r"Prompt:\s*(\d+\.?\d*)\s*t/s", line)
                if match:
                    prompt_rate = float(match.group(1))
            if "Generation:" in line:
                match = re.search(r"Generation:\s*(\d+\.?\d*)\s*t/s", line)
                if match:
                    gen_rate = float(match.group(1))
        
        if gen_rate == 0:
            return None
            
        return BenchmarkResult(
            model=model,
            backend="moxing",
            prompt_tokens_per_sec=prompt_rate,
            generation_tokens_per_sec=gen_rate,
            total_time_sec=total_time,
            test_prompt=prompt,
            timestamp=datetime.now().isoformat()
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT for {model}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def get_ollama_gguf_path(model: str) -> Optional[Path]:
    """Get the GGUF path for an Ollama model."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from moxing.ollama import OllamaClient
        client = OllamaClient()
        return client.get_model_gguf_path(model)
    except ImportError as e:
        print(f"    ERROR: moxing not available ({e})")
        return None
    except Exception as e:
        print(f"    ERROR getting GGUF path: {e}")
        return None


def check_moxing_compatibility(gguf_path: Path) -> bool:
    """Check if the GGUF is compatible with MoXing/llama.cpp."""
    llama_cli = Path(__file__).parent.parent / "moxing" / "bin" / "linux-x64-cuda" / "llama-cli"
    if not llama_cli.exists():
        print(f"    llama-cli not found at {llama_cli}")
        return False
    
    try:
        proc = subprocess.Popen(
            [str(llama_cli), "-m", str(gguf_path), "-n", "5", "-p", "hi", "-c", "128", "--no-display-prompt", "-ngl", "99"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(timeout=30)
        combined_output = stdout + stderr
        incompatible_patterns = [
            "error loading model",
            "wrong number of tensors", 
            "wrong shape",
            "missing tensor",
            "failed to load model",
            "done_getting_tensors",
        ]
        is_incompatible = any(p.lower() in combined_output.lower() for p in incompatible_patterns)
        
        if "Generation:" in combined_output or "t/s" in combined_output:
            return True
        return not is_incompatible
    except Exception as e:
        print(f"    Compatibility check error: {e}")
        return True
    
    try:
        result = subprocess.run(
            [str(llama_cli), "-m", str(gguf_path), "-n", "1", "-p", "test", "-c", "128", "--no-display-prompt"],
            capture_output=True,
            text=True,
            timeout=30,
            input="exit\n"
        )
        combined_output = (result.stdout + result.stderr).lower()
        incompatible_patterns = [
            "error loading model",
            "wrong number of tensors", 
            "wrong shape",
            "missing tensor",
            "failed to load model",
            "done_getting_tensors",
        ]
        is_incompatible = any(p in combined_output for p in incompatible_patterns)
        return not is_incompatible
    except Exception as e:
        print(f"    Compatibility check error: {e}")
        return False

def run_full_benchmark(test_prompt: str = "is it possible to mixuse rust and python") -> List[BenchmarkResult]:
    """Run complete benchmark suite."""
    
    models_to_test = [
        ("qwen3:8b", "official Ollama model"),
        ("huihui_ai/qwen3-abliterated:8b", "community model"),
        ("carstenuhlig/omnicoder-9b", "community model (Qwen2.5-based)"),
    ]
    
    results = []
    
    print("=" * 60)
    print("MoXing vs Ollama Performance Benchmark")
    print("=" * 60)
    print(f"Test prompt: '{test_prompt}'")
    print(f"GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)")
    
    for model, description in models_to_test:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"Type: {description}")
        print("-" * 60)
        
        ollama_result = run_ollama_benchmark(model, test_prompt)
        if ollama_result:
            results.append(ollama_result)
            print(f"    Ollama: {ollama_result.generation_tokens_per_sec:.1f} t/s (prompt: {ollama_result.prompt_tokens_per_sec:.1f} t/s)")
        
        gguf_path = get_ollama_gguf_path(model)
        if gguf_path and gguf_path.exists():
            if check_moxing_compatibility(gguf_path):
                moxing_result = run_moxing_benchmark(model, gguf_path, test_prompt)
                if moxing_result:
                    results.append(moxing_result)
                    print(f"    MoXing: {moxing_result.generation_tokens_per_sec:.1f} t/s (prompt: {moxing_result.prompt_tokens_per_sec:.1f} t/s)")
            else:
                print(f"    MoXing: SKIPPED (GGUF incompatible)")
        
        time.sleep(2)
    
    return results

def save_results(results: List[BenchmarkResult]):
    """Save benchmark results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    
    data = {
        "benchmark_date": datetime.now().isoformat(),
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)",
        "results": [
            {
                "model": r.model,
                "backend": r.backend,
                "prompt_tokens_per_sec": r.prompt_tokens_per_sec,
                "generation_tokens_per_sec": r.generation_tokens_per_sec,
                "total_time_sec": r.total_time_sec,
                "test_prompt": r.test_prompt,
                "timestamp": r.timestamp
            }
            for r in results
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

def print_summary_table(results: List[BenchmarkResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<40} {'Backend':<10} {'Prompt t/s':>12} {'Gen t/s':>12}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: (x.model, x.backend)):
        print(f"{r.model:<40} {r.backend:<10} {r.prompt_tokens_per_sec:>12.1f} {r.generation_tokens_per_sec:>12.1f}")
    
    print("-" * 80)
    
    models = set(r.model for r in results)
    print("\nComparison (Generation Speed):")
    for model in sorted(models):
        model_results = [r for r in results if r.model == model]
        if len(model_results) == 2:
            ollama_r = [r for r in model_results if r.backend == "ollama"][0]
            moxing_r = [r for r in model_results if r.backend == "moxing"][0]
            diff = moxing_r.generation_tokens_per_sec - ollama_r.generation_tokens_per_sec
            pct = (diff / ollama_r.generation_tokens_per_sec) * 100 if ollama_r.generation_tokens_per_sec > 0 else 0
            faster = "MoXing" if diff > 0 else "Ollama"
            print(f"  {model}: {faster} faster by {abs(pct):.1f}% ({abs(diff):.1f} t/s)")

def main():
    print("\n" + "="*60)
    print("Starting benchmark...")
    print("="*60)
    
    results = run_full_benchmark()
    
    if results:
        print_summary_table(results)
        save_results(results)
    else:
        print("\nNo results collected. Make sure Ollama is running and models are pulled.")
    
    return results

if __name__ == "__main__":
    main()