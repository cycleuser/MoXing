"""Run benchmarks on all device/backend combinations.

Dynamically discovers all devices and their supported backends,
then runs benchmarks for every valid combination.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def detect_devices_and_backends():
    """Discover all devices and their supported backends.

    Returns a list of (device_id, device_name, backend, backend_display) tuples.
    """
    from moxing.device import DeviceDetector, BackendType

    detector = DeviceDetector()
    devices = detector.detect()

    combinations = []

    for device in devices:
        if device.backend == BackendType.CPU:
            combinations.append(("cpu", "CPU", "cpu", "CPU"))
            continue

        device_id = f"gpu{device.index}"
        device_name = device.name
        vendor = device.vendor.lower()

        backends_for_device = []

        if vendor == "nvidia":
            backends_for_device.append(("cuda", "CUDA"))
            backends_for_device.append(("vulkan", "Vulkan"))
        elif vendor == "amd":
            backends_for_device.append(("vulkan", "Vulkan"))
            if sys.platform == "linux":
                backends_for_device.append(("rocm", "ROCm"))
        elif vendor == "intel":
            backends_for_device.append(("vulkan", "Vulkan"))
        elif vendor == "apple":
            backends_for_device.append(("metal", "Metal"))

        for backend, backend_display in backends_for_device:
            combinations.append((device_id, device_name, backend, backend_display))

    return combinations


def get_env_for_backend(backend, device_index=0):
    """Get environment variables for a specific backend."""
    env = dict(os.environ)

    if backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    elif backend == "vulkan":
        env["GGML_VK_VISIBLE_DEVICES"] = str(device_index)
    elif backend == "rocm":
        env["HIP_VISIBLE_DEVICES"] = str(device_index)

    return env


def run_benchmark(model_path, backend, device_id, prompt="standard", n_tokens=128, n_runs=3, ctx_size=4096):
    """Run a single benchmark and return results."""
    cmd = [
        "moxing", "bench",
        str(model_path),
        "-p", prompt,
        "-n", str(n_tokens),
        "-r", str(n_runs),
        "-w",
        "-c", str(ctx_size),
        "--json",
    ]

    device_index = 0
    if device_id.startswith("gpu"):
        try:
            device_index = int(device_id[3:])
        except ValueError:
            pass

    env = get_env_for_backend(backend, device_index)

    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        env=env,
    )

    if result.returncode == 0:
        # Parse JSON from stdout (may have other output before it)
        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        json_end = stdout.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(stdout[json_start:json_end])
            return {
                "success": True,
                "tokens_per_second": data.get("tokens_per_second", 0),
                "prompt_tokens_per_second": data.get("prompt_tokens_per_second", 0),
                "total_time_sec": data.get("total_time_sec", 0),
                "peak_memory_mb": data.get("peak_memory_mb", 0),
            }
        else:
            return {
                "success": False,
                "error": f"No JSON found in output: {stdout[:200]}",
            }
    else:
        return {
            "success": False,
            "error": result.stderr[:500],
        }


def main():
    models = {
        "Q4_K_M": "omnicoder-2-9b-q4_k_m.gguf",
        "Q5_K_M": "omnicoder-2-9b-q5_k_m.gguf",
    }

    combinations = detect_devices_and_backends()

    if not combinations:
        print("No devices found!")
        return

    print(f"\nDiscovered {len(combinations)} device/backend combinations:")
    for device_id, device_name, backend, backend_display in combinations:
        print(f"  {device_id}: {device_name} ({backend_display})")

    all_results = {}

    for quant_name, model_file in models.items():
        model_path = Path(model_file)
        if not model_path.exists():
            print(f"\nModel not found: {model_file}, skipping...")
            continue

        all_results[quant_name] = {}

        for device_id, device_name, backend, backend_display in combinations:
            label = f"{device_name} + {backend_display}"
            print(f"\n{'='*60}")
            print(f"Running: {label} + {quant_name}")
            print(f"{'='*60}")

            result = run_benchmark(model_path, backend, device_id)

            if result["success"]:
                all_results[quant_name][label] = {
                    "device": device_name,
                    "backend": backend_display,
                    "device_id": device_id,
                    "tokens_per_second": result["tokens_per_second"],
                    "prompt_tokens_per_second": result["prompt_tokens_per_second"],
                    "total_time_sec": result["total_time_sec"],
                    "peak_memory_mb": result["peak_memory_mb"],
                }
                print(f"  Speed: {result['tokens_per_second']:.2f} tok/s")
                print(f"  Prompt: {result['prompt_tokens_per_second']:.2f} tok/s")
                print(f"  Time: {result['total_time_sec']:.2f}s")
            else:
                all_results[quant_name][label] = {
                    "device": device_name,
                    "backend": backend_display,
                    "device_id": device_id,
                    "success": False,
                    "error": result["error"],
                }
                print(f"  FAILED: {result['error'][:200]}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(all_results, indent=2))

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
