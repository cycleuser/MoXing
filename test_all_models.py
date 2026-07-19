"""Batch-test all models in C:\\Users\\frede\\models on available GPUs.

For each model, runs a short generation on every (device, backend) combination
that is actually usable on this host, and reports which produce sane output.

Usage:
    python test_all_models.py                 # test all models
    python test_all_models.py --quick         # skip models > 20GB
    python test_all_models.py --model gemma-4-12b-it-Q4_K_M
    python test_all_models.py --device gpu0    # only test on gpu0
"""

import argparse
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

MODELS_DIR = Path(r"C:\Users\frede\models")
RESULTS_FILE = Path(__file__).parent / "test_all_models_results.json"

for _k in ["SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"]:
    os.environ.pop(_k, None)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def find_model_files() -> List[Tuple[str, Path]]:
    """Return list of (name, gguf_path) for all GGUF models in MODELS_DIR."""
    out: List[Tuple[str, Path]] = []
    if not MODELS_DIR.exists():
        return out
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            is_gguf = f.suffix.lower() == ".gguf"
            is_aux = "mmproj" in f.name.lower() or "mtp" in f.name.lower()
            if is_gguf and not is_aux:
                out.append((sub.name, f))
                break
    return out


def detect_device_backend_combos() -> List[Tuple[str, str, str]]:
    """Return list of (device_id, device_name, backend) to test."""
    from moxing.device import BackendType, DeviceDetector

    detector = DeviceDetector()
    devices = detector.detect()
    combos: List[Tuple[str, str, str]] = []

    for dv in devices:
        if dv.backend == BackendType.CPU:
            continue
        if dv.memory_mb < 2048:
            continue
        device_id = f"gpu{dv.index}"
        if dv.vendor == "nvidia":
            combos.append((device_id, dv.name, "cuda"))
            combos.append((device_id, dv.name, "vulkan"))
        elif dv.vendor == "amd":
            if sys.platform != "win32":
                combos.append((device_id, dv.name, "rocm"))
            combos.append((device_id, dv.name, "vulkan"))
    return combos


def get_backend_binary(backend: str) -> Optional[Path]:
    from moxing.server import _find_binary

    try:
        return _find_binary(backend)
    except Exception:
        return None


def resolve_dev_arg(device: str, backend: str) -> Optional[str]:
    from moxing.server import LlamaServer

    s = LlamaServer.__new__(LlamaServer)
    s.device = device
    s.gpu_backend = backend
    s.n_gpu_layers = 999
    return s._resolve_device_arg()


def wait_for_port(port: int, timeout: int = 180) -> bool:
    for _ in range(timeout):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def run_test(model_path: Path, backend: str, dev_arg: str, binary: Path, port: int) -> dict:
    """Run a single model test and return result dict."""
    args = [
        str(binary),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "4096",
        "-ngl",
        "999",
        "--flash-attn",
        "on",
        "--kv-unified",
        "--metrics",
    ]
    if dev_arg:
        args.extend(["-dev", dev_arg])

    lines: List[str] = []
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        cwd=str(binary.parent),
    )

    def reader():
        try:
            for line in proc.stdout:
                lines.append(line.rstrip())
        except Exception:
            pass

    threading.Thread(target=reader, daemon=True).start()

    loaded = False
    for _ in range(180):
        if any("model loaded" in line for line in lines):
            loaded = True
            break
        if proc.poll() is not None:
            loaded = False
            break
        time.sleep(1)

    if not loaded:
        proc.terminate()
        return {
            "success": False,
            "error": "model failed to load",
            "log_tail": lines[-15:],
        }

    time.sleep(2)

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                    "max_tokens": 60,
                    "temperature": 0.7,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120, context=_SSL_CTX)
        data = json.loads(resp.read())
        choice = data.get("choices", [{}])[0].get("message", {})
        content = choice.get("content", "")
        reasoning = choice.get("reasoning_content", "")
        full = (content + reasoning).strip()
        garbled = "<unused" in full or full.count(full[:5]) > 10 if full else True
        return {
            "success": True,
            "content": content[:200],
            "reasoning": reasoning[:200],
            "garbled": garbled,
            "tokens": data.get("usage", {}).get("completion_tokens", 0),
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "log_tail": lines[-10:]}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip models > 20GB")
    parser.add_argument("--model", help="only test this model dir name")
    parser.add_argument("--device", help="only test this device id (e.g. gpu0)")
    parser.add_argument("--port-start", type=int, default=8200)
    args = parser.parse_args()

    models = find_model_files()
    if args.model:
        models = [(n, p) for n, p in models if args.model.lower() in n.lower()]
    if args.quick:
        models = [(n, p) for n, p in models if p.stat().st_size < 20 * 1024**3]

    combos = detect_device_backend_combos()
    if args.device:
        combos = [(d, n, b) for d, n, b in combos if d == args.device]

    print(f"\nFound {len(models)} models, {len(combos)} device/backend combos:")
    for d, n, b in combos:
        print(f"  {d} + {b}: {n}")
    print()

    results = {}
    port = args.port_start

    for model_name, model_path in models:
        results[model_name] = {}
        for device_id, device_name, backend in combos:
            binary = get_backend_binary(backend)
            if binary is None or not binary.exists():
                results[model_name][f"{device_id}-{backend}"] = {
                    "success": False,
                    "error": f"no {backend} binary",
                }
                continue

            dev_arg = resolve_dev_arg(device_id, backend)
            label = f"{device_id}+{backend}"
            print(f"[TEST] {model_name} on {label} ({dev_arg})...", flush=True)
            port += 1
            r = run_test(model_path, backend, dev_arg, binary, port)
            results[model_name][label] = r
            if r.get("success"):
                garbled = r.get("garbled", False)
                status = "GARBLED" if garbled else "OK"
                print(f"  -> {status}: {r.get('content', '')[:80]!r}", flush=True)
            else:
                print(f"  -> FAIL: {r.get('error', '')[:120]}", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}")

    print("\n=== SUMMARY ===")
    for model_name, by_combo in results.items():
        for label, r in by_combo.items():
            if r.get("success"):
                status = "GARBLED" if r.get("garbled") else "OK"
            else:
                status = "FAIL"
            print(f"  {model_name:40s} {label:20s} {status}")


if __name__ == "__main__":
    main()
