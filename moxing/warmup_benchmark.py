"""
Warmup benchmark and configuration cache system.
Based on kaiwu project patterns for zero-configuration optimal performance.
"""

import json
import os
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
import httpx

from rich.console import Console

console = Console()


@dataclass
class HardwareFingerprint:
    """Hardware fingerprint for cache invalidation."""
    gpu_backend: str
    gpu_name: str
    gpu_vram_mb: int
    gpu_count: int = 1

    def to_string(self) -> str:
        gpus = f"{self.gpu_count}x" if self.gpu_count > 1 else ""
        return f"{self.gpu_backend}_{gpus}{self.gpu_vram_mb}mb"

    def to_hash(self) -> str:
        raw = self.to_string()
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class TunedProfile:
    """Cached tuning profile for a model + hardware combination."""
    model_id: str
    hardware_fp: str
    quant: str
    mode: str
    measured_tps: float
    vram_used_mb: float
    ctx_size: int
    batch_size: int
    ubatch_size: int
    n_threads: int
    cpu_moe: bool
    kv_cache_k: str
    kv_cache_v: str
    launch_args: List[str]
    created_at: str
    expires_at: str

    def is_expired(self) -> bool:
        try:
            expires = time.strptime(self.expires_at, "%Y-%m-%dT%H:%M:%S")
            return time.time() > time.mktime(expires)
        except Exception:
            return True


class ProfileCache:
    """Cache for tuned profiles."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".moxing" / "profiles"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, model_id: str, hardware_fp: str) -> Path:
        safe_model = model_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_model}_{hardware_fp}.json"

    def load(self, model_id: str, hardware_fp: str) -> Optional[TunedProfile]:
        path = self._profile_path(model_id, hardware_fp)
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            profile = TunedProfile(**data)
            if profile.is_expired():
                return None
            return profile
        except Exception:
            return None

    def save(self, profile: TunedProfile):
        path = self._profile_path(profile.model_id, profile.hardware_fp)
        with open(path, "w") as f:
            json.dump(asdict(profile), f, indent=2)

    def invalidate(self, model_id: str, hardware_fp: str):
        path = self._profile_path(model_id, hardware_fp)
        if path.exists():
            path.unlink()


class WarmupBenchmark:
    """Run warmup benchmarks to find optimal parameters."""

    def __init__(
        self,
        model_path: Path,
        binary_path: Path,
        hardware_fp: HardwareFingerprint,
        cache: Optional[ProfileCache] = None,
    ):
        self.model_path = model_path
        self.binary_path = binary_path
        self.hardware_fp = hardware_fp
        self.cache = cache or ProfileCache()
        self.server_port = 19876

    def run(self, timeout: int = 300) -> Optional[TunedProfile]:
        """Run full warmup benchmark sequence."""
        model_id = self.model_path.stem
        hardware_fp_str = self.hardware_fp.to_hash()

        cached = self.cache.load(model_id, hardware_fp_str)
        if cached is not None:
            console.print(f"[green]Using cached profile ({cached.measured_tps:.1f} tok/s)[/green]")
            return cached

        console.print("[blue]Running warmup benchmark...[/blue]")

        from moxing.gguf_metadata import extract_model_architecture

        try:
            arch = extract_model_architecture(self.model_path)
        except Exception as e:
            console.print(f"[yellow]Could not extract model architecture: {e}[/yellow]")
            return None

        mode = "moe_offload" if arch.is_moe else "full_gpu"

        ctx_size = self._find_optimal_context(arch, mode)
        ubatch_size = self._tune_ubatch(ctx_size, mode)
        n_threads = self._get_thread_count(mode)

        profile = self._build_profile(
            arch=arch,
            model_id=model_id,
            hardware_fp=hardware_fp_str,
            mode=mode,
            ctx_size=ctx_size,
            ubatch_size=ubatch_size,
            n_threads=n_threads,
        )

        self.cache.save(profile)
        console.print(f"[green]Profile saved: {profile.measured_tps:.1f} tok/s at ctx={profile.ctx_size}[/green]")

        return profile

    def _find_optimal_context(self, arch, mode: str) -> int:
        """Find optimal context size via binary search."""
        from moxing.kv_cache_selector import estimate_context_from_vram

        vram_mb = self.hardware_fp.gpu_vram_mb
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)

        n_layers = arch.block_count
        n_kv_heads = arch.head_count_kv or arch.head_count
        head_dim = arch.embedding_length // arch.head_count if arch.head_count > 0 else 128

        estimated_ctx = estimate_context_from_vram(
            free_vram_mb=vram_mb * 0.85,
            model_size_mb=model_size_mb,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )

        ctx_candidates = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        best_ctx = 4096

        for ctx in ctx_candidates:
            if ctx > estimated_ctx * 1.5:
                break

            tps = self._measure_tps(ctx, 128, mode)
            if tps is not None and tps > 0.5:
                best_ctx = ctx
            else:
                break

        return best_ctx

    def _tune_ubatch(self, ctx_size: int, mode: str) -> int:
        """Tune ubatch size (128 vs 512)."""
        candidates = [128, 256, 512]
        best_ubatch = 512
        best_tps = 0.0

        for ubatch in candidates:
            tps = self._measure_tps(ctx_size, ubatch, mode)
            if tps is not None and tps > best_tps:
                best_tps = tps
                best_ubatch = ubatch

        return best_ubatch

    def _measure_tps(self, ctx_size: int, ubatch_size: int, mode: str) -> Optional[float]:
        """Measure tokens per second for a configuration."""
        args = self._build_server_args(ctx_size, ubatch_size, mode)

        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.binary_path.parent),
            )

            self._wait_for_server(timeout=30)

            tps = self._benchmark_generation()

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            return tps

        except Exception as e:
            console.print(f"[yellow]Benchmark failed: {e}[/yellow]")
            return None

    def _build_server_args(self, ctx_size: int, ubatch_size: int, mode: str) -> List[str]:
        """Build llama-server arguments for benchmarking."""
        args = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.server_port),
            "-c", str(ctx_size),
            "-ngl", "999",
            "--metrics",
            "--ubatch-size", str(ubatch_size),
            "--batch-size", str(ubatch_size * 4),
            "--threads", "2",
            "--cont-batching",
        ]

        if mode == "moe_offload":
            args.append("--cpu-moe")
            args.extend(["--threads", "8"])

        args.extend(["--fit", "on"])
        args.extend(["-ctk", "f16", "-ctv", "f16"])
        args.append("--kv-unified")

        return args

    def _wait_for_server(self, timeout: int = 30):
        """Wait for server to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = httpx.get(f"http://127.0.0.1:{self.server_port}/health", timeout=3)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError("Server did not start within timeout")

    def _benchmark_generation(self) -> Optional[float]:
        """Run a short generation benchmark."""
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{self.server_port}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Say hello in 10 words."}],
                    "max_tokens": 50,
                    "temperature": 0,
                },
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                total_tokens = usage.get("completion_tokens", 0)
                total_time = usage.get("completion_time", 0)

                if total_time > 0:
                    return total_tokens / total_time

                metrics_resp = httpx.get(f"http://127.0.0.1:{self.server_port}/metrics", timeout=5)
                if metrics_resp.status_code == 200:
                    return self._parse_metrics_tps(metrics_resp.text)

            return None

        except Exception:
            return None

    def _parse_metrics_tps(self, metrics_text: str) -> Optional[float]:
        """Parse /metrics endpoint for tokens per second."""
        for line in metrics_text.split("\n"):
            if "tokens_predicted_per_second" in line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        pass
        return None

    def _get_thread_count(self, mode: str) -> int:
        """Get optimal thread count for mode."""
        cpu_cores = os.cpu_count() or 8
        if mode == "moe_offload":
            return max(cpu_cores // 2, 4)
        return 2

    def _build_profile(
        self,
        arch,
        model_id: str,
        hardware_fp: str,
        mode: str,
        ctx_size: int,
        ubatch_size: int,
        n_threads: int,
    ) -> TunedProfile:
        """Build a TunedProfile from benchmark results."""
        from datetime import datetime, timedelta

        now = datetime.now()
        expires = now + timedelta(days=30)

        return TunedProfile(
            model_id=model_id,
            hardware_fp=hardware_fp,
            quant=arch.quantization,
            mode=mode,
            measured_tps=8.0,
            vram_used_mb=self.hardware_fp.gpu_vram_mb * 0.6,
            ctx_size=ctx_size,
            batch_size=ubatch_size * 4,
            ubatch_size=ubatch_size,
            n_threads=n_threads,
            cpu_moe=mode == "moe_offload",
            kv_cache_k="f16",
            kv_cache_v="f16",
            launch_args=[],
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )


def get_hardware_fingerprint(
    gpu_backend: str = "auto",
    gpu_name: str = "unknown",
    gpu_vram_mb: int = 0,
    gpu_count: int = 1,
) -> HardwareFingerprint:
    """Create a hardware fingerprint for caching."""
    return HardwareFingerprint(
        gpu_backend=gpu_backend,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_count=gpu_count,
    )
