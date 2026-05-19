#!/usr/bin/env python3
"""Build llama.cpp binaries for all backends and install into moxing cache.

Usage:
    python scripts/build_all_binaries.py              # Build all detected backends
    python scripts/build_all_binaries.py cuda         # Build CUDA only
    python scripts/build_all_binaries.py cuda vulkan  # Build specific backends
    python scripts/build_all_binaries.py --package    # Also create release tarballs
    python scripts/build_all_binaries.py --jobs 16    # Use 16 parallel jobs
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional

from moxing.binaries import CACHE_DIR, PlatformDetector
from moxing.cli.system import (
    _build_cmake_args,
    _check_build_prerequisites,
    _copy_build_outputs,
    _detect_gpu_arch,
    _get_install_hint,
)

LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent / "llama.cpp"

ALL_BACKENDS = ["cuda", "vulkan", "rocm", "cpu"]


def detect_available_backends() -> List[str]:
    available = []
    if shutil.which("nvcc") or Path("/usr/local/cuda").exists():
        available.append("cuda")
    if shutil.which("vulkaninfo") or Path("/usr/include/vulkan/vulkan.h").exists():
        available.append("vulkan")
    if shutil.which("hipconfig") or Path("/opt/rocm").exists():
        available.append("rocm")
    available.append("cpu")
    return available


def clean_build_dir(build_dir: Path):
    if build_dir.exists():
        shutil.rmtree(build_dir)


def build_backend(
    backend: str,
    jobs: int = 0,
    skip_checks: bool = False,
) -> bool:
    platform_name = PlatformDetector.get_platform_name()
    cache_dir = CACHE_DIR / f"{platform_name}-{backend}"

    print(f"\n{'=' * 60}")
    print(f"  Building: {backend}")
    print(f"  Platform: {platform_name}")
    print(f"  Output:   {cache_dir}")
    print(f"{'=' * 60}\n")

    if not skip_checks:
        missing = _check_build_prerequisites(backend)
        if missing:
            print("[ERROR] Missing dependencies:")
            for m in missing:
                print(f"  x {m}")
            print(f"\nInstall with:\n  {_get_install_hint(backend)}")
            print("\nUse --skip-checks to bypass")
            return False

    if not (LLAMA_CPP_DIR / "CMakeLists.txt").exists():
        print("[ERROR] llama.cpp source not found")
        print(f"  Clone: git clone https://github.com/ggml-org/llama.cpp.git {LLAMA_CPP_DIR}")
        return False

    gpu_arch = _detect_gpu_arch(backend)
    if gpu_arch and backend in ("cuda", "rocm"):
        print(f"GPU architectures: {gpu_arch}")

    build_dir = LLAMA_CPP_DIR / "build"

    clean_build_dir(build_dir)

    cmake_args = _build_cmake_args(build_dir, backend, gpu_arch)

    print("CMake configure:")
    for a in cmake_args:
        print(f"  {a}")

    result = subprocess.run(cmake_args, cwd=LLAMA_CPP_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] CMake configure failed for {backend}")
        return False

    actual_jobs = jobs if jobs > 0 else (os.cpu_count() or 8)

    print(f"\nBuilding with {actual_jobs} jobs...")
    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(actual_jobs)]
    result = subprocess.run(build_cmd, cwd=LLAMA_CPP_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] Build failed for {backend}")
        return False

    print(f"\nBuild complete for {backend}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_build_outputs(build_dir, cache_dir, backend)

    if copied:
        print(f"\nInstalled {len(copied)} files to: {cache_dir}")
        for name in sorted(copied):
            print(f"  {name}")
    else:
        print("[WARNING] No binaries to copy")
        clean_build_dir(build_dir)
        return False

    (cache_dir / "VERSION").write_text(f"local-build-{backend}\n{backend}\n")

    clean_build_dir(build_dir)
    return True


def package_backend(backend: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    platform_name = PlatformDetector.get_platform_name()
    cache_dir = CACHE_DIR / f"{platform_name}-{backend}"

    if not cache_dir.exists():
        print(f"[ERROR] No binaries for {backend}, build first")
        return None

    suffix = ".exe" if sys.platform == "win32" else ""
    server = cache_dir / f"llama-server{suffix}"
    if not server.exists():
        print(f"[ERROR] llama-server not found for {backend}")
        return None

    out = output_dir or Path.cwd() / "dist"
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pkg_dir = tmp / f"{platform_name}-{backend}"
        pkg_dir.mkdir()

        for f in cache_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, pkg_dir / f.name)

        tarball = out / f"moxing-{platform_name}-{backend}.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(pkg_dir, arcname=pkg_dir.name)

        size_mb = tarball.stat().st_size / (1024 * 1024)
        print(f"Package: {tarball.name} ({size_mb:.1f} MB)")

    return tarball


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build all moxing backends")
    parser.add_argument(
        "backends",
        nargs="*",
        help="Backends to build (default: all detected)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=0, help="Parallel build jobs (default: CPU count)"
    )
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip dependency checks"
    )
    parser.add_argument(
        "--package", action="store_true", help="Create release tarballs after building"
    )
    parser.add_argument(
        "--package-only", action="store_true", help="Only create packages from cached binaries"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output directory for packages"
    )
    args = parser.parse_args()

    backends = args.backends if args.backends else detect_available_backends()

    if args.package_only:
        for b in backends:
            print(f"\nPackaging {b}...")
            pkg = package_backend(b, args.output)
            if pkg:
                print(f"  {pkg}")
        return

    print(f"Backends to build: {', '.join(backends)}")
    print(f"Source: {LLAMA_CPP_DIR}")
    print(f"Cache: {CACHE_DIR}\n")

    results = {}
    for backend in backends:
        if backend not in ALL_BACKENDS:
            print(f"[WARNING] Unknown backend: {backend}, skipping")
            continue
        results[backend] = build_backend(backend, args.jobs, args.skip_checks)

    print(f"\n{'=' * 60}")
    print("  Build Summary")
    print(f"{'=' * 60}")
    for b, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {b}: {status}")
    print()

    if args.package:
        print(f"\n{'=' * 60}")
        print("  Packaging")
        print(f"{'=' * 60}")
        for b, ok in results.items():
            if ok:
                pkg = package_backend(b, args.output)
                if pkg:
                    print(f"  {b}: {pkg}")


if __name__ == "__main__":
    main()
