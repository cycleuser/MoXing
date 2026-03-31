#!/usr/bin/env python3
"""
Build llama.cpp from Ollama's source with ROCm/Vulkan support.

Ollama's llama.cpp includes support for all model architectures (glm4moelite, etc.)
This script builds standalone llama-server binaries that can run all Ollama models.

Usage:
    python scripts/build_ollama_llama.py --backend rocm
    python scripts/build_ollama_llama.py --backend vulkan
    python scripts/build_ollama_llama.py --all
"""

import os
import sys
import shutil
import subprocess
import argparse
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "moxing" / "bin"

OLLAMA_PATH = Path("/home/fred/Documents/GitHub/Others/ollama")
OLLAMA_LLAMA_SRC = OLLAMA_PATH / "llama" / "llama.cpp"
OLLAMA_GGML_SRC = OLLAMA_PATH / "ml" / "backend" / "ggml" / "ggml" / "src"


def run_cmd(cmd, cwd=None, check=True, env=None):
    """Run command."""
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env
    )
    if check and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        print(f"STDOUT: {result.stdout}")
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result


def build_ollama_llama(backend, output_dir, jobs=8):
    """Build llama.cpp from Ollama source with specific backend."""
    print(f"\n{'='*60}")
    print(f"[BUILD] Ollama llama.cpp with {backend}")
    print(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = OLLAMA_LLAMA_SRC / f"build-{backend}"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    cmake_opts = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(OLLAMA_LLAMA_SRC),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={output_dir}",
        "-DGGML_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_CURL=OFF",
        f"-DLLAMA_GGML_PATH={OLLAMA_GGML_SRC}",
    ]
    
    if backend == "rocm":
        rocm_path = Path("/opt/rocm")
        rocm_core = list(rocm_path.glob("core-*"))
        if rocm_core:
            rocm_path = rocm_core[0]
        
        cmake_opts.extend([
            "-DGGML_HIPBLAS=ON",
            "-DGGML_CUDA=OFF",
            "-DGGML_VULKAN=OFF",
            f"-DCMAKE_C_COMPILER={rocm_path}/bin/hipcc",
            f"-DCMAKE_CXX_COMPILER={rocm_path}/bin/hipcc",
            f"-DCMAKE_PREFIX_PATH={rocm_path}",
        ])
    elif backend == "cuda":
        cmake_opts.extend([
            "-DGGML_CUDA=ON",
            "-DGGML_HIPBLAS=OFF",
            "-DGGML_VULKAN=OFF",
        ])
    elif backend == "vulkan":
        cmake_opts.extend([
            "-DGGML_VULKAN=ON",
            "-DGGML_CUDA=OFF",
            "-DGGML_HIPBLAS=OFF",
        ])
    else:
        cmake_opts.extend([
            "-DGGML_CUDA=OFF",
            "-DGGML_HIPBLAS=OFF",
            "-DGGML_VULKAN=OFF",
        ])
    
    run_cmd(cmake_opts, cwd=OLLAMA_LLAMA_SRC)
    run_cmd(["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs)], cwd=OLLAMA_LLAMA_SRC)
    
    copy_binaries(build_dir / "bin", output_dir, "linux")
    
    commit = "ollama-0.19.0"
    (output_dir / "VERSION").write_text(f"{commit}\n{backend}\nollama-source\n")
    
    count = len(list(output_dir.iterdir()))
    print(f"[OK] Built {backend}: {count} files")
    return True


def copy_binaries(bin_dir, output_dir, os_type):
    """Copy binaries from build directory."""
    if not bin_dir.exists():
        bin_dir = bin_dir.parent
    
    essential = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]
    
    for binary in essential:
        src = bin_dir / binary
        if src.exists():
            dst = output_dir / binary
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            print(f"  [COPY] {binary}")
    
    if os_type == "linux":
        for f in bin_dir.iterdir():
            if f.suffix == ".so" or ".so." in f.name:
                dst = output_dir / f.name
                if f.is_symlink():
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    dst.symlink_to(f.readlink())
                else:
                    shutil.copy2(f, dst)
                print(f"  [COPY] {f.name}")


def main():
    parser = argparse.ArgumentParser(description="Build Ollama llama.cpp")
    parser.add_argument("--backend", "-b", help="Backend: rocm, cuda, vulkan, cpu")
    parser.add_argument("--all", action="store_true", help="Build all backends")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="Parallel jobs")
    
    args = parser.parse_args()
    
    if args.all:
        backends = ["rocm", "cuda", "vulkan", "cpu"]
    elif args.backend:
        backends = [args.backend]
    else:
        parser.print_help()
        return 1
    
    arch = "x64" if platform.machine() in ["x86_64", "amd64"] else "arm64"
    
    for backend in backends:
        output_dir = BIN_DIR / f"linux-{arch}-{backend}-ollama"
        try:
            build_ollama_llama(backend, output_dir, args.jobs)
        except Exception as e:
            print(f"[ERROR] Failed to build {backend}: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())