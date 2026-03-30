#!/usr/bin/env python3
"""
Build llama.cpp binaries with Ollama patches.

Ollama maintains a forked llama.cpp with custom patches for:
- GLM-4.7-flash MLA flash attention
- Qwen3-VL interleave multi-rope
- Solar-Pro support
- And many other model architectures

This script builds llama.cpp with Ollama's patches applied.

Usage:
    python scripts/build_ollama_patched.py --backend cuda --platform linux --arch x64
    python scripts/build_ollama_patched.py --backend rocm --platform linux --arch x64
    python scripts/build_ollama_patched.py --backend vulkan --platform linux --arch x64

Requirements:
    - CMake
    - CUDA toolkit (for cuda backend)
    - ROCm (for rocm backend)
    - Vulkan SDK (for vulkan backend)
"""

import os
import sys
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path

OLLAMA_REPO = "https://github.com/ollama/ollama.git"
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = Path.home() / ".cache" / "moxing" / "ollama-build"


def clone_ollama_repo(dest: Path):
    """Clone Ollama repository."""
    if dest.exists():
        print(f"Repository already exists at {dest}")
        return
    
    print(f"Cloning Ollama repository to {dest}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", OLLAMA_REPO, str(dest)],
        check=True
    )


def apply_patches(llama_dir: Path):
    """Apply Ollama's patches to llama.cpp."""
    print("Patches are already applied in Ollama's vendored llama.cpp")
    print(f"llama.cpp location: {llama_dir}")


def build_llama_cpp(
    llama_dir: Path,
    backend: str,
    build_dir: Path,
    install_dir: Path,
):
    """Build llama.cpp with the specified backend."""
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    cmake_args = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(llama_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
    ]
    
    if backend == "cuda":
        cmake_args.extend([
            "-DLLAMA_CUBLAS=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90",
        ])
    elif backend == "rocm":
        cmake_args.extend([
            "-DLLAMA_HIPBLAS=ON",
            "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang",
            "-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++",
        ])
    elif backend == "vulkan":
        cmake_args.extend([
            "-DLLAMA_VULKAN=ON",
        ])
    elif backend == "metal":
        cmake_args.extend([
            "-DLLAMA_METAL=ON",
        ])
    elif backend == "cpu":
        cmake_args.extend([
            "-DLLAMA_NATIVE=OFF",
        ])
    
    print(f"Running cmake: {' '.join(cmake_args)}")
    subprocess.run(cmake_args, check=True, cwd=llama_dir)
    
    print(f"Building with {os.cpu_count()} threads...")
    build_cmd = [
        "cmake", "--build", str(build_dir),
        "--config", "Release",
        "-j", str(os.cpu_count() or 4),
    ]
    subprocess.run(build_cmd, check=True)
    
    install_dir.mkdir(parents=True, exist_ok=True)
    
    binaries = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]
    
    for binary in binaries:
        src = build_dir / "bin" / binary
        if not src.exists():
            src = build_dir / binary
        
        if src.exists():
            dst = install_dir / binary
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            print(f"Installed: {binary}")
    
    for lib in (build_dir / "lib").glob("*.so*"):
        dst = install_dir / lib.name
        if lib.is_symlink():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(os.readlink(lib), dst)
        else:
            shutil.copy2(lib, dst)
        print(f"Installed: {lib.name}")


def main():
    parser = argparse.ArgumentParser(description="Build llama.cpp with Ollama patches")
    parser.add_argument(
        "--backend", "-b",
        choices=["cuda", "rocm", "vulkan", "metal", "cpu"],
        required=True,
        help="GPU backend"
    )
    parser.add_argument(
        "--platform", "-p",
        choices=["linux", "darwin", "windows"],
        default="linux" if sys.platform != "darwin" else "darwin",
        help="Target platform"
    )
    parser.add_argument(
        "--arch", "-a",
        choices=["x64", "arm64"],
        default="x64",
        help="Target architecture"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    ollama_dir = CACHE_DIR / "ollama"
    clone_ollama_repo(ollama_dir)
    
    llama_dir = ollama_dir / "llama" / "vendor"
    if not llama_dir.exists():
        print(f"Error: llama.cpp not found at {llama_dir}")
        print("Make sure the Ollama repository was cloned correctly.")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "moxing" / "bin" / f"{args.platform}-{args.arch}-{args.backend}"
    build_dir = CACHE_DIR / "build" / f"{args.backend}"
    
    print(f"\nBuilding llama.cpp with {args.backend} backend...")
    print(f"Source: {llama_dir}")
    print(f"Build: {build_dir}")
    print(f"Output: {output_dir}")
    
    build_llama_cpp(llama_dir, args.backend, build_dir, output_dir)
    
    version_file = output_dir / "VERSION"
    version_file.write_text("ollama-patched\n" + args.backend + "\n")
    
    print(f"\n[green]Build complete![/green]")
    print(f"Binaries installed to: {output_dir}")


if __name__ == "__main__":
    main()