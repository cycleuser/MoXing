#!/usr/bin/env python3
"""
Build llama.cpp with Ollama patches for GLM-4.7-flash, GPT-OSS support.

This script builds llama.cpp binaries with all Ollama patches applied,
enabling support for model architectures not in standard llama.cpp:
- GLM-4.7-flash (glm4moelite)
- GPT-OSS
- And other models requiring Ollama's patches

Usage:
    python scripts/build_patched_llama.py --backend rocm
    python scripts/build_patched_llama.py --backend cuda
    python scripts/build_patched_llama.py --backend vulkan
    python scripts/build_patched_llama.py --backend cpu
    python scripts/build_patched_llama.py --all
"""

import os
import sys
import re
import subprocess
import shutil
import argparse
from pathlib import Path
from datetime import datetime

OLLAMA_PATH = Path("/home/fred/Documents/GitHub/Others/ollama")
LLAMA_CPP_SRC = Path("/home/fred/Documents/GitHub/cycleuser/llama.cpp")
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "moxing" / "bin"

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"


def run_cmd(cmd, cwd=None, check=True):
    """Run command and return result."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def get_ollama_llama_commit():
    """Get the llama.cpp commit that Ollama uses."""
    makefile = OLLAMA_PATH / "Makefile.sync"
    if makefile.exists():
        content = makefile.read_text()
        match = re.search(r'FETCH_HEAD\s*=\s*([a-f0-9]+)', content)
        if match:
            return match.group(1)
    return None


def prepare_llama_cpp():
    """Prepare llama.cpp source with Ollama's commit."""
    commit = get_ollama_llama_commit()
    if not commit:
        print("[ERROR] Could not find Ollama's llama.cpp commit")
        return False
    
    print(f"[INFO] Ollama uses llama.cpp commit: {commit}")
    
    if not LLAMA_CPP_SRC.exists():
        print(f"[ERROR] llama.cpp not found at {LLAMA_CPP_SRC}")
        return False
    
    result = run_cmd(["git", "cat-file", "-t", commit], cwd=LLAMA_CPP_SRC, check=False)
    if result.returncode != 0:
        print(f"[INFO] Fetching commit {commit}...")
        run_cmd(["git", "fetch", "origin"], cwd=LLAMA_CPP_SRC)
    
    print(f"[INFO] Checking out commit {commit}...")
    run_cmd(["git", "checkout", commit], cwd=LLAMA_CPP_SRC)
    
    return True


def apply_ollama_patches():
    """Apply Ollama's patches to llama.cpp."""
    patches_dir = OLLAMA_PATH / "llama" / "patches"
    
    if not patches_dir.exists():
        print(f"[ERROR] Patches directory not found: {patches_dir}")
        return 0
    
    patches = sorted(patches_dir.glob("*.patch"))
    print(f"[INFO] Found {len(patches)} patches")
    
    applied = 0
    for patch_file in patches:
        patch_name = patch_file.name
        
        result = run_cmd(
            ["git", "apply", "--check", str(patch_file)],
            cwd=LLAMA_CPP_SRC,
            check=False
        )
        
        if result.returncode == 0:
            run_cmd(["git", "apply", str(patch_file)], cwd=LLAMA_CPP_SRC)
            print(f"  [OK] {patch_name}")
            applied += 1
        else:
            print(f"  [SKIP] {patch_name}")
    
    return applied


def get_amd_targets():
    """Detect AMD GPU targets."""
    targets = []
    try:
        result = run_cmd(["rocminfo"], check=False)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Name:" in line and "gfx" in line.lower():
                    match = re.search(r'gfx(\d+[a-z]?)', line, re.I)
                    if match:
                        gfx = f"gfx{match.group(1)}"
                        if gfx not in targets:
                            targets.append(gfx)
    except Exception:
        pass
    
    if not targets:
        targets = ["gfx1100"]
        print(f"[WARN] Could not detect AMD targets, using default: {targets}")
    
    return targets


def get_cuda_archs():
    """Detect CUDA architectures."""
    archs = []
    try:
        result = run_cmd(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                cap = line.strip().replace(".", "")
                if cap and cap not in archs:
                    archs.append(cap)
    except Exception:
        pass
    
    if not archs:
        archs = ["89", "90"]
    
    return archs


def build_backend(backend, build_dir, output_path):
    """Build llama.cpp for specific backend."""
    import tempfile
    
    print(f"\n[BUILD] Backend: {backend}")
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmake_args = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(LLAMA_CPP_SRC),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + str(output_path),
        "-DGGML_NATIVE=OFF",
        "-DGGML_BACKEND_DL=OFF",
        "-DGGML_SHARED=ON",
        "-DLLAMA_BUILD_SERVER=ON",
    ]
    
    if backend == "cuda":
        archs = get_cuda_archs()
        arch_str = ";".join(archs)
        print(f"[INFO] CUDA architectures: {archs}")
        cmake_args.extend([
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch_str}",
            "-DGGML_CUDA_GRAPHS=ON",
            "-DGGML_CUDA_FA=ON",
        ])
    elif backend == "rocm":
        targets = get_amd_targets()
        target_str = ";".join(targets)
        print(f"[INFO] AMD targets: {targets}")
        
        rocm_paths = [
            "/opt/rocm/core-7.12/bin",
            "/opt/rocm/current/bin",
            "/opt/rocm/bin",
        ]
        clang_path = None
        for p in rocm_paths:
            if Path(p).joinpath("amdclang").exists():
                clang_path = Path(p)
                break
        
        if clang_path:
            clang = clang_path / "amdclang"
            clangxx = clang_path / "amdclang++"
        else:
            clang = Path("clang")
            clangxx = Path("clang++")
        
        print(f"[INFO] Using compiler: {clang}")
        
        cmake_args.extend([
            "-DGGML_HIP=ON",
            f"-DAMDGPU_TARGETS={target_str}",
            f"-DCMAKE_C_COMPILER={clang}",
            f"-DCMAKE_CXX_COMPILER={clangxx}",
        ])
    elif backend == "vulkan":
        cmake_args.extend(["-DGGML_VULKAN=ON"])
    elif backend == "cpu":
        cmake_args.extend([
            "-DGGML_CUDA=OFF",
            "-DGGML_HIP=OFF",
            "-DGGML_VULKAN=OFF",
        ])
    
    print(f"[INFO] Running cmake...")
    run_cmd(cmake_args, cwd=LLAMA_CPP_SRC)
    
    nproc = os.cpu_count() or 4
    print(f"[INFO] Building with {nproc} threads...")
    run_cmd(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(nproc)],
        cwd=LLAMA_CPP_SRC
    )
    
    print(f"[INFO] Installing binaries...")
    
    binaries = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]
    
    for binary in binaries:
        src = build_dir / "bin" / binary
        if not src.exists():
            src = build_dir / binary
        
        if src.exists():
            dst = output_path / binary
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            print(f"  [OK] {binary}")
        else:
            print(f"  [WARN] {binary} not found")
    
    for pattern in ["*.so*", "*.dylib", "*.a"]:
        for lib in build_dir.rglob(pattern):
            if lib.is_file() and not lib.is_symlink():
                dst = output_path / lib.name
                if not dst.exists():
                    shutil.copy2(lib, dst)
                    print(f"  [LIB] {lib.name}")
    
    timestamp = datetime.now().strftime("%Y%m%d")
    version_file = output_path / "VERSION"
    version_file.write_text(f"ollama-patched-{timestamp}\n{backend}\n")
    
    print(f"[OK] Build complete: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build llama.cpp with Ollama patches"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["cuda", "rocm", "vulkan", "cpu", "all"],
        default="all",
        help="GPU backend to build"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Skip applying patches (if already applied)"
    )
    
    args = parser.parse_args()
    
    if not OLLAMA_PATH.exists():
        print(f"[ERROR] Ollama not found at {OLLAMA_PATH}")
        sys.exit(1)
    
    if not LLAMA_CPP_SRC.exists():
        print(f"[ERROR] llama.cpp not found at {LLAMA_CPP_SRC}")
        sys.exit(1)
    
    print(f"[INFO] Ollama path: {OLLAMA_PATH}")
    print(f"[INFO] llama.cpp path: {LLAMA_CPP_SRC}")
    print(f"[INFO] Output: {OUTPUT_DIR}")
    
    if not prepare_llama_cpp():
        sys.exit(1)
    
    if not args.no_patch:
        applied = apply_ollama_patches()
        print(f"[INFO] Applied {applied} patches")
    
    platform = "linux" if sys.platform != "darwin" else "darwin"
    arch = "x64" if os.uname().machine in ["x86_64", "amd64"] else "arm64"
    
    build_root = Path.home() / ".cache" / "moxing" / "build-patched"
    
    backends = ["cuda", "rocm", "vulkan", "cpu"] if args.backend == "all" else [args.backend]
    
    results = {}
    for backend in backends:
        output_name = f"{platform}-{arch}-{backend}"
        output_path = OUTPUT_DIR / output_name if args.output is None else Path(args.output) / output_name
        build_dir = build_root / backend
        
        try:
            success = build_backend(backend, build_dir, output_path)
            results[backend] = "OK" if success else "FAILED"
        except Exception as e:
            print(f"[ERROR] Build failed for {backend}: {e}")
            results[backend] = f"FAILED: {e}"
    
    print("\n" + "=" * 50)
    print("BUILD RESULTS:")
    print("=" * 50)
    for backend, status in results.items():
        print(f"  {backend}: {status}")
    
    print(f"\nBinaries installed to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()