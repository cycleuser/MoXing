#!/usr/bin/env python3
"""
Build llama.cpp binaries with Ollama patches.

This script builds llama-server and related binaries with Ollama's patches applied,
supporting multiple GPU backends: CUDA, ROCm, Vulkan, CPU.

Usage:
    python scripts/build_ollama_binaries.py --backend cuda
    python scripts/build_ollama_binaries.py --backend rocm
    python scripts/build_ollama_binaries.py --backend vulkan
    python scripts/build_ollama_binaries.py --backend cpu
    python scripts/build_ollama_binaries.py --all

Requirements:
    - CMake >= 3.21
    - CUDA toolkit (for cuda backend)
    - ROCm (for rocm backend)
    - Vulkan SDK (for vulkan backend)
"""

import os
import sys
import re
import subprocess
import argparse
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

OLLAMA_PATH = Path("/home/fred/Documents/GitHub/Others/ollama")
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "moxing" / "bin"

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "ec98e2002"  # Ollama's tracking commit from Makefile.sync


def run_cmd(cmd, cwd=None, env=None, check=True):
    """Run a command and return output."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def clone_llama_cpp(dest: Path, commit: str):
    """Clone llama.cpp repository at specific commit."""
    if dest.exists():
        print(f"[INFO] llama.cpp already exists at {dest}")
        return
    
    print(f"[INFO] Cloning llama.cpp to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    run_cmd(["git", "clone", "--depth", "100", LLAMA_CPP_REPO, str(dest)])
    run_cmd(["git", "checkout", commit], cwd=str(dest))


def apply_ollama_patches(llama_cpp_dir: Path, ollama_dir: Path):
    """Apply Ollama patches to llama.cpp."""
    patches_dir = ollama_dir / "llama" / "patches"
    
    if not patches_dir.exists():
        print(f"[ERROR] Patches directory not found: {patches_dir}")
        return False
    
    patches = sorted(patches_dir.glob("*.patch"))
    
    print(f"[INFO] Found {len(patches)} patches to apply")
    
    for patch_file in patches:
        patch_name = patch_file.name
        print(f"[INFO] Applying patch: {patch_name}")
        
        result = run_cmd(
            ["git", "apply", "--check", str(patch_file)],
            cwd=str(llama_cpp_dir),
            check=False
        )
        
        if result.returncode == 0:
            run_cmd(
                ["git", "apply", str(patch_file)],
                cwd=str(llama_cpp_dir)
            )
            print(f"  [OK] Applied: {patch_name}")
        else:
            print(f"  [SKIP] Already applied or conflict: {patch_name}")
    
    return True


def get_amd_gfx_targets():
    """Detect AMD GPU architecture targets."""
    targets = []
    
    try:
        result = run_cmd(["rocm-smi", "--showid"], check=False)
        if result.returncode == 0:
            result2 = run_cmd(["rocminfo"], check=False)
            if result2.returncode == 0:
                for line in result2.stdout.split("\n"):
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
        print(f"[WARN] Could not detect AMD GPU targets, using default: {targets}")
    else:
        print(f"[INFO] Detected AMD GPU targets: {targets}")
    
    return targets


def get_cuda_architectures():
    """Detect CUDA architectures."""
    archs = []
    
    try:
        result = run_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], check=False)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                cap = line.strip().replace(".", "")
                if cap and cap not in archs:
                    archs.append(cap)
    except Exception:
        pass
    
    if not archs:
        archs = ["89", "90"]
        print(f"[WARN] Could not detect CUDA architectures, using default: {archs}")
    else:
        print(f"[INFO] Detected CUDA architectures: {archs}")
    
    return archs


def build_backend(llama_cpp_dir: Path, backend: str, build_dir: Path, install_dir: Path):
    """Build llama.cpp for a specific backend."""
    
    print(f"\n{'='*60}")
    print(f"[BUILD] Backend: {backend}")
    print(f"{'='*60}")
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    install_dir.mkdir(parents=True, exist_ok=True)
    
    cmake_args = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(llama_cpp_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=OFF",
        "-DGGML_BACKEND_DL=OFF",
        "-DGGML_SHARED=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
    ]
    
    if backend == "cuda":
        archs = get_cuda_architectures()
        arch_str = ";".join(archs)
        cmake_args.extend([
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch_str}",
            "-DGGML_CUDA_GRAPHS=ON",
            "-DGGML_CUDA_FA=ON",
        ])
    elif backend == "rocm":
        targets = get_amd_gfx_targets()
        target_str = ";".join(targets)
        cmake_args.extend([
            "-DGGML_HIP=ON",
            f"-DAMDGPU_TARGETS={target_str}",
            "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang",
            "-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++",
        ])
    elif backend == "vulkan":
        cmake_args.extend([
            "-DGGML_VULKAN=ON",
        ])
    elif backend == "cpu":
        cmake_args.extend([
            "-DGGML_CUDA=OFF",
            "-DGGML_HIP=OFF",
            "-DGGML_VULKAN=OFF",
        ])
    
    print(f"[CMAKE] {' '.join(cmake_args)}")
    result = run_cmd(cmake_args, cwd=str(llama_cpp_dir), check=False)
    if result.returncode != 0:
        print(f"[ERROR] CMake configure failed:\n{result.stderr}")
        return False
    
    nproc = os.cpu_count() or 4
    build_cmd = [
        "cmake", "--build", str(build_dir),
        "--config", "Release",
        "-j", str(nproc),
    ]
    
    print(f"[BUILD] Building with {nproc} threads...")
    result = run_cmd(build_cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Build failed:\n{result.stderr}")
        return False
    
    print(f"[INSTALL] Copying binaries to {install_dir}...")
    
    binaries = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]
    
    for binary in binaries:
        src = build_dir / "bin" / binary
        if not src.exists():
            src = build_dir / binary
        
        if src.exists():
            dst = install_dir / binary
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            print(f"  [OK] {binary}")
        else:
            print(f"  [WARN] {binary} not found")
    
    lib_patterns = ["*.so*", "*.dylib", "*.a"]
    for pattern in lib_patterns:
        for lib in build_dir.rglob(pattern):
            if lib.is_file() and not lib.is_symlink():
                dst = install_dir / lib.name
                if not dst.exists():
                    shutil.copy2(lib, dst)
                    print(f"  [OK] {lib.name}")
    
    version_file = install_dir / "VERSION"
    timestamp = datetime.now().strftime("%Y%m%d")
    version_file.write_text(f"ollama-patched-{timestamp}\n{backend}\n")
    
    print(f"\n[SUCCESS] Build complete for {backend}")
    print(f"[OUTPUT] {install_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build llama.cpp binaries with Ollama patches"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["cuda", "rocm", "vulkan", "cpu", "all"],
        default="all",
        help="GPU backend to build (default: all)"
    )
    parser.add_argument(
        "--ollama-path",
        default=str(OLLAMA_PATH),
        help="Path to Ollama repository"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for binaries"
    )
    parser.add_argument(
        "--skip-patch",
        action="store_true",
        help="Skip applying patches (if already applied)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    
    args = parser.parse_args()
    
    ollama_path = Path(args.ollama_path)
    if not ollama_path.exists():
        print(f"[ERROR] Ollama repository not found: {ollama_path}")
        print("Please clone it first: git clone https://github.com/ollama/ollama.git")
        sys.exit(1)
    
    build_root = Path.home() / ".cache" / "moxing" / "build-ollama"
    llama_cpp_dir = build_root / "llama.cpp"
    
    print(f"[INFO] Ollama path: {ollama_path}")
    print(f"[INFO] Build root: {build_root}")
    print(f"[INFO] Output: {OUTPUT_DIR}")
    
    commit_file = ollama_path / "Makefile.sync"
    if commit_file.exists():
        content = commit_file.read_text()
        match = re.search(r'FETCH_HEAD\s*=\s*([a-f0-9]+)', content)
        if match:
            global LLAMA_CPP_COMMIT
            LLAMA_CPP_COMMIT = match.group(1)
            print(f"[INFO] Using Ollama's llama.cpp commit: {LLAMA_CPP_COMMIT}")
    
    clone_llama_cpp(llama_cpp_dir, LLAMA_CPP_COMMIT)
    
    if not args.skip_patch:
        apply_ollama_patches(llama_cpp_dir, ollama_path)
    
    platform = "linux" if sys.platform != "darwin" else "darwin"
    arch = "x64" if os.uname().machine in ["x86_64", "amd64"] else "arm64"
    
    backends = ["cuda", "rocm", "vulkan", "cpu"] if args.backend == "all" else [args.backend]
    
    results = {}
    
    for backend in backends:
        output_name = f"{platform}-{arch}-{backend}"
        output_dir = OUTPUT_DIR / output_name
        build_dir = build_root / "build" / backend
        
        print(f"\n[INFO] Building {backend} -> {output_dir}")
        
        try:
            success = build_backend(llama_cpp_dir, backend, build_dir, output_dir)
            results[backend] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"[ERROR] Build failed for {backend}: {e}")
            results[backend] = f"FAILED: {e}"
    
    print(f"\n{'='*60}")
    print("[SUMMARY] Build Results:")
    print(f"{'='*60}")
    for backend, status in results.items():
        print(f"  {backend:10s}: {status}")
    
    print(f"\n[DONE] Binaries installed to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()