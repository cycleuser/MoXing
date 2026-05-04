#!/usr/bin/env python3
"""
Build llama.cpp binaries from official repo for MoXing.

Outputs to moxing/bin/official-{platform}-{backend}/ alongside ollama runners.

Supports: CUDA, ROCm, Vulkan, CPU backends.
Supports latest models including Qwen3.5 (Qwen35/Qwen35MoE) and Gemma4.

Usage:
    python scripts/build_official_llama.py --backend cuda
    python scripts/build_official_llama.py --backend rocm
    python scripts/build_official_llama.py --backend vulkan
    python scripts/build_official_llama.py --backend cpu
    python scripts/build_official_llama.py --all
    python scripts/build_official_llama.py --list-devices
"""

import os
import sys
import re
import subprocess
import shutil
import argparse
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LLAMA_CPP_SRC = PROJECT_ROOT / "llama.cpp"
BIN_DIR = PROJECT_ROOT / "moxing" / "bin"


def run_cmd(cmd, cwd=None, check=True, capture=True):
    """Run command and return result."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        print(f"STDOUT: {result.stdout}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def get_platform_info():
    """Get current platform info."""
    if platform.system() == "Darwin":
        os_name = "darwin"
    elif platform.system() == "Windows":
        os_name = "windows"
    else:
        os_name = "linux"
    
    machine = platform.machine().lower()
    if machine in ["arm64", "aarch64"]:
        arch = "arm64"
    else:
        arch = "x64"
    
    return os_name, arch


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
        print(f"[WARN] Could not detect CUDA archs, using default: {archs}")
    
    return archs


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
        try:
            result = run_cmd(["hipconfig", "-g"], check=False)
            if result.returncode == 0 and result.stdout.strip():
                gfx = result.stdout.strip()
                if gfx.startswith("gfx"):
                    targets.append(gfx)
        except Exception:
            pass
    
    if not targets:
        targets = ["gfx1100"]
        print(f"[WARN] Could not detect AMD targets, using default: {targets}")
    
    return targets


def detect_available_backends():
    """Detect which GPU backends are available."""
    backends = ["cpu"]
    
    if shutil.which("nvcc") or shutil.which("nvidia-smi"):
        backends.append("cuda")
    
    if shutil.which("hipcc") or Path("/opt/rocm").exists():
        backends.append("rocm")
    
    if shutil.which("vulkaninfo") or os.environ.get("VULKAN_SDK"):
        backends.append("vulkan")
    
    return backends


def build_backend(backend, build_dir, output_path, jobs=None):
    """Build llama.cpp for specific backend."""
    print(f"\n{'='*60}")
    print(f"[BUILD] Backend: {backend}")
    print(f"{'='*60}")
    
    if build_dir.exists():
        print(f"[INFO] Cleaning existing build dir: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmake_args = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(LLAMA_CPP_SRC),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
    ]
    
    if backend == "cuda":
        archs = get_cuda_archs()
        arch_str = ";".join(archs)
        print(f"[INFO] CUDA architectures: {archs}")
        cmake_args.extend([
            "-DGGML_CUDA=ON",
            "-DGGML_HIP=OFF",
            "-DGGML_VULKAN=OFF",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch_str}",
            "-DGGML_CUDA_GRAPHS=ON",
        ])
    elif backend == "rocm":
        targets = get_amd_targets()
        target_str = ";".join(targets)
        print(f"[INFO] AMD targets: {targets}")
        
        rocm_path = None
        for p in ["/opt/rocm/core-7.12", "/opt/rocm/current", "/opt/rocm"]:
            if Path(p).exists():
                rocm_path = Path(p)
                break
        
        if rocm_path:
            hipcc = rocm_path / "bin" / "hipcc"
            if hipcc.exists():
                cmake_args.extend([
                    "-DGGML_HIP=ON",
                    "-DGGML_CUDA=OFF",
                    "-DGGML_VULKAN=OFF",
                    f"-DAMDGPU_TARGETS={target_str}",
                    f"-DCMAKE_C_COMPILER={hipcc}",
                    f"-DCMAKE_CXX_COMPILER={hipcc}",
                    f"-DCMAKE_PREFIX_PATH={rocm_path}",
                ])
            else:
                print(f"[WARN] hipcc not found at {hipcc}, using default compiler")
                cmake_args.extend([
                    "-DGGML_HIP=ON",
                    "-DGGML_CUDA=OFF",
                    "-DGGML_VULKAN=OFF",
                    f"-DAMDGPU_TARGETS={target_str}",
                ])
        else:
            cmake_args.extend([
                "-DGGML_HIP=ON",
                "-DGGML_CUDA=OFF",
                "-DGGML_VULKAN=OFF",
                f"-DAMDGPU_TARGETS={target_str}",
            ])
    elif backend == "vulkan":
        print(f"[INFO] Vulkan backend")
        cmake_args.extend([
            "-DGGML_VULKAN=ON",
            "-DGGML_CUDA=OFF",
            "-DGGML_HIP=OFF",
        ])
    elif backend == "cpu":
        print(f"[INFO] CPU-only backend")
        cmake_args.extend([
            "-DGGML_CUDA=OFF",
            "-DGGML_HIP=OFF",
            "-DGGML_VULKAN=OFF",
        ])
    elif backend == "metal":
        print(f"[INFO] Metal backend (macOS)")
        cmake_args.extend([
            "-DGGML_METAL=ON",
            "-DGGML_CUDA=OFF",
            "-DGGML_HIP=OFF",
            "-DGGML_VULKAN=OFF",
        ])
    
    print(f"\n[INFO] Running cmake configure...")
    run_cmd(cmake_args, cwd=LLAMA_CPP_SRC)
    
    if jobs is None:
        jobs = os.cpu_count() or 4
    
    print(f"\n[INFO] Building with {jobs} parallel jobs...")
    run_cmd(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs)],
        cwd=LLAMA_CPP_SRC
    )
    
    print(f"\n[INFO] Copying binaries to: {output_path}")
    
    binaries = ["llama-server", "llama-cli", "llama-mtmd-cli", "llama-bench", "llama-quantize"]
    bin_dir = build_dir / "bin"
    
    if not bin_dir.exists():
        bin_dir = build_dir
    
    copied_files = []
    
    for binary in binaries:
        src = bin_dir / binary
        if src.exists():
            dst = output_path / binary
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            copied_files.append(binary)
            print(f"  [OK] {binary}")
        else:
            print(f"  [SKIP] {binary} not found")
    
    lib_patterns = ["*.so*", "*.dylib*", "*.dll", "*.a"]
    for pattern in lib_patterns:
        for lib in bin_dir.glob(pattern):
            if lib.is_file():
                dst = output_path / lib.name
                if not dst.exists():
                    if lib.is_symlink():
                        link_target = lib.readlink()
                        dst.symlink_to(link_target)
                    else:
                        shutil.copy2(lib, dst)
                    copied_files.append(lib.name)
                    print(f"  [LIB] {lib.name}")
    
    result = run_cmd(
        ["git", "describe", "--tags", "--always"],
        cwd=LLAMA_CPP_SRC,
        check=False
    )
    version = result.stdout.strip() if result.returncode == 0 else "unknown"
    
    version_file = output_path / "VERSION"
    version_file.write_text(f"{version}\n{backend}\nofficial\n")
    print(f"  [VERSION] {version} ({backend})")
    
    print(f"\n[OK] Built {backend}: {len(copied_files)} files -> {output_path.name}")
    return True


def list_devices():
    """List available GPU devices."""
    print("\n" + "=" * 60)
    print("GPU DEVICE DETECTION")
    print("=" * 60)
    
    print("\n[NVIDIA CUDA]")
    if shutil.which("nvidia-smi"):
        result = run_cmd(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv"],
            check=False
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("  No NVIDIA GPUs detected")
    else:
        print("  nvidia-smi not found")
    
    print("\n[AMD ROCm]")
    if shutil.which("rocminfo"):
        result = run_cmd(["rocminfo"], check=False)
        if result.returncode == 0:
            gfx_found = []
            for line in result.stdout.split("\n"):
                if "gfx" in line.lower() and "Name:" in line:
                    print(f"  {line.strip()}")
                    gfx_found.append(line)
            if not gfx_found:
                print("  No AMD GPUs detected")
    elif Path("/opt/rocm").exists():
        print("  ROCm installed but rocminfo not found")
    else:
        print("  ROCm not installed")
    
    print("\n[Vulkan]")
    if shutil.which("vulkaninfo"):
        result = run_cmd(["vulkaninfo", "--summary"], check=False)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for line in lines[:20]:
                if "deviceName" in line or "driverVersion" in line:
                    print(f"  {line.strip()}")
    else:
        print("  vulkaninfo not found")
    
    print("\n[Available Backends]")
    backends = detect_available_backends()
    for b in backends:
        print(f"  - {b}")


def main():
    parser = argparse.ArgumentParser(
        description="Build llama.cpp from official repo for MoXing"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["cuda", "rocm", "vulkan", "cpu", "metal", "all", "auto"],
        default="auto",
        help="GPU backend to build (default: auto-detect)"
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=None,
        help="Number of parallel build jobs"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: moxing/bin)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available GPU devices"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return 0
    
    if not LLAMA_CPP_SRC.exists():
        print(f"[ERROR] llama.cpp not found at {LLAMA_CPP_SRC}")
        print(f"[INFO] Please clone llama.cpp to: {LLAMA_CPP_SRC}")
        print(f"[INFO] git clone https://github.com/ggml-org/llama.cpp {LLAMA_CPP_SRC}")
        return 1
    
    result = run_cmd(
        ["git", "describe", "--tags", "--always"],
        cwd=LLAMA_CPP_SRC,
        check=False
    )
    version = result.stdout.strip() if result.returncode == 0 else "unknown"
    print(f"[INFO] llama.cpp version: {version}")
    
    os_name, arch = get_platform_info()
    platform_name = f"{os_name}-{arch}"
    print(f"[INFO] Platform: {platform_name}")
    
    output_base = Path(args.output) if args.output else BIN_DIR
    
    build_root = Path.home() / ".cache" / "moxing" / "build-official"
    
    if args.backend == "auto":
        backends = detect_available_backends()
        print(f"[INFO] Auto-detected backends: {backends}")
    elif args.backend == "all":
        backends = detect_available_backends()
        if os_name == "darwin":
            backends = ["metal", "cpu"]
    else:
        backends = [args.backend]
    
    results = {}
    for backend in backends:
        output_name = f"official-{platform_name}-{backend}"
        output_path = output_base / output_name
        build_dir = build_root / backend
        
        try:
            success = build_backend(backend, build_dir, output_path, args.jobs)
            results[backend] = "OK" if success else "FAILED"
        except Exception as e:
            print(f"[ERROR] Build failed for {backend}: {e}")
            results[backend] = f"FAILED: {str(e)[:50]}"
    
    print("\n" + "=" * 60)
    print("BUILD RESULTS")
    print("=" * 60)
    for backend, status in results.items():
        print(f"  {backend}: {status}")
    
    print(f"\n[INFO] Binaries installed to: {output_base}")
    print(f"[INFO] Use with moxing: moxing serve model.gguf -d gpu0 -b {backends[0]}")
    
    return 0 if all(s == "OK" for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())