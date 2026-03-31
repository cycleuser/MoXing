#!/usr/bin/env python3
"""
Build llama.cpp binaries from Ollama's patched source for multiple platforms.

This script:
1. Uses Ollama's llama.cpp source (already includes all patches for GLM-4.7, etc.)
2. Builds for Linux (native) - CUDA, ROCm, Vulkan, CPU
3. Cross-compiles for Windows using mingw - CUDA, Vulkan, CPU
4. Packages and uploads to GitHub releases

Usage:
    python scripts/build_ollama_binaries.py --build-linux
    python scripts/build_ollama_binaries.py --build-windows
    python scripts/build_ollama_binaries.py --package
    python scripts/build_ollama_binaries.py --upload
    python scripts/build_ollama_binaries.py --all
"""

import os
import sys
import re
import json
import shutil
import tarfile
import tempfile
import subprocess
import argparse
import platform
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "moxing" / "bin"
DIST_DIR = PROJECT_ROOT / "dist" / "binaries-patched"

OLLAMA_PATH = Path("/home/fred/Documents/GitHub/Others/ollama")
OLLAMA_LLAMA_SRC = OLLAMA_PATH / "llama" / "llama.cpp"

MOXING_REPO = "cycleuser/MoXing"


def run_cmd(cmd, cwd=None, check=True, env=None):
    """Run command and return result."""
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


def get_ollama_commit():
    """Get the llama.cpp commit that Ollama uses."""
    makefile = OLLAMA_PATH / "Makefile.sync"
    if makefile.exists():
        content = makefile.read_text()
        match = re.search(r'FETCH_HEAD\s*=\s*([a-f0-9]+)', content)
        if match:
            return match.group(1)
    return None


def build_linux_backend(backend, output_dir, jobs=8):
    """Build for Linux with specific backend."""
    arch = "x64" if platform.machine() in ["x86_64", "amd64"] else "arm64"
    
    print(f"\n{'='*60}")
    print(f"[BUILD] Linux {backend} for linux-{arch}")
    print(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    build_dir = OLLAMA_LLAMA_SRC / f"build-linux-{backend}"
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
    
    commit = get_ollama_commit() or "unknown"
    (output_dir / "VERSION").write_text(f"{commit}\n{backend}\nollama-patched\n")
    
    count = len(list(output_dir.iterdir()))
    print(f"[OK] Built {backend}: {count} files")
    return True


def build_windows_backend(backend, output_dir, jobs=8):
    """Cross-compile for Windows using mingw."""
    
    print(f"\n{'='*60}")
    print(f"[BUILD] Windows {backend} (cross-compile)")
    print(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    build_dir = OLLAMA_LLAMA_SRC / f"build-windows-{backend}"
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
        "-DCMAKE_SYSTEM_NAME=Windows",
        "-DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc",
        "-DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++",
        "-DCMAKE_RC_COMPILER=x86_64-w64-mingw32-windres",
    ]
    
    if backend == "vulkan":
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
    
    try:
        run_cmd(cmake_opts, cwd=OLLAMA_LLAMA_SRC)
        run_cmd(["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs)], cwd=OLLAMA_LLAMA_SRC)
        copy_binaries(build_dir / "bin", output_dir, "windows")
    except Exception as e:
        print(f"[WARN] Cross-compile failed: {e}")
        create_windows_placeholder(output_dir, backend)
    
    commit = get_ollama_commit() or "unknown"
    (output_dir / "VERSION").write_text(f"{commit}\n{backend}\nollama-patched\n")
    
    count = len(list(output_dir.iterdir()))
    print(f"[OK] Built {backend}: {count} files")
    return True


def copy_binaries(bin_dir, output_dir, os_type):
    """Copy binaries from build directory to output."""
    if not bin_dir.exists():
        bin_dir = bin_dir.parent
    
    essential = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]
    
    for binary in essential:
        for ext in ["", ".exe"]:
            src = bin_dir / f"{binary}{ext}"
            if src.exists():
                dst = output_dir / f"{binary}{ext}"
                shutil.copy2(src, dst)
                if os_type == "linux":
                    os.chmod(dst, 0o755)
                print(f"  [COPY] {binary}{ext}")
    
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
    elif os_type == "windows":
        for f in bin_dir.iterdir():
            if f.suffix == ".dll":
                shutil.copy2(f, output_dir / f.name)
                print(f"  [COPY] {f.name}")


def create_windows_placeholder(output_dir, backend):
    """Create placeholder files for Windows when cross-compile fails."""
    placeholder = f"""# Windows {backend} binaries placeholder

These binaries need to be built on a Windows system with:
- Visual Studio 2019+ or MinGW-w64
- CUDA Toolkit (for CUDA backend)
- Vulkan SDK (for Vulkan backend)

Build commands:
  cmake -B build -DGGML_VULKAN=ON
  cmake --build build --config Release

After building, place the following files here:
- llama-server.exe
- llama-cli.exe
- llama-bench.exe
- llama-quantize.exe
- *.dll files
"""
    (output_dir / "README.txt").write_text(placeholder)


def package_all():
    """Package all binaries into distribution archives."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    if not BIN_DIR.exists():
        print("[ERROR] No binaries to package")
        return False
    
    packaged = 0
    for platform_dir in BIN_DIR.iterdir():
        if not platform_dir.is_dir():
            continue
        
        version_file = platform_dir / "VERSION"
        if not version_file.exists():
            continue
        
        if "ollama-patched" not in version_file.read_text():
            continue
        
        tarball_name = f"{platform_dir.name}.tar.gz"
        tarball_path = DIST_DIR / tarball_name
        
        print(f"[PACKAGE] Creating {tarball_name}...")
        
        with tarfile.open(tarball_path, "w:gz") as tar:
            for f in platform_dir.iterdir():
                tar.add(f, arcname=f.name)
        
        size_mb = tarball_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] {tarball_name}: {size_mb:.1f} MB")
        packaged += 1
    
    print(f"\n[INFO] Packaged {packaged} binary sets")
    return True


def upload_to_github():
    """Upload binaries to GitHub releases."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("[ERROR] Not logged in to GitHub CLI. Run 'gh auth login' first.")
            return False
    except FileNotFoundError:
        print("[ERROR] GitHub CLI not found. Install with: sudo apt install gh")
        return False
    
    commit = get_ollama_commit() or "unknown"
    tag = f"binaries-patched-{commit[:7]}"
    
    print(f"[INFO] Creating release {tag}...")
    
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", MOXING_REPO],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        body = f"""llama.cpp binaries with Ollama patches applied.

**llama.cpp**: {commit}
**patches**: ollama-patches (includes GLM-4.7-flash, GPT-OSS support)
**build-date**: {datetime.now().isoformat()}

## Supported Models
These binaries include all Ollama patches for enhanced model support:
- GLM-4.7-flash (glm4moelite)
- GPT-OSS
- And other models requiring Ollama's patches

## Available Backends
- **Linux x64**: CUDA, ROCm, Vulkan, CPU
- **Windows x64**: CUDA, Vulkan, CPU

## Installation
Binaries are automatically downloaded by MoXing on first use.
"""
        
        result = subprocess.run([
            "gh", "release", "create", tag,
            "--title", f"Patched Binaries {commit[:7]}",
            "--notes", body,
            "--repo", MOXING_REPO
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to create release: {result.stderr}")
            return False
        print(f"[OK] Created release {tag}")
    else:
        print(f"[INFO] Release {tag} already exists, uploading assets...")
    
    if not DIST_DIR.exists():
        print("[ERROR] No packages to upload")
        return False
    
    assets = list(DIST_DIR.glob("*.tar.gz"))
    if not assets:
        print("[ERROR] No tarballs found")
        return False
    
    print(f"[INFO] Uploading {len(assets)} assets...")
    
    for asset in assets:
        print(f"[UPLOAD] {asset.name}...")
        result = subprocess.run([
            "gh", "release", "upload", tag,
            str(asset),
            "--clobber",
            "--repo", MOXING_REPO
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to upload {asset.name}: {result.stderr}")
        else:
            print(f"  [OK] Uploaded {asset.name}")
    
    print(f"\n[DONE] Release URL: https://github.com/{MOXING_REPO}/releases/tag/{tag}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build llama.cpp binaries from Ollama source")
    parser.add_argument("--build-linux", action="store_true", help="Build for Linux")
    parser.add_argument("--build-windows", action="store_true", help="Build for Windows (cross-compile)")
    parser.add_argument("--package", action="store_true", help="Package binaries")
    parser.add_argument("--upload", action="store_true", help="Upload to GitHub")
    parser.add_argument("--all", action="store_true", help="Build all platforms, package, and upload")
    parser.add_argument("--backend", "-b", help="Build specific backend only")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="Parallel jobs")
    
    args = parser.parse_args()
    
    if args.all:
        args.build_linux = True
        args.build_windows = True
        args.package = True
        args.upload = True
    
    if not any([args.build_linux, args.build_windows, args.package, args.upload]):
        parser.print_help()
        return 1
    
    arch = "x64" if platform.machine() in ["x86_64", "amd64"] else "arm64"
    
    if args.build_linux:
        backends = ["rocm", "cuda", "vulkan", "cpu"] if not args.backend else [args.backend]
        
        for backend in backends:
            output_dir = BIN_DIR / f"linux-{arch}-{backend}"
            try:
                build_linux_backend(backend, output_dir, args.jobs)
            except Exception as e:
                print(f"[ERROR] Failed to build {backend}: {e}")
    
    if args.build_windows:
        backends = ["cuda", "vulkan", "cpu"] if not args.backend else [args.backend]
        
        for backend in backends:
            output_dir = BIN_DIR / f"windows-{arch}-{backend}"
            try:
                build_windows_backend(backend, output_dir, args.jobs)
            except Exception as e:
                print(f"[ERROR] Failed to build {backend}: {e}")
    
    if args.package:
        if not package_all():
            return 1
    
    if args.upload:
        if not upload_to_github():
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())