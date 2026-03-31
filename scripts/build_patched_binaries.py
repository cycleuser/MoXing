#!/usr/bin/env python3
"""
Build llama.cpp binaries with Ollama patches and upload to GitHub releases.

This script:
1. Applies all Ollama patches to llama.cpp
2. Builds binaries for CUDA, ROCm, Vulkan, CPU backends
3. Packages binaries for distribution
4. Uploads to GitHub releases

Usage:
    python scripts/build_patched_binaries.py --build
    python scripts/build_patched_binaries.py --package
    python scripts/build_patched_binaries.py --upload
    python scripts/build_patched_binaries.py --all
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
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "moxing" / "bin"
DIST_DIR = PROJECT_ROOT / "dist" / "binaries-patched"
OLLAMA_PATH = Path("/home/fred/Documents/GitHub/Others/ollama")
LLAMA_CPP_SRC = Path("/home/fred/Documents/GitHub/cycleuser/llama.cpp")

MOXING_REPO = "cycleuser/MoXing"


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
    run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=LLAMA_CPP_SRC)
    
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
        print(f"[INFO] Applying {patch_name}...")
        
        result = run_cmd(
            ["git", "apply", "--check", str(patch_file)],
            cwd=LLAMA_CPP_SRC,
            check=False
        )
        
        if result.returncode == 0:
            run_cmd(["git", "apply", str(patch_file)], cwd=LLAMA_CPP_SRC)
            applied += 1
            print(f"  [OK] Applied")
        else:
            result2 = run_cmd(
                ["git", "apply", "--check", "--reverse", str(patch_file)],
                cwd=LLAMA_CPP_SRC,
                check=False
            )
            if result2.returncode == 0:
                print(f"  [SKIP] Already applied")
            else:
                print(f"  [WARN] Failed to apply: {patch_name}")
                print(f"  {result.stderr[:200]}")
    
    print(f"[INFO] Applied {applied} new patches")
    return applied


def build_backend(backend, build_dir, output_dir, jobs=8):
    """Build llama.cpp for a specific backend."""
    print(f"\n[BUILD] Building {backend} backend...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    cmake_opts = [
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={output_dir}",
        "-DGGML_BUILD_TESTS=OFF",
        "-DGGML_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=ON",
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
    
    print(f"[CMAKE] Configuring {backend}...")
    run_cmd(["cmake", "-B", str(build_dir), "-S", str(LLAMA_CPP_SRC)] + cmake_opts)
    
    print(f"[BUILD] Compiling {backend}...")
    run_cmd(["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs)])
    
    bin_dir = build_dir / "bin"
    if not bin_dir.exists():
        bin_dir = build_dir
    
    essential_binaries = ["llama-server", "llama-cli", "llama-bench", "llama-quantize"]
    
    for binary in essential_binaries:
        src = bin_dir / binary
        if src.exists():
            dst = output_dir / binary
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            print(f"  [COPY] {binary}")
    
    for f in bin_dir.iterdir():
        if f.suffix in [".so", ".dylib"] or ".so." in f.name or ".dylib." in f.name:
            dst = output_dir / f.name
            if f.is_symlink():
                dst.symlink_to(f.readlink())
            else:
                shutil.copy2(f, dst)
            print(f"  [COPY] {f.name}")
        elif f.suffix == ".dll":
            shutil.copy2(f, output_dir / f.name)
            print(f"  [COPY] {f.name}")
    
    commit = get_ollama_llama_commit()
    version_file = output_dir / "VERSION"
    version_file.write_text(f"{commit}\n{backend}\nollama-patched\n")
    
    count = len(list(output_dir.iterdir()))
    print(f"[OK] Built {backend}: {count} files")
    
    return True


def build_all_backends():
    """Build all backends."""
    if not prepare_llama_cpp():
        return False
    
    apply_ollama_patches()
    
    backends = []
    
    if Path("/opt/rocm").exists() or shutil.which("hipcc"):
        backends.append("rocm")
    if shutil.which("nvcc"):
        backends.append("cuda")
    if os.environ.get("VULKAN_SDK") or shutil.which("vulkaninfo"):
        backends.append("vulkan")
    backends.append("cpu")
    
    print(f"[INFO] Building backends: {backends}")
    
    import platform
    os_name = "linux" if platform.system() == "Linux" else "darwin" if platform.system() == "Darwin" else "windows"
    arch = "x64" if platform.machine() in ["x86_64", "amd64"] else "arm64"
    platform_name = f"{os_name}-{arch}"
    
    for backend in backends:
        build_dir = LLAMA_CPP_SRC / f"build-{backend}-patched"
        output_dir = BIN_DIR / f"{platform_name}-{backend}-patched"
        
        try:
            build_backend(backend, build_dir, output_dir)
        except Exception as e:
            print(f"[ERROR] Failed to build {backend}: {e}")
    
    return True


def package_binaries():
    """Package binaries into tarballs for distribution."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    if not BIN_DIR.exists():
        print("[ERROR] No binaries to package")
        return False
    
    packaged = 0
    for platform_dir in BIN_DIR.iterdir():
        if not platform_dir.is_dir():
            continue
        if "patched" not in platform_dir.name:
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
    
    print(f"[INFO] Packaged {packaged} binary sets")
    return True


def upload_to_github():
    """Upload binaries to GitHub releases."""
    try:
        import subprocess
        
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
    
    commit = get_ollama_llama_commit() or "unknown"
    tag = f"binaries-patched-{commit[:7]}"
    
    print(f"[INFO] Creating release {tag}...")
    
    result = subprocess.run(
        ["gh", "release", "view", tag],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"[INFO] Release {tag} already exists, uploading assets...")
    else:
        commit = get_ollama_llama_commit() or "unknown"
        body = f"""llama.cpp binaries with Ollama patches applied.

llama.cpp: {commit}
patches: ollama-patches
build-date: {datetime.now().isoformat()}

These binaries include all Ollama patches for enhanced model support.
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
    parser = argparse.ArgumentParser(description="Build and upload patched llama.cpp binaries")
    parser.add_argument("--build", action="store_true", help="Build binaries")
    parser.add_argument("--package", action="store_true", help="Package binaries")
    parser.add_argument("--upload", action="store_true", help="Upload to GitHub")
    parser.add_argument("--all", action="store_true", help="Build, package, and upload")
    parser.add_argument("--backend", "-b", help="Build specific backend")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="Parallel jobs")
    
    args = parser.parse_args()
    
    if args.all:
        args.build = args.package = args.upload = True
    
    if not any([args.build, args.package, args.upload]):
        parser.print_help()
        return 1
    
    if args.build:
        if args.backend:
            import platform
            os_name = "linux" if platform.system() == "Linux" else "darwin"
            arch = "x64" if platform.machine() in ["x86_64", "amd64"] else "arm64"
            platform_name = f"{os_name}-{arch}"
            
            if not prepare_llama_cpp():
                return 1
            apply_ollama_patches()
            build_dir = LLAMA_CPP_SRC / f"build-{args.backend}-patched"
            output_dir = BIN_DIR / f"{platform_name}-{args.backend}-patched"
            build_backend(args.backend, build_dir, output_dir, args.jobs)
        else:
            if not build_all_backends():
                return 1
    
    if args.package:
        if not package_binaries():
            return 1
    
    if args.upload:
        if not upload_to_github():
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())