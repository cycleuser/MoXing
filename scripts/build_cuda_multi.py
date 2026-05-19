#!/usr/bin/env python3
"""
独立脚本：检测所有已安装的CUDA版本，为每个版本编译二进制文件。

完全不修改已有的moxing代码，只做编译+安装到缓存。

Usage:
    python scripts/build_cuda_multi.py                          # 检测所有CUDA版本并编译
    python scripts/build_cuda_multi.py --list                   # 只列出检测到的CUDA版本
    python scripts/build_cuda_multi.py 12                       # 只编译CUDA 12
    python scripts/build_cuda_multi.py --package                # 编译后打包为tar.gz
    python scripts/build_cuda_multi.py --upload                 # 编译后打包并上传到GitHub Release

输出目录:
    ~/.cache/moxing/binaries/linux-x64-cuda12/
    ~/.cache/moxing/binaries/linux-x64-cuda13/
"""

import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

CACHE_DIR = Path.home() / ".cache" / "moxing" / "binaries"
LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent / "llama.cpp"
DIST_DIR = Path(__file__).resolve().parent.parent / "dist" / "binaries"
MOXING_REPO = "cycleuser/MoXing"
RELEASE_TAG = "binaries"

ESSENTIAL = [
    "llama-server",
    "llama-cli",
    "llama-mtmd-cli",
    "llama-bench",
    "llama-quantize",
]


def detect_cuda_versions() -> Dict[str, str]:
    """
    检测所有已安装的CUDA版本。

    Returns:
        Dict[cuda_label, toolkit_path]
        例如: {"cuda12": "/usr/local/cuda-12.8", "cuda13": "/usr/local/cuda-13.2"}
    """
    versions: Dict[str, str] = {}

    patterns = [
        Path("/usr/local/cuda"),
        *sorted(Path("/usr/local").glob("cuda-*")),
    ]

    for path in patterns:
        if not path.is_dir():
            continue
        nvcc = path / "bin" / "nvcc"
        if not nvcc.exists():
            continue

        try:
            result = subprocess.run(
                [str(nvcc), "--version"],
                capture_output=True, text=True, timeout=10,
            )
            m = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if m:
                major, minor = m.group(1), m.group(2)
                label = f"cuda{major}{minor}"
                versions[label] = str(path.resolve())
                print(f"  Found {label}: {path}")
        except Exception:
            pass

    return versions


def detect_gpu_arch() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().replace("\n", ";")
    except Exception:
        pass
    return "89"


def build_cuda(version_label: str, cuda_path: str, gpu_arch: str, jobs: int = 0) -> bool:
    """
    用指定的CUDA toolkit编译llama.cpp二进制文件。

    完全不依赖moxing的内部API，纯cmake + make流程。
    """
    cache_dir = CACHE_DIR / f"linux-x64-{version_label}"
    build_dir = LLAMA_CPP_DIR / f"build-{version_label}"

    print(f"\n{'=' * 60}")
    print(f"  Building: {version_label}")
    print(f"  CUDA path: {cuda_path}")
    print(f"  GPU arch: {gpu_arch}")
    print(f"  Output: {cache_dir}")
    print(f"{'=' * 60}")

    if not (LLAMA_CPP_DIR / "CMakeLists.txt").exists():
        print(f"[ERROR] llama.cpp source not found at {LLAMA_CPP_DIR}")
        return False

    if build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_args = [
        "cmake",
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=OFF",
        "-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={gpu_arch}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_path}",
    ]

    print("\nCMake configure:")
    for a in cmake_args:
        print(f"  {a}")

    result = subprocess.run(cmake_args, cwd=LLAMA_CPP_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] CMake failed for {version_label}")
        return False

    actual_jobs = jobs if jobs > 0 else (os.cpu_count() or 8)
    print(f"\nBuilding with {actual_jobs} jobs...")

    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", str(actual_jobs)],
        cwd=LLAMA_CPP_DIR,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Build failed for {version_label}")
        return False

    print(f"\nBuild complete: {version_label}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for f in cache_dir.iterdir():
        if f.is_file():
            f.unlink()

    copied: List[str] = []
    strip = shutil.which("strip")
    bin_dir = build_dir / "bin"
    if bin_dir.exists():
        for f in bin_dir.iterdir():
            if not f.is_file():
                continue
            if any(f.name.startswith(b) for b in ESSENTIAL):
                shutil.copy2(f, cache_dir / f.name)
                if strip:
                    subprocess.run([strip, str(cache_dir / f.name)], capture_output=True)
                os.chmod(cache_dir / f.name, 0o755)
                copied.append(f.name)

    for lib in build_dir.glob("*.so*"):
        if lib.is_file():
            shutil.copy2(lib, cache_dir / lib.name)
            copied.append(lib.name)

    version_file = cache_dir / "VERSION"
    version_file.write_text(f"{version_label}-local\n{version_label}\n")

    if copied:
        print(f"Installed {len(copied)} files to: {cache_dir}")
        for name in sorted(copied):
            print(f"  {name}")
    else:
        print("[ERROR] No binaries found")
        return False

    shutil.rmtree(build_dir)
    return True


def package_cuda(version_label: str) -> Optional[Path]:
    cache_dir = CACHE_DIR / f"linux-x64-{version_label}"
    server = cache_dir / "llama-server"
    if not server.exists():
        print(f"[ERROR] llama-server not found for {version_label}")
        return None

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    archive = DIST_DIR / f"linux-x64-{version_label}.tar.gz"

    with tarfile.open(archive, "w:gz") as tf:
        for f in sorted(cache_dir.iterdir()):
            if f.is_file():
                tf.add(f, f.name)

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"  {archive.name} ({size_mb:.1f} MB)")
    return archive


def upload_to_release(tag: str = RELEASE_TAG):
    if not DIST_DIR.exists() or not list(DIST_DIR.iterdir()):
        print(f"[ERROR] No packages in {DIST_DIR}")
        return

    if subprocess.run(["gh", "--version"], capture_output=True).returncode != 0:
        print("[ERROR] GitHub CLI (gh) not installed")
        print("  Install: https://cli.github.com/")
        return

    print(f"\nUploading to: github.com/{MOXING_REPO}/releases/tag/{tag}")

    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", MOXING_REPO],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"Creating release: {tag}")
        result = subprocess.run(
            [
                "gh", "release", "create", tag,
                "--repo", MOXING_REPO,
                "--title", f"Binaries ({tag})",
                "--notes", "CUDA version-specific binaries for MoXing",
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] Create release failed: {result.stderr}")
            return

    for archive in sorted(DIST_DIR.glob("linux-x64-cuda*.tar.gz")):
        size_mb = archive.stat().st_size / (1024 * 1024)
        print(f"\nUploading: {archive.name} ({size_mb:.1f} MB)")
        result = subprocess.run(
            [
                "gh", "release", "upload", tag,
                str(archive),
                "--repo", MOXING_REPO,
                "--clobber",
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Error: {result.stderr.strip()}")
        else:
            print("  Done")

    print(f"\nRelease: https://github.com/{MOXING_REPO}/releases/tag/{tag}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="为所有已安装的CUDA版本编译llama.cpp二进制文件"
    )
    parser.add_argument(
        "versions", nargs="*",
        help="要编译的CUDA主版本号 (如 12 13)。不指定则编译所有检测到的版本",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="只列出检测到的CUDA版本",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=0,
        help="并行编译任务数 (默认: CPU核心数)",
    )
    parser.add_argument(
        "--package", "-p", action="store_true",
        help="编译后打包为tar.gz",
    )
    parser.add_argument(
        "--upload", "-u", action="store_true",
        help="编译打包后上传到GitHub Release",
    )

    args = parser.parse_args()

    all_versions = detect_cuda_versions()
    if not all_versions:
        print("[ERROR] No CUDA installations found")
        print("  检查: /usr/local/cuda*")
        sys.exit(1)

    print(f"\nDetected CUDA versions: {len(all_versions)}")
    for label, path in sorted(all_versions.items()):
        print(f"  {label}: {path}")

    if args.list:
        return

    gpu_arch = detect_gpu_arch()
    print(f"\nGPU compute capability: {gpu_arch}")

    if args.versions:
        target_versions = {
            k: v for k, v in all_versions.items()
            if any(k.startswith(f"cuda{v}") for v in args.versions)
        }
        if not target_versions:
            print(f"[ERROR] No matching CUDA versions for: {args.versions}")
            print(f"  Available: {list(all_versions.keys())}")
            sys.exit(1)
    else:
        target_versions = all_versions

    print(f"\nWill build: {', '.join(sorted(target_versions.keys()))}")

    results: Dict[str, bool] = {}
    for label, path in sorted(target_versions.items()):
        results[label] = build_cuda(label, path, gpu_arch, args.jobs)

    print(f"\n{'=' * 60}")
    print("  Build Summary")
    print(f"{'=' * 60}")
    for label, ok in results.items():
        print(f"  {'[OK]' if ok else '[FAILED]'} {label}")

    if args.package or args.upload:
        print(f"\n{'=' * 60}")
        print("  Packaging")
        print(f"{'=' * 60}")
        for label, ok in results.items():
            if ok:
                package_cuda(label)

    if args.upload:
        upload_to_release()


if __name__ == "__main__":
    main()
