#!/usr/bin/env python3
"""
完整构建+发布流水线：检测llama.cpp更新，构建所有后端，上传到GitHub Release。

Usage:
    python scripts/release_binaries.py                    # 检查更新，有则构建+上传
    python scripts/release_binaries.py --force            # 强制重建（跳过版本检查）
    python scripts/release_binaries.py --check-only        # 只检查更新状态
    python scripts/release_binaries.py --build-only        # 只构建不发布
    python scripts/release_binaries.py --upload-only       # 只上传已有包
    python scripts/release_binaries.py --tag v0.1.37       # 指定版本tag（默认用binaries）

依赖:
    pip install httpx        # 用于GitHub API
    gh auth login            # 用于上传（只需一次）

上传目标:
    https://github.com/cycleuser/MoXing/releases/tag/binaries

运行时下载路径对应:
    https://github.com/cycleuser/MoXing/releases/download/binaries/linux-x64-cuda.tar.gz
"""

import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DIST_DIR = REPO_ROOT / "dist" / "binaries"
VERSION_CACHE_FILE = Path.home() / ".cache" / "moxing" / ".llama-cpp-version"
MOXING_REPO = "cycleuser/MoXing"
LLAMA_CPP_REPO = "ggml-org/llama.cpp"
RELEASE_TAG = "binaries"

ALL_BACKENDS = ["cuda", "vulkan", "rocm", "cpu"]


def check_gh_cli() -> bool:
    return subprocess.run(["gh", "--version"], capture_output=True).returncode == 0


def get_latest_llama_cpp_version() -> Optional[str]:
    try:
        import httpx

        url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "moxing-release"}
        token = os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))
        if token:
            headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json().get("tag_name")
    except Exception as e:
        print(f"[ERROR] Failed to get latest llama.cpp version: {e}")
        return None


def get_cached_llama_cpp_version() -> Optional[str]:
    if VERSION_CACHE_FILE.exists():
        return VERSION_CACHE_FILE.read_text().strip()
    return None


def save_cached_version(version: str):
    VERSION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_CACHE_FILE.write_text(version.strip())


def check_for_update() -> Tuple[bool, Optional[str], Optional[str]]:
    latest = get_latest_llama_cpp_version()
    if not latest:
        return False, None, None

    cached = get_cached_llama_cpp_version()
    if cached and cached == latest:
        print(f"llama.cpp is up to date: {latest}")
        return False, cached, latest

    if cached:
        print(f"llama.cpp update available: {cached} -> {latest}")
    else:
        print(f"llama.cpp latest version: {latest} (no cache)")

    return True, cached, latest


def detect_available_backends() -> List[str]:
    available = []
    if (
        shutil.which("nvcc")
        or Path("/usr/local/cuda").exists()
        or Path("/usr/local/cuda-13").exists()
    ):
        available.append("cuda")
    if shutil.which("vulkaninfo") or Path("/usr/include/vulkan/vulkan.h").exists():
        available.append("vulkan")
    if shutil.which("hipconfig") or Path("/opt/rocm").exists():
        available.append("rocm")
    available.append("cpu")
    return available


def ensure_llama_cpp_source() -> Path:
    llama_dir = REPO_ROOT / "llama.cpp"
    if not (llama_dir / "CMakeLists.txt").exists():
        print(f"Cloning llama.cpp into {llama_dir}...")
        clone_url = f"https://github.com/{LLAMA_CPP_REPO}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(llama_dir)],
            check=True,
        )
        print("Cloned llama.cpp")
    else:
        print(f"Using existing source: {llama_dir}")

        result = subprocess.run(
            ["git", "-C", str(llama_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip()
        if branch != "master" or branch != "main":
            subprocess.run(["git", "-C", str(llama_dir), "stash"], capture_output=True)
            subprocess.run(["git", "-C", str(llama_dir), "checkout", "master"], capture_output=True)
            subprocess.run(["git", "-C", str(llama_dir), "checkout", "main"], capture_output=True)

        print("Pulling latest llama.cpp...")
        subprocess.run(["git", "-C", str(llama_dir), "fetch", "--tags"], capture_output=True)
        latest_tag = subprocess.run(
            ["git", "-C", str(llama_dir), "describe", "--tags", "--abbrev=0"],
            capture_output=True, text=True,
        ).stdout.strip()
        if latest_tag:
            subprocess.run(
                ["git", "-C", str(llama_dir), "checkout", latest_tag],
                capture_output=True,
            )
            print(f"Checked out: {latest_tag}")
        else:
            subprocess.run(
                ["git", "-C", str(llama_dir), "pull", "origin", "master"],
                capture_output=True,
            )
            print("Pulled latest master")

    return llama_dir


def build_backend(backend: str, llama_dir: Path, jobs: int = 0) -> bool:
    from moxing.binaries import CACHE_DIR, PlatformDetector
    from moxing.cli.system import (
        _build_cmake_args,
        _check_build_prerequisites,
        _copy_build_outputs,
        _detect_gpu_arch,
        _get_install_hint,
    )

    platform_name = PlatformDetector.get_platform_name()
    cache_dir = CACHE_DIR / f"{platform_name}-{backend}"

    print(f"\n{'=' * 60}")
    print(f"  Building: {backend}")
    print(f"  Platform: {platform_name}")
    print(f"  Output:   {cache_dir}")
    print(f"{'=' * 60}")

    missing = _check_build_prerequisites(backend)
    if missing:
        print(f"\n[WARNING] Missing dependencies for {backend}:")
        for m in missing:
            print(f"  x {m}")
        print(f"\nInstall: {_get_install_hint(backend)}")
        return False

    gpu_arch = _detect_gpu_arch(backend)
    if gpu_arch and backend in ("cuda", "rocm"):
        print(f"GPU architectures: {gpu_arch}")

    build_dir = llama_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_args = _build_cmake_args(build_dir, backend, gpu_arch)
    print("\nCMake configure:")
    for a in cmake_args:
        print(f"  {a}")

    result = subprocess.run(cmake_args, cwd=llama_dir)
    if result.returncode != 0:
        print(f"\n[ERROR] CMake configure failed for {backend}")
        return False

    actual_jobs = jobs if jobs > 0 else (os.cpu_count() or 8)
    print(f"\nBuilding with {actual_jobs} jobs...")
    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(actual_jobs)]
    result = subprocess.run(build_cmd, cwd=llama_dir)
    if result.returncode != 0:
        print(f"\n[ERROR] Build failed for {backend}")
        return False

    print(f"\nBuild complete: {backend}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    for f in cache_dir.iterdir():
        if f.is_file():
            f.unlink()

    copied = _copy_build_outputs(build_dir, cache_dir, backend)

    if copied:
        print(f"Installed {len(copied)} files to: {cache_dir}")
        for name in sorted(copied):
            print(f"  {name}")
    else:
        print("[ERROR] No binaries to copy")
        shutil.rmtree(build_dir)
        return False

    (cache_dir / "VERSION").write_text(f"local-build-{backend}\n{backend}\n")
    shutil.rmtree(build_dir)
    return True


def package_backend(backend: str) -> Optional[Path]:
    from moxing.binaries import CACHE_DIR, PlatformDetector

    platform_name = PlatformDetector.get_platform_name()
    cache_dir = CACHE_DIR / f"{platform_name}-{backend}"

    if not cache_dir.exists():
        print(f"[ERROR] No cache for {backend}")
        return None

    suffix = ".exe" if sys.platform == "win32" else ""
    server = cache_dir / f"llama-server{suffix}"
    if not server.exists():
        print(f"[ERROR] llama-server not found for {backend}")
        return None

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = DIST_DIR / f"{platform_name}-{backend}.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tf:
        for file_path in sorted(cache_dir.iterdir()):
            if file_path.is_file():
                tf.add(file_path, file_path.name)
            elif file_path.is_symlink():
                info = tarfile.TarInfo(name=file_path.name)
                info.type = tarfile.SYMTYPE
                info.linkname = os.readlink(file_path)
                tf.addfile(info)

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"  Packaged: {archive_path.name} ({size_mb:.1f} MB)")
    return archive_path


def upload_to_github(tag: str = RELEASE_TAG):
    if not DIST_DIR.exists() or not list(DIST_DIR.iterdir()):
        print(f"[ERROR] No packages in {DIST_DIR}. Run --build first.")
        return

    if not check_gh_cli():
        print("[ERROR] GitHub CLI (gh) not installed.")
        print("  Install: https://cli.github.com/")
        print("  Then run: gh auth login")
        return

    print(f"\nUploading to: github.com/{MOXING_REPO}/releases/tag/{tag}")

    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", MOXING_REPO],
        capture_output=True,
    )

    if result.returncode != 0:
        print(f"Creating release: {tag}")
        notes = (
            f"llama.cpp binaries for MoXing\n\n"
            f"Build date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        )
        result = subprocess.run(
            [
                "gh", "release", "create", tag,
                "--repo", MOXING_REPO,
                "--title", f"Binaries ({tag})",
                "--notes", notes,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] Failed to create release: {result.stderr}")
            return
        print(f"Created release: {tag}")

    for archive_path in sorted(DIST_DIR.iterdir()):
        if archive_path.suffix not in (".gz", ".zip", ".tar"):
            continue
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"\nUploading: {archive_path.name} ({size_mb:.1f} MB)")
        result = subprocess.run(
            [
                "gh", "release", "upload", tag,
                str(archive_path),
                "--repo", MOXING_REPO,
                "--clobber",
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Error: {result.stderr.strip()}")
        else:
            print("  Done")

    url = f"https://github.com/{MOXING_REPO}/releases/tag/{tag}"
    print(f"\nRelease: {url}")
    print(f"Download URL prefix: https://github.com/{MOXING_REPO}/releases/download/{tag}/")


def build_all_backends(
    backends: List[str],
    jobs: int = 0,
) -> Dict[str, bool]:
    llama_dir = ensure_llama_cpp_source()
    results = {}

    for backend in backends:
        results[backend] = build_backend(backend, llama_dir, jobs)

    print(f"\n{'=' * 60}")
    print("  Build Summary")
    print(f"{'=' * 60}")
    for b, ok in results.items():
        status = "[OK]" if ok else "[FAILED]"
        print(f"  {status} {b}")

    return results


def package_all(backends: List[str]) -> Dict[str, Optional[Path]]:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for backend in backends:
        results[backend] = package_backend(backend)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build all backends and upload to GitHub Release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/release_binaries.py                     # Check update, build, upload
  python scripts/release_binaries.py --force             # Force rebuild
  python scripts/release_binaries.py --check-only         # Just check status
  python scripts/release_binaries.py --build-only         # Build only (no upload)
  python scripts/release_binaries.py --upload-only        # Upload existing packages
  python scripts/release_binaries.py --tag v0.1.37       # Release as version tag
  python scripts/release_binaries.py cuda vulkan          # Only these backends
        """,
    )
    parser.add_argument(
        "backends", nargs="*",
        help="Backends to build (default: auto-detect all available)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=0,
        help="Parallel build jobs (default: CPU count)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force rebuild even if llama.cpp is up to date",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check if llama.cpp has updates",
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="Build and package but don't upload",
    )
    parser.add_argument(
        "--upload-only", action="store_true",
        help="Only upload existing packages (skip build)",
    )
    parser.add_argument(
        "--tag", type=str, default=RELEASE_TAG,
        help=f"GitHub release tag (default: {RELEASE_TAG})",
    )
    parser.add_argument(
        "--no-package", action="store_true",
        help="Skip packaging step",
    )
    args = parser.parse_args()

    print("MoXing Binary Release Pipeline")
    print(f"  Repository: {MOXING_REPO}")
    print(f"  Release tag: {args.tag}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.upload_only:
        if not args.no_package:
            backends = args.backends or detect_available_backends()
            package_all(backends)
        upload_to_github(args.tag)
        return

    if args.check_only:
        has_update, cached, latest = check_for_update()
        if has_update:
            print(f"\nUpdate available: {cached} -> {latest}")
            sys.exit(1)
        else:
            print(f"\nUp to date: {latest}")
            sys.exit(0)

    if not args.force:
        has_update, cached, latest = check_for_update()
        if not has_update and cached:
            print("\nNo update needed. Use --force to rebuild.")
            return
    else:
        latest = get_latest_llama_cpp_version()
        print(f"Forcing rebuild. llama.cpp version: {latest}")

    backends = args.backends if args.backends else detect_available_backends()
    backends = [b for b in backends if b in ALL_BACKENDS]
    if not backends:
        print("[ERROR] No backends to build")
        sys.exit(1)
    print(f"Backends: {', '.join(backends)}\n")

    results = build_all_backends(backends, args.jobs)

    success_backends = [b for b, ok in results.items() if ok]
    if not success_backends:
        print("\n[ERROR] All builds failed")
        sys.exit(1)

    if args.no_package:
        if latest:
            save_cached_version(latest)
        return

    print(f"\n{'=' * 60}")
    print("  Packaging")
    print(f"{'=' * 60}")
    package_all(success_backends)

    if latest:
        save_cached_version(latest)
        print(f"\nSaved version cache: {latest}")

    if not args.build_only:
        upload_to_github(args.tag)


if __name__ == "__main__":
    main()
