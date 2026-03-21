#!/usr/bin/env python3
"""
Upload binaries to GitHub releases at:
    https://github.com/cycleuser/MoXing/releases/tag/binaries

Usage:
    # Package binaries
    python scripts/upload_binaries.py --package
    
    # Upload to GitHub (requires gh auth login)
    python scripts/upload_binaries.py --upload

The binaries will be downloaded by moxing from:
    https://github.com/cycleuser/MoXing/releases/tag/binaries
"""

import os
import sys
import argparse
import tarfile
import zipfile
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
BIN_DIR = REPO_ROOT / "moxing" / "bin"
DIST_DIR = REPO_ROOT / "dist" / "binaries"
MOXING_REPO = "cycleuser/MoXing"
RELEASE_TAG = "binaries"


def package_binaries():
    """Package each platform's binaries into tar.gz/zip files."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    for platform_dir in BIN_DIR.iterdir():
        if not platform_dir.is_dir():
            continue
        
        platform_name = platform_dir.name
        print(f"\nPackaging: {platform_name}")
        
        if "windows" in platform_name:
            archive_path = DIST_DIR / f"{platform_name}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in platform_dir.iterdir():
                    if file_path.is_file():
                        zf.write(file_path, file_path.name)
                        print(f"  {file_path.name}")
                    elif file_path.is_symlink():
                        print(f"  {file_path.name} -> {os.readlink(file_path)}")
        else:
            archive_path = DIST_DIR / f"{platform_name}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tf:
                for file_path in platform_dir.iterdir():
                    if file_path.is_file():
                        tf.add(file_path, file_path.name)
                        print(f"  {file_path.name}")
                    elif file_path.is_symlink():
                        info = tarfile.TarInfo(name=file_path.name)
                        info.type = tarfile.SYMTYPE
                        info.linkname = os.readlink(file_path)
                        tf.addfile(info)
                        print(f"  {file_path.name} -> {info.linkname}")
        
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"  Created: {archive_path.name} ({size_mb:.1f} MB)")
    
    print(f"\nAll packages created in: {DIST_DIR}")


def upload_to_github():
    """Upload binaries to GitHub releases using gh CLI."""
    
    if not DIST_DIR.exists():
        print("No packages found. Run with --package first.")
        return
    
    result = subprocess.run(["gh", "--version"], capture_output=True)
    if result.returncode != 0:
        print("Error: GitHub CLI (gh) not installed.")
        print("Install from: https://cli.github.com/")
        return
    
    print(f"\nUploading to release: {RELEASE_TAG}")
    
    # Check if release exists
    result = subprocess.run(
        ["gh", "release", "view", RELEASE_TAG, "--repo", MOXING_REPO],
        capture_output=True
    )
    
    if result.returncode != 0:
        # Create release
        result = subprocess.run([
            "gh", "release", "create", RELEASE_TAG,
            "--repo", MOXING_REPO,
            "--title", "Binaries",
            "--notes", "Pre-built llama.cpp binaries for MoXing",
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error creating release: {result.stderr}")
            return
        print(f"Created release: {RELEASE_TAG}")
    else:
        print(f"Release {RELEASE_TAG} exists. Uploading assets...")
    
    # Upload assets
    for archive_path in DIST_DIR.iterdir():
        if archive_path.suffix in (".zip", ".gz"):
            print(f"\nUploading: {archive_path.name}")
            result = subprocess.run([
                "gh", "release", "upload", RELEASE_TAG,
                str(archive_path),
                "--repo", MOXING_REPO,
                "--clobber"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  Error: {result.stderr}")
            else:
                print(f"  Done!")
    
    print(f"\nRelease URL: https://github.com/{MOXING_REPO}/releases/tag/{RELEASE_TAG}")


def main():
    parser = argparse.ArgumentParser(description="Package and upload binaries")
    parser.add_argument("--package", "-p", action="store_true", help="Package binaries")
    parser.add_argument("--upload", "-u", action="store_true", help="Upload to GitHub")
    
    args = parser.parse_args()
    
    if args.package:
        package_binaries()
    
    if args.upload:
        upload_to_github()
    
    if not args.package and not args.upload:
        parser.print_help()


if __name__ == "__main__":
    main()