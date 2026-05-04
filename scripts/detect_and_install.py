#!/usr/bin/env python3
"""
Automatic system detection and installation script for moxing-server.

This script detects:
- Operating system and version
- CPU architecture
- GPU hardware and available backends
- Missing dependencies

And provides:
- Automated installation commands
- Recommended configuration
- Troubleshooting suggestions
"""

import os
import sys
import platform
import subprocess
import shutil
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

# ANSI colors
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def color(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}[OK] {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}[X] {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}[!] {text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.BLUE}[i] {text}{Colors.RESET}")


class OSType(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class BackendType(Enum):
    CUDA = "cuda"
    VULKAN = "vulkan"
    ROCM = "rocm"
    METAL = "metal"
    CPU = "cpu"


@dataclass
class GPUDevice:
    name: str
    vendor: str
    memory_mb: int
    backend: BackendType
    driver_version: str = ""
    
    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024


@dataclass
class SystemInfo:
    os_type: OSType
    os_name: str
    os_version: str
    arch: str
    cpu_cores: int
    cpu_name: str
    total_ram_gb: float
    gpus: List[GPUDevice] = field(default_factory=list)
    python_version: str = ""
    recommended_backend: BackendType = BackendType.CPU
    missing_deps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SystemDetector:
    """Detect system configuration and recommend setup."""
    
    def __init__(self):
        self.info = SystemInfo(
            os_type=self._detect_os(),
            os_name=platform.system(),
            os_version=platform.version(),
            arch=platform.machine(),
            cpu_cores=os.cpu_count() or 1,
            cpu_name=self._get_cpu_name(),
            total_ram_gb=self._get_total_ram(),
            python_version=platform.python_version(),
        )
    
    def _detect_os(self) -> OSType:
        system = platform.system().lower()
        if system == "windows":
            return OSType.WINDOWS
        elif system == "linux":
            return OSType.LINUX
        elif system == "darwin":
            return OSType.MACOS
        return OSType.UNKNOWN
    
    def _get_cpu_name(self) -> str:
        try:
            if self._detect_os() == OSType.WINDOWS:
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True
                )
                lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
                return lines[-1] if len(lines) > 1 else "Unknown CPU"
            elif self._detect_os() == OSType.LINUX:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif self._detect_os() == OSType.MACOS:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                return result.stdout.strip()
        except:
            pass
        return "Unknown CPU"
    
    def _get_total_ram(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except:
            return 0.0
    
    def detect_gpus(self) -> List[GPUDevice]:
        """Detect all available GPUs."""
        gpus = []
        
        # Detect NVIDIA (CUDA)
        gpus.extend(self._detect_nvidia())
        
        # Detect AMD (ROCm/Windows)
        gpus.extend(self._detect_amd())
        
        # Detect Intel
        gpus.extend(self._detect_intel())
        
        # Detect Apple Silicon
        gpus.extend(self._detect_apple_silicon())
        
        # Detect Vulkan devices
        gpus.extend(self._detect_vulkan())
        
        # Remove duplicates based on name
        seen = set()
        unique_gpus = []
        for gpu in gpus:
            if gpu.name not in seen:
                seen.add(gpu.name)
                unique_gpus.append(gpu)
        
        self.info.gpus = unique_gpus
        return unique_gpus
    
    def _detect_nvidia(self) -> List[GPUDevice]:
        gpus = []
        
        # Try nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            name = parts[0]
                            mem_str = parts[1]
                            driver = parts[2] if len(parts) > 2 else ""
                            
                            mem_match = re.search(r"(\d+)", mem_str)
                            memory_mb = int(mem_match.group(1)) if mem_match else 0
                            
                            gpus.append(GPUDevice(
                                name=name,
                                vendor="nvidia",
                                memory_mb=memory_mb,
                                backend=BackendType.CUDA,
                                driver_version=driver
                            ))
            except Exception as e:
                pass
        
        return gpus
    
    def _detect_amd(self) -> List[GPUDevice]:
        gpus = []
        
        # Try rocm-smi on Linux
        if self.info.os_type == OSType.LINUX and shutil.which("rocm-smi"):
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                    capture_output=True, text=True, timeout=10
                )
                # Parse ROCm output
                lines = result.stdout.split("\n")
                for i, line in enumerate(lines):
                    if "Card" in line and "series" in line.lower():
                        name = line.split(":")[-1].strip()
                        gpus.append(GPUDevice(
                            name=name,
                            vendor="amd",
                            memory_mb=8192,  # Default, would need better parsing
                            backend=BackendType.ROCM
                        ))
            except:
                pass
        
        # Windows: Check for AMD in Vulkan devices
        return gpus
    
    def _detect_intel(self) -> List[GPUDevice]:
        gpus = []
        
        # Intel Arc detection via vulkaninfo or Windows
        if self.info.os_type == OSType.WINDOWS:
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True
                )
                for line in result.stdout.split("\n"):
                    if "intel" in line.lower() and "arc" in line.lower():
                        gpus.append(GPUDevice(
                            name=line.strip(),
                            vendor="intel",
                            memory_mb=8192,  # Approximate
                            backend=BackendType.VULKAN
                        ))
            except:
                pass
        
        return gpus
    
    def _detect_apple_silicon(self) -> List[GPUDevice]:
        gpus = []
        
        if self.info.os_type != OSType.MACOS:
            return gpus
        
        # Check for Apple Silicon
        if self.info.arch == "arm64":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                cpu_name = result.stdout.strip()
                
                # Determine chip series
                if "M1" in cpu_name or "M2" in cpu_name or "M3" in cpu_name or "M4" in cpu_name:
                    # Get memory (unified)
                    mem_result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True
                    )
                    total_mem = int(mem_result.stdout.strip()) / (1024 ** 3)
                    
                    gpus.append(GPUDevice(
                        name=cpu_name,
                        vendor="apple",
                        memory_mb=int(total_mem * 1024),
                        backend=BackendType.METAL
                    ))
            except:
                pass
        
        return gpus
    
    def _detect_vulkan(self) -> List[GPUDevice]:
        gpus = []
        
        # Try vulkaninfo
        if shutil.which("vulkaninfo"):
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True, text=True, timeout=30
                )
                
                current_device = None
                for line in result.stdout.split("\n"):
                    if "deviceName" in line:
                        name = line.split("=")[-1].strip().strip(",")
                        current_device = name
                    if "driverVersion" in line and current_device:
                        vendor = "unknown"
                        name_lower = current_device.lower()
                        if "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
                            vendor = "nvidia"
                        elif "amd" in name_lower or "radeon" in name_lower:
                            vendor = "amd"
                        elif "intel" in name_lower:
                            vendor = "intel"
                        
                        # Check if already in list
                        if not any(g.name == current_device for g in self.info.gpus):
                            gpus.append(GPUDevice(
                                name=current_device,
                                vendor=vendor,
                                memory_mb=0,  # Vulkan doesn't report VRAM easily
                                backend=BackendType.VULKAN
                            ))
                        current_device = None
            except:
                pass
        
        return gpus
    
    def recommend_backend(self) -> BackendType:
        """Recommend the best backend based on detected hardware."""
        if not self.info.gpus:
            self.detect_gpus()
        
        # Priority order
        priority = {
            BackendType.CUDA: 100,
            BackendType.METAL: 90,
            BackendType.ROCM: 85,
            BackendType.VULKAN: 70,
            BackendType.CPU: 0,
        }
        
        best_backend = BackendType.CPU
        best_score = 0
        
        for gpu in self.info.gpus:
            score = priority.get(gpu.backend, 0)
            # Bonus for more VRAM
            if gpu.memory_gb >= 8:
                score += 10
            elif gpu.memory_gb >= 6:
                score += 5
            
            if score > best_score:
                best_score = score
                best_backend = gpu.backend
        
        self.info.recommended_backend = best_backend
        return best_backend
    
    def check_dependencies(self) -> List[str]:
        """Check for missing dependencies."""
        missing = []
        
        # Check Python version
        py_version = tuple(map(int, platform.python_version().split(".")[:2]))
        if py_version < (3, 8):
            missing.append(f"Python 3.8+ required (found {platform.python_version()})")
        
        # Check pip
        if not shutil.which("pip") and not shutil.which("pip3"):
            missing.append("pip not found")
        
        # Platform-specific checks
        if self.info.os_type == OSType.LINUX:
            # Check Vulkan
            if not shutil.which("vulkaninfo"):
                self.info.warnings.append("Vulkan tools not installed. Run: sudo apt install vulkan-tools")
            
            # Check for GPU drivers
            if any(g.vendor == "nvidia" for g in self.info.gpus):
                if not shutil.which("nvidia-smi"):
                    missing.append("NVIDIA driver not installed")
            
            if any(g.vendor == "amd" for g in self.info.gpus):
                if not os.path.exists("/opt/rocm"):
                    self.info.warnings.append("ROCm not found. AMD GPU may work with Vulkan backend")
        
        self.info.missing_deps = missing
        return missing


class InstallationGuide:
    """Generate installation commands and guides."""
    
    def __init__(self, system_info: SystemInfo):
        self.info = system_info
    
    def get_install_commands(self) -> List[str]:
        """Get platform-specific installation commands."""
        commands = []
        
        if self.info.os_type == OSType.WINDOWS:
            commands.extend(self._windows_commands())
        elif self.info.os_type == OSType.LINUX:
            commands.extend(self._linux_commands())
        elif self.info.os_type == OSType.MACOS:
            commands.extend(self._macos_commands())
        
        return commands
    
    def _windows_commands(self) -> List[str]:
        commands = [
            "# Install moxing-server",
            "pip install moxing-server",
            "",
            "# Download pre-built binaries",
            f"moxing download-binaries --backend {self.info.recommended_backend.value}",
            "",
            "# Verify installation",
            "moxing devices",
        ]
        
        # Add GPU-specific setup
        if self.info.recommended_backend == BackendType.CUDA:
            commands.insert(0, "# For NVIDIA GPUs, ensure CUDA is installed:")
            commands.insert(1, "# Download from: https://developer.nvidia.com/cuda-downloads")
            commands.insert(2, "")
        elif self.info.recommended_backend == BackendType.VULKAN:
            commands.insert(0, "# Vulkan runtime is usually included with GPU drivers")
            commands.insert(1, "# Download latest drivers from your GPU vendor")
            commands.insert(2, "")
        
        return commands
    
    def _linux_commands(self) -> List[str]:
        commands = []
        distro = self._detect_distro()
        
        if distro in ("ubuntu", "debian"):
            commands.extend([
                "# Install system dependencies",
                "sudo apt update",
                "sudo apt install -y python3 python3-pip python3-venv vulkan-tools",
                "",
                "# Create virtual environment (recommended)",
                "python3 -m venv ~/moxing-env",
                "source ~/moxing-env/bin/activate",
                "",
            ])
        elif distro in ("fedora", "rhel"):
            commands.extend([
                "sudo dnf install -y python3 python3-pip vulkan-loader vulkan-tools",
                "",
            ])
        elif distro == "arch":
            commands.extend([
                "sudo pacman -S python python-pip vulkan-tools vulkan-icd-loader",
                "",
            ])
        else:
            commands.extend([
                "# Install Python 3.8+, pip, and Vulkan using your package manager",
                "",
            ])
        
        commands.extend([
            "# Install moxing-server",
            "pip install moxing-server",
            "",
            "# Download pre-built binaries",
            f"moxing download-binaries --backend {self.info.recommended_backend.value}",
            "",
            "# Verify installation",
            "moxing devices",
        ])
        
        return commands
    
    def _macos_commands(self) -> List[str]:
        commands = [
            "# Install Homebrew (if not installed)",
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
            "",
            "# Install Python",
            "brew install python@3.11",
            "",
            "# Create virtual environment",
            "python3.11 -m venv ~/moxing-env",
            "source ~/moxing-env/bin/activate",
            "",
            "# Install moxing-server",
            "pip install moxing-server",
            "",
        ]
        
        if self.info.recommended_backend == BackendType.METAL:
            commands.append("# Metal is automatically used on Apple Silicon")
        else:
            commands.append(f"moxing download-binaries --backend {self.info.recommended_backend.value}")
        
        commands.extend([
            "",
            "# Verify installation",
            "moxing devices",
        ])
        
        return commands
    
    def _detect_distro(self) -> str:
        try:
            with open("/etc/os-release") as f:
                content = f.read().lower()
                if "ubuntu" in content:
                    return "ubuntu"
                elif "debian" in content:
                    return "debian"
                elif "fedora" in content:
                    return "fedora"
                elif "rhel" in content or "centos" in content:
                    return "rhel"
                elif "arch" in content:
                    return "arch"
        except:
            pass
        return "unknown"


def print_system_report(info: SystemInfo):
    """Print a detailed system report."""
    print_header("System Information")
    
    # OS
    print(f"{Colors.BOLD}Operating System:{Colors.RESET}")
    print(f"  Type:    {color(info.os_type.value.upper(), Colors.CYAN)}")
    print(f"  Name:    {info.os_name}")
    print(f"  Version: {info.os_version[:50]}...")
    print(f"  Arch:    {info.arch}")
    print()
    
    # CPU
    print(f"{Colors.BOLD}CPU:{Colors.RESET}")
    print(f"  Name:  {info.cpu_name}")
    print(f"  Cores: {info.cpu_cores}")
    print()
    
    # Memory
    print(f"{Colors.BOLD}Memory:{Colors.RESET}")
    print(f"  RAM: {info.total_ram_gb:.1f} GB")
    print()
    
    # Python
    print(f"{Colors.BOLD}Python:{Colors.RESET}")
    print(f"  Version: {color(info.python_version, Colors.GREEN)}")
    py_ver = tuple(map(int, platform.python_version().split(".")[:2]))
    if py_ver >= (3, 8):
        print_success("Python version OK")
    else:
        print_error("Python 3.8+ required")
    print()
    
    # GPUs
    print_header("GPU Detection")
    
    if info.gpus:
        print(f"{Colors.BOLD}Detected GPUs:{Colors.RESET}")
        for i, gpu in enumerate(info.gpus):
            mem_str = f"{gpu.memory_gb:.1f}GB" if gpu.memory_mb > 0 else "N/A"
            print(f"\n  [{i}] {color(gpu.name, Colors.GREEN)}")
            print(f"      Vendor:  {gpu.vendor.upper()}")
            print(f"      Memory:  {mem_str}")
            print(f"      Backend: {color(gpu.backend.value.upper(), Colors.CYAN)}")
            if gpu.driver_version:
                print(f"      Driver:  {gpu.driver_version}")
    else:
        print_warning("No GPUs detected - will use CPU backend")
    
    print()
    
    # Recommended backend
    print_header("Recommendation")
    
    backend = info.recommended_backend
    print(f"  Recommended Backend: {color(backend.value.upper(), Colors.GREEN)}")
    
    if backend == BackendType.CUDA:
        print_info("NVIDIA GPU detected - CUDA will provide best performance")
    elif backend == BackendType.METAL:
        print_info("Apple Silicon detected - Metal will provide best performance")
    elif backend == BackendType.ROCM:
        print_info("AMD GPU with ROCm detected - ROCm will provide good performance")
    elif backend == BackendType.VULKAN:
        print_info("Vulkan backend recommended for cross-platform GPU acceleration")
    else:
        print_warning("No GPU acceleration available - using CPU (slow)")
    
    print()
    
    # Missing dependencies
    if info.missing_deps:
        print_header("Missing Dependencies")
        for dep in info.missing_deps:
            print_error(dep)
        print()
    
    if info.warnings:
        print_header("Warnings")
        for warning in info.warnings:
            print_warning(warning)
        print()


def print_installation_guide(guide: InstallationGuide):
    """Print installation commands."""
    print_header("Installation Commands")
    
    commands = guide.get_install_commands()
    
    print(f"{Colors.BOLD}Run the following commands:{Colors.RESET}\n")
    
    for cmd in commands:
        if cmd.startswith("#"):
            print(f"{Colors.BLUE}{cmd}{Colors.RESET}")
        elif cmd.strip():
            print(f"{Colors.GREEN}$ {cmd}{Colors.RESET}")
        else:
            print()
    
    print()
    print(f"{Colors.BOLD}After installation, run:{Colors.RESET}")
    print(f"  {Colors.GREEN}moxing devices{Colors.RESET}   # Verify GPU detection")
    print(f"  {Colors.GREEN}moxing models{Colors.RESET}    # List available models")
    print(f"  {Colors.GREEN}moxing run llama-3.2-3b -p 'Hello'{Colors.RESET}  # Quick test")
    print()


def auto_install():
    """Attempt automatic installation."""
    print_header("Automatic Installation")
    
    detector = SystemDetector()
    detector.detect_gpus()
    backend = detector.recommend_backend()
    
    print_info(f"Installing moxing-server with {backend.value} backend...")
    print()
    
    # Install package
    print(f"{Colors.BOLD}Installing moxing-server...{Colors.RESET}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "moxing-server"],
        text=True
    )
    
    if result.returncode != 0:
        print_error("Failed to install moxing-server")
        return False
    
    print_success("moxing-server installed")
    
    # Download binaries
    print(f"\n{Colors.BOLD}Downloading pre-built binaries...{Colors.RESET}")
    result = subprocess.run(
        [sys.executable, "-m", "moxing.cli", "download-binaries", "--backend", backend.value],
        text=True
    )
    
    if result.returncode != 0:
        print_warning("Failed to download binaries. Try manually: moxing download-binaries")
    else:
        print_success("Binaries downloaded")
    
    # Verify
    print(f"\n{Colors.BOLD}Verifying installation...{Colors.RESET}")
    result = subprocess.run(
        [sys.executable, "-m", "moxing.cli", "devices"],
        text=True
    )
    
    print()
    print_success("Installation complete!")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="moxing-server system detection and installation"
    )
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="Automatically install moxing-server"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Detect system
    detector = SystemDetector()
    detector.detect_gpus()
    backend = detector.recommend_backend()
    detector.check_dependencies()
    
    if args.json:
        import json
        data = {
            "os": {
                "type": detector.info.os_type.value,
                "name": detector.info.os_name,
                "version": detector.info.os_version,
                "arch": detector.info.arch,
            },
            "cpu": {
                "name": detector.info.cpu_name,
                "cores": detector.info.cpu_cores,
            },
            "memory_gb": detector.info.total_ram_gb,
            "python_version": detector.info.python_version,
            "gpus": [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "memory_gb": g.memory_gb,
                    "backend": g.backend.value,
                }
                for g in detector.info.gpus
            ],
            "recommended_backend": backend.value,
            "missing_deps": detector.info.missing_deps,
            "warnings": detector.info.warnings,
        }
        print(json.dumps(data, indent=2))
        return 0
    
    if args.install:
        return 0 if auto_install() else 1
    
    # Print report
    if not args.quiet:
        print_system_report(detector.info)
        
        guide = InstallationGuide(detector.info)
        print_installation_guide(guide)
    else:
        print(f"Backend: {backend.value}")
        print(f"GPUs: {len(detector.info.gpus)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())