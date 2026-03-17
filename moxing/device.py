"""
Device and backend detection for optimal performance
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

from rich.console import Console
from rich.table import Table

console = Console()


class BackendType(Enum):
    CUDA = "cuda"
    VULKAN = "vulkan"
    ROCM = "rocm"
    METAL = "metal"
    CPU = "cpu"
    
    def __lt__(self, other):
        order = {
            BackendType.CUDA: 0,
            BackendType.METAL: 1,
            BackendType.ROCM: 2,
            BackendType.VULKAN: 3,
            BackendType.CPU: 4,
        }
        return order[self] < order[other]


@dataclass
class Device:
    index: int
    name: str
    backend: BackendType
    memory_mb: int = 0
    free_memory_mb: int = 0
    vendor: str = ""
    
    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024
    
    @property
    def free_memory_gb(self) -> float:
        return self.free_memory_mb / 1024
    
    def __str__(self) -> str:
        mem_str = f"{self.memory_gb:.1f}GB" if self.memory_mb > 0 else "unknown"
        return f"{self.name} ({self.backend.value}, {mem_str})"


@dataclass
class DeviceConfig:
    backend: BackendType
    device: Device
    n_gpu_layers: int = -1
    recommended_ctx: int = 4096
    notes: str = ""


class DeviceDetector:
    """Detect and manage available compute devices."""
    
    def __init__(self, binary_path: Optional[Path] = None):
        self._binary_path = binary_path
        self._devices: List[Device] = []
        self._preferred_backend: Optional[BackendType] = None
    
    @property
    def binary_path(self) -> Path:
        if self._binary_path is None:
            from moxing.server import LlamaServer
            self._binary_path = LlamaServer.get_binary_path()
        return self._binary_path
    
    def detect(self) -> List[Device]:
        """Detect all available devices."""
        self._devices = []
        
        try:
            result = subprocess.run(
                [str(self.binary_path), "--list-devices"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.binary_path.parent)
            )
            
            output = result.stdout + result.stderr
            
            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if ":" in line and "MiB" in line:
                    match = re.match(r"(\w+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB\s*free)?\)", line)
                    if match:
                        backend_str = match.group(1).lower()
                        idx = int(match.group(2))
                        name = match.group(3).strip()
                        memory = int(match.group(4))
                        free_memory = int(match.group(5)) if match.group(5) else memory
                        
                        backend = BackendType.VULKAN
                        if backend_str == "cuda":
                            backend = BackendType.CUDA
                        elif backend_str == "rocm" or backend_str == "hip":
                            backend = BackendType.ROCM
                        elif backend_str == "metal" or backend_str == "mtl":
                            backend = BackendType.METAL
                        elif backend_str == "vulkan":
                            backend = BackendType.VULKAN
                        
                        vendor = self._detect_vendor(name)
                        
                        self._devices.append(Device(
                            index=idx,
                            name=name,
                            backend=backend,
                            memory_mb=memory,
                            free_memory_mb=free_memory,
                            vendor=vendor
                        ))
        except Exception as e:
            console.print(f"[yellow]Warning: Device detection failed: {e}[/yellow]")
        
        if not self._devices:
            self._devices.append(Device(
                index=0,
                name="CPU",
                backend=BackendType.CPU,
                memory_mb=0
            ))
        
        return self._devices
    
    def _detect_vendor(self, name: str) -> str:
        """Detect GPU vendor from name."""
        name_lower = name.lower()
        if "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
            return "nvidia"
        elif "amd" in name_lower or "radeon" in name_lower or "rx" in name_lower:
            return "amd"
        elif "intel" in name_lower or "arc" in name_lower:
            return "intel"
        elif "apple" in name_lower or "m1" in name_lower or "m2" in name_lower or "m3" in name_lower:
            return "apple"
        return "unknown"
    
    def get_best_device(self, model_size_gb: float = 0) -> DeviceConfig:
        """Get the best device configuration for the given model size."""
        if not self._devices:
            self.detect()
        
        best_device = None
        best_backend = BackendType.CPU
        best_score = -1
        
        for device in self._devices:
            if device.backend == BackendType.CPU:
                continue
            
            score = self._score_device(device, model_size_gb)
            if score > best_score:
                best_score = score
                best_device = device
                best_backend = device.backend
        
        if best_device is None:
            return DeviceConfig(
                backend=BackendType.CPU,
                device=Device(index=0, name="CPU", backend=BackendType.CPU),
                n_gpu_layers=0,
                recommended_ctx=4096,
                notes="No GPU available, using CPU"
            )
        
        n_gpu_layers, ctx, notes = self._calculate_config(best_device, model_size_gb)
        
        return DeviceConfig(
            backend=best_backend,
            device=best_device,
            n_gpu_layers=n_gpu_layers,
            recommended_ctx=ctx,
            notes=notes
        )
    
    def _score_device(self, device: Device, model_size_gb: float) -> int:
        """Score a device based on performance potential."""
        score = 0
        
        backend_scores = {
            BackendType.CUDA: 100,
            BackendType.METAL: 90,
            BackendType.ROCM: 85,
            BackendType.VULKAN: 70,
            BackendType.CPU: 0,
        }
        score += backend_scores.get(device.backend, 0)
        
        if device.free_memory_mb > 0:
            free_gb = device.free_memory_gb
            if model_size_gb > 0:
                if free_gb >= model_size_gb * 1.2:
                    score += 50
                elif free_gb >= model_size_gb:
                    score += 30
            else:
                score += min(int(free_gb * 5), 50)
        
        vendor_bonuses = {
            "nvidia": 10,
            "apple": 5,
        }
        score += vendor_bonuses.get(device.vendor, 0)
        
        return score
    
    def _calculate_config(self, device: Device, model_size_gb: float) -> Tuple[int, int, str]:
        """Calculate optimal GPU layers and context size."""
        notes = []
        
        if device.free_memory_gb <= 0:
            return -1, 4096, "Unknown GPU memory, using all layers"
        
        available_gb = device.free_memory_gb * 0.85
        
        if model_size_gb <= 0:
            return -1, 4096, f"Using all GPU layers (model size unknown)"
        
        if available_gb >= model_size_gb * 1.3:
            ctx = min(8192, int((available_gb - model_size_gb) * 1024))
            notes.append(f"Full GPU offload possible")
        elif available_gb >= model_size_gb:
            ctx = 4096
            notes.append(f"GPU offload with limited context")
        else:
            ratio = available_gb / model_size_gb
            notes.append(f"Partial GPU offload (~{int(ratio*100)}%)")
            ctx = 2048
        
        notes.append(f"GPU: {device.name}")
        
        return -1, ctx, "; ".join(notes)
    
    def list_devices(self) -> None:
        """Print a table of available devices."""
        if not self._devices:
            self.detect()
        
        table = Table(title="Available Devices")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="green")
        table.add_column("Backend", style="magenta")
        table.add_column("Memory", style="yellow")
        table.add_column("Free", style="blue")
        table.add_column("Vendor", style="white")
        
        for device in self._devices:
            mem = f"{device.memory_gb:.1f}GB" if device.memory_mb > 0 else "-"
            free = f"{device.free_memory_gb:.1f}GB" if device.free_memory_mb > 0 else "-"
            table.add_row(
                str(device.index),
                device.name,
                device.backend.value,
                mem,
                free,
                device.vendor
            )
        
        console.print(table)
    
    def get_backend_env(self, backend: BackendType) -> dict:
        """Get environment variables for the specified backend."""
        env = os.environ.copy()
        
        if backend == BackendType.VULKAN:
            pass
        elif backend == BackendType.CUDA:
            pass
        elif backend == BackendType.ROCM:
            env["HIP_VISIBLE_DEVICES"] = "0"
        elif backend == BackendType.METAL:
            pass
        
        return env


def detect_best_backend() -> BackendType:
    """Quick detection of the best available backend."""
    detector = DeviceDetector()
    devices = detector.detect()
    
    if not devices:
        return BackendType.CPU
    
    best = min([d.backend for d in devices if d.backend != BackendType.CPU], default=BackendType.CPU)
    return best


def get_device_config(model_path: Optional[str] = None, model_size_gb: float = 0) -> DeviceConfig:
    """Get optimal device configuration for a model."""
    detector = DeviceDetector()
    detector.detect()
    
    if model_size_gb <= 0 and model_path:
        model_size_gb = estimate_model_size_gb(model_path)
    
    return detector.get_best_device(model_size_gb)


def estimate_model_size_gb(model_path: str) -> float:
    """Estimate model size in GB from file path."""
    try:
        size = Path(model_path).stat().st_size
        return size / (1024 ** 3)
    except:
        return 0