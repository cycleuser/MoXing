"""
Device and backend detection for optimal performance
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
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
        self._amd_permission_ok: bool = True
        self._amd_permission_message: Optional[str] = None
    
    def check_amd_permission(self) -> Tuple[bool, Optional[str]]:
        from moxing.binaries import PlatformDetector
        return PlatformDetector.check_amd_permission()
    
    def _detect_amd_via_amdsmi(self) -> List[Device]:
        devices = []
        try:
            result = subprocess.run(
                ["amd-smi", "list", "--json"],
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                stdout = result.stdout
                json_start = stdout.find('[\n')
                if json_start >= 0:
                    stdout = stdout[json_start:]
                
                import json
                try:
                    data = json.loads(stdout)
                    for gpu in data:
                        gpu_id = gpu.get("gpu", 0)
                        
                        name = "AMD GPU"
                        try:
                            name_result = subprocess.run(
                                ["amd-smi", "static", "--gpu", str(gpu_id), "--json"],
                                capture_output=True,
                                encoding='utf-8',
                                errors='replace',
                                timeout=5
                            )
                            if name_result.returncode == 0 and name_result.stdout:
                                name_stdout = name_result.stdout
                                json_start = name_stdout.find('{\n')
                                if json_start >= 0:
                                    name_stdout = name_stdout[json_start:]
                                name_data = json.loads(name_stdout)
                                name = name_data.get("gpu_name", "AMD GPU")
                        except Exception:
                            pass
                        
                        vram_mb = 0
                        try:
                            vram_result = subprocess.run(
                                ["amd-smi", "static", "--gpu", str(gpu_id), "--json"],
                                capture_output=True,
                                encoding='utf-8',
                                errors='replace',
                                timeout=5
                            )
                            if vram_result.returncode == 0 and vram_result.stdout:
                                vram_stdout = vram_result.stdout
                                json_start = vram_stdout.find('{\n')
                                if json_start >= 0:
                                    vram_stdout = vram_stdout[json_start:]
                                vram_data = json.loads(vram_stdout)
                                vram = vram_data.get("gpu_vram_total", {}).get("value", 0)
                                vram_mb = int(vram / (1024 * 1024)) if vram else 0
                        except Exception:
                            pass
                        
                        devices.append(Device(
                            index=gpu_id,
                            name=name,
                            backend=BackendType.ROCM,
                            memory_mb=vram_mb,
                            free_memory_mb=vram_mb,
                            vendor="amd"
                        ))
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        return devices
    
    def _detect_amd_via_rocmsmi(self) -> List[Device]:
        devices = []
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showmeminfo", "vram"],
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split("\n")
                current_gpu = None
                for line in lines:
                    line = line.strip()
                    if line.startswith("GPU"):
                        match = re.match(r"GPU\[(\d+)\]", line)
                        if match:
                            current_gpu = int(match.group(1))
                    elif current_gpu is not None and "VRAM Total Memory" in line:
                        match = re.search(r"(\d+)\s*(MB|GB)", line)
                        if match:
                            mem = int(match.group(1))
                            if match.group(2) == "GB":
                                mem *= 1024
                            devices.append(Device(
                                index=current_gpu,
                                name="AMD GPU",
                                backend=BackendType.ROCM,
                                memory_mb=mem,
                                free_memory_mb=mem,
                                vendor="amd"
                            ))
                            current_gpu = None
        except Exception:
            pass
        return devices
    
    def _detect_via_sysfs(self) -> List[Device]:
        devices = []
        if sys.platform != "linux":
            return devices
        
        drm_path = Path("/sys/class/drm")
        if not drm_path.exists():
            return devices
        
        for card in sorted(drm_path.glob("card*"), key=lambda x: int(x.name.replace("card", ""))):
            try:
                uevent_path = card / "device/uevent"
                if uevent_path.exists():
                    content = uevent_path.read_text()
                    
                    if "amdgpu" not in content.lower() and "AMD" not in content:
                        continue
                    
                    vendor_match = re.search(r"PCI_ID=([0-9a-fA-F]+):([0-9a-fA-F]+)", content)
                    if vendor_match and vendor_match.group(1) != "1002":
                        continue
                    
                    idx = int(card.name.replace("card", ""))
                    
                    name = "AMD GPU"
                    name_path = card / "device/driver/module/drivers"
                    if name_path.exists():
                        for d in name_path.iterdir():
                            if "amdgpu" in d.name:
                                name = "AMD Radeon"
                                break
                    
                    vram_mb = 0
                    vram_path = card / "device/mem_info_vram_total"
                    if vram_path.exists():
                        try:
                            vram = int(vram_path.read_text().strip())
                            vram_mb = int(vram / (1024 * 1024))
                        except Exception:
                            pass
                    
                    devices.append(Device(
                        index=idx,
                        name=name,
                        backend=BackendType.ROCM,
                        memory_mb=vram_mb,
                        free_memory_mb=vram_mb,
                        vendor="amd"
                    ))
            except Exception:
                continue
        
        return devices
    
    @property
    def binary_path(self) -> Path:
        if self._binary_path is None:
            from moxing.server import LlamaServer
            self._binary_path = LlamaServer.get_binary_path()
        return self._binary_path
    
    def detect(self) -> List[Device]:
        """Detect all available devices."""
        self._devices = []
        
        has_amd, amd_msg = self.check_amd_permission()
        if not has_amd:
            self._amd_permission_ok = False
            self._amd_permission_message = amd_msg
            console.print(f"[yellow]Warning: {amd_msg}[/yellow]")
        
        try:
            result = subprocess.run(
                [str(self.binary_path), "--list-devices"],
                capture_output=True,
                encoding='utf-8',
                errors='replace',
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
            console.print(f"[yellow]Warning: Device detection via llama.cpp failed: {e}[/yellow]")
        
        amd_detected = any(d.backend == BackendType.ROCM for d in self._devices)
        if not amd_detected and self._amd_permission_ok:
            console.print("[dim]Trying alternative AMD GPU detection methods...[/dim]")
            
            for detect_func in [
                self._detect_amd_via_amdsmi,
                self._detect_amd_via_rocmsmi,
                self._detect_via_sysfs
            ]:
                try:
                    amd_devices = detect_func()
                    if amd_devices:
                        for dev in amd_devices:
                            exists = any(d.index == dev.index and d.backend == BackendType.ROCM for d in self._devices)
                            if not exists:
                                self._devices.append(dev)
                        console.print(f"[green]Found {len(amd_devices)} AMD GPU(s) via alternative detection[/green]")
                        break
                except Exception:
                    continue
        
        has_cuda = any(d.backend == BackendType.CUDA for d in self._devices)
        if not has_cuda:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=10
                )
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split(", ")
                        if len(parts) >= 4:
                            idx = int(parts[0])
                            name = parts[1].strip()
                            memory_mb = int(parts[2])
                            free_memory_mb = int(parts[3])
                            
                            exists = any(d.index == idx and d.backend == BackendType.CUDA for d in self._devices)
                            if not exists:
                                self._devices.append(Device(
                                    index=idx,
                                    name=name,
                                    backend=BackendType.CUDA,
                                    memory_mb=memory_mb,
                                    free_memory_mb=free_memory_mb,
                                    vendor="nvidia"
                                ))
            except Exception:
                pass
        
        vulkan_detected = any(d.backend == BackendType.VULKAN for d in self._devices)
        if not vulkan_detected:
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=10
                )
                
                if result.returncode == 0:
                    device_id = 0
                    for line in result.stdout.split("\n"):
                        line = line.strip()
                        if "deviceName" in line or "GPU" in line:
                            match = re.search(r"(?:deviceName|Name)\s*[=:]\s*(.+)", line)
                            if match:
                                name = match.group(1).strip()
                                vendor = self._detect_vendor(name)
                                
                                backend = BackendType.VULKAN
                                if vendor == "amd":
                                    backend = BackendType.ROCM
                                elif vendor == "nvidia":
                                    backend = BackendType.CUDA
                                
                                exists = any(d.name == name and d.backend == backend for d in self._devices)
                                if not exists:
                                    self._devices.append(Device(
                                        index=device_id,
                                        name=name,
                                        backend=BackendType.VULKAN,
                                        memory_mb=0,
                                        free_memory_mb=0,
                                        vendor=vendor
                                    ))
                                    device_id += 1
            except Exception:
                pass
        
        self._devices.append(Device(
            index=0,
            name="CPU",
            backend=BackendType.CPU,
            memory_mb=0,
            free_memory_mb=0,
            vendor="cpu"
        ))
        
        self._reassign_device_indices()
        
        return self._devices
    
    def _reassign_device_indices(self):
        unique_devices: List[Device] = []
        seen_names: Dict[str, int] = {}
        
        gpu_devices = [d for d in self._devices if d.backend != BackendType.CPU]
        cpu_devices = [d for d in self._devices if d.backend == BackendType.CPU]
        
        backend_order = [BackendType.CUDA, BackendType.ROCM, BackendType.METAL, BackendType.VULKAN]
        
        for backend in backend_order:
            backend_devices = [d for d in gpu_devices if d.backend == backend]
            backend_devices.sort(key=lambda x: x.memory_mb, reverse=True)
            
            for dev in backend_devices:
                name_key = f"{dev.name}_{dev.backend.value}"
                if name_key not in seen_names:
                    seen_names[name_key] = len(unique_devices)
                    unique_devices.append(dev)
        
        other_devices = [d for d in gpu_devices if d.backend not in backend_order]
        other_devices.sort(key=lambda x: x.memory_mb, reverse=True)
        for dev in other_devices:
            name_key = f"{dev.name}_{dev.backend.value}"
            if name_key not in seen_names:
                seen_names[name_key] = len(unique_devices)
                unique_devices.append(dev)
        
        for i, dev in enumerate(unique_devices):
            dev.index = i
        
        if cpu_devices:
            cpu_devices[0].index = 0
        
        self._devices = unique_devices + cpu_devices[:1]
    
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
    
    def get_device_by_name(self, device_name: str) -> Optional[Device]:
        """Get device by name like 'gpu0', 'gpu1', 'cpu'.
        
        Args:
            device_name: Device name (e.g., 'gpu0', 'gpu1', 'cpu', 'vulkan0', 'cuda0')
        
        Returns:
            Device or None if not found
        """
        if not self._devices:
            self.detect()
        
        device_name = device_name.lower().strip()
        
        if device_name == "cpu":
            for d in self._devices:
                if d.backend == BackendType.CPU:
                    return d
            return Device(index=0, name="CPU", backend=BackendType.CPU)
        
        if device_name.startswith("gpu"):
            try:
                idx = int(device_name[3:])
                for d in self._devices:
                    if d.index == idx and d.backend != BackendType.CPU:
                        return d
            except ValueError:
                pass
        
        for backend in ["vulkan", "cuda", "rocm", "metal"]:
            if device_name.startswith(backend):
                try:
                    idx = int(device_name[len(backend):])
                    for d in self._devices:
                        if d.index == idx and d.backend.value == backend:
                            return d
                except ValueError:
                    pass
        
        for d in self._devices:
            if d.name.lower() == device_name:
                return d
        
        return None
    
    def get_device_config_by_name(
        self, 
        device_name: str, 
        backend_name: Optional[str] = None,
        model_size_gb: float = 0
    ) -> DeviceConfig:
        """Get device config by device name and optional backend.
        
        Args:
            device_name: Device name (e.g., 'gpu0', 'gpu1', 'cpu')
            backend_name: Backend name (e.g., 'vulkan', 'cuda', 'auto')
            model_size_gb: Model size for context calculation
        
        Returns:
            DeviceConfig
        """
        device = self.get_device_by_name(device_name)
        
        if device is None:
            console.print(f"[yellow]Device '{device_name}' not found, using auto-detection[/yellow]")
            return self.get_best_device(model_size_gb)
        
        if backend_name and backend_name != "auto":
            try:
                backend = BackendType(backend_name.lower())
            except ValueError:
                console.print(f"[yellow]Unknown backend '{backend_name}', using device backend[/yellow]")
                backend = device.backend
        else:
            backend = device.backend
        
        if device.backend == BackendType.CPU or backend == BackendType.CPU:
            return DeviceConfig(
                backend=BackendType.CPU,
                device=device,
                n_gpu_layers=0,
                recommended_ctx=4096,
                notes="Using CPU only"
            )
        
        n_gpu_layers, ctx, notes = self._calculate_config(device, model_size_gb)
        
        return DeviceConfig(
            backend=backend,
            device=device,
            n_gpu_layers=n_gpu_layers,
            recommended_ctx=ctx,
            notes=notes
        )
    
    def list_devices(self) -> None:
        """Print a table of available devices."""
        if not self._devices:
            self.detect()
        
        table = Table(title="Available Devices (use -d option to select)")
        table.add_column("Device ID", style="cyan", width=10)
        table.add_column("Name", style="green")
        table.add_column("Backend", style="magenta")
        table.add_column("Memory", style="yellow")
        table.add_column("Free", style="blue")
        table.add_column("Vendor", style="white")
        
        for device in self._devices:
            mem = f"{device.memory_gb:.1f}GB" if device.memory_mb > 0 else "-"
            free = f"{device.free_memory_gb:.1f}GB" if device.free_memory_mb > 0 else "-"
            
            if device.backend == BackendType.CPU:
                device_id = "cpu"
            else:
                device_id = f"gpu{device.index}"
            
            table.add_row(
                device_id,
                device.name,
                device.backend.value,
                mem,
                free,
                device.vendor
            )
        
        console.print(table)
        
        amd_devices = [d for d in self._devices if d.vendor == "amd" and d.backend != BackendType.CPU]
        nvidia_devices = [d for d in self._devices if d.vendor == "nvidia" and d.backend != BackendType.CPU]
        
        if amd_devices:
            console.print()
            if not self._amd_permission_ok:
                console.print("[yellow bold]AMD GPU Permission Issue[/yellow bold]")
                console.print(f"[yellow]{self._amd_permission_message}[/yellow]")
                console.print()
                console.print("[blue]Alternative: You can still use AMD GPU with Vulkan backend![/blue]")
                for dev in amd_devices:
                    console.print(f"[dim]  moxing serve model.gguf -d gpu{dev.index} -b vulkan[/dim]")
            else:
                console.print("\n[green]AMD GPUs detected! You can use them with:[/green]")
                for dev in amd_devices:
                    console.print(f"[dim]  moxing serve model.gguf -d gpu{dev.index} -b rocm[/dim]")
                    console.print(f"[dim]  moxing serve model.gguf -d gpu{dev.index} -b vulkan[/dim]")
        
        if nvidia_devices:
            console.print("\n[green]NVIDIA GPUs detected! You can use them with:[/green]")
            for dev in nvidia_devices:
                console.print(f"[dim]  moxing serve model.gguf -d gpu{dev.index} -b cuda[/dim]")
                console.print(f"[dim]  moxing serve model.gguf -d gpu{dev.index} -b vulkan[/dim]")
        
        console.print("\n[dim]Usage: moxing serve model.gguf -d gpu0 -b cuda[/dim]")
        console.print("[dim]        moxing serve model.gguf -d gpu1 -b vulkan[/dim]")
        console.print("[dim]        moxing ollama serve model -d gpu0 -b cuda[/dim]")
    
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


def calculate_optimal_context(
    model_size_gb: float,
    available_vram_gb: float,
    ctx_size_requested: int = 0,
    vram_buffer_ratio: float = 0.15,
) -> tuple[int, int, str]:
    """Calculate optimal context size based on available VRAM.
    
    Args:
        model_size_gb: Model size in GB
        available_vram_gb: Available VRAM in GB
        ctx_size_requested: Requested context size (0 = auto)
        vram_buffer_ratio: Ratio of VRAM to leave free (default 15%)
    
    Returns:
        (ctx_size, n_gpu_layers, notes)
    """
    notes = []
    
    usable_vram_gb = available_vram_gb * (1 - vram_buffer_ratio)
    
    kv_cache_per_1k_ctx_gb = 0.002
    
    if model_size_gb >= usable_vram_gb:
        notes.append(f"Model ({model_size_gb:.1f}GB) exceeds usable VRAM ({usable_vram_gb:.1f}GB)")
        notes.append("Using CPU offloading for some layers")
        
        max_ctx_for_vram = int((usable_vram_gb * 0.3) / kv_cache_per_1k_ctx_gb) * 1024
        max_ctx = max(2048, min(max_ctx_for_vram, 16384))
        
        if ctx_size_requested > 0:
            ctx = min(ctx_size_requested, max_ctx)
        else:
            ctx = max_ctx
        
        notes.append(f"Context limited to {ctx}")
        return ctx, -1, "; ".join(notes)
    
    remaining_vram_gb = usable_vram_gb - model_size_gb
    
    max_ctx_for_vram = int(remaining_vram_gb / kv_cache_per_1k_ctx_gb) * 1024
    
    default_ctx = 4096
    if remaining_vram_gb >= 4:
        default_ctx = 16384
    elif remaining_vram_gb >= 2:
        default_ctx = 8192
    elif remaining_vram_gb >= 1:
        default_ctx = 4096
    else:
        default_ctx = 2048
    
    if ctx_size_requested > 0:
        if ctx_size_requested > max_ctx_for_vram:
            ctx = max_ctx_for_vram
            notes.append(f"Requested ctx {ctx_size_requested} exceeds VRAM, using {ctx}")
        else:
            ctx = ctx_size_requested
            notes.append(f"Using requested context size: {ctx}")
    else:
        ctx = default_ctx
        notes.append(f"Auto-detected context size: {ctx}")
    
    notes.append(f"Model: {model_size_gb:.1f}GB, Free VRAM: {remaining_vram_gb:.1f}GB")
    
    return ctx, -1, "; ".join(notes)


def get_vram_for_context(ctx_size: int) -> float:
    """Estimate VRAM needed for given context size (in GB)."""
    kv_cache_per_1k_ctx_gb = 0.002
    return ctx_size * kv_cache_per_1k_ctx_gb / 1024