"""
Device and backend detection for optimal performance
"""

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()


class BackendType(Enum):
    CUDA = "cuda"
    VULKAN = "vulkan"
    ROCM = "rocm"
    METAL = "metal"
    MLX = "mlx"
    MPS = "mps"
    CPU = "cpu"

    def __lt__(self, other):
        order = {
            BackendType.CUDA: 0,
            BackendType.METAL: 1,
            BackendType.MPS: 2,
            BackendType.MLX: 3,
            BackendType.ROCM: 4,
            BackendType.VULKAN: 5,
            BackendType.CPU: 6,
        }
        return order[self] < order[other]

    @classmethod
    def all_gpu(cls) -> List["BackendType"]:
        return [cls.CUDA, cls.ROCM, cls.VULKAN, cls.METAL, cls.MLX, cls.MPS]

    def is_gpu(self) -> bool:
        return self in self.all_gpu()

    def requires_offload_support(self) -> bool:
        return self in [BackendType.CUDA, BackendType.ROCM, BackendType.VULKAN, BackendType.METAL]


@dataclass
class Device:
    index: int
    name: str
    backend: BackendType
    memory_mb: int = 0
    free_memory_mb: int = 0
    vendor: str = ""
    total_layers: int = 0
    backend_index: int = -1

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory_mb / 1024

    @property
    def used_memory_gb(self) -> float:
        return (self.memory_mb - self.free_memory_mb) / 1024

    def __str__(self) -> str:
        mem_str = f"{self.memory_gb:.1f}GB" if self.memory_mb > 0 else "unknown"
        return f"{self.name} ({self.backend.value}, {mem_str})"


@dataclass
class DeviceConfig:
    backend: BackendType
    device: Device
    n_gpu_layers: int = -1
    n_cpu_layers: int = 0
    recommended_ctx: int = 4096
    notes: str = ""
    needs_offload: bool = False
    offload_suggested: int = 0

    @property
    def total_layers(self) -> int:
        if self.n_gpu_layers < 0:
            return -1
        return self.n_gpu_layers + self.n_cpu_layers


@dataclass
class OffloadPlan:
    can_fit_full: bool
    gpu_layers: int
    cpu_layers: int
    vram_required_gb: float
    vram_available_gb: float
    model_size_gb: float
    needs_offload: bool
    suggested_cpu_layers: int
    is_moe: bool = False
    use_cpu_moe: bool = False
    attention_layers_only: bool = False


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
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )

            if result.returncode == 0 and result.stdout:
                stdout = result.stdout
                json_start = stdout.find("[\n")
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
                                encoding="utf-8",
                                errors="replace",
                                timeout=5,
                            )
                            if name_result.returncode == 0 and name_result.stdout:
                                name_stdout = name_result.stdout
                                json_start = name_stdout.find("{\n")
                                if json_start >= 0:
                                    name_stdout = name_stdout[json_start:]
                                name_data = json.loads(name_stdout)
                                name = name_data.get("gpu_name", "AMD GPU")
                        except Exception as e:
                            logger.debug(
                                "AMD GPU detection via amd-smi failed: %s", e, exc_info=True
                            )
                            pass

                        vram_mb = 0
                        try:
                            vram_result = subprocess.run(
                                ["amd-smi", "static", "--gpu", str(gpu_id), "--json"],
                                capture_output=True,
                                encoding="utf-8",
                                errors="replace",
                                timeout=5,
                            )
                            if vram_result.returncode == 0 and vram_result.stdout:
                                vram_stdout = vram_result.stdout
                                json_start = vram_stdout.find("{\n")
                                if json_start >= 0:
                                    vram_stdout = vram_stdout[json_start:]
                                vram_data = json.loads(vram_stdout)
                                vram = vram_data.get("gpu_vram_total", {}).get("value", 0)
                                vram_mb = int(vram / (1024 * 1024)) if vram else 0
                        except Exception as e:
                            logger.debug(
                                "AMD GPU detection via amd-smi failed: %s", e, exc_info=True
                            )
                            pass

                        devices.append(
                            Device(
                                index=gpu_id,
                                name=name,
                                backend=BackendType.ROCM,
                                memory_mb=vram_mb,
                                free_memory_mb=vram_mb,
                                vendor="amd",
                            )
                        )
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug("Operation failed: %s", e, exc_info=True)
            pass
        return devices

    def _detect_amd_via_rocmsmi(self) -> List[Device]:
        devices = []
        gpu_names = {}
        gpu_vram = {}
        gpu_vram_used = {}

        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Device Name:" in line:
                        match = re.search(r"GPU\[(\d+)\].*Device Name:\s*(.+)", line)
                        if match:
                            gpu_id = int(match.group(1))
                            name = match.group(2).strip()
                            gpu_names[gpu_id] = name
        except Exception as e:
            logger.debug("ROCm GPU detection via rocm-smi failed: %s", e, exc_info=True)
            pass

        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "VRAM Total Memory (B):" in line:
                        match = re.search(r"GPU\[(\d+)\].*VRAM Total Memory \(B\):\s*(\d+)", line)
                        if match:
                            gpu_id = int(match.group(1))
                            vram_bytes = int(match.group(2))
                            gpu_vram[gpu_id] = int(vram_bytes / (1024 * 1024))

                    if "VRAM Total Used Memory (B):" in line:
                        match = re.search(
                            r"GPU\[(\d+)\].*VRAM Total Used Memory \(B\):\s*(\d+)", line
                        )
                        if match:
                            gpu_id = int(match.group(1))
                            used_bytes = int(match.group(2))
                            gpu_vram_used[gpu_id] = int(used_bytes / (1024 * 1024))
        except Exception as e:
            logger.debug("ROCm GPU detection via rocm-smi failed: %s", e, exc_info=True)
            pass

        for gpu_id in sorted(set(gpu_names.keys()) | set(gpu_vram.keys())):
            name = gpu_names.get(gpu_id, "AMD GPU")
            vram_mb = gpu_vram.get(gpu_id, 0)
            used_mb = gpu_vram_used.get(gpu_id, 0)
            free_mb = max(0, vram_mb - used_mb) if vram_mb > 0 else 0
            devices.append(
                Device(
                    index=gpu_id,
                    name=name,
                    backend=BackendType.ROCM,
                    memory_mb=vram_mb,
                    free_memory_mb=free_mb,
                    vendor="amd",
                )
            )

        return devices

    def _detect_via_sysfs(self) -> List[Device]:
        devices: List[Any] = []
        if sys.platform != "linux":
            return devices

        drm_path = Path("/sys/class/drm")
        if not drm_path.exists():
            return devices

        for card in drm_path.glob("card[0-9]*"):
            try:
                match = re.match(r"card(\d+)", card.name)
                if not match:
                    continue
                idx = int(match.group(1))

                uevent_path = card / "device/uevent"
                if not uevent_path.exists():
                    continue

                content = uevent_path.read_text()

                if "amdgpu" not in content.lower() and "AMD" not in content:
                    continue

                vendor_match = re.search(r"PCI_ID=([0-9a-fA-F]+):([0-9a-fA-F]+)", content)
                if vendor_match and vendor_match.group(1) != "1002":
                    continue

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
                    except Exception as e:
                        logger.debug("Operation failed: %s", e, exc_info=True)
                        pass

                if vram_mb > 0:
                    devices.append(
                        Device(
                            index=idx,
                            name=name,
                            backend=BackendType.ROCM,
                            memory_mb=vram_mb,
                            free_memory_mb=vram_mb,
                            vendor="amd",
                        )
                    )
            except Exception as e:
                logger.debug("Operation failed: %s", e, exc_info=True)
                continue

        return devices

    @property
    def binary_path(self) -> Path:
        if self._binary_path is None:
            self._binary_path = self._find_best_detection_binary()
        return self._binary_path

    def _find_best_detection_binary(self) -> Path:
        from moxing.binaries import BIN_DIR, PlatformDetector, get_binary_manager

        platform_name = PlatformDetector.get_platform_name()
        binary_name = "llama-server" if sys.platform != "win32" else "llama-server.exe"

        preferred_backends = ["rocm", "cuda", "vulkan", "cpu"]
        if PlatformDetector.get_os() == "darwin":
            preferred_backends = ["metal", "cpu"]

        for backend in preferred_backends:
            bundled_dir = BIN_DIR / f"{platform_name}-{backend}"
            if bundled_dir.exists():
                binary_path = bundled_dir / binary_name
                if binary_path.exists():
                    return binary_path

        try:
            manager = get_binary_manager("auto")
            if manager.has_binaries():
                return manager.get_binary_path("llama-server")
        except Exception as e:
            logger.debug("Binary manager lookup failed: %s", e, exc_info=True)
            pass

        raise FileNotFoundError("No llama-server binary found for device detection")

    def detect(self) -> List[Device]:
        """Detect all available devices."""
        self._devices = []

        has_amd, amd_msg = self.check_amd_permission()
        if not has_amd:
            self._amd_permission_ok = False
            self._amd_permission_message = amd_msg

        try:
            result = subprocess.run(
                [str(self.binary_path), "--list-devices"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=str(self.binary_path.parent),
            )

            output = result.stdout + result.stderr

            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if ":" in line and "MiB" in line:
                    match = re.match(
                        r"(\w+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB\s*free)?\)", line
                    )
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
                        elif backend_str == "mlx":
                            backend = BackendType.MLX
                        elif backend_str == "mps":
                            backend = BackendType.MPS
                        elif backend_str == "vulkan":
                            backend = BackendType.VULKAN

                        vendor = self._detect_vendor(name)

                        self._devices.append(
                            Device(
                                index=idx,
                                name=name,
                                backend=backend,
                                memory_mb=memory,
                                free_memory_mb=free_memory,
                                vendor=vendor,
                            )
                        )
        except Exception as e:
            logger.debug("Operation failed: %s", e, exc_info=True)
            pass

        if self._amd_permission_ok:
            for detect_func in [
                self._detect_amd_via_rocmsmi,
                self._detect_amd_via_amdsmi,
                self._detect_via_sysfs,
            ]:
                try:
                    amd_devices = detect_func()
                    if amd_devices:
                        for dev in amd_devices:
                            name_lower = dev.name.lower()
                            key = self._get_device_key(name_lower)

                            existing = None
                            for d in self._devices:
                                d_key = self._get_device_key(d.name.lower())
                                if d_key == key:
                                    existing = d
                                    break

                            if existing:
                                if dev.memory_mb > 0:
                                    existing.memory_mb = dev.memory_mb
                                    existing.free_memory_mb = dev.free_memory_mb
                                if dev.backend == BackendType.ROCM:
                                    existing.backend = BackendType.ROCM
                            else:
                                self._devices.append(dev)
                        break
                except Exception as e:
                    logger.debug("Operation failed: %s", e, exc_info=True)
                    continue

        has_cuda = any(d.backend == BackendType.CUDA for d in self._devices)
        if not has_cuda:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                )

                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split(", ")
                        if len(parts) >= 4:
                            idx = int(parts[0])
                            name = parts[1].strip()
                            memory_mb = int(parts[2])
                            free_memory_mb = int(parts[3])

                            exists = any(
                                d.index == idx and d.backend == BackendType.CUDA
                                for d in self._devices
                            )
                            if not exists:
                                self._devices.append(
                                    Device(
                                        index=idx,
                                        name=name,
                                        backend=BackendType.CUDA,
                                        memory_mb=memory_mb,
                                        free_memory_mb=free_memory_mb,
                                        vendor="nvidia",
                                    )
                                )
            except Exception as e:
                logger.debug("Operation failed: %s", e, exc_info=True)
                pass

        if sys.platform == "darwin":
            if not any(d.backend == BackendType.METAL for d in self._devices):
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPDisplaysDataType"],
                        capture_output=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=10,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "Chipset Model:" in line or "Model:" in line:
                                name = line.split(":")[-1].strip()
                                if (
                                    "Apple" in name
                                    or "M1" in name
                                    or "M2" in name
                                    or "M3" in name
                                    or "M4" in name
                                ):
                                    unified_memory_gb = self._get_macos_unified_memory()
                                    self._devices.append(
                                        Device(
                                            index=0,
                                            name=name,
                                            backend=BackendType.METAL,
                                            memory_mb=int(unified_memory_gb * 1024),
                                            free_memory_mb=int(unified_memory_gb * 1024 * 0.8),
                                            vendor="apple",
                                        )
                                    )
                                    break
                except Exception as e:
                    logger.debug("macOS device detection failed: %s", e, exc_info=True)
                    pass

        vulkan_detected = any(d.backend == BackendType.VULKAN for d in self._devices)
        if not vulkan_detected:
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                )

                if result.returncode == 0:
                    device_id = 0
                    for line in result.stdout.split("\n"):
                        line = line.strip()
                        if "deviceName" in line or "GPU" in line:
                            match = re.search(r"(?:deviceName|Name)\s*[=:]\s*(.+)", line)
                            if match:
                                name = match.group(1).strip()

                                if (
                                    "llvmpipe" in name.lower()
                                    or "swiftshader" in name.lower()
                                    or "software" in name.lower()
                                ):
                                    continue

                                vendor = self._detect_vendor(name)

                                exists = any(d.name == name for d in self._devices)
                                if not exists:
                                    self._devices.append(
                                        Device(
                                            index=device_id,
                                            name=name,
                                            backend=BackendType.VULKAN,
                                            memory_mb=0,
                                            free_memory_mb=0,
                                            vendor=vendor,
                                        )
                                    )
                                    device_id += 1
            except Exception as e:
                logger.debug("Operation failed: %s", e, exc_info=True)
                pass

        self._devices.append(
            Device(
                index=0,
                name="CPU",
                backend=BackendType.CPU,
                memory_mb=0,
                free_memory_mb=0,
                vendor="cpu",
            )
        )

        self._reassign_device_indices()

        return self._devices

    def _get_macos_unified_memory(self) -> float:
        """Get unified memory size on macOS."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, encoding="utf-8", timeout=5
            )
            if result.returncode == 0:
                bytes_mem = int(result.stdout.strip())
                return bytes_mem / (1024**3)
        except Exception as e:
            logger.debug("macOS unified memory detection failed: %s", e, exc_info=True)
            pass
        return 16.0

    def _get_device_key(self, name_lower: str) -> str:
        """Get a unique key for device deduplication."""
        if "7900" in name_lower or "7900xtx" in name_lower.replace(" ", ""):
            return "7900xtx"
        elif "610m" in name_lower:
            return "610m"
        elif "4070" in name_lower and "laptop" in name_lower:
            return "rtx4070_laptop"
        elif "4070" in name_lower:
            return "rtx4070"
        elif "4080" in name_lower:
            return "rtx4080"
        elif "4090" in name_lower:
            return "rtx4090"
        elif "radeon" in name_lower and ("rx" in name_lower or "7" in name_lower):
            if "7900" in name_lower:
                return "7900xtx"
            elif "610" in name_lower:
                return "610m"
            else:
                match = re.search(r"rx\s*(\d+)", name_lower)
                if match:
                    return f"rx{match.group(1)}"
        return name_lower.replace(" ", "_")[:30]

    def _reassign_device_indices(self):
        unique_devices: List[Device] = []
        seen_keys: Dict[str, Device] = {}

        gpu_devices = [d for d in self._devices if d.backend != BackendType.CPU]
        cpu_devices = [d for d in self._devices if d.backend == BackendType.CPU]

        software_renderers = ["llvmpipe", "swiftshader", "software", "mesa software"]

        for dev in gpu_devices:
            if any(sr in dev.name.lower() for sr in software_renderers):
                continue

            key = self._get_device_key(dev.name.lower())

            if key in seen_keys:
                existing = seen_keys[key]

                if dev.backend in [BackendType.ROCM, BackendType.CUDA]:
                    if dev.memory_mb > 0:
                        existing.memory_mb = dev.memory_mb
                        existing.free_memory_mb = dev.free_memory_mb
                        existing.name = dev.name
                    existing.backend = dev.backend
                elif dev.memory_mb > 0 and existing.memory_mb == 0:
                    existing.memory_mb = dev.memory_mb
                    existing.free_memory_mb = dev.free_memory_mb

                if existing.backend == BackendType.VULKAN and dev.backend in [
                    BackendType.ROCM,
                    BackendType.CUDA,
                ]:
                    existing.backend = dev.backend
                    if dev.memory_mb > 0:
                        existing.memory_mb = dev.memory_mb
                        existing.free_memory_mb = dev.free_memory_mb
            else:
                seen_keys[key] = dev

        rocm_devices = [
            d for d in seen_keys.values() if d.backend == BackendType.ROCM and d.memory_mb > 0
        ]
        cuda_devices = [d for d in seen_keys.values() if d.backend == BackendType.CUDA]
        metal_devices = [
            d
            for d in seen_keys.values()
            if d.backend in [BackendType.METAL, BackendType.MLX, BackendType.MPS]
        ]
        vulkan_devices = [d for d in seen_keys.values() if d.backend == BackendType.VULKAN]
        other_devices = [
            d
            for d in seen_keys.values()
            if d.backend
            not in [
                BackendType.CUDA,
                BackendType.ROCM,
                BackendType.METAL,
                BackendType.MLX,
                BackendType.MPS,
                BackendType.VULKAN,
            ]
        ]

        for devices in [cuda_devices, rocm_devices, metal_devices, vulkan_devices, other_devices]:
            devices.sort(key=lambda x: x.memory_mb, reverse=True)

        unique_devices = (
            cuda_devices + rocm_devices + metal_devices + vulkan_devices + other_devices
        )

        for i, dev in enumerate(unique_devices):
            if dev.backend_index == -1:
                dev.backend_index = dev.index
            dev.index = i

        if cpu_devices:
            cpu_devices[0].index = len(unique_devices)

        self._devices = unique_devices + cpu_devices[:1]

    def _detect_vendor(self, name: str) -> str:
        """Detect GPU vendor from name."""
        name_lower = name.lower()
        if (
            "nvidia" in name_lower
            or "geforce" in name_lower
            or "rtx" in name_lower
            or "gtx" in name_lower
        ):
            return "nvidia"
        elif "amd" in name_lower or "radeon" in name_lower or "rx" in name_lower:
            return "amd"
        elif "intel" in name_lower or "arc" in name_lower:
            return "intel"
        elif (
            "apple" in name_lower
            or "m1" in name_lower
            or "m2" in name_lower
            or "m3" in name_lower
            or "m4" in name_lower
        ):
            return "apple"
        return "unknown"

    def calculate_offload_plan(
        self,
        device: Device,
        model_size_gb: float,
        ctx_size: int = 4096,
        total_layers: int = 80,
        is_moe: bool = False,
        expert_count: int = 0,
        expert_used_count: int = 0,
    ) -> OffloadPlan:
        """Calculate GPU/CPU offload plan.

        Args:
            device: Target device
            model_size_gb: Model size in GB
            ctx_size: Context size
            total_layers: Total number of layers (default 80)
            is_moe: Whether this is a MoE model
            expert_count: Total number of experts (for MoE)
            expert_used_count: Number of active experts per token (for MoE)

        Returns:
            OffloadPlan with recommended configuration
        """
        if device.backend == BackendType.CPU:
            return OffloadPlan(
                can_fit_full=False,
                gpu_layers=0,
                cpu_layers=total_layers,
                vram_required_gb=0,
                vram_available_gb=0,
                model_size_gb=model_size_gb,
                needs_offload=True,
                suggested_cpu_layers=total_layers,
                is_moe=is_moe,
                use_cpu_moe=False,
            )

        available_vram_gb = device.free_memory_gb * 0.85

        if is_moe and expert_count > 0:
            return self._calculate_moe_offload(
                device=device,
                model_size_gb=model_size_gb,
                ctx_size=ctx_size,
                total_layers=total_layers,
                available_vram_gb=available_vram_gb,
                expert_count=expert_count,
                expert_used_count=expert_used_count,
            )

        kv_cache_gb = ctx_size * 0.000002

        vram_required_full = model_size_gb * 1.1 + kv_cache_gb

        if available_vram_gb >= vram_required_full:
            return OffloadPlan(
                can_fit_full=True,
                gpu_layers=-1,
                cpu_layers=0,
                vram_required_gb=vram_required_full,
                vram_available_gb=available_vram_gb,
                model_size_gb=model_size_gb,
                needs_offload=False,
                suggested_cpu_layers=0,
                is_moe=is_moe,
                use_cpu_moe=False,
            )

        layer_size_gb = model_size_gb / total_layers * 1.1
        max_gpu_layers = int(available_vram_gb / layer_size_gb)

        if max_gpu_layers >= total_layers:
            return OffloadPlan(
                can_fit_full=True,
                gpu_layers=-1,
                cpu_layers=0,
                vram_required_gb=vram_required_full,
                vram_available_gb=available_vram_gb,
                model_size_gb=model_size_gb,
                needs_offload=False,
                suggested_cpu_layers=0,
                is_moe=is_moe,
                use_cpu_moe=False,
            )

        cpu_layers_needed = total_layers - max_gpu_layers

        return OffloadPlan(
            can_fit_full=False,
            gpu_layers=max_gpu_layers,
            cpu_layers=cpu_layers_needed,
            vram_required_gb=available_vram_gb,
            vram_available_gb=device.free_memory_gb,
            model_size_gb=model_size_gb,
            needs_offload=True,
            suggested_cpu_layers=cpu_layers_needed,
            is_moe=is_moe,
            use_cpu_moe=False,
        )

    def _calculate_moe_offload(
        self,
        device: Device,
        model_size_gb: float,
        ctx_size: int,
        total_layers: int,
        available_vram_gb: float,
        expert_count: int,
        expert_used_count: int,
    ) -> OffloadPlan:
        """Calculate offload plan for MoE models with CPU expert offloading."""
        attention_layer_fraction = 0.3
        expert_layer_fraction = 0.7

        attention_size_gb = model_size_gb * attention_layer_fraction
        expert_size_gb = model_size_gb * expert_layer_fraction

        active_expert_ratio = expert_used_count / expert_count if expert_count > 0 else 0.1
        active_expert_memory_gb = expert_size_gb * active_expert_ratio

        vram_for_attention = attention_size_gb * 1.1
        kv_cache_gb = ctx_size * 0.000002
        total_vram_needed = vram_for_attention + kv_cache_gb + active_expert_memory_gb * 0.1

        if available_vram_gb >= total_vram_needed:
            return OffloadPlan(
                can_fit_full=True,
                gpu_layers=-1,
                cpu_layers=0,
                vram_required_gb=total_vram_needed,
                vram_available_gb=available_vram_gb,
                model_size_gb=model_size_gb,
                needs_offload=False,
                suggested_cpu_layers=0,
                is_moe=True,
                use_cpu_moe=True,
                attention_layers_only=True,
            )

        return OffloadPlan(
            can_fit_full=False,
            gpu_layers=-1,
            cpu_layers=0,
            vram_required_gb=available_vram_gb,
            vram_available_gb=available_vram_gb,
            model_size_gb=model_size_gb,
            needs_offload=True,
            suggested_cpu_layers=0,
            is_moe=True,
            use_cpu_moe=True,
            attention_layers_only=True,
        )

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
                n_cpu_layers=80,
                recommended_ctx=4096,
                notes="No GPU available, using CPU",
            )

        offload_plan = self.calculate_offload_plan(best_device, model_size_gb)

        return DeviceConfig(
            backend=best_backend,
            device=best_device,
            n_gpu_layers=offload_plan.gpu_layers,
            n_cpu_layers=offload_plan.cpu_layers,
            recommended_ctx=self._calculate_ctx(best_device, model_size_gb, offload_plan),
            notes=self._build_notes(best_device, offload_plan),
            needs_offload=offload_plan.needs_offload,
            offload_suggested=offload_plan.suggested_cpu_layers,
        )

    def _score_device(self, device: Device, model_size_gb: float) -> int:
        """Score a device based on performance potential."""
        score = 0

        backend_scores = {
            BackendType.CUDA: 100,
            BackendType.METAL: 95,
            BackendType.MPS: 95,
            BackendType.MLX: 90,
            BackendType.ROCM: 85,
            BackendType.VULKAN: 70,
            BackendType.CPU: 0,
        }
        score += backend_scores.get(device.backend, 0)

        if device.free_memory_mb > 0:
            free_gb = device.free_memory_gb
            if model_size_gb > 0:
                if free_gb >= model_size_gb * 1.3:
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

    def _calculate_ctx(
        self, device: Device, model_size_gb: float, offload_plan: OffloadPlan
    ) -> int:
        """Calculate optimal context size."""
        if offload_plan.can_fit_full:
            available_gb = device.free_memory_gb * 0.85 - model_size_gb
            if available_gb >= 4:
                return 16384
            elif available_gb >= 2:
                return 8192
            else:
                return 4096
        else:
            return 4096

    def _build_notes(self, device: Device, offload_plan: OffloadPlan) -> str:
        """Build notes string for device config."""
        notes = []
        notes.append(f"GPU: {device.name}")

        if offload_plan.can_fit_full:
            notes.append("Full GPU offload")
        else:
            notes.append(f"GPU layers: {offload_plan.gpu_layers}")
            notes.append(f"CPU layers: {offload_plan.cpu_layers}")
            notes.append(f"VRAM: {offload_plan.vram_available_gb:.1f}GB available")

        return "; ".join(notes)

    def get_device_by_name(
        self, device_name: str, backend: Optional[str] = None
    ) -> Optional[Device]:
        """Get device by name like 'gpu0', 'gpu1', 'cpu'.

        Args:
            device_name: Device identifier (e.g., 'gpu0', 'gpu1', 'cpu')
            backend: Optional backend filter (e.g., 'rocm', 'cuda', 'vulkan')
        """
        if not self._devices:
            self.detect()

        device_name = device_name.lower().strip()

        backend_type = None
        if backend:
            backend = backend.lower()
            if backend in ["rocm", "hip"]:
                backend_type = BackendType.ROCM
            elif backend == "cuda":
                backend_type = BackendType.CUDA
            elif backend == "vulkan":
                backend_type = BackendType.VULKAN
            elif backend in ["metal", "mtl"]:
                backend_type = BackendType.METAL
            elif backend == "cpu":
                backend_type = BackendType.CPU

        if device_name == "cpu":
            for d in self._devices:
                if d.backend == BackendType.CPU:
                    return d
            return Device(index=0, name="CPU", backend=BackendType.CPU)

        if device_name.startswith("gpu"):
            try:
                idx = int(device_name[3:])
                for d in self._devices:
                    if (
                        d.index == idx
                        and d.backend != BackendType.CPU
                        and (backend_type is None or d.backend == backend_type)
                    ):
                        return d
                if backend_type:
                    for d in self._devices:
                        if (
                            d.index == idx
                            and d.backend != BackendType.CPU
                            and (
                                backend_type == BackendType.ROCM
                                and d.vendor == "amd"
                                or backend_type == BackendType.CUDA
                                and d.vendor == "nvidia"
                                or backend_type == BackendType.VULKAN
                            )
                        ):
                            return d
            except ValueError:
                pass

        for backend_prefix in ["vulkan", "cuda", "rocm", "hip", "metal", "mtl", "mlx", "mps"]:
            if device_name.lower().startswith(backend_prefix):
                prefix_len = len(backend_prefix)
                try:
                    idx = int(device_name[prefix_len:])
                    canonical_backends = {
                        "hip": "rocm",
                        "mtl": "metal",
                    }
                    effective_backend = canonical_backends.get(backend_prefix, backend_prefix)
                    for d in self._devices:
                        if d.backend.value == effective_backend and (d.backend_index == idx):
                            return d
                    for d in self._devices:
                        if d.backend.value == effective_backend and d.index == idx:
                            return d
                    for d in self._devices:
                        if d.backend.value == effective_backend:
                            return d
                except ValueError:
                    pass

        for d in self._devices:
            if d.name.lower() == device_name:
                return d

        if backend_type:
            matching = [
                d
                for d in self._devices
                if d.backend == backend_type and d.backend != BackendType.CPU
            ]
            if matching:
                try:
                    clean = device_name.replace("gpu", "").replace("ROCm", "")
                    clean = clean.replace("CUDA", "").replace("Vulkan", "").replace("MTL", "")
                    idx = int(clean)
                    for d in matching:
                        if d.backend_index == idx:
                            return d
                except (ValueError, AttributeError):
                    pass
                return None

        return None

    def get_device_config_by_name(
        self,
        device_name: str,
        backend_name: Optional[str] = None,
        model_size_gb: float = 0,
        cpu_offload_layers: int = 0,
    ) -> DeviceConfig:
        """Get device config by device name and optional backend."""
        device = self.get_device_by_name(device_name)

        if device is None:
            console.print(
                f"[yellow]Device '{device_name}' not found, using auto-detection[/yellow]"
            )
            return self.get_best_device(model_size_gb)

        if backend_name and backend_name != "auto":
            try:
                backend = BackendType(backend_name.lower())
            except ValueError:
                console.print(
                    f"[yellow]Unknown backend '{backend_name}', using device backend[/yellow]"
                )
                backend = device.backend
        else:
            backend = device.backend

        if device.backend == BackendType.CPU or backend == BackendType.CPU:
            return DeviceConfig(
                backend=BackendType.CPU,
                device=device,
                n_gpu_layers=0,
                n_cpu_layers=80,
                recommended_ctx=4096,
                notes="Using CPU only",
            )

        offload_plan = self.calculate_offload_plan(device, model_size_gb)

        if cpu_offload_layers > 0:
            if offload_plan.gpu_layers > 0:
                gpu_layers = max(1, offload_plan.gpu_layers - cpu_offload_layers)
            else:
                gpu_layers = -1
        else:
            gpu_layers = -1 if offload_plan.can_fit_full else -1

        return DeviceConfig(
            backend=backend,
            device=device,
            n_gpu_layers=gpu_layers,
            n_cpu_layers=cpu_offload_layers if cpu_offload_layers > 0 else offload_plan.cpu_layers,
            recommended_ctx=self._calculate_ctx(device, model_size_gb, offload_plan),
            notes=self._build_notes(device, offload_plan),
            needs_offload=offload_plan.needs_offload,
            offload_suggested=offload_plan.suggested_cpu_layers,
        )

    def list_devices(self) -> None:
        """Print a table of available devices."""
        if not self._devices:
            self.detect()

        table = Table(title="Available Devices")
        table.add_column("ID", style="cyan", width=6)
        table.add_column("Name", style="green", width=32)
        table.add_column("Backend", style="magenta", width=8)
        table.add_column("Memory", style="yellow", width=10)
        table.add_column("Free", style="blue", width=10)
        table.add_column("Vendor", style="white", width=8)

        for device in self._devices:
            if device.memory_mb > 0:
                if device.memory_gb >= 1:
                    mem = f"{device.memory_gb:.1f}GB"
                else:
                    mem = f"{device.memory_mb}MB"
            else:
                mem = "-"

            if device.free_memory_mb > 0:
                if device.free_memory_gb >= 1:
                    free = f"{device.free_memory_gb:.1f}GB"
                else:
                    free = f"{device.free_memory_mb}MB"
            else:
                free = "-"

            device_id = "cpu" if device.backend == BackendType.CPU else f"gpu{device.index}"

            backend_display = device.backend.value.upper()

            table.add_row(device_id, device.name[:32], backend_display, mem, free, device.vendor)

        console.print(table)

        gpu_devices = [d for d in self._devices if d.backend != BackendType.CPU]

        if gpu_devices:
            console.print("\n[bold]Backend Support:[/bold]")

            nvidia_devices = [d for d in gpu_devices if d.vendor == "nvidia"]
            if nvidia_devices:
                console.print("  [green]NVIDIA:[/green] CUDA, Vulkan")

            amd_devices = [d for d in gpu_devices if d.vendor == "amd"]
            if amd_devices:
                console.print("  [green]AMD:[/green] ROCm, Vulkan")

            console.print("\n[bold]Usage:[/bold]")
            console.print("  [dim]moxing ollama serve model -d gpu0 -b cuda[/dim]")
            console.print("  [dim]moxing ollama serve model -d gpu1 -b rocm[/dim]")
            console.print("  [dim]moxing ollama serve model -d gpu0 -b vulkan[/dim]")

            console.print("\n[bold]CPU Offload Options:[/bold]")
            console.print("  [dim]moxing ollama serve model --cpu-offload 10[/dim]")
            console.print("  [dim]moxing ollama serve model --prompt-offload[/dim]")

    def get_backend_env(self, backend: BackendType, device: Optional[Device] = None) -> dict:
        """Get environment variables for the specified backend."""
        env = os.environ.copy()

        if backend == BackendType.ROCM:
            if device:
                backend_idx = device.backend_index if device.backend_index >= 0 else device.index
                env["HIP_VISIBLE_DEVICES"] = str(backend_idx)
            else:
                env["HIP_VISIBLE_DEVICES"] = "0"

            rocm_candidates = [
                "/opt/rocm/lib",
                "/opt/rocm/core/lib",
            ]
            import glob as _glob

            rocm_candidates.extend(_glob.glob("/opt/rocm/core-*/lib"))
            rocm_candidates.extend(_glob.glob("/opt/rocm-*/lib"))

            ld_path = env.get("LD_LIBRARY_PATH", "")
            for path in rocm_candidates:
                if Path(path).exists():
                    ld_path = f"{path}:{ld_path}"
            env["LD_LIBRARY_PATH"] = ld_path

        elif backend == BackendType.CUDA:
            if device:
                backend_idx = device.backend_index if device.backend_index >= 0 else device.index
                env["CUDA_VISIBLE_DEVICES"] = str(backend_idx)
            else:
                env["CUDA_VISIBLE_DEVICES"] = "0"

        elif backend == BackendType.VULKAN:
            if device:
                backend_idx = device.backend_index if device.backend_index >= 0 else device.index
                env["GGML_VK_VISIBLE_DEVICES"] = str(backend_idx)
            else:
                env["GGML_VK_VISIBLE_DEVICES"] = "0"

        elif backend in (BackendType.METAL, BackendType.MLX, BackendType.MPS):
            pass

        return env


def detect_best_backend() -> BackendType:
    """Quick detection of the best available backend."""
    detector = DeviceDetector()
    devices = detector.detect()

    if not devices:
        return BackendType.CPU

    gpu_devices = [d for d in devices if d.backend.is_gpu()]
    if not gpu_devices:
        return BackendType.CPU

    return min(gpu_devices, key=lambda d: d.backend).backend


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
        return size / (1024**3)
    except:  # noqa: E722
        return 0


def calculate_optimal_context(
    model_size_gb: float,
    available_vram_gb: float,
    ctx_size_requested: int = 0,
    vram_buffer_ratio: float = 0.15,
    kv_cache_bits: int = 16,
) -> tuple:
    """Calculate optimal context size based on available VRAM.

    Args:
        model_size_gb: Model size in GB
        available_vram_gb: Available VRAM in GB
        ctx_size_requested: Requested context size (0 = auto)
        vram_buffer_ratio: Buffer ratio for VRAM (default 0.15)
        kv_cache_bits: KV cache bits per element (16=f16, 8=q8_0, 4=q4_0)

    Returns:
        tuple: (ctx_size, n_gpu_layers, cpu_offload_layers, notes)
    """
    notes = []

    usable_vram_gb = available_vram_gb * (1 - vram_buffer_ratio)

    # KV cache memory per 1K context depends on quantization
    # F16: ~2MB per 1K ctx, Q4_0: ~0.5MB per 1K ctx (4x less)
    kv_cache_per_1k_ctx_gb = 0.002 * (kv_cache_bits / 16.0)

    if model_size_gb >= usable_vram_gb:
        notes.append(f"Model ({model_size_gb:.1f}GB) exceeds usable VRAM ({usable_vram_gb:.1f}GB)")
        notes.append("Using CPU offloading for some layers")

        max_ctx_for_vram = int((usable_vram_gb * 0.3) / kv_cache_per_1k_ctx_gb) * 1024
        # No hard limit - user knows best when using KV cache quantization
        max_ctx = max(2048, max_ctx_for_vram)

        ctx = min(ctx_size_requested, max_ctx) if ctx_size_requested > 0 else min(max_ctx, 16384)

        notes.append(f"Context limited to {ctx}")
        return ctx, -1, 0, "; ".join(notes)

    remaining_vram_gb = usable_vram_gb - model_size_gb

    max_ctx_for_vram = int(remaining_vram_gb / kv_cache_per_1k_ctx_gb) * 1024

    # Default context based on remaining VRAM
    default_ctx = 4096
    if remaining_vram_gb >= 8:
        default_ctx = 32768
    elif remaining_vram_gb >= 4:
        default_ctx = 16384
    elif remaining_vram_gb >= 2:
        default_ctx = 8192
    elif remaining_vram_gb >= 1:
        default_ctx = 4096
    else:
        default_ctx = 2048

    # Respect user's requested context size
    if ctx_size_requested > 0:
        if ctx_size_requested > max_ctx_for_vram:
            ctx = max_ctx_for_vram
            notes.append(f"Requested ctx {ctx_size_requested} exceeds VRAM, using {ctx}")
        else:
            ctx = ctx_size_requested
            notes.append(f"Using requested context size: {ctx}")
    else:
        ctx = min(default_ctx, max_ctx_for_vram)
        notes.append(f"Auto-detected context size: {ctx}")

    notes.append(f"Model: {model_size_gb:.1f}GB, Free VRAM: {remaining_vram_gb:.1f}GB")

    return ctx, -1, 0, "; ".join(notes)
