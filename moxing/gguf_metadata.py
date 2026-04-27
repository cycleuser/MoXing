"""
Enhanced GGUF metadata extraction for MoE detection and architecture analysis.
Based on kaiwu project patterns for performance optimization.
"""

import struct
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

GGUF_MAGIC = 0x46554747

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class ModelArchitecture:
    """Detailed model architecture information extracted from GGUF."""
    architecture: str
    is_moe: bool = False
    expert_count: int = 0
    expert_used_count: int = 0
    block_count: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    embedding_length: int = 0
    context_length: int = 4096
    quantization: str = "unknown"
    file_size_bytes: int = 0
    parameter_count: int = 0
    is_hybrid: bool = False
    has_ssm: bool = False
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_count_b(self) -> float:
        return self.parameter_count / 1e9

    @property
    def model_size_gb(self) -> float:
        return self.file_size_bytes / (1024 ** 3)

    @property
    def is_qwen3_moe(self) -> bool:
        return self.architecture == "qwen3" and self.is_moe

    @property
    def is_mixtral(self) -> bool:
        return self.architecture == "mixtral" or "moe" in self.architecture


@dataclass
class GGUFReader:
    """Low-level GGUF file reader for metadata extraction."""
    path: Path
    _fp: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensor_count: int = 0
    version: int = 0

    def read_metadata(self) -> Dict[str, Any]:
        with open(self.path, "rb") as f:
            self._fp = f
            magic = self._read_uint32()
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {hex(magic)}")

            self.version = self._read_uint32()
            self.tensor_count = self._read_uint64()
            metadata_kv_count = self._read_uint64()

            for _ in range(metadata_kv_count):
                key = self._read_string()
                value = self._read_value()
                self.metadata[key] = value

        return self.metadata

    def _read_uint32(self) -> int:
        return struct.unpack("<I", self._fp.read(4))[0]

    def _read_uint64(self) -> int:
        return struct.unpack("<Q", self._fp.read(8))[0]

    def _read_int32(self) -> int:
        return struct.unpack("<i", self._fp.read(4))[0]

    def _read_float32(self) -> float:
        return struct.unpack("<f", self._fp.read(4))[0]

    def _read_string(self) -> str:
        length = self._read_uint64()
        return self._fp.read(length).decode("utf-8", errors="replace")

    def _read_value(self) -> Any:
        value_type = self._read_uint32()

        if value_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", self._fp.read(1))[0]
        elif value_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", self._fp.read(1))[0]
        elif value_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", self._fp.read(2))[0]
        elif value_type == GGUF_TYPE_INT16:
            return struct.unpack("<h", self._fp.read(2))[0]
        elif value_type == GGUF_TYPE_UINT32:
            return self._read_uint32()
        elif value_type == GGUF_TYPE_INT32:
            return self._read_int32()
        elif value_type == GGUF_TYPE_FLOAT32:
            return self._read_float32()
        elif value_type == GGUF_TYPE_BOOL:
            return struct.unpack("<?", self._fp.read(1))[0]
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string()
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = self._read_uint32()
            array_length = self._read_uint64()
            return [self._read_value_by_type(array_type) for _ in range(array_length)]
        elif value_type == GGUF_TYPE_UINT64:
            return self._read_uint64()
        elif value_type == GGUF_TYPE_INT64:
            return struct.unpack("<q", self._fp.read(8))[0]
        elif value_type == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", self._fp.read(8))[0]
        else:
            return None

    def _read_value_by_type(self, value_type: int) -> Any:
        if value_type == GGUF_TYPE_UINT32:
            return self._read_uint32()
        elif value_type == GGUF_TYPE_INT32:
            return self._read_int32()
        elif value_type == GGUF_TYPE_FLOAT32:
            return self._read_float32()
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string()
        else:
            return self._read_value()


def extract_model_architecture(path: Path) -> ModelArchitecture:
    """Extract detailed model architecture from GGUF file."""
    reader = GGUFReader(path)
    metadata = reader.read_metadata()

    arch = metadata.get("general.architecture", "unknown")
    name = metadata.get("general.name", path.stem)
    quant = _get_quantization(metadata)
    file_size = path.stat().st_size

    param_count = metadata.get("general.parameter_count", 0)
    if isinstance(param_count, str):
        try:
            param_count = int(float(param_count))
        except (ValueError, TypeError):
            param_count = 0

    block_count = _get_int(metadata, f"{arch}.block_count", 0)
    head_count = _get_int(metadata, f"{arch}.attention.head_count", 0)
    head_count_kv = _get_int(metadata, f"{arch}.attention.head_count_kv", 0)
    embedding_length = _get_int(metadata, f"{arch}.embedding_length", 0)
    context_length = _get_int(metadata, f"{arch}.context_length", 4096)

    is_moe, expert_count, expert_used_count = _detect_moe(metadata, arch)
    is_hybrid, has_ssm = _detect_hybrid(metadata, arch)

    return ModelArchitecture(
        architecture=arch,
        is_moe=is_moe,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        block_count=block_count,
        head_count=head_count,
        head_count_kv=head_count_kv,
        embedding_length=embedding_length,
        context_length=context_length,
        quantization=quant,
        file_size_bytes=file_size,
        parameter_count=param_count,
        is_hybrid=is_hybrid,
        has_ssm=has_ssm,
        raw_metadata=metadata,
    )


def _get_int(metadata: Dict[str, Any], key: str, default: int = 0) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_quantization(metadata: Dict[str, Any]) -> str:
    file_type = metadata.get("general.file_type", 0)
    quant_map = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
        6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
        14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
        18: "Q6_K", 19: "Q8_K", 20: "IQ2_XXS", 21: "IQ2_XS",
        22: "IQ3_XXS", 23: "IQ1_S", 24: "IQ4_NL", 25: "IQ3_S",
        26: "IQ2_S", 27: "IQ4_XS", 28: "I8", 29: "I16", 30: "I32"
    }
    return quant_map.get(file_type, f"Unknown({file_type})")


def _detect_moe(metadata: Dict[str, Any], arch: str) -> Tuple[bool, int, int]:
    """Detect if model is MoE and extract expert counts."""
    is_moe = False
    expert_count = 0
    expert_used_count = 0

    moe_indicators = [
        f"{arch}.expert_count",
        f"{arch}.moe.expert_count",
        "general.architecture",
    ]

    for key in moe_indicators:
        if key in metadata:
            if "expert_count" in key:
                try:
                    expert_count = int(metadata[key])
                    if expert_count > 0:
                        is_moe = True
                except (ValueError, TypeError):
                    pass

    if "moe" in arch.lower():
        is_moe = True

    expert_used_keys = [
        f"{arch}.expert_used_count",
        f"{arch}.moe.expert_used_count",
        f"{arch}.attention.expert_used_count",
    ]

    for key in expert_used_keys:
        if key in metadata:
            try:
                expert_used_count = int(metadata[key])
            except (ValueError, TypeError):
                pass

    if is_moe and expert_used_count == 0:
        if "qwen3" in arch and expert_count == 128:
            expert_used_count = 8
        elif "mixtral" in arch and expert_count == 8:
            expert_used_count = 2
        elif "deepseek" in arch and expert_count >= 64:
            expert_used_count = 8

    return is_moe, expert_count, expert_used_count


def _detect_hybrid(metadata: Dict[str, Any], arch: str) -> Tuple[bool, bool]:
    """Detect hybrid architectures (SSM/DeltaNet/Mamba)."""
    is_hybrid = False
    has_ssm = False

    hybrid_patterns = [".ssm.", ".delta", ".recurrent", ".mamba"]
    arch_name = arch.lower()

    hybrid_archs = {"jamba", "rwkv", "mamba", "deltanet", "qwen3.5"}

    if arch_name in hybrid_archs:
        is_hybrid = True

    for key in metadata:
        for pattern in hybrid_patterns:
            if pattern in key.lower():
                has_ssm = True
                is_hybrid = True
                break

    return is_hybrid, has_ssm


def should_use_cpu_moe(arch: ModelArchitecture, force_gpu: bool = False) -> bool:
    """Determine if MoE experts should be offloaded to CPU."""
    if force_gpu:
        return False

    if not arch.is_moe:
        return False

    if arch.expert_count > 0 and arch.expert_count > arch.expert_used_count * 4:
        return True

    if arch.architecture in ("mixtral", "qwen3moe", "deepseek2", "deepseek3"):
        return True

    return False


def get_optimal_thread_count(arch: ModelArchitecture, cpu_cores: int, use_cpu_moe: bool = False) -> int:
    """Get optimal thread count based on model architecture."""
    if use_cpu_moe:
        return max(cpu_cores // 2, 4)
    return 2
