"""
Ollama integration for MoXing.

List and use models from local Ollama installation.
"""

import json
import os
import re
import socket
import struct
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError

from rich.console import Console
from rich.table import Table

console = Console()

OLLAMA_USER_MODELS_DIR = Path.home() / ".ollama" / "models"
OLLAMA_SYSTEM_MODELS_DIR = Path("/usr/share/ollama/.ollama/models")
OLLAMA_API_URL = "http://localhost:11434"

if sys.platform == "win32":
    OLLAMA_WINDOWS_MODELS_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ollama" / "models"

OLLAMA_MODELS_DIRS = [
    OLLAMA_USER_MODELS_DIR,
]

if sys.platform == "win32":
    OLLAMA_MODELS_DIRS.append(OLLAMA_WINDOWS_MODELS_DIR)
else:
    OLLAMA_MODELS_DIRS.append(OLLAMA_SYSTEM_MODELS_DIR)


def _check_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open with reliable timeout."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


@dataclass
class OllamaModel:
    """Represents a model in Ollama."""
    name: str
    tag: str
    id: str
    size: int
    modified: str
    digest: Optional[str] = None
    family: Optional[str] = None
    context_length: Optional[int] = None
    architecture: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None
    format: Optional[str] = None
    
    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)
    
    @property
    def full_name(self) -> str:
        if self.tag and self.tag != "latest":
            return f"{self.name}:{self.tag}"
        return self.name
    
    @property
    def context_str(self) -> str:
        if not self.context_length:
            return "-"
        if self.context_length >= 1024 * 100:
            return f"{self.context_length // 1024}K"
        elif self.context_length >= 1024:
            return f"{self.context_length // 1024}K"
        return str(self.context_length)


def read_gguf_context_length(gguf_path: str) -> Tuple[Optional[int], Optional[str]]:
    """Read context length and architecture from GGUF file metadata.
    
    Optimized version that reads only the first 32KB of metadata to avoid
    memory issues with large GGUF files.
    
    Args:
        gguf_path: Path to GGUF file
        
    Returns:
        Tuple of (context_length, architecture) or (None, None) if not found
    """
    context_length = None
    architecture = None
    
    try:
        with open(gguf_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return None, None
            
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            
            bytes_read = 20
            max_bytes = 32768
            
            for i in range(min(metadata_kv_count, 30)):
                if bytes_read > max_bytes:
                    break
                
                key_len_data = f.read(8)
                if len(key_len_data) < 8:
                    break
                key_len = struct.unpack('<Q', key_len_data)[0]
                
                if key_len > 256:
                    f.seek(key_len, 1)
                    bytes_read += 8 + key_len
                    value_type_data = f.read(4)
                    if len(value_type_data) < 4:
                        break
                    value_type = struct.unpack('<I', value_type_data)[0]
                    bytes_read += 4
                    if value_type == 8:
                        str_len_data = f.read(8)
                        if len(str_len_data) >= 8:
                            str_len = struct.unpack('<Q', str_len_data)[0]
                            f.seek(str_len, 1)
                            bytes_read += 8 + str_len
                    elif value_type == 9:
                        arr_type_data = f.read(4)
                        arr_len_data = f.read(8)
                        if len(arr_type_data) >= 4 and len(arr_len_data) >= 8:
                            arr_type = struct.unpack('<I', arr_type_data)[0]
                            arr_len = struct.unpack('<Q', arr_len_data)[0]
                            bytes_read += 12
                            f.seek(arr_len * 4, 1)
                            bytes_read += arr_len * 4
                    continue
                
                key_data = f.read(key_len)
                if len(key_data) < key_len:
                    break
                key = key_data.decode('utf-8', errors='replace')
                bytes_read += 8 + key_len
                
                value_type_data = f.read(4)
                if len(value_type_data) < 4:
                    break
                value_type = struct.unpack('<I', value_type_data)[0]
                bytes_read += 4
                
                value = None
                
                if value_type in [0, 1, 7]:
                    data = f.read(1)
                    if len(data) >= 1:
                        if value_type == 7:
                            value = struct.unpack('<?', data)[0]
                        else:
                            value = struct.unpack('<B', data)[0]
                    bytes_read += 1
                elif value_type in [2, 3]:
                    data = f.read(2)
                    if len(data) >= 2:
                        value = struct.unpack('<H', data)[0]
                    bytes_read += 2
                elif value_type in [4, 5, 6]:
                    data = f.read(4)
                    if len(data) >= 4:
                        value = struct.unpack('<I', data)[0]
                    bytes_read += 4
                elif value_type == 8:
                    str_len_data = f.read(8)
                    if len(str_len_data) < 8:
                        break
                    str_len = struct.unpack('<Q', str_len_data)[0]
                    if str_len > 256:
                        f.seek(str_len, 1)
                    else:
                        value = f.read(str_len).decode('utf-8', errors='replace')
                    bytes_read += 8 + str_len
                elif value_type == 9:
                    arr_type_data = f.read(4)
                    arr_len_data = f.read(8)
                    if len(arr_type_data) < 4 or len(arr_len_data) < 8:
                        break
                    arr_type = struct.unpack('<I', arr_type_data)[0]
                    arr_len = struct.unpack('<Q', arr_len_data)[0]
                    bytes_read += 12
                    
                    if arr_len > 10:
                        f.seek(arr_len * 8, 1)
                        bytes_read += arr_len * 8
                    else:
                        for _ in range(arr_len):
                            if arr_type in [0, 1, 7]:
                                f.read(1)
                                bytes_read += 1
                            elif arr_type in [2, 3]:
                                f.read(2)
                                bytes_read += 2
                            elif arr_type in [4, 5, 6]:
                                f.read(4)
                                bytes_read += 4
                
                if 'context_length' in key.lower() and value is not None and isinstance(value, int):
                    context_length = value
                    if architecture:
                        return context_length, architecture
                if key == 'general.architecture' and value is not None:
                    architecture = value
                    if context_length:
                        return context_length, architecture
                    
    except (MemoryError, OSError, struct.error):
        pass
    except Exception:
        pass
    
    return context_length, architecture


class OllamaClient:
    """Client for interacting with Ollama."""
    
    def __init__(self, api_url: str = OLLAMA_API_URL):
        self.api_url = api_url
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if self._available is not None:
            return bool(self._available)
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.api_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 11434
            
            if not _check_port_open(host, port, timeout=1.0):
                self._available = False
                return False
            
            req = Request(f"{self.api_url}/api/tags", method="GET")
            with urlopen(req, timeout=2) as resp:
                self._available = resp.status == 200
                return bool(self._available)
        except Exception:
            self._available = False
            return False
    
    def list_models(self) -> List[OllamaModel]:
        """List all models from Ollama with detailed information."""
        models = []
        
        if self.is_available():
            try:
                req = Request(f"{self.api_url}/api/tags")
                with urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                
                for m in data.get("models", []):
                    name = m.get("name", "")
                    if ":" in name:
                        base_name, tag = name.rsplit(":", 1)
                    else:
                        base_name = name
                        tag = "latest"
                    
                    details = m.get("details", {})
                    
                    model = OllamaModel(
                        name=base_name,
                        tag=tag,
                        id=m.get("digest", "")[:12],
                        size=m.get("size", 0),
                        modified=m.get("modified_at", ""),
                        digest=m.get("digest"),
                        family=details.get("family"),
                        parameter_size=details.get("parameter_size"),
                        quantization_level=details.get("quantization_level"),
                        format=details.get("format"),
                    )
                    
                    try:
                        info = self.get_model_info(name)
                        if info:
                            model_info_details = info.get("details", {})
                            if not model.parameter_size:
                                model.parameter_size = model_info_details.get("parameter_size")
                            if not model.quantization_level:
                                model.quantization_level = model_info_details.get("quantization_level")
                            if not model.format:
                                model.format = model_info_details.get("format")
                            
                            modelfile = info.get("modelfile", "")
                            if modelfile:
                                for line in modelfile.split("\n"):
                                    line_lower = line.lower()
                                    if line_lower.strip().startswith("parameter ") and "num_ctx" in line_lower:
                                        try:
                                            ctx_val = int(line.split()[-1])
                                            model.context_length = ctx_val
                                        except ValueError:
                                            pass
                    except Exception:
                        pass
                    
                    models.append(model)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get models from Ollama API: {e}[/yellow]")
        
        if not models:
            models = self._list_models_from_disk()
        
        return models
    
    def _list_models_from_disk(self) -> List[OllamaModel]:
        """List models by reading Ollama's manifest files."""
        models = []
        found = set()
        
        for models_dir in OLLAMA_MODELS_DIRS:
            manifests_dir = models_dir / "manifests" / "registry.ollama.ai"
            
            if not manifests_dir.exists():
                continue
            
            for owner_dir in manifests_dir.iterdir():
                if not owner_dir.is_dir():
                    continue
                
                for model_dir in owner_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    for tag_file in model_dir.iterdir():
                        if not tag_file.is_file():
                            continue
                        
                        model_name = f"{owner_dir.name}/{model_dir.name}" if owner_dir.name != "library" else model_dir.name
                        tag = tag_file.name
                        
                        key = f"{model_name}:{tag}"
                        if key in found:
                            continue
                        found.add(key)
                        
                        try:
                            size = self._get_model_size(tag_file, models_dir)
                        except:
                            size = 0
                        
                        models.append(OllamaModel(
                            name=model_name,
                            tag=tag,
                            id="",
                            size=size,
                            modified=""
                        ))
        
        return models
    
    def _get_model_size(self, manifest_path: Path, models_dir: Path) -> int:
        """Calculate model size from manifest."""
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            total_size = 0
            blobs_dir = models_dir / "blobs"
            
            for layer in manifest.get("layers", []):
                digest = layer.get("digest", "")
                if digest.startswith("sha256:"):
                    blob_path = blobs_dir / f"sha256-{digest[7:]}"
                    if blob_path.exists():
                        total_size += blob_path.stat().st_size
            
            return total_size
        except:
            return 0
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a model."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.api_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 11434
            
            if not _check_port_open(host, port, timeout=1.0):
                return None
            
            req = Request(
                f"{self.api_url}/api/show",
                data=json.dumps({"name": name}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None
    
    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the path to the model blob file (GGUF format)."""
        info = self.get_model_info(name)
        if info and "modelfile" in info:
            path = self._extract_path_from_modelfile(info["modelfile"])
            if path:
                return path
        
        return self._find_model_blob_from_manifest(name)
    
    def _extract_path_from_modelfile(self, modelfile: str) -> Optional[Path]:
        """Extract the GGUF path from modelfile's FROM line."""
        for line in modelfile.split("\n"):
            line = line.strip()
            if line.upper().startswith("FROM "):
                path_str = line[5:].strip()
                path = Path(path_str)
                if path.exists():
                    try:
                        with open(path, "rb") as f:
                            f.read(4)
                        return path
                    except PermissionError:
                        return None
        return None
    
    def check_model_access(self, name: str) -> tuple[bool, Optional[Path], str]:
        """Check if a model's GGUF file is accessible.
        
        Returns:
            (is_accessible, gguf_path, message)
        """
        info = self.get_model_info(name)
        if not info:
            return False, None, f"Model '{name}' not found in Ollama"
        
        if "modelfile" not in info:
            return False, None, f"Could not get modelfile for '{name}'"
        
        for line in info["modelfile"].split("\n"):
            line = line.strip()
            if line.upper().startswith("FROM "):
                path_str = line[5:].strip()
                path = Path(path_str)
                try:
                    if not path.exists():
                        return False, None, f"GGUF file not found: {path}"
                except PermissionError:
                    return False, path, (
                        f"Permission denied: {path}\n"
                        f"The model file is owned by the 'ollama' system user.\n"
                        f"Options:\n"
                        f"  1. Copy to user directory:\n"
                        f"     mkdir -p ~/ollama_models\n"
                        f"     sudo cp {path} ~/ollama_models/{name.replace(':', '_')}.gguf\n"
                        f"     sudo chown $USER:$USER ~/ollama_models/{name.replace(':', '_')}.gguf\n"
                        f"  2. Re-pull model (will store in ~/.ollama/models/):\n"
                        f"     ollama pull {name}\n"
                        f"  3. Use ollama directly instead of moxing:\n"
                        f"     ollama run {name}"
                    )
                try:
                    with open(path, "rb") as f:
                        f.read(4)
                    return True, path, "OK"
                except PermissionError:
                    return False, path, (
                        f"Permission denied: {path}\n"
                        f"The model file is owned by the 'ollama' system user.\n"
                        f"Options:\n"
                        f"  1. Copy to user directory:\n"
                        f"     mkdir -p ~/ollama_models\n"
                        f"     sudo cp {path} ~/ollama_models/{name.replace(':', '_')}.gguf\n"
                        f"     sudo chown $USER:$USER ~/ollama_models/{name.replace(':', '_')}.gguf\n"
                        f"  2. Re-pull model (will store in ~/.ollama/models/):\n"
                        f"     ollama pull {name}\n"
                        f"  3. Use ollama directly instead of moxing:\n"
                        f"     ollama run {name}"
                    )
        
        return False, None, f"No FROM line found in modelfile for '{name}'"
    
    def _find_model_blob_from_manifest(self, name: str) -> Optional[Path]:
        """Find model blob by checking manifests in all model directories."""
        manifest_path = self._find_manifest(name)
        if manifest_path:
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                
                for layer in manifest.get("layers", []):
                    if layer.get("mediaType") == "application/vnd.ollama.image.model":
                        digest = layer.get("digest", "")
                        if digest.startswith("sha256:"):
                            for models_dir in OLLAMA_MODELS_DIRS:
                                blob_path = models_dir / "blobs" / f"sha256-{digest[7:]}"
                                if blob_path.exists():
                                    return blob_path
            except:
                pass
        
        return None
    
    def get_model_gguf_path(self, name: str) -> Optional[Path]:
        """Get the GGUF file path for an Ollama model.
        
        Returns the actual GGUF blob file that can be used with llama.cpp.
        """
        return self.get_model_path(name)
    
    def _find_manifest(self, name: str) -> Optional[Path]:
        """Find the manifest file for a model in all model directories."""
        if ":" in name:
            model_name, tag = name.rsplit(":", 1)
        else:
            model_name = name
            tag = "latest"
        
        for models_dir in OLLAMA_MODELS_DIRS:
            manifests_dir = models_dir / "manifests" / "registry.ollama.ai"
            
            if not manifests_dir.exists():
                continue
            
            if "/" in model_name:
                owner, model = model_name.split("/", 1)
                manifest_path = manifests_dir / owner / model / tag
                if manifest_path.exists():
                    return manifest_path
            else:
                manifest_path = manifests_dir / "library" / model_name / tag
                if manifest_path.exists():
                    return manifest_path
        
        return None
    
    def is_embedding_model(self, name: str) -> bool:
        """Check if a model is an embedding model."""
        embedding_keywords = [
            "embed", "embedding", "bge", "nomic", "snowflake",
            "arctic-embed", "minilm", "granite-embedding"
        ]
        name_lower = name.lower()
        return any(kw in name_lower for kw in embedding_keywords)


def list_ollama_models() -> List[OllamaModel]:
    """List all models from local Ollama."""
    client = OllamaClient()
    return client.list_models()


def print_ollama_models(models: Optional[List[OllamaModel]] = None, 
                        show_embeddings: bool = True,
                        show_context: bool = True):
    """Print a table of Ollama models with detailed information from Ollama API."""
    if models is None:
        models = list_ollama_models()
    
    if not models:
        console.print("[yellow]No Ollama models found.[/yellow]")
        console.print("[dim]Make sure Ollama is installed and running.[/dim]")
        return
    
    if not show_embeddings:
        client = OllamaClient()
        models = [m for m in models if not client.is_embedding_model(m.full_name)]
    
    table = Table(title=f"Ollama Models ({len(models)} total)", expand=True)
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green", justify="right")
    if show_context:
        table.add_column("Context", style="magenta", justify="right")
    table.add_column("Params", style="blue", justify="right")
    table.add_column("Quant", style="yellow")
    table.add_column("Type", style="dim")
    
    client = OllamaClient()
    
    for m in sorted(models, key=lambda x: x.size, reverse=True):
        is_embed = client.is_embedding_model(m.full_name)
        model_type = "embedding" if is_embed else "llm"
        
        if m.size_gb >= 1:
            size_str = f"{m.size_gb:.1f}GB"
        else:
            size_str = f"{int(m.size / (1024**2))}MB"
        
        params_str = format_parameter_size(m.parameter_size)
        quant_str = m.quantization_level or "-"
        
        row = [m.full_name, size_str]
        
        if show_context:
            row.append(m.context_str)
        
        row.extend([params_str, quant_str, model_type])
        
        table.add_row(*row)
    
    console.print(table)


def format_parameter_size(param_str: Optional[str]) -> str:
    """Format parameter size string to compact representation.
    
    Examples:
        '31.3B' -> '31B'
        '4.3B' -> '4B'
        '334.0M' -> '334M'
        '999.8M' -> '1B'
        '33M' -> '33M'
        '30.15M' -> '30M'
    """
    if not param_str:
        return "-"
    
    param_str = param_str.strip()
    
    if param_str.endswith('B'):
        try:
            val = float(param_str[:-1])
            if val >= 1000:
                return f"{int(val)}B"
            elif val >= 1:
                return f"{int(round(val))}B"
            else:
                return f"{int(round(val * 1000))}M"
        except ValueError:
            return param_str
    elif param_str.endswith('M'):
        try:
            val = float(param_str[:-1])
            if val >= 950:
                return f"{int(round(val / 1000))}B"
            else:
                return f"{int(round(val))}M"
        except ValueError:
            return param_str
    else:
        return param_str


def get_ollama_model(name: str) -> Optional[OllamaModel]:
    """Get a specific Ollama model by name."""
    if ":" in name:
        base_name, tag = name.rsplit(":", 1)
    else:
        base_name = name
        tag = "latest"
    
    normalized_name = f"{base_name}:{tag}" if tag != "latest" else base_name
    
    models = list_ollama_models()
    for m in models:
        if m.name == base_name:
            if tag == "latest" and m.tag == "latest":
                return m
            if m.tag == tag:
                return m
        if m.full_name == normalized_name or m.full_name == name or m.name == name:
            return m
    return None