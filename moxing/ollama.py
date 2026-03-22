"""
Ollama integration for MoXing.

List and use models from local Ollama installation.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError

from rich.console import Console
from rich.table import Table

console = Console()

OLLAMA_USER_MODELS_DIR = Path.home() / ".ollama" / "models"
OLLAMA_SYSTEM_MODELS_DIR = Path("/usr/share/ollama/.ollama/models")
OLLAMA_API_URL = "http://localhost:11434"

OLLAMA_MODELS_DIRS = [
    OLLAMA_USER_MODELS_DIR,
    OLLAMA_SYSTEM_MODELS_DIR,
]


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
    
    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)
    
    @property
    def full_name(self) -> str:
        if self.tag and self.tag != "latest":
            return f"{self.name}:{self.tag}"
        return self.name


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
            req = Request(f"{self.api_url}/api/tags", method="GET")
            with urlopen(req, timeout=2) as resp:
                self._available = resp.status == 200
                return bool(self._available)
        except:
            self._available = False
            return False
    
    def list_models(self) -> List[OllamaModel]:
        """List all models from Ollama."""
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
                    
                    models.append(OllamaModel(
                        name=base_name,
                        tag=tag,
                        id=m.get("digest", "")[:12],
                        size=m.get("size", 0),
                        modified=m.get("modified_at", ""),
                        digest=m.get("digest"),
                        family=m.get("details", {}).get("family")
                    ))
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
            req = Request(
                f"{self.api_url}/api/show",
                data=json.dumps({"name": name}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except:
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
                        show_embeddings: bool = True):
    """Print a table of Ollama models."""
    if models is None:
        models = list_ollama_models()
    
    if not models:
        console.print("[yellow]No Ollama models found.[/yellow]")
        console.print("[dim]Make sure Ollama is installed and running.[/dim]")
        return
    
    if not show_embeddings:
        client = OllamaClient()
        models = [m for m in models if not client.is_embedding_model(m.full_name)]
    
    table = Table(title=f"Ollama Models ({len(models)} total)")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("ID", style="dim")
    
    client = OllamaClient()
    
    for m in sorted(models, key=lambda x: x.size, reverse=True):
        is_embed = client.is_embedding_model(m.full_name)
        model_type = "embedding" if is_embed else "llm"
        table.add_row(
            m.full_name,
            f"{m.size_gb:.1f} GB" if m.size_gb >= 1 else f"{m.size / (1024**2):.0f} MB",
            model_type,
            m.id[:8] if m.id else "-"
        )
    
    console.print(table)


def get_ollama_model(name: str) -> Optional[OllamaModel]:
    """Get a specific Ollama model by name."""
    models = list_ollama_models()
    for m in models:
        if m.full_name == name or m.name == name:
            return m
    return None