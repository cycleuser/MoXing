"""
Model downloading and management from ModelScope and HuggingFace
"""

import os
import re
import json
import time
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from rich.console import Console
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, BarColumn, TextColumn

console = Console()


DEFAULT_MODEL_DIR = Path.home() / ".cache" / "moxing" / "models"


@dataclass
class ModelInfo:
    name: str
    repo: str
    filename: str
    size_bytes: int = 0
    quantization: str = ""
    source: str = "huggingface"
    local_path: Optional[Path] = None
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)
    
    @property
    def is_downloaded(self) -> bool:
        if self.local_path:
            return self.local_path.exists()
        return False


class ModelDownloader:
    """Download GGUF models from ModelScope or HuggingFace."""
    
    HF_API = "https://huggingface.co/api"
    MS_API = "https://modelscope.cn/api/v1"
    HF_DOWNLOAD = "https://huggingface.co"
    MS_DOWNLOAD = "https://modelscope.cn/models"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_MODEL_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(follow_redirects=True, timeout=60)
    
    def search(
        self,
        query: str,
        source: str = "auto",
        limit: int = 20
    ) -> List[ModelInfo]:
        """Search for GGUF models."""
        results = []
        
        if source in ("auto", "huggingface"):
            results.extend(self._search_hf(query, limit))
        
        if source in ("auto", "modelscope") and len(results) < limit:
            results.extend(self._search_modelscope(query, limit - len(results)))
        
        return results[:limit]
    
    def _search_hf(self, query: str, limit: int) -> List[ModelInfo]:
        """Search HuggingFace for GGUF models."""
        results = []
        try:
            resp = self._client.get(
                f"{self.HF_API}/models",
                params={
                    "search": f"{query} gguf",
                    "limit": limit,
                    "filter": "gguf"
                }
            )
            resp.raise_for_status()
            
            for item in resp.json():
                repo_id = item.get("id", "")
                if not repo_id:
                    continue
                
                gguf_files = self._list_gguf_files_hf(repo_id)
                for filename, size in gguf_files[:3]:
                    quant = self._extract_quantization(filename)
                    results.append(ModelInfo(
                        name=repo_id.split("/")[-1],
                        repo=repo_id,
                        filename=filename,
                        size_bytes=size,
                        quantization=quant,
                        source="huggingface"
                    ))
        except Exception as e:
            console.print(f"[yellow]HuggingFace search error: {e}[/yellow]")
        
        return results
    
    def _search_modelscope(self, query: str, limit: int) -> List[ModelInfo]:
        """Search ModelScope for GGUF models."""
        results = []
        try:
            resp = self._client.get(
                f"{self.MS_API}/models",
                params={
                    "Name": query,
                    "PageSize": limit
                }
            )
            resp.raise_for_status()
            
            data = resp.json()
            models = data.get("Data", {}).get("Models", [])
            
            for item in models:
                repo_id = item.get("Path", "")
                if not repo_id:
                    continue
                
                gguf_files = self._list_gguf_files_modelscope(repo_id)
                for filename, size in gguf_files[:3]:
                    quant = self._extract_quantization(filename)
                    results.append(ModelInfo(
                        name=repo_id.split("/")[-1],
                        repo=repo_id,
                        filename=filename,
                        size_bytes=size,
                        quantization=quant,
                        source="modelscope"
                    ))
        except Exception as e:
            console.print(f"[yellow]ModelScope search error: {e}[/yellow]")
        
        return results
    
    def _list_gguf_files_hf(self, repo_id: str) -> List[tuple]:
        """List GGUF files in a HuggingFace repo."""
        files = []
        try:
            resp = self._client.get(f"{self.HF_API}/models/{repo_id}/tree/main")
            resp.raise_for_status()
            
            for item in resp.json():
                if item.get("type") == "file" and item.get("path", "").endswith(".gguf"):
                    files.append((item["path"], item.get("size", 0)))
            
            files.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            console.print(f"[dim]Could not list HF files: {e}[/dim]")
        
        return files
    
    def _list_gguf_files_modelscope(self, repo_id: str) -> List[tuple]:
        """List GGUF files in a ModelScope repo."""
        files = []
        try:
            resp = self._client.get(
                f"{self.MS_API}/models/{repo_id}/repo/files",
                timeout=30
            )
            resp.raise_for_status()
            
            data = resp.json()
            file_list = data.get("Data", {}).get("Files", [])
            if not file_list:
                file_list = data.get("Data", [])
            
            for item in file_list:
                name = item.get("Name", item.get("Path", ""))
                if name.endswith(".gguf"):
                    size = item.get("Size", 0)
                    files.append((name, size))
            
            files.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            console.print(f"[dim]Could not list ModelScope files: {e}[/dim]")
        
        return files
    
    def list_files(self, repo: str, source: str = "auto") -> List[tuple]:
        """List GGUF files in a repo."""
        source = self._detect_source(repo, source)
        
        if source == "modelscope":
            return self._list_gguf_files_modelscope(repo)
        else:
            return self._list_gguf_files_hf(repo)
    
    def _detect_source(self, repo: str, source: str) -> str:
        """Detect the source from repo string."""
        if source != "auto":
            return source
        
        if "modelscope" in repo.lower():
            return "modelscope"
        
        if repo.startswith("models--"):
            return "huggingface"
        
        return "huggingface"
    
    def _extract_quantization(self, filename: str) -> str:
        """Extract quantization type from filename."""
        filename = filename.lower()
        
        quants = [
            "q8_0", "q7_0", "q6_0", "q5_0", "q5_1", "q4_0", "q4_1",
            "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0",
            "iq4_xs", "iq4_nl", "iq3_m", "iq3_s", "iq2_xxs", "iq2_xs",
            "f32", "f16", "bf16"
        ]
        
        for q in quants:
            if q in filename:
                return q.upper()
        
        return ""
    
    def download(
        self,
        repo: str,
        filename: Optional[str] = None,
        source: str = "auto",
        output: Optional[Path] = None,
        progress: bool = True,
        callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """Download a GGUF model file."""
        source = self._detect_source(repo, source)
        
        files = self.list_files(repo, source)
        if not files:
            raise FileNotFoundError(f"No GGUF files found in {repo}")
        
        if filename:
            if "*" in filename:
                pattern = filename.replace("*", "").lower()
                matching = [f for f in files if pattern in f[0].lower()]
            else:
                matching = [f for f in files if f[0] == filename or f[0].endswith(filename)]
            if not matching:
                raise FileNotFoundError(f"File not found: {filename}")
            filename, size = matching[0]
        else:
            filename, size = files[0]
        
        if output is None:
            safe_repo = repo.replace("/", "__").replace("\\", "__")
            output = self.cache_dir / safe_repo / filename
        else:
            output = Path(output)
        
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if output.exists():
            console.print(f"[green]Model already exists: {output}[/green]")
            return output
        
        console.print(f"[blue]Downloading {filename} ({size / (1024**3):.2f} GB) from {source}...[/blue]")
        
        if source == "modelscope":
            url = f"{self.MS_DOWNLOAD}/{repo}/resolve/master/{filename}"
        else:
            url = f"{self.HF_DOWNLOAD}/{repo}/resolve/main/{filename}"
        
        temp_path = output.with_suffix(".downloading")
        
        try:
            self._download_file(url, temp_path, size, progress, callback)
            temp_path.rename(output)
            console.print(f"[green]Downloaded to: {output}[/green]")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
        
        return output
    
    def _download_file(
        self,
        url: str,
        output: Path,
        expected_size: int,
        show_progress: bool,
        callback: Optional[Callable[[int, int], None]]
    ):
        """Download a file with progress tracking."""
        with self._client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            
            total = int(resp.headers.get("content-length", expected_size))
            downloaded = 0
            
            if show_progress:
                progress = Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console
                )
                task = progress.add_task("Downloading", total=total)
                progress.start()
            else:
                progress = None
            
            try:
                with open(output, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192 * 16):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress:
                            progress.update(task, completed=downloaded)
                        
                        if callback:
                            callback(downloaded, total)
            finally:
                if progress:
                    progress.stop()
    
    def download_with_hf_hub(
        self,
        repo: str,
        filename: Optional[str] = None,
        token: Optional[str] = None
    ) -> Path:
        """Download using huggingface_hub library (faster for large files)."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")
        
        files = self.list_files(repo, "huggingface")
        if not files:
            raise FileNotFoundError(f"No GGUF files found in {repo}")
        
        if not filename:
            filename = files[0][0]
        
        return Path(hf_hub_download(
            repo_id=repo,
            filename=filename,
            token=token,
            local_dir=self.cache_dir / repo.replace("/", "__")
        ))
    
    def download_with_modelscope(
        self,
        repo: str,
        filename: Optional[str] = None
    ) -> Path:
        """Download using modelscope library."""
        try:
            from modelscope import snapshot_download
        except ImportError:
            raise ImportError("modelscope not installed. Run: pip install modelscope")
        
        files = self.list_files(repo, "modelscope")
        if not files:
            raise FileNotFoundError(f"No GGUF files found in {repo}")
        
        if not filename:
            filename = files[0][0]
        
        local_dir = snapshot_download(
            model_id=repo,
            allow_patterns=[filename]
        )
        
        return Path(local_dir) / filename
    
    def get_local_models(self) -> List[ModelInfo]:
        """Get list of locally cached models."""
        models = []
        
        if not self.cache_dir.exists():
            return models
        
        for repo_dir in self.cache_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            
            for gguf_file in repo_dir.glob("*.gguf"):
                size = gguf_file.stat().st_size
                quant = self._extract_quantization(gguf_file.name)
                
                repo = repo_dir.name.replace("__", "/")
                
                models.append(ModelInfo(
                    name=repo.split("/")[-1],
                    repo=repo,
                    filename=gguf_file.name,
                    size_bytes=size,
                    quantization=quant,
                    local_path=gguf_file
                ))
        
        return models
    
    def clear_cache(self, model: Optional[str] = None):
        """Clear model cache."""
        if model:
            for repo_dir in self.cache_dir.iterdir():
                if model in repo_dir.name:
                    import shutil
                    shutil.rmtree(repo_dir)
                    console.print(f"[green]Removed: {repo_dir}[/green]")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Cleared cache: {self.cache_dir}[/green]")


class ModelRegistry:
    """Registry of popular GGUF models."""
    
    POPULAR_MODELS = {
        "llama-3.2-3b": {
            "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "repo_ms": "LLM-Research/Llama-3.2-3B-Instruct-GGUF",
            "description": "Llama 3.2 3B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "llama-3.1-8b": {
            "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "repo_ms": "LLM-Research/Meta-Llama-3-8B-Instruct-GGUF",
            "description": "Llama 3.1 8B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "llama-3-8b": {
            "repo": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
            "repo_ms": "LLM-Research/Meta-Llama-3-8B-Instruct-GGUF",
            "description": "Llama 3 8B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "qwen2.5-7b": {
            "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "repo_ms": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "description": "Qwen 2.5 7B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "qwen2.5-3b": {
            "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
            "repo_ms": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "description": "Qwen 2.5 3B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "qwen2.5-14b": {
            "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
            "repo_ms": "Qwen/Qwen2.5-14B-Instruct-GGUF",
            "description": "Qwen 2.5 14B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M"]
        },
        "gemma-2-9b": {
            "repo": "bartowski/gemma-2-9b-it-GGUF",
            "repo_ms": "AI-ModelScope/gemma-2-9b-it-GGUF",
            "description": "Gemma 2 9B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "mistral-7b": {
            "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "repo_ms": "AI-ModelScope/Mistral-7B-Instruct-v0.3-GGUF",
            "description": "Mistral 7B Instruct v0.3",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "phi-3.5-mini": {
            "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
            "repo_ms": "LLM-Research/Phi-3.5-mini-instruct-GGUF",
            "description": "Phi 3.5 Mini Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "deepseek-coder-6.7b": {
            "repo": "bartowski/deepseek-coder-6.7B-instruct-GGUF",
            "repo_ms": "deepseek/deepseek-coder-6.7B-instruct-GGUF",
            "description": "DeepSeek Coder 6.7B Instruct",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
        "deepseek-v2-lite": {
            "repo": "bartowski/DeepSeek-V2-Lite-Chat-GGUF",
            "repo_ms": "deepseek/DeepSeek-V2-Lite-Chat-GGUF",
            "description": "DeepSeek V2 Lite 16B Chat",
            "sizes": ["Q4_K_M", "Q5_K_M"]
        },
        "yi-1.5-9b": {
            "repo": "bartowski/Yi-1.5-9B-Chat-GGUF",
            "repo_ms": "01ai/Yi-1.5-9B-Chat-GGUF",
            "description": "Yi 1.5 9B Chat",
            "sizes": ["Q4_K_M", "Q5_K_M", "Q8_0"]
        },
    }
    
    @classmethod
    def get_model_info(cls, name: str, source: str = "auto") -> Optional[Dict]:
        """Get info for a popular model."""
        info = cls.POPULAR_MODELS.get(name.lower())
        if not info:
            return None
        
        result = info.copy()
        if source == "modelscope" and "repo_ms" in info:
            result["repo"] = info["repo_ms"]
        
        return result
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """List all popular models."""
        return cls.POPULAR_MODELS


def download_model(
    repo: str,
    filename: Optional[str] = None,
    source: str = "auto",
    output: Optional[Path] = None,
    use_fast: bool = True
) -> Path:
    """Convenience function to download a model."""
    downloader = ModelDownloader()
    
    source = downloader._detect_source(repo, source)
    
    if use_fast:
        if source == "modelscope":
            try:
                return downloader.download_with_modelscope(repo, filename)
            except ImportError:
                pass
        else:
            try:
                return downloader.download_with_hf_hub(repo, filename)
            except ImportError:
                pass
    
    return downloader.download(repo, filename, source, output)