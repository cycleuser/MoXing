"""
Ollama Runner - 使用 moxing 编译的 Ollama runner 直接运行 Ollama 模型

核心创新点：
1. 使用 Ollama 的 patched llama.cpp 支持所有 Ollama 模型（包括 gemma4）
2. 支持灵活设备选择 (-d gpu0, gpu1, etc.)
3. 支持所有后端 (CUDA/ROCm/Vulkan/CPU)
4. 比系统 Ollama 更灵活的参数控制

用法：
    moxing ollama serve gemma4:31b -b cuda -d gpu0
    moxing ollama run gemma4:31b -b rocm -d gpu1 -c 65536
"""

import os
import sys
import json
import time
import signal
import socket
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError

from rich.console import Console
from rich.panel import Panel

console = Console()

OLLAMA_ONLY_ARCHITECTURES = [
    'gemma4', 'gemma4.it', 'gemma4-text',
    'deepseek3', 'deepseek3-text',
]


@dataclass
class RunnerConfig:
    """Runner 配置"""
    backend: str
    device: str
    port: int
    host: str
    ctx_size: int
    n_gpu_layers: int = -1
    threads: int = 0
    batch_size: int = 2048
    flash_attn: bool = True
    kv_cache: str = "f16"


class OllamaModelResolver:
    """解析 Ollama 模型路径"""
    
    OLLAMA_MODELS_DIRS = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
    ]
    
    def __init__(self):
        self._manifest_cache: Dict[str, Path] = {}
    
    def resolve(self, model_name: str) -> Optional[Path]:
        """
        解析模型名称到 GGUF 文件路径
        
        Args:
            model_name: 如 "gemma4:31b", "llama3:8b"
            
        Returns:
            GGUF 文件路径
        """
        # 解析名称和标签
        if ":" in model_name:
            name, tag = model_name.rsplit(":", 1)
        else:
            name = model_name
            tag = "latest"
        
        # 标准化名称
        if "/" not in name:
            name = f"library/{name}"
        
        # 查找 manifest
        manifest = self._find_manifest(name, tag)
        if not manifest:
            return None
        
        # 解析 manifest 获取 blobs
        return self._get_model_blob(manifest)
    
    def _find_manifest(self, name: str, tag: str) -> Optional[Path]:
        """查找 manifest 文件"""
        cache_key = f"{name}:{tag}"
        if cache_key in self._manifest_cache:
            return self._manifest_cache[cache_key]
        
        for models_dir in self.OLLAMA_MODELS_DIRS:
            manifest_path = models_dir / "manifests" / "registry.ollama.ai" / name / tag
            if manifest_path.exists():
                self._manifest_cache[cache_key] = manifest_path
                return manifest_path
        
        return None
    
    def _get_model_blob(self, manifest_path: Path) -> Optional[Path]:
        """从 manifest 获取模型 blob 路径"""
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            for layer in manifest.get("layers", []):
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    digest = layer.get("digest", "")
                    if digest.startswith("sha256:"):
                        # 查找 blobs
                        models_dir = manifest_path.parent.parent.parent.parent.parent
                        blob_path = models_dir / "blobs" / f"sha256-{digest[7:]}"
                        if blob_path.exists():
                            return blob_path
            
            # 尝试配置文件中的 FROM
            config_digest = manifest.get("config", {}).get("digest", "")
            if config_digest.startswith("sha256:"):
                models_dir = manifest_path.parent.parent.parent.parent.parent
                config_path = models_dir / "blobs" / f"sha256-{config_digest[7:]}"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    # 尝试从 modelfile 解析
                    
        except Exception as e:
            console.print(f"[yellow]解析 manifest 失败: {e}[/yellow]")
        
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用的 Ollama 模型"""
        models = []
        
        for models_dir in self.OLLAMA_MODELS_DIRS:
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
                        
                        name = f"{owner_dir.name}/{model_dir.name}" if owner_dir.name != "library" else model_dir.name
                        
                        # 获取大小
                        blob = self._get_model_blob(tag_file)
                        size = blob.stat().st_size if blob else 0
                        
                        models.append({
                            "name": name,
                            "tag": tag_file.name,
                            "full_name": f"{name}:{tag_file.name}",
                            "size": size,
                            "size_gb": size / (1024**3),
                            "blob_path": blob,
                        })
        
        return sorted(models, key=lambda x: x["size"], reverse=True)


class OllamaRunnerBinary:
    """管理 Ollama Runner 二进制文件"""
    
    def __init__(self, backend: str = "auto", device: str = "auto"):
        self.backend = backend
        self.device = device
        self.bin_dir = self._get_bin_dir()
        self.runner_path = self.bin_dir / "llama-server"
    
    def _get_bin_dir(self) -> Path:
        """获取二进制目录"""
        moxing_dir = Path(__file__).parent
        platform = self._detect_platform()
        
        backend = self.backend if self.backend != "auto" else "cpu"
        
        # 优先使用 Ollama 版本
        ollama_bin = moxing_dir / "bin" / f"ollama-{platform}-{backend}"
        if ollama_bin.exists():
            return ollama_bin
        
        # 回退到标准版本
        standard_bin = moxing_dir / "bin" / f"{platform}-{backend}"
        if standard_bin.exists():
            return standard_bin
        
        # 最后尝试从 binaries_ollama_new
        legacy_bin = Path(moxing_dir.parent) / "binaries_ollama_new" / f"linux-x64-{backend}"
        if legacy_bin.exists():
            return legacy_bin
        
        raise FileNotFoundError(f"找不到 {backend} 后端的二进制文件")
    
    def _detect_platform(self) -> str:
        """检测平台"""
        if sys.platform == "darwin":
            arch = "arm64" if os.uname().machine == "arm64" else "x64"
            return f"darwin-{arch}"
        elif sys.platform == "win32":
            return "windows-x64"
        else:
            return "linux-x64"
    
    def is_available(self) -> bool:
        """检查 runner 是否可用"""
        return self.runner_path.exists()
    
    def get_version(self) -> Optional[str]:
        """获取 runner 版本"""
        try:
            result = subprocess.run(
                [str(self.runner_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except:
            return None


class OllamaRunnerServer:
    """Ollama Runner 服务器"""
    
    def __init__(
        self,
        model_path: Path,
        config: RunnerConfig,
        verbose: bool = False
    ):
        self.model_path = model_path
        self.config = config
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        
        # 初始化 runner
        self.runner = OllamaRunnerBinary(config.backend, config.device)
    
    def start(self, wait_ready: bool = True) -> bool:
        """启动服务器"""
        if not self.runner.is_available():
            console.print(f"[red]Runner 不可用: {self.runner.runner_path}[/red]")
            return False
        
        # 准备命令
        cmd = self._build_command()
        
        # 准备环境
        env = self._prepare_env()
        
        if self.verbose:
            console.print(f"[dim]命令: {' '.join(cmd)}[/dim]")
        
        # 启动进程
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not self.verbose else None,
            stderr=subprocess.PIPE if not self.verbose else None,
            text=True,
            env=env,
            cwd=str(self.runner.bin_dir)
        )
        
        if wait_ready:
            return self._wait_for_ready()
        
        return True
    
    def _build_command(self) -> List[str]:
        """构建运行命令"""
        cmd = [
            str(self.runner.runner_path),
            "-m", str(self.model_path),
            "--port", str(self.config.port),
            "--host", self.config.host,
            "-c", str(self.config.ctx_size),
        ]
        
        # GPU 层数
        if self.config.n_gpu_layers >= 0:
            cmd.extend(["-ngl", str(self.config.n_gpu_layers)])
        else:
            cmd.extend(["-ngl", "999"])  # 默认全部 GPU
        
        # 线程数
        if self.config.threads > 0:
            cmd.extend(["-t", str(self.config.threads)])
        
        # Flash attention
        if self.config.flash_attn:
            cmd.append("-fa")
        
        # 批大小
        if self.config.batch_size > 0:
            cmd.extend(["-b", str(self.config.batch_size)])
        
        # KV 缓存类型
        if self.config.kv_cache and self.config.kv_cache != "f16":
            cmd.extend(["-ctk", self.config.kv_cache])
            cmd.extend(["-ctv", self.config.kv_cache])
        
        return cmd
    
    def _prepare_env(self) -> Dict[str, str]:
        """准备环境变量"""
        env = os.environ.copy()
        
        # 库路径
        lib_path = str(self.runner.bin_dir)
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = lib_path
        
        # 设备选择
        device = self.config.device
        backend = self.config.backend
        
        if device.startswith("gpu"):
            gpu_id = int(device.replace("gpu", ""))
            
            if backend == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                if self.verbose:
                    console.print(f"[blue]使用 CUDA GPU {gpu_id}[/blue]")
            elif backend == "rocm":
                env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
                if self.verbose:
                    console.print(f"[blue]使用 ROCm GPU {gpu_id}[/blue]")
            # Vulkan 通过 llama.cpp 参数处理
        
        return env
    
    def _wait_for_ready(self, timeout: float = 60.0) -> bool:
        """等待服务器就绪"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                # 进程已退出
                stdout, stderr = self.process.communicate()
                console.print(f"[red]服务器启动失败[/red]")
                if stderr:
                    console.print(f"[red]{stderr}[/red]")
                return False
            
            # 检查端口
            if self._check_port():
                return True
            
            time.sleep(0.5)
        
        console.print(f"[red]等待服务器超时[/red]")
        return False
    
    def _check_port(self) -> bool:
        """检查端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((self.config.host, self.config.port))
            sock.close()
            return result == 0
        except:
            return False
    
    def stop(self):
        """停止服务器"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.process is not None and self.process.poll() is None
    
    @property
    def base_url(self) -> str:
        """获取 API URL"""
        return f"http://{self.config.host}:{self.config.port}"


def is_ollama_specific_model(model_name: str) -> bool:
    """检查是否是 Ollama 特定模型"""
    model_lower = model_name.lower()
    for arch in OLLAMA_ONLY_ARCHITECTURES:
        if arch in model_lower:
            return True
    return False


def serve_ollama_model(
    model_name: str,
    backend: str = "auto",
    device: str = "auto",
    port: int = 8080,
    host: str = "127.0.0.1",
    ctx_size: int = 32768,
    verbose: bool = False,
    **kwargs
) -> Optional[OllamaRunnerServer]:
    """
    服务 Ollama 模型
    
    这是主要的入口函数，用于从 moxing CLI 调用
    
    Args:
        model_name: Ollama 模型名称，如 "gemma4:31b"
        backend: 后端类型 (cuda/rocm/vulkan/cpu/auto)
        device: 设备选择 (gpu0/gpu1/.../auto)
        port: 服务端口
        host: 服务主机
        ctx_size: 上下文大小
        verbose: 详细输出
        
    Returns:
        OllamaRunnerServer 实例
    """
    # 解析模型路径
    resolver = OllamaModelResolver()
    model_path = resolver.resolve(model_name)
    
    if not model_path:
        console.print(f"[red]找不到模型: {model_name}[/red]")
        console.print("[dim]请确保模型已下载: ollama pull {model_name}[/dim]")
        return None
    
    console.print(f"[blue]模型路径: {model_path}[/blue]")
    
    # 自动检测后端
    if backend == "auto":
        from moxing.device import DeviceDetector
        detector = DeviceDetector()
        detector.detect()
        best = detector.get_best_device(model_path.stat().st_size / (1024**3))
        backend = best.backend.value
        console.print(f"[blue]自动选择后端: {backend}[/blue]")
    
    # 创建配置
    config = RunnerConfig(
        backend=backend,
        device=device,
        port=port,
        host=host,
        ctx_size=ctx_size,
        **kwargs
    )
    
    # 创建并启动服务器
    server = OllamaRunnerServer(model_path, config, verbose)
    
    if server.start(wait_ready=True):
        console.print(Panel(
            f"[green]模型:[/green] {model_name}\n"
            f"[green]后端:[/green] {backend}\n"
            f"[green]设备:[/green] {device}\n"
            f"[green]API:[/green] {server.base_url}/v1\n"
            f"[yellow]按 Ctrl+C 停止[/yellow]",
            title="Ollama Runner 已启动"
        ))
        return server
    else:
        console.print("[red]启动失败[/red]")
        return None


def run_ollama_model(
    model_name: str,
    prompt: str,
    backend: str = "auto",
    device: str = "auto",
    **kwargs
) -> Optional[str]:
    """
    运行单次推理
    
    Args:
        model_name: 模型名称
        prompt: 提示词
        backend: 后端
        device: 设备
        
    Returns:
        生成的文本
    """
    import random
    
    port = random.randint(18000, 19000)
    
    server = serve_ollama_model(
        model_name=model_name,
        backend=backend,
        device=device,
        port=port,
        **kwargs
    )
    
    if not server:
        return None
    
    try:
        # 使用 OpenAI API 格式调用
        from moxing.client import Client
        client = Client(server.base_url)
        
        response = client.chat.completions.create(
            model="model",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        return response.choices[0].get("message", {}).get("content", "")
    finally:
        server.stop()


# CLI 测试
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python -m moxing.ollama_runner <model_name> [prompt]")
        print("  python -m moxing.ollama_runner gemma4:31b")
        print("  python -m moxing.ollama_runner gemma4:31b 'Hello, world!'")
        sys.exit(1)
    
    model = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else None
    
    if prompt:
        result = run_ollama_model(model, prompt, backend="auto", verbose=True)
        if result:
            print(f"\n结果:\n{result}")
    else:
        server = serve_ollama_model(model, backend="auto", verbose=True)
        if server:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n停止...")
            finally:
                server.stop()
