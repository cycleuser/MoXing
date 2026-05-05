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

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

from rich.console import Console
from rich.panel import Panel

console = Console()

OLLAMA_ONLY_ARCHITECTURES = [
    "gemma4",
    "gemma4.it",
    "gemma4-text",
    "deepseek3",
    "deepseek3-text",
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
    ubatch_size: int = 512
    flash_attn: bool = True
    kv_cache: str = "f16"
    backend_index: int = -1
    runner_type: str = "ollama"
    verbose_runner: bool = False
    fit_mode: str = "auto"
    lookahead: int = 0
    cache_prompts: bool = False
    slots: int = 1
    cont_batching: bool = True
    mlock: bool = False
    no_kv_offload: bool = False
    rope_scaling: str = "none"
    rope_scale: float = 1.0
    speculative_draft: Optional[str] = None
    speculative_max: int = 5
    speculative_pmin: float = 0.75
    cpu_moe: bool = False


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
            with open(manifest_path) as f:
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
                    with open(config_path) as f:
                        json.load(f)
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

                        name = (
                            f"{owner_dir.name}/{model_dir.name}"
                            if owner_dir.name != "library"
                            else model_dir.name
                        )

                        # 获取大小
                        blob = self._get_model_blob(tag_file)
                        size = blob.stat().st_size if blob else 0

                        models.append(
                            {
                                "name": name,
                                "tag": tag_file.name,
                                "full_name": f"{name}:{tag_file.name}",
                                "size": size,
                                "size_gb": size / (1024**3),
                                "blob_path": blob,
                            }
                        )

        return sorted(models, key=lambda x: x["size"], reverse=True)


class OllamaRunnerBinary:
    """管理 Ollama Runner 二进制文件"""

    def __init__(self, backend: str = "auto", device: str = "auto", runner_type: str = "ollama"):
        self.backend = backend
        self.device = device
        self.runner_type = runner_type  # "ollama" or "official"
        self.bin_dir = self._get_bin_dir()
        self.runner_path = self._get_runner_path()

    def _get_runner_path(self) -> Path:
        """获取 runner 可执行文件路径"""
        candidates = []
        if self.runner_type == "ollama":
            candidates = [
                self.bin_dir / f"ollama-runner-{self.backend}",
                self.bin_dir / "ollama-runner",
                self.bin_dir / "llama-server",
            ]
        else:
            candidates = [
                self.bin_dir / "llama-server",
            ]

        for path in candidates:
            if path.is_symlink():
                real_path = path.resolve()
                if real_path.exists():
                    return path
            elif path.exists():
                return path

        return candidates[0]

    def _get_bin_dir(self) -> Path:
        """获取二进制目录"""
        moxing_dir = Path(__file__).parent
        platform = self._detect_platform()

        backend = self.backend if self.backend != "auto" else "cpu"

        # Search in order: ollama-specific, official, standard
        candidate_names = []
        if self.runner_type == "ollama":
            candidate_names.extend(
                [
                    f"ollama-{platform}-{backend}",
                    f"{platform}-{backend}",
                ]
            )
        elif self.runner_type == "official":
            candidate_names.extend(
                [
                    f"official-{platform}-{backend}",
                    f"{platform}-{backend}",
                ]
            )
        else:
            candidate_names.extend(
                [
                    f"{self.runner_type}-{platform}-{backend}",
                    f"{platform}-{backend}",
                ]
            )

        for name in candidate_names:
            bin_path = moxing_dir / "bin" / name
            if bin_path.exists():
                return bin_path

        binaries_str = ', '.join(candidate_names)
        raise FileNotFoundError(
            f"找不到 {self.runner_type} {backend} 后端的二进制文件: "
            f"{moxing_dir}/bin/[{binaries_str}]"
        )

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
                [str(self.runner_path), "--version"], capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except:  # noqa: E722
            return None


class OllamaRunnerDownloader:
    """从 Ollama 官方下载预编译的 runner"""

    OLLAMA_REPO = "ollama/ollama"
    OLLAMA_VERSION = "v0.6.8"
    CACHE_DIR = Path.home() / ".cache" / "moxing" / "ollama-runners"

    PLATFORM_PACKAGES = {
        "linux-x64": {
            "cuda": "ollama-linux-amd64.tar.zst",
            "vulkan": "ollama-linux-amd64.tar.zst",
            "cpu": "ollama-linux-amd64.tar.zst",
            "rocm": "ollama-linux-amd64-rocm.tar.zst",
        },
        "linux-arm64": {
            "cuda_jetpack5": "ollama-linux-arm64-jetpack5.tar.zst",
            "cuda_jetpack6": "ollama-linux-arm64-jetpack6.tar.zst",
            "cpu": "ollama-linux-arm64.tar.zst",
        },
        "darwin-arm64": {
            "metal": "ollama-darwin.tgz",
        },
        "windows-x64": {
            "cuda": "ollama-windows-amd64.zip",
            "vulkan": "ollama-windows-amd64.zip",
            "cpu": "ollama-windows-amd64.zip",
            "rocm": "ollama-windows-amd64-rocm.zip",
            "mlx": "ollama-windows-amd64-mlx.zip",
        },
    }

    def __init__(self, version: Optional[str] = None):
        self.version = version or self.OLLAMA_VERSION
        self.cache_dir = self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def detect_platform(self) -> str:
        """检测当前平台"""
        if sys.platform == "darwin":
            arch = "arm64" if os.uname().machine == "arm64" else "x64"
            return f"darwin-{arch}"
        elif sys.platform == "win32":
            return "windows-x64"
        else:
            arch = "arm64" if os.uname().machine in ["arm64", "aarch64"] else "x64"
            return f"linux-{arch}"

    def get_bin_dir(self, backend: str) -> Path:
        """获取 runner 安装目录"""
        platform = self.detect_platform()
        moxing_dir = Path(__file__).parent
        return moxing_dir / "bin" / f"ollama-{platform}-{backend}"

    def has_runner(self, backend: str) -> bool:
        """检查 runner 是否已安装"""
        bin_dir = self.get_bin_dir(backend)
        if not bin_dir.exists():
            return False
        runner = bin_dir / f"ollama-runner-{backend}"
        return runner.exists()

    def list_available_backends(self) -> List[str]:
        """列出当前平台可用的后端"""
        platform = self.detect_platform()
        return list(self.PLATFORM_PACKAGES.get(platform, {}).keys())

    def download_runner(self, backend: str = "cuda", force: bool = False) -> Path:
        """下载并安装 runner"""
        platform = self.detect_platform()

        if platform not in self.PLATFORM_PACKAGES:
            raise ValueError(f"不支持的平台: {platform}")

        packages = self.PLATFORM_PACKAGES[platform]
        if backend not in packages:
            raise ValueError(f"平台 {platform} 不支持后端 {backend}，可用: {list(packages.keys())}")

        bin_dir = self.get_bin_dir(backend)

        if not force and self.has_runner(backend):
            console.print(f"[green]Runner 已安装: {bin_dir}[/green]")
            return bin_dir

        package_name = packages[backend]
        download_url = (
            f"https://github.com/{self.OLLAMA_REPO}/releases/download/{self.version}/{package_name}"
        )

        console.print(f"[blue]下载 Ollama runner ({backend})...[/blue]")
        console.print(f"[dim]版本: {self.version}[/dim]")
        console.print(f"[dim]包: {package_name}[/dim]")
        console.print(f"[dim]目标: {bin_dir}[/dim]")

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            archive_path = tmpdir / package_name

            self._download_file(download_url, archive_path)

            extract_dir = tmpdir / "extract"
            self._extract_archive(archive_path, extract_dir)

            bin_dir.mkdir(parents=True, exist_ok=True)
            self._copy_files(extract_dir, bin_dir, backend)

        version_file = bin_dir / "VERSION"
        version_file.write_text(f"{self.version}\n{backend}\nollama\n")

        console.print(f"[green]安装完成: {bin_dir}[/green]")
        return bin_dir

    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """下载所有可用的 runner"""
        results = {}
        for backend in self.list_available_backends():
            try:
                self.download_runner(backend, force=force)
                results[backend] = True
            except Exception as e:
                console.print(f"[red]下载 {backend} 失败: {e}[/red]")
                results[backend] = False
        return results

    def _download_file(self, url: str, dest: Path):
        """下载文件"""
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        req = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "moxing"})

        with urlopen(req, timeout=600) as response:
            total = int(response.headers.get("content-length", 0))

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("下载中", total=total)
                downloaded = 0
                chunk_size = 8192 * 16

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task, completed=downloaded)

    def _extract_archive(self, archive_path: Path, dest_dir: Path):
        """解压归档"""
        dest_dir.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == ".zst":
            try:
                result = subprocess.run(
                    ["zstd", "-d", "-c", str(archive_path)], capture_output=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"解压失败: {result.stderr.decode()}")

                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                    tmp.write(result.stdout)
                    tmp_path = Path(tmp.name)

                try:
                    import tarfile

                    with tarfile.open(tmp_path, "r") as tar:
                        tar.extractall(dest_dir)
                finally:
                    tmp_path.unlink()
            except FileNotFoundError as e:
                raise RuntimeError("需要安装 zstd: apt install zstd 或 brew install zstd") from e
        elif archive_path.suffix == ".tgz" or archive_path.name.endswith(".tar.gz"):
            import tarfile

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(dest_dir)
        elif archive_path.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        else:
            raise ValueError(f"不支持的归档格式: {archive_path.suffix}")

    def _copy_files(self, extract_dir: Path, bin_dir: Path, backend: str):
        """复制 runner 文件"""
        lib_dir = extract_dir / "lib" / "ollama"

        if not lib_dir.exists():
            lib_dir = extract_dir / "lib"

        backend_subdirs = {
            "cuda": ["cuda_v13", "cuda_v12"],
            "rocm": ["rocm"],
            "vulkan": ["vulkan"],
            "mlx": ["mlx"],
            "cpu": [],
            "cuda_jetpack5": ["cuda_jetpack5"],
            "cuda_jetpack6": ["cuda_jetpack6"],
        }

        copied = []

        if lib_dir.exists():
            for item in lib_dir.iterdir():
                if item.is_file() and (
                    item.suffix in [".so", ".dylib", ".dll"]
                    or ".so." in item.name
                    or ".dylib." in item.name
                ):
                    shutil.copy2(item, bin_dir / item.name)
                    copied.append(item.name)

        subdirs = backend_subdirs.get(backend, [])
        for subdir in subdirs:
            backend_dir = (
                lib_dir / subdir if lib_dir.exists() else extract_dir / "lib" / "ollama" / subdir
            )
            if backend_dir.exists():
                for item in backend_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, bin_dir / item.name)
                        copied.append(item.name)

        bin_subdir = extract_dir / "bin"
        if bin_subdir.exists():
            for item in bin_subdir.iterdir():
                if item.is_file() and item.name.startswith("ollama"):
                    target_name = (
                        f"ollama-runner-{backend}" if backend not in ["cpu"] else "ollama-runner"
                    )
                    shutil.copy2(item, bin_dir / target_name)
                    os.chmod(bin_dir / target_name, 0o755)
                    copied.append(target_name)

        for name in copied:
            console.print(f"  [green]{name}[/green]")

        if not copied:
            console.print("  [yellow]未找到文件[/yellow]")

    def use_system_ollama(self, backend: str = "cuda") -> Optional[Path]:
        """使用系统安装的 Ollama 库"""
        ollama_lib_dirs = [
            Path("/usr/local/lib/ollama"),
            Path("/usr/lib/ollama"),
            Path("/opt/ollama/lib"),
        ]

        for lib_dir in ollama_lib_dirs:
            if not lib_dir.exists():
                continue

            backend_map = {
                "cuda": ["cuda_v13", "cuda_v12"],
                "rocm": ["rocm"],
                "vulkan": ["vulkan"],
            }

            subdirs = backend_map.get(backend, [])
            for subdir in subdirs:
                backend_dir = lib_dir / subdir
                if backend_dir.exists():
                    console.print(f"[green]使用系统 Ollama 库: {backend_dir}[/green]")
                    return backend_dir

        return None


def download_ollama_runner(backend: str = "auto", version: Optional[str] = None) -> Path:
    """下载 Ollama runner (便捷函数)"""
    downloader = OllamaRunnerDownloader(version)

    if backend == "auto":
        backends = downloader.list_available_backends()
        backend = backends[0] if backends else "cuda"

    return downloader.download_runner(backend)


class OllamaRunnerServer:
    """Ollama Runner 服务器"""

    def __init__(self, model_path: Path, config: RunnerConfig, verbose: bool = False):
        self.model_path = model_path
        self.config = config
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()

        # 初始化 runner
        self.runner = OllamaRunnerBinary(config.backend, config.device, config.runner_type)

    def start(self, wait_ready: bool = True) -> bool:
        """启动服务器"""
        if not self.runner.is_available():
            console.print(f"[red]Runner 不可用: {self.runner.runner_path}[/red]")
            return False

        # 准备命令
        cmd = self._build_command()

        # 准备环境
        env = self._prepare_env()

        # 始终显示命令
        console.print("[blue]命令:[/blue]")
        console.print(f"[dim]{' '.join(cmd)}[/dim]")
        console.print(f"[blue]Runner:[/blue] {self.runner.runner_path}")
        console.print(f"[blue]工作目录:[/blue] {self.runner.bin_dir}")
        console.print("")

        # 启动进程 - 输出直接显示到终端
        self.process = subprocess.Popen(
            cmd,
            stdout=None,  # 直接输出到终端
            stderr=None,  # 直接输出到终端
            text=True,
            env=env,
            cwd=str(self.runner.bin_dir),
        )

        if wait_ready:
            return self._wait_for_ready()

        return True

    def _build_command(self) -> List[str]:
        """构建运行命令"""
        cmd = [
            str(self.runner.runner_path),
            "-m",
            str(self.model_path),
            "--port",
            str(self.config.port),
            "--host",
            self.config.host,
            "-c",
            str(self.config.ctx_size),
        ]

        # GPU 层数
        if self.config.cpu_moe:
            cmd.extend(["-ngl", "999"])
            cmd.append("--cpu-moe")
        elif self.config.n_gpu_layers >= 0:
            cmd.extend(["-ngl", str(self.config.n_gpu_layers)])
        else:
            cmd.extend(["-ngl", "999"])  # 默认全部 GPU

        # 线程数
        if self.config.threads > 0:
            cmd.extend(["-t", str(self.config.threads)])

        # Flash attention
        if self.config.flash_attn:
            cmd.extend(["-fa", "on"])

        # 批大小
        if self.config.batch_size > 0:
            cmd.extend(["-b", str(self.config.batch_size)])

        # UBatch 大小
        if self.config.ubatch_size > 0:
            cmd.extend(["-ub", str(self.config.ubatch_size)])

        # KV 缓存类型
        if self.config.kv_cache and self.config.kv_cache not in ("f16", "auto"):
            cmd.extend(["-ctk", self.config.kv_cache])
            cmd.extend(["-ctv", self.config.kv_cache])

        # Fit 模式
        if self.config.fit_mode and self.config.fit_mode != "auto":
            cmd.extend(["--fit", self.config.fit_mode])

        # Lookahead 解码
        if self.config.lookahead > 0:
            cmd.extend(["--lookahead", str(self.config.lookahead)])

        # Prompt 缓存
        if self.config.cache_prompts:
            cmd.append("--cache-prompts")

        # 并发槽位
        if self.config.slots > 1:
            cmd.extend(["--slots", str(self.config.slots)])

        # 连续批处理
        if self.config.cont_batching:
            cmd.append("--cont-batching")

        # 内存锁定
        if self.config.mlock:
            cmd.append("--mlock")

        # 禁用 KV offload
        if self.config.no_kv_offload:
            cmd.append("--no-kv-offload")

        # RoPE 缩放
        if self.config.rope_scaling != "none":
            cmd.extend(["--rope-scaling", self.config.rope_scaling])

        if self.config.rope_scale != 1.0:
            cmd.extend(["--rope-scale", str(self.config.rope_scale)])

        # 推测解码
        if self.config.speculative_draft:
            cmd.extend(["--draft", self.config.speculative_draft])
            cmd.extend(["--draft-max", str(self.config.speculative_max)])
            cmd.extend(["--draft-p-min", str(self.config.speculative_pmin)])

        # Runner verbose
        if self.config.verbose_runner:
            cmd.append("--verbose")

        return cmd

    def _prepare_env(self) -> Dict[str, str]:
        """准备环境变量"""
        env = os.environ.copy()

        # 库路径
        lib_paths = [str(self.runner.bin_dir)]

        # 添加后端特定的库路径
        backend = self.config.backend
        if backend == "rocm":
            # ROCm 库路径
            rocm_paths = [
                "/opt/rocm/lib",
                "/opt/rocm/core-7.12/lib",
                "/opt/rocm-7.1.1/lib",
            ]
            for path in rocm_paths:
                if os.path.exists(path):
                    lib_paths.append(path)

            # 设置 HIP 设备
            if self.config.backend_index >= 0:
                env["HIP_VISIBLE_DEVICES"] = str(self.config.backend_index)
                if self.verbose:
                    console.print(f"[blue]使用 ROCm GPU index {self.config.backend_index}[/blue]")
            elif self.config.device.startswith("gpu"):
                gpu_id = int(self.config.device.replace("gpu", ""))
                env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
                if self.verbose:
                    console.print(f"[blue]使用 ROCm GPU {gpu_id}[/blue]")

        elif backend == "cuda":
            cuda_paths = [
                "/usr/local/cuda/lib64",
                "/usr/local/cuda-12.8/lib64",
                "/usr/local/cuda-12.6/lib64",
                "/usr/local/cuda-12.5/lib64",
                "/usr/local/cuda-12.4/lib64",
                "/usr/local/cuda-12.3/lib64",
                "/usr/local/cuda-12.2/lib64",
                "/usr/local/cuda-12.1/lib64",
                "/usr/local/cuda-12.0/lib64",
                "/usr/local/cuda-13.2/lib64",
            ]
            for path in cuda_paths:
                if os.path.exists(path):
                    lib_paths.append(path)

            if self.config.device.startswith("gpu"):
                gpu_id = int(self.config.device.replace("gpu", ""))
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                if self.verbose:
                    console.print(f"[blue]使用 CUDA GPU {gpu_id}[/blue]")

        elif backend == "vulkan":
            if self.verbose:
                console.print("[blue]使用 Vulkan GPU 后端[/blue]")

        # 合并库路径
        if "LD_LIBRARY_PATH" in env:
            lib_paths.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths)

        return env

    def _wait_for_ready(self, timeout: float = 120.0) -> bool:
        """等待服务器就绪"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.process is not None and self.process.poll() is not None:
                console.print(f"[red]服务器启动失败 (进程退出码: {self.process.returncode})[/red]")
                return False

            if self._check_port():
                console.print("[green]服务已就绪[/green]")
                return True

            time.sleep(0.5)

        console.print(f"[red]等待服务器超时 ({timeout}秒)[/red]")
        if self.process is not None and self.process.poll() is None:
            console.print("[yellow]进程仍在运行，但端口未响应[/yellow]")
        return False

    def _check_port(self) -> bool:
        """检查端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((self.config.host, self.config.port))
            sock.close()
            return result == 0
        except:  # noqa: E722
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
    return any(arch in model_lower for arch in OLLAMA_ONLY_ARCHITECTURES)


def serve_ollama_model(
    model_name: str,
    backend: str = "auto",
    device: str = "auto",
    port: int = 8080,
    host: str = "127.0.0.1",
    ctx_size: int = 32768,
    verbose: bool = False,
    runner_type: str = "ollama",
    verbose_runner: bool = False,
    fit_mode: str = "auto",
    lookahead: int = 0,
    cache_prompts: bool = False,
    slots: int = 1,
    cont_batching: bool = True,
    mlock: bool = False,
    no_kv_offload: bool = False,
    rope_scaling: str = "none",
    rope_scale: float = 1.0,
    speculative_draft: Optional[str] = None,
    speculative_max: int = 5,
    speculative_pmin: float = 0.75,
    cpu_moe: bool = False,
    threads: int = 0,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    flash_attn: bool = True,
    kv_cache: str = "f16",
    n_gpu_layers: int = -1,
    **kwargs,
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
        runner_type: runner 类型 ("ollama" 或 "official")
        verbose_runner: runner 详细日志
        fit_mode: 参数拟合模式 ("auto", "on", "off")
        lookahead: 前瞻解码步数 (0=禁用, 2-4推荐)
        cache_prompts: 启用提示词缓存
        slots: 并发请求槽位数
        cont_batching: 启用连续批处理
        mlock: 锁定模型在内存中
        no_kv_offload: 禁用KV缓存卸载到CPU
        rope_scaling: RoPE缩放类型
        rope_scale: RoPE上下文缩放因子
        speculative_draft: 推测解码draft模型路径
        speculative_max: 最大draft token数
        speculative_pmin: 最小接受概率

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

    # 获取设备的 backend_index
    backend_index = -1
    if device.startswith("gpu"):
        from moxing.device import DeviceDetector

        detector = DeviceDetector()
        devices = detector.detect()
        try:
            gpu_id = int(device.replace("gpu", ""))
            for dev in devices:
                if dev.index == gpu_id:
                    backend_index = dev.backend_index if dev.backend_index >= 0 else dev.index
                    break
        except (ValueError, AttributeError):
            pass

    # 创建配置
    config = RunnerConfig(
        backend=backend,
        device=device,
        port=port,
        host=host,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        threads=threads,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        flash_attn=flash_attn,
        kv_cache=kv_cache,
        backend_index=backend_index,
        runner_type=runner_type,
        verbose_runner=verbose_runner or verbose,
        fit_mode=fit_mode,
        lookahead=lookahead,
        cache_prompts=cache_prompts,
        slots=slots,
        cont_batching=cont_batching,
        mlock=mlock,
        no_kv_offload=no_kv_offload,
        rope_scaling=rope_scaling,
        rope_scale=rope_scale,
        speculative_draft=speculative_draft,
        speculative_max=speculative_max,
        speculative_pmin=speculative_pmin,
        cpu_moe=cpu_moe,
        **kwargs,
    )

    # 创建并启动服务器
    server = OllamaRunnerServer(model_path, config, verbose)

    if server.start(wait_ready=True):
        console.print(
            Panel(
                f"[green]模型:[/green] {model_name}\n"
                f"[green]后端:[/green] {backend}\n"
                f"[green]设备:[/green] {device}\n"
                f"[green]API:[/green] {server.base_url}/v1\n"
                f"[yellow]按 Ctrl+C 停止[/yellow]",
                title="Ollama Runner 已启动",
            )
        )
        return server
    else:
        console.print("[red]启动失败[/red]")
        return None


def run_ollama_model(
    model_name: str, prompt: str, backend: str = "auto", device: str = "auto", **kwargs
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
        model_name=model_name, backend=backend, device=device, port=port, **kwargs
    )

    if not server:
        return None

    try:
        # 使用 OpenAI API 格式调用
        from moxing.client import Client

        client = Client(server.base_url)

        response = client.chat.completions.create(
            model="model", messages=[{"role": "user", "content": prompt}], stream=False
        )

        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].get("message", {}).get("content", "")
        return ""
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
