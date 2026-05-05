"""
MoXing 监控仪表板

提供实时监控页面，显示：
- GPU/CPU 内存使用
- 显卡算力
- Tokens 消耗
- 请求统计
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from rich.console import Console

logger = logging.getLogger(__name__)

console = Console()


@dataclass
class ServerMetrics:
    """服务器指标"""

    prompt_tokens_total: int = 0
    tokens_predicted_total: int = 0
    prompt_seconds_total: float = 0.0
    predicted_seconds_total: float = 0.0
    prompt_tokens_per_second: float = 0.0
    predicted_tokens_per_second: float = 0.0
    requests_processing: int = 0
    requests_deferred: int = 0
    n_decode_total: int = 0
    n_tokens_max: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryInfo:
    """内存信息"""

    gpu_total_mb: float = 0
    gpu_used_mb: float = 0
    gpu_free_mb: float = 0
    ram_total_mb: float = 0
    ram_used_mb: float = 0
    ram_free_mb: float = 0
    model_mb: float = 0
    kv_cache_mb: float = 0
    compute_mb: float = 0


@dataclass
class GPUInfo:
    """GPU 信息"""

    name: str = ""
    utilization_percent: float = 0
    memory_percent: float = 0
    temperature_c: float = 0
    power_w: float = 0


class MetricsCollector:
    """指标收集器"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

        self._last_metrics: Optional[ServerMetrics] = None
        self._metrics_history: List[ServerMetrics] = []
        self._max_history = 100

        self._last_prompt_tokens = 0
        self._last_predicted_tokens = 0

    def fetch_metrics(self) -> Optional[ServerMetrics]:
        """获取 Prometheus 格式的指标"""
        try:
            r = httpx.get(f"{self.base_url}/metrics", timeout=5)
            if r.status_code != 200:
                return None

            metrics = self._parse_prometheus_metrics(r.text)

            if self._last_metrics:
                metrics.prompt_tokens_per_second = (
                    (
                        (metrics.prompt_tokens_total - self._last_prompt_tokens)
                        / max(
                            0.001,
                            metrics.prompt_seconds_total - self._last_metrics.prompt_seconds_total,
                        )
                    )
                    if metrics.prompt_seconds_total > self._last_metrics.prompt_seconds_total
                    else 0
                )

                metrics.predicted_tokens_per_second = (
                    (
                        (metrics.tokens_predicted_total - self._last_predicted_tokens)
                        / max(
                            0.001,
                            metrics.predicted_seconds_total
                            - self._last_metrics.predicted_seconds_total,
                        )
                    )
                    if metrics.predicted_seconds_total > self._last_metrics.predicted_seconds_total
                    else 0
                )

            self._last_prompt_tokens = metrics.prompt_tokens_total
            self._last_predicted_tokens = metrics.tokens_predicted_total
            self._last_metrics = metrics

            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)

            return metrics

        except Exception as e:
            logger.debug("Metrics parsing failed: %s", e, exc_info=True)
            console.print(f"[red]Failed to fetch metrics: {e}[/red]")
            return None

    def _parse_prometheus_metrics(self, text: str) -> ServerMetrics:
        """解析 Prometheus 格式的指标"""
        metrics = ServerMetrics()

        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue

            try:
                if "llamacpp:prompt_tokens_total" in line:
                    metrics.prompt_tokens_total = int(float(line.split()[-1]))
                elif "llamacpp:tokens_predicted_total" in line:
                    metrics.tokens_predicted_total = int(float(line.split()[-1]))
                elif "llamacpp:prompt_seconds_total" in line:
                    metrics.prompt_seconds_total = float(line.split()[-1])
                elif "llamacpp:tokens_predicted_seconds_total" in line:
                    metrics.predicted_seconds_total = float(line.split()[-1])
                elif "llamacpp:prompt_tokens_seconds" in line:
                    metrics.prompt_tokens_per_second = float(line.split()[-1])
                elif "llamacpp:predicted_tokens_seconds" in line:
                    metrics.predicted_tokens_per_second = float(line.split()[-1])
                elif "llamacpp:requests_processing" in line:
                    metrics.requests_processing = int(float(line.split()[-1]))
                elif "llamacpp:requests_deferred" in line:
                    metrics.requests_deferred = int(float(line.split()[-1]))
                elif "llamacpp:n_decode_total" in line:
                    metrics.n_decode_total = int(float(line.split()[-1]))
                elif "llamacpp:n_tokens_max" in line:
                    metrics.n_tokens_max = int(float(line.split()[-1]))
            except:  # noqa: E722
                pass

        return metrics

    def fetch_slots(self) -> Optional[List[Dict]]:
        """获取槽位信息"""
        try:
            r = httpx.get(f"{self.base_url}/slots", timeout=5)
            if r.status_code == 200:
                return r.json()
        except:  # noqa: E722
            pass
        return None

    def fetch_props(self) -> Optional[Dict]:
        """获取模型属性"""
        try:
            r = httpx.get(f"{self.base_url}/props", timeout=5)
            if r.status_code == 200:
                return r.json()
        except:  # noqa: E722
            pass
        return None

    def get_memory_info(self, pid: Optional[int] = None) -> MemoryInfo:
        """获取内存信息"""
        import psutil

        info = MemoryInfo()

        info.ram_total_mb = psutil.virtual_memory().total / (1024 * 1024)
        info.ram_used_mb = psutil.virtual_memory().used / (1024 * 1024)
        info.ram_free_mb = psutil.virtual_memory().available / (1024 * 1024)

        if pid:
            try:
                process = psutil.Process(pid)
                info.gpu_used_mb = process.memory_info().rss / (1024 * 1024)
            except:  # noqa: E722
                pass

        return info

    def get_gpu_info(self) -> Optional[GPUInfo]:
        """获取 GPU 信息"""
        import platform

        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    info = GPUInfo()
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "Chipset Model:" in line:
                            info.name = line.split("Chipset Model:")[1].strip()
                        elif "VRAM (Total):" in line:
                            vram_str = line.split("VRAM (Total):")[1].strip()
                            if "GB" in vram_str:
                                info.memory_percent = float(vram_str.replace("GB", "").strip())
                    return info
            except:  # noqa: E722
                pass

        elif platform.system() == "Linux":
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 6:
                        info = GPUInfo()
                        info.name = parts[0]
                        info.utilization_percent = float(parts[1])
                        info.memory_percent = float(parts[2]) / float(parts[3]) * 100
                        info.temperature_c = float(parts[4])
                        info.power_w = float(parts[5])
                        return info
            except:  # noqa: E722
                pass

        return None


MONITOR_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoXing Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 2em;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .card h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #00d2ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        .metric-label { color: #888; }
        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
            color: #fff;
        }
        .metric-value.green { color: #00ff88; }
        .metric-value.blue { color: #00d2ff; }
        .metric-value.yellow { color: #ffd700; }
        .metric-value.red { color: #ff4444; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d2ff, #00ff88);
            transition: width 0.3s;
        }
        .tokens-display {
            display: flex;
            justify-content: space-around;
            text-align: center;
        }
        .token-stat {
            padding: 15px;
        }
        .token-stat .value {
            font-size: 2em;
            font-weight: bold;
            color: #00d2ff;
        }
        .token-stat .label {
            color: #888;
            font-size: 0.9em;
        }
        .slots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .slot {
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            text-align: center;
        }
        .slot.active { border: 2px solid #00ff88; }
        .slot.idle { border: 2px solid #444; }
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-dot.active { background: #00ff88; }
        .status-dot.idle { background: #666; }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 0.9em;
        }
        .model-info {
            text-align: center;
            padding: 10px;
            background: rgba(0,210,255,0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .model-name {
            font-size: 1.3em;
            color: #00d2ff;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .processing { animation: pulse 1s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 MoXing Monitor</h1>

        <div class="model-info">
            <div class="model-name" id="modelName">Loading...</div>
            <div id="modelDetails" style="color: #888; margin-top: 5px;"></div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>💾 Memory</h2>
                <div class="metric">
                    <span class="metric-label">GPU Memory</span>
                    <span class="metric-value" id="gpuMemory">-- MB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="gpuMemoryBar" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">RAM</span>
                    <span class="metric-value" id="ramMemory">-- MB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="ramMemoryBar" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">KV Cache</span>
                    <span class="metric-value blue" id="kvCache">-- MB</span>
                </div>
            </div>

            <div class="card">
                <h2>🎮 GPU</h2>
                <div class="metric">
                    <span class="metric-label">Device</span>
                    <span class="metric-value" id="gpuName">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Utilization</span>
                    <span class="metric-value" id="gpuUtil">--%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="gpuUtilBar" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Temperature</span>
                    <span class="metric-value" id="gpuTemp">--°C</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Power</span>
                    <span class="metric-value" id="gpuPower">-- W</span>
                </div>
            </div>

            <div class="card">
                <h2>⚡ Performance</h2>
                <div class="metric">
                    <span class="metric-label">Prompt Speed</span>
                    <span class="metric-value green" id="promptSpeed">-- tok/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Generate Speed</span>
                    <span class="metric-value green" id="genSpeed">-- tok/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Speed</span>
                    <span class="metric-value blue" id="avgSpeed">-- tok/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Decode Calls</span>
                    <span class="metric-value" id="decodeCalls">--</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📊 Tokens Statistics</h2>
            <div class="tokens-display">
                <div class="token-stat">
                    <div class="value" id="promptTokens">0</div>
                    <div class="label">Prompt Tokens</div>
                </div>
                <div class="token-stat">
                    <div class="value" id="predictedTokens">0</div>
                    <div class="label">Generated Tokens</div>
                </div>
                <div class="token-stat">
                    <div class="value" id="totalTokens">0</div>
                    <div class="label">Total Tokens</div>
                </div>
                <div class="token-stat">
                    <div class="value" id="maxTokens">0</div>
                    <div class="label">Max Batch</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🔄 Slots Status</h2>
            <div class="slots-grid" id="slotsGrid">
                <div class="slot idle">
                    <span class="status-dot idle"></span>
                    <span>Loading...</span>
                </div>
            </div>
        </div>

        <div class="refresh-info">
            Last update: <span id="lastUpdate">--</span> |
            Refresh interval: 1s |
            <a href="/v1/models" target="_blank" style="color: #00d2ff">API Docs</a>
        </div>
    </div>

    <script>
        let lastPromptTokens = 0;
        let lastPredictedTokens = 0;
        let lastTime = Date.now();

        async function fetchMetrics() {
            try {
                const response = await fetch('/metrics');
                const text = await response.text();
                parseMetrics(text);
            } catch (e) {
                console.error('Failed to fetch metrics:', e);
            }
        }

        function parseMetrics(text) {
            const lines = text.split('\\n');
            const metrics = {};

            lines.forEach(line => {
                if (line.startsWith('#') || !line.trim()) return;
                const parts = line.split(' ');
                if (parts.length >= 2) {
                    const name = parts[0].replace('llamacpp:', '');
                    metrics[name] = parseFloat(parts[1]);
                }
            });

            updateDisplay(metrics);
        }

        function updateDisplay(metrics) {
            const now = Date.now();
            const dt = (now - lastTime) / 1000;
            lastTime = now;

            // Tokens
            const promptTokens = Math.round(metrics['prompt_tokens_total'] || 0);
            const predictedTokens = Math.round(metrics['tokens_predicted_total'] || 0);
            const totalTokens = promptTokens + predictedTokens;

            document.getElementById('promptTokens').textContent = promptTokens.toLocaleString();
            document.getElementById('predictedTokens').textContent = (
                predictedTokens.toLocaleString());
            document.getElementById('totalTokens').textContent = totalTokens.toLocaleString();
            document.getElementById('maxTokens').textContent = (
                Math.round(metrics['n_tokens_max'] || 0));

            // Speed
            const promptSpeed = metrics['prompt_tokens_seconds'] || 0;
            const genSpeed = metrics['predicted_tokens_seconds'] || 0;
            const avgSpeed = promptSpeed > 0 && genSpeed > 0
                ? ((promptSpeed + genSpeed) / 2).toFixed(1)
                : Math.max(promptSpeed, genSpeed).toFixed(1);

            document.getElementById('promptSpeed').textContent = promptSpeed.toFixed(1) + ' tok/s';
            document.getElementById('genSpeed').textContent = genSpeed.toFixed(1) + ' tok/s';
            document.getElementById('avgSpeed').textContent = avgSpeed + ' tok/s';

            // Decode
            document.getElementById('decodeCalls').textContent = (
                Math.round(metrics['n_decode_total'] || 0));

            // Update time
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        }

        async function fetchSlots() {
            try {
                const response = await fetch('/slots');
                const slots = await response.json();
                updateSlots(slots);
            } catch (e) {
                console.error('Failed to fetch slots:', e);
            }
        }

        function updateSlots(slots) {
            const grid = document.getElementById('slotsGrid');
            grid.innerHTML = slots.map(slot => {
                const isActive = slot.is_processing;
                const statusClass = isActive ? 'active' : 'idle';
                const statusText = isActive ? 'Processing' : 'Idle';
                const tokens = slot.next_token?.[0]?.n_decoded || 0;

                return `
                    <div class="slot ${statusClass}">
                        <span class="status-dot ${statusClass}"></span>
                        <span>Slot ${slot.id}</span>
                        <div style="margin-top: 5px; color: #888;">
                            ${statusText}${tokens ? ' - ' + tokens + ' tokens' : ''}
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function fetchProps() {
            try {
                const response = await fetch('/props');
                const props = await response.json();
                updateProps(props);
            } catch (e) {
                console.error('Failed to fetch props:', e);
            }
        }

        function updateProps(props) {
            document.getElementById('modelName').textContent = (
                props.model_path?.split('/').pop() || 'Unknown');
            document.getElementById('modelDetails').textContent =
                `Context: ${props.n_ctx || '--'} | Batch: ${props.n_batch || '--'}`;
        }

        // Initial load
        fetchProps();

        // Refresh loop
        setInterval(() => {
            fetchMetrics();
            fetchSlots();
        }, 1000);
    </script>
</body>
</html>
"""


def create_monitor_page(output_path: Optional[Path] = None) -> Path:
    """创建监控页面"""
    if output_path is None:
        output_path = Path("/tmp/moxing_monitor.html")

    output_path.write_text(MONITOR_HTML_TEMPLATE)
    return output_path


def start_monitor_server(
    host: str = "127.0.0.1", port: int = 8080, llama_port: int = 8080, metrics_port: int = 9090
):
    """启动独立的监控服务器"""
    import http.server
    import socketserver
    from urllib.parse import urlparse

    class MonitorHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.llama_url = f"http://127.0.0.1:{llama_port}"
            super().__init__(*args, **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(MONITOR_HTML_TEMPLATE.encode())

            elif parsed.path == "/metrics":
                try:
                    resp = httpx.get(f"{self.llama_url}/metrics", timeout=5)
                    self.send_response(resp.status_code)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(resp.content)
                except Exception as e:
                    logger.debug("Metrics parsing failed: %s", e, exc_info=True)
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f"Error: {e}".encode())

            elif parsed.path == "/slots":
                try:
                    resp = httpx.get(f"{self.llama_url}/slots", timeout=5)
                    self.send_response(resp.status_code)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp.content)
                except Exception as e:
                    logger.debug("Metrics parsing failed: %s", e, exc_info=True)
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f"{{'error': '{e}'}}".encode())

            elif parsed.path == "/props":
                try:
                    resp = httpx.get(f"{self.llama_url}/props", timeout=5)
                    self.send_response(resp.status_code)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp.content)
                except Exception as e:
                    logger.debug("Model props fetch failed: %s", e, exc_info=True)
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f"{{'error': '{e}'}}".encode())

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    console.print(f"[green]Starting monitor server at http://{host}:{port}[/green]")
    console.print(f"[dim]Proxying to llama.cpp at http://127.0.0.1:{llama_port}[/dim]")

    with socketserver.TCPServer((host, port), MonitorHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor server stopped[/yellow]")


def print_live_metrics(host: str = "127.0.0.1", port: int = 8080):
    """打印实时指标到终端"""
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    collector = MetricsCollector(host, port)

    def generate_display():
        metrics = collector.fetch_metrics()
        slots = collector.fetch_slots()
        props = collector.fetch_props()

        Layout()

        # 模型信息
        if props:
            model_name = Path(props.get("model_path", "Unknown")).name
            model_info = f"[cyan]{model_name}[/cyan] | Context: {props.get('n_ctx', '--')}"
        else:
            model_info = "[red]Not connected[/red]"

        # Token 统计表
        table = Table(title="Token Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if metrics:
            table.add_row("Prompt Tokens", f"{metrics.prompt_tokens_total:,}")
            table.add_row("Generated Tokens", f"{metrics.tokens_predicted_total:,}")
            table.add_row(
                "Total Tokens", f"{metrics.prompt_tokens_total + metrics.tokens_predicted_total:,}"
            )
            table.add_row("Prompt Speed", f"{metrics.prompt_tokens_per_second:.1f} tok/s")
            table.add_row("Generate Speed", f"{metrics.predicted_tokens_per_second:.1f} tok/s")
            table.add_row("Requests Processing", str(metrics.requests_processing))
        else:
            table.add_row("Status", "[red]Failed to connect[/red]")

        # Slots 状态
        slots_table = Table(title="Slots", show_header=True)
        slots_table.add_column("ID", style="cyan")
        slots_table.add_column("Status", style="green")
        slots_table.add_column("Context", style="yellow")

        if slots:
            for slot in slots:
                status = (
                    "[green]Processing[/green]" if slot.get("is_processing") else "[dim]Idle[/dim]"
                )
                slots_table.add_row(str(slot.get("id", "?")), status, str(slot.get("n_ctx", "--")))

        return Panel(
            f"{model_info}\n\n{table}\n\n{slots_table}", title="MoXing Monitor", border_style="blue"
        )

    console.print("[blue]Starting live metrics display...[/blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        with Live(generate_display(), refresh_per_second=1) as live:
            while True:
                time.sleep(1)
                live.update(generate_display())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped[/yellow]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoXing Monitor")
    parser.add_argument("--host", default="127.0.0.1", help="Monitor host")
    parser.add_argument("--port", type=int, default=9090, help="Monitor port")
    parser.add_argument("--llama-port", type=int, default=8080, help="llama.cpp server port")
    parser.add_argument("--mode", choices=["server", "cli"], default="cli", help="Run mode")

    args = parser.parse_args()

    if args.mode == "server":
        start_monitor_server(args.host, args.port, args.llama_port)
    else:
        print_live_metrics(args.host, args.llama_port)
