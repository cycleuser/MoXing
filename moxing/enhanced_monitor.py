"""
MoXing 增强监控系统

提供完整的资源监控和历史记录：
- 内存/显存历史占用
- CPU/GPU 资源消耗
- Token 使用统计
- 上下文使用情况
"""

import contextlib
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()


@dataclass
class MetricSnapshot:
    """单个时间点的指标快照"""

    timestamp: datetime

    # Tokens
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0

    # 速度
    prompt_speed: float = 0.0
    generate_speed: float = 0.0

    # 内存 (MB)
    gpu_memory_mb: float = 0.0
    ram_used_mb: float = 0.0
    process_memory_mb: float = 0.0

    # GPU
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_w: float = 0.0

    # CPU
    cpu_percent: float = 0.0

    # 请求
    requests_processing: int = 0
    requests_deferred: int = 0

    # 上下文
    context_used: int = 0
    context_total: int = 0


@dataclass
class ServerInfo:
    """服务器静态信息"""

    model_name: str = ""
    model_path: str = ""
    model_size_gb: float = 0.0
    context_length: int = 0
    batch_size: int = 0
    n_gpu_layers: str = ""
    kv_cache_type: str = ""
    backend: str = ""
    device: str = ""


class MetricsHistory:
    """指标历史记录"""

    def __init__(self, max_points: int = 3600):
        self.max_points = max_points
        self.history: deque = deque(maxlen=max_points)
        self._lock = threading.Lock()

    def add(self, snapshot: MetricSnapshot):
        with self._lock:
            self.history.append(snapshot)

    def get_recent(self, seconds: int = 60) -> List[MetricSnapshot]:
        """获取最近 N 秒的数据"""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        with self._lock:
            return [s for s in self.history if s.timestamp >= cutoff]

    def get_all(self) -> List[MetricSnapshot]:
        with self._lock:
            return list(self.history)

    def get_stats(self, seconds: int = 60) -> Dict[str, Any]:
        """计算最近 N 秒的统计数据"""
        recent = self.get_recent(seconds)
        if not recent:
            return {}

        return {
            "avg_prompt_speed": sum(s.prompt_speed for s in recent) / len(recent),
            "avg_generate_speed": sum(s.generate_speed for s in recent) / len(recent),
            "max_gpu_memory": max(s.gpu_memory_mb for s in recent),
            "avg_gpu_memory": sum(s.gpu_memory_mb for s in recent) / len(recent),
            "max_ram": max(s.ram_used_mb for s in recent),
            "avg_cpu": sum(s.cpu_percent for s in recent) / len(recent),
            "avg_gpu_util": sum(s.gpu_utilization for s in recent) / len(recent),
            "total_tokens": recent[-1].total_tokens if recent else 0,
            "data_points": len(recent),
        }


class EnhancedMonitor:
    """增强的监控器"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, history_seconds: int = 3600):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

        self.history = MetricsHistory(max_points=history_seconds)
        self.server_info = ServerInfo()
        self.process: Optional[psutil.Process] = None

        self._running = False
        self._collector_thread: Optional[threading.Thread] = None

        self._last_prompt_tokens = 0
        self._last_generated_tokens = 0

    def start_collection(self, interval: float = 1.0):
        """启动后台收集线程"""
        if self._running:
            return

        self._running = True
        self._collector_thread = threading.Thread(
            target=self._collect_loop, args=(interval,), daemon=True
        )
        self._collector_thread.start()

    def stop_collection(self):
        """停止收集"""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join(timeout=2)

    def _collect_loop(self, interval: float):
        """后台收集循环"""
        while self._running:
            try:
                snapshot = self._collect_snapshot()
                self.history.add(snapshot)
            except Exception as e:
                logger.debug("Metrics collection failed: %s", e, exc_info=True)
                console.print(f"[red]Collect error: {e}[/red]")
            time.sleep(interval)

    def _collect_snapshot(self) -> MetricSnapshot:
        """收集单个快照"""
        snapshot = MetricSnapshot(timestamp=datetime.now())

        # 从 /metrics 获取指标
        try:
            r = httpx.get(f"{self.base_url}/metrics", timeout=5)
            if r.status_code == 200:
                self._parse_metrics(r.text, snapshot)
        except:  # noqa: E722
            pass

        # 从 /slots 获取上下文使用
        try:
            r = httpx.get(f"{self.base_url}/slots", timeout=5)
            if r.status_code == 200:
                slots = r.json()
                for slot in slots:
                    if slot.get("is_processing"):
                        snapshot.context_used = slot.get("n_ctx", 0)
        except:  # noqa: E722
            pass

        # 系统指标
        snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)
        snapshot.ram_used_mb = psutil.virtual_memory().used / (1024 * 1024)

        # 进程内存
        if self.process:
            try:
                snapshot.process_memory_mb = self.process.memory_info().rss / (1024 * 1024)
                snapshot.gpu_memory_mb = snapshot.process_memory_mb
            except:  # noqa: E722
                pass

        # GPU 信息 (macOS)
        import platform

        if platform.system() == "Darwin":
            gpu_info = self._get_macos_gpu_info()
            if gpu_info:
                snapshot.gpu_temperature = gpu_info.get("temperature", 0)

        return snapshot

    def _parse_metrics(self, text: str, snapshot: MetricSnapshot):
        """解析 Prometheus 指标"""
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue

            try:
                parts = line.split()
                if len(parts) < 2:
                    continue

                name = parts[0].replace("llamacpp:", "")
                value = float(parts[1])

                if name == "prompt_tokens_total":
                    snapshot.prompt_tokens = int(value)
                elif name == "tokens_predicted_total":
                    snapshot.generated_tokens = int(value)
                elif name == "prompt_tokens_seconds":
                    snapshot.prompt_speed = value
                elif name == "predicted_tokens_seconds":
                    snapshot.generate_speed = value
                elif name == "requests_processing":
                    snapshot.requests_processing = int(value)
                elif name == "requests_deferred":
                    snapshot.requests_deferred = int(value)
            except:  # noqa: E722
                pass

        snapshot.total_tokens = snapshot.prompt_tokens + snapshot.generated_tokens

    def _get_macos_gpu_info(self) -> Dict[str, Any]:
        """获取 macOS GPU 信息（不使用 sudo）"""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                info = {}
                for line in result.stdout.split("\n"):
                    if "Chipset Model:" in line:
                        info["gpu_model"] = line.split(":")[1].strip()
                return info
        except:  # noqa: E722
            pass
        return {}

    def fetch_server_info(self) -> bool:
        """获取服务器信息"""
        try:
            r = httpx.get(f"{self.base_url}/props", timeout=5)
            if r.status_code == 200:
                props = r.json()
                self.server_info.model_path = props.get("model_path", "")
                self.server_info.model_name = Path(self.server_info.model_path).name
                self.server_info.context_length = props.get("n_ctx", 0)
                self.server_info.batch_size = props.get("n_batch", 0)
                return True
        except:  # noqa: E722
            pass
        return False

    def set_process(self, pid: int):
        """设置要监控的进程"""
        with contextlib.suppress(BaseException):
            self.process = psutil.Process(pid)


ENHANCED_MONITOR_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoXing Enhanced Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 15px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 15px;
            font-size: 1.8em;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            font-size: 1em;
            margin-bottom: 12px;
            color: #00d2ff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 8px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            font-size: 0.9em;
        }
        .metric-label { color: #888; }
        .metric-value {
            font-weight: bold;
            color: #fff;
        }
        .metric-value.green { color: #00ff88; }
        .metric-value.blue { color: #00d2ff; }
        .metric-value.yellow { color: #ffd700; }
        .metric-value.red { color: #ff4444; }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 4px;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .progress-fill.green { background: linear-gradient(90deg, #00d2ff, #00ff88); }
        .progress-fill.yellow { background: linear-gradient(90deg, #ffd700, #ff8800); }
        .progress-fill.red { background: linear-gradient(90deg, #ff4444, #ff0000); }
        .chart-container {
            position: relative;
            height: 200px;
            margin-top: 10px;
        }
        .tokens-display {
            display: flex;
            justify-content: space-around;
            text-align: center;
        }
        .token-stat {
            padding: 10px;
        }
        .token-stat .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00d2ff;
        }
        .token-stat .label {
            color: #888;
            font-size: 0.8em;
        }
        .model-info {
            text-align: center;
            padding: 8px;
            background: rgba(0,210,255,0.1);
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .model-name {
            font-size: 1.2em;
            color: #00d2ff;
        }
        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        @media (max-width: 900px) {
            .chart-row { grid-template-columns: 1fr; }
        }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 10px;
            font-size: 0.85em;
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-indicator.running { background: #00ff88; }
        .status-indicator.idle { background: #666; }
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.85em;
        }
        .history-table th, .history-table td {
            padding: 6px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .history-table th { color: #00d2ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 MoXing Enhanced Monitor</h1>

        <div class="model-info">
            <div class="model-name" id="modelName">Loading...</div>
            <div id="modelDetails" style="color: #888; margin-top: 3px;"></div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>💾 Memory</h2>
                <div class="metric">
                    <span class="metric-label">GPU/Process Memory</span>
                    <span class="metric-value" id="gpuMemory">-- MB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill green" id="gpuMemoryBar" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">System RAM</span>
                    <span class="metric-value" id="ramMemory">-- MB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill green" id="ramMemoryBar" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">RAM Used</span>
                    <span class="metric-value blue" id="ramPercent">--%</span>
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
                    <span class="metric-label">Avg Speed (60s)</span>
                    <span class="metric-value blue" id="avgSpeed">-- tok/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value" id="cpuUsage">--%</span>
                </div>
            </div>

            <div class="card">
                <h2>📊 Tokens</h2>
                <div class="tokens-display">
                    <div class="token-stat">
                        <div class="value" id="promptTokens">0</div>
                        <div class="label">Prompt</div>
                    </div>
                    <div class="token-stat">
                        <div class="value" id="generatedTokens">0</div>
                        <div class="label">Generated</div>
                    </div>
                    <div class="token-stat">
                        <div class="value" id="totalTokens">0</div>
                        <div class="label">Total</div>
                    </div>
                </div>
                <div class="metric" style="margin-top: 10px;">
                    <span class="metric-label">Context Used</span>
                    <span class="metric-value" id="contextUsed">-- / --</span>
                </div>
            </div>

            <div class="card">
                <h2>🔄 Slots</h2>
                <div id="slotsGrid"></div>
            </div>
        </div>

        <div class="chart-row">
            <div class="card">
                <h2>📈 Memory History (Last 60s)</h2>
                <div class="chart-container">
                    <canvas id="memoryChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>⚡ Speed History (Last 60s)</h2>
                <div class="chart-container">
                    <canvas id="speedChart"></canvas>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📋 Session Summary</h2>
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Current</th>
                        <th>Average (60s)</th>
                        <th>Max (60s)</th>
                    </tr>
                </thead>
                <tbody id="summaryTable">
                    <tr><td colspan="4">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <div class="refresh-info">
            Last update: <span id="lastUpdate">--</span> |
            Refresh: 1s |
            <a href="/v1/models" target="_blank" style="color: #00d2ff">API</a> |
            <a href="/metrics" target="_blank" style="color: #00d2ff">Metrics</a>
        </div>
    </div>

    <script>
        // 历史数据存储
        const historyData = {
            memory: [],
            speed: [],
            labels: [],
            maxPoints: 60
        };

        // Charts
        let memoryChart, speedChart;

        // 初始化图表
        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' } }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#888', font: { size: 10 } }
                    }
                },
                animation: { duration: 0 }
            };

            const ctx1 = document.getElementById('memoryChart').getContext('2d');
            memoryChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'GPU Memory (MB)', data: [],
                            borderColor: '#00d2ff',
                            backgroundColor: 'rgba(0,210,255,0.1)',
                            fill: true, tension: 0.3 },
                        { label: 'RAM (GB)', data: [],
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0,255,136,0.1)',
                            fill: false, tension: 0.3 }
                    ]
                },
                options: chartOptions
            });

            const ctx2 = document.getElementById('speedChart').getContext('2d');
            speedChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Prompt (tok/s)', data: [],
                            borderColor: '#ffd700', fill: false, tension: 0.3 },
                        { label: 'Generate (tok/s)', data: [],
                            borderColor: '#00ff88', fill: false, tension: 0.3 }
                    ]
                },
                options: chartOptions
            });
        }

        // 获取指标
        async function fetchMetrics() {
            try {
                const response = await fetch('/metrics');
                const text = await response.text();
                return parseMetrics(text);
            } catch (e) {
                console.error('Failed to fetch metrics:', e);
                return null;
            }
        }

        function parseMetrics(text) {
            const metrics = {};
            lines = text.split('\\n');

            lines.forEach(line => {
                if (line.startsWith('#') || !line.trim()) return;
                const parts = line.split(' ');
                if (parts.length >= 2) {
                    const name = parts[0].replace('llamacpp:', '');
                    metrics[name] = parseFloat(parts[1]);
                }
            });

            return metrics;
        }

        // 获取 slots
        async function fetchSlots() {
            try {
                const response = await fetch('/slots');
                return await response.json();
            } catch (e) {
                return [];
            }
        }

        // 获取属性
        async function fetchProps() {
            try {
                const response = await fetch('/props');
                return await response.json();
            } catch (e) {
                return {};
            }
        }

        // 更新显示
        async function updateDisplay() {
            const metrics = await fetchMetrics();
            const slots = await fetchSlots();

            if (!metrics) return;

            const now = new Date();

            // Tokens
            const promptTokens = Math.round(metrics['prompt_tokens_total'] || 0);
            const generatedTokens = Math.round(metrics['tokens_predicted_total'] || 0);
            const totalTokens = promptTokens + generatedTokens;

            document.getElementById('promptTokens').textContent = promptTokens.toLocaleString();
            document.getElementById('generatedTokens').textContent = (
                generatedTokens.toLocaleString());
            document.getElementById('totalTokens').textContent = totalTokens.toLocaleString();

            // Speed
            const promptSpeed = metrics['prompt_tokens_seconds'] || 0;
            const genSpeed = metrics['predicted_tokens_seconds'] || 0;

            document.getElementById('promptSpeed').textContent = promptSpeed.toFixed(1) + ' tok/s';
            document.getElementById('genSpeed').textContent = genSpeed.toFixed(1) + ' tok/s';

            // 更新历史数据
            historyData.memory.push(metrics['process_memory_mb'] || 0);
            historyData.speed.push({ prompt: promptSpeed, generate: genSpeed });
            historyData.labels.push(now.toLocaleTimeString());

            if (historyData.memory.length > historyData.maxPoints) {
                historyData.memory.shift();
                historyData.speed.shift();
                historyData.labels.shift();
            }

            // 计算平均值
            const avgPrompt = historyData.speed.reduce(
                (a, b) => a + b.prompt, 0) / historyData.speed.length;
            const avgGen = historyData.speed.reduce(
                (a, b) => a + b.generate, 0) / historyData.speed.length;
            document.getElementById('avgSpeed').textContent =
                ((avgPrompt + avgGen) / 2).toFixed(1) + ' tok/s';

            // 更新图表
            memoryChart.data.labels = historyData.labels;
            memoryChart.data.datasets[0].data = historyData.memory;
            memoryChart.data.datasets[1].data = historyData.memory.map(m => m / 1024);
            memoryChart.update('none');

            speedChart.data.labels = historyData.labels;
            speedChart.data.datasets[0].data = historyData.speed.map(s => s.prompt);
            speedChart.data.datasets[1].data = historyData.speed.map(s => s.generate);
            speedChart.update('none');

            // 更新摘要表
            const maxMemory = Math.max(...historyData.memory);
            const avgMemory = historyData.memory.reduce(
                (a, b) => a + b, 0) / historyData.memory.length;

            document.getElementById('summaryTable').innerHTML = `
                <tr>
                    <td>GPU Memory</td>
                    <td>${(historyData.memory[historyData.memory.length-1] || 0).toFixed(0)} MB</td>
                    <td>${avgMemory.toFixed(0)} MB</td>
                    <td>${maxMemory.toFixed(0)} MB</td>
                </tr>
                <tr>
                    <td>Prompt Speed</td>
                    <td>${promptSpeed.toFixed(1)} tok/s</td>
                    <td>${avgPrompt.toFixed(1)} tok/s</td>
                    <td>${Math.max(...historyData.speed.map(s => s.prompt)).toFixed(1)} tok/s</td>
                </tr>
                <tr>
                    <td>Generate Speed</td>
                    <td>${genSpeed.toFixed(1)} tok/s</td>
                    <td>${avgGen.toFixed(1)} tok/s</td>
                    <td>${Math.max(...historyData.speed.map(s => s.generate)).toFixed(1)} tok/s</td>
                </tr>
                <tr>
                    <td>Total Tokens</td>
                    <td colspan="3">${totalTokens.toLocaleString()}</td>
                </tr>
            `;

            // Slots
            const slotsHtml = slots.map(slot => {
                const isActive = slot.is_processing;
                const statusClass = isActive ? 'active' : 'idle';
                const statusText = isActive ? 'Processing' : 'Idle';
                const tokens = slot.next_token?.[0]?.n_decoded || 0;

                return `
                    <div class="metric">
                        <span><span class="status-indicator ${statusClass}">
                            </span>Slot ${slot.id}</span>
                        <span class="metric-value">${statusText}</span>
                    </div>
                `;
            }).join('');
            const noSlots = '<div class="metric">No slots</div>';
            document.getElementById('slotsGrid').innerHTML =
                slotsHtml || noSlots;

            document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
        }

        // 初始化
        async function init() {
            initCharts();

            const props = await fetchProps();
            if (props.model_path) {
                const modelName = props.model_path.split('/').pop();
                document.getElementById('modelName').textContent = modelName;
                document.getElementById('modelDetails').textContent =
                    `Context: ${props.n_ctx || '--'} | Batch: ${props.n_batch || '--'}`;
            }

            updateDisplay();
            setInterval(updateDisplay, 1000);
        }

        init();
    </script>
</body>
</html>
"""


def print_enhanced_live_metrics(host: str = "127.0.0.1", port: int = 8080):
    """终端实时监控显示"""
    monitor = EnhancedMonitor(host, port)
    monitor.fetch_server_info()
    monitor.start_collection()

    def generate_display():
        snapshot = monitor._collect_snapshot()
        monitor.history.add(snapshot)
        stats = monitor.history.get_stats(60)

        Layout()

        # 模型信息
        model_info = f"[cyan]{monitor.server_info.model_name[:40]}[/cyan]"
        if monitor.server_info.context_length:
            model_info += f" | Context: {monitor.server_info.context_length:,}"

        # 资源使用表
        resource_table = Table(title="💾 Resources", show_header=True, title_style="cyan")
        resource_table.add_column("Resource", style="yellow")
        resource_table.add_column("Current", style="green")
        resource_table.add_column("Avg (60s)", style="blue")
        resource_table.add_column("Max (60s)", style="magenta")

        resource_table.add_row(
            "GPU Memory",
            f"{snapshot.gpu_memory_mb:.0f} MB",
            f"{stats.get('avg_gpu_memory', 0):.0f} MB",
            f"{stats.get('max_gpu_memory', 0):.0f} MB",
        )
        resource_table.add_row(
            "RAM Used",
            f"{snapshot.ram_used_mb / 1024:.2f} GB",
            f"{stats.get('max_ram', 0) / 1024:.2f} GB",
            "-",
        )
        resource_table.add_row(
            "CPU Usage", f"{snapshot.cpu_percent:.1f}%", f"{stats.get('avg_cpu', 0):.1f}%", "-"
        )

        # Tokens 表
        tokens_table = Table(title="📊 Tokens", show_header=True, title_style="cyan")
        tokens_table.add_column("Type", style="yellow")
        tokens_table.add_column("Count", style="green")
        tokens_table.add_column("Speed", style="blue")

        tokens_table.add_row(
            "Prompt", f"{snapshot.prompt_tokens:,}", f"{snapshot.prompt_speed:.1f} tok/s"
        )
        tokens_table.add_row(
            "Generated", f"{snapshot.generated_tokens:,}", f"{snapshot.generate_speed:.1f} tok/s"
        )
        tokens_table.add_row(
            "Total", f"{snapshot.total_tokens:,}", f"{stats.get('avg_generate_speed', 0):.1f} tok/s"
        )

        # 性能表
        perf_table = Table(title="⚡ Performance", show_header=False)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")

        avg_speed = (stats.get("avg_prompt_speed", 0) + stats.get("avg_generate_speed", 0)) / 2
        perf_table.add_row("Avg Speed (60s)", f"{avg_speed:.1f} tok/s")
        perf_table.add_row("Requests Processing", str(snapshot.requests_processing))
        perf_table.add_row(
            "Context",
            f"{monitor.server_info.context_length:,}"
            if monitor.server_info.context_length
            else "N/A",
        )

        return Panel(
            f"{model_info}\n\n{resource_table}\n\n{tokens_table}\n\n{perf_table}",
            title="🚀 MoXing Monitor",
            border_style="blue",
        )

    console.print("[blue]Starting enhanced monitor...[/blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        with Live(generate_display(), refresh_per_second=1) as live:
            while True:
                time.sleep(1)
                live.update(generate_display())
    except KeyboardInterrupt:
        monitor.stop_collection()
        console.print("\n[yellow]Monitor stopped[/yellow]")


def create_enhanced_monitor_page() -> str:
    """创建增强监控页面 HTML"""
    return ENHANCED_MONITOR_HTML


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoXing Enhanced Monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    print_enhanced_live_metrics(args.host, args.port)
