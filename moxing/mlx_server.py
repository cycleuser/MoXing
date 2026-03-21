"""
MLX-based server for Apple Silicon.

Provides OpenAI-compatible API using Apple's MLX framework.
Supports models that llama.cpp may not yet support (e.g., Gemma3, Qwen3.5).
"""

import os
import sys
import json
import time
import signal
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
import multiprocessing

from rich.console import Console

console = Console()

MLX_SERVER_SCRIPT = '''
import os
import json
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any
import threading
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

class MLXModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        
    def load(self):
        if self.model is None:
            print(f"Loading model from {self.model_path}...", flush=True)
            self.model, self.tokenizer = load(self.model_path)
            print("Model loaded successfully!", flush=True)
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, **kwargs) -> str:
        with self._lock:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
            response = generate(
                self.model, 
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False
            )
            return response

model_instance: Optional[MLXModel] = None

class OpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        elif self.path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{"id": "mlx-model", "object": "model", "owned_by": "mlx"}]
            })
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        global model_instance
        
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return
        
        if self.path == "/v1/chat/completions":
            self.handle_chat_completion(data)
        elif self.path == "/v1/completions":
            self.handle_completion(data)
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def handle_chat_completion(self, data: dict):
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        stream = data.get("stream", False)
        
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|system|>\\n{content}<|end|>\\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\\n{content}<|end|>\\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\\n{content}<|end|>\\n")
        
        if prompt_parts:
            prompt_parts.append("<|assistant|>\\n")
        prompt = "".join(prompt_parts)
        
        if not prompt:
            prompt = messages[-1].get("content", "") if messages else ""
        
        try:
            response_text = model_instance.generate(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=top_p
            )
            
            # Clean up response
            for tag in ["<|end|>", "<|assistant|>", "<|user|>", "<|system|>"]:
                response_text = response_text.replace(tag, "")
            response_text = response_text.strip()
            
            result = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data.get("model", "mlx-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)
    
    def handle_completion(self, data: dict):
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        
        try:
            response_text = model_instance.generate(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=top_p
            )
            
            result = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": data.get("model", "mlx-model"),
                "choices": [{
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop"
                }]
            }
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

def run_server(host: str, port: int, model_path: str):
    global model_instance
    
    model_instance = MLXModel(model_path)
    model_instance.load()
    
    server = HTTPServer((host, port), OpenAIHandler)
    print(f"MLX server running at http://{host}:{port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    model_path = sys.argv[3] if len(sys.argv) > 3 else "."
    run_server(host, port, model_path)
'''


class MLXServer:
    """
    MLX-based LLM server with OpenAI-compatible API.
    
    Usage:
        server = MLXServer(model="Qwen/Qwen2.5-3B-Instruct")
        server.start()
        
        # Or use as context manager
        with MLXServer(model="model_name") as s:
            # Server is running
            ...
    """
    
    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        **kwargs
    ):
        self.model = model
        self.host = host
        self.port = port
        self._process: Optional[multiprocessing.Process] = None
        self._base_url = f"http://{host}:{port}"
        
    @staticmethod
    def is_available() -> bool:
        """Check if MLX is available on this system."""
        if sys.platform != "darwin":
            return False
        try:
            import mlx.core as mx
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_gguf(path: str) -> bool:
        """Check if a file is a GGUF model."""
        return path.endswith(".gguf")
    
    def start(self, wait: bool = True, timeout: int = 120) -> "MLXServer":
        """Start the MLX server."""
        if self._process is not None:
            raise RuntimeError("Server is already running")
        
        import httpx
        
        # Write server script to temp file
        script_path = Path.home() / ".cache" / "moxing" / "mlx_server.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(MLX_SERVER_SCRIPT)
        
        console.print(f"[blue]Starting MLX server...[/blue]")
        
        self._process = subprocess.Popen(
            [sys.executable, str(script_path), self.host, str(self.port), self.model],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if wait:
            self._wait_for_server(timeout)
        
        return self
    
    def _wait_for_server(self, timeout: int = 120):
        """Wait for server to be ready."""
        import httpx
        
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=2)
                if resp.status_code == 200:
                    console.print(f"[green]MLX Server ready at {self._base_url}[/green]")
                    return
            except:
                pass
            
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                raise RuntimeError(f"MLX Server failed to start:\n{stdout}\n{stderr}")
            
            time.sleep(1)
        
        raise TimeoutError(f"MLX Server did not start within {timeout} seconds")
    
    def stop(self):
        """Stop the server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            console.print("[yellow]MLX Server stopped[/yellow]")
    
    def is_running(self) -> bool:
        """Check if server is running."""
        if self._process is None:
            return False
        return self._process.poll() is None
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def base_url(self) -> str:
        return self._base_url