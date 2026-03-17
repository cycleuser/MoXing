"""
OpenAI-compatible client for llama.cpp server
"""

import json
import time
from typing import Optional, List, Dict, Any, Generator, Union, AsyncGenerator
from dataclasses import dataclass, field

import httpx


@dataclass
class Message:
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletion:
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[Dict] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class Client:
    """
    OpenAI-compatible client for llama.cpp server.
    
    Usage:
        client = Client("http://localhost:8080")
        response = client.chat.completions.create(
            model="llama",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat = Chat(self)
        self.models = Models(self)
        
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def get(self, path: str, **kwargs) -> httpx.Response:
        return httpx.get(f"{self.base_url}{path}", headers=self._headers(), **kwargs)
    
    def post(self, path: str, **kwargs) -> httpx.Response:
        return httpx.post(f"{self.base_url}{path}", headers=self._headers(), **kwargs)
    
    def health(self) -> bool:
        """Check server health."""
        try:
            resp = self.get("/health")
            return resp.status_code == 200
        except:
            return False
    
    def props(self) -> Dict:
        """Get server properties."""
        resp = self.get("/props")
        return resp.json()


class Chat:
    def __init__(self, client: Client):
        self.client = client
        self.completions = ChatCompletions(client)


class ChatCompletions:
    def __init__(self, client: Client):
        self.client = client
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Union[ChatCompletion, Generator]:
        """
        Create a chat completion.
        
        Args:
            model: Model name (ignored, uses loaded model)
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: List of tool definitions for function calling
            tool_choice: Tool choice mode ("auto", "none", or specific tool)
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
            
        payload.update(kwargs)
        
        if stream:
            return self._stream_create(payload)
        
        resp = self.client.post("/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        return ChatCompletion(
            id=data.get("id", ""),
            model=data.get("model", model),
            choices=data.get("choices", []),
            usage=data.get("usage", {}),
        )
    
    def _stream_create(self, payload: Dict) -> Generator:
        """Stream chat completion."""
        with httpx.stream(
            "POST",
            f"{self.client.base_url}/v1/chat/completions",
            json=payload,
            headers=self.client._headers(),
            timeout=120
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue


class Models:
    def __init__(self, client: Client):
        self.client = client
    
    def list(self) -> Dict:
        """List available models."""
        resp = self.client.get("/v1/models")
        return resp.json()
    
    def retrieve(self, model_id: str) -> Dict:
        """Get model info."""
        resp = self.client.get(f"/v1/models/{model_id}")
        return resp.json()