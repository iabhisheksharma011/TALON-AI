"""
Thin wrapper around Ollama's local HTTP API.
Supports hardware options: num_gpu (GPU layers) and num_thread (CPU threads).
"""

import json
import requests
from typing import List, Dict, Optional, Generator

OLLAMA_BASE = "http://localhost:11434"


def get_models() -> List[Dict]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json().get("models", [])
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")
    except Exception as e:
        raise RuntimeError(f"Failed to contact Ollama: {e}")


def chat_stream(
    model: str,
    messages: List[Dict],
    num_gpu: int = 0,
    num_thread: int = 0,
) -> Generator[str, None, None]:
    """
    Stream a chat response from Ollama.
    num_gpu    : number of model layers to offload to GPU (0 = CPU only)
    num_thread : number of CPU threads (0 = Ollama default)
    """
    options = {}
    if num_gpu >= 0:
        options["num_gpu"] = num_gpu          # 0 = pure CPU, >0 = GPU layers
    if num_thread > 0:
        options["num_thread"] = num_thread

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }

    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")


def is_running() -> bool:
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False
