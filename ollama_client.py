"""
Thin wrapper around Ollama's local HTTP API.
"""

import json
import requests
from typing import List, Dict, Optional, Generator

OLLAMA_BASE = "http://localhost:11434"


def get_models() -> List[Dict]:
    """Return list of locally available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return data.get("models", [])
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Please start it with: ollama serve")
    except Exception as e:
        raise RuntimeError(f"Failed to contact Ollama: {e}")


def chat_stream(
    model: str,
    messages: List[Dict],
) -> Generator[str, None, None]:
    """
    Stream a chat response from Ollama.
    Yields text tokens as they arrive.
    messages: list of {"role": "user"|"assistant"|"system", "content": str}
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
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
        raise RuntimeError("Ollama is not running. Please start it with: ollama serve")


def is_running() -> bool:
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False
