"""Small standard-library client for reproducible local Ollama generation."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


SOLUTION_SCHEMA = {
    "type": "object",
    "properties": {"solution": {"type": "string"}},
    "required": ["solution"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class OllamaReply:
    content: dict[str, Any]
    elapsed_seconds: float
    prompt_tokens: int
    output_tokens: int
    output_tokens_per_second: float | None


def chat_json(
    endpoint: str,
    model: str,
    system: str,
    prompt: str,
    *,
    schema: dict[str, Any],
    seed: int,
    timeout: float,
    keep_alive: str = "10m",
) -> OllamaReply:
    """Request one non-streaming JSON response from Ollama."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": schema,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": seed,
            "num_predict": 1536,
        },
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            row = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    elapsed = time.monotonic() - started
    raw_content = row.get("message", {}).get("content", "")
    try:
        content = json.loads(raw_content)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        excerpt = str(raw_content).strip().replace("\n", " ")[:300]
        raise ValueError(
            "Ollama did not return valid structured JSON; "
            f"content starts with {excerpt!r}"
        ) from exc
    if not isinstance(content, dict):
        raise ValueError("Ollama JSON response is not an object")
    output_tokens = int(row.get("eval_count", 0))
    eval_duration = int(row.get("eval_duration", 0))
    rate = output_tokens / (eval_duration / 1e9) if output_tokens and eval_duration else None
    return OllamaReply(
        content=content,
        elapsed_seconds=elapsed,
        prompt_tokens=int(row.get("prompt_eval_count", 0)),
        output_tokens=output_tokens,
        output_tokens_per_second=rate,
    )
