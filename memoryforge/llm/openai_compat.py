"""Backend for any server that speaks the `/v1/chat/completions` wire format.

Works with Ollama, vLLM, LM Studio, llama.cpp server and most hosted gateways.
"""

from __future__ import annotations

import random
import time
from typing import Sequence

import httpx

from .base import Completion, LLMError, Message, Usage, UsageMeter

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleLLM:
    is_simulation = False

    def __init__(self, base_url: str, model: str, api_key: str = "", max_retries: int = 5) -> None:
        self.name = f"openai_compatible:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.meter = UsageMeter()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0),
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion:
        body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed  # best effort; many servers ignore it

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(f"{self.base_url}/chat/completions", json=body)
            except httpx.TransportError as e:
                last_err = e
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json())
                if resp.status_code not in RETRY_STATUS:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                last_err = LLMError(f"HTTP {resp.status_code}")
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(30.0, float(retry_after)))
                    continue
            time.sleep(min(30.0, 0.5 * 2**attempt) * random.uniform(0.5, 1.5))
        raise LLMError(f"gave up after {self.max_retries} attempts: {last_err}")

    def _parse(self, data: dict) -> Completion:
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens", 0)),
            completion_tokens=int(u.get("completion_tokens", len(text) // 4)),
        )
        self.meter.add(usage)
        return Completion(text=text, usage=usage, backend="openai_compatible")
