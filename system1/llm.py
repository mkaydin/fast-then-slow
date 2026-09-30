"""System-2 engine client: the local vLLM server over its OpenAI-compatible API.

Kept as an HTTP client on purpose. vLLM owns a ~10 GiB resident model and its own
CUDA context; sharing a process with the decision model would put the two on the
same device or force one CUDA context to fight for the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

# The two deliberate modes of the same checkpoint. Qwen3.5 is a hybrid reasoning
# model: `enable_thinking` decides whether the model spends tokens on a reasoning
# trace before answering. The gate picks between them.
FAST = False
DELIBERATE = True


@dataclass(frozen=True)
class Generation:
    text: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, messages, *, think: bool, max_tokens: int | None, temperature: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            # Qwen3.5 reads its reasoning switch from the chat template kwargs, so
            # it travels with the request rather than with the server's flags.
            "chat_template_kwargs": {"enable_thinking": think},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        think: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Generation:
        payload = self._payload(messages, think=think, max_tokens=max_tokens, temperature=temperature)
        response = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        return Generation(
            text=message.get("content") or "",
            reasoning=message.get("reasoning_content") or "",
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        think: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield content deltas as they arrive."""
        payload = self._payload(messages, think=think, max_tokens=max_tokens, temperature=temperature)
        payload["stream"] = True
        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False
