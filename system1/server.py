"""HTTP surface: an OpenAI-compatible endpoint with a System-1 gate in front.

`POST /v1/chat/completions` accepts the OpenAI chat schema so existing clients work
unchanged. The decision record is returned under a `laya` key, which OpenAI clients
ignore and which is the only way to see why a request was answered without the model.

Run:
    .venv-laya/bin/python -m system1.server --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config as config_module
from . import policy
from .gate import System1, flatten_state
from .llm import LLMClient
from .pipeline import Pipeline, PipelineResult

SERVED_NAME = "system1"


class Message(BaseModel):
    role: str
    content: Any = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float = 0.0
    # Escape hatch: force a reasoning mode instead of letting the gate pick.
    enable_thinking: bool | None = None


class SystemOneQuestion(BaseModel):
    type: str
    instructions: str
    criteria: Any = None
    labels: dict[str, str] | None = None


class SystemOneRequest(BaseModel):
    state: Any
    questions: dict[str, SystemOneQuestion] = Field(default_factory=dict)
    model: str | None = None
    max_len: int | None = None
    min_confidence: float | None = None


@dataclass
class State:
    pipeline: Pipeline
    llm: LLMClient


def _chat_body(result: PipelineResult, model: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.reasoning:
        message["reasoning_content"] = result.reasoning
    generation = result.generation
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": generation.finish_reason if generation else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": generation.prompt_tokens if generation else 0,
            "completion_tokens": generation.completion_tokens if generation else 0,
            "total_tokens": generation.total_tokens if generation else 0,
        },
        "laya": result.trace(),
    }


def _sse(chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Wrap text deltas in the chat-completion SSE frames OpenAI clients expect."""
    model = SERVED_NAME

    async def emit() -> AsyncIterator[bytes]:
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        async for delta in chunks:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
        final = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return emit()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = config_module.load(app.state.config_path)
        app.state.config = cfg

        gate = System1(
            questions=cfg.gate_questions(),
            verify_questions=cfg.verify_questions(),
            repo=cfg.laya.get("repo", ""),
            device=cfg.laya.get("device", "cuda:0"),
            expect_name=cfg.laya.get("expect_name", ""),
            preload=bool(cfg.laya.get("preload", True)),
            max_loaded=int(cfg.laya.get("max_loaded", 2)),
            min_confidence=cfg.laya.get("min_confidence") or None,
        )
        llm = LLMClient(
            base_url=cfg.llm["base_url"],
            model=cfg.llm["served_name"],
        )
        app.state.state = State(pipeline=Pipeline(cfg, gate, llm), llm=llm)
        try:
            yield
        finally:
            await llm.aclose()

    app = FastAPI(title="Laya System 1 + Qwen3.5-9B int4", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        state: State = app.state.state
        cfg: config_module.Config = app.state.config
        return {
            "status": "ok",
            "system1": {"model": cfg.laya.get("repo"), "device": cfg.laya.get("device")},
            "system2": {
                "model": cfg.llm.get("repo"),
                "served_name": cfg.llm.get("served_name"),
                "reachable": await state.llm.healthy(),
            },
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        state: State = app.state.state
        messages = [m.model_dump() for m in request.messages]
        if not messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        if request.stream:
            return StreamingResponse(
                _sse(state.pipeline.stream(messages, think_override=request.enable_thinking,
                                           max_tokens=request.max_tokens)),
                media_type="text/event-stream",
            )

        result = await state.pipeline.run(messages, think_override=request.enable_thinking,
                                          max_tokens=request.max_tokens)
        return _chat_body(result, SERVED_NAME)

    @app.post("/v1/systemone")
    async def systemone(request: SystemOneRequest) -> dict[str, Any]:
        """Laya passthrough, same request/response shape as TypeSafe's endpoint.

        Exposed so the decision model stays usable on its own -- the gate is a
        dependency of the chat path, not a private implementation detail.
        """
        state: State = app.state.state
        questions = {k: v.model_dump(exclude_none=True) for k, v in request.questions.items()}
        if not questions:
            questions = app.state.config.gate_questions()
        result = state.pipeline.gate.router.predict(
            request.state,
            questions,
            model=request.model,
            max_len=request.max_len,
            min_confidence=request.min_confidence,
        )
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args()

    import uvicorn

    cfg = config_module.load(args.config)
    host = cfg.server.get("host", "127.0.0.1")
    port = int(cfg.server.get("port", 8100))

    app = create_app()
    app.state.config_path = str(cfg.path)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
