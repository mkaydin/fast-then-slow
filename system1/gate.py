"""System-1 gate: the Laya decision model.

Laya is non-autoregressive. One forward pass answers every question about a state,
so the whole gate -- guardrail, intent and complexity -- costs one pass, not three.

The checkpoint lives on its own GPU (see PLAN.md §2) so the gate can never take
KV cache away from the System-2 engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GateResult:
    """One Laya forward pass, plus the bookkeeping the pipeline reports."""

    answers: dict[str, Any]
    routing: dict[str, Any]
    elapsed_ms: float
    model: str = ""

    @property
    def model_name(self) -> str:
        return self.model or str(self.routing.get("model", "unknown"))

    def answer(self, question_id: str) -> dict[str, Any]:
        return self.answers.get(question_id) or {}


def _assert_device(device: str, expect_name: str) -> None:
    """Fail fast if the gate is not on the GPU it was configured for.

    CUDA device indices are not stable across boots on this machine: default
    enumeration reports cuda:0 as the 5060 Ti, and PCI_BUS_ID ordering reports it
    as such too, but the two disagree about the 4060. Silently loading the gate
    onto the card the 9B engine is holding would cost KV cache and, in the other
    direction, would not fit at all. A wrong GPU should be a startup error.
    """
    if not expect_name or not str(device).startswith("cuda"):
        return
    import torch

    index = int(str(device).split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"System-1 device {device!r} does not exist; {torch.cuda.device_count()} CUDA "
            "device(s) visible. Check CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER."
        )
    actual = torch.cuda.get_device_name(index)
    if expect_name.lower() not in actual.lower():
        raise RuntimeError(
            f"System-1 device {device!r} is {actual!r}, which does not match the "
            f"configured {expect_name!r}. The two models would contend for VRAM."
        )



@dataclass
class System1:
    """Owns the Laya Router for the lifetime of the process."""

    questions: dict[str, Any]
    verify_questions: dict[str, Any]
    repo: str = "convaiinnovations/laya"
    device: str = "cuda:0"
    expect_name: str = ""
    preload: bool | list[str] = True
    max_loaded: int = 2
    min_confidence: float | None = None
    _router: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from laya import Router

        _assert_device(self.device, self.expect_name)

        kwargs: dict[str, Any] = {
            "device": self.device,
            "max_loaded": self.max_loaded,
        }
        if self.repo:
            kwargs["models"] = {"english": self.repo}
        # Router(preload=True) loads *every* checkpoint: ~3.2 GB rather than ~0.9 GB.
        # On a card shared with the 9B engine that starves the gate of cuBLAS
        # workspace and it dies with CUBLAS_STATUS_ALLOC_FAILED on the first request.
        # Passing a list preloads only those named.
        kwargs["preload"] = self.preload if isinstance(self.preload, bool) else False
        self._router = Router(**kwargs)
        if isinstance(self.preload, list) and self.preload:
            self._router.preload(self.preload)

    @property
    def router(self) -> Any:
        if self._router is None:  # pragma: no cover - only if __post_init__ is bypassed
            raise RuntimeError("System1 router is not initialised")
        return self._router

    def _run(self, state: Any, questions: dict[str, Any]) -> GateResult:
        started = time.perf_counter()
        result = self.router.predict(state, questions, min_confidence=self.min_confidence)
        elapsed_ms = (time.perf_counter() - started) * 1000
        routing = dict(result.get("routing") or {})
        return GateResult(
            answers=dict(result.get("answers") or {}),
            routing=routing,
            elapsed_ms=elapsed_ms,
            model=str(routing.get("model", "")),
        )

    def gate(self, state: Any, questions: dict[str, Any] | None = None) -> GateResult:
        """One pass over the user's request: guardrail + intent + complexity.

        `questions` overrides the configured set, which is what the question probe
        uses to compare formulations against the same labelled cases.
        """
        return self._run(state, questions or self.questions)

    def verify(
        self,
        request: str,
        draft: str,
        context: str = "",
        questions: dict[str, Any] | None = None,
    ) -> GateResult:
        """One pass over a draft answer, checking it against the request.

        The draft and the request go in as a structured state so the verifier sees
        the same text the model produced, not a paraphrase of it. `questions`
        overrides the configured set, for the verifier probe.
        """
        state: dict[str, str] = {"user_request": request, "draft_answer": draft}
        if context:
            state["context"] = context
        return self._run(state, questions or self.verify_questions)


def flatten_state(messages: list[Mapping[str, Any]]) -> str:
    """Render an OpenAI-style message list into the text Laya reads.

    Roles are kept as a prefix: "which team should handle this" depends on who is
    speaking, and dropping roles would silently change the answer.
    """
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if isinstance(content, list):
            # Multimodal parts: only text reaches a text-only decision model.
            content = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)
