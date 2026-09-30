"""The cascade: gate -> route -> generate -> verify -> (escalate | ship).

The ordering is the whole design. The System-1 pass runs before any token is
generated, so a request it resolves never pays the System-2 engine's time-to-first
token at all. The verifier runs after, so a draft the gate does not trust gets one
deliberate retry before it is shown to anyone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from . import policy
from .config import Config
from .gate import GateResult, System1, flatten_state
from .llm import DELIBERATE, FAST, Generation, LLMClient

REFUSAL = (
    "This request was blocked before generation by the System-1 safety gate. "
    "No model was called."
)


@dataclass
class PipelineResult:
    route: str
    content: str
    gate: GateResult
    reasoning: str = ""
    generation: Generation | None = None
    verdict: policy.Verdict | None = None
    escalated: bool = False
    think: bool = False
    # The gate's least-decisive margin, copied from the Route at decision time.
    # Recomputing it here would need the policy, which the result does not carry.
    gate_confidence: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def used_llm(self) -> bool:
        return self.generation is not None

    def trace(self) -> dict[str, Any]:
        """The decision record: what the gate saw, what it decided, what was paid."""
        return {
            "route": self.route,
            "think": self.think,
            "escalated": self.escalated,
            "gate_model": self.gate.model_name,
            "gate_answers": self.gate.answers,
            "gate_confidence": self.gate_confidence,
            "verdict": None
            if self.verdict is None
            else {
                "grounded": self.verdict.grounded,
                "declined": self.verdict.declined,
                "outcome": self.verdict.outcome,
                "reason": self.verdict.reason,
            },
            "tokens": {
                "prompt": self.generation.prompt_tokens if self.generation else 0,
                "completion": self.generation.completion_tokens if self.generation else 0,
            },
            "timings_ms": self.timings_ms,
        }


def render_system1(answers: dict[str, Any]) -> str:
    """Render the System-1 decision as a reply.

    Laya never writes text. This is a template over its typed answers, and it says
    so, because a rendered decision presented as a generated answer is exactly the
    kind of thing this pipeline is supposed to avoid.
    """
    lines = ["[System-1 decision - classified, not generated]"]
    classification = answers.get("classification") or {}
    deliberation = answers.get("deliberation") or {}

    if classification.get("noul") is not None:
        lines.append(f"classification request: {classification['noul']:.2f}")
    if deliberation.get("noul") is not None:
        lines.append(f"needs deliberation: {deliberation['noul']:.2f}")
    lines.append("This request was classified rather than answered; no model was called.")
    return "\n".join(lines)


class Pipeline:
    def __init__(self, config: Config, gate: System1, llm: LLMClient) -> None:
        self.config = config
        self.gate = gate
        self.llm = llm

    async def run(
        self,
        messages: list[dict[str, Any]],
        *,
        think_override: bool | None = None,
        max_tokens: int | None = None,
    ) -> PipelineResult:
        state = flatten_state(messages)
        timings: dict[str, float] = {}

        started = time.perf_counter()
        gate_result = self.gate.gate(state)
        timings["gate"] = (time.perf_counter() - started) * 1000

        route = policy.decide(gate_result.answers, self.config.policy)

        if route.action == policy.BLOCK:
            return PipelineResult(
                route=policy.BLOCK,
                content=REFUSAL,
                gate=gate_result,
                gate_confidence=route.gate_confidence,
                timings_ms=timings,
            )

        if route.action == policy.SYSTEM1:
            return PipelineResult(
                route=policy.SYSTEM1,
                content=render_system1(gate_result.answers),
                gate=gate_result,
                gate_confidence=route.gate_confidence,
                timings_ms=timings,
            )

        think = route.think if think_override is None else think_override

        started = time.perf_counter()
        draft = await self.llm.generate(messages, think=think, max_tokens=max_tokens)
        timings["generate"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        verdict = self._verify(messages, draft)
        timings["verify"] = (time.perf_counter() - started) * 1000

        generation, escalated = draft, False
        if verdict.escalate:
            # One deliberate retry. If the fast path was already the deliberate one,
            # a second pass would just spend the same tokens again.
            if not think:
                started = time.perf_counter()
                generation = await self.llm.generate(
                    messages, think=DELIBERATE, max_tokens=max_tokens
                )
                timings["escalate"] = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                verdict = self._verify(messages, generation)
                timings["reverify"] = (time.perf_counter() - started) * 1000
                escalated = True

        content = generation.text
        if not content.strip() and generation.finish_reason == "length":
            # Deliberate mode can burn the whole budget on reasoning and return no
            # answer. finish_reason is the available signal, not the reasoning text:
            # --reasoning-parser qwen3 on vLLM 0.30 reports reasoning_tokens in usage
            # but leaves reasoning_content empty.
            content = (
                "The model used the entire token budget without producing an answer. "
                "Raise max_tokens, or re-run with enable_thinking=false to skip deliberation."
            )

        return PipelineResult(
            route=policy.SYSTEM2,
            content=content,
            gate=gate_result,
            reasoning=generation.reasoning,
            generation=generation,
            verdict=verdict,
            escalated=escalated,
            think=think or escalated,
            gate_confidence=route.gate_confidence,
            timings_ms=timings,
        )

    def _verify(self, messages, generation: Generation) -> policy.Verdict:
        """One System-1 pass over the draft. Both verifier questions are answered
        in the same forward pass, so verification costs the same as the gate."""
        result = self.gate.verify(
            request=flatten_state(messages),
            draft=generation.text,
        )
        return policy.verify(result.answers, self.config.policy)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        think_override: bool | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream the System-2 answer.

        The gate is resolved first, exactly as in `run`, so a blocked or classified
        request never opens a stream against the engine at all.
        """
        state = flatten_state(messages)
        gate_result = self.gate.gate(state)
        route = policy.decide(gate_result.answers, self.config.policy)

        if route.action == policy.BLOCK:
            yield render_system1(gate_result.answers) if route.action == policy.SYSTEM1 else REFUSAL
            return
        if route.action == policy.SYSTEM1:
            yield render_system1(gate_result.answers)
            return

        think = route.think if think_override is None else think_override
        async for delta in self.llm.stream(messages, think=think, max_tokens=max_tokens):
            yield delta
