"""Routing policy for the System-1 -> System-2 cascade.

Pure functions of (gate answers, policy) -> route. No torch, laya, vllm or network
imports, so the decision logic is testable without a GPU.

Every gate question is a `noul`: a probability that a statement is true. That shapes
two things here:

  * thresholds are compared against probabilities, and a threshold of 0.5 is a
    coin flip rather than the default;
  * "confidence" cannot be the raw probability. A `deliberation` answer of 0.12
    against a threshold of 0.15 is a near miss, not a confident "no" -- so
    confidence is the normalised distance from the decision threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

BLOCK = "block"
SYSTEM1 = "system1"
SYSTEM2 = "system2"

# Cascade outcomes for a drafted answer.
SHIP = "ship"
HANDOFF = "handoff"


@dataclass(frozen=True)
class Route:
    """What the pipeline decided to do, and why."""

    action: str
    think: bool
    reason: str
    gate_confidence: float = 0.0
    uncertain: bool = False

    @property
    def uses_llm(self) -> bool:
        return self.action == SYSTEM2


@dataclass(frozen=True)
class Verdict:
    """The verifier's reading of a draft answer."""

    outcome: str
    grounded: float
    declined: float
    escalate: bool = False
    reason: str = ""


def _probability(answers: Mapping[str, Any], question_id: str) -> float:
    """Read a `noul` answer, defaulting to 0.0 for a missing or malformed one."""
    answer = answers.get(question_id) or {}
    if answer.get("type") != "noul":
        return 0.0
    try:
        return float(answer.get("noul"))
    except (TypeError, ValueError):
        return 0.0


def margin_confidence(probability: float, threshold: float) -> float:
    """How decisive a probability is, relative to the threshold it is judged against.

    For a 0.5 threshold this reduces to max(p, 1-p), the usual binary confidence. For
    any other threshold it is the distance from the decision boundary, scaled by how
    far that boundary is from the nearer edge. A 0.12 against a 0.15 threshold scores
    0.04, not 0.88: it is a near miss, and treating it as confident is exactly how a
    gate starts escalating on requests it has not actually understood.
    """
    scale = max(threshold, 1.0 - threshold)
    if scale <= 0:
        return 0.0
    return min(1.0, abs(probability - threshold) / scale)


def gate_confidence(answers: Mapping[str, Any], policy: Mapping[str, Any]) -> float:
    """The gate's overall confidence: its least decisive decision.

    A request is only routed confidently if every decision taken about it was
    decisive, so this is a minimum, not a mean.
    """
    margins = [
        margin_confidence(_probability(answers, "guardrail"), float(policy.get("block_threshold", 0.5))),
        margin_confidence(_probability(answers, "classification"), float(policy.get("system1_threshold", 0.5))),
        margin_confidence(_probability(answers, "deliberation"), float(policy.get("think_threshold", 0.15))),
    ]
    return min(margins)


def decide(answers: Mapping[str, Any], policy: Mapping[str, Any]) -> Route:
    """Map the System-1 answers onto a route.

    Order matters. The guardrail is evaluated first and can only block, so a request
    that trips it never reaches the model however the other questions scored.
    """
    block_at = float(policy.get("block_threshold", 0.5))
    system1_at = float(policy.get("system1_threshold", 0.5))
    think_at = float(policy.get("think_threshold", 0.15))
    floor = float(policy.get("uncertain_below", 0.5))

    guardrail = _probability(answers, "guardrail")
    classification = _probability(answers, "classification")
    deliberation = _probability(answers, "deliberation")

    if guardrail >= block_at:
        return Route(
            action=BLOCK,
            think=False,
            reason=f"guardrail {guardrail:.3f} >= {block_at}",
            gate_confidence=margin_confidence(guardrail, block_at),
        )

    confidence = gate_confidence(answers, policy)
    uncertain = confidence < floor

    if uncertain:
        # The gate is not sure how it read the request. Take the cheap path: a
        # deliberate answer to a misread question is the most expensive mistake here.
        return Route(
            action=SYSTEM2,
            think=False,
            reason=f"gate margin {confidence:.3f} < {floor} -> cheap path",
            gate_confidence=confidence,
            uncertain=True,
        )

    if classification >= system1_at:
        return Route(
            action=SYSTEM1,
            think=False,
            reason=f"classification {classification:.3f} >= {system1_at}",
            gate_confidence=confidence,
        )

    if deliberation >= think_at:
        return Route(
            action=SYSTEM2,
            think=True,
            reason=f"deliberation {deliberation:.3f} >= {think_at} -> deliberate path",
            gate_confidence=confidence,
        )

    return Route(
        action=SYSTEM2,
        think=False,
        reason=f"deliberation {deliberation:.3f} < {think_at} -> fast path",
        gate_confidence=confidence,
    )


def verify(answers: Mapping[str, Any], policy: Mapping[str, Any]) -> Verdict:
    """Turn the verifier's answers into a cascade outcome.

    A declined draft is different in kind from an unsupported one: the model said it
    has no answer, so escalating to a bigger model would spend more tokens to reach
    the same conclusion. That goes to a human instead.
    """
    accept_at = float(policy.get("accept_confidence", 0.6))
    decline_at = float(policy.get("decline_confidence", 0.5))

    grounded = _probability(answers, "grounded")
    declined = _probability(answers, "declined")

    if declined >= decline_at:
        return Verdict(HANDOFF, grounded, declined, escalate=False, reason="draft declined")

    if grounded >= accept_at:
        return Verdict(SHIP, grounded, declined, escalate=False, reason="draft grounded")

    return Verdict(SHIP, grounded, declined, escalate=True, reason="draft not grounded")
