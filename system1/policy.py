"""Routing policy for the System-1 -> System-2 cascade.

Pure functions of (gate answers, decisions) -> route. No torch, laya, vllm, yaml or
network imports, so the decision logic is testable without a GPU.

The gate's shape is data, not code. A *decision* binds a role to a question id and
the threshold that question is judged against:

    Decision(role="block", question="guardrail", threshold=0.5)

Roles are the pipeline's vocabulary and live here; which question answers which
role, and at what threshold, is configured. Every role is optional — a project
with no guardrail simply omits it, and nothing blocks.

Every gate question is a `noul`: a probability that a statement is true. That
shapes two things:

  * thresholds are probabilities, and 0.5 is a coin flip rather than the default;
  * "confidence" cannot be the raw probability. A `deliberation` answer of 0.12
    against a threshold of 0.15 is a near miss, not a confident "no" -- so
    confidence is the normalised distance from the decision threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# Gate roles, evaluated against the user's request.
BLOCK = "block"
SYSTEM1 = "system1"
THINK = "think"
GATE_ROLES = (BLOCK, SYSTEM1, THINK)

# Verifier roles, evaluated against a draft answer.
GROUNDED = "grounded"
DECLINED = "declined"
VERIFY_ROLES = (GROUNDED, DECLINED)
ALL_ROLES = GATE_ROLES + VERIFY_ROLES

# Route action when no gate decision fired. Not a role: nothing is configured to
# trigger it, it is simply the default, so it is absent from ALL_ROLES.
SYSTEM2 = "system2"

# Cascade outcomes for a drafted answer.
SHIP = "ship"
HANDOFF = "handoff"


@dataclass(frozen=True)
class Decision:
    """One configured decision: role -> question, judged at a probability threshold."""

    role: str
    question: str
    threshold: float

    def triggered(self, answers: Mapping[str, Any]) -> bool:
        return probability(answers, self.question) >= self.threshold


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


def probability(answers: Mapping[str, Any], question_id: str) -> float:
    """Read a `noul` answer, defaulting to 0.0 for a missing or malformed one.

    A non-`noul` answer is never coerced into a probability. Reading a `score` of
    2.0 as a probability would silently flip a threshold comparison.
    """
    answer = answers.get(question_id) or {}
    if answer.get("type") != "noul":
        return 0.0
    try:
        return float(answer.get("noul"))
    except (TypeError, ValueError):
        return 0.0


def index(decisions: Iterable[Decision]) -> dict[str, Decision]:
    """Role -> decision. Later entries win, so a config can override a base."""
    return {decision.role: decision for decision in decisions}


def margin_confidence(value: float, threshold: float) -> float:
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
    return min(1.0, abs(value - threshold) / scale)


def gate_confidence(
    answers: Mapping[str, Any],
    by_role: Mapping[str, Decision],
) -> float:
    """The gate's overall confidence: its least decisive decision.

    A request is only routed confidently if every decision taken about it was
    decisive, so this is a minimum, not a mean. Verifier roles are excluded: they
    judge a draft, not the request.
    """
    margins = [
        margin_confidence(probability(answers, d.question), d.threshold)
        for role, d in by_role.items()
        if role in GATE_ROLES
    ]
    return min(margins) if margins else 1.0


def decide(
    answers: Mapping[str, Any],
    by_role: Mapping[str, Decision],
    uncertain_below: float = 0.5,
) -> Route:
    """Map the System-1 answers onto a route.

    Order matters. Blocking is evaluated first and can only block, so a request that
    trips it never reaches the model however the other questions scored.
    """
    block = by_role.get(BLOCK)
    if block is not None and block.triggered(answers):
        return Route(
            action=BLOCK,
            think=False,
            reason=f"{block.question} {probability(answers, block.question):.3f} "
                   f">= {block.threshold}",
            gate_confidence=margin_confidence(
                probability(answers, block.question), block.threshold
            ),
        )

    confidence = gate_confidence(answers, by_role)
    uncertain = confidence < uncertain_below

    if uncertain:
        # The gate is not sure how it read the request. Take the cheap path: a
        # deliberate answer to a misread question is the most expensive mistake here.
        return Route(
            action=SYSTEM2,
            think=False,
            reason=f"gate margin {confidence:.3f} < {uncertain_below} -> cheap path",
            gate_confidence=confidence,
            uncertain=True,
        )

    system1 = by_role.get(SYSTEM1)
    if system1 is not None and system1.triggered(answers):
        return Route(
            action=SYSTEM1,
            think=False,
            reason=f"{system1.question} {probability(answers, system1.question):.3f} "
                   f">= {system1.threshold}",
            gate_confidence=confidence,
        )

    think = by_role.get(THINK)
    if think is not None and think.triggered(answers):
        return Route(
            action=SYSTEM2,
            think=True,
            reason=f"{think.question} {probability(answers, think.question):.3f} "
                   f">= {think.threshold} -> deliberate path",
            gate_confidence=confidence,
        )

    return Route(
        action=SYSTEM2,
        think=False,
        reason="no gate decision triggered -> fast path",
        gate_confidence=confidence,
    )


def verify(
    answers: Mapping[str, Any],
    by_role: Mapping[str, Decision],
) -> Verdict:
    """Turn the verifier's answers into a cascade outcome.

    A declined draft is different in kind from an unsupported one: the model said it
    has no answer, so escalating to a bigger model would spend more tokens to reach
    the same conclusion. That goes to a human instead.
    """
    grounded_d = by_role.get(GROUNDED)
    declined_d = by_role.get(DECLINED)

    grounded = probability(answers, grounded_d.question) if grounded_d else 0.0
    declined = probability(answers, declined_d.question) if declined_d else 0.0

    if declined_d is not None and declined >= declined_d.threshold:
        return Verdict(HANDOFF, grounded, declined, escalate=False, reason="draft declined")

    if grounded_d is not None and grounded >= grounded_d.threshold:
        return Verdict(SHIP, grounded, declined, escalate=False, reason="draft grounded")

    if grounded_d is None and declined_d is None:
        # No verifier configured: ship rather than refuse to answer.
        return Verdict(SHIP, grounded, declined, escalate=False, reason="no verifier configured")

    return Verdict(SHIP, grounded, declined, escalate=True, reason="draft not grounded")