"""Unit tests for the cascade routing policy.

These are the tests that matter for correctness: a wrong route here either answers
without the model when it should not, or silently burns a deliberate model call.
They run without torch, laya, vllm or a GPU.

    .venv-laya/bin/python -m pytest tests/ -q
"""

from system1.policy import (
    BLOCK,
    HANDOFF,
    SHIP,
    SYSTEM1,
    SYSTEM2,
    decide,
    gate_confidence,
    margin_confidence,
    verify,
)

POLICY = {
    "block_threshold": 0.5,
    "system1_threshold": 0.5,
    "think_threshold": 0.15,
    "accept_confidence": 0.6,
    "decline_confidence": 0.5,
    # 0.1, not 0.5: this checkpoint's deliberation probabilities occupy 0.03-0.24,
    # so a 0.5 floor would mark every request uncertain and disable both the
    # deliberate path and the System-1 shortcut.
    "uncertain_below": 0.1,
}

# Values measured on this machine by scripts/probe_questions2.py.
MEASURED = {
    "capital":   {"guardrail": 0.00, "classification": 0.05, "deliberation": 0.04},
    "summarise": {"guardrail": 0.00, "classification": 0.21, "deliberation": 0.03},
    "proof":     {"guardrail": 0.00, "classification": 0.06, "deliberation": 0.24},
}


def answers(guardrail=0.0, classification=0.0, deliberation=0.0):
    return {
        "guardrail": {"type": "noul", "noul": guardrail},
        "classification": {"type": "noul", "noul": classification},
        "deliberation": {"type": "noul", "noul": deliberation},
    }


def vq(**kwargs):
    return {
        "grounded": {"type": "noul", "noul": kwargs.get("grounded", 0.0)},
        "declined": {"type": "noul", "noul": kwargs.get("declined", 0.0)},
    }


# --- margin_confidence ---------------------------------------------------

def test_margin_at_half_threshold_is_the_usual_binary_confidence():
    assert margin_confidence(0.9, 0.5) == 0.8
    assert margin_confidence(0.1, 0.5) == 0.8


def test_margin_against_a_low_threshold_is_not_the_raw_probability():
    # 0.12 against a 0.15 threshold is a near miss, not a confident "no".
    assert margin_confidence(0.12, 0.15) == 0.03 / 0.85


def test_margin_is_zero_exactly_on_the_threshold():
    assert margin_confidence(0.15, 0.15) == 0.0
    assert margin_confidence(0.5, 0.5) == 0.0


# --- guardrail -----------------------------------------------------------

def test_guardrail_blocks_before_anything_else():
    # A maximally clear, deliberate request still blocks.
    route = decide(answers(guardrail=0.99, classification=0.9, deliberation=0.9), POLICY)
    assert route.action == BLOCK
    assert route.think is False
    assert not route.uses_llm


def test_guardrail_threshold_is_inclusive():
    assert decide(answers(guardrail=0.5), POLICY).action == BLOCK
    assert decide(answers(guardrail=0.49), POLICY).action != BLOCK


# --- uncertainty ---------------------------------------------------------

def test_uncertain_gate_falls_back_to_the_cheap_path_not_the_deep_one():
    # 0.13 sits 0.02 below the 0.15 deliberation threshold: too close to call.
    route = decide(answers(deliberation=0.13), POLICY)
    assert route.action == SYSTEM2
    assert route.uncertain is True
    assert route.think is False


def test_uncertainty_also_blocks_the_system1_shortcut():
    # classification clears its threshold, but deliberation is a near miss.
    route = decide(answers(classification=0.52, deliberation=0.14), POLICY)
    assert route.uncertain is True
    assert route.action == SYSTEM2
    assert route.uses_llm


def test_clearly_clear_gate_is_not_uncertain():
    assert decide(answers(deliberation=0.0), POLICY).uncertain is False


def test_system1_threshold_crossing():
    # The uncertainty band means a threshold is only crossed with margin: with
    # t=0.5 and uncertain_below=0.1, anything within 0.05 of 0.5 is "unsure".
    assert decide(answers(classification=0.6, deliberation=0.0), POLICY).action == SYSTEM1
    assert decide(answers(classification=0.52, deliberation=0.0), POLICY).uncertain is True
    assert decide(answers(classification=0.49, deliberation=0.0), POLICY).action == SYSTEM2


def test_think_threshold_crossing():
    assert decide(answers(deliberation=0.24), POLICY).think is True
    assert decide(answers(deliberation=0.2), POLICY).uncertain is True
    assert decide(answers(deliberation=0.06), POLICY).think is False


def test_a_probability_exactly_on_a_threshold_is_maximally_uncertain():
    # Sitting on the boundary is the one value where the gate genuinely cannot tell
    # which way to go, so it must take the cheap path rather than flip a coin.
    assert decide(answers(deliberation=0.15), POLICY).uncertain is True
    assert decide(answers(classification=0.5, deliberation=0.0), POLICY).uncertain is True


def test_missing_or_malformed_questions_do_not_crash_the_gate():
    assert decide({}, POLICY).action == SYSTEM2
    # A `score` answer where a `noul` is expected must not be read as a probability.
    assert decide({"deliberation": {"type": "score", "score": 2.0}}, POLICY).action == SYSTEM2
    assert decide({"deliberation": {"type": "noul", "noul": "high"}}, POLICY).action == SYSTEM2


# --- measured inputs -----------------------------------------------------

def test_measured_factual_lookup_takes_the_fast_path():
    route = decide(answers(**MEASURED["capital"]), POLICY)
    assert route.action == SYSTEM2
    assert route.think is False


def test_measured_summarise_takes_the_fast_path():
    route = decide(answers(**MEASURED["summarise"]), POLICY)
    assert route.action == SYSTEM2
    assert route.think is False


def test_measured_proof_escalates_to_deliberate():
    route = decide(answers(**MEASURED["proof"]), POLICY)
    assert route.action == SYSTEM2
    assert route.think is True


def test_gate_confidence_is_the_least_decisive_decision():
    confidence = gate_confidence(answers(**MEASURED["capital"]), POLICY)
    assert confidence == min(
        margin_confidence(0.00, 0.5),
        margin_confidence(0.05, 0.5),
        margin_confidence(0.04, 0.15),
    )


# --- verifier ------------------------------------------------------------

def test_grounded_draft_ships():
    verdict = verify(vq(grounded=0.9), POLICY)
    assert verdict.outcome == SHIP
    assert verdict.escalate is False


def test_ungrounded_draft_escalates():
    assert verify(vq(grounded=0.2), POLICY).escalate is True


def test_accept_threshold_is_inclusive():
    assert verify(vq(grounded=0.6), POLICY).escalate is False
    assert verify(vq(grounded=0.59), POLICY).escalate is True


def test_declined_draft_hands_off_without_escalating():
    verdict = verify(vq(grounded=0.1, declined=0.9), POLICY)
    assert verdict.outcome == HANDOFF
    assert verdict.escalate is False


def test_decline_wins_over_grounding():
    # Unsupported *and* a refusal is a handoff, not an escalation.
    assert verify(vq(grounded=0.05, declined=0.8), POLICY).outcome == HANDOFF
