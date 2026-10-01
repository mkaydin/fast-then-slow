"""Unit tests for the cascade routing policy.

These are the tests that matter for correctness: a wrong route here either answers
without the model when it should not, or silently burns a deliberate model call.
They run without torch, laya, vllm or a GPU.

    .venv-laya/bin/python -m pytest tests/ -q
"""

import pytest

from system1.policy import (
    BLOCK,
    HANDOFF,
    SHIP,
    SYSTEM1,
    SYSTEM2,
    THINK,
    Decision,
    decide,
    gate_confidence,
    margin_confidence,
    probability,
    verify,
)

FLOOR = 0.1  # matches config/pipeline.yaml's uncertain_below

# The shipped decision set, as a test would read it from config.
DECISIONS = (
    Decision(role=BLOCK, question="guardrail", threshold=0.5),
    Decision(role=SYSTEM1, question="classification", threshold=0.5),
    Decision(role=THINK, question="deliberation", threshold=0.15),
    Decision(role="grounded", question="grounded", threshold=0.6),
    Decision(role="declined", question="declined", threshold=0.8),
)
BY_ROLE = {d.role: d for d in DECISIONS}


def with_think(threshold: float):
    """Swap only the think threshold, the way the sweep does."""
    return {
        d.role: (Decision(d.role, d.question, threshold) if d.role == THINK else d)
        for d in DECISIONS
    }


def answers(guardrail=0.0, classification=0.0, deliberation=0.0):
    return {
        "guardrail": {"type": "noul", "noul": guardrail},
        "classification": {"type": "noul", "noul": classification},
        "deliberation": {"type": "noul", "noul": deliberation},
    }


def vq(grounded=0.0, declined=0.0):
    return {
        "grounded": {"type": "noul", "noul": grounded},
        "declined": {"type": "noul", "noul": declined},
    }


def route(state, by_role=None):
    return decide(state, by_role or BY_ROLE, FLOOR)


# --- probability ---------------------------------------------------------

def test_non_noul_answers_are_never_read_as_a_probability():
    # Reading a score of 2.0 as a probability would silently flip a comparison.
    assert probability({"x": {"type": "score", "score": 2.0}}, "x") == 0.0
    assert probability({"x": {"type": "choice", "choice": "yes"}}, "x") == 0.0


def test_missing_and_malformed_answers_default_to_zero():
    assert probability({}, "x") == 0.0
    assert probability({"x": None}, "x") == 0.0
    assert probability({"x": {"type": "noul", "noul": "high"}}, "x") == 0.0


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
    result = route(answers(guardrail=0.99, classification=0.9, deliberation=0.9))
    assert result.action == BLOCK
    assert result.think is False
    assert not result.uses_llm


def test_guardrail_threshold_is_inclusive():
    assert route(answers(guardrail=0.5)).action == BLOCK
    assert route(answers(guardrail=0.49)).action != BLOCK


# --- uncertainty ---------------------------------------------------------

def test_uncertain_gate_falls_back_to_the_cheap_path_not_the_deep_one():
    # 0.13 sits 0.02 below the 0.15 deliberation threshold: too close to call.
    result = route(answers(deliberation=0.13))
    assert result.action == SYSTEM2
    assert result.uncertain is True
    assert result.think is False


def test_uncertainty_also_blocks_the_system1_shortcut():
    result = route(answers(classification=0.52, deliberation=0.14))
    assert result.uncertain is True
    assert result.action == SYSTEM2


def test_clearly_clear_gate_is_not_uncertain():
    assert route(answers(deliberation=0.0)).uncertain is False


def test_a_probability_exactly_on_a_threshold_is_maximally_uncertain():
    # Sitting on the boundary is the one value where the gate cannot tell which way
    # to go, so it must take the cheap path rather than flip a coin.
    assert route(answers(deliberation=0.15)).uncertain is True
    assert route(answers(classification=0.5, deliberation=0.0)).uncertain is True


# --- system 1 only -------------------------------------------------------

def test_clear_classification_is_answered_by_system1_alone():
    result = route(answers(classification=0.9, deliberation=0.0))
    assert result.action == SYSTEM1
    assert not result.uses_llm


def test_system1_threshold_crossing():
    assert route(answers(classification=0.6, deliberation=0.0)).action == SYSTEM1
    assert route(answers(classification=0.52, deliberation=0.0)).uncertain is True
    assert route(answers(classification=0.49, deliberation=0.0)).action == SYSTEM2


# --- deliberation switch -------------------------------------------------

def test_think_threshold_crossing():
    assert route(answers(deliberation=0.24)).think is True
    assert route(answers(deliberation=0.2)).uncertain is True
    assert route(answers(deliberation=0.06)).think is False


def test_lowering_the_think_threshold_can_disable_the_system1_route():
    # The coupling that kept think_threshold at 0.15. margin confidence normalises by
    # max(t, 1-t), so a lower threshold shrinks the uncertainty band and a
    # classification scoring 0.000 deliberation reads as uncertain instead of
    # confidently "not deliberate". If this ever stops being true the config comment
    # justifying 0.15 is wrong.
    confident_classification = answers(classification=0.9, deliberation=0.0)
    assert decide(confident_classification, BY_ROLE, FLOOR).action == SYSTEM1
    assert decide(confident_classification, with_think(0.05), FLOOR).uncertain is True
    assert decide(confident_classification, with_think(0.05), FLOOR).action == SYSTEM2


def test_missing_questions_do_not_crash_the_gate():
    assert route({}).action == SYSTEM2
    assert route({"deliberation": {"type": "score", "score": 2.0}}).action == SYSTEM2


# --- optional roles ------------------------------------------------------

def test_a_project_with_no_guardrail_never_blocks():
    by_role = {r: d for r, d in BY_ROLE.items() if r != BLOCK}
    assert decide(answers(guardrail=0.99), by_role, FLOOR).action != BLOCK


def test_a_project_with_no_deliberation_never_thinks():
    by_role = {r: d for r, d in BY_ROLE.items() if r != THINK}
    assert decide(answers(deliberation=0.99), by_role, FLOOR).think is False


def test_a_project_with_no_system1_role_always_calls_the_model():
    by_role = {r: d for r, d in BY_ROLE.items() if r != SYSTEM1}
    assert decide(answers(classification=0.9), by_role, FLOOR).action == SYSTEM2


def test_an_empty_decision_set_still_routes_somewhere():
    result = decide(answers(deliberation=0.9), {}, FLOOR)
    assert result.action == SYSTEM2
    assert result.think is False
    # Nothing to be unsure about, so nothing marks the gate uncertain.
    assert gate_confidence(answers(), {}) == 1.0


# --- measured inputs -----------------------------------------------------

def test_measured_factual_lookup_takes_the_fast_path():
    result = route(answers(guardrail=0.00, classification=0.05, deliberation=0.04))
    assert result.action == SYSTEM2
    assert result.think is False


def test_measured_proof_escalates_to_deliberate():
    result = route(answers(guardrail=0.00, classification=0.06, deliberation=0.24))
    assert result.action == SYSTEM2
    assert result.think is True


def test_gate_confidence_is_the_least_decisive_decision():
    confidence = gate_confidence(answers(guardrail=0.0, classification=0.05, deliberation=0.04), BY_ROLE)
    assert confidence == min(
        margin_confidence(0.00, 0.5),
        margin_confidence(0.05, 0.5),
        margin_confidence(0.04, 0.15),
    )


def test_gate_confidence_ignores_verifier_roles():
    """Verifier roles judge a draft, not the request, so they must not gate routing."""
    by_role = dict(BY_ROLE)
    by_role["grounded"] = Decision("grounded", "grounded", 0.99)
    assert gate_confidence(answers(), by_role) == gate_confidence(answers(), BY_ROLE)


# --- verifier ------------------------------------------------------------

def test_grounded_draft_ships():
    verdict = verify(vq(grounded=0.9), BY_ROLE)
    assert verdict.outcome == SHIP
    assert verdict.escalate is False


def test_ungrounded_draft_escalates():
    assert verify(vq(grounded=0.2), BY_ROLE).escalate is True


def test_accept_threshold_is_inclusive():
    assert verify(vq(grounded=0.6), BY_ROLE).escalate is False
    assert verify(vq(grounded=0.59), BY_ROLE).escalate is True


def test_declined_draft_hands_off_without_escalating():
    verdict = verify(vq(grounded=0.1, declined=0.9), BY_ROLE)
    assert verdict.outcome == HANDOFF
    assert verdict.escalate is False


def test_decline_wins_over_grounding():
    # Unsupported *and* a refusal is a handoff, not an escalation.
    assert verify(vq(grounded=0.05, declined=0.8), BY_ROLE).outcome == HANDOFF


def test_a_project_with_no_verifier_ships_everything():
    verdict = verify(vq(grounded=0.0, declined=0.9), {})
    assert verdict.outcome == SHIP
    assert verdict.escalate is False


def test_verifier_role_binds_to_a_different_question():
    """The question id is config's choice, not the role's name."""
    by_role = {
        "grounded": Decision("grounded", "is_it_correct", 0.5),
        "declined": Decision("declined", "did_it_refuse", 0.5),
    }
    assert verify({"is_it_correct": {"type": "noul", "noul": 0.9}}, by_role).outcome == SHIP
    assert verify({"did_it_refuse": {"type": "noul", "noul": 0.9}}, by_role).outcome == HANDOFF


# --- config validation ---------------------------------------------------

def test_pipeline_config_loads_and_binds_every_role():
    from system1 import config as config_module

    cfg = config_module.load()
    roles = {d.role for d in cfg.decisions}
    assert {"block", "system1", "think", "grounded", "declined"} <= roles
    for decision in cfg.decisions:
        assert decision.question in cfg.questions


def test_a_deployment_file_is_not_required_to_read_policy(tmp_path):
    from system1 import config as config_module

    cfg = config_module.load(deployment=tmp_path / "absent.yaml")
    assert cfg.decisions  # policy still loads
    assert cfg.llm == {}  # deployment half is empty, not an error


@pytest.mark.parametrize(
    "bad,expected",
    [
        ([{"role": "nope", "question": "guardrail", "threshold": 0.5}], "known roles"),
        ([{"role": "block", "question": "guardrail", "threshold": 2.0}], r"within \[0,1\]"),
        (
            [
                {"role": "block", "question": "guardrail", "threshold": 0.5},
                {"role": "block", "question": "guardrail", "threshold": 0.6},
            ],
            "configured twice",
        ),
    ],
)
def test_malformed_decisions_are_rejected_with_a_useful_message(tmp_path, bad, expected):
    from system1 import config as config_module

    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "questions:\n"
        "  guardrail:\n"
        "    type: noul\n"
        "    instructions: Is this unsafe?\n"
        f"decisions: {bad!r}\n"
        "uncertain_below: 0.1\n"
    )
    with pytest.raises(ValueError, match=expected):
        config_module.load(pipeline=path, deployment=tmp_path / "absent.yaml")