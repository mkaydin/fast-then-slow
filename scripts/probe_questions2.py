"""Probe round 2: pick gate questions on measured discrimination, not on how they read.

Round 1 found two configuration-level defects:

  * the `score` primitive clustered in 0.45-1.29, so the configured
    `think_threshold: 2.0` was unreachable and `max_complexity: 0` almost never
    fired -- the escalation path could never turn on;
  * a 5-way `choice` for intent peaked at ~0.29 probability, so `uncertain_below:
    0.6` classified almost every request as uncertain, which also disables the
    System-1-only shortcut.

Both say the same thing: wide answer spaces wash out. Laya was trained with
strictly proper scoring rules, and `noul` is the primitive that shows sharp
separation (easy 0.042 vs hard 0.186 on a 4x ratio). This round measures several
binary `noul` questions on the same labelled cases and reports AUC, so the
configured set is the one that actually separates.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 USE_TF=0 \
      .venv-laya/bin/python scripts/probe_questions2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system1 import config as config_module
from system1.gate import System1

# (label, tier) where tier 0 = trivial, 1 = moderate, 2 = hard.
CASES = [
    ("capital", 0, "user: What is the capital of France?"),
    ("define", 0, "user: Define photosynthesis in one sentence."),
    ("classify", 0, "user: Label this support ticket as billing, bug or feature: 'The app crashes on export.'"),
    ("translate", 0, "user: Translate 'good morning' into German."),
    ("summarise", 1, "user: Summarise the plot of Hamlet in three sentences."),
    ("debug", 1, "user: Why does my Python script throw a KeyError on this dictionary lookup?"),
    ("compare", 1, "user: Compare Postgres and MySQL for a small analytics workload."),
    ("proof", 2, "user: Prove that the sum of the first n odd numbers is n squared."),
    ("code", 2, "user: Write a Python function that reverses a linked list and explain the recursion termination condition."),
    ("plan", 2, "user: Design a migration plan to move this monolith to microservices over two quarters."),
]

# Each candidate: question id -> noul definition. Labelled with which tier it should fire on.
CANDIDATES = {
    "q_needs_deliberation": {
        "tiers": {2},
        "definition": {
            "type": "noul",
            "instructions": (
                "Would answering this correctly require sustained multi-step reasoning, a "
                "derivation, or writing and debugging code?"
            ),
        },
    },
    "q_is_classification": {
        "tiers": {0},
        "definition": {
            "type": "noul",
            "instructions": (
                "Is the user asking you to label, categorise, tag, score, or sort something "
                "they supplied, rather than asking you for an answer?"
            ),
        },
    },
    "q_is_factual_lookup": {
        "tiers": {0},
        "definition": {
            "type": "noul",
            "instructions": (
                "Can this be answered with a single short factual statement, with no "
                "reasoning, calculation or analysis required?"
            ),
        },
    },
    "q_is_trivial_transform": {
        "tiers": {0},
        "definition": {
            "type": "noul",
            "instructions": (
                "Is this a routine text transformation -- translating, reformatting, or "
                "rewording a short piece of text -- that needs no reasoning?"
            ),
        },
    },
    "q_needs_derivation": {
        "tiers": {1, 2},
        "definition": {
            "type": "noul",
            "instructions": (
                "Does answering this correctly require you to work through several steps of "
                "reasoning, or to work out something rather than recall it?"
            ),
        },
    },
    "q_would_you_deliberate": {
        "tiers": {1, 2},
        "definition": {
            "type": "noul",
            "instructions": (
                "Would a careful assistant need to think carefully for a while before "
                "answering this, rather than reply straight away?"
            ),
        },
    },
}


def auc(positives: list[float], negatives: list[float]) -> float:
    """Probability a random positive outranks a random negative (ties count 0.5)."""
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positives
        for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def main() -> int:
    cfg = config_module.load()
    gate = System1(
        questions=cfg.gate_questions(),
        verify_questions=cfg.verify_questions(),
        repo=cfg.laya.get("repo", ""),
        device=cfg.laya.get("device", "cuda:0"),
        expect_name=cfg.laya.get("expect_name", ""),
        preload=False,
    )
    gate.gate(CASES[0][2])  # warm

    print(f"{'question':22s} {'target':>8s} {'AUC':>6s}   per-case")
    results = []
    for name, spec in CANDIDATES.items():
        qid = "probe"
        questions = {**cfg.gate_questions(), qid: spec["definition"]}
        scores: dict[str, float] = {}
        for label, tier, state in CASES:
            answer = gate.gate(state, questions).answer(qid)
            scores[label] = float(answer.get("noul", 0.0))

        targets = spec["tiers"]
        positives = [v for (l, t, _), v in zip(CASES, scores.values()) if t in targets]
        negatives = [v for (l, t, _), v in zip(CASES, scores.values()) if t not in targets]
        score_auc = auc(positives, negatives)
        results.append((name, score_auc))

        detail = " ".join(f"{l}={scores[l]:.2f}" for l, _, _ in CASES)
        print(f"{name:22s} {str(sorted(targets)):>8s} {score_auc:6.3f}   {detail}")

    print("\nbest by separation:")
    for name, score_auc in sorted(results, key=lambda r: (r[1] != r[1], -r[1]))[:3]:
        print(f"  {name:22s} {score_auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
