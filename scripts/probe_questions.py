"""Probe: are the System-1 questions in config/pipeline.yaml actually discriminating?

The first smoke run returned a complexity score of ~1.09 for "capital of France",
~1.09 for a ticket classification and ~1.09 for a linked-list coding task. A score
that never moves is not a decision input, it is a constant. This script measures
the spread of several question formulations over a fixed labelled set so the
question text can be chosen on evidence rather than on how it reads.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 USE_TF=0 \
      .venv-laya/bin/python scripts/probe_questions.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system1 import config as config_module
from system1.gate import System1

# (label, expected complexity tier 0/1/2, state)
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

# Alternative formulations, evaluated against the same cases.
VARIANTS = {
    "score_ordinal": {
        "complexity": {
            "type": "score",
            "instructions": (
                "How much deliberate reasoning would a correct answer require? "
                "0 means the answer is immediately available. 2 means it needs "
                "sustained multi-step reasoning, calculation, or code."
            ),
            "criteria": [
                "immediately available",
                "some thought needed",
                "sustained multi-step reasoning",
            ],
        }
    },
    "score_yesno_binary": {
        "complexity": {
            "type": "score",
            "instructions": (
                "How much work is this? 0 = answer it right now with what you already "
                "know. 1 = it takes real thought: derivation, debugging, multi-part answers."
            ),
            "criteria": ["trivially answerable right now", "requires real work"],
        }
    },
    "noul_hard": {
        "complexity": {
            "type": "noul",
            "instructions": (
                "Would answering this correctly require sustained multi-step reasoning, "
                "a derivation, or writing and debugging code?"
            ),
        }
    },
    "noul_thinking": {
        "complexity": {
            "type": "noul",
            "instructions": (
                "Is this the kind of request where a careful model should deliberate "
                "for a while before answering, rather than reply immediately?"
            ),
        }
    },
}


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

    # Warm the kernels so the first timed call is not a compile.
    gate.gate(CASES[0][2])

    for name, question in VARIANTS.items():
        probe_questions = {**cfg.gate_questions(), **question}
        rows = []
        for label, expected, state in CASES:
            result = gate.gate(state, probe_questions)
            rows.append((label, expected, result.answer("complexity")))

        if question["complexity"]["type"] == "noul":
            value = lambda a: a.get("noul", 0.0)  # noqa: E731
        else:
            value = lambda a: a.get("score", 0.0)  # noqa: E731

        easy = [value(a) for _, e, a in rows if e == 0]
        hard = [value(a) for _, e, a in rows if e == 2]
        spread = max(value(a) for _, _, a in rows) - min(value(a) for _, _, a in rows)

        print(f"\n=== {name} ===")
        for label, expected, answer in rows:
            print(f"  {label:10s} expected={expected} got={value(answer):.3f}  {json.dumps(answer)[:90]}")
        print(f"  easy mean {statistics.mean(easy):.3f} | hard mean {statistics.mean(hard):.3f} | spread {spread:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
