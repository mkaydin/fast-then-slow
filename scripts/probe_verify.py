"""Probe the verifier questions.

The first end-to-end run flagged a correct answer as `declined` (0.75) and handed it
off. A verifier that cries wolf on every draft would stop the pipeline from ever
shipping anything, so the wording has to be measured like any other question.

Each draft is paired with what the pipeline should do with it.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 USE_TF=0 \
      .venv-laya/bin/python scripts/probe_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system1 import config as config_module
from system1 import policy
from system1.gate import System1

REQUEST = "In one sentence: what is a semaphore?"

# (label, draft, expected outcome)
DRAFTS = [
    (
        "good",
        "A semaphore is a synchronisation primitive that limits how many threads may "
        "hold a shared resource at once, using a counter with atomic wait and signal.",
        policy.SHIP,
    ),
    (
        # Verbatim output of the 9B engine, which is what the verifier actually sees.
        # The hand-written draft above is not a representative sample.
        "real model output",
        "A semaphore is a synchronization mechanism used in concurrent programming to "
        "control access to shared resources by multiple threads or processes, preventing "
        "race conditions through the use of atomic operations like wait and signal.",
        policy.SHIP,
    ),
    (
        "fabricated",
        "A semaphore was invented in 1847 by Thomas Edison for the Pneumatic Telegraph "
        "Company and stores exactly four threads.",
        policy.HANDOFF,  # escalated here, but it is certainly not a refusal
    ),
    (
        "explicit refusal",
        "I can't help with that request.",
        policy.HANDOFF,
    ),
    (
        "hedged refusal",
        "I'm not able to answer this question.",
        policy.HANDOFF,
    ),
]

VARIANTS = {
    "current": {},
    "declines_to_answer": {
        "declined": {
            "type": "noul",
            "instructions": (
                "Is the assistant refusing or declining to give an answer, for example by "
                "saying it cannot help, is unable to answer, or that the question should not "
                "be asked? Answer no if the assistant gave a substantive answer."
            ),
        }
    },
    "no_substantive_answer": {
        "declined": {
            "type": "noul",
            "instructions": (
                "Does the assistant's reply contain no actual answer to the question -- for "
                "instance only an apology, a refusal, or a statement that it does not know?"
            ),
        }
    },
}


def main() -> int:
    cfg = config_module.load()
    gate = System1(
        questions=cfg.gate_questions,
        verify_questions=cfg.verify_questions,
        repo=cfg.laya.get("repo", ""),
        device=cfg.laya.get("device", "cuda:0"),
        expect_name=cfg.laya.get("expect_name", ""),
        preload=bool(cfg.laya.get("preload", True)),
        max_loaded=int(cfg.laya.get("max_loaded", 2)),
    )
    gate.verify(REQUEST, DRAFTS[0][1])  # warm

    for name, override in VARIANTS.items():
        questions = {**cfg.verify_questions, **override}
        print(f"\n=== {name} ===")
        mismatches = 0
        for label, draft, expected in DRAFTS:
            answers = gate.verify(REQUEST, draft, questions=questions).answers
            verdict = policy.verify(answers, cfg.by_role)
            got = "escalate" if verdict.escalate else verdict.outcome
            ok = got == expected or (expected == policy.HANDOFF and got == "escalate")
            mismatches += 0 if ok else 1
            print(
                f"  {label:18s} grounded={answers['grounded']['noul']:.3f} "
                f"declined={answers['declined']['noul']:.3f} -> {got:9s} "
                f"(want {expected:8s}) {'ok' if ok else 'MISMATCH'}"
            )
        print(f"  mismatches: {mismatches}/{len(DRAFTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
