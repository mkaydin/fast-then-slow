"""Sweep the routing thresholds on a labelled set.

`config/pipeline.yaml` says `think_threshold: 0.15` is provisional, and it is: it was
picked from a 10-case probe where the two classes were separated by 0.01. This runs
the real gate over a wider labelled set and reports, for each candidate threshold,
how often the route would have been right.

What is being predicted here is the *route the policy should take*, not whether the
model is a good model. The labels are the intended behaviour:

  think      -> the request should run in deliberate mode
  fast       -> deliberate mode would be wasted; the fast path should answer it
  system1    -> a classification request, answerable from the decision alone

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 USE_TF=0 \
      .venv-laya/bin/python scripts/sweep_thresholds.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system1 import config as config_module
from system1 import policy
from system1.gate import System1

THINK, FAST, SYSTEM1 = "think", "fast", "system1"

# (state, intended route)
CASES = [
    ("What is the capital of France?", FAST),
    ("Define photosynthesis in one sentence.", FAST),
    ("Translate 'good morning' into German.", FAST),
    ("What time zone is UTC+2?", FAST),
    ("Who wrote Pride and Prejudice?", FAST),
    ("What is the boiling point of water?", FAST),
    ("Label this ticket as billing, bug or feature: 'The app crashes on export.'", SYSTEM1),
    ("Tag this review as positive, negative or neutral: 'Delivery was quick, packaging was fine.'", SYSTEM1),
    ("Classify this email as spam or not: 'Congratulations, you have won a prize!'", SYSTEM1),
    ("Sort these into buckets: alpha, beta, gamma.", SYSTEM1),
    ("Rate the urgency of this ticket: 'The checkout page is down for all customers.'", SYSTEM1),
    ("Is this sentence sentiment positive or negative: 'I love this product'?", SYSTEM1),
    ("What is the HTTP status code for not found?", FAST),
    ("How do I reverse a string in Python?", THINK),
    ("Why does my Python script throw a KeyError here?", THINK),
    ("Compare Postgres and MySQL for a small analytics workload.", THINK),
    ("Summarise the plot of Hamlet in three sentences.", THINK),
    ("What are the tradeoffs between REST and gRPC?", THINK),
    ("Explain the difference between a mutex and a semaphore.", THINK),
    ("Debug this: my Docker build fails at the pip install step.", THINK),
    ("Prove that the sum of the first n odd numbers is n squared.", THINK),
    ("Write a Python function that reverses a linked list.", THINK),
    ("Design a migration plan to move this monolith to microservices.", THINK),
    ("Derive the closed form for the sum of a geometric series.", THINK),
    ("Find the bug in this SQL query and fix it.", THINK),
    ("What are the symptoms of iron deficiency?", FAST),
    ("Convert 30 degrees Celsius to Fahrenheit.", FAST),
    ("What does the acronym HTTP stand for?", FAST),
    ("Is the Pacific the largest ocean?", FAST),
    ("Give me three ideas for a birthday gift under $20.", THINK),
    ("Which of these two regexes is faster and why?", THINK),
    ("Refactor this function to be iterative and explain the tradeoffs.", THINK),
    ("Work out how many distinct ways to climb a staircase of n steps.", THINK),
    ("Explain why the sky appears blue.", THINK),
    ("What version of Python introduced match statements?", FAST),
    ("Rate this code review comment as minor or blocking.", SYSTEM1),
    ("Classify this transaction as fraudulent or legitimate.", SYSTEM1),
    ("Is this customer message expressing frustration? Score it 0-2.", SYSTEM1),
    ("Walk me through setting up a virtualenv and installing packages.", FAST),
    ("Critique the architectural tradeoffs of event sourcing.", THINK),
]

CANDIDATES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]


def main() -> int:
    cfg = config_module.load()
    base_policy = dict(cfg.policy)
    gate = System1(
        questions=cfg.gate_questions(),
        verify_questions=cfg.verify_questions(),
        repo=cfg.laya.get("repo", ""),
        device=cfg.laya.get("device", "cuda:0"),
        expect_name=cfg.laya.get("expect_name", ""),
        preload=bool(cfg.laya.get("preload", True)),
        max_loaded=int(cfg.laya.get("max_loaded", 2)),
    )
    gate.gate(CASES[0][0])  # warm

    # One pass over the set; reuse the answers for every candidate threshold.
    scores: list[tuple[str, dict]] = []
    latencies: list[float] = []
    for state, _ in CASES:
        result = gate.gate(state)
        scores.append((state, result.answers))
        latencies.append(result.elapsed_ms)

    print(f"cases: {len(CASES)}   gate latency median {statistics.median(latencies):.1f} ms\n")

    print(f"{'threshold':>9} {'think acc':>10} {'think FP':>9} {'think FN':>9} {'s1 hits':>8}")
    best = None
    for threshold in CANDIDATES:
        trial = {**base_policy, "think_threshold": threshold}
        tp = fp = fn = 0
        s1_hits = 0
        for (state, want), (_, answers) in zip(CASES, scores):
            route = policy.decide(answers, trial)
            if route.action == policy.SYSTEM1:
                if want == SYSTEM1:
                    s1_hits += 1
                continue
            got_think = route.think
            want_think = want == THINK
            if got_think and want_think:
                tp += 1
            elif got_think and not want_think:
                fp += 1
            elif not got_think and want_think:
                fn += 1
        # `want` is one of three labels; count correct as think/fast/system1 agreement.
        correct = 0
        for (state, want), (_, answers) in zip(CASES, scores):
            route = policy.decide(answers, trial)
            if route.action == policy.SYSTEM1:
                correct += 1 if want == SYSTEM1 else 0
            elif route.think:
                correct += 1 if want == THINK else 0
            else:
                correct += 1 if want == FAST else 0
        accuracy = correct / len(CASES)
        print(f"{threshold:9.2f} {accuracy:10.3f} {fp:9d} {fn:9d} {s1_hits:8d}")
        if best is None or accuracy > best[1]:
            best = (threshold, accuracy)

    configured = float(base_policy["think_threshold"])
    print(f"\nconfigured think_threshold = {configured}")
    print(f"best on this set           = {best[0]} (accuracy {best[1]:.3f})")

    deliberate = [a["deliberation"]["noul"] for _, a in scores]
    print(f"\ndeliberation range: {min(deliberate):.3f} - {max(deliberate):.3f}")
    print("per-case deliberation (sorted):")
    for (state, want), value in sorted(
        zip([c for c in CASES], deliberate), key=lambda pair: pair[1]
    ):
        print(f"  {value:.3f}  {want:8s} {state[:62]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
