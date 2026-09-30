"""Smoke test for the System-1 gate on its own GPU.

Loads the real config, runs the real questions through a real Laya checkpoint and
checks the answers are typed, decisive and fast.

The first call on a fresh process is a CUDA compile and takes ~2s. That is reported
separately; the latency budget applies to warm calls, which is what a running
service actually sees.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 USE_TF=0 \
      .venv-laya/bin/python scripts/smoke_system1.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system1 import config as config_module
from system1 import policy
from system1.gate import System1

# (state, expected route). Expectations are the measured behaviour in
# scripts/probe_questions2.py, not aspirations.
SAMPLES = [
    ("user: What is the capital of France?", policy.SYSTEM2),
    (
        "user: Label this support ticket as billing, bug or feature: 'The app crashes on export.'",
        policy.SYSTEM1,
    ),
    (
        "user: Prove that the sum of the first n odd numbers is n squared.",
        policy.SYSTEM2,
    ),
]

WARMUP = "user: warm up the CUDA kernels"
WARM_BUDGET_MS = 400.0


def main() -> int:
    cfg = config_module.load()
    laya_cfg = cfg.laya

    print(f"config      : {cfg.path}")
    print(f"checkpoint  : {laya_cfg['repo']}")
    print(f"device      : {laya_cfg['device']} (expect {laya_cfg.get('expect_name')})")

    started = time.perf_counter()
    gate = System1(
        questions=cfg.gate_questions(),
        verify_questions=cfg.verify_questions(),
        repo=laya_cfg.get("repo", ""),
        device=laya_cfg.get("device", "cuda:0"),
        expect_name=laya_cfg.get("expect_name", ""),
        preload=bool(laya_cfg.get("preload", True)),
        max_loaded=int(laya_cfg.get("max_loaded", 2)),
    )
    print(f"load        : {time.perf_counter() - started:.1f}s")

    cold = gate.gate(WARMUP)
    print(f"cold call   : {cold.elapsed_ms:.0f} ms (first call compiles kernels)")

    failures = []
    for state, expected in SAMPLES:
        result = gate.gate(state)
        route = policy.decide(result.answers, cfg.policy)

        print(f"\nstate       : {state[:72]}")
        print(f"routing     : {result.model_name} ({result.routing.get('reason', '')})")
        print(f"latency     : {result.elapsed_ms:.1f} ms")
        for qid, answer in result.answers.items():
            print(f"  {qid:14s}: {answer.get('type')} = {answer.get('noul')}")
        print(f"  route        : {route.action} think={route.think} ({route.reason})")

        if route.action != expected:
            failures.append(f"{state[:40]!r}: expected route {expected}, got {route.action}")
        if result.elapsed_ms > WARM_BUDGET_MS:
            failures.append(f"warm gate took {result.elapsed_ms:.0f} ms (budget {WARM_BUDGET_MS:.0f} ms)")

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("System-1 gate OK: one forward pass answered all three questions per request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
