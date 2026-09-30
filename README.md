# Laya System 1 + Qwen3.5-9B int4

A local two-tier inference pipeline. **Laya** is the System-1 reflex: one
non-autoregressive forward pass answers typed decision questions in ~28 ms.
**Qwen3.5-9B int4** on vLLM is System 2: it generates. Laya decides *whether* to
generate, *how much deliberation* it deserves, and *whether to trust the result*.
It never writes text, so there is nothing to parse and nothing to hallucinate.

```
   POST /v1/chat/completions
              │
              ▼
   ┌──── Laya, one forward pass (~28 ms, RTX 4060) ────┐
   │  guardrail     noul   block?           → refuse,   │  no model call
   │  classification noul  answerable by S1? → reply    │  no model call
   │  deliberation  noul   needs thought?    → think?    │
   └────────────────────────────┬───────────────────────┘
                                │ system2
                ┌───────────────┴────────────────┐
                ▼                                ▼
      fast path (enable_thinking=false)   deliberate path (thinking=true)
                └───────────────┬────────────────┘
                                ▼
                     ┌──── Laya verifier (~28 ms) ────┐
                     │  grounded  → ship               │
                     │  declined  → hand off to human  │
                     │  neither   → one escalation     │
                     └─────────────────────────────────┘
```

## Measured on this machine

RTX 5060 Ti 16 GB (System 2) + RTX 4060 8 GB (System 1), vLLM 0.30.0 / torch 2.13.0+cu130,
Laya 0.3.22 / torch 2.14.0.

| | |
|---|---|
| System-1 gate, warm | **27.8 ms** median, all three questions in one pass |
| System-1 gate, first call | ~640 ms (CUDA kernel compile) |
| System-2 decode | **~44 tok/s** (39–45 across runs) |
| System-2 fast path | 1282 ms median for ~56 tokens |
| System-2 deliberate | 26.3 s for 1200 reasoning tokens |
| Model weights | 9.56 GiB resident (AWQ int4, vision dropped) |
| KV cache | 2.53 GiB = 67,691 tokens, 4.1x concurrency at 16k |
| Context | 32,768 tokens (native is 262,144) |

A request the gate resolves never touches the engine: **~28 ms vs ~1282 ms**, a 46x
difference, and zero GPU-seconds on the 16 GB card.

## Run it

Two services, two virtualenvs. They must be separate: vLLM pins `torch==2.13.0`
while Laya requires `torch 2.14`, and a single venv resolves that conflict by
breaking one of them.

```bash
uv venv --python 3.12 .venv-vllm && VIRTUAL_ENV=.venv-vllm uv pip install vllm==0.30.0
uv venv --python 3.12 .venv-laya && VIRTUAL_ENV=.venv-laya uv pip install -e ".[dev]"

./scripts/serve_vllm.sh       # System 2, cuda:0, port 8000
./scripts/serve_pipeline.sh   # System 1 gate + cascade, port 8100
```

```bash
curl localhost:8100/healthz
curl localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "system1",
  "messages": [{"role": "user", "content": "Label this ticket as billing/technical/other: duplicate charge"}]
}'
```

Every response carries a `laya` key with the full decision record — route, gate
answers, verdict, tokens and per-stage timings:

```json
"laya": {
  "route": "system1", "think": false, "escalated": false,
  "gate_model": "english", "gate_confidence": 0.140,
  "gate_answers": {"guardrail": 0.046, "classification": 0.776, "deliberation": 0.031},
  "verdict": null, "tokens": {"prompt": 0, "completion": 0},
  "timings_ms": {"gate": 27.4}
}
```

`enable_thinking` on the request overrides the gate's deliberation decision.

## Layout

```
config/pipeline.yaml   questions, thresholds, model and device config — all policy is data
system1/policy.py      routing rules. Pure functions, no torch/laya/vllm/network imports
system1/gate.py        Laya Router wrapper + GPU identity assertion
system1/llm.py         vLLM OpenAI-compatible client
system1/pipeline.py    gate -> route -> generate -> verify -> escalate
system1/server.py      FastAPI: /v1/chat/completions, /v1/systemone, /healthz
tests/                 21 routing tests, no GPU required
scripts/               launch + the measurement probes described below
```

`system1/policy.py` imports nothing heavy on purpose: the decision logic is testable
without loading 9.56 GiB of weights.

```bash
.venv-laya/bin/python -m pytest tests/ -q          # 21 passed
```

## Configuration is measured, not guessed

Every gate question in `config/pipeline.yaml` was chosen by running
`scripts/probe_questions2.py` against a labelled set, and every threshold that
moved has the number that moved it in a comment. The first design did not survive
contact:

- A 5-way `choice` for intent peaked at **0.29** probability. Flat, so
  `uncertain_below: 0.6` marked nearly every request uncertain.
- A 3-level `score` for complexity clustered in **0.45–1.29**, so the configured
  `think_threshold: 2.0` was **unreachable** — the escalation path could never
  turn on.

Both were replaced with binary `noul` questions, which is the primitive Laya was
trained on (RLCD, strictly proper scoring rules) and the one that actually
separates. Measured AUC: `needs_deliberation` **1.000**, `needs_derivation` 0.750,
`is_classification` clean at 0.78 versus ≤0.21 elsewhere.

Two things measurement forced that are worth knowing:

**Confidence is distance from the threshold, not the probability.**
`system1/policy.py::margin_confidence`. A `deliberation` answer of 0.12 against a
0.15 threshold is a near miss, not a confident "no" — the naive `max(p, 1-p)`
would score it 0.88. A probability sitting exactly on a threshold is the most
uncertain value there is, and routes to the cheap path.

**The verifier was crying wolf.** The obvious wording ("does this draft decline to
answer?") scored a *correct* answer at **0.767** and handed it off. Re-asked as
"does the reply contain no actual answer", and with `decline_confidence` raised to
0.8, `scripts/probe_verify.py` routes 5/5 correctly. Missing a refusal is
recoverable; a false handoff silently discards a good answer.

## Known limitations

- **The 4060 is not optional in practice.** Both models on the 5060 Ti was tried and
  measured: 14,308 MiB free, ~11,892 MiB for engine weights and CUDA context,
  ~2,000 MiB for the gate, leaving ~400 MiB of KV cache — about 4k tokens, one
  request. The gate benchmarks at the same ~28 ms on either card, so the split
  costs nothing measurable.
- **MTP speculative decoding is off.** The draft layer loads unquantized — a 1.9 GiB
  contiguous allocation on top of 10.1 GiB of AWQ weights — which consumed the
  entire KV cache and OOMed. Re-enable with `QWEN_SPEC_TOKENS=2` when the gate is
  stopped. This is the main throughput headroom left on the table.
- **The deliberation margin is thin.** `needs_deliberation` separates its classes by
  0.01 on a 10-case set (0.11 vs 0.12). AUC 1.000 on 10 points is not a robust
  estimate; `think_threshold: 0.15` is provisional until swept on real traffic.
- **Refusal detection is weak on the base checkpoint.** An explicit "I can't help"
  scores 0.238 where a hedged refusal scores 0.544. Fine-tuning is the documented
  fix (0.362 → 0.766 on typed-decisions).
- **No vision.** `--language-model-only` drops the 0.85 GiB encoder; the VRAM buys
  KV cache instead and the gate is text-only.
- **The eval set is 10 cases.** Enough to catch the design defects it did catch, not
  enough to call any threshold tuned. Phase 4 of `PLAN.md` is the real sweep.

## Reproducing the measurements

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python scripts/smoke_system1.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python scripts/probe_questions2.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python scripts/probe_verify.py
```

Stop `serve_pipeline.sh` first for the probes: a second Laya instance alongside
the service exhausts the 4060 and silently falls back to CPU.
