# fast-then-slow

> Skip the LLM when a 28 ms model can answer. Check its answer when it can't.

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

RTX 5060 Ti 16 GB (System 2) + RTX 4060 8 GB (System 1). vLLM 0.30.0 /
torch 2.13.0+cu130, Laya 0.3.22 / torch 2.14.0.

| | |
|---|---|
| System-1 gate, warm | **27.8 ms** median, all three questions in one pass |
| System-1 gate, first call | ~640 ms (CUDA kernel compile) |
| System-2 decode | **~44 tok/s** (39–45 across runs) |
| System-2 fast path | 821–1282 ms for 30–56 tokens |
| System-2 deliberate | 26.3 s for 1200 reasoning tokens |
| Model weights | 9.56 GiB resident (AWQ int4, vision dropped) |
| KV cache | 2.53 GiB = 67,691 tokens, 4.1x concurrency at 16k |
| Context | 32,768 tokens (native is 262,144) |

A request the gate resolves never touches the engine: **28 ms** median wall time
(27.5 ms gate, over 8 runs), against **821–1282 ms** for the System-2 fast path — a
**29–46x** difference, and zero tokens on the 16 GB card. The `system1` route is the
reason this pipeline exists, and the sweep section below is about keeping it alive.

## Run it

Two services, two virtualenvs. They must be separate: vLLM pins `torch==2.13.0`
while Laya requires `torch 2.14`, and a single venv resolves that conflict by
breaking one of them.

```bash
uv venv --python 3.12 .venv-vllm
VIRTUAL_ENV=.venv-vllm uv pip install vllm==0.30.0

uv venv --python 3.12 .venv-laya
VIRTUAL_ENV=.venv-laya uv pip install -e ".[dev]"

cp config/deployment.example.yaml config/deployment.yaml   # then edit for your machine

./scripts/serve_vllm.sh       # System 2, cuda:0, port 8000
./scripts/serve_pipeline.sh   # System 1 gate + cascade, port 8100
```

Configuration is two files. `config/pipeline.yaml` is portable and committed: which
question plays which role, at what threshold. `config/deployment.yaml` is
gitignored because its contents are this machine's — checkpoints, GPU indices, VRAM
budgets, ports. Only the first is the part a fine-tune is trained against.

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
`POST /v1/systemone` exposes Laya on its own, with the same request shape as
TypeSafe's endpoint, so the decision model stays usable without the engine.

## Layout

```
config/pipeline.yaml   portable policy: questions, role bindings, thresholds
config/deployment.yaml machine-specific: checkpoints, GPUs, VRAM, ports (gitignored)
system1/policy.py      routing rules. Pure functions, no torch/laya/vllm/network imports
system1/config.py      config load + validation (thresholds must be probabilities)
system1/gate.py        Laya Router wrapper + GPU identity assertion
system1/llm.py         vLLM OpenAI-compatible client
system1/pipeline.py    gate -> route -> generate -> verify -> escalate
system1/server.py      FastAPI: /v1/chat/completions, /v1/systemone, /healthz
tests/                 36 routing and config tests, no GPU required
scripts/               launch scripts and the measurement probes described below
```

## Retargeting it at another project

The decision surface is config, not code. `config/pipeline.yaml` binds each question
to a **role** with a threshold:

```yaml
decisions:
  - role: block     # stop before the model is called
    question: guardrail
    threshold: 0.5
  - role: think     # deliberate mode on System 2
    question: deliberation
    threshold: 0.15
```

Roles are `block`, `system1`, `think` (gate) and `grounded`, `declined` (verifier).
**Every role is optional** — delete the `block` row and nothing blocks, which is what
a domain without a guardrail wants. Question ids are yours to rename; `policy.py`
never names one.

To adapt it:

1. Point `deployment.yaml` at your fine-tuned checkpoint — one line.
2. Rewrite the question wording and criteria in `pipeline.yaml`.
3. Re-bind roles and thresholds.
4. Re-run `scripts/sweep_thresholds.py` against your own labelled cases and treat the
   configured threshold as provisional until the sweep agrees.

Only adding a genuinely new *behaviour* needs code. Adding a question, renaming one,
or dropping a role does not.

`system1/policy.py` imports nothing heavy on purpose: the decision logic is testable
without loading 9.56 GiB of weights.

```bash
.venv-laya/bin/python -m pytest tests/ -q          # 36 passed
```

## Configuration is measured, not guessed

Every gate question in `config/pipeline.yaml` was chosen by running a probe against
a labelled set, and every threshold that moved has the number that moved it in a
comment. The first design did not survive contact:

- A 5-way `choice` for intent peaked at **0.29** probability. Flat, so
  `uncertain_below: 0.6` marked nearly every request uncertain.
- A 3-level `score` for complexity clustered in **0.45–1.29**, so the configured
  `think_threshold: 2.0` was **unreachable** — the escalation path could never turn
  on at all.

Both were replaced with binary `noul` questions, the primitive Laya was trained on
(RLCD, strictly proper scoring rules). On the 10-case probe, `needs_deliberation`
measured AUC 1.000, `needs_derivation` 0.750, and `is_classification` came out clean
at 0.78 against ≤0.21 elsewhere. **The 1.000 did not survive a larger set** — see
the sweep below.

Two things measurement forced that are worth knowing:

**Confidence is distance from the threshold, not the probability.**
`system1/policy.py::margin_confidence`. A `deliberation` answer of 0.12 against a
0.15 threshold is a near miss, not a confident "no" — the naive `max(p, 1-p)` would
score it 0.88. A probability sitting exactly on a threshold is the most uncertain
value there is, and routes to the cheap path.

**The verifier was crying wolf.** The obvious wording ("does this draft decline to
answer?") scored a *correct* answer at **0.767** and handed it off. Re-asked as
"does the reply contain no actual answer", and with `decline_confidence` raised to
0.8, `scripts/probe_verify.py` routes 5/5 correctly. Missing a refusal is
recoverable; a false handoff silently discards a good answer.

## The threshold sweep, and what it cost

`scripts/sweep_thresholds.py` runs the real gate over 40 labelled cases and reports,
for each candidate threshold, how often the route would have been right.

| `think_threshold` | route accuracy | false "needs thinking" | missed hard requests |
|---|---|---|---|
| 0.05 | **0.800** | 1 | 7 |
| **0.15** (configured) | 0.775 | 0 | 9 |
| 0.30 | 0.600 | 0 | 16 |

Two results, both negative, and the second is the more useful one.

**The probe's AUC 1.000 was a small-sample artifact.** On 40 cases even the best
threshold reaches 0.800 and still misses 7–9 of 20 genuinely hard requests. This
matches Laya's own model card: the base English checkpoint scores **0.362** on
typed-decisions against **0.766** fine-tuned. The guardrail and classification
questions are sharp; deliberation is the weak link, and the docs say so rather than
presenting it as a tuned optimisation.

**The best-scoring threshold is the wrong one.** `margin_confidence` normalises by
`max(t, 1-t)`, so lowering `think_threshold` shrinks the uncertainty band around it.
A classification request scoring 0.000 deliberation then reads as *uncertain* instead
of confidently "not deliberate", and the System-1-only route switches off — the
latency win the pipeline exists for. A request that is merely clear now looks
undecidable, which is the worst possible failure for a gate.

Two things establish this, and neither is the sweep's aggregate: the sweep's
`s1 hits` column is too noisy to isolate the effect, because whether a case reaches
`system1` depends on the classification question crossing 0.5 as well, and that
varies by phrasing. Instead it is pinned by a unit test
(`test_lowering_the_think_threshold_can_disable_the_system1_route`) and by the live
case: at 0.05 a ticket classification is uncertain and routes to the model; at 0.15
it scores 0.776 on classification and answers in ~28 ms with no engine call.

So `think_threshold` stays at **0.15**, which is *not* the sweep's optimum — the
2.5-point accuracy gap is within noise for 40 hand-written labels, and it buys back
the route that carries the pipeline. If you change it, re-check that a
classification request still takes the `system1` route; `scripts/smoke_system1.py`
asserts exactly that.

## Known limitations

- **The 4060 is not optional in practice.** Both models on the 5060 Ti was tried and
  measured: 14,308 MiB free, ~11,892 MiB for engine weights and CUDA context, ~2,000
  MiB for the gate, leaving ~400 MiB of KV cache — about 4k tokens, one request. The
  gate fails at startup with `cuDevicePrimaryCtxRetain` OOM. The gate benchmarks at
  the same ~28 ms on either card, so the split costs nothing measurable.
- **MTP speculative decoding is off.** The draft layer loads unquantized — a 1.9 GiB
  contiguous allocation on top of 10.1 GiB of AWQ weights — which consumed the entire
  KV cache and OOMed. Re-enable with `QWEN_SPEC_TOKENS=2` when the gate is stopped.
  This is the main throughput headroom left on the table.
- **The deliberation signal is weak**, as the sweep above is the evidence for. Do not
  read the deliberate path as a tuning knob until the checkpoint is fine-tuned.
- **Refusal detection is weak on the base checkpoint.** An explicit "I can't help"
  scores 0.238 where a hedged refusal scores 0.544. Fine-tuning is the documented
  fix (0.362 → 0.766 on typed-decisions).
- **No vision.** `--language-model-only` drops the 0.85 GiB encoder; the VRAM buys KV
  cache instead and the gate is text-only.
- **40 labelled cases, all written by hand.** Enough to catch three design defects and
  to set a threshold defensibly; not enough to call it tuned against real traffic.
- **Deliberate mode can return nothing.** If the model spends the whole budget on
  reasoning, `finish_reason` is `length` and the content is empty. The pipeline
  substitutes an explanatory message rather than returning an empty string, but the
  caller still needs a large `max_tokens` on hard questions. Note that vLLM 0.30's
  `--reasoning-parser qwen3` reports `reasoning_tokens` in usage yet leaves
  `reasoning_content` empty, so there is no reasoning text to show.

## Licensing

This project is Apache-2.0 (see `LICENSE`). The components it depends on:

| Component | License |
|---|---|
| `Qwen/Qwen3.5-9B` | Apache-2.0 |
| `convaiinnovations/laya` | Apache-2.0 |
| vLLM | Apache-2.0 |
| `nicklas373/Qwen3.5-9B-AWQ` | third-party quantization — check its own terms |

Apache-2.0 permits modification, fine-tuning, commercial use and redistribution, so
fine-tuning Laya on your own dataset and shipping the result is fine. The conditions
that survive are attribution and the licence/notice files: keep the upstream
attribution for Qwen and Laya, and if you redistribute a fine-tuned checkpoint,
carry the licence and any NOTICE file with it. Weights are not committed to this
repository, so nothing here needs that treatment yet.

## Reproducing the measurements

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python scripts/smoke_system1.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python scripts/probe_questions2.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python scripts/probe_verify.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python scripts/sweep_thresholds.py
```

Stop `serve_pipeline.sh` first for the probes: a second Laya instance alongside the
service exhausts the 4060 and silently falls back to CPU, which quietly invalidates
the numbers.
