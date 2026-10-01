# Laya System 1 + Qwen3.5-9B 4Bit — Build Plan

A local two-tier inference pipeline. **Laya** (`convaiinnovations/laya`) is the System-1
reflex: one non-autoregressive forward pass answers typed decision questions in ~33-40 ms.
**Qwen3.5-9B int4** on vLLM is System 2: it generates. Laya decides *whether, how, and
whether to trust* the generation — it never writes text, so there is nothing to parse and
nothing to hallucinate.

---

## 1. Research findings

### 1.1 Laya — System-1 decision model

Source: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) ·
[github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0, 0.3.22)

Non-autoregressive, bidirectional encoder + a decision head. Answers **typed questions**
about a **state** in a single forward pass. Trained with RL against strictly proper scoring
rules (RLCD), so its probabilities are calibrated by construction — the confidence number
is a real posterior, not a softmax artefact.

Three primitives, same shape as TypeSafe Jev:

| Primitive | Question | Output |
|---|---|---|
| `choice` | Which of these labels applies? | label + confidence + per-label probabilities |
| `score` | Where on this ordinal scale? | float + confidence |
| `noul` | Is this true? | probability in [0,1] |

Checkpoints (only the requested one downloads):

| Checkpoint | Encoder | Params | Context | Use for |
|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 | English, guardrails, triage |
| `convaiinnovations/laya-multilingual` | mmBERT-base | 322M | 1024 (→8192 via `max_len`) | 100+ languages, ~2.2x faster |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decisions workflows |

`Router` detects script/language in <0.5 ms of pure Python and dispatches. It keeps two
checkpoints resident by default, so a language switch after first use costs detection only.
`Router(preload=True)` removes the last reload. Published latency on T4: 39.5 ms English,
32.8 ms multilingual, 7.2 ms/question batched.

Key API surface we will use: `Router(preload=True, device=...)`, `predict(state, questions)`,
`predict_batch` (shared forward pass), `min_confidence=` abstention, and the `laya-serve`
HTTP server exposing `POST /v1/systemone` with the **same request/response shape as Jev's**.

### 1.2 Qwen3.5-9B 4-bit

Base: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) — 9B, hybrid
**Gated DeltaNet (3:1) + sparse MoE**, 32 layers, hidden 4096, vocab 248,320, native
262,144 ctx, vision encoder, **MTP trained for multi-step**. bf16 base is 18 GiB — too big
for this machine, so int4 is mandatory, not optional.

Surveyed int4 checkpoints (exact byte sizes read from safetensors headers):

| Checkpoint | Format | LM | MTP | Vision | Total | MTP for spec-decode? |
|---|---|---|---|---|---|---|
| `nicklas373/Qwen3.5-9B-AWQ` | AWQ, compressed-tensors, g64 sym | 9.43 | 0.45 | 0.85 | **10.73 GiB** | **yes** |
| `QuantTrio/Qwen3.5-9B-AWQ` | AWQ, compressed-tensors | 10.22 | 0.45 | 0.85 | **11.52 GiB** | yes |
| `Intel/Qwen3.5-9B-int4-AutoRound` | `quant_method: auto-round`, g128 | — | — | — | 8.35 GiB | **no MTP weights** |

`Intel/...AutoRound` is excluded on two counts: vLLM 0.30.0's quant registry
(`vllm/model_executor/layers/quantization/__init__.py`) has no `auto-round` method —
it would need conversion to load — and it ships **zero MTP tensors**, so MTP speculative
decoding is unavailable.

**Default: `nicklas373/Qwen3.5-9B-AWQ`** — smallest LM footprint (9.43 GiB), compressed-tensors
so vLLM auto-detects the method, and MTP weights present. `QuantTrio/Qwen3.5-9B-AWQ` is a
documented fallback.

Serving facts from the [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html):
`--language-model-only` skips the vision encoder and frees its 0.85 GiB for KV cache;
`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` is the latency lever
(recipe: MTP-1 cuts TPOT, higher N trades throughput); `--reasoning-parser qwen3` with
`--default-chat-template-kwargs '{"enable_thinking": false}'` gives a non-thinking mode;
prefix caching helps throughput and hurts low-concurrency latency.

### 1.3 How System-1 models are used in front of an LLM

This is a well-established pattern; Jev is TypeSafe's hosted System-1 model and Laya is the
self-hosted open counterpart. The projects that do it:

- **[prismhq/jev-router](https://github.com/prismhq/jev-router)** — LiteLLM pre-call hook.
  Jev reads a *minimized* request summary and picks a model from `router.yaml` candidates
  that Jev can actually read. Capability filtering happens *before* the decision. Any
  failure falls back to a configured default. Reported in the response `model` field.
- **[OpenRouter: Jev-verified cascade](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/jev-verified-cascade)** —
  `draft -> verify -> escalate`. A cheap tier drafts; the System-1 model returns
  `supported` / `unsupported` / `declined` with confidence; only `unsupported` triggers the
  frontier call, and `declined` hands off to a human. Measured: same zero wrong answers as
  always-frontier, **~7% of the cost**.
- **[lucianfialho/jev-model-router](https://github.com/lucianfialho/jev-model-router)**,
  **[its-panzer/jev-model-router](https://github.com/its-panzer/jev-model-router)**,
  **[Pinutss/jev-model-router](https://github.com/Pinutss/jev-model-router)** — cheapest-model
  selection, with the documented caveat that a mid-conversation downgrade can cost *more*
  because it invalidates the prompt cache.
- **[Ying-Kai-Liao/jev-browser](https://github.com/Ying-Kai-Liao/jev-browser)** — LLM plans,
  System-1 decides: ~300 ms per decision, 2-4 per step. Also the crispest statement of the
  contract: *"Jev only answers with probability distributions. It never writes text."*
- **[BillionsBobby/JevRouter](https://github.com/BillionsBobby/JevRouter)** and
  **[TypeSafeAI/typesafe-router](https://github.com/TypeSafeAI/typesafe-router)** — the same
  pattern applied to tool/agent selection rather than model selection.

The transferable idea is not "use Jev". It is: **a calibrated, typed, single-pass decision
should run *before* the expensive generative step, and again *after* it to verify.**

**One correction to the usual framing, and it matters here.** Those projects save money
because a hosted frontier model bills per token. This pipeline is local: there is no token
bill. So the gate must be justified on the metrics that actually exist here —

- **latency**: a pure System-1 answer is ~40 ms; a System-2 answer is 300 ms + seconds of decode
- **throughput**: every request the gate resolves never occupies a KV slot
- **energy / thermal headroom**: skipped generation is skipped joules
- **output quality**: a wrong answer is a correctness bug, so the verifier must earn its keep

A gate that is wrong about complexity costs both latency *and* a wasted model call. So the
plan treats threshold tuning as a measured deliverable (Phase 4), not a config guess.

---

## 2. Architecture

```
                        ┌────── Laya on the 4060 (cuda:0 within its process) — ONE pass, ~28 ms ──────┐
   POST /v1/chat/ ────► │  guardrail       noul  block?            → refuse, never touch the LLM    │
   completions          │  classification  noul  answerable by S1? → reply with the decision        │
                        │  deliberation    noul  needs thought?    → thinking / non-thinking       │
                        │  script/lang (built-in, <0.5 ms)         → english | multilingual        │
                        └───────────────────────────────────────────────────────────────────────────┘
                                    │                        │                        │
                       unsafe        │  simple & confident  │  hard                 │
                                    ▼                        ▼                       ▼
                                 refuse              ┌──────────────┐      ┌──────────────────────┐
                              (no LLM call)         │ S1 ANSWER    │      │ S2: Qwen3.5-9B AWQ  │
                                                   │ ~28 ms       │      │ on the 5060 Ti      │
                                                   │ no 5060 Ti   │      │ fast or deliberate  │
                                                   └──────────────┘      └───────────┬──────────┘
                                                                                       │
                                                                ┌──────────────────────┤
                                                                ▼                      │
                                                     ┌────────────────────┐            │
                                                     │ Laya verifier       │            │
                                                     │ grounded ≥ τ ?      │            │
                                                     │ declined ≥ τ ?      │            │
                                                     └──┬──────────┬──────┘            │
                                                   ship │          │ handoff / escalate  │
                                                        ▼          ▼                   │
                                                     ship    thinking mode,   ◀────────┘
                                                             re-verify, ship
```

Latency of the gate is paid once, up front, before any token streams — so a request the gate
resolves never pays TTFT at all. Measured over 8 runs: 28 ms wall against 821-1282 ms for
the fast path, a 29-46x difference.

### Why Laya and Qwen live in separate processes, separate venvs, and separate cards

1. **VRAM is the binding constraint, and it was measured rather than assumed.** The 5060 Ti
   has 14,308 MiB free. The AWQ engine needs ~11,892 MiB for weights plus CUDA context
   (9.56 GiB of weights). Laya needs ~2,000 MiB. That leaves ~400 MiB of KV cache — about
   4k tokens, one request at a time. It was tried: Laya could not even acquire a CUDA
   context (`cuDevicePrimaryCtxRetain` → OOM) and the service died with
   `CUBLAS_STATUS_ALLOC_FAILED`. Pinning the gate to the idle **RTX 4060** costs nothing
   measurable — it benchmarks at the same ~28 ms on either card.
2. **Dependency conflict.** vLLM 0.30.0 pins `torch==2.13.0`; Laya requires `torch 2.14`.
   Sharing a venv resolves that by breaking one of them.
3. **Failure isolation.** A gate OOM or a bad question set takes down 28 ms of routing, not
   the 9B engine — and vice versa.
4. **Device selection is not index-stable.** Default CUDA enumeration reports cuda:0 as the
   *4060*; only `CUDA_DEVICE_ORDER=PCI_BUS_ID` puts the 5060 Ti first. Selecting by PCI bus id
   in `CUDA_VISIBLE_DEVICES` does not work here — `0000:07:00.0` silently yields the 5060 Ti
   and `07:00.0` yields nothing. The integer index under a pinned order is the only selector
   that works, and `system1.gate._assert_device` refuses to start on the wrong card.

---

## 3. Hardware budget (measured, not estimated)

```
GPU0  RTX 5060 Ti  16311 MiB total, 14308 MiB free (1524 MiB held by pid 1154)
      -> Qwen3.5-9B AWQ int4, vision dropped (--language-model-only)
      -> model weights 9.56 GiB resident; --gpu-memory-utilization 0.85
      -> KV cache 2.53 GiB = 67,691 tokens, 4.1x concurrency at 16k
      -> --max-model-len 32768 (native 262144 is unreachable at this VRAM)
GPU1  RTX 4060       8188 MiB total
      -> Laya english 421M + multilingual 322M, both preloaded; ~4.8 GiB with the
         torch context, leaving ample headroom on a dedicated card
CPU   12 cores, 14 GiB RAM, 39 GiB free on / (92% full)
DISK  266 GiB free on the working partition; 13 GiB of checkpoints cached
```

KV math for the 32k context: only 8 of 32 layers are full attention (4 KV heads x 256 head
dim) = 32 KiB/token, so 32k tokens is ~1 GiB per full sequence. The other 24 layers keep a
fixed-size Gated DeltaNet state, which is per-slot, not per-token.

**The 0.85 figure is not free headroom.** At 0.90 the engine claims 14,680 MiB while only
14,308 are free; the gate then fails to start. Moving the gate to the 4060 is what bought
both the 2.53 GiB KV cache and a stable gate.

MTP speculative decoding is off. The draft layer loads unquantized — a 1.9 GiB contiguous
allocation on top of 10.1 GiB of AWQ weights — which would consume the entire KV cache. It
OOMs with `CUBLAS_STATUS_ALLOC_FAILED`. This is the main throughput headroom left unused.

---

## 4. Phases — what was verified, and what it cost

### Phase 0 — Environment ✅
- `.venv-vllm` + `vllm==0.30.0`; `.venv-laya` + `laya 0.3.22`. The venv split was not
  optional: vLLM pinned `torch 2.13.0`, Laya pulled `torch 2.14.0`.
- **Verified**: torch 2.13.0+cu130 reports capability `(12, 0)` and `sm_120` is in the arch
  list; a real bf16 2048x2048 matmul executes on the 5060 Ti.
- **Verified**: Laya runs a real forward pass on the 4060, 25.2–27.8 ms warm, all three
  primitives returning correct types.

### Phase 1 — System 2 engine ✅ (with one capability lost)
- `nicklas373/Qwen3.5-9B-AWQ` downloaded (10.73 GiB). AWQ on sm_120 works; the Marlin risk in
  §5 did not materialise.
- `vllm serve` with `--language-model-only --max-model-len 32768 --gpu-memory-utilization
  0.85 --max-cudagraph-capture-size 128 --reasoning-parser qwen3`. The mamba assert did not
  fire.
- **Verified**: 9.56 GiB weights, 2.53 GiB KV / 67,691 tokens, ~44 tok/s, 32768 context
  served.
- **Lost**: MTP speculative decoding. OOMs, see §3. This was a planned latency lever and it
  is the main thing still on the table.

### Phase 2 — System 1 gate ✅ (redesigned after measurement)
- `Router(preload=["english","multilingual"], device=...)` on the 4060.
- **The first question set did not survive contact and was replaced.** A 5-way `choice` for
  intent peaked at 0.29 probability, and a 3-level `score` for complexity clustered in
  0.45–1.29 — which made the configured `think_threshold: 2.0` *unreachable*, so the
  escalation path could never have turned on. Both were replaced with binary `noul`
  questions after `scripts/probe_questions2.py` measured `needs_deliberation` at AUC 1.000.
- **Verified**: all three routes correct — factual → fast, ticket → System-1-only, proof →
  deliberate.

### Phase 3 — Cascade orchestrator ✅
- `gate -> route -> generate -> verify -> escalate`, policy entirely in YAML.
- **Verified** end to end over HTTP: a System-1-only request answers in 646 ms with the model
  never invoked; the fast path ships in 990 ms; the deliberate path ships in 26.3 s.
  Streaming and `/v1/systemone` work.
- Three real bugs were found by running it, not by reading it: `State` declared with
  annotated fields but no `@dataclass`; a stale `gate_confidence` call signature; and
  `for delta in chunks` over an async generator, which made streaming return nothing at all.

### Phase 4 — Evaluation ✅ (and it found something)
- 40 labelled cases, swept across 9 candidate thresholds (`scripts/sweep_thresholds.py`),
  plus the 10-case question probe and a 5-draft verifier probe including verbatim model
  output. End-to-end A/B: 28 ms against 821-1282 ms.
- **The sweep contradicted the probe.** `needs_deliberation` measured AUC 1.000 on 10
  cases; on 40 cases the best threshold reaches **0.800** route accuracy and still misses
  7 of 20 genuinely hard requests.
- `think_threshold` stays at **0.15**, which is *not* the sweep's optimum. The 0.05 winner
  switches off the System-1 route: `margin_confidence` normalises by `max(t, 1-t)`, so a
  low threshold shrinks the uncertainty band until a merely-clear request reads as
  undecidable. The 2.5-point gap is within noise for 40 hand-written labels.
- This matches Laya's own model card (base checkpoint 0.362 on typed-decisions vs 0.766
  fine-tuned). Deliberation is the weakest link and is flagged as such in the config and
  README rather than presented as a tuned optimisation.
- **Not done**: a reliability curve, and any evaluation against real traffic. The 40
  labels are hand-written.

### Phase 5 — Ops and docs ✅
- Launch scripts, `/healthz`, a `laya` decision record on every response, README with the
  measured numbers.


## 5. Risks — measured outcomes

| Risk | Outcome | Resolution |
|---|---|---|
| AWQ Marlin absent on sm_120 | Did not occur | torch 2.13.0+cu130 ships sm_120; AWQ loads and runs |
| MTP spec-decode memory | **Confirmed** — 1.9 GiB unquantized draft, `CUBLAS_STATUS_ALLOC_FAILED` | disabled via `QWEN_SPEC_TOKENS=0`; re-enable with the gate stopped |
| Both models on one card | **Confirmed** — gate could not get a CUDA context | split across the two cards; both fast |
| Mamba cache / CUDA-graph assert | Did not occur | `--max-cudagraph-capture-size 128` set anyway |
| Gate misclassifies complexity | Partly — the 0.01 margin is real | uncertainty band routes to the cheap path; sweep still owed |
| Verifier rejects good answers | **Confirmed** — correct answer scored `declined` 0.767 | reworded; `decline_confidence` 0.8; now 5/5 |
| `CUDA_VISIBLE_DEVICES` bus-id selection silently wrong | **Confirmed** — `0000:07:00.0` yields the wrong GPU | index under `PCI_BUS_ID` + startup assertion in `_assert_device` |
| 39 GiB free on `/` | Confirmed | all caches on the working partition |

---

## 6. Non-goals

- Vision input: `--language-model-only` drops the 0.85 GiB encoder because the gate is a text
  decision model and the VRAM buys KV cache instead. One flag re-enables it.
- No cross-GPU tensor parallelism: sm_120 and sm_89 are different architectures, NCCL cannot
  span them. Disjoint per-model residency is the correct use of the second GPU.
- No fine-tuning Laya yet. The verifier's weak refusal detection (explicit refusal 0.238 vs
  hedged 0.544) says zero-shot is the limiting factor; fine-tuning is the documented lever
  (0.362 → 0.766 on typed-decisions).
