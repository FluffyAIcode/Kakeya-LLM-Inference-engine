# Kakeya Inference Engine Architecture Design

Status: Accepted (v1 core); Date: 2026-06-15. Governing decision: [ADR 0015](../adr/0015-kakeya-attention-and-engine-substrate.md).

## 1. Purpose

The Kakeya Inference Engine is a **product-grade LLM inference engine whose goal
is to replace vLLM**. Its native, first-class attention algorithm is **Kakeya
Attention** — sink+window bound + f_θ KV-projection + dLLM-proposer restoration,
as one primitive. The engine is designed **bounded-KV-native**: the full token
history is never resident; evicted context is reconstructed on demand. Every
subsystem (prefill, KV layout, admission, decode) is built around that invariant,
the way vLLM is built around full-KV PagedAttention.

The engine **replaces** vLLM rather than extending it. Of vLLM's three
prefill-engineering pieces:

- **Chunked prefill** → adopted, reinterpreted as **chunked restoration**.
- **FlashAttention** → wrapped and called as a kernel (table stakes).
- **Paged KV** → **not used** — paging manages a growing full KV that this
  engine never holds.

## 2. Core invariant

For a session of logical length `T`, the **resident KV is bounded** and
independent of `T` beyond the exact-layer term:

```
resident_KV(session) = Σ_exact_layers (full T) + Σ_other_layers (sink + window)
```

- **Exact layers** keep full-context KV (the recall-critical layers).
- **All other layers** keep only `sink + window` resident positions; positions
  outside that window are **evicted** and, when a query needs them, **restored on
  demand** by the restoration policy (§5).
- The full per-layer KV for evicted positions is **never materialized or stored**.

Memory is provisioned for the **peak resident window**, not the conversation
length. This is the structural source of the engine's concurrency advantage over
full-KV engines.

## 3. Subsystems

```
        prompt stream                          token stream
             │                                      ▲
             ▼                                      │
  ┌───────────────────────┐   bounded KV   ┌────────────────────┐
  │ Chunked Restoration   │──────────────▶ │ Bounded-KV Decode   │
  │ Prefill (§4)          │  + restore idx │ Engine (§6)         │
  └───────────────────────┘                └────────────────────┘
             ▲                                      ▲
             │            ┌──────────────────┐      │
             └────────────│ Peak-Window      │──────┘
            admit/reject  │ Admission (§7)   │ schedule cohort
                          └──────────────────┘
                                   │
                          ┌──────────────────┐
                          │ Restoration       │  (§5: native-hybrid | f_θ)
                          │ Policy            │
                          └──────────────────┘
```

## 4. Chunked Restoration Prefill

**Interface**

```
prefill(prompt_ids: int[T], policy: RestorationPolicy, chunk: int = 2048)
    -> BoundedKVState
```

**Behavior** — consume the prompt in fixed `chunk`-sized blocks. For each block:

1. Run the verifier forward over the block against the in-progress
   `BoundedKVState` (the resident KV so far).
2. Emit/update each layer's resident KV per the core invariant (§2): exact layers
   accumulate full; other layers retain only `sink + window`.
3. The restoration policy (§5) supplies the K/V for any evicted positions a block
   needs to attend to (for full-attention models); on hybrid models with native
   sliding the block simply does not attend beyond its window.

**Invariants**

- Per-block working memory is **O(N · chunk · cache_len)** for the mask/activation
  — never **O(N · T²)**. Memory is decoupled from total prompt length.
- The LM head is evaluated only where needed (last position for next-token), never
  as a full `[N, T, vocab]` tensor.
- No `[·, ·, T, T]` attention mask is ever materialized — attention is computed by
  the wrapped flash kernel (§6) with window/causal as kernel parameters.

## 5. Restoration Policy

A pluggable policy supplies the K/V for evicted positions, keeping the rest of the
engine model-agnostic:

| Policy | Model class | Mechanism | Restoration cost |
| --- | --- | --- | --- |
| **NativeHybridBounded** | hybrid-attention (e.g. Gemma-4: full + sliding layers) | exact full-attn layers carry recall; other layers are natively local → no reconstruction needed | none (free) |
| **FThetaRestored** | full-attention (e.g. Qwen/Llama) | dLLM proposer produces transient K/V over history; f_θ projects to verifier K/V at evicted positions | one proposer forward |

The policy decides **where recall comes from** and **whether reconstruction
runs**. The bounded-KV layout, prefill chunking, admission, and decode are
identical across policies.

## 6. Bounded-KV Decode Engine

**Interface**

```
decode(cohort: list[BoundedKVState], max_new_tokens) -> list[token[]]
```

**Behavior** — decode an admitted cohort in one batched step per token. Each
session is one batch row over its bounded KV. Attention is the **wrapped flash
kernel** with the Kakeya window / exact-layer-full as kernel parameters; the
decode step is **graph-capturable** (static shapes per cohort). The resident KV
grows only within the bound (sink+window for non-exact layers).

**Invariants**

- Decode-step memory is bounded by the cohort's resident KV (§2), not by total
  generated length beyond the exact-layer term.
- No full-KV cache and no paged store: the resident set is the only KV that exists.

## 7. Peak-Window Admission

**Interface**

```
admit(memory_budget_bytes, model_bytes, sessions) -> admitted_cohort
```

**Behavior** — a session's cost is its **bounded resident KV** (§2), computed from
the model's layer layout and the engine's `(sink, window)` — **not** its token
count. Max concurrency:

```
max_concurrent = (memory_budget - model_bytes) // resident_KV_per_session
```

Admission is by **peak window**, so concurrency does **not** degrade as
conversations lengthen — the defining difference from full-KV admission, where a
session's cost grows with its history and concurrency collapses at long context.

## 8. Where the engine wins vs vLLM

The advantage scales with the model's **full-attention fraction** (the exact-layer
term in §2):

- **Hybrid models** (Gemma-4, 25/30 natively sliding): vLLM already bounds the
  sliding layers and the 5 full-attention layers dominate KV in both engines →
  the engine is **competitive** (it removes the O(N·T²) prefill cost and provisions
  by peak window) but has no large structural KV edge.
- **Full-attention models** (Qwen/Llama, no native sliding): vLLM must keep **all**
  layers' full KV; the Kakeya engine keeps only `exact + sink + window` and
  restores the rest via f_θ+proposer → a **large** resident-KV edge → it admits
  many more concurrent long-context sessions. **This is the engine's target
  regime.**

## 9. v1 scope, status, and sequencing

- **v1 (implemented):** Chunked Restoration Prefill + Bounded-KV Decode +
  Peak-Window Admission, NativeHybridBounded policy (Gemma-4), flash kernel via
  SDPA. `inference_engine.engine.{admission,kakeya_engine}` (admission: 9 unit
  tests). **Measured @62k:** recall 1.0, chunked prefill lifts concurrency
  N=2→N=4 by removing the O(N·T²) prefill mask; admission model gives a 2.56 GB/
  session bound and a 34-session ceiling (`docs/reports/kakeya-engine-vs-vllm-h200.md`).
- **v1.1 (next):** realize the bounded-KV bound at runtime with the
  **sliding-window-evicting cache** (transformers' hybrid-aware evicting cache
  drops sliding-layer KV correctly, but `cache_implementation="static"` currently
  segfaults under CUDA-graph capture with this model + chunked prefill) →
  graph-captured + fused-MoE decode. Lifts concurrency from N=4 toward the 34
  ceiling.
- **v1.2 (the decisive win):** FThetaRestored policy fully wired for a
  **full-attention verifier** (Qwen/Llama) — the configuration where restoration
  is load-bearing and the bounded-KV edge over vLLM is ~6× (§8), not the marginal
  gemma-4 case.
