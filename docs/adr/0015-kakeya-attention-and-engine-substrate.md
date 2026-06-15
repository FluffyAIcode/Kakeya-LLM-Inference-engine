# ADR 0015 — Kakeya Inference Engine: a product-grade vLLM replacement, Kakeya Attention native

- Status: Accepted (north star + algorithm definition); engine = in design
- Date: 2026-06-15 (rev. 2026-06-15)
- Supersedes/extends: the "Kakeya Attention vs PagedAttention/RadixAttention"
  framing in README; ADR 0014 §3.4.

## North star (highest goal — this governs everything below)

The **Kakeya Inference Engine is a product-grade inference engine whose goal is
to replace vLLM.** It is **not** a research script, **not** a technique bolted
onto HuggingFace transformers, and **not** "vLLM with a different cache". Its
**native, first-class attention algorithm is Kakeya Attention** — sink+window
bound + f_θ KV-projection + dLLM-proposer restoration, as **one primitive**. The
whole engine (prefill, KV management, admission/scheduling, kernels) is designed
**around bounded-KV + on-demand restoration as the default invariant**, exactly
the way vLLM is designed around full-KV PagedAttention.

Everything else in this ADR serves that objective. Explicitly:

- The product target is measured against vLLM by the **engine**, never by the
  research bench. The bench (`k3_cuda_multitenant_parallel_bench.py`, eager HF
  transformers) is **only a correctness/feasibility probe** and is not a thing we
  ship or benchmark as "Kakeya".
- "Parity → win" with vLLM is an engine deliverable, not a roadmap of vLLM
  features to copy.

## Why "borrow vLLM's pipeline" is the wrong plan (rejected)

vLLM's architecture is **full-KV-centric**: paged blocks store the *whole*
history; chunked prefill and flash masking are optimizations for *processing and
storing the full KV*. Porting Kakeya's bounded cache onto that pipeline inherits
a full-KV-shaped engine and caps the advantage at whatever the cache layout
saves — i.e. it makes Kakeya a vLLM feature, not a replacement.

A product Kakeya engine instead makes **bounded-KV the native invariant**:

- the **full history is never resident**; evicted context is reconstructed by
  the proposer on demand (Kakeya Attention);
- **prefill** produces the bounded resident set **+** the restoration path in one
  pass — it must never materialize the O(N·T²) attention mask or full-vocab
  logits that sink the research bench;
- **admission/scheduling** sizes sessions by their **peak window**, not their
  total token count — this is the structural source of the concurrency win;
- graph-captured decode, fused-MoE, efficient masking are **table stakes** any
  product engine needs, implemented *in service of* the Kakeya-native design —
  not as a port of vLLM's full-KV pipeline.

### vLLM's three prefill techniques, through the Kakeya lens (adopt / wrap / drop)

vLLM gets its 62k concurrency from three prefill-engineering pieces. They are
**not** copied wholesale; each is reinterpreted against the bounded-KV-native
design — two are adopted (one reinterpreted, one wrapped), the third is
structurally **unnecessary**:

| vLLM technique | what it solves for vLLM | Kakeya engine stance |
| --- | --- | --- |
| **Chunked prefill** | process a long prompt in fixed token blocks so mask/activation memory is O(N·chunk·d), not O(N·T²) | **Adopt — it *is* our chunked restoration.** Restoration is inherently incremental: consume the prompt in fixed blocks, emitting the bounded resident set + restoration path per block. Chunking is native to how restoration works, not a bolt-on. |
| **FlashAttention** (native causal + sliding-window kernel) | compute attention without materializing a `[.,.,T,T]` mask/score tensor | **Wrap and use directly.** Window is a kernel parameter; we call the flash kernel over the Kakeya window. It is a table-stakes kernel, not an architecture — no reason to reinvent it. |
| **Paged KV** | store the **whole growing KV** in non-contiguous pages so a large, ever-growing cache fits and shares | **Not needed — structurally.** Paging is a solution to the problem of *storing a growing full KV*. Kakeya is **on-demand KV restoration**: the resident KV is bounded (sink+window + exact layers) and the full history is never stored, so there is no growing full-KV to page. The problem paging solves **does not exist** in a Kakeya-native engine. |

So the engineering Kakeya needs is **chunked restoration + a wrapped flash kernel
+ native bounded-KV management** — *not* PagedAttention. This is the concrete
sense in which the engine **replaces** vLLM's design rather than extending it.

## Kakeya Attention — the native algorithm

**Kakeya Attention** = sink+window bound + f_θ KV-projection + dLLM-proposer
restoration, taken as one primitive. Peer of, and replacement for, the attention
layer in current engines:

| Algorithm | Replaces | Keeps full KV? |
| --- | --- | --- |
| eager / **FlashAttention** | the **compute** layer | yes |
| vLLM **PagedAttention** / SGLang **RadixAttention** | the **storage** layer | yes |
| **Kakeya Attention** | **compute + storage** | **no — bounded; evicted KV reconstructed on demand** |

FlashAttention makes attention compute cheaper; Paged/Radix make the same *total*
KV cheaper to allocate/share. **Kakeya Attention makes the total itself bounded.**

## Where the win is real (model architecture matters)

Resident KV is dominated by the **full-attention** layers (they hold full context
in any engine). So the engine's advantage over vLLM scales with the model's
full-attention fraction:

- **gemma-4-26B-A4B** is 25/30 **natively sliding** → vLLM already bounds those
  layers, and gemma-4 keeps recall 1.0 at `sliding_window=68` *with no
  restoration at all* → **no Kakeya moat on gemma-4** (probed; see below). It is
  the wrong showcase model.
- **Full-attention models** (Qwen/Llama, no native sliding): shrinking the window
  without restoration **destroys recall**, so f_θ+proposer restoration is the
  *only* way to bound memory at full recall — and vLLM, having no restoration,
  **must keep full KV and cannot match it**. This is the engine's target regime.

## Feasibility probes so far (informed the design — NOT the product)

These ran on the eager-transformers research bench; they validate correctness and
locate the engine's required invariants. They are not the product engine.

- **Restoration prefill, memory-efficient (SDPA + chunked logits + bf16 K/V)** —
  unblocked long-context execution at recall 1.0 (16k N=1-only→N=4, 32k OOM→N=2,
  62k OOM→N=1). Confirms the engine must avoid O(N·T²) masks / full-vocab logits.
- **gemma-4 bounded decode** — recall 1.0 at `sliding_window=68` natively (no
  restoration); native bounded decode still fits only N=2 @62k vs vLLM's 15.5
  because the bench does non-chunked prefill. Confirms (a) gemma-4 is the wrong
  showcase, (b) the engine — not bench retrofits — is what must beat vLLM.

(`docs/reports/kakeya-vs-vllm-longcontext-h200.md`.)

## Consequences

- The repo's highest engineering goal is the **product Kakeya Inference Engine
  replacing vLLM**, with Kakeya Attention as its native algorithm; the eager
  research bench is a probe only and is never reported as "Kakeya performance".
- The engine is designed bounded-KV-native (admission by peak window; restoration
  fused into prefill/decode), not as a port of vLLM's full-KV pipeline.
- The vLLM-beating demonstration is to be run on a **full-attention** verifier,
  where restoration is load-bearing.

## Milestone tracking (task encoding)

Engine work is coded **KIE-v1.x** (Kakeya Inference Engine), governed by this ADR
§9 of `docs/design/kakeya-inference-engine-architecture.md`. One milestone = one
PR, so development context stays per-task:

| Code | Milestone | Status | PR |
| --- | --- | --- | --- |
| **KIE-v1** | engine core: chunked restoration prefill + bounded-KV decode + peak-window admission (NativeHybridBounded) | done (core); concurrency gated on v1.1 | #135 |
| **KIE-v1.1** | realize the bounded-KV bound at runtime: sliding-window-**evicting** cache without the CUDA-graph segfault (evicting cache, graph capture off) + push concurrency toward the ceiling | **done** — gemma-4 62k concurrency **N=4→N=24** (recall 1.0; chunk-size tuning), **1.55× vLLM's 15.5**. Decoupled prefill/decode implemented (correct) but fragmentation-limited. | #136 |
| **KIE-v1.1.x** | exact-layer KV quantization toward the N=34+ ceiling | **partial** — int8/int4 exact-layer quant **de-risked recall-safe** (recall 1.0 @62k); genuine int8 storage **implemented + correct** (halves stored bytes). BUT **N=34 still OOMs**: the dequant-on-read returns full bf16 per exact layer (transients coexist), so peak doesn't drop. `v04.kv_compressor` doesn't help (round-trips, no RAM cut); `QuantizedCache` not hybrid-aware. | #137 |
| **KIE-v1.1.y** | **quantized attention** (tiled/dequant-in-kernel SDPA over int8 K/V — no full bf16 materialization) to convert the int8 storage into concurrency (N→69 at int8); + graph-captured decode | planned | — |
| **KIE-v1.2** | FThetaRestored policy on a full-attention verifier (Qwen/Llama) — the decisive vLLM win | planned | — |

## Evidence

- `docs/reports/kakeya-vs-vllm-multitenant-h200.md` (ctx-1238, same H200)
- `docs/reports/kakeya-vs-vllm-longcontext-h200.md` (16k–62k probes + KV model)
- `results/research/{k3_cuda_multitenant_parallel,vllm_multitenant_parallel,gemma_bounded_decode}_h200nvl_*.{json,log}`
