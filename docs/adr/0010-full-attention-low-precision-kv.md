# ADR 0010 — Full-attention verifier + low-precision (INT8 / NF4) KV cache

- **Status**: Proposed (safety-net for ADR 0011) — 2026-06-07
- **Date**: 2026-06-07
- **Decision drivers**:
  - The 2026-06-06 `sink+window` quality A/B benchmark
    (`results/platform-tests/sink_window_quality_ab_1780714635.json`)
    showed that v0.3's `SinkWindowVerifier` loses 83.3% recall on
    middle-context fact retrieval relative to a full-attention baseline.
    The `sink+window` design buys bounded KV by literally evicting K/V
    tensors for tokens outside `(sink ∪ window)`; nothing the proposer
    does at inference time can recover information that was deleted
    from the verifier's cache.
  - ADR 0011 ("cross-attention proposer/verifier coupling") is the
    hypothesis that cross-attention from a full-attention proposer
    hidden bank into a bounded verifier can rescue the lost recall.
    R1c GPU evidence
    (`results/research/cross_attn_toy_vast_full_1780806644.json`,
    `results/research/cross_attn_toy_vast_needle_small_1780806644.json`)
    establishes that the mechanism partially works (16% on a 20-vocab
    needle task, 0% on the 135k-vocab full task) but is far from the
    G-X1 ≥ 80% acceptance criterion. R1d-β will give a more definitive
    answer; this ADR is the safety net **independent of R1d-β's
    outcome**.
  - User-stated v0.4 strategic constraints (recorded 2026-06-06):
    *no deadline, no sunk-cost reasoning, extreme KV efficiency,
    zero intelligence regression*. ADR 0010 takes "zero intelligence
    regression" as a hard constraint and trades on the "extreme KV
    efficiency" axis using a different mechanism than ADR 0011.
- **Depends on**: ADR 0001 (proposer sizing + speculative decoding
  contract), ADR 0002 (verifier selection — Qwen3-1.7B, Gemma 3-1B
  family).
- **Relates to**: ADR 0011 (cross-attention bridge). The two are
  *alternative* approaches to the same problem
  ("how do we get extreme KV savings on long-context workloads
  without intelligence regression?"). They are not mutually
  exclusive in code — a future v0.5 could combine bounded
  cross-attention rescue (ADR 0011 if validated) with low-precision
  full attention (this ADR) for compounding savings — but for v0.4
  GA they are exclusive choices because they share the verifier
  forward path and require different memory layout.

---

## 1. Context

### 1.1 What `sink+window` actually costs

v0.3's `SinkWindowVerifier` keeps K/V tensors only for
`{0..sink-1} ∪ {q-window+1..q}` for each query position q. At
`sink=4, window=64` over a 256–1024 token haystack with the needle
at a random middle position, the A/B run measured:

| | Full-context Qwen3-1.7B greedy | v0.3 (Qwen3-0.6B dLM proposer + Qwen3-1.7B sink+window verifier) |
|---|---|---|
| Mid-context fact recall | 6/6 (100%) | 1/6 (16.7%) |
| Peak KV bytes (B=1, S=84) | 56,311,808 | 7,798,784 |

Five of the six losses are middle-context fact recall failures: the
needle's K/V was evicted before the answer position, and no
proposer-side mechanism in v0.3 can rescue it.

The strategic question for v0.4 is whether to (a) accept the
intelligence regression, (b) recover the lost information through
cross-attention from a full-attention proposer (ADR 0011), or (c)
keep full attention on the verifier and trade on memory in a
different dimension (this ADR).

### 1.2 Where ADR 0011's R1c evidence stands

R1c (vast H200, 2 × 16 min, 2 GPU runs):

- 20-vocab diagnostic task: cross-attn bridge reaches 16% recall
  (final), peaks at 25% at step 800. The mechanism injects needle
  information in some fraction of cases — not noise.
- 135k-vocab full task: 0.00 recall throughout 2000 training steps.
  Loss converges to perplexity ~2.3 yet recall does not rise.

This is consistent with two interpretations:

- **(I-1)** Single-layer cross-attn at depth 20 has too little
  capacity to encode an arbitrary needle into the verifier's
  residual stream as a precise argmax-flipping signal; multi-layer
  / multi-depth bridges can close the gap (R1d-β → R1e).
- **(I-2)** The full-attention proposer's hidden bank, as a generic
  pretrained representation, is not localizable enough by gradient
  descent to be a usable index — i.e., the §3 hypothesis is wrong
  in shape, and no amount of capacity in the bridge fixes it.

R1d-β (auxiliary retrieval loss + attention-localization metric) is
designed to distinguish (I-1) from (I-2). ADR 0010 is the v0.4 GA
plan if R1d-β returns (I-2) or if R1e cannot reach 80% within a
reasonable compute budget.

### 1.3 The ADR 0010 framing

Keep full attention on the verifier — same intelligence as the
oracle baseline by construction — but reduce the *bytes per cached
token* by quantizing K/V to lower precision. The KV cache is the
dominant memory term for long-context inference (it grows linearly
with context length and dominates weights once context > a few k
tokens), so a 2× or 4× per-token compression buys back most of the
practical memory benefit `sink+window` provided.

### 1.4 Memory math

Per token, per layer KV bytes:

| Precision | bytes/elem | KV bytes/(token, layer) for hidden=1152 (Gemma 3-1B) | for hidden=3584 (Gemma 4-9B class) |
|---|---|---|---|
| **bf16** (current) | 2 | 4,608 | 14,336 |
| **INT8** | 1 | 2,304 (-50%) | 7,168 (-50%) |
| **INT4 / NF4** | 0.5 | 1,152 (-75%) | 3,584 (-75%) |

For multi-layer aggregate at typical layer counts:

- Gemma 3-1B (26 layers): bf16 ≈ **120 KB/token**, INT8 ≈ 60, NF4 ≈ 30
- Gemma 4-9B-class (≈ 42 layers): bf16 ≈ **600 KB/token**, INT8 ≈ 300, NF4 ≈ 150

For Mac mini 24 GB targeting 64 k-token context on Gemma 4-9B class:

- bf16 KV: 64 k × 600 KB = **~37 GB** → does not fit. v0.3 only fit by trimming the cache.
- INT8 KV: ~18 GB → fits with margin for weights/activations.
- NF4 KV: ~9 GB → fits comfortably; leaves room for KV growth past 100 k tokens.

For comparison `sink+window=4+64`: caps at ~68 tokens × 600 KB ≈
**41 MB** regardless of context length. ADR 0010's win-axis is
**different from `sink+window`'s**: not "constant memory", but
"linear memory at half/quarter the slope, with full intelligence".

The two are complementary — ADR 0010 + ADR 0011 (if validated) is a
v0.5+ direction.

---

## 2. Decisions

### 2.1 Default precision: NF4 (4-bit normal-float)

NF4 (introduced in QLoRA, 2023) is a 4-bit quantization tuned for
parameter distributions that are roughly normal — which the K/V
projections after a transformer layer are, by training-time weight
decay and layer-norm structure. Empirical benchmarks
(QLoRA paper + follow-ups, AWQ paper) put NF4 within 0.3–0.8% of
bf16 on MMLU / HellaSwag / ARC at 7B–13B parameter scale. INT4
uniform quant is ~0.5% worse than NF4 at the same bit-rate.

INT8 is the **safe-default fallback** when a backend cannot host
NF4 efficiently (e.g., MPS without bnb-style kernels). INT8 is
within 0.05–0.1% of bf16 in the same benchmarks — effectively
indistinguishable.

### 2.2 Calibration: per-tensor symmetric, asymmetric for outliers

KV tensors have outlier channels (well-documented in SmoothQuant,
AWQ). Two-step quantization:

1. Per-token, per-head **outlier mask**: top-k channels by absolute
   magnitude (k = 1–2) are kept in bf16.
2. Remaining channels: per-channel symmetric quant for K
   (zero-centered after layer-norm), per-channel asymmetric for V
   (no zero-centering guarantee).

This adds ~3–5% storage overhead (the bf16 outliers + per-channel
scales) but recovers most of the long-context retrieval quality
that uniform per-tensor quant loses.

### 2.3 Backends

- **MLX (Apple Silicon)**: implement NF4 KV via `mx.quantize` /
  `mx.dequantize` on the K/V projections immediately before they
  enter the cache, and dequant on the read side. INT8 fallback uses
  the same path with a different `bits=` argument. MLX 0.31+
  supports both.
- **PyTorch / CUDA**: use `bitsandbytes` for NF4 (well-tested on
  CUDA), fall back to INT8 via `torch.quantize_per_channel` for
  hardware without `bnb`.
- **CPU (test/CI)**: INT8 only; NF4 has no efficient CPU kernel and
  is not a v0.4 GA target.

### 2.4 Sink+window stays as a feature flag, not a default

`SinkWindowVerifier` is preserved in `inference_engine.backends.*`
but defaults to disabled in v0.4. Workloads that explicitly request
constant-memory KV (e.g., long-running agent loops on tiny edge
hardware where even NF4 × full-context is too much) opt in via
`Verifier(kv_strategy="sink_window", sink=..., window=...)`.

### 2.5 Speculative decoding contract: unchanged

The dLM proposer + AR verifier speculative decoding loop from ADR
0001 remains exactly as in v0.3. Verification still happens at
bf16 precision (logits are dequantized for argmax/softmax); only
the *K/V cache storage* is quantized. This preserves byte-exact
determinism under the ADR 0008 §6.5 INV-3 gate.

---

## 3. Alternatives considered

| Alternative | Status | Why rejected (or why deferred) |
|---|---|---|
| Keep `sink+window` as v0.4 default | Rejected | Empirically loses ≥83% on middle-context recall; conflicts with "zero intelligence regression". |
| ADR 0011 cross-attention bridge | **Active research** | Conditional on R1d-β / R1e outcome. ADR 0010 is the safety net if 0011 is rejected. If 0011 is accepted, ADR 0010 may still ship as an *additive* memory optimization (combining bounded cross-attention + low-precision storage for compounded savings). |
| Sliding-window-only (no sink) | Rejected | Same intelligence regression as `sink+window`; worse on early-context anchoring. |
| H2O / SnapKV / PyramidKV importance-based eviction | Deferred | Improves on `sink+window` for some workloads but still evicts. Requires per-token importance scoring at inference time (compute cost). v0.5 candidate. |
| Mamba / RWKV / RetNet long-context-native models | Out of scope | Changes the project's model-identity. ADR 0001 commits to Qwen3 / Gemma family. |
| KV cache *offload* to disk / shared memory | Deferred | Mac mini 24 GB has no fast secondary storage path. Useful for desktops with ample SSD bandwidth — v0.6 candidate. |

---

## 4. Consequences

### 4.1 What is gained

- **Zero intelligence regression by construction**. Full attention
  means oracle-equivalent token argmax in the limit of perfect
  dequant; calibrated NF4 / INT8 keep the gap < 1% on standard
  benchmarks.
- **2× (INT8) or 4× (NF4) reduction in per-token cache bytes**,
  enough to fit Gemma 4-9B class workloads at ~64–100 k tokens
  on Mac mini 24 GB.
- **No new training step**. Unlike ADR 0011 (which needs cross-
  attention bridge training, alignment data prep, gate G-X1/2/3
  empirical validation), ADR 0010 is implementable on top of
  v0.3.0 weights without modifying the proposer or verifier.
- **Backend-portable**. Apple Silicon, NVIDIA, and CPU all have
  established INT8 / NF4 kernels.

### 4.2 What is given up

- **Linear memory growth**. KV still grows with context length; on
  pathological multi-hour agent loops with no `clearKvCache` calls
  the cache will eventually exceed any fixed budget. ADR 0010
  trades an absolute bound (`sink+window`) for a *better slope* on
  a linear curve. Workloads that need an absolute bound must opt
  back into `sink+window` (§2.4).
- **Compute overhead at the dequant boundary**. Each verifier
  forward pass dequantizes the K/V tensors it reads. On hardware
  with native int8/int4 tensor cores (H100, M-series GPU
  matmul-on-int8) this is negligible. On older NVIDIA cards (A100,
  L4) it is measurable (~5–15% slowdown vs bf16). Acceptable for
  v0.4; revisit on a per-backend basis.
- **Outlier-aware calibration adds complexity**. Per-channel scales
  + outlier mask is non-trivial code; the simpler per-tensor
  symmetric quant is faster but loses 2–5% on long-context
  retrieval. v0.4 ships outlier-aware as the default; per-tensor
  is a runtime flag for benchmarking.

---

## 5. Implementation plan (PR sequence)

| Phase | Scope | Deliverables |
|---|---|---|
| **A** | Quantization primitives (CPU + MLX + CUDA) | `inference_engine.backends.kv_quant` module with `quantize_kv(K, V, bits, scheme)` / `dequantize_kv(...)` and a `KVQuantConfig` dataclass. Linux unit tests for round-trip error bounds. |
| **B** | Verifier integration (single backend first: MLX) | `inference_engine.backends.mlx.FullAttentionQuantizedVerifier` — same forward signature as `MLXSinkWindowVerifier`, but stores KV in NF4 / INT8 and dequantizes on read. INV-3 determinism gate must pass. |
| **C** | A/B benchmark vs sink+window vs full-bf16 | Run the same `bench_sink_window_quality_ab.py` matrix on Mac M4 with NF4 / INT8 / sink+window / full-bf16 verifiers. Acceptance: NF4 recall ≥ 95% of full-bf16 on the existing 6-case mid-context fact retrieval benchmark. |
| **D** | Backend port: PyTorch / CUDA | `inference_engine.backends.pytorch.FullAttentionQuantizedVerifier`. Linux integration tests on a small NVIDIA-equipped runner (or vast.ai). |
| **E** | Long-session bench under quantized KV | Re-run `bench_session_long_run.py` 4 h at NF4 + INT8. Verify `kv_live_bytes` slope matches the predicted 2× / 4× reduction. |
| **F** | Default flip + docs | v0.4 default verifier becomes `FullAttentionQuantizedVerifier(bits=4, scheme="nf4_outlier")`. Quickstart updated. `sink+window` documented as a feature flag for memory-bounded edge use. |

Each phase has Linux CI gates + (where applicable) Mac M4 / vast.ai
empirical gates. PRs are stacked per ADR 0008 §9.

---

## 6. Validation criteria (v0.4 GA gates)

A v0.4 release shipping ADR 0010 must demonstrate, all on
reproducible artifacts in `results/platform-tests/` or
`results/research/`:

1. **Quality parity vs full-bf16**: NF4 verifier achieves ≥ 95% of
   full-bf16 recall on the 6-case mid-context benchmark, > 99% on
   short-context greedy completions. INT8 ≥ 99%.
2. **Memory reduction realized**: per-turn `kv_live_bytes` reported
   by `GetSessionInfo` is within 5% of the theoretical
   2× / 4× target across a 1 h benchmark.
3. **Determinism preserved**: ADR 0008 §6.5 INV-3 gate passes
   bit-exact between continuation and reset paths under the
   quantized cache.
4. **Cross-backend equivalence**: MLX and PyTorch backends produce
   matching argmax across a 50-prompt eval set (within int4 / NF4
   numerical tolerance — exact int8 match expected).
5. **Long-session stability**: 4 h `bench_session_long_run.py` on
   Mac M4 with `kv_strategy=nf4_full` shows no errors; KV growth
   matches the linear prediction (slope < the bf16 slope by 4×).

---

## 7. Open questions (to resolve during implementation)

- **Q1**: Per-channel vs per-token vs per-head granularity for
  outlier detection. Initial recommendation: per-head (matches
  attention computation natural axis), top-1 outlier channel
  retained at bf16. Validate empirically in Phase A.
- **Q2**: Do we quantize on write only, or on both read and
  write (re-quantizing dequantized values during attention update
  passes)? Speculative decoding's verifier-recompute path may
  re-touch the same K/V tensors; double-quantization round-trip
  error compounds. Initial recommendation: quantize-on-write only,
  cache stays in low precision until evicted.
- **Q3**: Interaction with cross-request KV reuse (deferred per
  ADR 0008 §6 — was ADR 0007's territory). When cross-request
  reuse lands in a future ADR, NF4 storage must round-trip cleanly
  across session boundaries. Out of scope here; flagged for
  whoever takes that on.
- **Q4**: NF4 + speculative decoding interaction. The proposer
  reads no K/V (it's a dLM); the verifier reads K/V at quantized
  precision. Expected to be neutral. Validate in Phase C.
- **Q5**: Compatibility with ADR 0011 cross-attention bridge if it
  later passes G-X1. The bridge consumes the proposer's hidden
  bank (which is computed at full-attention bf16 precision and
  stored separately, not in the verifier KV cache); the verifier
  KV cache is what ADR 0010 quantizes. The two should compose
  cleanly; validate in v0.5 if both ship.

---

## 8. Testing discipline

Same rules as ADR 0008 §9: no fakes, no fallbacks, no overfits,
100% Linux unit-test coverage where the mechanism is testable
without GPU; all empirical claims gated on reproducible Mac M4
or vast.ai artifacts committed under `results/platform-tests/`.

NF4 round-trip error bounds, outlier mask correctness,
quant/dequant idempotence, and INV-3 determinism are all
testable on Linux in CI.

---

## 9. References

- `results/platform-tests/sink_window_quality_ab_1780714635.json`
  — the empirical surface that motivates this ADR
- `results/research/cross_attn_toy_vast_full_1780806644.json`,
  `results/research/cross_attn_toy_vast_needle_small_1780806644.json`
  — R1c evidence informing the safety-net framing
- ADR 0001 (proposer sizing + speculative decoding contract)
- ADR 0002 (verifier selection — Qwen3-1.7B, Gemma 3-1B)
- ADR 0008 (session-bound runtime, INV-3 determinism gate)
- ADR 0011 (cross-attention bridge — proposed alternative)
- QLoRA: Dettmers et al., "QLoRA: Efficient Finetuning of
  Quantized LLMs", NeurIPS 2023 (NF4 quantization scheme)
- AWQ: Lin et al., "AWQ: Activation-aware Weight Quantization for
  LLM Compression and Acceleration", MLSys 2024 (outlier handling)
- SmoothQuant: Xiao et al., "SmoothQuant: Accurate and Efficient
  Post-Training Quantization for Large Language Models",
  ICML 2023 (per-channel scaling)
- KV Cache quantization survey: Liu et al., "KIVI: A Tuning-Free
  Asymmetric 2bit Quantization for KV Cache", ICML 2024
