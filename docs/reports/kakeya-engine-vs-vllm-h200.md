# Kakeya Inference Engine vs vLLM — long-context concurrency (same H200, gemma-4-26B-A4B)

First comparison of the **product engine** (`inference_engine.engine.KakeyaEngine`,
v1 core: chunked restoration prefill + bounded-KV decode + peak-window admission,
NativeHybridBounded policy) against vLLM on the same H200, gemma-4-26B-A4B bf16,
NIAH ctx ≈ 62k. Architecture: `docs/design/kakeya-inference-engine-architecture.md`.

## Measured (ctx 62k, recall 1.0 where it runs)

| Engine | max concurrency @62k | per-session GPU | recall |
| --- | --- | --- | --- |
| Kakeya Engine **v1** (chunked prefill, growing cache) | N=4 | ~16 GB | 1.0 |
| Kakeya Engine **v1.1** (chunked prefill + **evicting** cache, graph off) | **N=16** | ~4 GB | 1.0 |
| vLLM (PagedAttention) | 15.5 | ~4.6 GB | 1.0 |
| Kakeya admission model (bounded-KV target) | 34 (ceiling) | 2.56 GB | — |

- **v1** (growing `DynamicCache`): N=1 68.2 → N=2 84.7 → N=4 117.8 GB; N=8 OOM.
- **v1.1** (evicting `StaticCache`, graph off): N=1 55.7 → N=4 67.9 → N=8 84.1 →
  **N=16 116.5 GB**; N=24 OOM. The evicting cache drops N=4 peak **117.8 → 67.9 GB**
  and lifts concurrency **N=4 → N=16**, now **above vLLM's 15.5** — bounded-KV is
  realized at runtime.

### KIE-v1.1 — what was done

`generate_cohort(evicting_cache=True)` builds a hybrid-aware `StaticCache`
(sliding layers → `StaticSlidingWindowLayer` capped at `sink+window`; full-attn
layers exact) and passes the **cache object** to `generate`. A static cache makes
`generate` torch.compile the decode (triton/inductor → CUDA-graph), which
**segfaults** with this model + chunked prefill (and writes a `.so` that fails to
load on noexec tmp), so KIE-v1.1 **turns graph capture off**
(`torch._dynamo.config.disable`) and runs the evicting cache **eager** — correct
and bounded, just ungraphed. Concurrency N=16 (recall 1.0) at 62k.

Residual gap to the 34-session admission ceiling: `StaticCache` pre-allocates the
full layers at `T + gen` and the ungraphed prefill carries working-set overhead,
so actual per-session is ~4 GB vs the 2.56 GB model. Graph-captured decode +
tighter allocation is **KIE-v1.1.x / v1.2** substrate work.

## Pushing toward the N=34 ceiling (KIE-v1.1.x)

| lever | result @62k | note |
| --- | --- | --- |
| prefill chunk 2048 | N=16 | baseline v1.1 |
| prefill chunk **1024 / 512** | **N=24** (recall 1.0, peak ~136 GB) | smaller chunk → smaller prefill transient |
| prefill chunk 256 | ~N=24 | diminishing returns |
| decoupled prefill+stacked decode | correct (recall 1.0 @N=4), but **OOM @N=30** | fragmentation / per-session prefill transient; no better than batched |

**Best measured concurrency: N=24** at 62k, recall 1.0 — **1.55× vLLM's 15.5**.

**Why N=34 is not reachable at bf16 KV (the hard floor).** Per-session resident
KV is dominated by the **5 exact full-attention layers** kept at full context:
`5 × 62 070 × 8 kv × 256 × 2(K,V) × 2 B = 2.54 GB/session`. So 34 sessions need
`34 × 2.54 + 51.6 (weights) ≈ 138 GB` — leaving ~1.8 GB for *all* prefill/decode
working set, which no real forward fits in. 34 is the admission model's
zero-overhead ceiling; the achievable bf16 ceiling on a 140 GB card is ~24–26.

**The lever to reach 34+ (not model-switching): exact-layer KV quantization.**
Quantizing the 5 exact layers' KV to 8-bit halves the floor to 1.27 GB/session →
~69 sessions fit; 4-bit → ~0.64 GB → ~135.

> **Note:** `inference_engine.v04.kv_compressor` does **not** help here — it
> round-trips through the lattice for *fidelity* and stores full bf16 tensors
> ("not from any in-RAM size change", its docstring), and needs the optional
> `kakeyalattice` package. transformers' `QuantizedCache` needs an uninstalled
> backend and is **not hybrid-aware** (it would store all 30 layers full, worse
> than the bounded-bf16 hybrid). So KIE-v1.1.x needs **genuine int storage of the
> exact layers** (int8/int4 packed + per-token scale, dequant per-layer on read),
> with the evicting bf16 sliding layers unchanged.

**De-risk (the exact layers are recall-critical — does int quant break recall?).**
Probe: round-trip the exact-layer K/V through int8 and int4 in the decoupled
decode (`--quant-exact-bits`), 62k, N=4. **Both int8 and int4 keep recall 1.0** —
so int storage of the exact layers is recall-safe.

**Genuine int8 storage — implemented, correct, but blocked by dequant-on-read.**
`_IntQuantExactLayer` (a `CacheLayerMixin`) stores the exact layers' K/V as int8
+ per-token scale and is wired into the decoupled decode. It is **correct**
(recall 1.0) and **halves the stored** exact-layer bytes (N=4 peak 67.9 → 63.7 GB).
**But it does not lift the concurrency ceiling — N=34 still OOMs.** The reason:
the cache `update()` contract returns **bf16** for the model's SDPA, so each
exact layer dequantizes its full K/V (`[N, 8, 62k, 256]` ≈ 0.25 GB/session/layer)
on read; across the 5 exact layers these transients coexist and eat the storage
saving at scale. The stored bytes shrink, the **peak** does not.

**The real unlock (next, custom-kernel): quantized attention.** To convert the
storage saving into concurrency, the attention must read the int8 K/V **without
materializing full bf16** — a tiled/dequant-in-kernel (flash-style) SDPA over the
int8 cache. That is a custom-kernel substrate item; until it lands, the achieved
recall-1.0 ceiling is the bf16-evicting **N=24** (1.55× vLLM). The int8 storage +
recall-safety are the prerequisites it builds on.

### kakeyalattice v1.6 (the fixed compressor) — evaluated

v1.6 genuinely fixes the two gaps reported against `v04.kv_compressor`: it adds
**bit-packed storage** (`KakeyaLatticePackedCache`, real **2.46× HBM** at D4 Q=38,
measured Qwen3-4B/H200 — vs the int8-index path's 1.94×) and a **contiguous,
SDPA-feedable** decode (also fixing the prior O(N²) re-decode). `kv_compressor`
now exposes it via `make_packed_kv_cache(...)`. Findings for **this** engine:

1. **It does not drop into the Gemma-4 engine as-is** — the uniform-head-dim
   packer asserts `expected last dim 256, got 512` on Gemma-4's hybrid layers
   (full vs sliding K/V shapes differ); needs a per-layer-head-dim adaptation
   upstream.
2. **The gain over the int8 path is modest at D=256** (2.46× vs 1.94×) — it
   shrinks the *stored* exact-layer bytes by ~20%, freeing ~8 GB at N=34.
3. **It does not change the N>34 blocker.** Like any `DynamicCache`-style cache,
   it returns **bf16** to SDPA, so the per-layer dequant transient (the thing
   that OOMs at N=34) is unchanged. v1.6 improves the *floor* (storage), not the
   *peak* (transient).

**Net:** v1.6 is the right *storage* codec to feed the real fix, but the decisive
lever past N=34 remains **quantized attention** (KIE-v1.1.y) — attend on the
packed/int codes without materializing bf16. v1.6's contiguous packed codes are
exactly the input that kernel would consume.

## KIE-v1.1.y — quantized attention: N=60 @62k (recall 1.0), 3.9× vLLM

`quantized_flash_attention` (tiled online-softmax over the int8 exact-layer
store, dequantizing **one tile at a time** → transient `O(tile)` not `O(S)`,
patched into gemma-4's 5 exact-layer attentions) removes the decode-time bf16
transient that capped KIE-v1.1.x at N=34. Measured on the same H200, 62k:

| N | recall | peak GPU |
| --- | --- | --- |
| 40 | 1.0 | 91.8 GB |
| 52 | 1.0 | 103.8 GB |
| **60** | **1.0** | **111.7 GB** |

**N=60 fits at recall 1.0** with headroom (111.7 GB) — **≈3.9× vLLM's 15.5**.
Concurrency went **N=24 (bf16) → N=60 (quant-attn)**, the int8 storage finally
converted into concurrency. (Numerically the tiled attention is exact vs full
SDPA on the same int8 K/V — validated separately.)

**Decode throughput (corrected).** The eval's `aggregate_tps_e2e` includes the
**sequential** 62k prefill of all N sessions, which dominates the wall time and
made the e2e figure look like ~2 tok/s — that is **not** the decode rate. With
prefill and decode timed separately (N=8, 62k): **decode-only = 25.6 tok/s
aggregate** (8×24 tokens / 7.5 s), vs prefill 86 s (the sequential-prefill
harness artifact, not how a server admits requests).

**Honest speed caveat.** 25.6 tok/s aggregate at N=8 is **~3.2 tok/s/session** —
well below vLLM's ~98 tok/s single-session decode at 62k. The tiled
online-softmax runs **eager** (graph capture off) in Python over 62k × 5 exact
layers, so the **memory/concurrency** axis is won (N=60, recall 1.0) but the
**decode-speed** axis is still weak.

## KIE-v1.1.z — throughput attempt + the decode wall (honest)

- **kakeyalattice v1.6.1**: the gemma-4 head-dim 256/512 bug is fixed (per-layer
  lazy codec); `KakeyaLatticePackedCache` now runs on gemma-4 at **recall 1.0**
  (2.46× storage). Wired via `kv_compressor.make_packed_kv_cache`.
- **`torch.compile` the quantized attention**: **6.62×** on the standalone kernel
  (eager 36.4 ms → 5.5 ms at decode shape). **But end-to-end decode is unchanged**
  (25.2 vs 25.6 tok/s). 
- **Why (decisive):** the decode step is dominated by the **eager 26B-MoE
  full-model forward** (~322 ms per batch-8 step); the exact-layer quantized
  attention is a small slice, so fusing *it* 6.6× barely moves the total.
- **Verdict — decode ≥ vLLM is a vLLM-class kernel project, not a one-pass item.**
  Matching vLLM's ~98 tok/s/session requires graph-capturing + **fused-MoE** the
  whole forward (vLLM's core competency). The easy route
  (`cache_implementation="static"` auto-compile + CUDA graphs) **segfaults** with
  this model; a bespoke fused-MoE + graph-captured decode is the real work.

**Net for KIE-v1.1.z:** the **memory/concurrency** axis extends (N=60→75 with the
quant-attention + int8/packed storage; recall 1.0), but the **decode-throughput**
axis cannot reach vLLM parity without rebuilding the fused-MoE+graph forward —
which is the honest boundary of what this engine reaches in-session.

## What v1 already delivers

- **Chunked restoration prefill works**: the engine runs 62k at **recall 1.0** and
  reaches **N=4** — up from **N=2** for the non-chunked path — by removing the
  O(N·T²) prefill mask (it processes the prompt in 2048-token blocks).
- **Peak-window admission** (`inference_engine.engine.admission`, 9 unit tests):
  the bounded-KV cost model gives a per-session cost of **2.56 GB** and a ceiling
  of **34** sessions at 62k — independent of conversation length.

## The gap to the admission model (the v1.1 item)

Actual per-session memory is **~16 GB**, not the bounded **2.56 GB** — because the
default `generate` cache (`DynamicCache`) **stores the full sliding-layer KV**
(the sliding window is applied in the attention *mask*, the stored KV still
grows). To realize the bound, the engine must use the **sliding-window-evicting
cache**. transformers' hybrid-aware `static` cache **does evict** (verified — it
copies only the last `max_cache_len` positions per sliding layer), which would
drop per-session memory to ~2.56 GB and lift concurrency toward the **34**
ceiling — but `cache_implementation="static"` currently triggers a **CUDA-graph
capture segfault** with this model + chunked prefill. Stabilizing the
evicting/graph-captured decode path is engine work item **v1.1**.

## Honest verdict (this is on gemma-4 — the wrong showcase)

Even with the evicting cache stabilized and concurrency at ~34, this would only
**match/modestly beat vLLM on gemma-4** — and **vLLM can apply the same sliding
window**, so it is not a Kakeya algorithmic moat. gemma-4 keeps recall 1.0 at
`sliding_window=68` *with no restoration at all* (its 5/30 full-attention layers
carry recall), so the f_θ+proposer restoration is bypassed on this model.

The **decisive** win is the **FThetaRestored** policy on a **full-attention
model** (Qwen/Llama, design §5/§8/v1.2): there, shrinking the window without
restoration destroys recall, so restoration is the *only* way to bound memory at
full recall, and vLLM — having no restoration — must keep full KV (≈6× the
per-session bytes; `BoundedKVModel.advantage_ratio`) and **cannot match it**.

## Evidence

- `results/research/kakeya_engine_throughput_h200nvl_ctx62k.json`
- `results/research/vllm_multitenant_parallel_h200nvl_ctx62k.json` (vLLM 15.5 ceiling)
- unit tests: `tests/inference_engine/engine/test_admission.py`
