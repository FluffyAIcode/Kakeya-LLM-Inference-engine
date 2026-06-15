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
~69 sessions fit; 4-bit → ~135. The repo already has KV-compression machinery
(`inference_engine.v04.kv_compressor`); wiring it into the bounded decode cache is
**KIE-v1.1.x** and unlocks N=34 and well beyond. (Graph-captured decode for
throughput is the parallel v1.1.x item.)

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
