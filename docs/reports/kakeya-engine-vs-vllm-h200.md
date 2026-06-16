# Kakeya Inference Engine vs vLLM — long-context concurrency (same H200, gemma-4-26B-A4B)

First comparison of the **product engine** (`inference_engine.engine.KakeyaEngine`,
v1 core: chunked restoration prefill + bounded-KV decode + peak-window admission,
NativeHybridBounded policy) against vLLM on the same H200, gemma-4-26B-A4B bf16,
NIAH ctx ≈ 62k. Architecture: `docs/design/kakeya-inference-engine-architecture.md`.

## Measured (ctx 62k, recall 1.0 where it runs)

| Engine | max concurrency @62k | per-session GPU | recall |
| --- | --- | --- | --- |
| **Kakeya Engine v1** (chunked prefill, generate-cache) | **N=4** | ~16 GB | 1.0 |
| Kakeya admission model (bounded-KV target) | **34** (ceiling) | **2.56 GB** | — |
| vLLM (PagedAttention) | **15.5** | ~4.6 GB | 1.0 |

(Engine N=1 68.2 GB → N=2 84.7 → N=4 117.8 GB; N=8 OOM.)

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
