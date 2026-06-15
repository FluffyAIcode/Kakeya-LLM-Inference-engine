# Long-context: Kakeya bounded-KV vs vLLM full-KV (same H200, gemma-4-26B-A4B)

Tests the long-context (16k–64k) memory + concurrency thesis: does Kakeya's
bounded resident KV let it serve **more concurrent sessions** than vLLM at long
context — the parallel-forward "sweet spot" where aggregate throughput would
overtake vLLM? Same H200 as `kakeya-vs-vllm-multitenant-h200.md`.

**TL;DR — the answer is architecture-dependent, and the current research engine
can't realize it on gemma-4 yet:**

1. **vLLM scales to 62k cleanly** (max concurrency ≈ 15.5 sessions, recall 1.0).
2. **The current Kakeya *eager* research engine cannot do long context at all** —
   N=1-only at 16k (138.8 GB peak), **OOM at 32k even N=1**. This is purely the
   **eager O(T²) prefill substrate**, not the bounded-KV algorithm.
3. **On gemma-4 the bounded-KV memory edge is small (~7% resident KV)** because
   gemma-4 is natively **25/30 sliding-window** layers — vLLM already bounds
   those. The 5 full-attention layers dominate long-context KV and are full in
   **both**. The large bounded-KV win requires a **full-attention** model.

## Empirical (measured on this H200)

| Engine | ctx | N (sessions) | decode tok/s | recall | peak GPU | note |
| --- | --- | --- | --- | --- | --- | --- |
| Kakeya eager restored-S5 | 16k | 1 | 17.7 | 1.0 | **138.8 GB** | barely fits at N=1 |
| Kakeya eager restored-S5 | 16k | 2 | — | — | **OOM** | needs +30.7 GiB |
| Kakeya eager restored-S5 | 32k | 1 | — | — | **OOM** | needs +61.2 GiB |
| vLLM (PagedAttention) | 62k | 1 | 98.3 | 1.0 | 126.7 GB pool | runs |
| vLLM (PagedAttention) | 62k | — | — | — | — | **max concurrency ≈ 15.5** (71 GiB KV → 1.02M tokens) |

(vLLM N=2/4 at 62k decode-rate is prefill-dominated metric noise — 62k prefill ×
N vastly exceeds 32 decode tokens; the load-bearing vLLM number is its **reported
max concurrency 15.51×** for 66k-token requests.)

The Kakeya OOM is the **eager prefill**: the restoration forward materializes
O(T²) attention scores for the 5 full-attention layers, runs
`capture_verifier_own_kv` (a redundant full verifier forward), and materializes
the full `[N,T,vocab=262144]` logits — at 16k that already costs ~139 GB at N=1.
**None of this is the bounded resident KV** (which is a decode-time property);
it is the unoptimized prefill substrate masking the algorithm.

## Resident-KV model (decode-time, the real serving-capacity axis)

gemma-4-26B-A4B: 30 layers (25 sliding @ window 1024 + 5 full-attention),
8 KV-heads × 256 head-dim, bf16 ⇒ **8192 B / token / layer** (K+V).

Per-session **resident** KV at context C:

| config | formula | C = 62k |
| --- | --- | --- |
| **Kakeya S5** (sink 4 + win 64; 5 full layers full-ctx) | `(5·C + 25·68)·8192` | **2.55 GB** |
| **vLLM on gemma-4** (5 full + 25 sliding@1024) | `(5·C + 25·1024)·8192` | 2.75 GB → **1.08×** |
| **vLLM on a full-attention 30-layer model** (no native sliding) | `30·C·8192` | 15.24 GB → **5.97×** |

The decisive term is `5·C` — the **5 full-attention layers**, which **both**
engines keep at full context on gemma-4. So on gemma-4 Kakeya's resident-KV
advantage is only ~7%. **On a full-attention model** (where vLLM must keep all
30 layers full while Kakeya bounds 25 of them) the advantage is **~6×** — that is
where bounded-KV produces a real long-context concurrency sweet spot.

> This is the same "S5 free-lunch" caveat recorded earlier: gemma-4's native
> sliding window means "keep 5 full-attention layers exact" already covers the
> recall-critical attention, so f_θ/restoration is *replaced* by the S5 shortcut
> on this particular model — and vLLM gets the sliding-window saving too.

## Sweet spot — verdict

- **Memory sweet spot exists, but its size is set by the model's full-attention
  fraction**, not by Kakeya alone. gemma-4 (5/30 full) ⇒ ~7% resident-KV edge ⇒
  marginal concurrency gain. A full-attention model ⇒ ~6× edge ⇒ a large gain.
- **It cannot be realized on the current eager engine** at all: the eager prefill
  OOMs at 32k. Realizing it requires the **optimized substrate** (ADR 0015):
  memory-efficient restoration prefill (SDPA/FlashAttention, chunked logits, no
  redundant full forward) + bounded decode cache + CUDA graphs + fused MoE.
- Until that substrate lands, **vLLM is the better long-context engine on
  gemma-4** on this hardware (it runs 62k at ~15× concurrency; Kakeya eager does
  not run past 16k N=1).

## Evidence

- `results/research/vllm_multitenant_parallel_h200nvl_ctx62k.json`
- Kakeya eager 16k/32k OOM ceiling — `/dev/shm/klong.log` on `vastgpu4`
  (16k N=1 restored 17.7 tok/s, recall 1.0, peak 138.84 GB; 16k N=2 + 32k N=1 OOM)
- ctx-1238 baseline: `kakeya-vs-vllm-multitenant-h200.md`
