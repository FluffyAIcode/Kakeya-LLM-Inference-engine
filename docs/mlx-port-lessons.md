# Porting the K3 GPU beta (#107) to MLX — lessons & plan

Audience: whoever ports the validated CUDA restored-verifier engine
(`inference_engine/v04/…`, PR #107) to the Apple-Silicon MLX backend
(`inference_engine/backends/mlx/…`). The current MLX blocker is **decode
token-throughput collapse**. This doc distills *why* #107 is fast and exactly
which mechanisms must be reproduced in MLX.

## TL;DR — the throughput collapse is the O(T²) re-forward

On MLX today, `restored_logits` (`backends/mlx/cross_model_dlm_verifier.py`) does
a **full-position forward over the whole sequence every step**, and the Mac
harness calls it per generated token → **O(T²)** → collapse (the same harness
also shows the *oracle* is fast because it uses mlx_lm's **native incremental KV
cache**). The fix is the #107 **Gap-A** trick, ported verbatim:

> **Capture the restored K/V into a persistent (sink+window) cache at prefill,
> then decode with mlx_lm's native incremental step (O(L)/block) — never
> re-forward the whole sequence per token.**

This alone takes the restored path from "collapsed" to **= native AR decode
speed** (on CUDA: 1.3–2.8 tok/s re-forward → ~21 tok/s incremental = AR).

## What makes #107 fast — and the MLX analog of each

| # | #107 (CUDA) mechanism | MLX analog / gotcha |
|---|---|---|
| 1 | **Gap-A incremental decode**: capture restored K/V (per layer, post-norm/RoPE) into a `transformers.DynamicCache` at prefill; decode L new tokens against it. | Capture into `inference_engine/backends/mlx/cache.SinkWindowKVCache` (already exists) and decode via **`mlx_lm.generate.generate_step`** with `prompt_cache=` — its **chunked prefill + `mx.async_eval` pipelined decode** is the throughput-critical part. A hand-rolled per-token loop with `mx.eval` each step is itself a collapse cause. |
| 2 | **S5 carries recall** via the 5 full-attention layers' **exact own K/V**; f_θ restores only the sliding layers (masked at decode). | Same: store the 5 full-attn evicted own K/V (KakeyaLattice-compressible); **do not** invest in f_θ sliding fidelity for recall. The needle reaches output through the full-attn layers only. |
| 3 | **Eliminate the extra `capture_own_kv` forward**: in #107 the full-attn own K/V are captured once at prefill (not recomputed per step). PR #108 showed removing it via *f_θ full-attn* breaks recall — wrong fix. | The Mac harness's 12.4s `build_restoration` is this extra forward. Right fix: capture own K/V from the **prefill** forward / store as positions evict — **not** f_θ-restore the full-attn layers. |
| 4 | **Fused spec-decode (>AR)** = three prefill-built, incrementally-extended caches: (A) verifier aux hidden from the verify forward, (B) drafter context K/V cache, (C) Gap-A restored KV. Per-block O(L). | Port `draft_block_cached` + `make/extend_context_kv` semantics to the MLX drafter path; capture aux from the MLX verify forward. Only after #1 works. |
| 5 | **Stabilization**: load verifier **without `device_map`** (no accelerate per-forward hooks) + **full-length warmup** (pre-size the allocator) → removed per-block variance. | MLX analog of the variance source is **graph (re)compilation + lazy eval**: warm up the *exact* shapes (prefill chunk size + 1-token decode) before timing; avoid shape churn; force `mx.eval` only where measuring. |
| 6 | **Gap-B drafter fidelity**: drafter query embedding is a **plain lookup — no Gemma `×sqrt(hidden)`** (port bug; fixed). | Same fix on the MLX drafting path: do not scale the shared embedding fed to the drafter. (z-lab acceptance 0.05→reference parity.) |

## MLX-specific gotchas already learned

- **MPS/MLX SDPA materializes scores** (no flash kernel for some shapes) → OOM at
  long context. Use **bounded attention** (decode only attends sink+window+restored
  evicted, not a transient full O(T) matrix) and/or **query-chunked SDPA**
  (`KAKEYA_DFLASH_ATTN_QCHUNK`). Bounded decode (Gap-A) avoids the transient full
  cache that OOM'd the ctx280 runs.
- **Lazy eval**: MLX is lazy; throughput depends on `mx.async_eval` pipelining
  (mlx_lm's `generate_step` does this). Per-token `mx.eval().item()` serializes →
  collapse. Mirror the native loop.
- **`make_sink_window_cache(model, *, sink_size, window_size)`** is keyword-only
  (a past bug was positional args). The cache is a drop-in `_BaseCache`.
- **Cross-runtime bridge**: verifier in MLX, drafter+f_θ in PyTorch (MPS/CPU) is
  workable, but the per-step tensor bridging must not re-forward; bridge only at
  the K/V-injection boundary, once per block.

## MLX port plan (ordered; each gates the next)

1. **Incremental decode (kills the collapse).** Add an MLX analog of
   `CrossModelRestoredSinkWindowVerifier(incremental=True)`: prefill → capture
   restored K/V into `SinkWindowKVCache` (full-attn = own/exact; sliding = f_θ or
   window-masked) → decode via `generate_step(prompt_cache=…)`. **Gate: decode
   tok/s ≈ native mlx_lm AR; recall 1.0** (carried by S5).
2. **Drop the extra build forward.** Capture full-attn own K/V at prefill; do not
   re-run a clean verifier forward per request beyond prefill. **Gate:
   `build_restoration` from ~12s → ~prefill cost.**
3. **Gap-B drafter embed fix** (no `×sqrt`) on the MLX/Bridge drafting path.
   **Gate: acceptance toward reference on code prompts.**
4. **Fused spec-decode** (A+B+C incremental caches). **Gate: tok/s > AR.**

## Validation gates (match #107 evidence)

- Recall **1.0** vs oracle (S5).
- Bounded resident KV (sink+window), reported via `kv_memory_report`.
- Decode tok/s: incremental **≥ native AR**; fused **> AR**.
- Reference: #107 on H200 — incremental = 1.0× AR (KV 16.9–43.9× smaller),
  fused 1.27× AR, recall 1.0. (`docs/k3-gpu-beta.md`,
  `results/research/k3_e2e_gpu_bench_incremental.json`,
  `k3_specdecode_fused_stable.json`.)

## Do-not-repeat (anti-patterns)

- ❌ Re-forwarding the full sequence per generated token (the current collapse).
- ❌ A custom decode loop with per-token `mx.eval` (no async pipelining).
- ❌ f_θ-restoring the **full-attention** layers (PR #108: breaks recall; those
  K/V are not reconstructable from the shallow drafter — α-sweep proven). Keep S5.
- ❌ Scaling the drafter's shared embedding by `×sqrt(hidden)` (Gap-B port bug).
- ❌ Materializing a transient full-T attention score matrix on MPS (OOM).
