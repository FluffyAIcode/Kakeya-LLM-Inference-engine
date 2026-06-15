# ADR 0015 — Kakeya Attention as an attention algorithm + engine substrate roadmap

- Status: Accepted (definition); substrate = roadmap
- Date: 2026-06-15
- Supersedes/extends: the "Kakeya Attention vs PagedAttention/RadixAttention"
  framing in README and ADR 0014 §3.4 (MLX serial-only / CUDA batched).

## Context

Kakeya's KV-restoration work (AR verifier + dLLM proposer + f_θ + S5 sink/window)
has been described as a *technique inside* the engine. A same-H200 benchmark vs
vLLM (`docs/reports/kakeya-vs-vllm-multitenant-h200.md`) and a long-context probe
(`docs/reports/kakeya-vs-vllm-longcontext-h200.md`) forced two clarifications:

1. The throughput gap to vLLM (~8–14×) is a **substrate** gap (eager HF
   transformers decode loop, O(T²) eager prefill), **not** an algorithmic one.
2. The bounded-KV memory advantage is **architecture-dependent** and is a
   **decode-time** property that the eager prefill currently masks (OOM at 32k).

This ADR fixes the vocabulary and the roadmap.

## Decision

### 1. Kakeya Attention is a first-class attention algorithm

Define **Kakeya Attention** = **sink+window bound + f_θ KV-projection +
dLLM-proposer restoration, as one primitive**. It is a peer of, and a drop-in
replacement for, the attention layer in current engines:

| Algorithm | Replaces | Keeps full KV? |
| --- | --- | --- |
| eager / **FlashAttention** | the **compute** layer | yes |
| vLLM **PagedAttention** / SGLang **RadixAttention** | the **storage** layer | yes |
| **Kakeya Attention** | **compute + storage** | **no — bounded; evicted KV reconstructed on demand** |

FlashAttention makes attention compute cheaper; Paged/Radix make the same *total*
KV cheaper to allocate/share. **Kakeya Attention makes the total itself bounded**,
and is **composable** with all of them (a flash kernel can compute a Kakeya
window; a paged/radix store can hold it).

### 2. The engine substrate is Kakeya Attention + CUDA graphs + fused MoE

The **Kakeya Inference Engine** = Kakeya Attention on a production substrate
(**CUDA graphs + fused-MoE kernels + memory-efficient prefill**), targeting vLLM
on absolute throughput. This is a **build target**, not yet shipped. The current
research path runs the algorithm on **eager HF transformers**, which is correct
and recall-preserving but (a) ~8–14× slower than vLLM at ctx-1238 and (b) OOMs on
long prefills (N=1-only at 16k, OOM at 32k) due to O(T²) eager scores +
full-vocab logits + a redundant `capture_verifier_own_kv` forward.

**Substrate work items (ordered):**
1. ✅ **Memory-efficient restoration prefill** — patched forward routes to
   **SDPA** (`--attn-impl sdpa`), LM-head logits chunked (`logits_to_keep=1` on
   the restored forward + `capture_verifier_own_kv`), restored K/V held in bf16.
   **Unblocked long context** (recall 1.0): 16k went N=1-only→N=4, 32k OOM→N=2,
   62k OOM→N=1; 16k N=1 peak 138.8→74.5 GB. (`docs/reports/kakeya-vs-vllm-longcontext-h200.md`.)
2. **Bounded decode cache as the native KV layout** — resident sink+window +
   5 exact full-attention layers; no Python per-step `DynamicCache`. **This is
   the gating item for the long-context concurrency win**: today the bench still
   captures full-T K/V at decode (~17 GB/session @16k), so the bounded advantage
   is not yet realized — #1 only made the prefill *run*.
3. **CUDA graphs** for the decode step; **fused-MoE** kernels for the verifier.
4. (Optional) integrate Kakeya Attention as a **vLLM attention backend** so the
   bounded window rides vLLM's paged store + scheduler.

### 3. The bounded-KV win is architecture-dependent

Resident KV is dominated by the **full-attention** layers (they hold full
context in any engine). gemma-4-26B-A4B is 25/30 **natively sliding** → vLLM
already bounds 25 layers → Kakeya's resident-KV edge is **~7 % at 62k**. On a
**full-attention** model (no native sliding) the edge is **~6×**. Long-context
concurrency "sweet spots" must be claimed **per model architecture**, not
universally. (Consistent with the recorded "S5 free-lunch" caveat: on gemma-4 the
5-exact-layer S5 shortcut already covers recall, so the proposer/f_θ restoration
is *replaced* by S5 on this model.)

## Consequences

- README and docs now present Kakeya Attention as an algorithm peer to
  Flash/Paged/Radix, with the substrate explicitly a roadmap.
- Throughput parity with vLLM is gated on the substrate work items above, not on
  the algorithm.
- Memory/sweet-spot claims are scoped by full-attention fraction; the strong
  long-context demonstration should be run on a full-attention verifier.

## Evidence

- `docs/reports/kakeya-vs-vllm-multitenant-h200.md` (ctx-1238, same H200)
- `docs/reports/kakeya-vs-vllm-longcontext-h200.md` (16k–62k, OOM ceiling + KV model)
- `results/research/{k3_cuda_multitenant_parallel,vllm_multitenant_parallel}_h200nvl_*.json`
