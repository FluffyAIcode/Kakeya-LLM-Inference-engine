# Feasibility: Kakeya Attention as a vLLM attention backend — can decode reach vLLM throughput?

Scoping for **KIE-v2** (run Kakeya Attention *inside* vLLM instead of rebuilding
vLLM's runtime). The question this note answers: **does plugging Kakeya's
bounded-restored-quantized cache in as a vLLM attention backend recover
vLLM-level decode token throughput?**

## TL;DR — Yes, with low throughput-parity risk (the integration effort is the cost, not the speed).

The decode throughput comes from the parts of the forward Kakeya **does not**
touch (those stay vLLM's), and the part Kakeya **does** own (attention) is a
minor decode fraction over *smaller* KV — so it won't become the bottleneck if
implemented as a proper graph-capturable kernel.

## 1. Decode-cost decomposition (why parity is the low-risk axis)

Per decode token, the 26B-A4B forward (30 layers) is:

| component | who owns it under KIE-v2 | share of decode cost |
| --- | --- | --- |
| QKV/o projections, norms, RoPE, residuals | **vLLM** (fused/graphed) | small |
| **attention op** | **Kakeya backend** | **minor at decode** (1 query token) |
| **MoE FFN** (router + 128 experts) | **vLLM fused-MoE** (graphed) | **dominant (~the forward)** |
| LM head | vLLM | small |

Measured here (eager, no vLLM): the MoE-dominated step is ~322 ms at batch-8;
the attention is a small slice. **vLLM's throughput is its fused-MoE + CUDA-graph
+ scheduler.** By plugging in as the attention backend, Kakeya **inherits all of
that for the dominant ~90%** of the forward — it is not rebuilding the runtime.

## 2. Is Kakeya's attention (the part it owns) a slowdown vs vLLM's?

No — it operates on **less** KV than vLLM's full-context attention:

- **Sliding layers (25/30 on gemma-4):** bounded window (sink+window). vLLM
  already bounds these (native 1024); Kakeya bounds tighter (e.g. 68). Attention
  over a *smaller* window is **≤** vLLM's cost → not a bottleneck. Reuse vLLM's
  existing flash/paged kernel on the small window.
- **Exact layers (5/30):** full-context, but over **quantized** KV. vLLM already
  ships **fp8 KV-cache attention with in-kernel dequant**; an int8/lattice
  variant is the same class of kernel (dequant-in-kernel flash). At decode this
  is comparable to vLLM's full-attention layers — **not slower**.

So the attention op is, at worst, vLLM-comparable and likely cheaper (smaller
KV). It does not regress decode throughput **provided it is a real
graph-capturable kernel** (not the current Python tiled loop, which was a
research stand-in).

## 3. Does decode need restoration (the slow proposer forward)? — No.

Restoration (proposer + f_θ reconstruction of evicted K/V) is a **prefill-time**
operation, run **once** to build the bounded resident set. At **decode**:

- gemma-4: sliding layers are natively local (no reconstruction); exact layers
  keep full K/V. **No restoration runs per token.**
- full-attention models (Qwen): exact layers keep full K/V; the sliding/bounded
  layers' evicted positions were reconstructed **at prefill** into the resident
  set. **No per-token proposer forward.**

So the decode step is **restoration-free** → fixed-shape → **graph-capturable**.
This is what makes vLLM's CUDA-graph decode applicable to the Kakeya backend.

## 4. Verdict on throughput parity

**Decode tok/s ≈ vLLM: achievable, low risk.** The throughput is dominated by
vLLM's fused-MoE + graph + scheduler, which KIE-v2 inherits unchanged; Kakeya
owns only the attention op, which is a minor decode fraction over smaller/
quantized KV and is restoration-free at decode. The single hard requirement is a
**graph-capturable quantized-exact attention kernel** (extend vLLM's fp8-KV
attention to int8/lattice) — a bounded, well-scoped kernel, not a runtime rebuild.

## 5. Honest caveats / where the real work and risk are

1. **vLLM attention-backend + KV-manager conformance (moderate–high effort).**
   Kakeya's hybrid bounded layout (per-layer different sizes; quantized exact
   layers) must fit vLLM V1's `AttentionBackend` + paged KV-manager / block-table
   abstractions. vLLM already supports gemma-4's hybrid sliding/full cache, which
   de-risks the per-layer-type handling, but a quantized + tighter-window layout
   is non-standard integration work.
2. **Quantized-exact attention kernel** — must exist and be graph-safe. vLLM's
   fp8-KV attention is the precedent; int8/lattice is an extension, not net-new
   research.
3. **Restoration at prefill** must be injected into vLLM's prefill pipeline
   (proposer forward + f_θ). On **gemma-4 this is unnecessary** (S5 free lunch),
   so a gemma-4 KIE-v2 backend is much simpler; on **full-attention models** it is
   the real integration (and the real memory differentiator).
4. **Memory differentiation is model-dependent.** On gemma-4, vLLM *already*
   hybrid-bounds the sliding layers → Kakeya's extra saving is modest (~7% at
   62k). The large bounded-KV win (and thus the reason to do KIE-v2 at all over
   plain vLLM) is on **full-attention models**, where vLLM keeps all layers full
   and Kakeya keeps exact+window — the ~6× resident-KV edge.

## 6. Recommendation

KIE-v2 (Kakeya-as-vLLM-backend) is the **right path to "N high *and* decode ≥
vLLM"**: it gets vLLM throughput by inheritance and contributes Kakeya's actual
differentiator (bounded-KV + restoration + quantized attention) as the attention
backend. The throughput-parity risk is low; the work is integration + one
quantized-attention kernel. The showcase model should be **full-attention**
(Qwen/Llama), where the bounded-KV memory win is large — on gemma-4 the win is
small enough that plain vLLM is already close.
