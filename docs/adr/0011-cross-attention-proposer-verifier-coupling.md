# ADR 0011 — Cross-attention proposer/verifier coupling for bounded-KV global-context inference

* **Status**: Proposed
* **Date**: 2026-06-06
* **Supersedes**: nothing
* **Extends**: [ADR 0004](0004-alignment-training-data-preparation-policy.md) (alignment training data + LoRA pipeline)
* **Coexists with**: [ADR 0008](0008-session-bound-runtime-and-grpc-protocol.md) (session-bound gRPC runtime — load-bearing for v0.3 GA, unchanged)
* **Targets**: v0.5-A (research → ship). Companion ADR 0010 (queued) covers v0.4-Q's pragmatic full-attention + INT8 KV path that ships before this lands.

## 1. Context

### 1.1 The principle violation that forced this ADR

[ADR 0001](0001-proposer-sizing-and-alignment.md) committed the project to **"no intelligence loss"** as a non-negotiable principle alongside the KV memory bound. The v0.3 GA shipped a `SinkWindowVerifier` that achieves the memory bound by **dropping K/V tensors for tokens outside a (sink + window) range**. The June 6 2026 A/B benchmark (`results/platform-tests/sink_window_quality_ab_1780714635.json`) measured the cost:

| | Full-context Qwen3-1.7B greedy | Kakeya v0.3 (Qwen3-0.6B dLM proposer + Qwen3-1.7B sink+window verifier) |
| --- | --- | --- |
| Long-context exact-recall accuracy | **6 / 6 (100 %)** | **1 / 6 (16.7 %)** |
| Peak verifier KV bytes | 56 MB | 7.6 MB |
| KV plateau | 2048 tokens | 68 tokens |

Five out of six failures are middle-context fact recall — the verifier's sink+window has evicted the K/V for the relevant tokens before generation. The remaining proposer cannot rescue them: in strict-greedy speculative decoding the verifier's `argmax` is the source of truth, and the verifier is operating on a partial cache.

This is not an implementation defect. **Any token-evicting KV strategy** (sink+window, H2O, SnapKV, PyramidKV) violates the ADR 0001 principle for any case where the evicted token's information matters for the output. ADR 0011 asks: is there a KV strategy that satisfies *both* "no intelligence loss" *and* the ADR 0008 memory-bound contract?

### 1.2 Multimodal makes the problem load-bearing, not optional

Text long-context recall is the canonical failure mode but it understates the real product motivation. The intended deployment target is local agent infrastructure on Apple Silicon ([ADR 0006](0006-local-agent-infrastructure-positioning.md)) that supports **text-video** workloads — Gemma 4-class multimodal models running locally on Mac mini, generating video conditioned on long contexts.

Memory math at the multimodal scale (Gemma 4 family, 28 layers, hidden 3584, bf16):

| Workload | Tokens | Full-attention KV | Sink+window (sink=4, window=512) | Cross-attention (this ADR) |
| --- | --- | --- | --- | --- |
| 1-turn text chat (1 k tokens) | 1 024 | 230 MB | 116 MB | ~120 MB |
| 1-turn long doc (8 k tokens) | 8 192 | 1.84 GB | 116 MB | ~150 MB |
| 5-second video (30 fps × 256 visual tokens / frame) | 38 400 | 8.6 GB | 116 MB | ~280 MB |
| 30-second video | 230 400 | **51.6 GB** | 116 MB | ~1.2 GB |

The 5-second video case is where the architectural choice becomes irreducible: **full attention runs out of unified memory on a 24 GB Mac mini, sink+window loses temporal coherence (the same 17 % failure mode as text but for visual tokens), and cross-attention is the only path that preserves both bound and quality**.

### 1.3 What's been ruled out

The two-week period between the A/B run and this ADR explored:

- **W1 (NF4 KV quantization, ADR 0010)**: ships in v0.4-Q. Full attention + INT8/NF4 quantization. Satisfies ADR 0001 (intelligence preserved up to <1 % perplexity drift on INT8). **Memory bound only at small context** — at 8 k it's 230-460 MB / session, at 30-second video it's 12-30 GB. Adequate for text agent loops, **fails at video scale**. ADR 0010 is the v0.4 pragmatic path; ADR 0011 is what unblocks v0.5+ video deployment.

- **W2 (importance-based eviction: H2O / SnapKV / PyramidKV)**: still token-eviction. Drops to ~85 % recall on RULER instead of 17 %, but **still violates** ADR 0001 strict reading. Not the path forward post-A/B clarification.

- **InfLLM-style hot/cold storage**: hot working set + compressed cold pool with retrieval. Quality similar to cross-attention; complexity comparable. Falls back into the implementation space of this ADR — could be one of the "alternatives considered" if cross-attention doesn't pan out.

- **Long-context-native model substitution (Mamba / RWKV / RetNet)**: changes the project's model identity. Would require dropping Qwen3 / Gemma and re-doing the alignment work. Out of scope for ADR 0011; potential v0.6 direction.

## 2. Decision

### 2.1 Architecture

Two coupled models with a thin cross-attention bridge:

```
                 ┌─────────────────────────────────────┐
                 │          Proposer (dLM)             │
                 │                                     │
   prompt ─────► │   full attention over T tokens      │
   T tokens      │   28 layers, hidden_p              │
                 │                                     │
                 │   →  hidden bank h_p[0..T-1]        │
                 │      [T × hidden_p × bf16]          │
                 └─────────────────────────────────────┘
                                  │
                                  │   K, V projections
                                  ▼
   ┌─────────────────────────────────────────────────────────┐
   │                  Verifier (Qwen3 / Gemma family)         │
   │                                                          │
   │   layer  1 ─────────  bounded local attention            │
   │   layer  2 ─────────  bounded local attention            │
   │   ...                                                     │
   │   layer  K ─────────  bounded local attention             │
   │   layer K+ ─────────  bounded local + cross-attn(h_p)    │  ← NEW LAYER
   │   layer K+1 ─────────  bounded local attention            │
   │   ...                                                     │
   │   layer 28 ─────────  bounded local attention             │
   │                                                          │
   │   verifier KV: bounded by (sink + window) — unchanged    │
   │   cross-attention KV: proposer hidden bank h_p           │
   └─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                            output logits
                            (greedy argmax = output)
```

**The cross-attention layer's contract**:

* `Q ← W_q · verifier_hidden_at_depth_K+`
* `K ← W_k · proposer_hidden_bank`
* `V ← W_v · proposer_hidden_bank`
* `attn = softmax(Q K^T / √d) V`
* `output = W_o · attn`, residual-added back to verifier's stream

Critically:

1. **Verifier's local attention path is unchanged**. Sink+window (or any other bounded-KV policy) still applies to verifier's self-attention. The cross-attention is **additive**, not replacement.
2. **Proposer's hidden bank is the K/V** — the bank IS the long-term memory. It's a single tensor `[T, hidden_p]`, not a per-layer KV cache. That's the memory savings vs full attention.
3. **`O_proj` is initialized to zero**. At training step 0 the cross-attention contributes nothing; the verifier behaves identically to its pre-ADR-0011 self. As training progresses, gradients flow into `W_o` and the cross-attention output gradually mixes in. This is the single most important training stability trick.

### 2.2 Training (extends ADR 0004)

ADR 0004's existing pipeline (7-domain prompt pool, hidden state collection, LoRA on `o_proj`, per-slice eval) is **reused**. ADR 0011 adds Stage 3b on top of the existing Stage 3a (Repr-Align):

| Stage | Source | Purpose | Trainable |
| --- | --- | --- | --- |
| 2 (data) | ADR 0004, **extended** | Collect (prompt, hidden_v_full, argmax_v_full); for video also (visual tokens, full-attention argmax of generated frames) | — |
| 3a (Repr-Align) | ADR 0004 | proposer's hidden ≈ verifier's hidden at corresponding position | LoRA on proposer `o_proj` |
| **3b (cross-attention)** | **NEW** | bounded verifier + cross-attn(proposer hidden bank) → matches full-attention argmax | (a) cross-attention layer weights, (b) LoRA on proposer (shared with 3a) |
| 4 (deploy) | ADR 0004 | inference: bounded verifier + cross-attention bridge + proposer hidden bank | — frozen |

Stage 3b's loss is single-objective:

```
L_3b = CE(verifier_output_with_xattn(prompt, h_p), argmax_v_full[prompt])
```

Stage 3a + 3b can be co-trained as multi-task (`L = α · L_3a + β · L_3b`) or sequentially (3a converges first, then introduce 3b with curriculum). Empirical decision; toy prototype validates which. Default starting point: **sequential (3a first), then 3b with `α = 0` (pure cross-attention objective)**.

### 2.3 Multimodal extension is mechanically free

The cross-attention bridge is **modality-agnostic** by construction. `Q`, `K`, `V` are linear projections of hidden states; they don't care whether those hiddens encode text tokens, vision patches, or audio frames. To extend from text to multimodal:

1. Replace verifier with Gemma 4 multimodal class (or whatever the current SOTA open multimodal verifier is).
2. Replace proposer with a multimodal-capable proposer of the same family (Gemma 4-2B as proposer, Gemma 4-9B as verifier — typical EAGLE-3 ratio).
3. Stage 2 prompt pool extends to include video-conditioning prompts.
4. Stage 3b cross-attention layer is the same code; the hidden_p tensor now has a mix of text + visual + audio token positions.

The only modality-specific work is in **Stage 2 data preparation**: collecting full-attention ground-truth outputs at the multimodal scale is expensive (each 5-second video eval requires a full-attention forward at 38 k tokens against a 9 B verifier — needs A100/H100, not Mac). The ADR 0004 cluster + GPU rental budget covers this.

### 2.4 Memory and compute

Per-session memory at the deployment point (Mac mini 24 GB, Gemma 4-9B verifier, Gemma 4-2B proposer):

| Component | Size | Why |
| --- | --- | --- |
| Verifier weights (bf16) | 18 GB | weights resident across all sessions |
| Proposer weights (bf16) | 4 GB | weights resident across all sessions |
| Verifier local KV per session (sink + window = 1024 tokens) | 116 MB | bounded |
| **Proposer hidden bank per session** | **T × hidden_p × bf16** | **the new memory** |
| Cross-attention layer weights | < 30 MB | trainable, single layer |
| Activations during inference | ~500 MB | transient |

Proposer hidden bank for typical workloads (Gemma 4-2B has hidden_p = 2304):

| T | hidden bank | total per-session memory |
| --- | --- | --- |
| 1 k (text chat) | 4.7 MB | ~120 MB |
| 8 k (text long doc) | 38 MB | ~150 MB |
| 38 k (5-s video) | 175 MB | ~290 MB |
| 230 k (30-s video) | 1.05 GB | ~1.2 GB |

**With weights amortized across sessions, 24 GB Mac mini supports**:

* ~50 concurrent text-chat sessions, OR
* ~30 concurrent 8 k-token long-doc sessions, OR
* ~10 concurrent 5-second video sessions, OR
* ~3 concurrent 30-second video sessions

This is the headline memory result of ADR 0011: **bounded sufficient at 24 GB for the realistic Mac M4 multimodal agent workload, while preserving (target) 99 %+ of full-attention quality**.

Compute overhead: cross-attention adds one extra attention layer's worth of FLOPs per verifier forward — small relative to the verifier's existing 28 layers.

## 3. Alternatives considered

| Alternative | Decision | Why |
| --- | --- | --- |
| **Pure full attention + INT8 KV (ADR 0010)** | Will ship as v0.4-Q; this ADR is for v0.5-A | Adequate at text scale; fails at video memory budget |
| **W2 H2O eviction** | Rejected | Still token-eviction → still violates ADR 0001 |
| **InfLLM hot/cold** | Reserved as fallback | If cross-attention research bet fails, this is the next-best memory-vs-quality point |
| **Memory tokens (Compressive Transformer style)** | Future v0.6+ | Stronger compression; 12-18 month research bet AFTER cross-attention ships |
| **Replace Qwen3/Gemma with Mamba/RWKV** | Out of scope | Changes project model identity; redoes alignment work |
| **Drop KV bound, accept full attention** | Possible if research fails | Falls back to ADR 0010-only, video deployment becomes "single-session at a time" |

## 4. Validation criteria (research bet → ship)

This ADR is conditional on empirical validation. Three gates must pass before v0.5-A ships:

### Gate G-X1: Toy prototype convergence

* Setup: small Gemma family (Gemma 3-1B / Gemma 4-2B as both proposer and verifier), text-only, needle-in-haystack
* Compute: <$500 GPU rental, 2-4 weeks
* Pass criterion: with bounded verifier (sink+window=128 over 1-2 k context) + cross-attention from full-attention proposer's hidden bank, recall on synthetic NIAH ≥ 80 % (vs full-attention baseline 100 %, vs bounded baseline ~20 %)
* Fail action: stop. Switch to ADR 0010 + InfLLM hot/cold for v0.5

### Gate G-X2: Production-scale text validation

* Setup: Gemma 4-9B verifier + Gemma 4-2B proposer, real prompts (RULER, NIAH, NarrativeQA short subset)
* Compute: ~$15-30k GPU, 3-5 months
* Pass criterion: cross-attention recall ≥ 95 % of full-attention baseline on RULER 4 k–8 k subtasks, on Mac mini at <300 MB / session
* Fail action: regression analysis; consider scaled-up cross-attention (multi-layer, multi-head, larger hidden bank)

### Gate G-X3: Video-modality validation

* Setup: Gemma 4 multimodal verifier + smaller multimodal proposer, text-video generation tasks (5-second video continuation, video QA)
* Compute: ~$30-60k GPU, 6-9 months
* Pass criterion: bounded-KV cross-attention generates temporally coherent 5-second video on Mac mini in <30 GB peak memory, indistinguishable from full-attention baseline by human eval pairwise > 50 % preference (i.e., not worse)
* Fail action: scope ADR 0011 to text-only v0.5-A, defer video to v0.6+ with InfLLM-style hierarchical alternative

## 5. Phasing

| Phase | Output | Time | Cost |
| --- | --- | --- | --- |
| P0 (this ADR) | ADR 0011 ratified, prototype scaffold | 1 week | engineer time |
| P1 (toy prototype) | Gate G-X1 pass/fail | 2-4 weeks | <$500 |
| P2 (Stage 2 data extension) | full-attention ground-truth dataset (text + later video) | 1-2 months | ~$5-10k |
| P3 (text production training) | Gate G-X2 pass/fail | 3-5 months | ~$15-30k |
| P4 (cross-attention productionization) | v0.5-A merge to main; integration test gate green on Mac M4 | 1-2 months | engineer time |
| P5 (multimodal extension training) | Gate G-X3 pass/fail | 6-9 months | ~$30-60k |
| P6 (video productionization) | v0.6-A merge with multimodal cross-attention | 1-2 months | engineer time |

Total ADR 0011 lifecycle: ~12-18 months, ~$50-100k GPU + engineer time. Gate G-X1 (4 weeks, <$500) is the critical decision point — go/no-go for the whole program.

## 6. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| **Cross-attention short-circuits** (verifier ignores local KV, relies entirely on proposer hidden bank) | Medium | Information bottleneck on hidden bank (dropout, quantization); curriculum (3a first); regularization on cross-attention output magnitude |
| **Proposer collapse** (proposer's hidden bank becomes a trivial encoding of the answer) | Medium | KL regularization; evaluate generalization to held-out prompts; vary hidden bank source layer |
| **Joint training instability** | High initially | Identity initialization on `W_o`; sequential training (3a converges → 3b introduced); gradient clipping; mixed-precision care |
| **OOD generalization weak** | Medium | 7-domain prompt pool already in ADR 0004; expand with multimodal early in Stage 2 |
| **Inference latency overhead from cross-attention** | Low | Single layer with bounded compute; can fuse into verifier's existing forward kernel |
| **MLX implementation complexity** | Medium | Stage 4 work; PyTorch reference first, MLX port after Gate G-X2 |
| **Multimodal scaling (G-X3) fails** | Medium-high | Acceptable: ship v0.5-A as text-only, multimodal becomes v0.6+ research |
| **Gemma 4 multimodal isn't open-weights as expected** | Low-medium | Substitute closest open multimodal model (Llama-3.2-Vision, InternVL, etc.) |

## 7. Open questions

* **Where in the verifier is the cross-attention layer inserted?** — Default proposal: middle (layer K = 14 of 28). Ablation in toy prototype: try {final-layer, mid-layer, multi-layer (every 4th)}.
* **Which proposer hidden state feeds the bank?** — Default: proposer's final-layer hidden after the last diffusion step. Ablation: try {final-layer, mid-layer, attention-weighted combination}.
* **Single cross-attention head or multi-head?** — Default: 8 heads (matches verifier KV head count). Ablation in toy prototype.
* **Is `α = 0, β = 1` (pure 3b) better than `α = 1, β = 1` (joint)?** — Empirical; default pure-3b after sequential 3a.
* **Does the cross-attention work for generation, not just predicting token N+1 from full prefix?** — Critical for video (long generation). Test in G-X3 explicitly.

## 8. References

* Speculative decoding correctness contract: [ADR 0001 §2.2](0001-proposer-sizing-and-alignment.md), [`kv_cache_proposer/speculative.py`](../../kv_cache_proposer/speculative.py)
* Sink+window invariant: [ADR 0001 §3, §5](0001-proposer-sizing-and-alignment.md), [ADR 0008 §2.4](0008-session-bound-runtime-and-grpc-protocol.md)
* A/B benchmark surfacing the failure: `results/platform-tests/sink_window_quality_ab_1780714635.json`
* Compressive Transformer: Rae et al., "Compressive Transformers for Long-Range Sequence Modelling," ICLR 2020.
* Memorizing Transformers: Wu et al., "Memorizing Transformers," ICLR 2022.
* InfLLM: Xiao et al., "InfLLM: Training-Free Long-Context Extrapolation," ICML 2024.
* RETRO: Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens," ICML 2022.
* EAGLE-3: Li et al., "EAGLE-3: Scaling up Inference Acceleration via Dynamic Draft Trees," 2024.
* Repr-Align training pipeline: [ADR 0004](0004-alignment-training-data-preparation-policy.md).

## 9. Implementation pointer

Toy prototype scaffold lives at `scripts/research/cross_attn_toy_prototype.py` (this PR). Phase 1 (G-X1) target: text-only Gemma 3/4 family, 1-2 week feasibility study on Mac M4. Multimodal hooks documented inline so Phase 2/3 are mechanical extensions, not architectural rewrites.
