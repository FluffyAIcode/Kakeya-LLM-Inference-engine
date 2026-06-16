# Kakeya Inference Engine — Build Skill (SOP)

**Audience / when to use this skill.** Read this before building, extending,
benchmarking, or *validating* any part of the Kakeya Inference Engine (KIE) — on
CUDA or Mac/MLX. It distills the full build journey (v0.4 → KIE-v1.x → KIE-v2 →
v0.5-cuda) into: what the engine is, where the code lives, how to run/benchmark
it, the milestone roadmap, the hard-won bugs+fixes, and — most important — the
**validation honesty standards** (the rules that keep claims defensible).

> If you only read one section, read **§7 Validation & honesty standards**. The
> most expensive mistakes in this project were *overclaims*, not bugs.

---

## 1. North star (governs everything)

The Kakeya Inference Engine is a **product-grade inference engine whose goal is to
replace vLLM**, with **Kakeya Attention as its native, first-class attention
algorithm**. It is **not** a research script, **not** a technique bolted onto HF
transformers, and **not** "vLLM with a different cache". The whole engine
(prefill, KV management, admission/scheduling, kernels) is designed
**bounded-KV-native**: the full history is never resident; evicted context is
reconstructed on demand. Authoritative source: `docs/adr/0015-kakeya-attention-and-engine-substrate.md`
and `docs/design/kakeya-inference-engine-architecture.md`.

### Kakeya Attention (the algorithm — one primitive)

**sink+window bound + f_θ KV-projection + dLLM-proposer restoration, taken as one
primitive.** It is a peer / drop-in replacement for eager attention,
FlashAttention, vLLM PagedAttention, SGLang RadixAttention. Those keep the *whole*
KV (memory grows with the conversation); Kakeya Attention bounds *how much* is
resident and **reconstructs evicted context on demand** (proposer + f_θ), so the
resident footprint does not grow with the session.

- **Compute axis**: composable with FlashAttention (a flash kernel can compute a
  Kakeya window).
- **Storage axis**: composable with paged/radix stores (they can hold the bounded
  window).
- **The Kakeya-only axis**: the *total itself* is bounded. Cost = restoration
  compute (a proposer forward at prefill).

---

## 2. Architecture & where the code lives

| Component | Role | Code |
| --- | --- | --- |
| **AR verifier** (Gemma-4 26B-A4B, frozen) | the model being served; carries recall | `inference_engine/v04/dlm_restored_verifier.py`, `build_restored.py` |
| **dLLM proposer** (DFlash) | reconstructs evicted K/V (restoration) | `inference_engine/v04/dflash_drafter.py`, `cross_model_dlm_verifier.py` |
| **f_θ projection** | trained map from proposer hidden → verifier K/V | `inference_engine/v04/f_theta.py`; training: `docs/design/k3-f-theta-training-pipeline.md` |
| **KV capture / merge / compress** | capture own K/V at prefill; pack/quantize | `inference_engine/v04/kv_capture.py`, `kv_merge.py`, `kv_compressor.py` |
| **Engine runtime** (KIE-v1.x) | chunked restoration prefill + bounded-KV decode | `inference_engine/engine/kakeya_engine.py` |
| **Admission / bounded-KV math** | peak-window admission, concurrency ceiling (pure stdlib) | `inference_engine/engine/admission.py` |
| **Quantized attention** | tiled online-softmax over int8 KV (no bf16 transient) | `inference_engine/engine/quant_attention.py` |
| **KakeyaVLLM (v0.5 entrypoint, KIE-v2)** | Kakeya window **on the vLLM runtime** | `inference_engine/engine/kakeya_vllm.py` |
| **MLX backend** | Apple-Silicon port (`v0.4-mac`) | `inference_engine/backends/mlx/*` |
| **gRPC session runtime** | session-bound serving (ADR 0008) | `inference_engine/session/*`, `inference_engine/server/*` |

**Two engine substrates exist — know which you're touching:**
1. **`KakeyaEngine`** (`engine/kakeya_engine.py`) — the eager HF-transformers
   research/feasibility substrate. Wins the **memory/concurrency** axis (N=75 @62k,
   recall 1.0) but decode speed is weak (eager 26B-MoE forward dominates). **Never
   ship or benchmark this as "Kakeya performance"** — it is a correctness probe.
2. **`KakeyaVLLM`** (`engine/kakeya_vllm.py`) — the **product** path: Kakeya's
   bounded window **on vLLM**, inheriting vLLM's (Apache-2.0) fused-MoE Triton
   kernel + CUDA graphs + continuous-batching scheduler. This is the v0.5-cuda
   release artifact.

> **Not a contradiction with §1's "not vLLM with a different cache".** The
> *north-star* engine is bounded-KV-native and does not live inside vLLM.
> `KakeyaVLLM` (KIE-v2 / v0.5) is the **pragmatic interim**: rebuilding vLLM's
> fused-MoE + graphs + scheduler from scratch was attempted (KIE-v1.1.z2) and
> shown to be a multi-week kernel project, so v0.5 wins the **decode-speed axis
> now** by running Kakeya Attention *on* vLLM. ADR 0015 reconciles this
> explicitly: vLLM's runtime is inherited; Kakeya owns the bounded-KV attention
> layer; the native bounded-KV engine matures alongside.

---

## 3. Platforms & how to run

### 3.1 CUDA (Vast.ai H200) — the primary benchmark platform

GPU access is via SSH to a Vast.ai instance. Connection details live in injected
secrets: `vast_ssh_host`, `VAST_SSH_PORT`, `VAST_SSH_USER`, `vast_ssh_key`. The
**host/port changes between sessions** — the user supplies a fresh
`ssh -p <port> root@<host>` each time; trust the user's latest details over the
stale env vars.

> **GOTCHA — the SSH key's newlines are collapsed.** `vast_ssh_key` is stored as a
> single line (PEM newlines stripped) → `ssh` fails with `error in libcrypto`. You
> MUST reconstruct a valid PEM before use:
> ```python
> import os, re
> k = os.environ["vast_ssh_key"]
> b, e = "-----BEGIN OPENSSH PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----"
> body = re.sub(r"\s+", "", k.split(b,1)[1].split(e,1)[0])
> pem = b + "\n" + "\n".join(body[i:i+70] for i in range(0,len(body),70)) + "\n" + e + "\n"
> open("/tmp/vk","w").write(pem); os.chmod("/tmp/vk",0o600)
> ```
> Validate with `ssh-keygen -y -f /tmp/vk`. Then `ssh -i /tmp/vk -p <port> root@<host>`.

> **GOTCHA — disk is often tiny.** Some instances have ~4 GB free on the overlay
> (`/workspace`, `/root`); the multi-TB devices shown by `df` are bind-mount
> artifacts (NVIDIA libs, `/etc/hosts`), **not usable dirs**. Check
> `findmnt`/writable space before assuming you can download a 26B model (~52 GB).
> A pre-existing venv with vLLM is usually at `/root/venv-vllm`; HF cache under
> `$HF_HOME` (e.g. `/workspace/.hf_home`). Set `HF_HUB_OFFLINE=1` to use cached models.

Provisioning helper: `scripts/research/run_on_vast.sh` (creates `.venv-vast`,
installs CUDA torch + transformers, verifies GPU). Run scripts on the host with
`PYTHONPATH=.:sdks/python`.

### 3.2 Mac / MLX (`v0.4-mac`)

MLX runs only on Apple Silicon; the cloud agent reaches a Mac M4 via the **Mac
bridge** (`docs/design/mac-bridge-cloud-agent-access.md`, `docs/mac-bridge.md`).
Port lessons: `docs/mlx-port-lessons.md`.

### 3.3 Key benchmark / test entrypoints

| Goal | Command |
| --- | --- |
| vLLM vs Kakeya-on-vLLM (KIE-v2 / v0.5) | `scripts/research/vllm_multitenant_parallel_bench.py --sliding-window 68` |
| KIE eager engine throughput/concurrency | `scripts/eval/kakeya_engine_throughput_eval.py` (`--quant-attn`, `--compile-attn`, `--decoupled`) |
| CUDA multi-tenant feasibility probe | `scripts/research/k3_cuda_multitenant_parallel_bench.py` |
| MLX batched multi-tenant | `scripts/research/mlx_batched_multitenant_bench.py` |
| Admission math unit tests | `pytest tests/inference_engine/engine/test_admission.py` |
| v0.5 wrapper config unit tests | `pytest tests/inference_engine/engine/test_kakeya_vllm.py` |

---

## 4. Milestone roadmap & current status

| Code | What | Status |
| --- | --- | --- |
| **v0.4-cuda / v0.4-mac** | restored Gemma-4 verifier + fused DFlash spec-decode | shipped — CUDA fused **≈1.79× AR** (committed scorecard); up to **~2.06–2.20× co-located** (ADR 0014). MLX **≈AR parity (~0.93–1.05× AR)** — a memory win, not a Mac speed win |
| **KIE-v1** (#135) | engine core: chunked restoration prefill + bounded-KV decode + peak-window admission | done (core); concurrency gated on v1.1 |
| **KIE-v1.1** (#136) | realize the bound at runtime: sliding-window-**evicting** StaticCache, graph capture OFF | done — 62k N=4→**N=16** (recall 1.0) with the evicting cache alone; **N=24** (1.55× vLLM) only after **prefill chunk-size tuning** (1024/512), see §below |
| **KIE-v1.1.x** (#137) | int8/int4 exact-layer KV quant toward N=34+ | partial — recall-safe + halves stored bytes, but **N=34 OOMs** (dequant-on-read transient). The N=16→N=24 chunk-tuning lives here too |
| **KIE-v1.1.y** (#138) | **quantized attention** (tiled online-softmax over int8, no bf16 transient) | done — **N=60 @62k** (peak 111.7 GB), recall 1.0, ~3.9× vLLM's ≈15.5 |
| **KIE-v1.1.z** (#139) | throughput + N=75 | **N=75 MET** (recall 1.0, 126.7 GB, ~4.8× vLLM; ~31 tok/s aggregate); **decode ≥ vLLM NOT met** (eager 26B-MoE wall) |
| **KIE-v1.1.z2** | rebuild fused-MoE + graph forward | **abandoned** — superseded by KIE-v2 (run *on* vLLM) |
| **KIE-v2** (#140) | **Kakeya Attention on vLLM** | decode **≥ vLLM (1.15–1.23×)** @16k, recall 1.0, measured to N=70 — inherits vLLM runtime |
| **v0.5-cuda** (#141) | release `KakeyaVLLM` + consolidated reports | done (gemma-4 instantiation). Product concurrency claim = **`KakeyaVLLM` N→70 @16k** on vLLM; the **N=75 @62k is the *eager* `KakeyaEngine` substrate**, not the v0.5 product path — do not conflate. See §7 for exact validation scope |
| **v0.6** (= ADR 0015 KIE-v1.2) | **restoration backend on full-attention models** (Qwen/Llama): train f_θ/proposer + inject restoration at vLLM prefill + graph-capturable quantized-exact kernel | **planned — the real memory differentiator (~6×)** |

> **N=16 vs N=24 (KIE-v1.1 precaution).** The evicting StaticCache alone at the
> default prefill chunk (2048) tops out at **N=16** @62k; **N=24** required smaller
> prefill chunks (1024/512) and is tracked under KIE-v1.1.x. Don't credit N=24 to
> the evicting cache alone (`docs/reports/kakeya-engine-vs-vllm-h200.md`,
> `docs/design/kakeya-inference-engine-architecture.md` §9).

---

## 5. Hard-won bugs & fixes (don't re-discover these)

| Symptom | Root cause | Fix |
| --- | --- | --- |
| MLX batched decode recall 0.125 | MLX **core kernel** bug for `B>1, L=1` quantized/rope decode (confirmed `0.31.2/0.31.3`) | `L>=2` padded decode workaround (recall 1.0, 0.67× tput); upstream bug, not ours |
| MLX O(T²) throughput collapse | `restored_logits` did a full-sequence forward **per token** | Gap-A: capture restored K/V into native cache at prefill, decode incrementally (`mlx_lm.generate_step`) |
| Eager prefill OOM (16k N=2, 32k N=1) | O(T²) scores + full-vocab logits + redundant forwards | SDPA + `logits_to_keep=1` + bf16 f_θ K/V |
| StaticCache CUDA-graph **segfault** (chunked + long) | gemma-4 has non-graph-capturable ops (windowed `copy_` eviction; data-dependent MoE routing) — **structural** | pre-build StaticCache, `TORCHDYNAMO_DISABLE=1` (run evicting cache eager) |
| `StaticSlidingWindowLayer` `AttributeError: device` | manual cache stacking dropped metadata | copy all metadata attrs in `_stack_caches` |
| int8 exact-layer misclassified as `LinearAttention` | not subclassing `CacheLayerMixin` | lazy factory subclassing `transformers.cache_utils.CacheLayerMixin` |
| `KakeyaLatticePackedCache` `expected last dim 256, got 512` | codec assumed uniform head_dim; gemma-4 full layers = 512, sliding = 256 | `kakeyalattice` v1.6.1 per-layer lazy head_dim (upstream) |
| int8 storage halves bytes but **N=34 still OOMs** | cache `update()` returns **bf16** → each exact layer dequantizes full K/V on read; transients coexist | the real fix is **quantized attention** (KIE-v1.1.y) — attend on int8 without materializing bf16 |
| `torch.compile` attention 6.6× but **0% e2e decode gain** | decode dominated by **eager 26B-MoE full-model forward**, not attention | need fused-MoE + full-forward graph capture → that's vLLM's job → **KIE-v2** |
| fused-MoE port blocked | HF `kernels` incompatible w/ transformers 5.12; vLLM `fused_moe` cross-venv surgery; from-scratch = multi-week | **run Kakeya ON vLLM** instead of rebuilding it (KIE-v2) |
| `KakeyaVLLM` crash on text-only model | unconditional `text_config` nesting (gemma multimodal) breaks Qwen/Llama (`num_attention_heads` missing) | **auto-detect** `text_config` via `AutoConfig`: nested for gemma-4, flat for Qwen/Llama |

---

## 6. Engineering workflow (how this project ships)

- **One milestone = one PR, stacked.** KIE-v1 (#135) → v1.1 (#136) → v1.1.x (#137)
  → v1.1.y (#138) → v1.1.z (#139) → KIE-v2 (#140) → v0.5-cuda (#141), each based on
  the previous branch so the diff stays per-task. Branch prefix `AgentMemory/…`.
- **ADR + report discipline.** Every milestone updates `docs/adr/0015-…` milestone
  table and a report under `docs/reports/`. Decisions and *honest caveats* are
  written down, not just code.
- **Hypothesis-driven, runtime-evidenced.** Never claim a fix from code alone —
  reproduce, instrument, measure on the real GPU. Each optimization revealed the
  *next* bottleneck (eager prefill OOM → bf16 KV floor → dequant transient → MoE
  forward → vLLM runtime). Expect this ladder; don't skip rungs.
- **Pragmatism over heroics.** Python-only workarounds and leveraging existing
  libraries (vLLM, kakeyalattice) beat multi-week from-scratch kernels within a
  session — *as long as the claim matches what was actually built*.

---

## 7. Validation & honesty standards (READ THIS)

The single most damaging error pattern in this project is **overclaiming a
validation**. Follow these rules rigidly.

### 7.1 What counts as validating "the engine" vs "the plumbing"

- **Engine/algorithm validation** = the actual claim (recall, memory, throughput)
  measured **on the release model, through the release code path, exercising the
  mechanism being claimed.**
- **Plumbing/smoke test** = "the wrapper constructs, the config is applied, it
  generates" — proves the code runs, proves **nothing** about the algorithm.
- **Label every artifact as one or the other.** Never let a smoke test masquerade
  as engine validation. (Case study: a Qwen3-4B run of `KakeyaVLLM` was wrongly
  presented as "end-to-end validation". It was plumbing-only — see §7.3.)

### 7.2 The Gemma-4 "S5 free lunch" — and why it does NOT generalize

- On **gemma-4-26B-A4B**, recall is **1.0 at `sliding_window=68` with NO
  restoration**, because **5 of 30 layers are native full-attention and carry
  recall**. So the gemma-4 instantiation (v0.5-cuda) is honest **without a trained
  f_θ/proposer** — restoration is *bypassed*, not exercised.
- Therefore the gemma-4 **memory win over vLLM is small (~7% @62k)**: vLLM already
  hybrid-bounds the 25 sliding layers, and the 5 full layers dominate both engines.
- **The large bounded-KV win (~6×) requires a FULL-ATTENTION model** (Qwen/Llama,
  all layers full), where shrinking the window **without restoration destroys
  recall** — so restoration is the *only* way to bound memory at full recall, and
  vLLM (no restoration) must keep full KV.

### 7.3 HARD RULE: never validate Kakeya Attention on a model without trained f_θ/proposer

A bounded window **without** trained restoration is **naive truncation, not Kakeya
Attention.** On a full-attention model with no trained f_θ/proposer:
- restoration never runs;
- short prompts (< window) never even trigger eviction → the mechanism is untested;
- long prompts lose recall (expected — that's *why* restoration is needed).

So you **cannot** demonstrate the engine on such a model. The v0.6 work is exactly
"train f_θ/proposer for a full-attention model **then** validate". Until then, the
only defensible engine evidence is gemma-4 (§7.2).

### 7.4 Decode-speed honesty

- The **eager `KakeyaEngine`** wins memory/concurrency but is slow at decode
  (~25–31 tok/s aggregate; the eager 26B-MoE forward dominates). Report decode-only
  tok/s **separately from prefill** — the `aggregate_tps_e2e` figure folds in the
  sequential 62k prefill and looks like ~2 tok/s, which is a harness artifact, not
  the decode rate.
- The **product** decode-speed story is **KakeyaVLLM** (≥ vLLM), because it
  inherits vLLM's fused-MoE + CUDA graphs + scheduler. Don't claim product decode
  speed from the eager engine.

### 7.5 Checklist before writing "validated" anywhere

1. Did the **release code path** run (not a side script that approximates it)?
2. Was the claim's **mechanism actually exercised** (restoration ran? eviction
   triggered? quant attention hit?)?
3. Is it on the **release model**, or are you extrapolating from a proxy? If a
   proxy, say so and say what's still unproven.
4. Is recall measured with a **real NIAH/needle test**, not vibes from a short prompt?
5. Is the **artifact labelled** smoke-test vs engine-validation?
6. Are the **caveats** (model-dependence, prefill-vs-decode, untrained components)
   in the report, not just the happy numbers?

If any answer is "no", write the weaker, true claim.

---

## 8. Pointers

- North star + algorithm + milestones: `docs/adr/0015-kakeya-attention-and-engine-substrate.md`
- Engine architecture: `docs/design/kakeya-inference-engine-architecture.md`
- KIE-v2 feasibility (decode-cost decomposition): `docs/design/kakeya-vllm-backend-feasibility.md`
- v0.5-cuda scorecard (+ honest §5): `docs/reports/kakeya-inference-engine-v0.5-cuda.md`
- Engine vs vLLM long-context journey: `docs/reports/kakeya-engine-vs-vllm-h200.md`, `docs/reports/kakeya-vs-vllm-longcontext-h200.md`
- MLX port lessons: `docs/mlx-port-lessons.md`
- f_θ training pipeline: `docs/design/k3-f-theta-training-pipeline.md`
- Session capacity / cross-host: `docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md`
