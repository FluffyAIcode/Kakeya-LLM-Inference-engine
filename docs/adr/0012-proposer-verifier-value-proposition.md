# ADR 0012 — Proposer/verifier value proposition: bounded-memory + recall (all platforms), platform-forked throughput

- **Status**: Accepted (2026-06-13)
- **Date**: 2026-06-13
- **Decision drivers**:
  - A recurring question keeps being re-opened by new contributors (human
    and agent): *"is the proposer still worth it, given Step-1 reaches 1.0× AR
    on Mac without using it?"* and *"is speculative decoding dead on Mac?"*
    This ADR settles the value map so the decision tree is not re-derived
    every time the Mac throughput number looks bad in isolation.
  - 2026-06-12 Mac ctx280 validation (`results/research/k3_mlx_fused_fair_ctx280_n5_gen32_*.json`):
    Step-1 recall 5/5 vs oracle 5/5, bounded resident KV 132.9 MB vs naive
    1308.9 MB (89.8 % saving) at 4406–5810-token prompts.
  - 2026-06-11 H200 #107 evidence: fused spec-decode 1.27× AR, recall 1.0.
  - 2026-06-13 `verify(L)` calibration sweep (`results/research/verify_l_sweep.json`,
    ctx 4096): measured kernel-dedup headroom 3.92× at L=16, ≈87 % of the
    router-measured expert-union bound (4.52×).
  - Builds on / re-affirms ADR 0001 (proposer sizing + alignment),
    ADR 0004 (alignment data policy), ADR 0006 (local-agent-infra
    positioning), ADR 0008 §11 (dLM K/V-Restoration architecture),
    ADR 0009 (capability exchange),     ADR 0010 (full-attention low-precision
    KV / affine4), ADR 0011 (cross-attention coupling, falsified by R1e).

## ⚠️ Revision (2026-06-13) — the Step-1 / S5-coupon result is a *validation trap*, not architecture evidence

A 2026-06-13 directive supersedes the optimistic reading of "Step-1 = realised
deliverable" below. The correction:

- **Step-1 (incremental restored decode) and the native-cache path get their
  recall from Gemma-4's *native* retained 5 full-attention layers + native
  sliding-window eviction — they never exercise f_θ or proposer KV
  restoration.** So "Step-1 recall 5/5 / 1.0× AR" is **Gemma-4 native
  behaviour, not evidence that the K/V-Restoration architecture (ADR 0008 §11)
  works.**
- The path is structurally **incapable of failing in a way that tests the
  architecture**: the full-attention coupon always carries recall regardless of
  whether f_θ/restoration is correct or even present. Citing it as a deliverable
  **corrupts the integrity assessment**.
- Sharper: **on Gemma-4 no configuration makes proposer/f_θ restoration the
  recall source** (the 5 full-attn layers' own exact K/V always do; f_θ only
  touches sliding layers, which are window-masked at decode). Gemma-4 is
  therefore the **wrong model to validate the restoration architecture**.
- **Step-1 / native-cache bypass is forbidden for any architecture-validation
  attempt.** The bounded-memory + recall *architecture* claim is **unvalidated
  on a falsifiable model** and must be re-validated on a **pure sliding-window
  model (Qwen3, the K1/K2 path)** where recall is mathematically impossible
  without proposer/f_θ restoration. Gemma-4 may still be used as a *product*
  model, but never as the validation vehicle for architectural integrity.

The §1 "won / realised" framing below is retained for history but must be read
through this revision: the **memory-saving numbers are real**, but they are not
proof the *restoration mechanism* works — only that Gemma-4 + a bounded sliding
cache works, which Gemma-4 does natively.

## Context

ADR 0008 §11 (K-series) changed the proposer's primary role from *drafter*
to *history reconstructor*: the dLM proposer has no KV cache and can produce
transient K/V for the **entire** history, which is used to restore the
verifier's attention at structurally-evicted positions. Speculative decoding
is the **second** product line on the same architecture, not the first.

The trap is to evaluate the architecture on a single cell of its value
table — "Mac, single host, generic chat, current un-aligned DFlash" — see a
weak throughput number, and conclude the proposer (or spec-decode) has no
value. That conclusion does not generalise. This ADR records the full value
map and prices the open options explicitly.

## Decision

The proposer/verifier value proposition is realised on **two axes**, and its
status is **platform- and workload-dependent**, not a single scalar:

### 1. The core value is "bounded memory + recall", not "fast"

Since ADR 0008 §11, the proposer's first-class role is **history
reconstruction**: no KV cache, transient full-history K/V → restore the
verifier's evicted-position attention. The main line has already **won**, but
the value is realised on the **memory axis**, not the throughput axis:
Step-1 = **1.0× AR throughput + recall 5/5 + KV 132.9 MB vs naive 1308.9 MB
(89.8 % saving; ~48 MB after affine4 / ADR 0010)**. That is the
proposer/verifier deliverable.

A finer honesty note: in the Mac **S5-native** shipping configuration,
Gemma-4's *native hybrid attention* means keeping the **5 full-attention
layers exact** is already enough to carry recall — so on this specific model
the f_θ/proposer reconstruction is **replaced by the S5 shortcut**. But on a
**pure sliding-window architecture** (the K1/K2 Qwen3 case — no
full-attention layers to preserve) and on the **CUDA full-restoration path**,
proposer reconstruction remains the **only** source of recall. The
architecture's domain of applicability is unchanged; Gemma-4 simply handed us
a free coupon.

### 2. Speculative-decoding value forks by platform — the Mac negative does not extrapolate

- **H200 (#107 measured)**: fused = **1.27× AR, recall 1.0** — the *same*
  proposer/verifier code; on a platform where verify-batch is nearly free,
  spec-decode value holds.
- **Mac's 0.26×** has a **concrete, movable** bottleneck: real per-token
  acceptance is **30–40 %**. The vLLM reference reports the *same* drafter at
  **44.7 %**, and our own drafter docs say "the precise EAGLE-3 ↔ block-fusion
  alignment is a Stage-2 task". Alignment fine-tuning (the plan that has been
  queued in ADR 0001 / 0004 all along) lifting acceptance to **~70 %** makes
  the block-4 arithmetic `3.5 × 43.8 / 140 ≈ 1.1×` — Mac clears the bar too.
  So the Mac status is **"waiting for the alignment asset"**, not
  **"architecture pronounced dead"**.

### 3. Option value of the verification primitive: correctness containment makes any draft source plug-and-play

The v3 loop + byte-level consistency guarantees one thing: a draft source can
only affect **throughput**, never **pollute output**. This is an open
interface; the drafter can be swapped for anything. A concrete, this-week,
testable Mac route: **NGramProposer** (the zero-weight prompt-lookup proposer
already in PR #105) + the v3 loop — draft cost ≈ 0, no alignment dependency,
naturally high acceptance on agentic workloads (tool-call JSON, templated
replies, highly self-repetitive sessions). Arithmetic: `draft ≈ 0 +
verify(4) = 120 ms`, committing 2.5/block → 0.78×, 3.5/block → 1.1× — on
Kakeya's target workload (ADR 0006: local agent infrastructure) this is
entirely plausible. One bridge command verifies it.

### 4. Beyond single-host throughput, the split is the foundation for a multi-host architecture

ADR 0009 / PR #105's capability-exchange plane is built with the
proposer/verifier roles as primitives: the proposer is a fleet capability that
can be gossip-discovered and remote-invoked. Even if a single Mac never runs
spec-decode, the "verifier on host A, proposer capability on host B / cloud"
shape (including the dev/eval tool plane) is **already running** — the Mac
bridge used over the last two days is itself an instance of this
architecture's tool plane.

### Bottom line

If the proposition is narrowed to *"Mac single-host + generic chat + the
current un-aligned DFlash"* — yes, the proposer has **no runtime value in
that one cell today**, and Step-1 reaches 1.0× without it. But the
architecture's value map is: **the realised bounded-memory story (all
platforms) + the realised throughput story (CUDA) + two explicitly-priced Mac
throughput options (alignment fine-tuning / n-gram drafting) + the foundation
for the multi-host capability plane.**

## Consequences

- The proposer/verifier split is **retained**; "Step-1 doesn't use the
  proposer on Mac/Gemma-4" is **not** grounds to deprecate it (it is the only
  recall source on pure-sliding-window models and on CUDA full-restoration).
- Memory-axis claims (bounded KV, S5, affine4) are the **primary**,
  all-platform deliverable and should be reported as such; throughput claims
  must be qualified by platform (CUDA: realised; Mac: option-pending).
- Two priced Mac throughput options are tracked as next steps:
  (a) **alignment fine-tuning** (ADR 0001/0004) to lift DFlash acceptance
  toward the 44.7 % reference / ~70 % target; (b) **NGramProposer × v3 loop**
  for agentic workloads (draft ≈ 0, no training). Option (b) is the cheapest
  and is verifiable in a single bridge run.
- Any future "is spec-decode worth it?" discussion must specify the
  **(platform, workload, drafter)** cell; a negative in one cell does not
  generalise.

## Alternatives considered

- **"The Mac throughput number kills the architecture."** Rejected: the Mac
  cell has a concrete, movable bottleneck (acceptance 30–40 % vs 44.7 %
  reference), and the negative does not extrapolate to CUDA (1.27×, realised)
  or to the memory axis (89.8 % saving, realised, all platforms).
- **"Deprecate the proposer because Step-1 reaches 1.0× without it on
  Gemma-4."** Rejected: S5 is a Gemma-4-specific free coupon (native hybrid
  attention); on pure sliding-window models (K1/K2 Qwen3) and the CUDA
  full-restoration path the proposer is the only recall source.
- **"Only ship spec-decode if it beats AR everywhere."** Rejected: the value
  is platform-forked; CUDA already clears it, and the verification primitive's
  correctness containment makes the Mac throughput a strictly-additive option
  (it can never regress correctness).

## Evidence pointers

- Mac bounded-memory + recall: `results/research/k3_mlx_fused_fair_ctx280_n5_gen32_*.json`
  (recall 5/5, KV 132.9 MB vs 1308.9 MB), `docs/pr109-mac-ctx280-validation.md`.
- CUDA throughput: PR #107, `docs/k3-gpu-beta.md`.
- verify(L) headroom: `results/research/verify_l_sweep.json` (3.92× measured @
  L=16 vs 4.52× expert-union bound).
- Drafter alignment status: ADR 0001/0004; `inference_engine/v04/dflash_drafter.py`
  ("Stage-2" fidelity note).
- Capability plane / multi-host: ADR 0009, PR #105; `scripts/mac_bridge/`.
