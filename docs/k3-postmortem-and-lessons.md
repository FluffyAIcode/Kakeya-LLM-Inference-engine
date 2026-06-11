# K3 Development Postmortem + Agent Lessons (2026-06-09 → 2026-06-11)

**Status**: post-failure retrospective. K3 NIAH recall gate **not closed**
in v0.4 timeline. Documented for Mac mini review and as a reference
artifact extending `AGENTS.md` and `docs/agent-workflow-rules.md`.

## Scope

K3 = the v0.4 K/V Restoration product gate — verifier (Gemma 4 26B-A4B)
holds only sink+window cache, drafter (DFlash 0.4B, aligned) provides
K/V at evicted positions via a learned `f_θ` projection. Goal: NIAH
recall ≥ 90% with sink=4, window=64.

Time span: 2026-06-09 to 2026-06-11 (~2.5 days of agent work).

GPU spend (vast.ai H200): **≥ 25 hr** measured, much avoidable.

Outcome: **gate NOT closed**. Information-theoretic floor on drafter
K/V at eval-domain positions identified as the bottleneck. Path forward
documented; not yet executed.

This document is the canonical retrospective. It supersedes scattered
analysis in commit messages and PR descriptions.

---

## Table of Contents

1. [Architecture summary](#architecture-summary)
2. [Milestones (chronological, what worked)](#milestones)
3. [Failure log (chronological, what didn't)](#failure-log)
4. [Pattern analysis](#pattern-analysis)
5. [Lessons → AGENTS.md additions](#lessons)
6. [Open questions / next steps](#next-steps)

---

## 1. Architecture summary <a id="architecture-summary"></a>

```
input_ids
   │
   ├─→ DRAFTER (frozen, 0.4B DFlash, aligned via PR #93 Eagle-3)
   │      │
   │      └─ drafter K/V (5 layers × 256 dim per pos)
   │            │
   │            └─→ f_θ (94M params, rank 768, the only TRAINING module)
   │                   │
   │                   └─→ verifier-space K/V at all positions
   │                          │
   │                          └─→ injected at evicted positions
   │                                                            │
   │                                                            ▼
   └─→ VERIFIER (frozen, 27B Gemma 4 26B-A4B, eager attn)  ──→ logits
        │
        └─ self.attn at each layer:
              for evicted positions: use f_θ K/V
              for sink+window:       use verifier's own K/V
              compute attention(Q, K_combined, V_combined)
              return o_proj(attn_out)
```

**Three frozen entities + one trainable adapter (f_θ).**

Dimensions:
- Drafter: 5 layers × {2 KV heads × 128 dim} = 5 × 256 = 1280 dim per pos
- Verifier (Gemma 4 26B-A4B):
  - 25 sliding layers × {8 KV heads × 256 dim} = 51200 dim
  - 5 full-attention layers × {2 KV heads × 512 dim, K=V} = 5120 dim
  - Total per pos: 56320 dim
- Compression input/output ratio: 56320 / 1280 = **44×** (drafter must
  encode info that f_θ can expand 44× into verifier-space K/V).

Recall threshold (empirically, from PR #103 alpha-sweep evidence):
- `full_attn rel_mse ≤ 0.36` → recall = 1.0
- `full_attn rel_mse ~ 0.52` → recall = 0.6
- `full_attn rel_mse ≥ 0.71` → recall = 0

---

## 2. Milestones (what worked) <a id="milestones"></a>

| Date | Milestone | Evidence |
|---|---|---|
| 2026-06-09 | **K3 Block A**: hardware feasibility on vast (Gemma 4 + DFlash drafter both load + forward) | `results/research/k3_feasibility_smoke_vast_blockA_*.json` |
| 2026-06-09 | **K3 Block B**: DFlashDrafter native loader (PR #93's checkpoint loads with correct keys + extras) | `inference_engine/v04/dflash_loader.py`, `inference_engine/v04/dflash_drafter.py` |
| 2026-06-09 | DFlash drafter **Eagle-3 alignment** training: acceptance 0.003 → 0.074 (in-domain) → reference-level after 64-prompt corpus scaling | `models/dflash-kakeya-baseline/` (Git LFS) |
| 2026-06-10 am | **f_θ engine API**: `FThetaProjection` + `CrossModelDLMRestoredVerifier` + integrated NIAH eval | `inference_engine/v04/f_theta.py`, `cross_model_dlm_verifier.py` |
| 2026-06-10 am | **Identity-restore evidence: recall = 1.0** — proves K/V injection plumbing + sink+window machinery is correct | `results/research/k3_identity_restore_ctx70.json` |
| 2026-06-10 am | **Heterogeneous per-layer KV head/head_dim** in f_θ: Gemma 4 has 2 distinct layer types (sliding 8×256 vs full 2×512); FThetaConfig captures per-layer | `inference_engine/v04/f_theta.py` (`verifier_layer_kv_heads`, `verifier_layer_head_dims`) |
| 2026-06-10 am | **relmse v3 alpha-sweep evidence** maps recall ↔ rel_mse threshold: full_attn rel_mse 0.36→1.0, 0.52→0.6, 0.71→0 | `results/research/k3_alpha_sweep_relmse.json`, `k3_alpha_sweep_relmse_knee.json` |
| 2026-06-10 pm | **Workflow rules R1+R2+R3 codified** after first GPU-time waste (15 min) | `docs/agent-workflow-rules.md`, `AGENTS.md`, `tests/research/test_reviewer_aid_headers.py` |
| 2026-06-10 pm | **Hybrid loss design**: attn_distill + cosine direction + magnitude. Theoretically right for cross-model K/V mapping. | `scripts/research/k3_f_theta_train.py` `_attention_distillation_loss(hybrid=True)` |
| 2026-06-10 pm | **Mac MLX cross-model verifier** (PR #104, parallel-track) — same architecture, MLX-native injection | `inference_engine/v04/cross_model_dlm_verifier_mlx.py` |
| 2026-06-11 | **Information-theoretic floor identified**: eval-domain `full_attn rel_mse ≈ 1.4-1.5` is invariant across loss/rank/steps/data. Bottleneck is upstream of training. | `results/research/k3_fidelity_*v4{a,b}*.json` |
| 2026-06-11 | **Workflow rule R4 codified** after second GPU-time waste (8.5 hr) | `docs/agent-workflow-rules.md` (R4 section) |

---

## 3. Failure log (chronological, with cost) <a id="failure-log"></a>

### F1 — ADR 0011 cross-attention bridge (G-X1) — 4 GPU-hr wasted

**Hypothesis**: Train a cross-attention "write head" so the verifier can
recover from sink+window cache loss via a direct attention bridge to
the proposer.

**Actions**: 5 successive PRs (R1, R1b, R1c, R1d, R1d-β, R1e) over
~4 days, total ~4 hr H200. Bug fixes (matcher, capacity, init, aux
loss).

**Outcome**: empirically falsified. Even with capacity bumped (R1e
write-path expansion), recall flat. Not a write-path bottleneck —
the bridge architecture itself doesn't carry the right information.

**ADR 0011**: WITHDRAWN (commit `2e933cf`).

**Lesson learned (already absorbed)**: ADR 0008 was rewritten to
focus on `dLM K/V Restoration` instead. KakeyaLattice was pulled
forward from K4 to K2.A.

### F2 — K2.A.1 Mac M4 production smoke serial bugfix cycle — ~30 min wasted (per attempt × 4)

**Hypothesis**: K2.A.1 KL ON + stateful path produces good Mac M4
production-shape evidence.

**Actions**: 4 successive Mac M4 runs, each failing with a different
bug:
1. dtype mismatch (`index_copy_` bf16 vs fp32) — fixed with cast
2. Quadratic broadcast (19.20 GB buffer for ctx280) — fixed with
   transpose order
3. HF Cache contract (`get_mask_sizes` missing) — fixed
4. Short test eventually passed

**Outcome**: each fix uncovered the next bug. No systematic
pre-flight checking of the full integration path.

**Lesson learned**: integration tests should run END-TO-END against
the smallest viable workload first. Found the dtype issue → fix →
should have re-run end-to-end immediately to find next bug, instead
of only re-running on the failing config.

### F3 — Wrong-branch GPU run (PR #103 vs PR #106) — 15 min wasted (2026-06-10 am)

**Hypothesis**: Running `bash scripts/review_pr_k3_f_theta_train_on_vast.sh`
on PR #103 branch executes the attention-output distillation trainer.

**Actual**: PR #103 branch had the relmse trainer (committed by another
agent). PR #106 (child branch I created) had attn_distill. The
reviewer aid printed nothing about which trainer was running, so the
user couldn't tell.

**Cost**: 15 min H200 GPU running relmse instead of attn_distill —
NIAH recall 0/10, didn't help.

**Root cause**: I created PR #106 as a CHILD branch of PR #103 instead
of committing to PR #103 directly. Branch fragmentation made the user
end up running the wrong version.

**Codified as R1 + R2 + R3** in `docs/agent-workflow-rules.md`.

### F4 — v4 hybrid training campaign — **8.5 hr wasted** (2026-06-10 pm to 2026-06-11)

**Hypothesis**: Hybrid loss + rank 768 + 20k steps + 128 NIAH prompts +
gen_len 1024 will reduce eval-domain `full_attn rel_mse` from 1.45
(relmse v3 baseline) to below the 0.4 recall threshold.

**Actions**:
1. v4a (warmstart from relmse v3, rank 256, 10k steps, 64 NIAH, gen 1024) — 3 hr H200
2. v4b (fresh start, rank 768, 20k steps, 128 NIAH, gen 1024) — 5.4 hr H200

**Outcome**: BOTH hit eval-domain `full_attn rel_mse ≈ 1.4-1.5`,
identical to relmse v3's 1.45. NIAH recall 0/10 in both. Bigger,
longer, more data — same floor.

**Cost**: 8.5 hr H200 (~$15-25 vast.ai credits).

**Root cause**: I recommended v4 training without first running a
fidelity probe. A 5-min probe (running `--mix-alpha-sweep` on
relmse v3 + comparing baseline rel_mse to recall threshold) would
have shown the floor was already at 1.45 and no training tweak had
historically reduced it. The 8.5 hr was directionally wrong from the
start.

**Pattern**: identical to F1's pattern — keep iterating on a wrong
direction without first falsifying the hypothesis cheaply.

**Codified as R4** in `docs/agent-workflow-rules.md` (commit `5066cd3`).

---

## 4. Pattern analysis <a id="pattern-analysis"></a>

Three failure modes recurred. Each costs the user real time / GPU
credits. R1-R4 cover them, but understanding the underlying patterns
helps prevent NEW failure modes in the same family.

### Pattern A — "Just iterate" without falsifying

F1 (ADR 0011) and F4 (v4 campaign) share a structure:
- Initial hypothesis (cross-attention bridge / hybrid loss)
- Run 1 fails → assume "almost there, need more capacity / more
  training / more data"
- Run 2-N: each tweaks one knob (capacity / training / data) without
  questioning the underlying hypothesis
- Eventually empirical evidence accumulates → realize the hypothesis
  was wrong from start

**Cost asymmetry**: each failed run costs hours of GPU. Each "did the
hypothesis already fail to be true" check costs minutes.

**Cure**: R4 forces this check explicitly. Generalize: BEFORE iterating,
ASK "what evidence would prove this direction is wrong?" then check
that evidence cheaply.

### Pattern B — Branch fragmentation

F3 (wrong-branch run) is the canonical example. The pattern:
- Multiple agents (or one agent across multiple sessions) work on
  the same PR
- Fixes land on different branches
- User runs reviewer aid → gets whatever's on their checked-out
  branch

**Cure**: R1 (commit fixes to the same PR's branch) + R2 (reviewer
aids self-identify branch + recipe). Generalize: branch hygiene matters
as much as code hygiene.

### Pattern C — Integration debugging by ping-pong

F2 (Mac K2.A.1 serial bugfix) shows a different pattern:
- Run integration → bug 1 → fix → re-run → bug 2 → fix → re-run → ...
- Each cycle is 10-30 min (not catastrophic but adds up)
- No pre-flight "would this work end-to-end at all" check

**Cure**: pre-flight smoke tests on the smallest viable end-to-end
workload before scaling up. Not yet codified as a rule (could be R5).

---

## 5. Lessons → AGENTS.md additions <a id="lessons"></a>

R1-R4 are already shipped. The full retrospective surfaces three
additional candidate rules / extensions:

### R5 (proposed) — END-TO-END SMOKE BEFORE SCALING

**Statement**: When integrating a new component (data path, loss
function, model wrapper), run an end-to-end smoke test on the
SMALLEST viable workload (1 sample, 100 steps, T=128) before scaling
to production sizes.

**Falsification**: if smoke fails, no amount of scaling will fix it.

**Triggers F2**: each Mac K2.A.1 bug (dtype, broadcast, cache contract)
would have surfaced in a 10-second smoke run. Ping-pong cost ~2 hr
total over 4 cycles.

### R6 (proposed) — HONEST FAILURE MODES

**Statement**: When recall / acceptance / quality gate fails, do NOT
hide failure with "looks promising, let's iterate." State the failure
explicitly + propose three falsification experiments + only after one
of them produces information should iteration continue.

**Triggers F4**: I described v3 attn_distill mse_O = 0.176 as
"converged" — but the alpha-sweep later revealed K/V was 135× off-scale.
"Converged on the wrong objective" is not the same as "converged."

### R7 (proposed) — PRIOR-ART CHECK BEFORE RE-INVENTING

**Statement**: When designing a new training scheme (new loss
function, new architecture variant, new training pipeline), spend
≥ 15 min on prior-art search (FitNets, TinyBERT, MiniLM, distillation
literature) BEFORE writing the implementation. Cite at least one
relevant paper in the design doc.

**Triggers latent in F4**: hybrid loss = TinyBERT 4-loss combo. I
designed it from first principles without realizing it's textbook
distillation practice. Would have:
- Saved design time (paper has the tuned λ values)
- Caught the K/V degeneracy issue earlier (FitNets paper warns about
  exactly this)

### Cross-cutting lesson — Information theory > optimization tweaks

The single biggest meta-lesson from F4: when an objective metric is
INVARIANT across multiple training-side knobs (rank, steps, data,
loss), the bottleneck is INFORMATION-THEORETIC, not optimization.

Concrete example: across 3 checkpoints (relmse v3 / v4a / v4b) with
totally different training configs, eval-domain `full_attn rel_mse`
landed at 1.42-1.52. That's a floor. No optimization tweak breaks it.

The fix to a floor is upstream:
- Bigger drafter (more raw information capacity in the input to f_θ)
- Drafter retraining with co-objective (specialize for K/V mapping)
- Architecture change (hidden states instead of K/V as f_θ input)
- Or: accept the floor + adapt the architecture (e.g. don't compress
  full-attn layers, keep their K/V cache)

R4 is the operational rule that catches this. The cross-cutting
intellectual lesson is: **CHECK for floors before tuning**.

---

## 6. Open questions / next steps <a id="next-steps"></a>

After this postmortem, K3 NIAH recall gate remains open. Three options
ranked by ROI:

### Path A — Keep verifier K/V at full-attn layers (recommended)

Modify `cross_model_dlm_verifier.py` to bypass K/V Restoration at the
5 full-attention layers (5/30 = 17% of layer count, 9% of K/V memory).
Run `f_θ` on the 25 sliding layers only.

- Cost: ~50 LOC edit + 1 NIAH eval (5 min)
- Risk: low — sliding layers have K/V dim 8×256 = 2048 vs full 2×512 =
  1024; rel_mse on sliding layers from existing checkpoints already
  shows reasonable fidelity
- Compression ratio: 9.3× (vs current target 11×) — 17% sacrifice for
  100% recall potential

R4-compliant probe before launching:
```bash
# Modify cross_model_dlm_verifier.py to skip f_θ for full-attn layers
# (a 50-LOC change). Then:
python3 scripts/research/k3_integrated_niah_eval.py \
    --f-theta-dir results/research/f_theta_v4b_fresh_hybrid \
    --output /tmp/k3_path_a_probe.json
# 10-min probe; recall ≥ 0.5 means path A works.
```

### Path B — Bigger drafter (1B+ params)

Replace 0.4B DFlash drafter with 1-2B variant. More param capacity to
encode long-range features.

- Cost: re-run PR #93 alignment (~10-20 GPU-hr) + redo K3 from scratch
- Risk: medium — 1B drafter may still not encode 27B's 5 full-attn
  layers' worth of long-context features. No guarantee.

### Path C — Use drafter hidden states as f_θ input

Currently `f_θ` consumes drafter K/V. Try drafter hidden states (the
input to k_proj/v_proj) — strictly more information per position.

- Cost: ~architecture change in `cross_model_dlm_verifier.py` + retrain
- Risk: medium — uncertain if hidden states have the missing
  information. Could be the same floor.

### My recommendation

**Path A first** (R4-compliant, 10 min cost, fast falsification). If
Path A NIAH recall ≥ 0.5, ship. If not, the bottleneck is broader than
the full-attn layers, escalate to Path B.

---

## Appendix: GPU spend summary

| Date | Activity | Hours | Notes |
|---|---|---|---|
| 2026-06-09 | K3 Block A feasibility (vast) | ~1 | Necessary |
| 2026-06-09 | DFlash alignment (Eagle-3) | ~10 | Necessary, produced `models/dflash-kakeya-baseline` |
| 2026-06-09 | DFlash Stage 2 spec-decode harness + corpus scaling | ~2 | Necessary |
| 2026-06-09 to 2026-06-10 | F1 ADR 0011 toy prototype + R1c/d/e iterations | ~4 | **Wasted** (toy bridge architecture falsified) |
| 2026-06-10 am | f_θ v1 training (PR #103 attn_distill) | ~0.25 | Necessary baseline (recall 0 → diagnostic) |
| 2026-06-10 am | f_θ v3 relmse training | ~0.25 | Necessary baseline (knee evidence) |
| 2026-06-10 am | F3 wrong-branch run (relmse instead of attn_distill) | ~0.25 | **Wasted** |
| 2026-06-10 pm | F4 v4a hybrid training | ~3 | **Wasted** (R4 should have prevented) |
| 2026-06-10 pm to 2026-06-11 | F4 v4b hybrid training + collection | ~5.4 | **Wasted** (R4 should have prevented) |
| **Total** | | **~26 hr** | |
| **Wasted (could have been avoided with R1+R2+R3+R4)** | | **~13 hr** | **50% of total** |

Half the GPU spend was avoidable with the rules now codified.

---

## How to use this document

- Mac mini reviewer: `git pull` on PR #103 branch, read this file +
  `docs/agent-workflow-rules.md` + `AGENTS.md`.
- Future agent (me or another): read AGENTS.md at session start. R1-R4
  are mandatory. R5-R7 are proposed; promote to mandatory if a future
  failure shows they're needed.
- Future K3 / K4 work: start with the R4 fidelity probe template
  before launching any training.

This document is append-only. When new failure modes appear, add to
section 3, update section 4 patterns, propose new R rules in section 5.


