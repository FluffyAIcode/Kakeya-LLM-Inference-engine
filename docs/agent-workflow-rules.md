# Agent Workflow Rules

**Status**: ENFORCED (2026-06-10) — failures here cost real GPU time.

These rules exist because we burned ~15 min of vast.ai GPU time on
2026-06-10 running a relmse-fix trainer on PR #103 branch when the
intended principled fix (attention-output distillation) was sitting
on PR #106's child branch. Root cause: branch fragmentation +
reviewer-aid scripts not self-identifying what they were running +
agent assuming the user knew which branch to be on.

The three rules below are **minimum sufficient** to prevent that
specific failure mode. Each has a concrete enforcement mechanism.

## R1 · SAME-PR FIX RULE

**Statement**: When fixing a still-open PR's training/inference
output, the fix MUST commit to that PR's branch. **No child PRs for
fixes.**

**When does this apply**:

| Scenario | What to do |
|---|---|
| PR open, evidence failed, you have a fix | Commit to **the same** PR's branch |
| PR merged, fix needed | New branch off `main` (the parent is gone) |
| Fix is independent in scope (e.g. Mac MLX wrapper for a CUDA fix) | New branch off the parent — **but** the parent's branch must also pin a `MUST_USE_BRANCH=<name>` banner in its reviewer aid so users don't accidentally run the obsolete config |
| User explicitly asks for a separate PR | Honor it, but explicitly tell the user "to use this fix you MUST `git checkout <branch>`" |

**Enforcement**: Pre-push hook checks: if the local branch's history
contains "Fix" / "fix recall" / "supersedes" in commit messages AND
the branch is a child of an open PR's branch (not `main`), the hook
warns.

**Example failure (the mistake we're fixing)**:

```
PR #103 branch         e18f2fc → ce25dfa (relmse fix by another agent)
                                ↘
PR #106 branch (mine)             6c2fc23 → 6f168dd (attn_distill, MY fix)

Reviewer aid runs from PR #103 branch (the user's checkout) → executes
relmse, not attn_distill. GPU time wasted.
```

The fix would have been: commit attn_distill **directly to PR #103
branch**. Then the same reviewer aid command runs the right code.

## R2 · REVIEWER AID SELF-IDENTIFICATION

**Statement**: Every `scripts/review_*.sh` script MUST print, at
startup (before any pre-flight check), an identification block:

```
==> <script name>
    Branch:        <git rev-parse --abbrev-ref HEAD>
    HEAD commit:   <git rev-parse --short HEAD> "<commit subject>"
    Trainer/eval:  <full path of the .py being invoked>
    Config recipe: <derived from CLI flags + env, e.g. "loss_type=attn_distill rank=768">
```

This block lets the user (and the reviewing agent) immediately verify
"the code that's about to spend GPU time is the code I think it is."

**Implementation**: a sourceable lib at
`scripts/_lib/reviewer_aid_header.sh` that all reviewer aids invoke
via `source "$ROOT/scripts/_lib/reviewer_aid_header.sh"; print_aid_header
"$0" "<recipe>"`.

**Enforcement**: a CI check (`tests/ci/test_reviewer_aid_headers.sh`
or similar) greps every `scripts/review_*.sh` for `print_aid_header`
or the explicit identification fields. Missing = CI fail.

## R3 · PRE-GPU CONFIRMATION (AGENT BEHAVIOR)

**Statement**: Before instructing the user to spend GPU/training time
("go run this on vast / Mac M4 / cluster"), the agent MUST in the
same response include:

1. The **exact branch** the user must be on (`git checkout <branch>`)
2. The **HEAD commit short SHA + subject** for that branch
3. The **key configuration knobs** the run will use (loss type, rank,
   steps, etc.)
4. Confirmation the agent **verified the code path** (read the
   trainer/eval file on that branch, not just the design doc)

Format:

```
Run this on vast:

    git checkout <branch>
    git pull
    HF_TOKEN=hf_xxx bash scripts/review_<aid>.sh

Verified:
  - Branch: <branch>  HEAD: <sha> "<subject>"
  - Trainer: scripts/research/<file>.py uses <loss_type> by default
  - Key configs: <list>
  - Expected wall: <time>
```

**Enforcement**: this is agent behavior — codified in
`AGENTS.md` (the agent's own ruleset, read at session start) so the
agent self-checks before every "go run on vast/CUDA/Mac" instruction.

## R4 · PRE-TRAINING FIDELITY PROBE (AGENT BEHAVIOR)

**Statement**: Before recommending or launching ANY GPU training
that costs **> 30 min wall**, the agent MUST first design + run a
**fidelity probe** that:

1. Takes **≤ 10%** of the planned training wall
2. Tests the **specific hypothesis** the training is supposed to
   validate (not just "does training run?")
3. Has a clear **falsification criterion** — "if the probe shows
   X, the training will not achieve Y"

If the probe falsifies the hypothesis, **abort the training** and
reframe — do not "run it anyway to see what happens." The user's
GPU credits are real money + real time.

**When does R4 apply**:

| Scenario | Probe |
|---|---|
| Scale-up training (more steps, bigger model, more data) | Run a 1-epoch / smoke-size version. If smoke metric on the **eval distribution** matches existing checkpoints, scale-up will likely also match. |
| New loss function | Run 500-1000 steps; verify loss decomposes as designed (each loss term is non-zero + decreases). Catches loss collapse / degeneracy. |
| New architecture (rank ↑, layers ↑, etc.) | Train tiny version (2-3 layers, rank 32). Verify shape contracts + sample loss curve. |
| New dataset / corpus extension | Verify the new data raises out-of-domain metric. Run the existing checkpoint on the new data — if metric is identical to training data, the data isn't actually new. |
| Cross-checkpoint comparison | If old + new checkpoints converge to the **same** out-of-domain metric, the bottleneck is upstream of the training change. Stop scaling that direction. |

**Fidelity probe template** (for K3 f_θ training):

```bash
# Probe (5-10 min):
python3 scripts/research/k3_integrated_niah_eval.py \
    --f-theta-dir <existing_best_checkpoint> \
    --mix-alpha-sweep "0.0,0.5,1.0" \
    --output /tmp/k3_probe_baseline.json

# Read off the f_theta_baseline_rel_mse.full_attn value.
# This is the eval-domain rel_mse FLOOR.
# If the floor is X for ALL prior checkpoints (regardless of training
# config differences), and X > recall_threshold (~0.4), then the
# bottleneck is NOT in training — DO NOT launch another training run.
# Pivot to architecture / drafter / inputs.
```

**Enforcement**: agent behavior — codified in `AGENTS.md`. Before any
shell command that initiates a > 30 min GPU training run, the agent
MUST emit a "Probe done" verification block (analogous to R3's
"Verified by agent") containing:

```
Probe verified:
  - Hypothesis: <e.g. "rank 768 + 20k steps + 128 NIAH will reduce
                 eval-domain full_attn rel_mse below 0.4">
  - Probe wall: <minutes>
  - Probe outcome: <PASS / FAIL>
  - Reasoning: <why the probe outcome supports / falsifies hypothesis>
```

If the probe FAILS, the agent MUST NOT issue the training command.

## Failure log

| Date | Cost | Failure | Rule introduced |
|---|---|---|---|
| 2026-06-10 (am) | ~15 min vast H200 | Ran relmse v3 on PR #103 instead of attn_distill on PR #106 (branch fragmentation) | R1, R2, R3 |
| 2026-06-10 (pm) | **~8.5 hr vast H200** | Ran v4a (3 hr) + v4b (5.4 hr) hybrid training assuming "bigger rank + more steps + more NIAH data" would close NIAH recall gate. Post-hoc evidence (alpha-sweep on 3 different checkpoints) showed eval-domain `full_attn rel_mse` is fixed at **~1.4-1.5 across relmse v3 / v4a / v4b**, independent of rank (256/768), steps (4k/10k/20k), NIAH count (0/64/128), or sequence length (128/1024). The bottleneck is information-theoretic (drafter K/V at eval positions), not training. A 5-min fidelity probe on relmse v3 — measuring its eval-domain `full_attn rel_mse = 1.45` and comparing to the recall threshold (0.4) — would have shown the bottleneck is upstream of any training tweak, and **the entire 8.5 hr was avoidable**. | R4 |

When new failures surface, add a row + a new rule (or strengthen an
existing one). The rule list is **append-only** — never delete a
rule because we think we've outgrown it.
