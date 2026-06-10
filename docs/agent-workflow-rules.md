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

## Failure log

| Date | Cost | Failure | Rule introduced |
|---|---|---|---|
| 2026-06-10 | ~15 min vast H200 GPU | Ran relmse v3 on PR #103 instead of attn_distill on PR #106 (separate child PR) | R1, R2, R3 (this document) |

When new failures surface, add a row + a new rule (or strengthen an
existing one). The rule list is **append-only** — never delete a
rule because we think we've outgrown it.
