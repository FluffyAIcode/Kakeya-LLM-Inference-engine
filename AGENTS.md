# Agent Rules

This file is read by AI coding agents (Cursor Cloud, Claude Code, etc.) at
the start of every session in this repository. It codifies the
**non-negotiable workflow rules** that prevent specific failure modes we've
hit. See `docs/agent-workflow-rules.md` for the long-form rationale + the
failure log that motivated each rule.

## Mandatory Pre-flight (every session)

1. **Read `docs/agent-workflow-rules.md`** before any non-trivial change.
2. If you are about to recommend the user spend GPU/cluster time, first
   complete the **R3 verification block** (below).

## The Rules

### R1 — SAME-PR FIX RULE

When fixing a still-open PR's failed evidence, **commit to that PR's
branch**. Do NOT open a child PR for the fix.

Permitted exceptions (each requires explicit user sign-off OR clearly
independent scope):
- Parent PR is merged → start fresh from `main`
- Fix's scope is genuinely independent (e.g. Mac MLX wrapper for a
  CUDA fix)
- User explicitly says "open a separate PR for this"

When in doubt: commit to the existing PR's branch.

### R2 — REVIEWER AID SELF-IDENTIFICATION

Every `scripts/review_*.sh` script MUST:
1. Source `scripts/_lib/reviewer_aid_header.sh`
2. Call `print_aid_header "$0" "<recipe>"` at startup, BEFORE any
   pre-flight check or GPU-time-spending operation

Recipe string MUST list the most-impactful knobs (loss type, rank,
steps, etc.) so the user can verify the run's intent at a glance.

CI enforces this via `tests/research/test_reviewer_aid_headers.py`.
Pre-existing aids that haven't been retrofitted yet are listed in
that test's `_GRANDFATHERED` set; that set must only shrink.

### R3 — PRE-GPU CONFIRMATION (agent behavior)

Before instructing the user to spend GPU/cluster/training time
("go run this on vast / Mac M4 / cluster"), include in the same
response a **verification block** with:

```
Verified by agent:
  - Branch:    <name>
  - HEAD:      <short SHA>  "<commit subject>"
  - Code path: <full path of the .py / .sh that will execute>
  - Recipe:    <key knobs: loss type, rank, steps, etc.>
  - Expected wall: <time>
```

The verification is only valid if you have, IN THIS SESSION:
1. Read the file at `Code path` (Read tool)
2. Confirmed by code inspection that the recipe knobs match the
   intended design (not just the design doc)
3. Confirmed the user is on / will checkout the correct branch

If any of those is unverifiable, stop and tell the user — do NOT
recommend the GPU-time-spending command.

## Failure Log Pointer

`docs/agent-workflow-rules.md` maintains an append-only failure log.
When a new workflow failure costs the user time, add a row + a new
rule (or strengthen an existing one). Never delete a rule.

## Repository conventions

- All write operations go through the standard tooling (StrReplace,
  Write, EditNotebook). Never use `sed` / `awk` / heredoc to modify
  tracked files.
- Reviewer aid scripts go to `scripts/review_*.sh`, with a matching
  reviewer-target naming `_on_mac.sh` / `_on_vast.sh` / `_on_cluster.sh`.
- Research scripts go to `scripts/research/`. Production engine code
  goes to `inference_engine/`. SDK code goes to `sdks/`.
- Tests mirror source paths: `tests/<package>/test_<module>.py`.
