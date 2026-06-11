# Agent Session Context Dump

**Honest scope**: this directory contains every raw artifact accessible
from the cloud agent's VM. The literal user↔agent message-by-message
transcript **is NOT in this dump** — Cursor stores conversation
messages server-side, not on the agent's VM filesystem (verified by
checking `~/.cursor-server/data`, `$AGENT_TRANSCRIPTS`, and the entire
filesystem for files matching the conversation ID
`bc-51150fb2-5f33-444d-b21d-9baf584d8e7f`; no transcript file exists).

**To get the literal conversation transcript**: use the Cursor web UI's
session export feature. Conversation ID above identifies this session
to Cursor's web UI.

What this dump DOES contain (raw, byte-for-byte):

```
agent-session-context/
├── README.md                    (this file)
├── terminals/                   raw output of every Shell tool call
│                                  (PID + cwd + commands + stdout/stderr)
├── git-commits/                 full commit messages + diff stats
│                                  for every branch involved in K3 work,
│                                  since 2026-06-09 (start of v4 cycle)
├── evidence/                    every results/research/*.json
│                                  (training reports, NIAH evals,
│                                  alpha-sweeps, fidelity probes)
├── scripts/                     every script the agent created or
│                                  modified during the session
└── docs/                        AGENTS.md + workflow rules +
                                   postmortem markdown
```

## Directory contents

### `terminals/`

Each `<id>.txt` is the raw output buffer of one Shell tool invocation.
Format: header lines (pid, cwd, command, started_at, running_for_ms /
exit_code) + stdout + stderr. This is the **closest thing to a tool-call
trace** the VM has.

Files prefixed with hex IDs (e.g. `5bff6af5-…txt`) are older session
artifacts from earlier conversations on this same project (the
"DLM proposer + AR verifier" framework discussion from May 2026), kept
here for completeness.

### `git-commits/`

`ALL-branches-since-2026-06-09.log` (~6000 lines) is the master log:
every commit's full message body + filenames-changed + insertion/
deletion counts, across all K3-relevant branches.

Per-branch logs (`AgentMemory_v04-pr-k3-…log`) are the same data
filtered to one branch each, for easier reading.

The commit messages on this project are LONG and detailed (often
50-150 lines each) because the workflow rules R1-R4 emerged during
this session and the messages document reasoning + evidence inline.
Effectively, the commit log is **the agent's reasoning timeline**.

### `evidence/`

Every JSON file under `results/research/`. Includes:

- Training reports: `f_theta_v1.json`, `f_theta_v3.json`,
  `f_theta_v4a_warmstart_hybrid.json`, `f_theta_v4b_fresh_hybrid.json`
- Integrated NIAH evals: `k3_integrated_niah_*.json`
- Alpha sweeps: `k3_alpha_sweep_*.json`, `k3_fidelity_*.json`
- Identity-restore: `k3_identity_restore_ctx70.json`
- DFlash spec-decode harness: `k3_dflash_specdecode_*.json`
- Mac smoke evidence: `k3_feasibility_smoke_*.json`,
  `k3_dflash_specdecode_mac_*.json`

### `scripts/`

The Python + Bash code central to K3:

- `f_theta.py` — the trainable projection module
- `cross_model_dlm_verifier.py` — CUDA cross-model wrapper
- `cross_model_dlm_verifier_mlx.py` — Mac MLX variant
- `k3_f_theta_train.py` — trainer (relmse + attn_distill + hybrid)
- `k3_integrated_niah_eval.py` — eval harness with `--mix-alpha-sweep`
- `k3_v4_analyze.py` — auto-analyzer used by the polling loop
- `k3_v4_evidence_watcher.sh` — polling watcher used during v4b wait
- `review_pr_k3_*.sh` — reviewer aids
- `reviewer_aid_header.sh` — R2 enforcement library

### `docs/`

- `AGENTS.md` — the agent's mandatory ruleset (R1-R4)
- `agent-workflow-rules.md` — long-form rules + failure log
- `k3-postmortem-and-lessons.md` — the comprehensive postmortem
  (394 lines), section §5 proposes R5/R6/R7

## Reading order for Mac mini review

If you have ~30 min:

1. `docs/k3-postmortem-and-lessons.md` (the abstracted summary)
2. `docs/agent-workflow-rules.md` (the rules)
3. `docs/AGENTS.md` (rule enforcement format)

If you have ~2 hr and want the raw timeline:

1. `git-commits/ALL-branches-since-2026-06-09.log` — read commit
   messages chronologically. The reasoning is in the messages.
2. `evidence/k3_alpha_sweep_*.json` — the sweep data that finally
   pinpointed the information-theoretic floor
3. `evidence/f_theta_v*.json` — training reports for v1, v3, v4a, v4b
4. `terminals/827446.txt` and `962370.txt` — the v4b training watcher
   logs
5. Compare `docs/k3-postmortem-and-lessons.md` vs the raw timeline
   above to see what the abstraction kept vs. dropped

If you want the literal user↔agent conversation:

- Use Cursor's web UI session export, conversation ID
  `bc-51150fb2-5f33-444d-b21d-9baf584d8e7f`. The agent VM does not
  have this data.

## Why the abstracted document exists

The user asked for "the complete development process" and I produced
`docs/k3-postmortem-and-lessons.md` which is an abstracted summary,
not raw context. The user pushed back: "我要的是完整的 context 记录。
不是你抽象之后的文档记录" (I want the complete context record, not your
abstracted document).

This dump is the response to that pushback. It's the **rawest possible
form of agent-VM-side data**. The abstract document is now correctly
positioned as ONE artifact among the raw files — not a replacement for
them.

## Limitations of this dump

1. **No raw conversation messages.** Stored server-side. Not in this dump.
2. **No tool-call structured trace.** I have access to terminal stdout
   from Shell tool calls, but not a structured "tool call → tool args
   → tool result" trace for non-Shell tools (Read, Write, StrReplace,
   etc.). Those tool calls happened in-session and left no
   filesystem-readable record.
3. **No agent reasoning steps.** My internal chain-of-thought is not
   accessible from inside the agent. Reasoning lives only in (a) the
   commit messages I wrote, (b) the markdown docs I wrote, (c) what
   the user saw in chat.
4. **Conversation summaries provided at session start.** When this
   session was initialized, the host provided a "[Previous conversation
   summary]" block (an LLM-generated summary of prior turns). That
   summary is not in this dump because (a) it was provided in-context
   to me, not written to disk, and (b) it's itself an abstracted
   artifact — not raw.

## Mac mini access

```bash
# On Mac mini:
git pull origin AgentMemory/v04-pr-k3-block-c-f-theta-design-and-skeleton-8e7f
cd <repo>/agent-session-context

# Option 1: read individual files in your editor.
# Option 2: zip the whole thing and unpack elsewhere:
tar czf /tmp/k3_session_context.tar.gz agent-session-context/

# Total size: ~2.3 MB compressed.
```
