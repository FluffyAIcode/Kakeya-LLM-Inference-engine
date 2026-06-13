# Design — Mac bridge: cloud-agent access to the self-hosted `kakeya-mac-m4`

> **New here?** Start with the reader-friendly guide
> [`docs/mac-bridge.md`](../mac-bridge.md) (what the soft link is, the
> request/response flow, security, quickstart). This document is the deeper
> design record (transports M1/M2/M3 + fleet-integration evaluation).

- **Status**: M1 implemented (git-bus transport); M2/M3 designed
- **Relates to**: ADR 0009 (multi-host plane), PR #105 (CapabilityService),
  PR #109 evidence gate (`inference_engine/bench/k3_report_gate.py`),
  [`docs/ops/mac-m4-runner-setup.md`](../ops/mac-m4-runner-setup.md)
- **Implementation**: [`inference_engine/bridge/`](../../inference_engine/bridge/),
  [`scripts/mac_bridge/`](../../scripts/mac_bridge/),
  [`.github/workflows/mac-bridge.yaml`](../../.github/workflows/mac-bridge.yaml)

## 1. Problem

Kakeya development now happens substantially through cloud agents running
on **Linux x86 VMs with no Metal**. Everything MLX-dependent — the MLX
verifier, `mlx.distributed`, the K3 Mac harness, the PR #109 evidence-gate
reruns — needs Apple Silicon. The project owns exactly one such machine:
the Mac mini registered as the self-hosted runner
`[self-hosted, macOS, ARM64, kakeya-mac-m4]`, sitting behind NAT with
**outbound-only** connectivity (the Actions runner long-polls GitHub).

Constraints that shape the design:

- **C1 — No inbound path to the Mac.** No public IP, no port forwarding.
  Any transport must be initiated from the Mac side or relayed.
- **C2 — Cloud agents are ephemeral and git-native.** They reliably have:
  a repo checkout, git push permission, and read-only `gh`. They do NOT
  reliably have: VPN keys, SSH keys to the Mac, or workflow-dispatch
  permission.
- **C3 — The Mac executes whatever lands on it.** A bridge that forwards
  arbitrary shell from an internet-facing queue to a desk machine is a
  remote-shell backdoor. Command surface must be an allowlist.
- **C4 — Evidence discipline.** Results coming back from the Mac must
  flow through the PR #109 evidence gate, not around it.

## 2. Architecture: three transports, one capability model

```
M1 (this PR)            M2 (queued)                M3 (queued)
┌─────────────────┐    ┌──────────────────┐      ┌──────────────────────┐
│ git-bus         │    │ tailnet SSH      │      │ Kakeya fleet member  │
│                 │    │                  │      │                      │
│ agent ──push──► │    │ agent ──SSH──►   │      │ agent ──gRPC──►      │
│  mac-bridge/*   │    │  Mac (tailscaled)│      │  CapabilityService   │
│  branch+manifest│    │  interactive REPL│      │  ProposerService     │
│ Mac runner:     │    │  lldb / py-spy / │      │  (ADR 0009 plane,    │
│  run preset,    │    │  mlx debugging   │      │   PR #105, over the  │
│  commit results │    │                  │      │   M2 tailnet)        │
│  back to branch │    │                  │      │                      │
└─────────────────┘    └──────────────────┘      └──────────────────────┘
 async, batch,          interactive,              programmatic,
 zero new secrets       needs TS authkey          inference-native
```

### 2.1 M1 — git-bus (implemented)

The only transport that satisfies C1+C2 with **zero new infrastructure**:
git is the RPC bus, the Actions runner is the executor, the branch is the
session.

Protocol:

1. **Request.** The agent runs `scripts/mac_bridge/request_run.py
   --preset <name> [--param k=v ...] [--ref <workload-ref>]`. The client:
   - branches `mac-bridge/<preset>-<nonce>` from the workload ref,
   - overlays the bridge files if the ref predates them (workflow +
     executor must exist on the pushed branch — `on: push` workflows
     execute the pushed commit's definition),
   - writes `.mac-bridge/request.json` (the manifest), commits, pushes.
2. **Execute.** `.github/workflows/mac-bridge.yaml` triggers on
   `push: branches: ['mac-bridge/**']`, runs on `kakeya-mac-m4`,
   serialized via a `mac-bridge` concurrency group (one Mac). It calls
   `scripts/mac_bridge/run_preset.py --manifest .mac-bridge/request.json`,
   which validates the manifest against the **preset allowlist**
   (`inference_engine/bridge/manifest.py`) and executes the preset's
   fixed argv list — no shell interpolation of any user-controlled
   string (C3).
3. **Respond.** The runner commits `.mac-bridge/logs/` + any new
   `results/research/*.json` back to the same branch and pushes; it also
   uploads them as workflow artifacts. K3 acceptance reports are passed
   through `scripts/validate_k3_reports.py` **on the Mac** so a
   non-conforming report fails the bridge run itself (C4).
4. **Fetch.** The agent polls with read-only `gh run list/view` (or plain
   `git fetch` until the result commit appears) via
   `scripts/mac_bridge/fetch_results.py`.

Latency profile: ~10 s dispatch + queue + workload runtime. Right for
test/eval/bench cycles (minutes-scale), wrong for interactive debugging —
that is M2's job, not a reason to widen M1's command surface.

### 2.2 Preset allowlist (M1 command surface)

| Preset | What runs on the Mac | Typical use |
| --- | --- | --- |
| `mlx-env-probe` | `backends.mlx.env.probe_environment()` + `distributed.mlx_ring.probe_ring_environment()` (when present on the ref) | "is Metal/mlx healthy, which versions" |
| `mlx-backend-tests` | `pytest tests/backends/mlx/ -q` | real-mlx truth for the fake-mlx Linux suites |
| `integration-tests` | `pytest -m integration tests/integration/ -q` | the v0.3 GA gate on demand |
| `k3-step1-incremental` | hardened Mac harness `--incremental` (n/gen/ctx bounded params) | PR #109 Step-1 decode-only evidence |
| `k3-step2-fused` | hardened Mac harness `--fused-specdecode` | PR #109 Step-2 `blocks>0` evidence |
| `k3-native-baseline` | hardened Mac harness `--native-baseline-bypass` | labelled oracle baseline |
| `k3-evidence-gate` | `scripts/validate_k3_reports.py results/research` | re-validate committed reports on-device |
| `pytest-path` | `pytest <path> -q` with the path validated against a repo-relative allowlisted-prefix rule (`tests/`) | targeted debugging of one test file |

Parameters are **typed and bounded** (`n_samples ≤ 50`,
`max_new_tokens ≤ 512`, `block_size ≤ 16`, paths must resolve under
`tests/`); anything else is rejected at manifest validation, before any
process starts. Machine-local facts (verifier/model paths) come from the
runner's environment (`KAKEYA_MAC_VERIFIER_PATH`, …), never from the
manifest.

### 2.3 M2 — tailnet SSH (designed, needs one secret + one install)

For interactive MLX debugging (lldb, py-spy, Metal captures, REPL):

- Mac: `brew install tailscale`, join the tailnet with `--ssh`
  (Tailscale SSH; respects tailnet ACLs), tag `tag:kakeya-mac`.
- Cloud agent: `TAILSCALE_AUTHKEY` (ephemeral, pre-authorized,
  tag-scoped key) added in Cursor Dashboard → Cloud Agents → Secrets;
  `scripts/mac_bridge/connect_tailscale.sh` brings up `tailscaled` in
  userspace-networking mode and opens `ssh kakeya@kakeya-mac-m4`.
- ACL: the agent-side tag may reach `tag:kakeya-mac:22` only; the Mac
  initiates nothing toward agents. Ephemeral nodes garbage-collect when
  the agent VM dies.

This is the same outbound-only trust shape as the Actions runner (C1),
with per-session ephemeral identity. It is deliberately **not** part of
M1: it requires a secret a fresh clone does not have.

### 2.4 M3 — fleet membership (evaluation in §4)

With the tailnet up, the Mac's Kakeya runtime serves the ADR 0009 gRPC
plane (`CapabilityService` + `ProposerService`, PR #105) and the cloud
agent joins as a fleet peer — capability gossip, placement, and remote
block proposal over the same wire contract used between Mac minis on a
desk LAN.

## 3. Security model (M1)

- **Command surface**: presets only; fixed argv; no manifest string ever
  reaches a shell. `pytest-path` constrains to repo-relative `tests/`.
- **Trigger surface**: anyone with push permission to `mac-bridge/**` —
  identical to the existing surface (any PR labelled `needs-mac-m4`
  already executes its code on the runner via `integration.yaml`). The
  bridge does not widen who can run code on the Mac; it widens *what can
  be conveniently requested* while **narrowing** it to an allowlist.
- **Result integrity**: results are commits on the request branch —
  reviewable, attributable, and evidence-gated before merge anywhere.
- **Resource protection**: `concurrency: mac-bridge` serializes the
  single Mac; per-preset `timeout-minutes`; runs are cancellable from
  the Actions UI.

## 4. Evaluation — folding the bridge into Kakeya distributed inference

The question: should "cloud agent ⇄ kakeya-mac-m4" become a first-class
part of the engine's distributed-inference feature (ADR 0009 / PR #105),
rather than repo tooling?

### 4.1 What maps cleanly

| Bridge concept | ADR 0009 plane concept |
| --- | --- |
| Mac runner with presets | `NodeCapability` with `CAPABILITY_ROLE_TOOL` entries (the enum slot already exists in `distributed.proto`) — e.g. `tool:mlx-eval`, `tool:integration-tests` |
| preset manifest | `ProposeBlock`-style typed request messages (one RPC per tool capability) |
| git-bus branch session | durable async job with attributable artifacts — the property worth **keeping** even after gRPC exists |
| evidence gate on results | the same gate, already shared library code |

The capability model was designed for exactly this shape: the Mac
advertises what it can do; placement picks it; the work request is typed
and the accept/reject of its *output* happens on the consumer side. A
`remote-executor` tool role is a natural, small extension of PR #105 —
the registry, gossip, TTL, and placement code need **zero changes**;
only a new `ModelCapability(role=TOOL)` convention plus one service.

### 4.2 What does not map: WAN data-plane spec decode

The latency budget kills token-level speculative decoding across the
cloud↔desk boundary, and the integration should say so explicitly:

- LAN (two Mac minis, ADR 0009's target): `ProposeBlock` RTT ~0.3–1 ms
  against block compute of tens of ms → negligible overhead. ✔
- WAN (cloud agent ⇄ home/office Mac through a relay): RTT 30–150 ms,
  *per block*. A Gemma-4-26B 4-bit verifier on M4 verifies an 8-token
  block in roughly 50–100 ms — the network would add 30–300 % overhead
  per block, and any acceptance-rate gain is consumed by transport.
  Drafts are latency-critical; **proposer and verifier must share a
  LAN** (or a Thunderbolt ring, per ADR 0009 §2). ✘
- WAN-tolerant flows: capability gossip (seconds-scale TTLs), placement,
  eval/test/bench jobs, artifact return — all fine. ✔

So the correct integration boundary is: **WAN = control plane + tool
plane; LAN = data plane.** This is the same hybrid conclusion as ADR
0009, extended one tier outward.

### 4.3 Recommendation

1. **Keep M1 in-repo now** (this PR): it unblocks PR #109's required Mac
   reruns and all future agent-driven MLX work, with no new secrets.
2. **M2 next**: one Tailscale authkey secret + one Mac install; gives
   interactive debugging and the channel M3 needs. Low effort, high
   leverage.
3. **M3 as a v0.5 roadmap item, scoped**: add a `remote-executor` TOOL
   capability + a small `ToolService` to the ADR 0009 plane so fleet
   nodes (including the Mac) advertise *evaluation* capabilities the
   same way they advertise verifier/proposer roles. Explicitly do
   **not** route spec-decode draft traffic over WAN; placement should
   treat `ring_address`/RTT class as a hard constraint for data-plane
   pairings (a one-line addition to `plan_spec_decode_placement`'s
   candidate filter when WAN nodes appear).
4. mTLS + node identity (already queued for v0.5 GA) becomes a
   prerequisite for M3 leaving the tailnet's closed world.

## 5. One-click install & run

### Mac mini side — one command

```bash
# On the Mac, from the repo root.
# Existing kakeya-mac-m4 host (runner already registered):
bash scripts/mac_bridge/setup_mac.sh

# Fresh Mac (also installs + registers the Actions runner; token from
# GitHub → Settings → Actions → Runners → New self-hosted runner):
bash scripts/mac_bridge/setup_mac.sh \
    --runner-token <TOKEN> --repo-url https://github.com/<owner>/<repo>

# Optionally prepare M2 interactive SSH too:
bash scripts/mac_bridge/setup_mac.sh --with-tailscale
```

The script is idempotent and ends with a bridge self-test; a green exit
means the next `mac-bridge/**` push executes. What it covers: host
shape (arm64 + Python ≥3.12), Python deps (`scripts/setup_mac.sh`),
Actions runner install/registration with the
`[self-hosted, macOS, ARM64, kakeya-mac-m4]` labels, model-location
checks for the `k3-*` presets (with repo-variable instructions when
paths differ), HF-cache pre-warm check, and executor dry-run.

### Cloud agent side — zero install, two commands

The bridge client is stdlib-only: a fresh cloud agent needs **no
configuration** beyond what it already has (repo checkout, git push,
read-only `gh`). Optional: `TAILSCALE_AUTHKEY` in Cursor Dashboard →
Cloud Agents → Secrets enables M2 interactive SSH later.

```bash
# 0. Sanity-check this environment (push rights, gh, bridge files):
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py doctor

# 1. Run on the Mac and wait for results:
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py run \
    --preset mlx-env-probe --wait 600

# Evidence reruns for PR #109 (hardened-harness ref):
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py run \
    --preset k3-step2-fused --ref origin/AgentMemory/v04-mlx-port-incremental-decode-2815 \
    --param n_samples=5 --param max_new_tokens=64 --param block_size=4 --wait 7200

# 2. Check any request later:
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py status \
    --branch <request-branch> --wait 0
```

`kakeya_mac.py run` auto-detects cloud-agent branch policy: on an
`AgentMemory/<name>-<suffix>` checkout it creates the request as
`AgentMemory/mac-bridge-<preset>-<nonce>-<suffix>` (the workflow accepts
both namespaces), so agents never leave their allowed branch template.
After pushing, the client returns the worktree to the original branch.

Lower-level pieces (`request_run.py`, `fetch_results.py`,
`run_preset.py`) remain directly usable; machine-local configuration on
the runner lives in env / repo Actions variables
(`KAKEYA_MAC_VERIFIER_PATH`, `KAKEYA_MAC_DRAFTER_ID`,
`KAKEYA_MAC_FTHETA_DIR` — see `docs/ops/mac-m4-runner-setup.md`).
