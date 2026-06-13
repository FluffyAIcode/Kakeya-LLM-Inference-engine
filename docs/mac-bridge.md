# The Mac bridge — a soft link between a cloud agent and a local Mac mini M4

This is the canonical, reader-friendly guide to how a **Linux cloud agent**
(no Apple Silicon, no Metal) drives a **local Mac mini M4** to run everything
MLX-dependent — the MLX verifier, the K3 spec-decode harness, the evidence-gate
reruns — without any inbound network path to the Mac.

It is a "soft link" in the literal sense: there is **no socket, no SSH, no VPN,
no tunnel** between the two machines. The link is built entirely out of things a
cloud agent already has — a git checkout, git push permission, and a read-only
`gh` — plus the Mac's existing outbound-only GitHub Actions runner. **Git is the
wire.**

- **Deep-dive design** (transports M1/M2/M3, fleet-integration evaluation):
  [`docs/design/mac-bridge-cloud-agent-access.md`](design/mac-bridge-cloud-agent-access.md)
- **Runner operator setup** (register runner, model paths, HF pre-warm):
  [`docs/ops/mac-m4-runner-setup.md`](ops/mac-m4-runner-setup.md)
- **Implementation**: [`scripts/mac_bridge/`](../scripts/mac_bridge/),
  [`inference_engine/bridge/manifest.py`](../inference_engine/bridge/manifest.py),
  [`.github/workflows/mac-bridge.yaml`](../.github/workflows/mac-bridge.yaml)

---

## 1. Why a soft link (the constraints)

| # | Constraint | Consequence |
| --- | --- | --- |
| C1 | **No inbound path to the Mac.** It sits behind NAT, no public IP, no port-forward. | The transport must be initiated *from* the Mac, or relayed. The Mac's Actions runner already long-polls GitHub outbound-only — reuse that. |
| C2 | **Cloud agents are ephemeral and git-native.** They reliably have a repo checkout, git push, read-only `gh`. They do **not** reliably have VPN keys, SSH keys, or workflow-dispatch rights. | The control channel must be plain git. Zero new secrets. |
| C3 | **The Mac executes whatever lands on it.** A queue that forwards arbitrary shell to a desk machine is a remote-shell backdoor. | The command surface must be a typed **allowlist**, never free-form shell. |
| C4 | **Evidence discipline.** Benchmark/eval results must pass the K3 evidence gate, not route around it. | The gate runs **on the Mac**, so a non-conforming report fails the bridge run itself. |

The git-bus bridge is the only transport that satisfies **C1 + C2 with zero new
infrastructure**: git is the RPC bus, the Actions runner is the executor, and the
request branch is the session.

## 2. Architecture at a glance

```
   Linux cloud agent (no Metal)                 GitHub                    Mac mini M4 (kakeya-mac-m4)
   ───────────────────────────                 ────────                  ───────────────────────────
   kakeya_mac.py run                                                      self-hosted Actions runner
        │  --preset <name> --param k=v                                    [self-hosted, macOS, ARM64,
        │                                                                  kakeya-mac-m4], outbound-only
        ▼
   request_run.py
     • branch AgentMemory/mac-bridge-<preset>-<nonce>-<suffix>
     • write .mac-bridge/request.json  (the manifest)
     • git push  ───────────────────────►  push event
                                                │  on: push
                                                │  branches: mac-bridge/** ,
                                                │            AgentMemory/mac-bridge-*
                                                ▼
                                          mac-bridge.yaml  ──long-poll──►  runner picks up the job
                                          concurrency: mac-bridge                  │
                                          (one Mac, never cancel)                  ▼
                                                                          run_preset.py --manifest …
                                                                            • validate vs PRESET allowlist
                                                                              (typed, bounded params)
                                                                            • execute the preset's FIXED argv
                                                                              (no manifest string hits a shell)
                                                                            • K3 reports → evidence gate ON-device
                                                                                         │
                                          push results  ◄──────────────────────────────┘
                                          (commit .mac-bridge/logs/ +
                                           results/research/*.json back
                                           to the SAME branch; also
                                           upload as run artifacts)
        ▲                                       │
        │  git fetch / gh run view (read-only)  │
   fetch_results.py  ◄───────────────────────────
     • poll until the result commit appears
     • read logs + result JSON from the branch
```

Latency profile: **~10 s dispatch + queue + workload runtime**. This is the right
shape for test / eval / bench cycles (minutes-scale); it is deliberately *not* an
interactive debugger (that is the queued M2 tailnet-SSH transport — see the design
doc). The fix for "too slow" is never to widen the command surface.

## 3. The five components

| Component | Where | Role |
| --- | --- | --- |
| **Client** (`kakeya_mac.py`, `request_run.py`, `fetch_results.py`) | cloud agent | stdlib-only front door: `doctor` / `run` / `status`. Branches, writes the manifest, pushes, polls, returns the worktree to the original branch. |
| **Manifest + allowlist** (`inference_engine/bridge/manifest.py`) | shared library | The security contract: the set of allowed presets, their fixed argv templates, and the typed/bounded params. Unit-tested at 100 % on the Linux CI gate. |
| **Request manifest** (`.mac-bridge/request.json`) | the request branch | The serialized RPC: preset name + validated params + a nonce. |
| **Executor** (`mac-bridge.yaml` → `run_preset.py`) | Mac runner | Triggered by the push; materializes LFS checkpoints, validates the manifest against the allowlist, runs the preset's fixed argv, evidence-gates K3 reports, commits results back. |
| **Runner** (`[self-hosted, macOS, ARM64, kakeya-mac-m4]`) | the Mac mini | The Apple-Silicon executor. Outbound-only; the same trust shape as any self-hosted Actions runner. |

## 4. Security model

- **Command surface = the preset allowlist.** Each preset is a *fixed* argv list;
  no user-controlled manifest string is ever interpolated into a shell. `pytest-path`
  is the only path-taking preset and it constrains the path to a repo-relative
  `tests/` prefix. Params are **typed and bounded** at manifest-validation time —
  before any process starts:
  - `n_samples ≤ 50`, `max_new_tokens ≤ 512`, `block_size ≤ 16`.
  Machine-local facts (verifier/model paths) come from the **runner's environment**
  (`KAKEYA_MAC_VERIFIER_PATH`, `KAKEYA_MAC_DRAFTER_ID`, `KAKEYA_MAC_FTHETA_DIR`),
  never from the manifest.
- **Trigger surface = push permission on `mac-bridge/**` (or `AgentMemory/mac-bridge-*`).**
  This is the *same* population that can already run code on the runner via the
  `needs-mac-m4` PR label (`integration.yaml`). The bridge does **not** widen *who*
  can run code on the Mac — it widens *what can be conveniently requested* while
  **narrowing** it to an allowlist.
- **Result integrity.** Results are commits on the request branch — reviewable,
  attributable, and evidence-gated before they merge anywhere.
- **Resource protection.** `concurrency: mac-bridge` serializes the single Mac
  (never cancels a running job); each preset carries its own `timeout-minutes`,
  with a hard job cap (150 min) above it.

## 5. Quickstart

### Cloud-agent side — zero install, two commands

The client is stdlib-only; a fresh cloud agent needs no configuration beyond what
it already has (repo checkout, git push, read-only `gh`).

```bash
# 0. Sanity-check THIS environment (push rights, gh, bridge files):
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py doctor

# 1. Run a preset on the Mac and wait for the result:
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py run \
    --preset mlx-env-probe --wait 600

# 1b. A bench preset with bounded params:
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py run \
    --preset k3-beta-scorecard \
    --param n_samples=5 --param max_new_tokens=32 --param block_size=8 --wait 5400

# 2. Poll an earlier request later:
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py status \
    --branch <request-branch>
```

`run` auto-detects the cloud-agent branch policy: on an `AgentMemory/<name>-<suffix>`
checkout it creates the request branch as `AgentMemory/mac-bridge-<preset>-<nonce>-<suffix>`
(the workflow accepts both namespaces), so the agent never leaves its allowed
branch template. After pushing it returns the worktree to the original branch.

### Mac-mini side — one command (operator, one-time)

```bash
# Existing kakeya-mac-m4 host (runner already registered):
bash scripts/mac_bridge/setup_mac.sh

# Fresh Mac (also installs + registers the Actions runner):
bash scripts/mac_bridge/setup_mac.sh \
    --runner-token <TOKEN> --repo-url https://github.com/<owner>/<repo>
```

The script is idempotent and ends with a bridge self-test; a green exit means the
next `mac-bridge/**` push executes. Full operator details (labels, model symlink
locations, HF pre-warm, repo Actions variables) are in
[`docs/ops/mac-m4-runner-setup.md`](ops/mac-m4-runner-setup.md).

## 6. Current preset allowlist

Authoritative source: `inference_engine/bridge/manifest.py` (the list below is a
snapshot; the manifest and its unit tests are the contract).

| Preset | What runs on the Mac |
| --- | --- |
| `mlx-env-probe` | MLX/Metal environment + ring probe — "is Apple Silicon ML healthy, which versions" |
| `mlx-backend-tests` | `pytest tests/backends/mlx/` — real-MLX truth for the fake-MLX Linux suites |
| `integration-tests` | `pytest -m integration` — the v0.3 GA gate on demand |
| `pytest-path` | `pytest <path>` with the path validated against a `tests/` prefix rule |
| `k3-step1-incremental` | hardened Mac harness `--incremental` (Gap-A incremental decode evidence) |
| `k3-step2-fused` / `k3-step2-fused-allmlx` | hardened Mac harness `--fused-specdecode` (fused spec-decode evidence) |
| `k3-fused-allmlx-code` / `k3-fused-allmlx-code-trim` | all-MLX fused spec-decode on a code workload (CUDA-trim variant) |
| `k3-fused-allmlx-natural` | all-MLX fused, natural stop (NIAH) |
| `k3-fused-singlefused-probe` | single-fused-graph Metal-stability probe |
| `k3-beta-scorecard` | NIAH ctx280 all-MLX fused + CUDA-trim — Kakeya-vs-MLX-only scorecard |
| `k3-native-baseline` | labelled native-MLX AR oracle baseline |
| `k3-kv-quant-eval` | 4-bit KV-quantization evaluation |
| `k3-drafter-parity` / `k3-drafter-parity-fp32` | drafter port-fidelity / acceptance parity checks |
| `k3-evidence-gate` | re-validate committed K3 reports on-device |

## 7. Where this is heading

The git-bus bridge is **M1** of a three-tier plan. M2 adds an optional tailnet-SSH
transport for *interactive* MLX debugging (one Tailscale authkey secret), and M3
folds the Mac into the ADR 0009 multi-host capability plane as a `remote-executor`
TOOL capability. A key boundary, fixed by the latency analysis in the design doc:
**WAN = control + tool plane; LAN = data plane** — token-level spec-decode drafts
are latency-critical and must never cross the cloud↔desk boundary. See
[`docs/design/mac-bridge-cloud-agent-access.md`](design/mac-bridge-cloud-agent-access.md) §4.
