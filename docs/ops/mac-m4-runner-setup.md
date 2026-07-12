# Mac M4 self-hosted runner setup

This runner backs the **Integration (Mac M4)** and **Mac bridge** workflows.
Integration is triggered directly by runtime/model/proto/integration path
changes; it does not depend on a label being applied by another workflow.

## Hardware requirements

| Resource | Minimum |
| --- | --- |
| Chip | Apple Silicon (M-series); M4 or newer recommended |
| Unified memory | 24 GB (16 GB works for Qwen3-0.6B alone but no headroom for concurrent work) |
| Free disk | ~50 GB (HF cache + venv + checkout history) |
| Network | Reachable to github.com for runner registration; outbound to HF Hub for the one-time pre-warm |
| OS | macOS 14 (Sonoma) or newer |

## One-time setup

### 1. Register the self-hosted runner

Follow [GitHub's docs](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners) to add a runner to the repository:

1. Repository → Settings → Actions → Runners → New self-hosted runner.
2. Choose macOS / ARM64.
3. Run the install + configure commands GitHub provides.
4. **Important**: when prompted for labels, add `kakeya-mac-m4`
   in addition to the default `self-hosted, macOS, ARM64`. The
   workflow's `runs-on:` clause specifically requires that label.
5. Run the runner as a launchd service (`./svc.sh install && ./svc.sh start`)
   so it survives reboots.

### 2. Pre-warm the HF cache

The integration workflow runs with `HF_HUB_OFFLINE=1` so it never
hits HuggingFace at test time (avoids 90-min runs blocking on a 4 GB
download). Pre-warm the cache once per runner:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
"
```

The model lands at `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/`.
The workflow's "Verify Qwen3-0.6B in HF cache" step fails fast with
a clear error if that directory is missing.

If a future test adds a new model id, update the pre-warm command
(and the workflow's verify step) accordingly.

### 3. Install Python toolchain

The runner needs Python 3.12+. Use Homebrew or pyenv:

```bash
brew install python@3.12
# or:
pyenv install 3.12.7
pyenv global 3.12.7
```

Confirm `python3 --version` returns 3.12.x and `python3 -c 'import platform; print(platform.machine())'` returns `arm64`.

### 4. Pin the MLX workload venv

Set `KAKEYA_MAC_PYTHON` in the runner service environment to the long-lived
venv that imports `mlx_lm`, `torch` and `pytest`. Real MLX gates resolve this
interpreter before the legacy Qwen integration environment is installed.

## Runtime expectations

| Phase | Wall time on M4 24 GB |
| --- | --- |
| Checkout + verify host | <5 s |
| Verify HF cache | <1 s |
| `pip install -e .` (warm pip) | 20-40 s |
| `pytest -m integration` (80 tests, post-PR-N1..N4) | 60-120 s |
| Artifact upload | <5 s |
| **Total** | **~2-3 min** |

The 90-minute timeout in the workflow is a safety margin. A run
that exceeds 5 min should be investigated — likely a model-load
regression or a runaway test.

## Maintenance

### Cache hygiene

The runner's HF cache and pip cache grow over time. Recommend a
monthly cron:

```bash
# ~/clean-kakeya-runner.sh
find ~/.cache/huggingface/hub -type d -mtime +60 -prune -name 'models--*' -exec rm -rf {} +
python3 -m pip cache purge
```

The Qwen3-0.6B cache is touched on every run, so `mtime +60` only
prunes models added by future test additions that aren't currently
exercised.

### Runner upgrades

GitHub publishes new runner versions ~monthly. Update via:

```bash
cd ~/actions-runner
./svc.sh stop
./config.sh remove --token <repo-config-token>
# download the new tarball per GitHub UI instructions
./config.sh --url https://github.com/<owner>/<repo> --token <new-token>
./svc.sh install && ./svc.sh start
```

### Failure triage

Workflow failures are visible at `Actions → Integration (Mac M4)`. The "Surface failure summary" step inlines the test names + first-line error messages so triage doesn't require downloading the JUnit XML.

If the runner itself is offline (multiple differently-labelled jobs remain
`queued` and no first step starts), check on the Mac:

```bash
cd ~/actions-runner
sudo ./svc.sh status
tail -200 ~/Library/Logs/actions-runner/Runner_*.log
bash /path/to/repo/scripts/mac_bridge/runner_healthcheck.sh
```

Common causes:
- macOS auto-update rebooted the host; service didn't auto-start (rare with `launchd` but possible).
- HF cache was purged; the verify step fails. Re-warm.
- Disk full from accumulated pip downloads; clear cache.

### Headless reboot recovery

A user LaunchAgent runs only after GUI login. For an unattended Mac mini,
install the system watchdog once so the runner recovers before login:

```bash
cd /path/to/Kakeya-LLM-Inference-engine
sudo "$HOME/actions-runner/svc.sh" install 2>/dev/null || true
sudo "$HOME/actions-runner/svc.sh" start 2>/dev/null || true
bash scripts/mac_bridge/install_autorecover_launchagent.sh --system
bash scripts/mac_bridge/recover_runner_after_reboot.sh
```

For jobs that are already queued, the last command is the immediate recovery
action. GitHub assigns queued jobs automatically once `Runner.Listener`
reconnects. The system LaunchDaemon retries every 60 seconds and does not
require a logged-in desktop session.

## Mac bridge (cloud-agent access)

The same runner also serves the **Mac bridge**
(`.github/workflows/mac-bridge.yaml`): pushes to `mac-bridge/**`
branches execute an allowlisted preset (see
`inference_engine/bridge/manifest.py`) and commit logs/results back to
the request branch. Full protocol + security model:
`docs/design/mac-bridge-cloud-agent-access.md`.

**One-click setup**: on the Mac, from the repo root —
`bash scripts/mac_bridge/setup_mac.sh` (add
`--runner-token <TOKEN> --repo-url <URL>` on a fresh machine to install
and register the Actions runner too). The script is idempotent and ends
with a bridge self-test.

Operator setup beyond the standard runner install:

1. **Model locations** (used by the `k3-*` harness presets) are read
   from the environment / repo Actions variables, never from the
   request manifest. Set repo variables (Settings → Secrets and
   variables → Actions → Variables) when the on-disk layout differs
   from the defaults:
   - `KAKEYA_MAC_VERIFIER_PATH` — MLX 4-bit Gemma-4 verifier directory
   - `KAKEYA_MAC_DRAFTER_ID` — DFlash drafter HF id or local path
   - `KAKEYA_MAC_FTHETA_DIR` — trained f_θ checkpoint directory
2. Bridge runs are serialized (`concurrency: mac-bridge`) and capped at
   150 minutes; cancel stuck runs from the Actions UI.
3. K3 acceptance reports produced by bridge runs are validated by the
   evidence gate on this machine; a non-conforming report fails the
   bridge run (exit ≠ 0) by design.
