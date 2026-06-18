# Skill: Pin a self-hosted runner's Python env (survive reboots, reproducible heavy ML deps)

**Reusable across agents (Claude / Codex / Cursor).** Copy this file or paste the
prompt in the appendix. It is written to be repo-agnostic; the concrete examples
use a GitHub Actions self-hosted Mac runner driving MLX (`mlx_lm`/`torch`/
`transformers`), but the pattern applies to any self-hosted runner (Mac or Linux)
that runs heavy ML/native deps from a virtualenv.

---

## 1. When to use this skill

Trigger it when **a self-hosted runner job fails on a missing module that "used to
work"**, especially after a host **reboot / OS or Python upgrade / runner
re-register**. Classic signatures:

- `ModuleNotFoundError: No module named 'mlx_lm'` (or `torch`, `transformers`, …)
  in a job that previously passed.
- The failure is **fast** (seconds) — it dies at `import`, before any real work.
- A **lightweight probe** (one that only needs stdlib + a base package) still
  passes, proving the runner is *online* but pointing at the **wrong interpreter**.
- The interpreter version changed (e.g. `python=3.14.3` where it used to be
  `3.13.x`), or `pkg=None` for a package that should be installed.

Root cause is almost always: the workflow invokes a **bare `python3`**, and after
the reboot the default `python3` on `PATH` is no longer the venv that has the
deps. The venv still exists; nothing points at it.

---

## 2. Diagnose first (don't guess)

Run the **cheapest possible probe** through the same runner path to read the
interpreter + module state, instead of assuming. Example (adapt the import list):

```bash
python3 - <<'PY'
import sys
def v(m):
    try:
        mod = __import__(m); return getattr(mod, "__version__", "ok")
    except Exception as e:
        return f"MISSING ({e.__class__.__name__})"
print("python =", sys.version.split()[0], "| exe =", sys.executable)
for m in ("mlx", "mlx_lm", "torch", "transformers"):
    print(f"{m} = {v(m)}")
PY
```

Decision rule:
- **Runner online + probe shows wrong `python`/`exe` or `MISSING` deps** → this skill (interpreter pinning).
- **Probe itself never starts (job stuck `queued`/`pending`)** → the runner *agent*
  is down; restart the agent first (different problem).

> In CI-driven runners, route the probe through the same executor the real jobs
> use (so `PATH`/env match). A one-liner like the above, committed as a tiny
> "env-probe" job/preset, is worth keeping permanently.

---

## 3. Fix — three layers (do all three; they are defense-in-depth)

### Layer A — Pin the interpreter the runner *agent* sees (host side, durable)

Make the venv's `bin` the first thing on the **runner agent's** `PATH`, so a bare
`python3` resolves to the venv even across reboots. Pick the mechanism for how the
agent is launched:

- **GitHub Actions runner as a service (recommended).** The runner reads a
  `.env` and a `.path` file in its install dir at start:
  ```bash
  cd ~/actions-runner
  echo "$HOME/kakeya-venv/bin"  > .path          # prepended to PATH
  echo "VIRTUAL_ENV=$HOME/kakeya-venv" >> .env
  ./svc.sh stop && ./svc.sh start                # reload
  ```
  (`.path` is concatenated ahead of the system PATH for every job; `.env` injects
  process env. Both persist across reboots because the service re-reads them.)
- **launchd plist (macOS), if not using `svc.sh`.** In the runner's
  `~/Library/LaunchAgents/<runner>.plist`, set:
  ```xml
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/Users/&lt;you&gt;/kakeya-venv/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  ```
  then `launchctl unload/load` the plist.
- **systemd (Linux self-hosted).** In the runner unit:
  `Environment="PATH=/opt/kakeya-venv/bin:%h/.local/bin:/usr/bin:/bin"`, then
  `systemctl daemon-reload && systemctl restart <runner>`.

Verify: `python3 -c "import mlx_lm, torch, transformers; print('ok')"` from a job.

### Layer B — Make the workflow/executor resolve a *pinned* interpreter (repo side, robust)

Never call a bare `python3` for the heavy job. Resolve an explicit interpreter so
the repo is robust even if Layer A drifts:

1. Add a repo/runner variable, e.g. `KAKEYA_MAC_PYTHON`, pointing at the venv
   python (`/Users/<you>/kakeya-venv/bin/python`). Default-discover if unset:
   ```bash
   PYBIN="${KAKEYA_MAC_PYTHON:-}"
   for c in "$PYBIN" "$HOME/kakeya-venv/bin/python" "$(command -v python3.13)" "$(command -v python3)"; do
     [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import mlx_lm' 2>/dev/null && { PYBIN="$c"; break; }
   done
   ```
2. Use `$PYBIN` (or substitute a `${PYTHON}` token in your command templates)
   instead of `python3` for the actual workload. If your executor spawns argv
   lists (no shell), resolve the token to `$PYBIN` before `subprocess.run`.

### Layer C — Fail fast with a clear message (repo side, observability)

Before the expensive step, assert the deps and **print a fix hint** so the next
failure is self-explanatory instead of a deep `ModuleNotFoundError`:

```bash
"$PYBIN" - <<'PY' || { echo "::error::runner python missing ML deps — see pin-selfhosted-runner-python-env-skill.md (Layer A)"; exit 90; }
import mlx_lm, torch, transformers  # noqa
PY
```

---

## 4. Verify the fix

1. Re-run the lightweight env-probe → correct `python`/`exe`, all deps present.
2. Re-run one **real** (heavy) job → no `ModuleNotFoundError`, completes.
3. **Reboot the host and re-run** (the actual regression you are fixing) → still
   green. This step is the whole point; do not skip it.

---

## 5. Generalizing to a *Cloud Agent* VM env setup (different machine!)

Do **not** confuse the self-hosted runner with the Cloud Agent VM:
- The **Cloud Agent VM** is typically Linux; it runs the *client* that dispatches
  jobs and the unit-test gate. **Mac-only deps (MLX) do not belong there.** Put
  only what the client/tests need into the Cloud Agent env setup (base image +
  startup script), and pin versions.
- The **self-hosted runner** is where the heavy/native/Mac deps live. Pin them
  there (Layers A–C above), not in the Cloud VM env setup.

For the Cloud Agent VM specifically: bake stable deps into the **base image**, do
slow-changing installs in the **startup script**, and pin versions so a new VM is
reproducible. (In Cursor, this is the "env setup agent" config.)

---

## 6. Anti-patterns

- ❌ `pip install` the missing dep into whatever `python3` happens to be active
  (often a too-new system Python with no wheels for `torch`/`mlx_lm`). Pin to the
  known-good venv instead.
- ❌ Hardcoding an absolute interpreter path in many places. Resolve once
  (variable + discovery) and reuse.
- ❌ "It works now" without a reboot test — the regression is reboot-triggered.
- ❌ Relying on an interactive shell's `source venv/bin/activate`; CI jobs and
  services don't run your `.zshrc`.

---

## Appendix — ready-to-paste prompt for a setup agent

> **Task: make our self-hosted CI runner's Python environment reboot-proof.**
>
> Symptom: jobs on our self-hosted runner fail fast with
> `ModuleNotFoundError: No module named 'mlx_lm'` after the host rebooted; a
> lightweight env-probe shows the runner's default `python3` switched to a newer
> interpreter that lacks our ML stack (`mlx_lm`/`torch`/`transformers`), while the
> known-good venv still exists but is no longer on `PATH`.
>
> Do all of the following, smallest-diff first, and verify each:
> 1. **Diagnose:** run a tiny probe that prints `sys.version`, `sys.executable`,
>    and import status of `mlx_lm, torch, transformers` through the same path the
>    real jobs use. Confirm the wrong interpreter / missing modules.
> 2. **Host (runner agent):** pin the venv's `bin` ahead of system `PATH` for the
>    runner service so a bare `python3` resolves to the venv across reboots — via
>    the runner's `.path`/`.env` files (GitHub Actions `svc.sh`), or the
>    launchd/systemd unit's `PATH` env. Reload the service.
> 3. **Repo (workflow/executor):** stop calling bare `python3` for the heavy job.
>    Resolve a pinned interpreter from a `*_PYTHON` repo/runner variable, with a
>    discovery fallback that picks the first candidate where `import mlx_lm`
>    succeeds; use it for the workload commands.
> 4. **Repo (fail-fast):** before the expensive step, assert
>    `import mlx_lm, torch, transformers` and emit a clear `::error::` with a link
>    to this skill if missing (exit non-zero).
> 5. **Verify, including a reboot:** env-probe green, one real heavy job green,
>    then reboot the host and re-run the same job — must still be green.
> 6. **Pin versions** in the venv (freeze a lockfile) and document the venv path +
>    rebuild steps so the environment is reproducible, not just patched.
>
> Keep the heavy/native deps on the self-hosted runner only; do NOT add Mac-only
> deps to the Cloud Agent (Linux) VM env setup.
