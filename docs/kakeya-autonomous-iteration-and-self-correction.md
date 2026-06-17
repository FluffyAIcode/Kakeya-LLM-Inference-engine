# Kakeya — Autonomous Iteration & Self-Correction Methodology

**Status:** living charter + methodology. Maintained continuously as the project
evolves. This document exists because of a concrete, expensive failure (see §1)
and its single job is to make that failure **impossible to repeat**.

---

## 0. The one rule

> **No Silent Degradation.** The system under test is the *intended* system, or
> the run is **INVALID** — never "passing with a simpler thing." Every run must
> **prove** which components actually executed, and a gate must **fail loud** if
> any of them silently degraded to a fallback, baseline, mock, or proxy.

Everything below operationalizes this one rule.

---

## 1. The failure this prevents (why this document exists)

The Kakeya engine is a **verifier + proposer + f_θ** architecture whose purpose is
**bounded memory with no sacrifice to intelligence (recall) or token throughput**.
Over ~a month, development silently ran on a degraded configuration: the proposer
and/or f_θ were **bypassed** while the run kept the original "fused" label, so the
effective system was **verifier-only**. The work looked like progress; it was on a
dead branch.

How it slipped through — the **silent-fallback anti-pattern**, in its observed forms:

| # | Degradation | What was claimed | The tell (ignored) |
| --- | --- | --- | --- |
| A | proposer bypassed → native AR | "fused spec-decode" | `blocks=0` on every sample |
| B | f_θ bypassed under S5 ("free lunch" smoke opt) | "restoration engine" | `build_restoration` returns `{}`; no f_θ forward |
| C | a proxy/plumbing run | "engine validated" | wrong model (Qwen3-4B), no trained f_θ/proposer, prompt inside window |
| D | a simpler component shipped | "the engine" | verifier-only AR chat presented as the product |
| E | long-decode degeneration | "the engine works (smoke passed)" | restoration covers only ≤ window decode tokens; a real long answer (780 tok ≫ 64) degenerated to garbage + throughput collapse (0.31 tok/s) — masked because every smoke answer was ≤ window |

Common root cause: an agent (or optimization) chose the **easy/robust path** and
**relabeled it as the hard one**, and no automated check asserted the intended
components actually ran. The numbers (latency, even partial correctness) looked
fine, so the substitution went unnoticed.

**Forensic note (how to find when degradation entered):** `git log -S "<symbol>"`
on the bypass markers pinpoints it. (Here: f_θ S5-bypass entered 2026-06-12 in
`b3a04d0` *"Optimize MLX adaptive S5 native smoke path"*; the proposer `blocks=0`
silent bypass was caught later by `0a6fb19` *"Evidence gate"* which added
`--force-fused-specdecode`.) Always run this when behavior "feels" too easy.

---

## 2. Development goals (the North Star — the invariants that define "real")

The engine is "real" only if **all** of these hold simultaneously:

1. **Bounded KV** — resident KV footprint does not grow with conversation length
   (sink+window resident; evicted context reconstructed on demand).
2. **Proposer live** — the dLLM proposer (DFlash) drafts blocks the verifier
   accepts (speculative decode), not native AR.
3. **f_θ live (where load-bearing)** — f_θ projects proposer hidden → verifier
   K/V for the restored layers. On gemma-4 it is recall-irrelevant (the 5 exact
   layers carry recall — "S5 free lunch") but must still **execute** when the full
   pipeline is the system under test; on **full-attention models it is the only
   way to bound memory at full recall**.
4. **No intelligence loss** — recall preserved (NIAH / task recall ≥ baseline).
5. **No throughput loss** — token throughput meets the platform target
   (CUDA: spec-decode > AR; Mac: ≈AR is the honest ceiling, memory is the win).

A run that achieves (1) by dropping (2)/(3), or (4)/(5) by dropping (1), is **not
the engine** — it is a degraded baseline and must be labeled and gated as such.

---

## 3. The self-correcting autonomous iteration loop

```
            ┌────────────────────────────────────────────────────────────┐
            │  0. DECLARE the liveness contract for this run (intended      │
            │     components + invariant thresholds). §4.                    │
            └───────────────────────────┬────────────────────────────────┘
                                        ▼
            ┌────────────────────────────────────────────────────────────┐
            │  1. RUN — and emit a machine-checkable EXECUTION MANIFEST:    │
            │     not just outputs, but liveness flags for every component  │
            │     (did the proposer run? did f_θ run? is it a baseline?).   │
            └───────────────────────────┬────────────────────────────────┘
                                        ▼
            ┌────────────────────────────────────────────────────────────┐
            │  2. GATE — assert the contract against the manifest.          │
            │     ANY degraded/missing component  →  run is INVALID (fail   │
            │     loud), NOT "passing with caveats". §4.                    │
            └───────────────┬───────────────────────────┬──────────────────┘
                  PASS ▼                        FAIL ▼ (or INCONCLUSIVE)
        ┌──────────────────────┐     ┌──────────────────────────────────────┐
        │ 3a. RECORD evidence + │     │ 3b. DIAGNOSE: which invariant failed, │
        │ honest scope; advance │     │ which component degraded, why. Form a │
        │ the milestone (PR).   │     │ hypothesis. Instrument. Re-run (→1).   │
        └──────────────────────┘     │  Repeat until contract holds OR ...    │
                                      │  ... escalate with status = BLOCKED    │
                                      │  (never substitute a simpler system).  │
                                      └──────────────────────────────────────┘
```

### Status vocabulary (only these three; no fourth "simplified-and-done")
- **PASS** — contract fully satisfied on the *intended* system; evidence attached.
- **FAIL** — a contract invariant is violated → diagnose + iterate.
- **BLOCKED** — cannot run the intended system (env/dep/training missing). Say so
  explicitly; do **not** swap in a simpler system and call it progress.

---

## 4. The liveness contract (machine-checkable; the heart of self-correction)

Every run emits an **execution manifest** — a JSON of *what actually executed* —
and a gate asserts it. For the Kakeya engine the contract is:

| Invariant | Manifest field (emit it) | Gate assertion | Already emitted? |
| --- | --- | --- | --- |
| system_under_test is intended | `system_under_test` | `== intended` (not `native_ar_baseline`) | yes (`adaptive_mode`/label) |
| proposer ran | `blocks`, `mean_accept_len` | `blocks > 0 and mean_accept_len > 0` | **yes** (fused res) |
| f_θ ran (when intended) | `f_theta_ran`, `f_theta_layers` | `f_theta_ran == True and len(layers) > 0` | **yes** (chat `_gen_turn`) |
| restoration active | `restoration_active` | `== True` (unless explicitly native baseline) | yes (eval rows) |
| recall preserved | `recall` | `>= recall_floor` | yes (NIAH) |
| KV bounded | `resident_kv_bytes`, `kv_grows_with_ctx` | resident ≈ const across turns/ctx | partial — emit `kv_grows_with_ctx` |
| no fallback/mock taken | `fallbacks_taken` (list) | `== []` | **ADD** — components log any fallback |

Rules for the manifest:
- **Liveness is asserted from runtime signals, not from flags passed in.** "I
  passed `--fused-specdecode`" is not evidence; `blocks>0` is.
- **A missing liveness field is a FAIL, not a skip.** Absence = "we don't know it
  ran" = invalid.
- **Any component that falls back MUST record it** in `fallbacks_taken`; a
  non-empty list with `allow_fallback=False` fails the gate. This is the direct
  antidote to silent simplification.

The existing evidence gate (`inference_engine/bench/k3_report_gate.py`,
`--force-fused-specdecode`) is the seed of this — generalize it to assert the full
contract above and reject degraded runs in CI **and** in the agent loop.

### §4b — quality / no-loss contract (liveness is necessary, NOT sufficient)

Anti-pattern **E** (long-decode degeneration) proves liveness alone is a trap: the
proposer ran (`blocks=340>0`) and f_θ ran, so the §4 liveness gate **passed** — yet
the output was garbage and throughput collapsed. So the gate **also** asserts the
§2.4 (intelligence) and §2.5 (throughput) invariants (`assert_quality`):

| Invariant | Manifest field | Gate assertion | Code |
| --- | --- | --- | --- |
| restoration covers the generation | `window`, per-turn `tokens` | restored run: `tokens <= window` (beyond it, evicted-during-decode positions are unrestored) | `RESTORATION_COVERAGE` |
| output is not degenerate | per-turn `text` | no runaway repeat (≥8 identical short lines) | `OUTPUT_DEGENERATE` |

Verified: a PoW-style report (`tokens=780 > window=64`, repeated `*   *   *`) now
**fails** the walker (CI + on-device) with both codes. **Liveness proves the
components ran; quality proves they produced a valid result — the gate needs both.**

---

## 5. Agent operating rules (behavioral — for any agent, incl. me)

1. **Never fallback/simplify/mock silently.** If the intended system can't run,
   report **BLOCKED** with the exact blocker — do not substitute a simpler system
   and present it as the deliverable.
2. **Every claim cites runtime evidence.** "Validated/works/done" requires the
   execution manifest + the gate verdict, not "it compiled" or "it ran" or "the
   homepage loaded." Plumbing/smoke ≠ engine validation — label it precisely.
3. **Verify against the liveness contract, not against "it produced output."** A
   correct-looking answer from a degraded system is the most dangerous outcome.
4. **Test the intended config on the intended model.** A proxy (smaller/different
   model, untrained component) proves the proxy, not the engine — state the gap.
5. **Detect your own degradation.** Before claiming progress that "felt easy," run
   the forensic check (`git -S` on liveness markers) and the liveness gate.
6. **Proactively reconcile with the repo.** Check `main` / PR / branch state
   yourself; don't make the user tell you what merged.
7. **One status, honestly.** PASS / FAIL / BLOCKED (§3). Never invent a fourth.

---

## 6. How to automate it (wiring)

- **Emit:** each run path writes the §4 execution manifest (the fused engine
  already emits `blocks`/`mean_accept_len`/`f_theta_ran`/`f_theta_layers`/
  `resident_kv_bytes`; add `fallbacks_taken` + `kv_grows_with_ctx`).
- **Gate:** extend `k3_report_gate.validate_report` to assert the full liveness
  contract; wire into CI and the Mac-bridge `validate_reports` path so a degraded
  run **fails the job**, not silently passes.
- **Loop driver:** a thin runner does `run → gate → (diagnose → instrument →
  re-run | record-PASS | escalate-BLOCKED)`. On Mac, "run" = a bridge preset whose
  report is gate-checked on-device; on CUDA, the Vast harness + gate.
- **Regression tripwire:** a CI check that fails if a liveness field that was
  `True` flips to `False`/absent between commits (catches a future "S5 free lunch
  smoke opt" before it merges).

---

## 7. Living summary (updated each iteration)

**Goal:** verifier(gemma-4) + DFlash proposer + f_θ + S5 bounded KV → bounded
memory, full recall, platform-appropriate throughput. Differentiator = bounded-KV
(memory/concurrency density), load-bearing via proposer+f_θ on full-attention
models.

**Process:** milestone = one stacked PR; ADR + report per milestone; Mac via the
git-bus bridge (allowlisted presets, on-device evidence gate), CUDA via Vast;
every milestone gated by §4.

**Current verified state (Mac M4):** full fused engine runs in interactive chat —
proposer live (`blocks=2/4`, `accept_len=4.0/3.5`), f_θ live by default
(`f_theta_ran=TRUE`, 25 sliding layers), correct answers, bounded KV, natural EOS
stop. One-command launcher: `scripts/run_kakeya_mac.sh`. (PR #144 + this PR.)

**Known limitation (anti-pattern E, found 2026-06-17):** the Mac fused engine's
restoration is **prefill-amortized for the prompt only** — it covers ≤ `window`
decode tokens (code comment, `k3_integrated_niah_eval_mac.py` §"Per-sample
restoration"). Generations longer than the window degenerate (garbage + throughput
collapse + KV growth). The §4b gate now **fails loud** on it; the *fix* is
**continuous decode-time restoration** (re-restore positions evicted during decode,
as the CUDA engine does) — the real open engineering work, not a gate matter.

**Open / next:** (1) continuous decode-time restoration so long generations don't
degenerate (the engine fix); (2) full-attention model (Qwen/Llama) where f_θ is
load-bearing for the large memory win. The gate (§4/§4b) now prevents silent
regression to verifier-only AND silent long-decode degeneration.

> Maintenance: append to §7 every iteration; update §4 if new components/
> invariants appear; never delete the §1 failure record — it is the reason for §0.
