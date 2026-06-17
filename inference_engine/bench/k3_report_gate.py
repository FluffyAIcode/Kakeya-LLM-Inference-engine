"""K3 Mac evidence gate — machine-checkable report constraints.

Born from the PR #109 review of the CUDA→MLX port evidence. The
committed Mac reports exhibited four failure modes that a human had to
catch by reading raw JSON:

1. Reports labelled ``free_gen_fused_specdecode`` in which the fused
   spec-decode engine executed **zero blocks** (silent greedy
   fallback) — the system under test never ran.
2. A "cross vs oracle" speedup (2.584×) where the cross arm was the
   **native-cache baseline itself** (adaptive bypass), i.e. the system
   was compared against itself and the ratio was run-order noise
   (oracle prefill varied 35–146 s for the identical computation).
3. Headline throughput derived from n=1 / gen=8 smokes whose wall
   time was ~95 % prefill.
4. An analytical S5 memory table (``sink_window=68``) attached to a
   run that actually used the un-trimmed native cache.

Every one of those is now a hard, mechanical rule. The Mac harness
(``scripts/research/k3_integrated_niah_eval_mac.py``) runs
:func:`validate_report` on its own output and exits non-zero on
violation; CI (``scripts/validate_k3_reports.py``) re-validates every
committed report so non-conforming evidence cannot land on a branch
silently. Reports with ``schema_version < 2`` predate the gate and are
grandfathered as **non-evidence** (CI prints them as legacy warnings).

This module is deliberately dependency-free (stdlib only) so the CI
step and the Mac harness share the exact same rule implementation.

Rule codes
----------

================================  ============================================
``LEGACY_SCHEMA``                 schema_version < 2: pre-gate report,
                                  grandfathered, never citable as evidence.
``MISSING_STAGE_TIMINGS``         cross arm has no per-sample stage rows.
``MISSING_RESTORATION_FLAG``      a cross stage row lacks ``restoration_active``.
``MIXED_RESTORATION_PATHS``       cross samples mix restored and native paths.
``BASELINE_AS_SUT``               a native-baseline run occupies the
                                  system-under-test slot without declaring
                                  ``system_under_test = "native_ar_baseline"``.
``BASELINE_RECALL_CLAIM``         a native-baseline run claims
                                  ``gate.recall_cross_model``.
``RECALL_SCOPE``                  ``recall_cross_model`` claimed but not every
                                  cross sample ran with restoration active.
``FUSED_NEVER_RAN``               fused eval_mode with zero executed blocks.
``SPEEDUP_SELF_COMPARISON``       speedup claimed on a native-baseline run.
``SPEEDUP_SAMPLES``               speedup claimed with < MIN_PERF_SAMPLES.
``SPEEDUP_DECODE_TOKENS``         speedup claimed with median decode tokens
                                  < MIN_MEDIAN_DECODE_TOKENS (prefill-dominated).
``SPEEDUP_DECODE_ONLY_MISSING``   headline speedup without a decode-only
                                  median comparison alongside it.
``SPEEDUP_SCOPE_MISMATCH``        cross and oracle timing scopes differ.
``SPEEDUP_ORACLE_LOOP``           oracle decode loop is not ``generate_step``
                                  (hand-rolled per-token ``mx.eval`` baselines
                                  are the documented MLX anti-pattern).
``SPEEDUP_PREFILL_VARIANCE``      within-arm prefill spread exceeds
                                  MAX_PREFILL_SPREAD; e2e ratios are noise.
``MEMORY_CLAIM_MISMATCH``         memory savings claimed from an analytical
                                  formula that does not describe the run.
================================  ============================================
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

MAC_REPORT_KIND = "k3_integrated_niah_acceptance_mac"

# Reports older than this schema predate the gate: grandfathered,
# treated as non-evidence (warning, not failure) by the CI walker.
GATED_SCHEMA_VERSION = 2

# Minimum statistical strength for any cross-vs-oracle speedup claim.
MIN_PERF_SAMPLES = 5
MIN_MEDIAN_DECODE_TOKENS = 32

# Max allowed (max/min) prefill wall-time spread within one arm before
# an e2e throughput ratio is ruled noise. The ctx280 report that
# motivated the gate showed a 4.1× spread on the oracle arm.
MAX_PREFILL_SPREAD = 3.0

# The only oracle decode loop admissible for a headline speedup: the
# async-pipelined mlx_lm primitive. Per-token ``mx.eval`` loops are the
# anti-pattern documented in docs/mlx-port-lessons.md.
CLAIM_ORACLE_DECODE_LOOP = "generate_step"

NATIVE_BASELINE_LABEL = "native_ar_baseline"

# §4 liveness contract (docs/kakeya-autonomous-iteration-and-self-correction.md):
# report kinds that carry per-turn component-liveness signals. The gate proves
# the INTENDED components actually ran — the antidote to silent fallback /
# simplification (proposer→AR, f_θ→bypass) that kept the "fused" label.
LIVENESS_REPORT_KINDS = frozenset({"mac_gemma4_kakeya_fused_chat"})


@dataclass(frozen=True)
class GateViolation:
    """One violated evidence rule."""

    code: str
    message: str


def is_gated_report(report: Any) -> bool:
    """True when ``report`` is a K3 Mac acceptance report (any schema)."""
    return isinstance(report, dict) and report.get("kind") == MAC_REPORT_KIND


def is_liveness_report(report: Any) -> bool:
    """True when ``report`` carries the §4 component-liveness contract."""
    return isinstance(report, dict) and report.get("kind") in LIVENESS_REPORT_KINDS


def assert_liveness(report: Dict[str, Any]) -> List[GateViolation]:
    """§4 liveness contract — prove the intended components actually executed.

    Asserts, from RUNTIME signals (never from flags passed in):
      * proposer ran      — total proposer ``blocks`` across turns > 0,
      * f_θ ran           — when ``f_theta_intended`` is true, every turn has
                            ``f_theta_ran == true``,
      * no silent fallback — ``fallbacks_taken`` (report- and turn-level) empty.
    A missing liveness field is itself a violation (absence = "we don't know it
    ran" = invalid), not a skip.
    """
    violations: List[GateViolation] = []
    turns = report.get("turns")
    if not isinstance(turns, list) or not turns:
        return [GateViolation(
            "MISSING_LIVENESS",
            "liveness report has no 'turns'; component liveness cannot be "
            "asserted (absence of evidence = invalid run)",
        )]

    # --- proposer liveness: blocks > 0 (else it silently fell back to AR) ---
    total_blocks = 0
    missing_blocks = False
    for t in turns:
        b = t.get("blocks") if isinstance(t, dict) else None
        if isinstance(b, (int, float)) and not isinstance(b, bool):
            total_blocks += int(b)
        else:
            missing_blocks = True
    if missing_blocks:
        violations.append(GateViolation(
            "MISSING_LIVENESS",
            "a turn lacks numeric 'blocks'; proposer liveness unknown",
        ))
    elif total_blocks <= 0:
        violations.append(GateViolation(
            "PROPOSER_NEVER_RAN",
            "fused chat executed 0 proposer blocks across all turns — the "
            "proposer silently fell back to native AR (verifier-only)",
        ))

    # --- f_θ liveness: if intended, it must run on every turn ---
    if report.get("f_theta_intended") is True:
        ran = [t.get("f_theta_ran") if isinstance(t, dict) else None for t in turns]
        if any(r is None for r in ran):
            violations.append(GateViolation(
                "MISSING_LIVENESS",
                "f_theta_intended=true but a turn lacks 'f_theta_ran'",
            ))
        elif not all(bool(r) for r in ran):
            violations.append(GateViolation(
                "FTHETA_NOT_RUN",
                "f_theta_intended=true but f_theta_ran is false on >=1 turn — "
                "f_θ restoration was silently bypassed",
            ))

    # --- no silent fallback: declared fallbacks (report- + turn-level) empty ---
    fallbacks: List[str] = [str(x) for x in (report.get("fallbacks_taken") or [])]
    for t in turns:
        if isinstance(t, dict):
            fallbacks += [str(x) for x in (t.get("fallbacks_taken") or [])]
    if fallbacks:
        violations.append(GateViolation(
            "SILENT_FALLBACK",
            f"fallbacks_taken is non-empty: {sorted(set(fallbacks))} — a "
            "component degraded to a fallback; the system under test is not "
            "the intended one",
        ))
    return violations


def _looks_degenerate(text: Any) -> bool:
    """True when text has collapsed into a runaway repeat — the long-decode
    failure mode (e.g. many identical short lines like ``*   *   *``). Strict:
    >= 8 consecutive identical stripped non-empty lines of <= 12 chars."""
    if not isinstance(text, str):
        return False
    run = 0
    prev = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == prev and len(line) <= 12:
            run += 1
            if run >= 7:  # prev + 7 repeats = 8 identical lines
                return True
        else:
            run = 0
            prev = line
    return False


def assert_quality(report: Dict[str, Any]) -> List[GateViolation]:
    """§2.4/§2.5 contract — prove the run did not lose intelligence or throughput.

    Catches the long-decode failure the liveness gate cannot see (proposer/f_θ
    ran, yet the output is garbage + throughput collapsed):
      * RESTORATION_COVERAGE — a restored run generated more tokens than the
        resident window, beyond which the prefill-amortized restoration does NOT
        cover the evicted positions (outputs become unrestored/degenerate),
      * OUTPUT_DEGENERATE — a turn's text collapsed into a runaway repeat.
    """
    violations: List[GateViolation] = []
    turns = report.get("turns")
    if not isinstance(turns, list) or not turns:
        return violations
    window = report.get("window")
    restored = report.get("f_theta_intended") is True
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        toks = t.get("tokens")
        if (restored and isinstance(window, int) and window > 0
                and isinstance(toks, (int, float)) and not isinstance(toks, bool)
                and int(toks) > window):
            violations.append(GateViolation(
                "RESTORATION_COVERAGE",
                f"turn {i} generated {int(toks)} tokens > resident window "
                f"{window}: the prefill-amortized restoration covers only "
                "<= window decode tokens; positions evicted during decode are "
                "UNRESTORED, so the output beyond the window is degenerate",
            ))
        if _looks_degenerate(t.get("text")):
            violations.append(GateViolation(
                "OUTPUT_DEGENERATE",
                f"turn {i} output collapsed into a runaway repeat (long-decode "
                "degeneration) — not usable text",
            ))
    return violations


def is_legacy_report(report: Dict[str, Any]) -> bool:
    """True when the report predates the evidence gate (schema < 2)."""
    try:
        version = int(report.get("schema_version", 1))
    except (TypeError, ValueError):
        return True
    return version < GATED_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Shared helpers (also used by the Mac harness when assembling reports)
# ---------------------------------------------------------------------------


def row_prefill_seconds(row: Dict[str, Any]) -> Optional[float]:
    """Per-sample prefill seconds; accepts both row spellings."""
    value = row.get("prefill_s", row.get("restored_prefill_s"))
    return float(value) if isinstance(value, (int, float)) else None


def prefill_spread(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """(max / min) prefill seconds across rows; None when undeterminable."""
    values = [
        v for v in (row_prefill_seconds(r) for r in rows)
        if v is not None and v > 0
    ]
    if len(values) < 2:
        return None
    return max(values) / min(values)


def decode_only_block(
    cross_rows: Sequence[Dict[str, Any]],
    cross_tokens: Sequence[int],
    oracle_rows: Sequence[Dict[str, Any]],
    oracle_tokens: Sequence[int],
) -> Optional[Dict[str, float]]:
    """Decode-only median tok/s for both arms + their ratio.

    Prefill is identical machinery on both arms and noise-dominated on
    Apple Silicon, so the decode-only ratio is the only throughput
    comparison the gate accepts as a headline. Returns None when either
    arm lacks usable (decode_s > 0, tokens > 0) samples.
    """

    def _per_sample(rows: Sequence[Dict[str, Any]], tokens: Sequence[int]) -> List[float]:
        out: List[float] = []
        for row, n_tok in zip(rows, tokens):
            decode_s = row.get("decode_s")
            if isinstance(decode_s, (int, float)) and decode_s > 0 and n_tok > 0:
                out.append(float(n_tok) / float(decode_s))
        return out

    cross = _per_sample(cross_rows, cross_tokens)
    oracle = _per_sample(oracle_rows, oracle_tokens)
    if not cross or not oracle:
        return None
    cross_median = statistics.median(cross)
    # Both medians are > 0 by construction: _per_sample only admits
    # samples with decode_s > 0 and tokens > 0.
    oracle_median = statistics.median(oracle)
    return {
        "cross_median_tok_s": round(cross_median, 4),
        "oracle_median_tok_s": round(oracle_median, 4),
        "speedup": round(cross_median / oracle_median, 3),
    }


def summarize_violations(violations: Sequence[GateViolation]) -> str:
    """Stable multi-line rendering for logs and CI output."""
    return "\n".join(f"  [{v.code}] {v.message}" for v in violations)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _cross_arm(report: Dict[str, Any]) -> Dict[str, Any]:
    return (report.get("throughput") or {}).get("k3_cross_model") or {}


def _oracle_arm(report: Dict[str, Any]) -> Dict[str, Any]:
    return (report.get("throughput") or {}).get("oracle_native_ar") or {}


def _stage_rows(arm: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = arm.get("stage_timings")
    return list(rows) if isinstance(rows, list) else []


def validate_report(report: Dict[str, Any]) -> List[GateViolation]:
    """Validate one K3 Mac report against every evidence rule.

    Returns an empty list when the report is admissible evidence.
    Non-gated kinds validate trivially; legacy schemas return exactly
    one ``LEGACY_SCHEMA`` violation (the CI walker downgrades that one
    code to a warning — everything else fails the build).
    """
    if is_liveness_report(report):
        return assert_liveness(report) + assert_quality(report)
    if not is_gated_report(report):
        return []
    if is_legacy_report(report):
        return [GateViolation(
            "LEGACY_SCHEMA",
            f"schema_version < {GATED_SCHEMA_VERSION}: pre-gate report; "
            "grandfathered as NON-EVIDENCE (rerun with the hardened harness "
            "to make claims)",
        )]

    violations: List[GateViolation] = []
    cross = _cross_arm(report)
    rows = _stage_rows(cross)
    results = report.get("results") or {}
    results_cross = results.get("k3_cross_model") or {}
    gate = report.get("gate") or {}

    # --- Path identity: every sample must declare what actually ran ---
    if not rows:
        violations.append(GateViolation(
            "MISSING_STAGE_TIMINGS",
            "cross arm has no per-sample stage_timings; per-sample "
            "prefill_s/decode_s/restoration_active are mandatory at schema 2",
        ))
    flags = [row.get("restoration_active") for row in rows]
    if rows and any(flag is None for flag in flags):
        violations.append(GateViolation(
            "MISSING_RESTORATION_FLAG",
            "one or more cross stage rows lack restoration_active",
        ))
    known = [bool(flag) for flag in flags if flag is not None]
    all_active = bool(known) and all(known)
    none_active = bool(known) and not any(known)
    if known and not all_active and not none_active:
        violations.append(GateViolation(
            "MIXED_RESTORATION_PATHS",
            "cross samples mix restored and native paths in one report",
        ))

    # --- A baseline run may never occupy the SUT slot undeclared ---
    if none_active:
        if results_cross.get("system_under_test") != NATIVE_BASELINE_LABEL:
            violations.append(GateViolation(
                "BASELINE_AS_SUT",
                "no cross sample ran restoration, but the report does not "
                f"declare system_under_test={NATIVE_BASELINE_LABEL!r}",
            ))
        if gate.get("recall_cross_model") is not None:
            violations.append(GateViolation(
                "BASELINE_RECALL_CLAIM",
                "native-baseline run claims gate.recall_cross_model; "
                "baseline recall must be reported as recall_native_baseline",
            ))

    # --- Recall claims are scoped to the restored path ---
    if gate.get("recall_cross_model") is not None and not (known and all_active):
        violations.append(GateViolation(
            "RECALL_SCOPE",
            "gate.recall_cross_model is claimed but not every cross sample "
            "ran with restoration_active=true",
        ))

    # --- A fused report must have executed the fused engine ---
    if cross.get("eval_mode") == "free_gen_fused_specdecode":
        total_blocks = 0
        for row in rows:
            fused = row.get("fused") or {}
            blocks = fused.get("blocks")
            if isinstance(blocks, (int, float)):
                total_blocks += int(blocks)
        if total_blocks <= 0:
            violations.append(GateViolation(
                "FUSED_NEVER_RAN",
                "eval_mode=free_gen_fused_specdecode but the fused engine "
                "executed 0 blocks across all samples (silent fallback); "
                "the system under test never ran",
            ))

    # --- Speedup claims: only decode-isolated, variance-controlled,
    # adequately powered comparisons may carry a headline number ---
    throughput = report.get("throughput") or {}
    speedup = throughput.get("cross_model_speedup_vs_oracle_ar")
    if speedup is not None:
        oracle = _oracle_arm(report)
        oracle_rows = _stage_rows(oracle)
        if none_active:
            violations.append(GateViolation(
                "SPEEDUP_SELF_COMPARISON",
                "speedup claimed on a native-baseline run: the cross arm IS "
                "the oracle computation; the ratio is run-order noise",
            ))
        if len(rows) < MIN_PERF_SAMPLES or len(oracle_rows) < MIN_PERF_SAMPLES:
            violations.append(GateViolation(
                "SPEEDUP_SAMPLES",
                f"speedup claimed with n cross={len(rows)} oracle="
                f"{len(oracle_rows)}; minimum is {MIN_PERF_SAMPLES} per arm",
            ))
        cross_tokens = results_cross.get("per_sample_decode_tokens") or []
        oracle_tokens = (results.get("oracle") or {}).get("per_sample_decode_tokens") or []
        medians = [
            statistics.median(t) for t in (cross_tokens, oracle_tokens) if t
        ]
        if len(medians) < 2 or min(medians) < MIN_MEDIAN_DECODE_TOKENS:
            violations.append(GateViolation(
                "SPEEDUP_DECODE_TOKENS",
                f"speedup claimed with median decode tokens {medians or 'missing'}; "
                f"minimum is {MIN_MEDIAN_DECODE_TOKENS} per arm (otherwise the "
                "wall time is prefill noise, not decode throughput)",
            ))
        decode_only = throughput.get("decode_only") or {}
        if decode_only.get("speedup") is None:
            violations.append(GateViolation(
                "SPEEDUP_DECODE_ONLY_MISSING",
                "headline speedup present without throughput.decode_only "
                "medians; prefill-inclusive ratios alone are inadmissible",
            ))
        if cross.get("timing_scope") != oracle.get("timing_scope"):
            violations.append(GateViolation(
                "SPEEDUP_SCOPE_MISMATCH",
                f"cross timing_scope={cross.get('timing_scope')!r} != oracle "
                f"timing_scope={oracle.get('timing_scope')!r}",
            ))
        if oracle.get("decode_loop") != CLAIM_ORACLE_DECODE_LOOP:
            violations.append(GateViolation(
                "SPEEDUP_ORACLE_LOOP",
                f"oracle decode_loop={oracle.get('decode_loop')!r}; headline "
                f"speedups require {CLAIM_ORACLE_DECODE_LOOP!r} (per-token "
                "mx.eval loops are the documented MLX anti-pattern and "
                "depress the baseline)",
            ))
        for arm_name, arm_rows in (("cross", rows), ("oracle", oracle_rows)):
            spread = prefill_spread(arm_rows)
            if spread is not None and spread > MAX_PREFILL_SPREAD:
                violations.append(GateViolation(
                    "SPEEDUP_PREFILL_VARIANCE",
                    f"{arm_name} arm prefill spread {spread:.2f}× exceeds "
                    f"{MAX_PREFILL_SPREAD}×; e2e ratios under this variance "
                    "are noise — claim decode-only or control variance",
                ))

    # --- Memory claims must describe the run that was measured ---
    memory = report.get("memory") or {}
    if memory.get("savings_vs_naive_pct") is not None:
        s5 = memory.get("s5") or {}
        if s5.get("formula_matches_run") is not True:
            violations.append(GateViolation(
                "MEMORY_CLAIM_MISMATCH",
                "memory.savings_vs_naive_pct is claimed but memory.s5."
                "formula_matches_run is not true: the analytical sink+window "
                "table does not describe the cache the run actually used",
            ))

    return violations
