"""Unit tests for inference_engine.bench.k3_report_gate.

Each rule is pinned by mutating exactly one aspect of a fully valid
schema-2 report, so a future schema drift that silently disables a
rule fails here. The fixtures mirror the real committed report shapes
(see results/research/k3_mlx_fused_fair_ctx280_n5_gen32_*.json on this
branch — the report whose failure modes created this gate).

Coverage target: 100% on ``inference_engine/bench/k3_report_gate.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from inference_engine.bench.k3_report_gate import (
    CLAIM_ORACLE_DECODE_LOOP,
    GATED_SCHEMA_VERSION,
    LIVENESS_REPORT_KINDS,
    MAC_REPORT_KIND,
    MAX_PREFILL_SPREAD,
    MIN_MEDIAN_DECODE_TOKENS,
    MIN_PERF_SAMPLES,
    NATIVE_BASELINE_LABEL,
    GateViolation,
    assert_liveness,
    decode_only_block,
    is_gated_report,
    is_legacy_report,
    is_liveness_report,
    prefill_spread,
    row_prefill_seconds,
    summarize_violations,
    validate_report,
)


# ---------------------------------------------------------------------------
# §4 liveness contract (proposer / f_θ / no-fallback)
# ---------------------------------------------------------------------------


def _live_report(**over: Any) -> Dict[str, Any]:
    """A fused-chat liveness report that passes the §4 contract."""
    rep = {
        "kind": next(iter(LIVENESS_REPORT_KINDS)),
        "schema_version": 1,
        "f_theta_intended": True,
        "fallbacks_taken": [],
        "turns": [
            {"user": "q1", "blocks": 2, "mean_accept_len": 4.0,
             "f_theta_ran": True, "fallbacks_taken": []},
            {"user": "q2", "blocks": 4, "mean_accept_len": 3.5,
             "f_theta_ran": True, "fallbacks_taken": []},
        ],
    }
    rep.update(over)
    return rep


def test_liveness_report_detection_and_pass():
    rep = _live_report()
    assert is_liveness_report(rep) and not is_gated_report(rep)
    assert validate_report(rep) == []      # dispatches to assert_liveness
    assert assert_liveness(rep) == []


def test_liveness_missing_turns_is_invalid():
    codes = {v.code for v in assert_liveness(_live_report(turns=[]))}
    assert codes == {"MISSING_LIVENESS"}


def test_liveness_proposer_never_ran():
    rep = _live_report(turns=[
        {"blocks": 0, "f_theta_ran": True}, {"blocks": 0, "f_theta_ran": True}])
    codes = {v.code for v in assert_liveness(rep)}
    assert "PROPOSER_NEVER_RAN" in codes


def test_liveness_missing_blocks_field():
    rep = _live_report(turns=[{"f_theta_ran": True}])  # no 'blocks'
    codes = {v.code for v in assert_liveness(rep)}
    assert "MISSING_LIVENESS" in codes


def test_liveness_bool_blocks_not_counted_as_int():
    # True is an int subclass — must NOT be accepted as a block count.
    rep = _live_report(turns=[{"blocks": True, "f_theta_ran": True}])
    codes = {v.code for v in assert_liveness(rep)}
    assert "MISSING_LIVENESS" in codes


def test_liveness_ftheta_not_run_when_intended():
    rep = _live_report(turns=[
        {"blocks": 2, "f_theta_ran": True}, {"blocks": 2, "f_theta_ran": False}])
    codes = {v.code for v in assert_liveness(rep)}
    assert "FTHETA_NOT_RUN" in codes


def test_liveness_ftheta_missing_flag_when_intended():
    rep = _live_report(turns=[{"blocks": 2}])  # f_theta_intended True but no flag
    codes = {v.code for v in assert_liveness(rep)}
    assert "MISSING_LIVENESS" in codes


def test_liveness_ftheta_not_required_when_not_intended():
    # all-MLX fast path: f_θ bypassed by design → no FTHETA_NOT_RUN.
    rep = _live_report(f_theta_intended=False, turns=[
        {"blocks": 2, "f_theta_ran": False}, {"blocks": 3, "f_theta_ran": False}])
    assert assert_liveness(rep) == []


def test_liveness_silent_fallback_report_and_turn_level():
    rep = _live_report(fallbacks_taken=["proposer->ar"])
    assert any(v.code == "SILENT_FALLBACK" for v in assert_liveness(rep))
    rep2 = _live_report(turns=[
        {"blocks": 2, "f_theta_ran": True, "fallbacks_taken": ["f_theta->identity"]}])
    assert any(v.code == "SILENT_FALLBACK" for v in assert_liveness(rep2))


# ---------------------------------------------------------------------------
# §2.4/§2.5 quality contract (degeneration / restoration coverage)
# ---------------------------------------------------------------------------

from inference_engine.bench.k3_report_gate import assert_quality, _looks_degenerate


def _quality_report(turns, window=64, restored=True):
    return {
        "kind": next(iter(LIVENESS_REPORT_KINDS)), "schema_version": 1,
        "f_theta_intended": restored, "window": window,
        "fallbacks_taken": [], "turns": turns,
    }


def test_quality_passes_clean_short_turn():
    rep = _quality_report([{"tokens": 12, "text": "The capital of France is Paris."}])
    assert assert_quality(rep) == []


def test_quality_restoration_coverage_exceeded():
    # the PoW failure: restored run generated way past the window
    rep = _quality_report([{"tokens": 780, "text": "ok"}], window=64, restored=True)
    codes = {v.code for v in assert_quality(rep)}
    assert "RESTORATION_COVERAGE" in codes
    # and it surfaces through validate_report (dispatch wires assert_quality)
    assert any(v.code == "RESTORATION_COVERAGE" for v in validate_report(rep))


def test_quality_no_coverage_check_when_not_restored():
    # all-MLX path (f_θ bypassed): no restoration to exceed.
    rep = _quality_report([{"tokens": 780, "text": "ok"}], window=64, restored=False)
    assert all(v.code != "RESTORATION_COVERAGE" for v in assert_quality(rep))


def test_quality_coverage_skipped_without_window():
    rep = _quality_report([{"tokens": 780, "text": "ok"}], window=None)
    assert all(v.code != "RESTORATION_COVERAGE" for v in assert_quality(rep))


def test_quality_bool_tokens_not_counted():
    rep = _quality_report([{"tokens": True, "text": "ok"}], window=64)
    assert all(v.code != "RESTORATION_COVERAGE" for v in assert_quality(rep))


def test_quality_output_degenerate_detected():
    garbage = "Answer:\n" + "\n".join(["*   *   *"] * 12)
    rep = _quality_report([{"tokens": 50, "text": garbage}])
    assert any(v.code == "OUTPUT_DEGENERATE" for v in assert_quality(rep))


def test_quality_empty_and_nondict_turns():
    assert assert_quality(_quality_report([])) == []
    assert assert_quality({"kind": next(iter(LIVENESS_REPORT_KINDS)),
                           "turns": "nope"}) == []
    # a non-dict turn element is skipped without error
    assert assert_quality(_quality_report(["not-a-dict",
                                           {"tokens": 5, "text": "fine"}])) == []


def test_looks_degenerate_helper():
    assert _looks_degenerate("\n".join(["*   *   *"] * 10)) is True
    assert _looks_degenerate("a normal coherent sentence about proof of work") is False
    assert _looks_degenerate(123) is False
    # long repeated lines (>12 chars) are NOT flagged (could be legit content)
    assert _looks_degenerate("\n".join(["this line is definitely longer than twelve"] * 10)) is False
    # blank lines are skipped; a short line then different lines resets the run
    assert _looks_degenerate("x\n\ny\nz\nw\nq\nr\ns") is False


def _valid_report(n: int = MIN_PERF_SAMPLES) -> Dict[str, Any]:
    """A schema-2 report that passes every rule."""
    cross_rows = [
        {
            "sample": i,
            "prefill_s": 30.0 + i,
            "decode_s": 2.0,
            "e2e_s": 32.0 + i,
            "restoration_active": True,
            "decode_loop": "fused_specdecode",
            "fused": {"blocks": 8, "mean_accept_len": 1.5},
        }
        for i in range(n)
    ]
    oracle_rows = [
        {"sample": i, "prefill_s": 31.0 + i, "decode_s": 4.0, "e2e_s": 35.0 + i}
        for i in range(n)
    ]
    return {
        "schema_version": GATED_SCHEMA_VERSION,
        "kind": MAC_REPORT_KIND,
        "results": {
            "k3_cross_model": {
                "recall": 1.0,
                "per_sample_decode_tokens": [MIN_MEDIAN_DECODE_TOKENS] * n,
                "system_under_test": "restored_cross_model",
            },
            "oracle": {
                "recall": 1.0,
                "per_sample_decode_tokens": [MIN_MEDIAN_DECODE_TOKENS] * n,
            },
        },
        "gate": {"recall_cross_model": 1.0, "recall_oracle": 1.0},
        "memory": {
            "s5": {"formula_matches_run": True},
            "savings_vs_naive_pct": 89.8,
        },
        "throughput": {
            "k3_cross_model": {
                "eval_mode": "free_gen_fused_specdecode",
                "timing_scope": "e2e_prefill_plus_decode",
                "stage_timings": cross_rows,
            },
            "oracle_native_ar": {
                "timing_scope": "e2e_prefill_plus_decode",
                "decode_loop": CLAIM_ORACLE_DECODE_LOOP,
                "stage_timings": oracle_rows,
            },
            "decode_only": {
                "cross_median_tok_s": 16.0,
                "oracle_median_tok_s": 8.0,
                "speedup": 2.0,
            },
            "cross_model_speedup_vs_oracle_ar": 1.18,
        },
    }


def _codes(report: Dict[str, Any]) -> set:
    return {v.code for v in validate_report(report)}


# ---------------------------------------------------------------------------
# Scope / legacy handling
# ---------------------------------------------------------------------------


def test_valid_report_has_no_violations():
    assert validate_report(_valid_report()) == []


def test_non_gated_kinds_validate_trivially():
    assert validate_report({"kind": "k3_f_theta_train", "schema_version": 1}) == []
    assert validate_report("not even a dict") == []  # type: ignore[arg-type]


def test_is_gated_report():
    assert is_gated_report(_valid_report())
    assert not is_gated_report({"kind": "other"})
    assert not is_gated_report(None)


def test_legacy_schema_is_single_grandfather_violation():
    report = _valid_report()
    report["schema_version"] = 1
    violations = validate_report(report)
    assert [v.code for v in violations] == ["LEGACY_SCHEMA"]
    assert "NON-EVIDENCE" in violations[0].message


def test_is_legacy_report_handles_garbage_versions():
    assert is_legacy_report({"schema_version": 1})
    assert is_legacy_report({"schema_version": "not-a-number"})
    assert is_legacy_report({})
    assert not is_legacy_report({"schema_version": GATED_SCHEMA_VERSION})


# ---------------------------------------------------------------------------
# Path identity rules
# ---------------------------------------------------------------------------


def test_missing_stage_timings_flagged():
    report = _valid_report()
    report["throughput"]["k3_cross_model"]["stage_timings"] = []
    # No rows ⇒ also no restoration evidence for the recall claim.
    assert {"MISSING_STAGE_TIMINGS", "RECALL_SCOPE"} <= _codes(report)


def test_stage_timings_wrong_type_treated_as_missing():
    report = _valid_report()
    report["throughput"]["k3_cross_model"]["stage_timings"] = "oops"
    assert "MISSING_STAGE_TIMINGS" in _codes(report)


def test_missing_restoration_flag_flagged():
    report = _valid_report()
    del report["throughput"]["k3_cross_model"]["stage_timings"][2]["restoration_active"]
    assert "MISSING_RESTORATION_FLAG" in _codes(report)


def test_mixed_restoration_paths_flagged():
    report = _valid_report()
    report["throughput"]["k3_cross_model"]["stage_timings"][0]["restoration_active"] = False
    codes = _codes(report)
    assert "MIXED_RESTORATION_PATHS" in codes
    assert "RECALL_SCOPE" in codes  # recall claim no longer covered


def _native_baseline_report() -> Dict[str, Any]:
    """A correctly-declared native-baseline run (no claims) — admissible."""
    report = _valid_report()
    for row in report["throughput"]["k3_cross_model"]["stage_timings"]:
        row["restoration_active"] = False
        row["fused"] = {"blocks": 0, "mean_accept_len": 0.0}
        row["decode_loop"] = "per_token_eval"
    report["throughput"]["k3_cross_model"]["eval_mode"] = "native_ar_baseline"
    report["results"]["k3_cross_model"]["system_under_test"] = NATIVE_BASELINE_LABEL
    report["gate"]["recall_cross_model"] = None
    report["gate"]["recall_native_baseline"] = 1.0
    report["throughput"]["cross_model_speedup_vs_oracle_ar"] = None
    report["memory"]["savings_vs_naive_pct"] = None
    report["memory"]["s5"]["formula_matches_run"] = False
    return report


def test_declared_native_baseline_is_admissible():
    assert validate_report(_native_baseline_report()) == []


def test_undeclared_baseline_as_sut_flagged():
    report = _native_baseline_report()
    report["results"]["k3_cross_model"]["system_under_test"] = "restored_cross_model"
    assert "BASELINE_AS_SUT" in _codes(report)


def test_baseline_recall_claim_flagged():
    report = _native_baseline_report()
    report["gate"]["recall_cross_model"] = 1.0
    codes = _codes(report)
    assert "BASELINE_RECALL_CLAIM" in codes
    assert "RECALL_SCOPE" in codes


def test_recall_scope_requires_all_samples_restored():
    report = _valid_report()
    report["throughput"]["k3_cross_model"]["stage_timings"] = []
    assert "RECALL_SCOPE" in _codes(report)


# ---------------------------------------------------------------------------
# Fused execution rule — the blocks=0 reports that motivated the gate
# ---------------------------------------------------------------------------


def test_fused_never_ran_flagged():
    report = _valid_report()
    for row in report["throughput"]["k3_cross_model"]["stage_timings"]:
        row["fused"] = {"blocks": 0, "mean_accept_len": 0.0}
    assert "FUSED_NEVER_RAN" in _codes(report)


def test_fused_missing_blocks_counts_as_zero():
    report = _valid_report()
    for row in report["throughput"]["k3_cross_model"]["stage_timings"]:
        row["fused"] = {}
        row.pop("fused")
    assert "FUSED_NEVER_RAN" in _codes(report)


def test_fused_rule_only_applies_to_fused_eval_mode():
    report = _valid_report()
    report["throughput"]["k3_cross_model"]["eval_mode"] = "free_gen_incremental"
    for row in report["throughput"]["k3_cross_model"]["stage_timings"]:
        row.pop("fused")
        row["decode_loop"] = "generate_step"
    assert validate_report(report) == []


# ---------------------------------------------------------------------------
# Speedup claim rules
# ---------------------------------------------------------------------------


def test_withheld_speedup_skips_all_speedup_rules():
    report = _valid_report(n=1)  # tiny smoke...
    report["throughput"]["cross_model_speedup_vs_oracle_ar"] = None  # ...no claim
    report["results"]["k3_cross_model"]["per_sample_decode_tokens"] = [8]
    report["results"]["oracle"]["per_sample_decode_tokens"] = [8]
    assert validate_report(report) == []


def test_speedup_on_baseline_is_self_comparison():
    report = _native_baseline_report()
    report["throughput"]["cross_model_speedup_vs_oracle_ar"] = 2.584
    assert "SPEEDUP_SELF_COMPARISON" in _codes(report)


def test_speedup_sample_floor():
    report = _valid_report(n=MIN_PERF_SAMPLES - 1)
    assert "SPEEDUP_SAMPLES" in _codes(report)


def test_speedup_decode_token_floor():
    report = _valid_report()
    report["results"]["k3_cross_model"]["per_sample_decode_tokens"] = (
        [8] * MIN_PERF_SAMPLES  # the gen=8 smokes that motivated the rule
    )
    assert "SPEEDUP_DECODE_TOKENS" in _codes(report)


def test_speedup_decode_token_floor_missing_lists():
    report = _valid_report()
    report["results"]["oracle"]["per_sample_decode_tokens"] = []
    assert "SPEEDUP_DECODE_TOKENS" in _codes(report)


def test_speedup_requires_decode_only_block():
    report = _valid_report()
    report["throughput"]["decode_only"] = {}
    assert "SPEEDUP_DECODE_ONLY_MISSING" in _codes(report)
    del report["throughput"]["decode_only"]
    assert "SPEEDUP_DECODE_ONLY_MISSING" in _codes(report)


def test_speedup_scope_mismatch_flagged():
    report = _valid_report()
    report["throughput"]["oracle_native_ar"]["timing_scope"] = "decode_only"
    assert "SPEEDUP_SCOPE_MISMATCH" in _codes(report)


def test_speedup_oracle_loop_rule():
    report = _valid_report()
    report["throughput"]["oracle_native_ar"]["decode_loop"] = "per_token_eval"
    assert "SPEEDUP_ORACLE_LOOP" in _codes(report)


def test_speedup_prefill_variance_rule_each_arm():
    report = _valid_report()
    # The ctx280 oracle arm: 35.3s..146.3s on identical work.
    rows = report["throughput"]["oracle_native_ar"]["stage_timings"]
    rows[0]["prefill_s"] = 35.3
    rows[3]["prefill_s"] = 146.3
    assert "SPEEDUP_PREFILL_VARIANCE" in _codes(report)

    report2 = _valid_report()
    rows2 = report2["throughput"]["k3_cross_model"]["stage_timings"]
    rows2[0]["prefill_s"] = 10.0
    rows2[1]["prefill_s"] = 10.0 * (MAX_PREFILL_SPREAD + 0.1)
    assert "SPEEDUP_PREFILL_VARIANCE" in _codes(report2)


# ---------------------------------------------------------------------------
# Memory claim rule
# ---------------------------------------------------------------------------


def test_memory_claim_requires_formula_match():
    report = _valid_report()
    report["memory"]["s5"]["formula_matches_run"] = False
    assert "MEMORY_CLAIM_MISMATCH" in _codes(report)
    report["memory"]["s5"] = {}
    assert "MEMORY_CLAIM_MISMATCH" in _codes(report)


def test_memory_rule_skipped_when_no_savings_claim():
    report = _valid_report()
    report["memory"]["savings_vs_naive_pct"] = None
    report["memory"]["s5"]["formula_matches_run"] = False
    assert validate_report(report) == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_row_prefill_seconds_accepts_both_spellings():
    assert row_prefill_seconds({"prefill_s": 1.5}) == 1.5
    assert row_prefill_seconds({"restored_prefill_s": 2.5}) == 2.5
    assert row_prefill_seconds({"prefill_s": "bad"}) is None
    assert row_prefill_seconds({}) is None


def test_prefill_spread():
    assert prefill_spread([{"prefill_s": 10.0}, {"prefill_s": 40.0}]) == 4.0
    assert prefill_spread([{"prefill_s": 10.0}]) is None
    assert prefill_spread([{"prefill_s": 0.0}, {"prefill_s": -1}]) is None
    assert prefill_spread([]) is None


def test_decode_only_block_happy_path():
    cross = [{"decode_s": 2.0}, {"decode_s": 4.0}]
    oracle = [{"decode_s": 8.0}, {"decode_s": 8.0}]
    block = decode_only_block(cross, [32, 32], oracle, [32, 32])
    assert block == {
        "cross_median_tok_s": 12.0,   # median(16, 8)
        "oracle_median_tok_s": 4.0,
        "speedup": 3.0,
    }


def test_decode_only_block_unusable_samples_return_none():
    assert decode_only_block([{"decode_s": 0.0}], [8], [{"decode_s": 1.0}], [8]) is None
    assert decode_only_block([{"decode_s": 1.0}], [0], [{"decode_s": 1.0}], [8]) is None
    assert decode_only_block([{"decode_s": 1.0}], [8], [{}], [8]) is None


def test_summarize_violations_renders_codes():
    text = summarize_violations([
        GateViolation("A_CODE", "first"),
        GateViolation("B_CODE", "second"),
    ])
    assert text == "  [A_CODE] first\n  [B_CODE] second"


def test_the_committed_ctx280_report_shape_would_now_fail():
    """Regression lock: a report shaped like the real
    k3_mlx_fused_fair_ctx280_n5_gen32 run (baseline-as-SUT, blocks=0,
    gen=8, 4.1x oracle prefill spread, formula memory table) violates
    multiple rules at schema 2 instead of presenting as a 2.584x win."""
    report = _valid_report()
    for row in report["throughput"]["k3_cross_model"]["stage_timings"]:
        row["restoration_active"] = False          # native bypass ran
        row["fused"] = {"blocks": 0, "mean_accept_len": 0.0}
    report["results"]["k3_cross_model"]["per_sample_decode_tokens"] = [8, 7, 8, 8, 8]
    report["results"]["oracle"]["per_sample_decode_tokens"] = [8, 7, 8, 8, 8]
    oracle_rows = report["throughput"]["oracle_native_ar"]["stage_timings"]
    oracle_rows[0]["prefill_s"] = 35.3
    oracle_rows[3]["prefill_s"] = 146.3
    report["throughput"]["oracle_native_ar"]["decode_loop"] = "per_token_eval"
    del report["throughput"]["decode_only"]
    report["throughput"]["cross_model_speedup_vs_oracle_ar"] = 2.584
    # The analytical sink+window table did not describe the native cache
    # that actually ran.
    report["memory"]["s5"]["formula_matches_run"] = False

    codes = _codes(report)
    assert {
        "BASELINE_AS_SUT",
        "BASELINE_RECALL_CLAIM",
        "RECALL_SCOPE",
        "FUSED_NEVER_RAN",
        "SPEEDUP_SELF_COMPARISON",
        "SPEEDUP_DECODE_TOKENS",
        "SPEEDUP_DECODE_ONLY_MISSING",
        "SPEEDUP_ORACLE_LOOP",
        "SPEEDUP_PREFILL_VARIANCE",
        "MEMORY_CLAIM_MISMATCH",
    } <= codes
