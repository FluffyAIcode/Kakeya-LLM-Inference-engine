"""K3 v4 evidence auto-analyzer.

Given a JSON evidence path (training report, NIAH eval, or alpha-sweep),
extracts the key metrics + compares to thresholds + emits a structured
markdown report. Used by the agent's polling loop to analyze evidence
as it arrives.

Recall threshold (from PR #103 v3 alpha-sweep evidence):
    - full_attn rel_mse <= 0.40 → recall >= 0.9 (PASS gate)
    - full_attn rel_mse  ~ 0.50 → recall ~  0.6
    - full_attn rel_mse >= 0.70 → recall = 0   (FAIL)

Usage:
    python3 scripts/research/k3_v4_analyze.py <path_to_evidence.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


# Thresholds derived from PR #103 v3 alpha-sweep evidence (commit 3643b74).
RECALL_THRESHOLDS = {
    "pass":   0.40,   # recall >= 0.9
    "knee":   0.52,   # recall ~ 0.6
    "fail":   0.70,   # recall = 0
}


def _classify_recall(full_attn_rel_mse: float) -> str:
    if full_attn_rel_mse <= RECALL_THRESHOLDS["pass"]:
        return "PASS-EXPECTED (recall ≥ 90%)"
    if full_attn_rel_mse <= RECALL_THRESHOLDS["knee"]:
        return "MARGINAL (recall ~50-80%)"
    if full_attn_rel_mse <= RECALL_THRESHOLDS["fail"]:
        return "FAIL-LIKELY (recall < 30%)"
    return "FAIL-CERTAIN (recall = 0)"


def analyze_training_report(d: Dict[str, Any]) -> str:
    """Analyse a k3_f_theta_train report."""
    cfg = d.get("config", {})
    fcfg = d.get("f_theta_config", {})
    diag = d.get("final_diagnostic", {})
    out = []
    out.append("# K3 f_θ Training Report Analysis")
    out.append("")
    out.append("## Config")
    out.append(f"- Loss type: **{cfg.get('loss_type', '?')}**")
    out.append(f"- Steps: {cfg.get('steps', '?')}")
    out.append(f"- Rank: {cfg.get('rank', '?')}")
    out.append(f"- Gen len: {cfg.get('gen_len', '?')}")
    out.append(f"- N general / NIAH prompts: {cfg.get('n_prompts', '?')} / {cfg.get('n_niah_prompts', '?')}")
    out.append(f"- LR schedule: {cfg.get('lr_schedule', '?')} (warmup {cfg.get('warmup_steps', '?')})")
    out.append(f"- Init from: {cfg.get('init_from', '(scratch)')}")
    if cfg.get("loss_type") == "attn_distill_hybrid":
        out.append(
            f"- λ: kDir={cfg.get('lambda_k_dir', '?')} vDir={cfg.get('lambda_v_dir', '?')} "
            f"kMag={cfg.get('lambda_k_mag', '?')} vMag={cfg.get('lambda_v_mag', '?')}"
        )
    out.append("")
    out.append("## f_θ Architecture")
    out.append(f"- Params: {d.get('n_params', '?'):,}" if isinstance(d.get('n_params'), int) else f"- Params: {d.get('n_params', '?')}")
    out.append(f"- Drafter layers: {fcfg.get('drafter_num_layers', '?')}")
    out.append(f"- Verifier layers: {fcfg.get('verifier_num_layers', '?')}")
    out.append(f"- Bottleneck rank: {fcfg.get('rank', '?')}")
    out.append("")
    out.append("## Training Metrics")
    init_loss = d.get("initial_loss", 0)
    final_loss = d.get("final_loss", 0)
    reduction = d.get("loss_reduction_factor", 0)
    out.append(f"- Initial loss: {init_loss:.4f}")
    out.append(f"- Final loss:   {final_loss:.4f}")
    out.append(f"- Reduction:    {reduction:.2f}×")
    out.append(f"- Train wall:   {d.get('train_seconds', 0)/60:.1f} min")
    out.append(f"- Collect wall: {d.get('collect_seconds', 0)/60:.1f} min")
    out.append(f"- N sequences:  {d.get('n_sequences', '?')}")
    out.append("")
    out.append("## Diagnostic Metrics")
    if diag:
        for k, v in diag.items():
            if isinstance(v, (int, float)):
                out.append(f"- {k}: {v:.6f}")
            else:
                out.append(f"- {k}: {v}")
        # Compute attn output rel-err if both present
        mse_o = diag.get("mse_O_mean")
        abs_o = diag.get("abs_O_target_mean")
        if mse_o and abs_o:
            ratio = mse_o / max(abs_o ** 2, 1e-12)
            out.append(f"- **attn_output rel-err = {ratio:.4f}** (lower = better; v3 attn_distill = 0.38; collapse-degenerate)")
        if diag.get("k_dir_mean") is not None:
            cos_k = 1 - diag["k_dir_mean"]
            out.append(f"- **K cos sim = {cos_k:.3f}** (target > 0.95)")
        if diag.get("v_dir_mean") is not None:
            cos_v = 1 - diag["v_dir_mean"]
            out.append(f"- **V cos sim = {cos_v:.3f}** (target > 0.95)")
    else:
        out.append("- (no diagnostic data)")
    return "\n".join(out)


def analyze_niah_eval(d: Dict[str, Any]) -> str:
    """Analyse a k3_integrated_niah_acceptance JSON."""
    out = []
    out.append("# K3 Integrated NIAH Eval Analysis")
    out.append("")
    cfg = d.get("config", {})
    out.append("## Config")
    out.append(f"- f_θ dir: {cfg.get('f_theta_dir', '?')}")
    out.append(f"- N samples: {cfg.get('n_samples', '?')}")
    out.append(f"- Sink/window: {cfg.get('sink_size', '?')}/{cfg.get('window_size', '?')}")
    out.append(f"- Haystack lines: {cfg.get('haystack_min_lines', '?')}-{cfg.get('haystack_max_lines', '?')}")
    out.append("")
    results = d.get("results", {})
    cm = results.get("k3_cross_model", {})
    oracle = results.get("oracle", {})
    out.append("## Recall")
    cm_recall = cm.get("recall", 0)
    cm_corr = cm.get("samples_correct", 0)
    cm_tot = cm.get("samples_total", 0)
    out.append(f"- **k3_cross_model**: {cm_recall:.3f} ({cm_corr}/{cm_tot})")
    if oracle:
        or_recall = oracle.get("recall", 0)
        or_corr = oracle.get("samples_correct", 0)
        or_tot = oracle.get("samples_total", 0)
        out.append(f"- **oracle (verifier vanilla)**: {or_recall:.3f} ({or_corr}/{or_tot})")
        delta = abs(cm_recall - or_recall) * 100
        gate = "PASS ✅" if delta <= 5.0 else "FAIL ❌"
        out.append(f"- **|Δ vs oracle| = {delta:.1f} pp** — gate ≤5pp: **{gate}**")
    out.append("")
    gate_d = d.get("gate", {})
    if gate_d:
        out.append("## Gate Booleans")
        for k, v in gate_d.items():
            out.append(f"- {k}: {v}")
        out.append("")
    out.append("## Per-sample Decoded (k3_cross_model)")
    decoded = cm.get("per_sample_decoded", [])[:5]
    for i, s in enumerate(decoded):
        out.append(f"  {i}. `{s[:90]}{'...' if len(s) > 90 else ''}`")
    if oracle:
        out.append("")
        out.append("## Per-sample Decoded (oracle, for comparison)")
        decoded = oracle.get("per_sample_decoded", [])[:5]
        for i, s in enumerate(decoded):
            out.append(f"  {i}. `{s[:90]}{'...' if len(s) > 90 else ''}`")
    return "\n".join(out)


def analyze_alpha_sweep(d: Dict[str, Any]) -> str:
    """Analyse a k3_s6_fidelity_recall_sweep JSON."""
    out = []
    out.append("# K3 Alpha-Sweep (Fidelity → Recall) Analysis")
    out.append("")
    cfg = d.get("config", {})
    out.append("## Config")
    out.append(f"- f_θ dir: {cfg.get('f_theta_dir', '?')}")
    out.append(f"- N samples: {cfg.get('n_samples', '?')}")
    out.append("")
    base = d.get("f_theta_baseline_rel_mse", {})
    out.append("## Baseline rel_mse (α=0, pure f_θ)")
    out.append(f"- Overall:   {base.get('overall', '?'):.4f}" if isinstance(base.get('overall'), (int, float)) else f"- Overall:   {base.get('overall', '?')}")
    full_attn = base.get("full_attn", 0)
    out.append(f"- Full-attn: {full_attn:.4f}" if isinstance(full_attn, (int, float)) else f"- Full-attn: {full_attn}")
    if isinstance(full_attn, (int, float)):
        out.append(f"- → predicted recall: **{_classify_recall(full_attn)}**")
    out.append("")
    sweep = d.get("sweep", [])
    out.append("## Sweep")
    out.append("| α | recall | overall_rel_mse | full_attn_rel_mse |")
    out.append("|---|---|---|---|")
    for entry in sweep:
        a = entry.get("alpha")
        r = entry.get("recall")
        o = entry.get("eff_rel_mse_overall")
        f = entry.get("eff_rel_mse_full_attn")
        out.append(f"| {a} | {r} | {o:.4f} | {f:.4f}" if isinstance(o, float) else f"| {a} | {r} | {o} | {f} |")
    out.append("")
    # Find recall knee
    last_zero = None
    first_pass = None
    for entry in sweep:
        if entry.get("recall", 0) == 0:
            last_zero = entry
        elif entry.get("recall", 0) >= 0.9 and first_pass is None:
            first_pass = entry
    if last_zero and first_pass:
        out.append(f"## Recall Knee")
        out.append(f"- Last α with recall=0: α={last_zero.get('alpha')} (full_attn rel_mse {last_zero.get('eff_rel_mse_full_attn'):.3f})")
        out.append(f"- First α with recall≥0.9: α={first_pass.get('alpha')} (full_attn rel_mse {first_pass.get('eff_rel_mse_full_attn'):.3f})")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to evidence JSON")
    args = ap.parse_args()
    p = Path(args.path)
    if not p.is_file():
        print(f"NOT FOUND: {p}", file=sys.stderr)
        return 1
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"INVALID JSON: {p}: {e}", file=sys.stderr)
        return 2

    kind = d.get("kind", "")
    print(f"# Evidence: `{p}`")
    print(f"## Kind: `{kind}`")
    print()
    if kind == "k3_f_theta_train":
        print(analyze_training_report(d))
    elif kind == "k3_integrated_niah_acceptance":
        print(analyze_niah_eval(d))
    elif kind == "k3_s6_fidelity_recall_sweep":
        print(analyze_alpha_sweep(d))
    else:
        print(f"(unknown kind {kind!r}; raw keys: {sorted(d.keys())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
