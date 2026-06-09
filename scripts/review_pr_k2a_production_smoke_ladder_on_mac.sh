#!/usr/bin/env bash
# Mac M4 K2.A production-shape ladder (per user directive 2026-06-09).
#
# Runs scripts/review_pr_k2a_production_smoke_on_mac.sh at two context
# rungs sequentially and aggregates the four product-relevant metrics
# from each per-rung JSON into a single ladder JSON suitable for ADR
# §11 citation as 'Mac M4 production-shape empirical bound'.
#
# Why two rungs (default 70 + 280):
#
#   The 2026-06-09 v4 Mac M4 production smoke at ctx280 (~6413 tokens)
#   produced:
#
#     recall              = 0.0   (0/1)
#     mean_latency        = 264.22 s for 24 tokens   (~11 sec/token)
#     mean_throughput     = 0.091 tok/s
#     driver_allocated    = 26.63 GB on a 24 GB unified-memory box
#                           → physical memory exceeded → swap thrashing
#     effective_attn_frac = 1.0  (architecture correct: full context attended)
#
#   The single ctx280 datapoint mixes three signals:
#
#     (a) memory pressure dominating latency
#     (b) KL D4 Q=38 lossy on Mac MPS bf16
#     (c) single-sample Bernoulli variance
#
#   Adding ctx70 (~1400 tokens) — well within Mac M4 24 GB physical
#   memory — disambiguates (a) from (b) + (c). If ctx70 produces
#   recall=hit and reasonable latency, the architecture is correct
#   and the ctx280 failure is the memory-thrashing upper bound. If
#   ctx70 also fails, there's a non-memory bug that this PR has not
#   yet caught.
#
#   Either way the ladder JSON gives ADR §11 a falsifiable empirical
#   row: 'Mac M4 24 GB + Gemma 3-1B + KL D4 Q=38 + K2.A.2 stateful is
#   product-viable up to ctx X tokens; above that, swap thrashing
#   dominates.'
#
# What this script does NOT do:
#
#   * KL Q=38 vs Q=76 A/B (scope: deferred — that's a separate
#     hypothesis test for KL precision, addressable in a follow-up
#     reviewer aid if ctx70 ALSO fails).
#   * stateful=on vs stateful=off A/B (scope: deferred — same
#     reasoning).
#   * MLX/Metal native path (out of scope: K3 deliverable per ADR
#     §11.7.0; this ladder establishes the PyTorch MPS upper bound
#     that K3 MLX must beat).
#
# Usage
# -----
#
#   bash scripts/review_pr_k2a_production_smoke_ladder_on_mac.sh
#
# Default rungs: 70 280  (~1.4k + ~5.6k tokens). Override:
#
#   RUNGS='70'              # just ctx70 (architecture sanity, ~1-2 min)
#   RUNGS='70 280 800'      # add ctx16k (will OOM Mac M4 24 GB; useful
#                             only on bigger boxes)
#
# Other env knobs propagate to the per-rung smoke script:
#
#   KL_LATTICE   (D4)
#   KL_Q_RANGE   (38)
#   STATEFUL     (1)
#   ABORT_ON_FAIL (0)        # set to 1 to stop the ladder at the
#                              first rung that produces no JSON
#                              report (default: continue, so the
#                              ladder JSON records all rungs even
#                              when some fail)
#
# Output
# ------
#
#   results/research/k2a_production_smoke_ladder_mac_<stamp>.json
#   results/research/logs/k2a_production_smoke_ladder_mac_<stamp>.log
#
#   Plus the per-rung JSONs and logs that the inner smoke script
#   produces (k2a_production_smoke_mac_ctxN_*.json each).
#
# JSON schema
# -----------
#
#   {
#     "schema_version": 1,
#     "kind": "k2a_production_smoke_ladder_mac",
#     "platform": "mac",
#     "stamp": <unix-int>,
#     "config": { kl_lattice, kl_q_range, stateful, rungs_lines },
#     "rungs": [
#       {
#         "ctx_lines": <int>,
#         "exit_code": <int>,
#         "report_path": <str|null>,
#         "log_path": <str>,
#         "metrics": {                          # null if exit_code != 0
#           "recall": 0.0|1.0,
#           "samples_correct": <int>,
#           "samples_total": <int>,
#           "mean_latency_s": <float>,
#           "mean_throughput_tokens_per_sec": <float>,
#           "sec_per_token": <float>,
#           "current_allocated_bytes": <int>,
#           "driver_allocated_bytes": <int>,
#           "driver_allocated_gb": <float>,
#           "effective_attention_fraction": <float>,
#           "decoded": <str>
#         },
#         "summary": {
#           "architecture_ok": <bool|null>,     # eff_attn_frac == 1.0
#           "memory_under_24gb": <bool|null>,   # driver_alloc < 24 GiB
#           "recall_hit": <bool|null>           # recall == 1.0
#         }
#       },
#       ...
#     ],
#     "ladder_summary": {
#       "rungs_completed": <int>,
#       "rungs_with_recall_hit": <int>,
#       "rungs_under_24gb": <int>,
#       "narrative": <str>
#     }
#   }

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUNGS="${RUNGS:-70 280}"
KL_LATTICE="${KL_LATTICE:-D4}"
KL_Q_RANGE="${KL_Q_RANGE:-38}"
STATEFUL="${STATEFUL:-1}"
ABORT_ON_FAIL="${ABORT_ON_FAIL:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

ladder_json="${out_dir}/k2a_production_smoke_ladder_mac_${stamp}.json"
ladder_log="${log_dir}/k2a_production_smoke_ladder_mac_${stamp}.log"

# ---------------------------------------------------------------------------
# Pre-flight: confirm the inner smoke script exists and has its own
# pre-flight checks. Fail fast if either is missing.
# ---------------------------------------------------------------------------
inner="scripts/review_pr_k2a_production_smoke_on_mac.sh"
if [[ ! -x "$inner" ]]; then
    echo "ERROR: $inner not found or not executable." | tee "$ladder_log" >&2
    echo "       Run from a checkout of the mac reviewer branch:" >&2
    echo "       git checkout AgentMemory/mac-k2a1-reviewer-production-shape-8e7f" >&2
    exit 1
fi

echo "==> Mac M4 K2.A production-shape ladder"  | tee "$ladder_log"
echo "    Rungs:           $RUNGS  (padding lines)" | tee -a "$ladder_log"
echo "    KL lattice:      $KL_LATTICE  (Q=$KL_Q_RANGE)" | tee -a "$ladder_log"
echo "    Stateful:        $STATEFUL" | tee -a "$ladder_log"
echo "    Abort on fail:   $ABORT_ON_FAIL" | tee -a "$ladder_log"
echo "    Output ladder:   $ladder_json" | tee -a "$ladder_log"
echo "    Output log:      $ladder_log" | tee -a "$ladder_log"
echo | tee -a "$ladder_log"

# ---------------------------------------------------------------------------
# Per-rung loop: invoke inner smoke, capture exit code + report path,
# accumulate JSON-builder fragments via Python.
# ---------------------------------------------------------------------------

rung_results=()  # space-separated tuples of "ctx:exit_code:report_path:log_path"

for ctx in $RUNGS; do
    echo "==================================================" | tee -a "$ladder_log"
    echo "  Rung ctx${ctx}  (~$(( ctx * 20 )) tokens)" | tee -a "$ladder_log"
    echo "==================================================" | tee -a "$ladder_log"

    rung_stamp="$(date +%s)"
    rung_report="${out_dir}/k2a_production_smoke_mac_ctx${ctx}_${rung_stamp}.json"
    rung_log="${log_dir}/k2a_production_smoke_mac_ctx${ctx}_${rung_stamp}.log"

    set +e
    CTX_LINES="$ctx" \
    KL_LATTICE="$KL_LATTICE" \
    KL_Q_RANGE="$KL_Q_RANGE" \
    STATEFUL="$STATEFUL" \
        bash "$inner" 2>&1 | tee -a "$ladder_log"
    rc=${PIPESTATUS[0]}
    set -e

    # The inner script writes its own output JSON at a path it picks
    # (k2a_production_smoke_mac_ctxN_<innerstamp>.json). Recover the
    # latest matching file as the rung's report.
    actual_report="$(
        ls -t "${out_dir}/k2a_production_smoke_mac_ctx${ctx}_"*.json 2>/dev/null \
            | head -n 1 || true
    )"
    if [[ -n "$actual_report" ]]; then
        rung_report="$actual_report"
    else
        # Inner script didn't produce a JSON — record empty path.
        rung_report=""
    fi

    rung_results+=("${ctx}|${rc}|${rung_report}|${rung_log}")
    echo | tee -a "$ladder_log"
    echo "  Rung ctx${ctx} exit=${rc}, report=${rung_report:-<none>}" | tee -a "$ladder_log"
    echo | tee -a "$ladder_log"

    if [[ "$rc" -ne 0 && "$ABORT_ON_FAIL" == "1" ]]; then
        echo "==> ABORT_ON_FAIL=1 set; stopping ladder at first failure." | tee -a "$ladder_log"
        break
    fi
done

# ---------------------------------------------------------------------------
# Aggregation: parse every rung's JSON (where present) for the four
# product-relevant metrics, build the ladder JSON.
# ---------------------------------------------------------------------------

# Pass rung_results through env so Python can parse without bash quoting hell.
export RUNG_RESULTS_RAW="$(printf '%s\n' "${rung_results[@]}")"
export LADDER_JSON_PATH="$ladder_json"
export LADDER_STAMP="$stamp"
export LADDER_KL_LATTICE="$KL_LATTICE"
export LADDER_KL_Q_RANGE="$KL_Q_RANGE"
export LADDER_STATEFUL="$STATEFUL"
export LADDER_RUNGS_LINES="$RUNGS"

python3 - <<'PYEOF' | tee -a "$ladder_log"
import json
import os
import sys

raw = os.environ.get("RUNG_RESULTS_RAW", "").strip()
if not raw:
    print("ERROR: no rungs ran", file=sys.stderr)
    sys.exit(1)

ladder_path = os.environ["LADDER_JSON_PATH"]
stamp = int(os.environ["LADDER_STAMP"])
kl_lattice = os.environ["LADDER_KL_LATTICE"]
kl_q_range = int(os.environ["LADDER_KL_Q_RANGE"])
stateful = os.environ["LADDER_STATEFUL"] == "1"
rungs_lines_raw = os.environ["LADDER_RUNGS_LINES"]
rungs_lines = [int(x) for x in rungs_lines_raw.split() if x.strip()]

GIB_24 = 24 * (1 << 30)
rung_records = []

for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split("|", 3)
    if len(parts) != 4:
        continue
    ctx_lines = int(parts[0])
    rc = int(parts[1])
    report_path = parts[2] or None
    log_path = parts[3]

    metrics = None
    summary = {
        "architecture_ok": None,
        "memory_under_24gb": None,
        "recall_hit": None,
    }

    if report_path and os.path.exists(report_path):
        try:
            doc = json.load(open(report_path))
        except Exception as e:
            metrics = {"parse_error": str(e)}
        else:
            v04 = doc.get("results", {}).get("v04_dlm_restored", {}) or {}
            mem_per = (
                doc.get("memory", {})
                   .get("per_config", {})
                   .get("v04_dlm_restored", {})
                or {}
            )
            attn = (
                doc.get("attention_window", {})
                   .get("per_config", {})
                   .get("v04_dlm_restored", {})
                or {}
            )

            recall = v04.get("recall")
            samples_correct = v04.get("samples_correct")
            samples_total = v04.get("samples_total")
            mean_latency = v04.get("mean_latency_s")
            mean_throughput = v04.get("mean_throughput_tokens_per_sec")
            decoded_list = v04.get("per_sample_decoded") or []
            decoded = decoded_list[0] if decoded_list else None
            current_alloc = mem_per.get("current_allocated_bytes")
            driver_alloc = mem_per.get("driver_allocated_bytes")
            eff_attn = attn.get("effective_attention_fraction_mean")

            sec_per_token = None
            decode_tokens_list = v04.get("per_sample_decode_tokens") or []
            decode_tokens = decode_tokens_list[0] if decode_tokens_list else None
            if mean_latency and decode_tokens and decode_tokens > 0:
                sec_per_token = float(mean_latency) / float(decode_tokens)

            metrics = {
                "recall": recall,
                "samples_correct": samples_correct,
                "samples_total": samples_total,
                "mean_latency_s": mean_latency,
                "mean_throughput_tokens_per_sec": mean_throughput,
                "sec_per_token": sec_per_token,
                "current_allocated_bytes": current_alloc,
                "driver_allocated_bytes": driver_alloc,
                "driver_allocated_gb": (
                    driver_alloc / (1 << 30) if driver_alloc else None
                ),
                "effective_attention_fraction": eff_attn,
                "decoded": decoded,
                "decode_tokens": decode_tokens,
            }
            summary["architecture_ok"] = (
                eff_attn is not None and abs(eff_attn - 1.0) < 1e-6
            )
            summary["memory_under_24gb"] = (
                driver_alloc is not None and driver_alloc < GIB_24
            )
            summary["recall_hit"] = (
                recall is not None and abs(recall - 1.0) < 1e-6
            )

    rung_records.append({
        "ctx_lines": ctx_lines,
        "ctx_tokens_approx": ctx_lines * 20,
        "exit_code": rc,
        "report_path": report_path,
        "log_path": log_path,
        "metrics": metrics,
        "summary": summary,
    })

rungs_completed = sum(1 for r in rung_records if r["exit_code"] == 0)
rungs_recall_hit = sum(1 for r in rung_records if r["summary"]["recall_hit"] is True)
rungs_under_24 = sum(1 for r in rung_records if r["summary"]["memory_under_24gb"] is True)

# Auto-generated narrative for the ADR row author. Honest about what
# the ladder does and does NOT establish.
narrative_lines = []
narrative_lines.append(
    f"Mac M4 24 GB unified memory, Gemma 3-1B (bf16), KL {kl_lattice} Q={kl_q_range}, "
    f"stateful={stateful}, single request per rung, PyTorch MPS sdpa."
)
for r in rung_records:
    if r["metrics"] is None or "parse_error" in (r["metrics"] or {}):
        narrative_lines.append(
            f"  ctx{r['ctx_lines']} (~{r['ctx_tokens_approx']} tokens): "
            f"FAILED (exit={r['exit_code']}, no parsable JSON)"
        )
        continue
    m = r["metrics"]
    s = r["summary"]
    drv_gb = m.get("driver_allocated_gb")
    drv_str = f"{drv_gb:.2f} GB" if drv_gb is not None else "n/a"
    spt = m.get("sec_per_token")
    spt_str = f"{spt:.2f}s/tok" if spt is not None else "n/a"
    recall = m.get("recall")
    recall_str = "hit" if s["recall_hit"] else ("miss" if recall == 0.0 else "n/a")
    arch_str = "✓" if s["architecture_ok"] else "✗"
    mem_ok = "✓" if s["memory_under_24gb"] else "✗"
    narrative_lines.append(
        f"  ctx{r['ctx_lines']} (~{r['ctx_tokens_approx']} tokens): "
        f"recall={recall_str}, {spt_str}, mem={drv_str} ({mem_ok} <24GB), "
        f"arch_window=100% ({arch_str})"
    )

ladder = {
    "schema_version": 1,
    "kind": "k2a_production_smoke_ladder_mac",
    "platform": "mac",
    "stamp": stamp,
    "config": {
        "model": "google/gemma-3-1b-it",
        "kl_lattice": kl_lattice,
        "kl_q_range": kl_q_range,
        "stateful": stateful,
        "rungs_lines": rungs_lines,
    },
    "rungs": rung_records,
    "ladder_summary": {
        "rungs_attempted": len(rung_records),
        "rungs_completed": rungs_completed,
        "rungs_with_recall_hit": rungs_recall_hit,
        "rungs_under_24gb": rungs_under_24,
        "narrative": "\n".join(narrative_lines),
    },
}

os.makedirs(os.path.dirname(ladder_path), exist_ok=True)
with open(ladder_path, "w") as f:
    json.dump(ladder, f, indent=2)

print()
print("=" * 60)
print("LADDER SUMMARY")
print("=" * 60)
print(ladder["ladder_summary"]["narrative"])
print()
print(f"Ladder JSON: {ladder_path}")
print(f"Rungs attempted: {len(rung_records)}, completed: {rungs_completed}, "
      f"recall hit: {rungs_recall_hit}, under 24GB: {rungs_under_24}")
PYEOF

agg_rc=${PIPESTATUS[0]}

if [[ "$agg_rc" -ne 0 ]]; then
    echo "==> Aggregation failed (exit=$agg_rc); ladder JSON may be incomplete." | tee -a "$ladder_log"
    exit "$agg_rc"
fi

echo | tee -a "$ladder_log"
echo "==> Done." | tee -a "$ladder_log"
echo "Commit ladder evidence:" | tee -a "$ladder_log"
echo "    git add $ladder_json $ladder_log" | tee -a "$ladder_log"
echo "    git add $out_dir/k2a_production_smoke_mac_ctx*_${stamp}*.json 2>/dev/null || true" | tee -a "$ladder_log"
echo "    git add $log_dir/k2a_production_smoke_mac_ctx*_${stamp}*.log  2>/dev/null || true" | tee -a "$ladder_log"
echo "    git commit -m 'Mac M4 K2.A production-shape ladder evidence (ctx70 + ctx280)'" | tee -a "$ladder_log"
echo "    git push" | tee -a "$ladder_log"

exit 0
