#!/usr/bin/env bash
# vast.ai (CUDA) reviewer aid for K3 integrated NIAH eval —
# the complete Kakeya inference engine product evidence on CUDA.
#
# Combines CrossModelDLMRestoredVerifier (verifier with sink+window
# cache + drafter K/V Restoration via f_θ) with the K1.E NIAH
# evaluation harness. This is the **K3 product gate**.
#
# Pre-flight requires:
#   1. HF_TOKEN (Gemma 4 is gated)
#   2. models/dflash-kakeya-baseline/ Git LFS pulled
#   3. f_θ checkpoint at $F_THETA_DIR (default
#      results/research/f_theta_v1/) — produced by
#      scripts/review_pr_k3_f_theta_train_on_vast.sh
#   4. CUDA + transformers 5.x (Gemma 4 support)
#
# Validates (per ADR 0008 §11.8 release gates):
#
#   1. Architectural correctness:
#      effective_attention_fraction = 1.0 at every NIAH ladder rung.
#      Verifier "sees" full context despite sink+window-only cache.
#
#   2. Memory bounded:
#      Sustained cross-model verifier KV-cache memory ≤ O(sink+window).
#
#   3. Recall preservation:
#      |recall_cross_model - recall_oracle| ≤ 5 pp at every rung
#      (ADR §11.8 criterion 1a). This is the architecturally-meaningful
#      gate (independent of base-model long-context capability).
#
# Env knobs (defaults):
#
#   F_THETA_DIR        results/research/f_theta_v1
#   N_SAMPLES          10        per ladder rung
#   SINK_SIZE          4
#   WINDOW_SIZE        64
#   MAX_NEW_TOKENS     24
#   SEED               42
#   CONTEXT_LADDER     '70 280'  padding-line counts; '70'≈1.4k, '280'≈5.6k tokens
#   SKIP_ORACLE=1                skip the full-attention oracle baseline
#                                (saves ~50% time but loses recall_delta gate)
#
# Usage:
#
#   HF_TOKEN=hf_xxx bash scripts/review_pr_k3_integrated_niah_on_vast.sh
#
# Quick sanity (1.4k context, 4 samples):
#
#   N_SAMPLES=4 CONTEXT_LADDER='70' \
#       HF_TOKEN=hf_xxx bash $0
#
# Output JSONs at:
#   results/research/k3_integrated_niah_ctx<N>_<stamp>.json (per rung)
#   results/research/logs/k3_integrated_niah_<stamp>.log (combined log)
#
# This is the production-evidence reviewer aid. After it passes:
#   * ADR §11.8 K3 product gate is empirically closed
#   * K3 production-scale Kakeya inference is validated on CUDA
#   * Mac MLX path follows (separate PR — instrument mlx_lm directly)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

F_THETA_DIR="${F_THETA_DIR:-results/research/f_theta_v1}"
N_SAMPLES="${N_SAMPLES:-10}"
SINK_SIZE="${SINK_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-42}"
CONTEXT_LADDER="${CONTEXT_LADDER:-70 280}"
SKIP_ORACLE="${SKIP_ORACLE:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
log="${log_dir}/k3_integrated_niah_vast_${stamp}.log"

echo "==> K3 integrated NIAH eval (vast.ai CUDA)"
echo "    Verifier:        google/gemma-4-26B-A4B-it"
echo "    Drafter:         models/dflash-kakeya-baseline"
echo "    f_θ checkpoint:  $F_THETA_DIR"
echo "    N samples / rung: $N_SAMPLES"
echo "    Sink × window:   ${SINK_SIZE} × ${WINDOW_SIZE}"
echo "    Context ladder:  $CONTEXT_LADDER"
echo "    Skip oracle:     $SKIP_ORACLE"
echo "    Log:             $log"
echo

# Pre-flight 1: HF token
if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "ERROR: no HF auth detected. Run 'huggingface-cli login' or 'export HF_TOKEN=...'."
    exit 1
fi

# Pre-flight 2: f_θ checkpoint
if [[ ! -d "$F_THETA_DIR" ]]; then
    echo "ERROR: f_θ directory '$F_THETA_DIR' missing."
    echo "       Train it first via:"
    echo "           HF_TOKEN=hf_xxx bash scripts/review_pr_k3_f_theta_train_on_vast.sh"
    exit 2
fi
if [[ ! -f "$F_THETA_DIR/f_theta_config.json" ]] || [[ ! -f "$F_THETA_DIR/f_theta_weights.pt" ]]; then
    echo "ERROR: '$F_THETA_DIR' missing f_theta_config.json or f_theta_weights.pt."
    ls -la "$F_THETA_DIR" 2>&1 | head -10
    exit 2
fi

# Pre-flight 3: drafter checkpoint
if [[ ! -f "models/dflash-kakeya-baseline/model.safetensors" ]]; then
    echo "ERROR: models/dflash-kakeya-baseline/ missing or LFS not pulled."
    echo "       Run: git lfs install && git lfs pull"
    exit 3
fi

# Pre-flight 4: CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>&1; then
    echo "ERROR: CUDA not available."
    exit 4
fi

flags=(
    --f-theta-dir "$F_THETA_DIR"
    --n-samples "$N_SAMPLES"
    --sink-size "$SINK_SIZE"
    --window-size "$WINDOW_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
)
[[ "$SKIP_ORACLE" == "1" ]] && flags+=(--skip-oracle)

# Run per-rung
exit_code=0
for n_lines in $CONTEXT_LADDER; do
    lo=$(( (n_lines * 85 + 50) / 100 ))
    hi=$(( (n_lines * 115 + 50) / 100 ))
    [[ $lo -lt 10 ]] && lo=10
    [[ $hi -lt $((lo + 1)) ]] && hi=$((lo + 1))
    rung_report="${out_dir}/k3_integrated_niah_ctx${n_lines}_${stamp}.json"

    echo "==> ctx${n_lines}: lines [$lo, $hi]  → $rung_report"
    PYTHONPATH=.:sdks/python python3 scripts/research/k3_integrated_niah_eval.py \
        --haystack-min-lines "$lo" \
        --haystack-max-lines "$hi" \
        --output "$rung_report" \
        "${flags[@]}" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        echo "==> ctx${n_lines} FAILED (exit=$rc); continuing to next rung"
        exit_code="$rc"
    fi
done

echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "==> K3 integrated NIAH eval PASS (all rungs)"
    echo "    Reports:"
    for n_lines in $CONTEXT_LADDER; do
        echo "        ${out_dir}/k3_integrated_niah_ctx${n_lines}_${stamp}.json"
    done
    echo
    echo "Inspect aggregates per rung:"
    echo "    for f in ${out_dir}/k3_integrated_niah_ctx*_${stamp}.json; do"
    echo "        python3 -c 'import json,sys; r=json.load(open(sys.argv[1]));"
    echo "            print(\"file:\", sys.argv[1])"
    echo "            print(\"  cross-model recall:\", r[\"results\"][\"k3_cross_model\"][\"recall\"])"
    echo "            print(\"  oracle recall:    \", r[\"results\"].get(\"oracle\",{}).get(\"recall\"))"
    echo "            print(\"  effective_attn:   \", r[\"attention_window\"][\"per_config\"][\"k3_cross_model\"][\"effective_attention_fraction_mean\"])"
    echo "            print(\"  recall_delta_pp:  \", r[\"gate\"][\"recall_delta_vs_oracle_pp\"])"
    echo "            print(\"  gate_5pp:         \", r[\"gate\"][\"recall_delta_within_5pp\"])'  \"\$f\""
    echo "    done"
    echo
    echo "Commit evidence:"
    echo "    git add ${out_dir}/k3_integrated_niah_ctx*_${stamp}.json $log"
    echo "    git commit -m 'K3 integrated NIAH evidence (cross-model + f_θ + sink+window)'"
    echo "    git push"
else
    echo "==> Some rungs FAILED (last exit=$exit_code)"
    echo "    See $log for details"
fi

exit "$exit_code"
