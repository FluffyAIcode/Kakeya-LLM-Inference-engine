"""Mac-bridge preset allowlist + request-manifest schema.

Security posture (design doc §3): the Mac executes ONLY presets defined
here, with typed, bounded parameters. No string from a manifest is ever
interpolated into a shell — :func:`build_commands` returns argv lists
that the executor passes to ``subprocess.run`` without ``shell=True``.
Machine-local facts (model paths) come from the runner's environment,
referenced here as ``${ENV:VAR}`` placeholders the executor resolves
from ``os.environ`` — never from the manifest.

Pure stdlib so the Linux CI gate pins the allowlist semantics at 100%
coverage; the Mac executor imports exactly this module, so what CI
verifies is what the Mac enforces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MANIFEST_PATH = ".mac-bridge/request.json"
MANIFEST_SCHEMA_VERSION = 1

BRANCH_PREFIX = "mac-bridge/"

# Parameter bounds (design doc §2.2). Deliberately conservative: the
# bridge is for evidence runs and debugging, not for monopolizing the
# single Mac with open-ended workloads.
MAX_N_SAMPLES = 50
MAX_NEW_TOKENS = 2048  # backstop for chat; natural EOS stops well before this
MAX_BLOCK_SIZE = 16

_ENV_PLACEHOLDER = re.compile(r"^\$\{ENV:([A-Z][A-Z0-9_]*)\}$")
_NONCE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{3,63}$")


class ManifestError(ValueError):
    """A bridge manifest failed validation; nothing was executed."""


@dataclass(frozen=True)
class Preset:
    """One allowlisted Mac workload.

    ``command_templates`` are argv lists. Tokens may be:

    - plain strings (passed through),
    - ``${ENV:NAME}`` — resolved from the executor host's environment
      (missing variable = hard error, no fallback),
    - ``{param}`` — substituted with the validated parameter value.
    """

    name: str
    description: str
    command_templates: Tuple[Tuple[str, ...], ...]
    timeout_minutes: int
    # name -> (kind, default). kind ∈ {"int:n_samples", "int:max_new_tokens",
    # "int:block_size", "path:tests"}; None default = required.
    params: Mapping[str, Tuple[str, Optional[str]]] = field(default_factory=dict)
    # Run the K3 evidence gate over results/research after the commands.
    validate_reports: bool = False


def _harness_preset(
    name: str, description: str, mode_flag: str, *extra_flags: str,
) -> Preset:
    """The hardened-harness presets share everything but the mode flags."""
    return Preset(
        name=name,
        description=description,
        command_templates=(
            (
                "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                "--s5-exact-full-attn", mode_flag, *extra_flags,
                # Evidence runs decode the full budget: without this the
                # Gemma-4 <turn|> stop caps decode at ~8 tokens and the
                # report fails the SPEEDUP_DECODE_TOKENS gate rule.
                "--ignore-turn-stop",
                "--n-samples", "{n_samples}",
                "--max-new-tokens", "{max_new_tokens}",
                "--block-size", "{block_size}",
                "--prefill-chunk-size", "512",
                "--output",
                f"results/research/k3_mac_bridge_{name.replace('-', '_')}.json",
            ),
        ),
        timeout_minutes=120,
        params={
            "n_samples": ("int:n_samples", "5"),
            "max_new_tokens": ("int:max_new_tokens", "64"),
            "block_size": ("int:block_size", "4"),
        },
        validate_reports=True,
    )


PRESETS: Dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            name="mlx-env-probe",
            description="Probe Metal/MLX + mlx.distributed availability.",
            command_templates=(
                (
                    "python3", "-c",
                    "from inference_engine.backends.mlx.env import "
                    "probe_environment; print(probe_environment().render())",
                ),
            ),
            timeout_minutes=10,
        ),
        Preset(
            name="mlx-upgrade",
            description="Upgrade mlx + mlx-lm to the latest release on the Mac "
                        "runner, then re-probe the batch>1 L=1 quantized-decode "
                        "kernel bug after the upstream change. Prints mlx/mlx_lm "
                        "versions BEFORE, runs pip install --upgrade, prints "
                        "versions AFTER.",
            command_templates=(
                (
                    "python3", "-c",
                    "from inference_engine.backends.mlx.env import "
                    "probe_environment; print('BEFORE:', "
                    "probe_environment().render())",
                ),
                (
                    "python3", "-m", "pip", "install", "--upgrade",
                    "mlx", "mlx-lm",
                ),
                (
                    "python3", "-c",
                    "import importlib.metadata as m; "
                    "print('AFTER: mlx=' + m.version('mlx') + "
                    "' mlx_lm=' + m.version('mlx-lm'))",
                ),
            ),
            timeout_minutes=30,
        ),
        Preset(
            name="mlx-upstream-batch-probe",
            description="Self-contained probe (no inference_engine imports, "
                        "native model.make_cache(), L=1 batched decode): re-test "
                        "whether the upstream MLX B>1,L=1 quantized-decode kernel "
                        "bug is fixed after an mlx/mlx-lm upgrade. Reports "
                        "batched vs serialized per-session recall + "
                        "upstream_l1_batch_bug_fixed.",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_upstream_batch_probe.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--sessions", "8", "--haystack-lines", "60",
                    "--max-new-tokens", "24",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_upstream_batch_probe.json",
                ),
            ),
            timeout_minutes=90,
            validate_reports=False,
        ),
        Preset(
            name="mlx-backend-tests",
            description="Real-mlx truth for the MLX backend test suites.",
            command_templates=(
                ("python3", "-m", "pytest", "tests/backends/mlx/", "-q"),
            ),
            timeout_minutes=45,
        ),
        Preset(
            name="integration-tests",
            description="v0.3 GA integration gate (real Qwen3-0.6B).",
            command_templates=(
                ("python3", "-m", "pytest", "-m", "integration",
                 "tests/integration/", "-q"),
            ),
            timeout_minutes=60,
        ),
        Preset(
            name="mlx-batched-layer-diff",
            description="Localize the mlx_lm gemma-4 batch>1 decode bug: "
                        "per-layer hidden-state diff (batched row-i vs "
                        "serialized-i) at decode step 1; prints the first "
                        "divergent layer + its type/shared-KV status.",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_batched_layer_diff_diag.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--rows", "2", "--haystack-lines", "15",
                ),
            ),
            timeout_minutes=60,
            validate_reports=False,
        ),
        Preset(
            name="mlx-batched-manual-sdpa",
            description="Candidate fix: MLX batched multi-tenant with a manual "
                        "matmul-softmax SDPA replacing mx.fast.scaled_dot_"
                        "product_attention (works around the batch>1 + GQA "
                        "fast-kernel bug). Expect per-session recall -> 1.0.",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_batched_multitenant_bench.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--sessions", "8", "--haystack-lines", "60",
                    "--max-new-tokens", "24", "--manual-sdpa",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_batched_manual_sdpa.json",
                ),
            ),
            timeout_minutes=90,
            validate_reports=False,
        ),
        Preset(
            name="mlx-batched-layer-diff-concat",
            description="Layer-diff with the concat SinkWindowKVCache (no "
                        "in-place write) — if layer-0 output then matches, the "
                        "in-place cache write is the batch>1 bug.",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_batched_layer_diff_diag.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--rows", "2", "--haystack-lines", "15", "--kakeya-cache",
                ),
            ),
            timeout_minutes=60,
            validate_reports=False,
        ),
        Preset(
            name="mlx-batched-pad-decode",
            description="Candidate fix: MLX batched multi-tenant with the L>=2 "
                        "padded decode workaround (duplicate the new token so "
                        "every decode forward is length-2 and avoids mlx's L=1 "
                        "B>1 single-token quantized kernel — the suspected "
                        "core-kernel bug). Stays batched/parallel over "
                        "sessions, Python-only; forces the trimmable Kakeya S5 "
                        "cache. Expect per-session batched recall -> serialized "
                        "(1.0).",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_batched_multitenant_bench.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--sessions", "8",
                    "--haystack-lines", "60",
                    "--max-new-tokens", "24",
                    "--pad-decode", "--sink", "4", "--window", "64",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_batched_pad_decode.json",
                ),
            ),
            timeout_minutes=90,
            validate_reports=False,
        ),
        Preset(
            name="mlx-batched-kakeya-cache",
            description="Fix test: MLX batched multi-tenant with Kakeya's "
                        "concat-based SinkWindowKVCache (S5) instead of "
                        "mlx_lm's in-place buffer cache — should restore "
                        "per-session recall at batch>1 + bound memory.",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_batched_multitenant_bench.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--sessions", "8",
                    "--haystack-lines", "60",
                    "--max-new-tokens", "24",
                    "--kakeya-cache", "--sink", "4", "--window", "64",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_batched_kakeya_cache.json",
                ),
            ),
            timeout_minutes=90,
            validate_reports=False,
        ),
        Preset(
            name="mlx-batched-diag-short",
            description="Diagnostic: MLX batched multi-tenant on SHORT prompts "
                        "(haystack 8, below the sliding window so no "
                        "RotatingKVCache rotation) — isolates whether the "
                        "batched-recall bug is rotation-under-batch. Logs "
                        "per-row batched-vs-serialized first tokens.",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_batched_multitenant_bench.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--sessions", "4",
                    "--haystack-lines", "15",
                    "--max-new-tokens", "16",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_batched_diag_short.json",
                ),
            ),
            timeout_minutes=60,
            validate_reports=False,
        ),
        Preset(
            name="mlx-batched-multitenant",
            description="Mac analog of the §3.7 batched scheduler: N sessions "
                        "decoded in one batched MLX forward over the gemma "
                        "verifier vs serialized; reports aggregate tok/s, "
                        "speedup, per-session recall (recall-preserving native "
                        "cache).",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_batched_multitenant_bench.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--sessions", "{n_samples}",
                    "--haystack-lines", "60",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_batched_multitenant.json",
                ),
            ),
            timeout_minutes=90,
            params={
                "n_samples": ("int:n_samples", "8"),
                "max_new_tokens": ("int:max_new_tokens", "24"),
            },
            validate_reports=False,
        ),
        Preset(
            name="agent-capacity-loadtest",
            description="Test case 1: ramp concurrent agent connections "
                        "(independent gRPC channel + session each) against a "
                        "single RuntimeService; report max concurrent agents, "
                        "per-session bounded KV, node KV upper bound, latency "
                        "curve, server RSS. Uses the cpu Qwen3-0.6B verifier "
                        "(the integration-gate model; connection/admission "
                        "scaling is model-independent — the served MLX gemma "
                        "path is a separate v0.4 item).",
            command_templates=(
                (
                    "python3", "scripts/research/grpc_agent_capacity_loadtest.py",
                    "--backend", "cpu",
                    "--verifier-id", "Qwen/Qwen3-0.6B",
                    "--capacity", "256",
                    "--sink", "4", "--window", "64",
                    "--levels", "1,2,4,8,16,32,64,128,256",
                    "--gen-tokens", "4",
                    "--output",
                    "results/research/k3_mac_bridge_agent_capacity.json",
                ),
            ),
            timeout_minutes=90,
            validate_reports=False,
        ),
        Preset(
            name="mlx-multitenant-pressure",
            description="Multi-tenant resident-window pressure test + A/B vs "
                        "MLX-native: per-agent KV and max concurrent agents in "
                        "a memory budget, Kakeya S5 sink+window vs gemma's "
                        "native hybrid cache, on the real MLX gemma verifier.",
            command_templates=(
                (
                    "python3", "scripts/research/mlx_multitenant_pressure.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--mode", "both",
                    "--context-len", "2048",
                    "--sink", "4", "--window", "64",
                    "--max-agents", "64",
                    "--mem-budget-mb", "21000",
                    "--decode-steps", "16",
                    "--output",
                    "results/research/k3_mac_bridge_multitenant_pressure.json",
                ),
            ),
            timeout_minutes=120,
            validate_reports=False,
        ),
        Preset(
            name="agent-capacity-stress",
            description="Test case 1 (stress): push concurrent agents to 2048 "
                        "with a per-agent prefilled context (window 256), "
                        "raised FD limit, to probe the true connection ceiling "
                        "and the bounded-memory behavior (RSS vs agents) on the "
                        "Mac. cpu Qwen3-0.6B verifier.",
            command_templates=(
                (
                    "python3", "scripts/research/grpc_agent_capacity_loadtest.py",
                    "--backend", "cpu",
                    "--verifier-id", "Qwen/Qwen3-0.6B",
                    "--capacity", "2048",
                    "--sink", "4", "--window", "256",
                    "--context-len", "256",
                    "--levels", "1,4,8,16,32,48,64,96",
                    "--gen-tokens", "1",
                    "--output",
                    "results/research/k3_mac_bridge_agent_capacity_stress.json",
                ),
            ),
            timeout_minutes=120,
            validate_reports=False,
        ),
        _harness_preset(
            "k3-step1-incremental",
            "PR #109 Step-1 evidence: incremental restored decode.",
            "--incremental",
        ),
        _harness_preset(
            "k3-step2-fused",
            "PR #109 Step-2 evidence: fused engine must execute (blocks>0).",
            "--fused-specdecode",
        ),
        _harness_preset(
            "k3-native-baseline",
            "Labelled native-AR baseline run (cannot claim recall/speedup).",
            "--native-baseline-bypass",
        ),
        _harness_preset(
            "k3-step2-fused-allmlx",
            "Step-2 rescue evidence: fused engine with the ALL-MLX drafter "
            "(zero per-block bridge crossings).",
            "--fused-specdecode",
            "--all-mlx-drafter",
        ),
        Preset(
            name="k3-drafter-parity",
            description="All-MLX (bf16, shipping dtype) vs torch DFlash "
                        "drafter token parity.",
            command_templates=(
                (
                    "python3", "scripts/research/k3_mlx_drafter_parity.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--n-samples", "{n_samples}",
                    "--block-size", "{block_size}",
                    "--output",
                    "results/research/k3_mlx_drafter_parity.json",
                ),
            ),
            timeout_minutes=60,
            params={
                "n_samples": ("int:n_samples", "3"),
                "block_size": ("int:block_size", "8"),
            },
        ),
        Preset(
            name="k3-drafter-parity-fp32",
            description="Port-bug discriminator: all-MLX drafter at fp32 vs "
                        "the fp32 torch reference must match EXACTLY.",
            command_templates=(
                (
                    "python3", "scripts/research/k3_mlx_drafter_parity.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--mlx-dtype", "fp32",
                    "--n-samples", "{n_samples}",
                    "--block-size", "{block_size}",
                    "--output",
                    "results/research/k3_mlx_drafter_parity_fp32.json",
                ),
            ),
            timeout_minutes=60,
            params={
                "n_samples": ("int:n_samples", "3"),
                "block_size": ("int:block_size", "8"),
            },
        ),
        Preset(
            name="k3-kv-quant-eval",
            description="Rate-distortion shoot-out on the full-attn K/V: "
                        "mlx-native affine 8/4-bit vs KakeyaLattice D4/E8, "
                        "with real recall per arm. Decides whether an MLX "
                        "port of the KL codec is justified.",
            command_templates=(
                (
                    "python3", "scripts/research/k3_kv_quant_eval.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--n-samples", "{n_samples}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--output", "results/research/k3_kv_quant_eval.json",
                ),
            ),
            timeout_minutes=90,
            params={
                "n_samples": ("int:n_samples", "5"),
                "max_new_tokens": ("int:max_new_tokens", "32"),
            },
        ),
        Preset(
            name="k3-evidence-gate",
            description="Re-validate committed K3 Mac reports on-device.",
            command_templates=(
                ("python3", "scripts/validate_k3_reports.py",
                 "results/research"),
            ),
            timeout_minutes=10,
        ),
        Preset(
            name="pytest-path",
            description="One pytest target under tests/ (debugging).",
            command_templates=(
                ("python3", "-m", "pytest", "{path}", "-q"),
            ),
            timeout_minutes=45,
            params={"path": ("path:tests", None)},
        ),
        Preset(
            name="k3-fused-singlefused-probe",
            description="PROBE: single-fused (one drafter+26B graph) vs two-phase, "
                        "to classify the Metal instability. Small (n=2, gen=16) so a "
                        "pathological per-block eval is bounded. Compare block_eval_s "
                        "vs k3-fused-allmlx-code-trim (two-phase).",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode",
                    "--all-mlx-drafter", "--code-prompts", "--cuda-trim",
                    "--single-fused",
                    "--n-samples", "{n_samples}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--block-size", "{block_size}",
                    "--prefill-chunk-size", "512",
                    "--output",
                    "results/research/k3_mac_bridge_k3_fused_singlefused_probe.json",
                ),
            ),
            timeout_minutes=60,
            params={
                "n_samples": ("int:n_samples", "2"),
                "max_new_tokens": ("int:max_new_tokens", "16"),
                "block_size": ("int:block_size", "4"),
            },
            validate_reports=False,
        ),
        Preset(
            name="k3-beta-scorecard",
            description="Beta scorecard: all-MLX fused + CUDA-trim on NIAH ctx280 "
                        "(S5), natural stop. Reports Kakeya vs MLX-only oracle: "
                        "bounded KV (S5 vs naive), recall, context length, decode tok/s.",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode",
                    "--all-mlx-drafter", "--cuda-trim",
                    "--n-samples", "{n_samples}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--block-size", "{block_size}",
                    "--prefill-chunk-size", "512",
                    "--output",
                    "results/research/k3_mac_bridge_k3_beta_scorecard.json",
                ),
            ),
            timeout_minutes=120,
            params={
                "n_samples": ("int:n_samples", "5"),
                "max_new_tokens": ("int:max_new_tokens", "32"),
                "block_size": ("int:block_size", "8"),
            },
            validate_reports=False,
        ),
        Preset(
            name="k3-fused-allmlx-code-trim",
            description="CUDA-parity rollback test: all-MLX fused + --cuda-trim "
                        "(all-KVCache + native trim, keep accepted / drop rejected, "
                        "no re-forward) on the code-completion workload. Compare "
                        "decode-only tok/s vs k3-fused-allmlx-code (v3 carry).",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode",
                    "--all-mlx-drafter", "--code-prompts", "--cuda-trim",
                    "--n-samples", "{n_samples}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--block-size", "{block_size}",
                    "--prefill-chunk-size", "512",
                    "--output",
                    "results/research/k3_mac_bridge_k3_fused_allmlx_code_trim.json",
                ),
            ),
            timeout_minutes=120,
            params={
                "n_samples": ("int:n_samples", "8"),
                "max_new_tokens": ("int:max_new_tokens", "128"),
                "block_size": ("int:block_size", "4"),
            },
            validate_reports=False,
        ),
        Preset(
            name="k3-fused-allmlx-code",
            description="HONEST spec-decode throughput probe: all-MLX fused on a "
                        "code-completion workload (naturally-long, predictable gen "
                        "= the spec-decode sweet spot), natural stop. Reports "
                        "decode-only tok/s (fused vs oracle AR) + acceptance.",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode",
                    "--all-mlx-drafter", "--code-prompts",
                    # natural stop (no --ignore-turn-stop); code finishes itself
                    "--n-samples", "{n_samples}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--block-size", "{block_size}",
                    "--prefill-chunk-size", "512",
                    "--output",
                    "results/research/k3_mac_bridge_k3_fused_allmlx_code.json",
                ),
            ),
            timeout_minutes=120,
            params={
                "n_samples": ("int:n_samples", "8"),
                "max_new_tokens": ("int:max_new_tokens", "128"),
                "block_size": ("int:block_size", "4"),
            },
            validate_reports=False,
        ),
        Preset(
            name="k3-fused-allmlx-natural",
            description="Acceptance probe: all-MLX fused, NATURAL stop (no "
                        "--ignore-turn-stop) so generation ends at the real "
                        "answer. Compare mean_accept_len vs the forced "
                        "k3-step2-fused-allmlx (which over-generates).",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode",
                    "--all-mlx-drafter",
                    # deliberately NO --ignore-turn-stop (natural stop)
                    "--n-samples", "{n_samples}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--block-size", "{block_size}",
                    "--prefill-chunk-size", "512",
                    "--output",
                    "results/research/k3_mac_bridge_k3_fused_allmlx_natural.json",
                ),
            ),
            timeout_minutes=120,
            params={
                "n_samples": ("int:n_samples", "5"),
                "max_new_tokens": ("int:max_new_tokens", "48"),
                "block_size": ("int:block_size", "4"),
            },
            validate_reports=False,
        ),
        Preset(
            name="mlx-kakeya-chat-smoke",
            description="Run gemma-4 on the Kakeya-for-Mac (MLX) engine via the "
                        "interactive chat CLI in NON-interactive --scripted mode: "
                        "single-stream generation over the Kakeya S5 bounded "
                        "sink+window cache (sliding layers bounded; full-attn "
                        "layers full). Writes a transcript JSON so we can verify "
                        "gemma-4 responds coherently on the engine; the operator "
                        "runs the same script without --scripted for a real "
                        "interactive REPL on the Mac.",
            command_templates=(
                (
                    "python3", "scripts/chat_mlx_kakeya.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--sink", "4", "--window", "64",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--scripted",
                    "What is the capital of France? Answer in one short sentence."
                    "||Explain how proof-of-work works, step by step."
                    "||Name three primary colors.",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_kakeya_chat.json",
                ),
            ),
            timeout_minutes=45,
            params={"max_new_tokens": ("int:max_new_tokens", "64")},
            validate_reports=False,
        ),
        Preset(
            name="mlx-kakeya-fused-chat-smoke",
            description="Run gemma-4 on the FULL Kakeya fused engine (verifier + "
                        "DFlash proposer + f_θ + S5 bounded KV) via the harness "
                        "--chat --chat-scripted mode — NOT verifier-only. Verifies "
                        "the proposer is live (blocks>0, mean_accept_len>0) AND the "
                        "answer is correct AND KV is bounded, per turn. Writes a "
                        "transcript JSON.",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode",
                    "--all-mlx-drafter", "--cuda-trim",
                    "--sink-size", "4", "--window-size", "64",
                    "--block-size", "{block_size}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--prefill-chunk-size", "512",
                    "--chat",
                    "--chat-scripted",
                    "What is the capital of France? Answer in one short sentence."
                    "||Name three primary colors.",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_kakeya_fused_chat.json",
                ),
            ),
            timeout_minutes=60,
            params={
                "max_new_tokens": ("int:max_new_tokens", "64"),
                "block_size": ("int:block_size", "4"),
            },
            validate_reports=True,  # §4 liveness gate on-device (proposer/f_θ/fallback)
        ),
        Preset(
            name="mlx-kakeya-fused-chat-ftheta",
            description="Like mlx-kakeya-fused-chat-smoke but on the TORCH drafter "
                        "+ f_θ path with --force-f-theta: f_θ restoration ACTUALLY "
                        "RUNS each turn (projects proposer hidden → verifier K/V, "
                        "injected into the sliding layers) even though on gemma-4 "
                        "those K/V are recall-irrelevant (the exact layers carry "
                        "recall). Verifies the FULL verifier/proposer/f_θ pipeline: "
                        "report shows f_theta_ran=true + blocks>0. (No "
                        "--all-mlx-drafter; torch bridge path is slower.)",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode",
                    "--sink-size", "4", "--window-size", "64",
                    "--block-size", "{block_size}",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--prefill-chunk-size", "512",
                    "--chat",
                    "--chat-scripted",
                    "What is the capital of France? Answer in one short sentence."
                    "||Name three primary colors.",
                    "--output",
                    "results/research/k3_mac_bridge_mlx_kakeya_fused_chat_ftheta.json",
                ),
            ),
            timeout_minutes=90,
            params={
                "max_new_tokens": ("int:max_new_tokens", "32"),
                "block_size": ("int:block_size", "4"),
            },
            validate_reports=True,  # §4 liveness gate: asserts f_theta_ran on-device
        ),
        Preset(
            name="mlx-kakeya-launcher-smoke",
            description="Verify the one-command local launcher "
                        "scripts/run_kakeya_mac.sh runs the engine end-to-end on "
                        "the Mac: invokes it in --fast scripted mode (all-MLX "
                        "proposer path) with a fixed prompt and writes a "
                        "transcript. Proves launcher → harness → engine wiring + "
                        "env resolution + preflight on the real machine.",
            command_templates=(
                (
                    "bash", "scripts/run_kakeya_mac.sh", "--fast",
                    "--max-new-tokens", "{max_new_tokens}",
                    "--chat-scripted",
                    "What is the capital of France? Answer in one short sentence.",
                    "--output",
                    "results/research/k3_mac_bridge_launcher_smoke.json",
                ),
            ),
            timeout_minutes=45,
            params={"max_new_tokens": ("int:max_new_tokens", "64")},
            validate_reports=True,  # §4 liveness gate on-device
        ),
        Preset(
            name="mlx-kakeya-degen-probe",
            description="DEBUG (Phase-1): full f_θ fused engine on a LONG prompt "
                        "(--ignore-turn-stop, default 256 tokens) to characterize "
                        "the long-decode degeneration onset. Emits KDBG NDJSON to "
                        "stderr (captured in the bridge log) + transcript JSON. NOT "
                        "gated — the degeneration is the thing being measured.",
            command_templates=(
                (
                    "python3", "scripts/research/k3_integrated_niah_eval_mac.py",
                    "--verifier-path", "${ENV:KAKEYA_MAC_VERIFIER_PATH}",
                    "--drafter-id", "${ENV:KAKEYA_MAC_DRAFTER_ID}",
                    "--f-theta-dir", "${ENV:KAKEYA_MAC_FTHETA_DIR}",
                    "--s5-exact-full-attn", "--fused-specdecode", "--force-f-theta",
                    "--sink-size", "4", "--window-size", "64", "--block-size", "4",
                    "--max-new-tokens", "{max_new_tokens}", "--ignore-turn-stop",
                    "--chat", "--chat-native-ref",
                    "--chat-scripted", "请详细解释POW的工作原理",
                    "--output", "results/research/phase1_degeneration_chat.json",
                ),
            ),
            timeout_minutes=90,
            params={"max_new_tokens": ("int:max_new_tokens", "256")},
            validate_reports=False,
        ),
    )
}


@dataclass(frozen=True)
class BridgeRequest:
    """A validated bridge request (the parsed manifest)."""

    preset: Preset
    params: Mapping[str, str]
    ref: str
    requested_by: str
    nonce: str

    @property
    def branch_name(self) -> str:
        return f"{BRANCH_PREFIX}{self.preset.name}-{self.nonce}"

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "preset": self.preset.name,
            "params": dict(self.params),
            "ref": self.ref,
            "requested_by": self.requested_by,
            "nonce": self.nonce,
        }


def _validate_param(name: str, kind: str, raw: str) -> str:
    if kind.startswith("int:"):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ManifestError(f"param {name}={raw!r} is not an integer")
        bound = {
            "int:n_samples": MAX_N_SAMPLES,
            "int:max_new_tokens": MAX_NEW_TOKENS,
            "int:block_size": MAX_BLOCK_SIZE,
        }[kind]
        if not (1 <= value <= bound):
            raise ManifestError(
                f"param {name}={value} out of bounds [1, {bound}]")
        return str(value)
    # kind == "path:tests": repo-relative path under tests/, no traversal.
    if not isinstance(raw, str) or not raw:
        raise ManifestError(f"param {name} must be a non-empty string")
    if raw.startswith(("/", "~")) or ".." in raw.split("/"):
        raise ManifestError(
            f"param {name}={raw!r} must be repo-relative without traversal")
    if not (raw == "tests" or raw.startswith("tests/")):
        raise ManifestError(
            f"param {name}={raw!r} must resolve under tests/")
    return raw


def parse_manifest(data: Any) -> BridgeRequest:
    """Validate a decoded manifest dict into a :class:`BridgeRequest`.

    Raises :class:`ManifestError` on any deviation — unknown preset,
    unknown/missing/out-of-bounds params, malformed nonce. Nothing about
    a rejected manifest reaches a subprocess.
    """
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version={data.get('schema_version')!r}"
            f" (expected {MANIFEST_SCHEMA_VERSION})")
    preset_name = data.get("preset")
    preset = PRESETS.get(preset_name) if isinstance(preset_name, str) else None
    if preset is None:
        raise ManifestError(
            f"unknown preset {preset_name!r}; allowlist: {sorted(PRESETS)}")

    raw_params = data.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ManifestError("params must be an object")
    unknown = sorted(set(raw_params) - set(preset.params))
    if unknown:
        raise ManifestError(
            f"preset {preset.name!r} does not accept params: {unknown}")
    params: Dict[str, str] = {}
    for name, (kind, default) in preset.params.items():
        raw = raw_params.get(name, default)
        if raw is None:
            raise ManifestError(
                f"preset {preset.name!r} requires param {name!r}")
        params[name] = _validate_param(name, kind, str(raw))

    nonce = data.get("nonce")
    if not isinstance(nonce, str) or not _NONCE_RE.match(nonce):
        raise ManifestError(
            "nonce must match [a-z0-9][a-z0-9-]{3,63} (got "
            f"{nonce!r})")

    ref = data.get("ref")
    if not isinstance(ref, str) or not ref:
        raise ManifestError("ref must be a non-empty string")

    requested_by = data.get("requested_by")
    if not isinstance(requested_by, str) or not requested_by:
        raise ManifestError("requested_by must be a non-empty string")

    return BridgeRequest(
        preset=preset,
        params=params,
        ref=ref,
        requested_by=requested_by,
        nonce=nonce,
    )


def parse_manifest_text(text: str) -> BridgeRequest:
    """Parse + validate a manifest from its JSON text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    return parse_manifest(data)


def build_commands(
    request: BridgeRequest, env: Mapping[str, str],
) -> List[List[str]]:
    """Resolve a validated request into concrete argv lists.

    ``${ENV:NAME}`` tokens resolve from ``env``; a missing variable is a
    hard :class:`ManifestError` (no fallback — the runner must be
    configured per docs/ops/mac-m4-runner-setup.md). ``{param}`` tokens
    substitute already-validated parameter values. Output is ready for
    ``subprocess.run(argv)`` with no shell anywhere.
    """
    commands: List[List[str]] = []
    for template in request.preset.command_templates:
        argv: List[str] = []
        for token in template:
            env_match = _ENV_PLACEHOLDER.match(token)
            if env_match:
                var = env_match.group(1)
                if var not in env or not env[var]:
                    raise ManifestError(
                        f"preset {request.preset.name!r} needs runner env "
                        f"{var} (see docs/ops/mac-m4-runner-setup.md)")
                argv.append(env[var])
            elif token.startswith("{") and token.endswith("}"):
                argv.append(request.params[token[1:-1]])
            else:
                argv.append(token)
        commands.append(argv)
    return commands
