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
MAX_NEW_TOKENS = 512
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
