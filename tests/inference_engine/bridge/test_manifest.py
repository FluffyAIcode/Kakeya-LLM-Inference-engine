"""Unit tests for inference_engine.bridge.manifest (Mac bridge core).

This allowlist is the bridge's entire security argument (design doc
§3): what these tests pin is exactly what the Mac runner enforces,
because the executor imports this module. Every rejection path is
covered so a refactor cannot silently widen the command surface.

Coverage target: 100% on ``inference_engine/bridge/manifest.py``.
"""

from __future__ import annotations

import json

import pytest

from inference_engine.bridge.manifest import (
    BRANCH_PREFIX,
    MANIFEST_SCHEMA_VERSION,
    MAX_BLOCK_SIZE,
    MAX_N_SAMPLES,
    MAX_NEW_TOKENS,
    PRESETS,
    BridgeRequest,
    ManifestError,
    build_commands,
    parse_manifest,
    parse_manifest_text,
)

HARNESS_ENV = {
    "KAKEYA_MAC_VERIFIER_PATH": "/models/gemma-4-26B-A4B-it-mlx-4bit",
    "KAKEYA_MAC_DRAFTER_ID": "z-lab/gemma-4-26B-A4B-it-DFlash",
    "KAKEYA_MAC_FTHETA_DIR": "results/research/f_theta_v5_s5_sliding",
}


def _manifest(**overrides):
    data = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "preset": "mlx-env-probe",
        "params": {},
        "ref": "main",
        "requested_by": "test",
        "nonce": "1760000000-abc123",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Allowlist shape
# ---------------------------------------------------------------------------


def test_allowlist_contains_exactly_the_documented_presets():
    assert sorted(PRESETS) == [
        "integration-tests",
        "k3-drafter-parity",
        "k3-drafter-parity-fp32",
        "k3-evidence-gate",
        "k3-native-baseline",
        "k3-step1-incremental",
        "k3-step2-fused",
        "k3-step2-fused-allmlx",
        "mlx-backend-tests",
        "mlx-env-probe",
        "pytest-path",
    ]


def test_every_preset_has_timeout_and_description():
    for preset in PRESETS.values():
        assert preset.timeout_minutes > 0
        assert preset.description
        assert preset.command_templates


def test_harness_presets_validate_reports_others_do_not():
    gated = {name for name, p in PRESETS.items() if p.validate_reports}
    assert gated == {
        "k3-step1-incremental", "k3-step2-fused", "k3-native-baseline",
        "k3-step2-fused-allmlx",
    }


def test_allmlx_preset_carries_both_mode_flags():
    request = parse_manifest(_manifest(preset="k3-step2-fused-allmlx"))
    (argv,) = build_commands(request, HARNESS_ENV)
    assert "--fused-specdecode" in argv
    assert "--all-mlx-drafter" in argv
    assert "--ignore-turn-stop" in argv


def test_drafter_parity_preset_resolves():
    request = parse_manifest(_manifest(
        preset="k3-drafter-parity", params={"block_size": "8"}))
    (argv,) = build_commands(request, HARNESS_ENV)
    assert argv[1].endswith("k3_mlx_drafter_parity.py")
    assert HARNESS_ENV["KAKEYA_MAC_DRAFTER_ID"] in argv
    assert argv[argv.index("--block-size") + 1] == "8"


# ---------------------------------------------------------------------------
# parse_manifest acceptance
# ---------------------------------------------------------------------------


def test_minimal_valid_manifest_parses():
    request = parse_manifest(_manifest())
    assert request.preset.name == "mlx-env-probe"
    assert request.params == {}
    assert request.branch_name == f"{BRANCH_PREFIX}mlx-env-probe-1760000000-abc123"


def test_round_trip_through_manifest_dict():
    request = parse_manifest(_manifest())
    again = parse_manifest(request.to_manifest_dict())
    assert again == request


def test_parse_manifest_text_valid_and_invalid_json():
    request = parse_manifest_text(json.dumps(_manifest()))
    assert isinstance(request, BridgeRequest)
    with pytest.raises(ManifestError, match="not valid JSON"):
        parse_manifest_text("{nope")


def test_harness_preset_defaults_apply():
    request = parse_manifest(_manifest(preset="k3-step2-fused"))
    assert request.params == {
        "n_samples": "5", "max_new_tokens": "64", "block_size": "4",
    }


def test_harness_preset_params_override_within_bounds():
    request = parse_manifest(_manifest(
        preset="k3-step1-incremental",
        params={"n_samples": "10", "max_new_tokens": "128", "block_size": "8"},
    ))
    assert request.params == {
        "n_samples": "10", "max_new_tokens": "128", "block_size": "8",
    }


# ---------------------------------------------------------------------------
# parse_manifest rejection paths
# ---------------------------------------------------------------------------


def test_rejects_non_dict_and_wrong_schema():
    with pytest.raises(ManifestError, match="JSON object"):
        parse_manifest(["not", "a", "dict"])
    with pytest.raises(ManifestError, match="schema_version"):
        parse_manifest(_manifest(schema_version=99))


def test_rejects_unknown_preset_listing_allowlist():
    with pytest.raises(ManifestError, match="allowlist"):
        parse_manifest(_manifest(preset="rm-rf-everything"))
    with pytest.raises(ManifestError, match="allowlist"):
        parse_manifest(_manifest(preset=None))


def test_rejects_unknown_params():
    with pytest.raises(ManifestError, match="does not accept params"):
        parse_manifest(_manifest(params={"shell": "evil"}))
    with pytest.raises(ManifestError, match="params must be an object"):
        parse_manifest(_manifest(params="evil"))


def test_rejects_missing_required_param():
    with pytest.raises(ManifestError, match="requires param 'path'"):
        parse_manifest(_manifest(preset="pytest-path"))


def test_rejects_out_of_bounds_ints():
    for name, bad in (
        ("n_samples", str(MAX_N_SAMPLES + 1)),
        ("max_new_tokens", str(MAX_NEW_TOKENS + 1)),
        ("block_size", str(MAX_BLOCK_SIZE + 1)),
        ("n_samples", "0"),
        ("n_samples", "-3"),
    ):
        with pytest.raises(ManifestError, match="out of bounds"):
            parse_manifest(_manifest(
                preset="k3-step2-fused", params={name: bad}))


def test_rejects_non_integer_int_params():
    with pytest.raises(ManifestError, match="not an integer"):
        parse_manifest(_manifest(
            preset="k3-step2-fused", params={"n_samples": "five; rm -rf /"}))


def test_pytest_path_traversal_and_escape_rejected():
    for bad in ("/etc/passwd", "~/x", "tests/../scripts/serve.py",
                "scripts/serve.py", ""):
        with pytest.raises(ManifestError):
            parse_manifest(_manifest(
                preset="pytest-path", params={"path": bad}))


def test_pytest_path_accepts_tests_subpaths():
    for ok in ("tests", "tests/backends/mlx/test_fused_specdecode.py",
               "tests/integration/"):
        request = parse_manifest(_manifest(
            preset="pytest-path", params={"path": ok}))
        assert request.params["path"] == ok


def test_rejects_bad_nonce_ref_requested_by():
    with pytest.raises(ManifestError, match="nonce"):
        parse_manifest(_manifest(nonce="UPPER CASE!"))
    with pytest.raises(ManifestError, match="nonce"):
        parse_manifest(_manifest(nonce=None))
    with pytest.raises(ManifestError, match="ref"):
        parse_manifest(_manifest(ref=""))
    with pytest.raises(ManifestError, match="requested_by"):
        parse_manifest(_manifest(requested_by=""))


# ---------------------------------------------------------------------------
# build_commands
# ---------------------------------------------------------------------------


def test_simple_preset_builds_fixed_argv():
    request = parse_manifest(_manifest(preset="mlx-backend-tests"))
    commands = build_commands(request, {})
    assert commands == [
        ["python3", "-m", "pytest", "tests/backends/mlx/", "-q"],
    ]


def test_pytest_path_param_substitution_is_argv_level():
    request = parse_manifest(_manifest(
        preset="pytest-path",
        params={"path": "tests/backends/mlx/test_env.py"}))
    commands = build_commands(request, {})
    assert commands == [
        ["python3", "-m", "pytest", "tests/backends/mlx/test_env.py", "-q"],
    ]


def test_harness_preset_resolves_env_and_params():
    request = parse_manifest(_manifest(
        preset="k3-step2-fused",
        params={"n_samples": "7", "max_new_tokens": "96", "block_size": "8"},
    ))
    (argv,) = build_commands(request, HARNESS_ENV)
    assert argv[0:2] == ["python3", "scripts/research/k3_integrated_niah_eval_mac.py"]
    assert HARNESS_ENV["KAKEYA_MAC_VERIFIER_PATH"] in argv
    assert "--fused-specdecode" in argv
    assert "--ignore-turn-stop" in argv  # full decode budget (gate rule)
    assert argv[argv.index("--n-samples") + 1] == "7"
    assert argv[argv.index("--max-new-tokens") + 1] == "96"
    assert argv[argv.index("--block-size") + 1] == "8"
    # No unresolved placeholders of either kind survive.
    assert not [t for t in argv if t.startswith("${ENV:")]
    assert not [t for t in argv if t.startswith("{") and t.endswith("}")]


def test_step1_and_baseline_presets_carry_their_mode_flags():
    incr = parse_manifest(_manifest(preset="k3-step1-incremental"))
    (argv_incr,) = build_commands(incr, HARNESS_ENV)
    assert "--incremental" in argv_incr
    base = parse_manifest(_manifest(preset="k3-native-baseline"))
    (argv_base,) = build_commands(base, HARNESS_ENV)
    assert "--native-baseline-bypass" in argv_base


def test_missing_runner_env_is_a_hard_error():
    request = parse_manifest(_manifest(preset="k3-step2-fused"))
    with pytest.raises(ManifestError, match="KAKEYA_MAC_VERIFIER_PATH"):
        build_commands(request, {})
    partial = dict(HARNESS_ENV)
    partial["KAKEYA_MAC_DRAFTER_ID"] = ""
    with pytest.raises(ManifestError, match="KAKEYA_MAC_DRAFTER_ID"):
        build_commands(request, partial)
