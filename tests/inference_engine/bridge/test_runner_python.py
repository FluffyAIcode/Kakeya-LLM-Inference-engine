"""Unit tests for the mac-bridge workload interpreter pinning (Layers B/C).

Pure / dependency-injected logic from ``inference_engine.bridge.runner_python``;
the CLI ``scripts/mac_bridge/run_preset.py`` is a thin caller (coverage-exempt).
"""

from __future__ import annotations

from inference_engine.bridge.runner_python import (
    GATE_MODULE,
    SKILL_DOC,
    ResolvedPython,
    gate_error_message,
    preset_requires_gate,
    resolve_workload_python,
    substitute_python,
    workload_python_candidates,
)


# --------------------------------------------------------------------------- #
# workload_python_candidates
# --------------------------------------------------------------------------- #
def test_candidates_prioritise_pin_then_venvs_then_path():
    env = {"KAKEYA_MAC_PYTHON": "/pin/bin/python"}
    which = {"python3.13": "/usr/bin/python3.13", "python3": "/usr/bin/python3"}.get
    cands = workload_python_candidates(
        env, which=which, expanduser=lambda p: p.replace("~", "/home/me"))
    assert cands == [
        "/pin/bin/python",
        "/home/me/kakeya-venv/bin/python",
        "/home/me/.venv/bin/python",
        "/home/me/Documents/Kakeya-LLM-Inference-engine-pr109/.venv-mac/bin/python3.13",
        "/home/me/Documents/Kakeya-LLM-Inference-engine-pr109/.venv-mac/bin/python",
        "/usr/bin/python3.13",
        "/usr/bin/python3",
    ]


def test_candidates_drop_empty_and_dedupe():
    # no pin, python3.13 missing, and python3 == an expanded venv path (dedupe).
    env: dict = {}
    which = {"python3.13": None, "python3": "/home/me/.venv/bin/python"}.get
    cands = workload_python_candidates(
        env, which=which, expanduser=lambda p: p.replace("~", "/home/me"))
    assert cands == [
        "/home/me/kakeya-venv/bin/python",
        "/home/me/.venv/bin/python",
        "/home/me/Documents/Kakeya-LLM-Inference-engine-pr109/.venv-mac/bin/python3.13",
        "/home/me/Documents/Kakeya-LLM-Inference-engine-pr109/.venv-mac/bin/python",
    ]
    assert None not in cands


# --------------------------------------------------------------------------- #
# resolve_workload_python
# --------------------------------------------------------------------------- #
def test_resolve_picks_first_importable():
    cands = ["/a/py", "/b/py", "/c/py"]
    r = resolve_workload_python(cands, lambda p: p == "/b/py", pinned="/a/py")
    assert r == ResolvedPython(path="/b/py", gate_module_ok=True, from_pin=False)


def test_resolve_marks_from_pin_when_pinned_is_importable():
    r = resolve_workload_python(["/pin/py", "/x/py"], lambda p: True,
                                pinned="/pin/py")
    assert r.path == "/pin/py" and r.gate_module_ok is True and r.from_pin is True


def test_resolve_falls_back_to_first_when_none_importable():
    r = resolve_workload_python(["/a/py", "/b/py"], lambda p: False,
                                pinned="/a/py")
    assert r == ResolvedPython(path="/a/py", gate_module_ok=False, from_pin=True)


def test_resolve_returns_none_without_candidates():
    assert resolve_workload_python([], lambda p: True) is None


# --------------------------------------------------------------------------- #
# preset_requires_gate
# --------------------------------------------------------------------------- #
def test_gate_required_for_mlx_and_k3_engine_presets():
    assert preset_requires_gate("mlx-kakeya-launcher-full") is True
    assert preset_requires_gate("k3-step2-fused") is True


def test_gate_skips_diagnostic_and_installer_and_non_engine():
    assert preset_requires_gate("mlx-env-probe") is False     # diagnostic
    assert preset_requires_gate("mlx-upgrade") is False       # installer
    assert preset_requires_gate("integration-tests") is False
    assert preset_requires_gate("agent-capacity-stress") is False


# --------------------------------------------------------------------------- #
# substitute_python / gate_error_message
# --------------------------------------------------------------------------- #
def test_substitute_rewrites_only_leading_bare_python3():
    assert substitute_python(["python3", "a.py", "--x"], "/v/py") == [
        "/v/py", "a.py", "--x"]
    # non-python3 argv0 (e.g. the launcher) is untouched.
    assert substitute_python(["bash", "run.sh"], "/v/py") == ["bash", "run.sh"]
    # empty argv is safe.
    assert substitute_python([], "/v/py") == []


def test_gate_error_message_names_module_preset_and_skill():
    msg = gate_error_message("mlx-kakeya-launcher-full", "/usr/bin/python3")
    assert GATE_MODULE in msg
    assert "mlx-kakeya-launcher-full" in msg
    assert "/usr/bin/python3" in msg
    assert SKILL_DOC in msg
