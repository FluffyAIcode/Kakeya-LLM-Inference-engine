"""Pin the Mac-bridge workload interpreter (Layer B) + import self-check (Layer C).

A self-hosted runner's default ``python3`` can silently change across reboots /
OS upgrades (observed 2026-06-18: it flipped to a Python 3.14 without ``mlx_lm``,
breaking every full-engine preset with a deep ``ModuleNotFoundError``). The
mac-bridge executor used to invoke a bare ``python3`` for the workload, so it
inherited whatever interpreter happened to be first on ``PATH``.

This module makes the workload interpreter **explicit and verified**:

* **Layer B — resolution.** Build an ordered candidate list (a pinned
  ``KAKEYA_MAC_PYTHON``, common venv paths, then ``PATH`` pythons) and pick the
  first one that can import the gate module (``mlx_lm``); fall back to the first
  existing candidate otherwise.
* **Layer C — gate.** For presets whose workload needs ``mlx_lm`` (the ``mlx-`` /
  ``k3-`` engine families, minus the env-probe / upgrade tools that exist to
  diagnose/repair the env), fail fast with a clear message instead of a deep
  import error when no capable interpreter exists.

All functions here are pure / dependency-injected so they are unit-tested on the
Linux gate (the CLI ``scripts/mac_bridge/run_preset.py`` is a thin caller). See
``docs/skills/pin-selfhosted-runner-python-env-skill.md``.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence

# The single module whose absence broke the runner; importing it implies the
# full MLX-LM stack is wired for the interpreter.
GATE_MODULE = "mlx_lm"

# ``mlx-``/``k3-`` presets that must NOT be import-gated: these exist precisely
# to probe or repair the environment, so they must run even when mlx_lm is gone.
_IMPORT_GATE_SKIP = frozenset({"mlx-env-probe", "mlx-upgrade"})

SKILL_DOC = "docs/skills/pin-selfhosted-runner-python-env-skill.md"


def workload_python_candidates(
    environ: Mapping[str, str],
    *,
    which: Callable[[str], Optional[str]] = shutil.which,
    expanduser: Callable[[str], str] = os.path.expanduser,
) -> List[str]:
    """Ordered, de-duplicated interpreter candidates for the heavy workload.

    Priority: the explicit pin (``KAKEYA_MAC_PYTHON``), then conventional venv
    locations, then ``PATH`` pythons (a pinned minor version before the bare
    ``python3`` that a reboot may have repointed)."""
    raw = [
        environ.get("KAKEYA_MAC_PYTHON"),
        expanduser("~/kakeya-venv/bin/python"),
        expanduser("~/.venv/bin/python"),
        which("python3.13"),
        which("python3"),
    ]
    out: List[str] = []
    for c in raw:
        if c and c not in out:
            out.append(c)
    return out


@dataclass(frozen=True)
class ResolvedPython:
    """The interpreter chosen for the workload."""

    path: str
    gate_module_ok: bool   # whether ``path`` can import GATE_MODULE
    from_pin: bool         # whether it came from ``KAKEYA_MAC_PYTHON``


def resolve_workload_python(
    candidates: Sequence[str],
    can_import: Callable[[str], bool],
    *,
    pinned: Optional[str] = None,
) -> Optional[ResolvedPython]:
    """Pick the first candidate that can import :data:`GATE_MODULE`; otherwise
    the first candidate (a fallback whose ``gate_module_ok`` is ``False``).
    Returns ``None`` only when there are no candidates at all."""
    first: Optional[str] = None
    for c in candidates:
        if first is None:
            first = c
        if can_import(c):
            return ResolvedPython(c, True, c == pinned)
    if first is None:
        return None
    return ResolvedPython(first, False, first == pinned)


def preset_requires_gate(preset_name: str) -> bool:
    """True iff a preset's workload needs :data:`GATE_MODULE` (so a missing
    import must fail fast). The ``mlx-`` / ``k3-`` engine presets do; the
    env-probe and upgrade tools (which diagnose/repair the env) are exempt."""
    if preset_name in _IMPORT_GATE_SKIP:
        return False
    return preset_name.startswith(("mlx-", "k3-"))


def substitute_python(argv: Sequence[str], pybin: str) -> List[str]:
    """Rewrite a leading bare ``python3`` to the resolved interpreter ``pybin``.
    Non-``python3`` argv (e.g. ``bash run_kakeya_mac.sh``, which reads
    ``KAKEYA_MAC_PYTHON`` itself) is returned unchanged."""
    a = list(argv)
    if a and a[0] == "python3":
        a[0] = pybin
    return a


def gate_error_message(preset_name: str, pybin: str) -> str:
    """The fail-fast message when a gated preset has no mlx_lm-capable python."""
    return (
        f"runner python '{pybin}' cannot import {GATE_MODULE!r}, which preset "
        f"'{preset_name}' requires. The runner's default python likely changed "
        f"(e.g. after a reboot). Pin the venv via KAKEYA_MAC_PYTHON or the runner "
        f"agent PATH and reinstall the ML stack — see {SKILL_DOC}."
    )
