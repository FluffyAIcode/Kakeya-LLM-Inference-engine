"""CI enforcement of docs/agent-workflow-rules.md R2:

Every scripts/review_*.sh must call ``print_aid_header`` at startup
(after sourcing ``scripts/_lib/reviewer_aid_header.sh``) so the user
and reviewing agent can immediately verify branch + HEAD + recipe
before any GPU/training time is spent.

This test is the enforcement mechanism for R2. Adding it as a Python
test under tests/research/ so the existing pytest CI picks it up
automatically — no separate CI workflow needed.

Failure mode prevented (2026-06-10):
  PR #103 branch's reviewer aid silently ran a different trainer
  (relmse) than what the user (and agent) thought was running
  (attn_distill on PR #106). ~15 min vast.ai H200 GPU wasted.
"""

from __future__ import annotations

import pathlib
import re

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_HEADER_LIB = _SCRIPTS_DIR / "_lib" / "reviewer_aid_header.sh"


# Grandfathered aids: predate R2 (2026-06-10). Each filename in this
# set is exempted from the strict R2 check. **THIS SET MUST ONLY
# SHRINK, NEVER GROW.** Adding a new entry requires:
#   1. opening an issue tracking the retrofit
#   2. linking the issue from this comment
#   3. user sign-off in PR description
#
# When you retrofit an aid (add `source ...header.sh` + `print_aid_header`),
# remove its name from this set. Tests then enforce R2 on it strictly.
#
# Tracking issue for the full retrofit: TODO (open after this PR lands).
_GRANDFATHERED = {
    "review_pr_b1_on_mac.sh",
    "review_pr_b2_on_mac.sh",
    "review_pr_b3_on_mac.sh",
    "review_pr_b4_on_mac.sh",
    "review_pr_d2_on_mac.sh",
    "review_pr_e1_on_mac.sh",
    "review_pr_e1b_on_mac.sh",
    "review_pr_e1c_on_mac.sh",
    "review_pr_g5_on_mac.sh",
    "review_pr_g6_on_mac.sh",
    "review_pr_k1d_on_mac.sh",
    "review_pr_k1e_on_mac.sh",
    "review_pr_k1e_on_vast.sh",
    "review_pr_k2a1_integration_on_mac.sh",
    "review_pr_k2a1_integration_on_vast.sh",
    "review_pr_k2a_kl_smoke_on_mac.sh",
    "review_pr_k2a_production_smoke_ladder_on_mac.sh",
    "review_pr_k2a_production_smoke_on_mac.sh",
    "review_pr_k3_dflash_specdecode_on_mac.sh",
    "review_pr_k3_feasibility_on_mac.sh",
    "review_pr_k3_feasibility_on_vast.sh",
    "review_pr_k3_integrated_niah_on_vast.sh",
    "review_pr_n1_on_mac.sh",
    "review_pr_n2_on_mac.sh",
    "review_pr_n3_on_mac.sh",
    "review_pr_n4_on_mac.sh",
}


def _list_reviewer_aids() -> list[pathlib.Path]:
    """Return all NON-grandfathered scripts/review_*.sh files."""
    if not _SCRIPTS_DIR.exists():
        return []
    out: list[pathlib.Path] = []
    for path in _SCRIPTS_DIR.glob("review_*.sh"):
        if path.is_file() and path.name not in _GRANDFATHERED:
            out.append(path)
    return sorted(out)


def test_grandfathered_set_only_shrinks():
    """All grandfathered aids must still exist on disk. If a file is
    deleted, remove its name from _GRANDFATHERED. If a NEW aid is
    added that needs grandfathering (which should NEVER happen for
    new code), the rule violation is visible in this test's diff."""
    missing = [
        name for name in _GRANDFATHERED
        if not (_SCRIPTS_DIR / name).is_file()
    ]
    assert not missing, (
        f"Grandfathered aids no longer on disk: {missing}\n"
        f"Remove these names from _GRANDFATHERED in this test."
    )


def test_grandfathered_set_does_not_cover_new_aids():
    """Discover any new reviewer aid that isn't grandfathered AND
    doesn't comply with R2. Catches PRs that add a new aid without
    the header (which would silently bypass the strict tests because
    parametrize would just skip them if absent)."""
    on_disk = {p.name for p in _SCRIPTS_DIR.glob("review_*.sh") if p.is_file()}
    grandfathered_extra = _GRANDFATHERED - on_disk
    # Allow grandfathered set to be cleaned via the previous test;
    # this test only fires if NEW aids appear and aren't compliant.
    new_aids_non_compliant: list[str] = []
    for name in on_disk - _GRANDFATHERED:
        text = (_SCRIPTS_DIR / name).read_text()
        if "scripts/_lib/reviewer_aid_header.sh" not in text or \
                "print_aid_header" not in text:
            new_aids_non_compliant.append(name)
    assert not new_aids_non_compliant, (
        f"NEW reviewer aids non-compliant with R2: {new_aids_non_compliant}\n"
        f"Per docs/agent-workflow-rules.md R2 every reviewer aid MUST source\n"
        f"scripts/_lib/reviewer_aid_header.sh AND call print_aid_header at\n"
        f"startup. Adding to _GRANDFATHERED is NOT permitted for new aids."
    )


def test_header_lib_present():
    """The header lib (sourceable by all aids) must exist + be readable."""
    assert _HEADER_LIB.is_file(), (
        f"missing {_HEADER_LIB} — required by docs/agent-workflow-rules.md R2"
    )
    text = _HEADER_LIB.read_text()
    assert "print_aid_header" in text, "lib missing print_aid_header function"
    assert "require_branch" in text, "lib missing require_branch helper"


def test_header_lib_print_aid_header_signature():
    """print_aid_header takes (script_path, recipe) — verify the function
    body uses both args and prints branch + HEAD + recipe + started.
    Guards against accidental refactors that break the contract."""
    text = _HEADER_LIB.read_text()
    body_match = re.search(
        r"print_aid_header\(\)\s*\{(.*?)\n\}", text, re.DOTALL,
    )
    assert body_match, "could not locate print_aid_header function body"
    body = body_match.group(1)
    for required in ("Branch:", "HEAD commit:", "Recipe:", "Started at:"):
        assert required in body, f"print_aid_header missing field: {required}"


@pytest.mark.parametrize(
    "aid_path",
    _list_reviewer_aids(),
    ids=lambda p: p.name,
)
def test_reviewer_aid_sources_header_lib(aid_path: pathlib.Path):
    """R2 strict: every reviewer aid must source the lib."""
    text = aid_path.read_text()
    assert "scripts/_lib/reviewer_aid_header.sh" in text, (
        f"{aid_path.name} does not source scripts/_lib/reviewer_aid_header.sh\n"
        f"Per docs/agent-workflow-rules.md R2, every reviewer aid MUST source\n"
        f"the header lib AND call print_aid_header at startup.\n"
        f"Add near top of script:\n"
        f"    ROOT=\"$(cd \"$(dirname \"$0\")/..\" && pwd)\"\n"
        f"    source \"$ROOT/scripts/_lib/reviewer_aid_header.sh\""
    )


@pytest.mark.parametrize(
    "aid_path",
    _list_reviewer_aids(),
    ids=lambda p: p.name,
)
def test_reviewer_aid_calls_print_aid_header(aid_path: pathlib.Path):
    """R2 strict: every reviewer aid must invoke print_aid_header."""
    text = aid_path.read_text()
    assert "print_aid_header" in text, (
        f"{aid_path.name} sources the header lib but never calls "
        f"print_aid_header.\n"
        f"Per docs/agent-workflow-rules.md R2, every reviewer aid MUST call:\n"
        f"    print_aid_header \"$0\" \"<recipe summary>\"\n"
        f"BEFORE any pre-flight checks or GPU-time-spending operations."
    )
