from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
LEAN_SOURCE = ROOT / "KakeyaLeanGate" / "WeilPositivity.lean"
ROUTE_DOC = ROOT / "docs" / "weil-positivity-route.md"


def test_weil_module_has_no_proof_shortcuts():
    source = LEAN_SOURCE.read_text()
    forbidden = ("sorry", "admit", "axiom")
    code = re.sub(r"/-.*?-/", "", source, flags=re.DOTALL)
    code = "\n".join(line.split("--", 1)[0] for line in code.splitlines())
    for token in forbidden:
        assert token not in code


def test_weil_module_compiles_under_pinned_toolchain():
    result = subprocess.run(
        ["lake", "env", "lean", str(LEAN_SOURCE.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_route_document_records_exact_unproved_bridge():
    text = ROUTE_DOC.read_text()
    assert "BridgeObligation W := IsPositive W ↔ RiemannHypothesis" in text
    assert re.search(r"supplies no\s+inhabitant", text, flags=re.IGNORECASE)
    assert "C_c^\\infty((0,\\infty),\\mathbb C)" in text
