import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CARDS_PATH = ROOT / "docs/research/rh-jensen-source-cards.json"


def test_rh_jensen_source_cards_are_complete_and_honest() -> None:
    payload = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["route_id"] == "JENSEN_LAGUERRE_POLYA"

    cards = {card["id"]: card for card in payload["cards"]}
    assert {
        "gorz-2019-jensen",
        "osullivan-2021-xi-lp",
        "griffin-et-al-2022-effective",
        "mathlib-4.32.0-rc1-riemann",
        "mathlib-4.32.0-rc1-analysis",
    } <= cards.keys()

    allowed_statuses = {
        "SOURCE_VERIFIED_FORMAL_BRIDGE_OPEN",
        "SOURCE_VERIFIED_NOT_FORMALIZED",
        "FORMAL_LIBRARY_VERIFIED",
        "FORMAL_LIBRARY_AUDITED",
    }
    for card in cards.values():
        assert card["type"]
        assert card["citation"]
        assert card["url"].startswith("https://")
        assert card["locations"]
        assert card["claims"]
        assert card["status"] in allowed_statuses
        assert card["status"] not in {"PROVED", "FORMALIZED_EQUIVALENCE"}


def test_eventual_hyperbolicity_is_not_mislabeled_as_rh() -> None:
    payload = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    gorz = next(card for card in payload["cards"] if card["id"] == "gorz-2019-jensen")
    assert "EVENTUAL_HYPERBOLICITY_FIXED_DEGREE" in gorz["claims"]
    assert gorz["status"] == "SOURCE_VERIFIED_FORMAL_BRIDGE_OPEN"


def test_polya_schur_and_derivative_shift_are_source_pinned() -> None:
    payload = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    osullivan = next(
        card for card in payload["cards"] if card["id"] == "osullivan-2021-xi-lp"
    )
    assert {
        "LAGUERRE_POLYA_COMPACT_UNIFORM_CRITERION",
        "JENSEN_DERIVATIVE_SHIFT_IDENTITY",
        "DERIVATIVE_HYPERBOLICITY_PROPAGATION",
    } <= set(osullivan["claims"])
    locations = " ".join(osullivan["locations"])
    assert "Theorem 3.1" in locations
    assert "Equation (3.1)" in locations
    assert "d >= 1" in locations


def test_pinned_mathlib_gap_is_explicit() -> None:
    payload = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    analysis = next(
        card for card in payload["cards"] if card["id"] == "mathlib-4.32.0-rc1-analysis"
    )
    assert "NO_PINNED_DERIVATIVE_HYPERBOLICITY_CLOSURE_FOUND" in analysis["claims"]
