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
