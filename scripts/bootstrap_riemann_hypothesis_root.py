#!/usr/bin/env python3
"""Bootstrap the pinned Mathlib Riemann Hypothesis root exactly once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoresearch.prefill.orchestration_state import load_checkpoint
from autoresearch.prefill.root_bootstrap import bootstrap_rh_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        default="~/.kakeya/agent_gan_proof_ledger.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.kakeya/autoresearch/proof_orchestration.json",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    args = parser.parse_args()
    ledger_path = Path(args.ledger).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    project_root = Path(args.project_root).expanduser().resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise SystemExit("orchestration checkpoint is missing")
    result = bootstrap_rh_root(
        project_root=project_root,
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        ledger=ledger,
    )
    print(json.dumps({
        "root_id": result.root_id,
        "proposition_hash": result.proposition_hash,
        "certificate_hash": result.certificate_hash,
        "ledger_version": result.ledger_version,
        "changed": result.changed,
        "source_cards": [item.card_id for item in result.source_cards],
        "candidates": [{
            "candidate_id": item.candidate_id,
            "elaborated": item.elaborated,
            "accepted": item.accepted,
            "rejection_codes": list(item.rejection_codes),
            "proposition_hash": item.proposition_hash,
        } for item in result.candidates],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
