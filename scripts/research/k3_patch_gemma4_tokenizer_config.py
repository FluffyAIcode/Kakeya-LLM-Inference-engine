"""Patch Gemma 4 multimodal tokenizer_config.json so transformers 5.x
``_set_model_specific_special_tokens`` accepts it.

Background (2026-06-09)
-----------------------

The user-side K3 Mac smoke (after PR #99 merged) loaded the
``FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit`` verifier via mlx_lm and
hit:

    AttributeError: 'list' object has no attribute 'keys'

at::

    transformers/tokenization_utils_base.py:1210
        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES \\
            + list(special_tokens.keys())

The ``special_tokens`` argument is ``self.extra_special_tokens``,
which transformers 5.x expects to be a ``dict`` mapping token-name →
token-string for the multimodal extras (audio_token, image_token,
video_token, boi_token, eoi_token, boa_token, eoa_token).

The MLX-quantized variant's ``tokenizer_config.json`` ships
``extra_special_tokens`` as a ``list`` (or a different shape) which
breaks the ``.keys()`` call. This is an upstream-checkpoint bug that
this script patches locally.

Usage
-----

    python scripts/research/k3_patch_gemma4_tokenizer_config.py \\
        models/gemma-4-26B-A4B-it-mlx-4bit

The script:

  1. Reads ``<dir>/tokenizer_config.json``
  2. Inspects ``extra_special_tokens`` shape:
     * already a dict → no-op, exit 0
     * a list of strings → converts using known Gemma 4 token-name
       order (audio_token, image_token, video_token, ...)
     * a list of dicts (each {"name": ..., "value": ...} style) →
       reduces to flat dict
     * unrecognised shape → prints diagnostic + exits non-zero
  3. Backs up the original to ``tokenizer_config.json.pre-k3-patch.bak``
  4. Writes the patched config in place
  5. Prints a clear diff summary

Idempotent: re-running on an already-patched file is a no-op. The
``.bak`` file is created only on the first successful patch (subsequent
runs preserve the original backup).

Exit codes
----------

  0  patched (or already a dict — no-op)
  1  tokenizer_config.json missing
  2  tokenizer_config.json present but unparseable JSON
  3  ``extra_special_tokens`` shape unrecognised — manual fix required;
     diagnostic printed to stderr with the file's content
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


# Gemma 4 multimodal extra-special-token name → expected order.
# Order matters when extra_special_tokens is a flat list of strings
# without keys: we map them positionally.
#
# This list is derived from Gemma 4's published config.json fields
# (audio_token_id, image_token_id, video_token_id, boi_token_id,
# eoi_token_id, boa_token_id, eoa_token_id) — see
# https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json.
GEMMA4_EXTRA_SPECIAL_TOKEN_NAMES = [
    "audio_token",
    "image_token",
    "video_token",
    "boi_token",
    "eoi_token",
    "boa_token",
    "eoa_token",
]


def _looks_like_token_dict_entry(item: Any) -> bool:
    """Detect entries like ``{"name": "audio_token", "content": "<audio>"}``
    which some HF tokenizer configs ship as the per-entry shape."""
    return isinstance(item, dict) and (
        ("name" in item and ("content" in item or "value" in item))
        or "token" in item
    )


def _convert_list_to_dict(
    extra: list, expected_names: list = GEMMA4_EXTRA_SPECIAL_TOKEN_NAMES,
) -> Dict[str, str]:
    """Convert an ``extra_special_tokens`` list to a dict.

    Three list shapes handled:

      1. Empty list → return empty dict (transformers expects dict
         even when there are no extras; no-op semantically but
         shape-correct).
      2. List of strings → positional mapping using ``expected_names``.
         Length must match (≤) ``expected_names``; surplus entries
         get auto-named ``extra_token_N``.
      3. List of dicts ``[{"name": "x", "content": "y"}, ...]`` →
         reduce to flat dict ``{"x": "y", ...}``.
    """
    if not extra:
        return {}

    if all(isinstance(x, str) for x in extra):
        out: Dict[str, str] = {}
        for i, val in enumerate(extra):
            name = (
                expected_names[i] if i < len(expected_names)
                else f"extra_token_{i}"
            )
            out[name] = val
        return out

    if all(_looks_like_token_dict_entry(x) for x in extra):
        out = {}
        for entry in extra:
            name = entry.get("name") or entry.get("token")
            value = entry.get("content") or entry.get("value") or entry.get("token")
            if name is None or value is None:
                raise ValueError(
                    f"unrecognised dict entry shape in extra_special_tokens: "
                    f"{entry!r} (need 'name' + 'content' / 'value' keys)"
                )
            out[name] = value
        return out

    raise ValueError(
        f"extra_special_tokens list contains mixed or unrecognised entry "
        f"types: {[type(x).__name__ for x in extra[:5]]}"
    )


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("verifier_dir", help="Local directory containing tokenizer_config.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change without writing anything")
    args = ap.parse_args(argv)

    d = Path(args.verifier_dir)
    tc_path = d / "tokenizer_config.json"
    if not tc_path.is_file():
        print(f"ERROR: {tc_path} does not exist.", file=sys.stderr)
        return 1

    try:
        original_text = tc_path.read_text(encoding="utf-8")
        cfg = json.loads(original_text)
    except json.JSONDecodeError as e:
        print(f"ERROR: {tc_path} is not valid JSON: {e}", file=sys.stderr)
        return 2

    extra = cfg.get("extra_special_tokens")

    if extra is None:
        print(
            f"[k3-patch] {tc_path}: 'extra_special_tokens' field absent. "
            "Nothing to patch.", file=sys.stderr,
        )
        return 0

    if isinstance(extra, dict):
        print(
            f"[k3-patch] {tc_path}: 'extra_special_tokens' is already a "
            f"dict ({len(extra)} entries: {list(extra.keys())}). "
            "Nothing to patch.", file=sys.stderr,
        )
        return 0

    if not isinstance(extra, list):
        print(
            f"ERROR: {tc_path}'s 'extra_special_tokens' is unexpected "
            f"type {type(extra).__name__} (expected list or dict).",
            file=sys.stderr,
        )
        print(f"Content: {json.dumps(extra)[:200]}", file=sys.stderr)
        return 3

    # extra is a list — try to convert.
    print(
        f"[k3-patch] {tc_path}: 'extra_special_tokens' is a list "
        f"(len={len(extra)}); converting to dict.",
        file=sys.stderr,
    )
    print(f"[k3-patch]   list content: {json.dumps(extra)[:200]}",
          file=sys.stderr)

    try:
        converted = _convert_list_to_dict(extra)
    except ValueError as e:
        print(
            f"ERROR: cannot convert {tc_path}'s 'extra_special_tokens' "
            f"list to a dict automatically: {e}",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print("Manual fix needed. The list content is:", file=sys.stderr)
        print(json.dumps(extra, indent=2), file=sys.stderr)
        print(file=sys.stderr)
        print(
            "Edit tokenizer_config.json so 'extra_special_tokens' is a dict "
            "mapping token-name → token-string. Common Gemma 4 names: "
            f"{GEMMA4_EXTRA_SPECIAL_TOKEN_NAMES}.",
            file=sys.stderr,
        )
        return 3

    print(f"[k3-patch]   converted dict: {converted}", file=sys.stderr)

    if args.dry_run:
        print("[k3-patch] --dry-run set; not writing.", file=sys.stderr)
        return 0

    cfg["extra_special_tokens"] = converted

    bak_path = tc_path.with_suffix(".json.pre-k3-patch.bak")
    if not bak_path.exists():
        bak_path.write_text(original_text, encoding="utf-8")
        print(f"[k3-patch] backup written: {bak_path}", file=sys.stderr)
    else:
        print(
            f"[k3-patch] backup already exists at {bak_path}; not overwriting.",
            file=sys.stderr,
        )

    tc_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[k3-patch] patched: {tc_path}", file=sys.stderr)
    print(file=sys.stderr)
    print("Re-run the smoke to verify the tokenizer load succeeds:", file=sys.stderr)
    print(
        f"    bash scripts/research/k3_feasibility_smoke.py "
        f"--platform mac --verifier-path {args.verifier_dir} --skip-drafter",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
