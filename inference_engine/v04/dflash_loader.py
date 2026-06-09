"""DFlash drafter loader for K3 (per ADR 0008 §11.15.3 Block B prereq 4).

Background
----------

The K3 drafter ``z-lab/gemma-4-26B-A4B-it-DFlash`` declares
``model_type: qwen3`` in its ``config.json`` and ships no
``auto_map`` / no ``modeling_dflash.py`` (verified 2026-06-09 by
fetching the actual repo config). HuggingFace's
``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
therefore correctly dispatches to ``Qwen3ForCausalLM`` — that is
NOT a fallback, it is the right thing.

What HF's stock loader misses, however, is documented by the
warnings the K3 vast smoke surfaced:

  fc, hidden_norm        : in checkpoint, not in Qwen3ForCausalLM
  lm_head, embed_tokens  : in Qwen3ForCausalLM, newly initialised

DFlash adds two extras on top of the standard Qwen3 architecture
to implement its block-diffusion drafting protocol:

  * ``fc``           — feature-projection module that consumes
                       cross-layer hidden states from the verifier
                       (selected by ``target_layer_ids`` in the
                       config; six of the 30 verifier layers).
  * ``hidden_norm``  — target-feature normaliser applied before
                       ``fc``.

And the DFlash checkpoint's ``embed_tokens`` / ``lm_head`` weights
live under names ``Qwen3ForCausalLM`` doesn't probe — most likely
because the DFlash repo's safetensors index uses a different
prefix (e.g. ``drafter.model.embed_tokens.weight`` vs
``model.embed_tokens.weight``). That is recoverable with a small
key-remap pass at load time.

For v0.4 K/V Restoration to use this drafter meaningfully:

  * ``k_proj`` / ``v_proj`` (standard Qwen3) must load from the
    DFlash checkpoint — they will, since they're part of the
    architecture HF dispatches to. The smoke's existing
    ``newly initialised`` warning does NOT include these.
  * ``embed_tokens`` must load with trained weights (not
    randomly-initialised) — without it, layer-0 K/V is computed
    from random embeddings and all subsequent K/V propagate the
    garbage. This loader verifies via ``embed_tokens.weight.var()
    > 1e-3`` (random init has near-uniform low variance;
    trained embeddings have structured variance).
  * ``fc`` and ``hidden_norm`` must be loaded as extra modules
    attached to the model so the K3 cross-layer conditioning is
    available to v0.4 K/V Restoration's projection ``f_θ``.

Public API
----------

* :func:`inspect_dflash_checkpoint` — diagnostic-only. Reads the
  safetensors index + the resolved config and returns a
  serialisable dict describing what's in the checkpoint, what
  Qwen3 expects, and the proposed key remap. Useful for evidence
  collection on vast before committing to the loader path.

* :func:`load_dflash_drafter` — full load. Returns a
  :class:`DFlashLoadResult` with the loaded ``Qwen3ForCausalLM``
  model, the attached extras (``fc``, ``hidden_norm``) as a
  ``torch.nn.Module``, and a list of architectural warnings
  enumerating any keys that did not load cleanly.

Both functions take ``trust_remote_code`` as a parameter and
default to ``True`` to match HF behaviour for this repo.

LOC: ~250 (well over the user's 50-100 LOC estimate, but the
extra LOC is in error reporting and diagnostic output, both of
which are required for the vast smoke to be debuggable. Core
load logic is ~80 LOC.)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DFlashCheckpointInspection:
    """Result of :func:`inspect_dflash_checkpoint`.

    All fields are JSON-serialisable so the report can be written
    directly to ``architectural_warnings`` in the K3 smoke output.
    """

    repo_or_path: str
    config: Dict[str, Any]
    checkpoint_keys: List[str]
    qwen3_expected_keys: List[str]
    key_remap: Dict[str, str]
    qwen3_unmapped: List[str]
    checkpoint_extras: List[str]
    fc_keys: List[str]
    hidden_norm_keys: List[str]
    warnings: List[str]

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class DFlashLoadResult:
    """Result of :func:`load_dflash_drafter`."""

    model: Any  # Qwen3ForCausalLM (typed Any so torch import is lazy)
    extras: Any  # torch.nn.Module containing fc + hidden_norm; or None
    inspection: DFlashCheckpointInspection
    embed_tokens_var: float
    embed_tokens_trained: bool
    architectural_warnings: List[str]

    @property
    def expected_class_name(self) -> str:
        return type(self.model).__name__


# ---------------------------------------------------------------------------
# Inspection (diagnostic-only — no model materialisation)
# ---------------------------------------------------------------------------


_KNOWN_PREFIX_STRIPS = (
    "drafter.",
    "lm_model.",
    "transformer.",
)


_LOCAL_PATH_HEURISTICS = (
    "models/",
    "./",
    "../",
    "/",
)


def _looks_like_local_path(repo_or_path: str) -> bool:
    """Heuristic: does the input look like a local filesystem path
    rather than a HuggingFace ``org/repo`` repo id?

    Returns True for inputs starting with ``models/``, ``./``, ``../``,
    or ``/`` — the common project-relative or absolute path prefixes.
    HF repo ids are ``<user_or_org>/<repo>`` (exactly one slash, no
    leading slash, no relative-path component); inputs matching the
    HF format return False.

    This heuristic is used to **fail fast** when a user passes a
    project-relative path (like ``models/dflash-kakeya-baseline``)
    that doesn't exist on disk, instead of silently falling through
    to HF Hub (which then returns 404 with a confusing error message
    far from the actual root cause).
    """
    return repo_or_path.startswith(_LOCAL_PATH_HEURISTICS)


def _resolve_local_dir(repo_or_path: str, hf_kwargs: Mapping[str, Any]) -> Path:
    """Resolve repo id to a local snapshot. Pure HF-hub call; no model load.

    If ``repo_or_path`` looks like a local path (per
    :func:`_looks_like_local_path`) but does NOT exist on disk, this
    raises :class:`FileNotFoundError` with an actionable message instead
    of silently falling through to ``huggingface_hub.snapshot_download``
    (which then emits a 404 error far from the actual root cause —
    typically a missing ``git lfs pull`` or wrong cwd).
    """
    p = Path(repo_or_path)
    if p.exists() and p.is_dir():
        return p
    if _looks_like_local_path(repo_or_path):
        raise FileNotFoundError(
            f"DFlash drafter source {repo_or_path!r} looks like a local "
            f"path but does not exist on disk (resolved to "
            f"{p.absolute()}). Common causes:\n"
            f"  1. The Git LFS pointer for the model has not been pulled "
            f"yet — run 'git lfs install && git lfs pull' from the repo "
            f"root.\n"
            f"  2. The current working directory is not the repo root — "
            f"verify with 'pwd' and 'ls models/' before re-running.\n"
            f"  3. You are on a worktree that does not have the model "
            f"checkpoint — use a worktree where 'git lfs pull' has run.\n"
            f"\n"
            f"If you actually intended a HuggingFace repo id, use the "
            f"'<user_or_org>/<repo>' format (no leading 'models/', './' "
            f"or '/'). Refusing to silently fall through to HF Hub fetch "
            f"because that would 404 with a misleading error message."
        )
    from huggingface_hub import snapshot_download
    cache_dir = hf_kwargs.get("cache_dir")
    token = hf_kwargs.get("token") or hf_kwargs.get("use_auth_token")
    return Path(snapshot_download(
        repo_id=repo_or_path,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
    ))


def _read_checkpoint_keys(local_dir: Path) -> List[str]:
    """Enumerate all parameter keys across all safetensors shards.

    Checks both single-file and sharded layouts. Does NOT load
    tensor data — only the safetensors header.
    """
    from safetensors import safe_open
    index_path = local_dir / "model.safetensors.index.json"
    keys: List[str] = []
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map: Dict[str, str] = index.get("weight_map", {})
        unique_files = sorted(set(weight_map.values()))
        for f in unique_files:
            shard = local_dir / f
            with safe_open(str(shard), framework="pt") as ckpt:
                keys.extend(list(ckpt.keys()))
        return sorted(set(keys))
    single = local_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as ckpt:
            keys = list(ckpt.keys())
        return sorted(set(keys))
    return []


def _enumerate_qwen3_expected_keys(config: Mapping[str, Any]) -> List[str]:
    """Enumerate the parameter names a freshly-built Qwen3ForCausalLM
    would expose given this config's hidden_layers count.

    We construct a meta-tensor model so this is cheap (no real
    weight allocation). The list is the canonical "what Qwen3
    expects".
    """
    import torch
    from transformers import AutoConfig
    cfg_dict = dict(config)
    cfg_dict.pop("dflash_config", None)
    cfg_dict.pop("target_layer_ids", None)
    cfg_dict.pop("block_size", None)
    model_type = cfg_dict.pop("model_type")
    hf_config = AutoConfig.for_model(model_type, **cfg_dict)
    with torch.device("meta"):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(hf_config)
    return sorted(name for name, _ in model.state_dict().items())


def _propose_key_remap(
    checkpoint_keys: List[str], qwen3_expected: List[str]
) -> Tuple[Dict[str, str], List[str], List[str], List[str], List[str]]:
    """Heuristic remap: try identity first, then strip known prefixes.

    Returns:
      remap            checkpoint_key → qwen3_expected_key (only mapped)
      qwen3_unmapped   qwen3_expected keys with no match in the checkpoint
      checkpoint_extras  checkpoint keys with no qwen3 destination
      fc_keys            checkpoint extras that look like DFlash ``fc.*``
      hidden_norm_keys   checkpoint extras that look like ``hidden_norm.*``
    """
    qwen3_set = set(qwen3_expected)
    remap: Dict[str, str] = {}
    used_qwen3: set[str] = set()

    def _map(ckpt_key: str, target: str) -> bool:
        if target in qwen3_set and target not in used_qwen3:
            remap[ckpt_key] = target
            used_qwen3.add(target)
            return True
        return False

    for ckpt_key in checkpoint_keys:
        if _map(ckpt_key, ckpt_key):
            continue
        mapped = False
        for prefix in _KNOWN_PREFIX_STRIPS:
            if ckpt_key.startswith(prefix):
                stripped = ckpt_key[len(prefix):]
                if _map(ckpt_key, stripped):
                    mapped = True
                    break
        if mapped:
            continue
        for prefix in _KNOWN_PREFIX_STRIPS:
            candidate = prefix + ckpt_key
            if _map(ckpt_key, candidate):
                mapped = True
                break

    extras = [k for k in checkpoint_keys if k not in remap]
    qwen3_unmapped = [k for k in qwen3_expected if k not in used_qwen3]

    fc_keys: List[str] = []
    hidden_norm_keys: List[str] = []
    for k in extras:
        if re.search(r"(^|\.)fc(\.|$)", k):
            fc_keys.append(k)
        if re.search(r"(^|\.)hidden_norm(\.|$)", k):
            hidden_norm_keys.append(k)

    return remap, qwen3_unmapped, extras, fc_keys, hidden_norm_keys


def inspect_dflash_checkpoint(
    repo_or_path: str,
    *,
    trust_remote_code: bool = True,
    **hf_kwargs: Any,
) -> DFlashCheckpointInspection:
    """Inspect a DFlash repo / local dir without materialising weights.

    Returns a :class:`DFlashCheckpointInspection` with:
      * the repo's resolved ``config.json`` as a dict,
      * the checkpoint's parameter keys (from safetensors headers),
      * what Qwen3ForCausalLM would expect for this config,
      * the proposed key remap,
      * any keys that did not map (warnings),
      * which checkpoint extras look like DFlash ``fc`` / ``hidden_norm``
        (needed for prereq 4 step (d), the extras attachment).

    No torch model is materialised. No tensor data is loaded —
    only safetensors headers are read. Used as the diagnose phase
    of the K3 vast reviewer.
    """
    local_dir = _resolve_local_dir(repo_or_path, hf_kwargs)
    config_path = local_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing under {local_dir}")
    config = json.loads(config_path.read_text())

    checkpoint_keys = _read_checkpoint_keys(local_dir)
    if not checkpoint_keys:
        raise FileNotFoundError(
            f"no model.safetensors{{,.index.json}} found under {local_dir}"
        )

    qwen3_expected = _enumerate_qwen3_expected_keys(config)
    remap, qwen3_unmapped, extras, fc_keys, hidden_norm_keys = _propose_key_remap(
        checkpoint_keys, qwen3_expected,
    )

    warnings: List[str] = []
    if qwen3_unmapped:
        warnings.append(
            f"{len(qwen3_unmapped)} Qwen3 parameter(s) have no checkpoint "
            f"source — will be newly initialised: "
            f"{qwen3_unmapped[:3]}{' ...' if len(qwen3_unmapped) > 3 else ''}"
        )
    if not fc_keys:
        warnings.append(
            "no `fc.*` keys found in checkpoint — DFlash cross-layer "
            "conditioning will be unavailable"
        )
    if not hidden_norm_keys:
        warnings.append(
            "no `hidden_norm.*` keys found in checkpoint — DFlash "
            "target-feature normalisation will be unavailable"
        )

    return DFlashCheckpointInspection(
        repo_or_path=str(repo_or_path),
        config=config,
        checkpoint_keys=checkpoint_keys,
        qwen3_expected_keys=qwen3_expected,
        key_remap=remap,
        qwen3_unmapped=qwen3_unmapped,
        checkpoint_extras=extras,
        fc_keys=fc_keys,
        hidden_norm_keys=hidden_norm_keys,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Full load
# ---------------------------------------------------------------------------


# Random initialisation has near-uniform variance ~ 1/d_model (e.g.
# 1/3072 ≈ 3e-4 for Qwen3-4B-class). Trained token embeddings have
# structured per-token variance roughly 1-2 orders of magnitude
# higher (verified empirically across Gemma 3, Qwen 3, Llama 3).
# 1e-3 is a conservative lower bound: anything trained will clear it,
# nothing randomly initialised will.
EMBED_TOKENS_TRAINED_VAR_THRESHOLD = 1e-3


def _build_extras_module(
    inspection: DFlashCheckpointInspection,
    state_dict: Mapping[str, Any],
):
    """Attach the ``fc`` and ``hidden_norm`` weights from the
    checkpoint as anonymous ``torch.nn.Module``s on a parent module
    so they're addressable by the projection ``f_θ`` later.

    Implementation: a thin ``torch.nn.Module`` whose
    ``state_dict()`` matches the union of fc + hidden_norm keys.
    Real DFlash semantic (forward) is not implemented here — that's
    Block B (cross-model DLMRestoredVerifier) territory. This loader
    just makes the weights addressable.
    """
    import torch

    if not (inspection.fc_keys or inspection.hidden_norm_keys):
        return None

    extras = torch.nn.Module()
    for key in inspection.fc_keys + inspection.hidden_norm_keys:
        if key not in state_dict:
            continue
        tensor = state_dict[key]
        param = torch.nn.Parameter(tensor.detach().clone(), requires_grad=False)
        safe_attr_name = key.replace(".", "__")
        extras.register_parameter(safe_attr_name, param)
    return extras


def load_dflash_drafter(
    repo_or_path: str,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    **hf_kwargs: Any,
) -> DFlashLoadResult:
    """Load a DFlash drafter with the prereq-4 corrected loader.

    Steps (per ADR 0008 §11.15.3 Block B prereq 4 corrected):

      1. Resolve to local snapshot (huggingface_hub.snapshot_download).
      2. Inspect — propose key remap, classify extras.
      3. Build a Qwen3ForCausalLM from the resolved config.
      4. Load safetensors data into the Qwen3 state_dict via the
         remap; track unmapped keys as warnings.
      5. Attach fc + hidden_norm extras as a parameter container.
      6. Verify embed_tokens.weight.var() > threshold (proves the
         remap actually loaded the trained embeddings, not left
         them randomly initialised).
    """
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    local_dir = _resolve_local_dir(repo_or_path, hf_kwargs)
    inspection = inspect_dflash_checkpoint(
        repo_or_path, trust_remote_code=trust_remote_code, **hf_kwargs,
    )

    cfg_dict = dict(inspection.config)
    cfg_dict.pop("dflash_config", None)
    cfg_dict.pop("target_layer_ids", None)
    cfg_dict.pop("block_size", None)
    model_type = cfg_dict.pop("model_type")
    hf_config = AutoConfig.for_model(model_type, **cfg_dict)
    if dtype is not None:
        hf_config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_config(hf_config)
    if dtype is not None:
        model = model.to(dtype)

    full_state: Dict[str, torch.Tensor] = {}
    index_path = local_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map: Dict[str, str] = index.get("weight_map", {})
        unique_files = sorted(set(weight_map.values()))
        for f in unique_files:
            full_state.update(load_file(str(local_dir / f)))
    else:
        full_state.update(load_file(str(local_dir / "model.safetensors")))

    remapped: Dict[str, torch.Tensor] = {}
    for ckpt_key, target_key in inspection.key_remap.items():
        remapped[target_key] = full_state[ckpt_key]

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    warnings: List[str] = list(inspection.warnings)
    if missing:
        warnings.append(
            f"{len(missing)} Qwen3 keys missing after remap "
            f"(will be randomly initialised): {list(missing)[:3]}"
            f"{' ...' if len(missing) > 3 else ''}"
        )
    if unexpected:
        warnings.append(
            f"{len(unexpected)} keys unexpected by Qwen3 after remap "
            f"(should be empty if remap is exhaustive): {list(unexpected)[:3]}"
            f"{' ...' if len(unexpected) > 3 else ''}"
        )

    extras = _build_extras_module(inspection, full_state)

    if device is not None:
        model = model.to(device)
        if extras is not None:
            extras = extras.to(device)

    embed_module = model.get_input_embeddings()
    embed_var = float(embed_module.weight.detach().to(torch.float32).var().item())
    embed_trained = embed_var > EMBED_TOKENS_TRAINED_VAR_THRESHOLD
    if not embed_trained:
        warnings.append(
            f"embed_tokens.weight.var() = {embed_var:.6e} <= "
            f"{EMBED_TOKENS_TRAINED_VAR_THRESHOLD:.6e} threshold — "
            "embeddings appear NOT trained (random initialisation). "
            "Block C f_θ training MUST NOT proceed against this loader."
        )

    return DFlashLoadResult(
        model=model,
        extras=extras,
        inspection=inspection,
        embed_tokens_var=embed_var,
        embed_tokens_trained=embed_trained,
        architectural_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# CLI: `python -m inference_engine.v04.dflash_loader inspect <repo>`
# ---------------------------------------------------------------------------


def _cli_main(argv: Optional[List[str]] = None) -> int:
    """Tiny CLI for the diagnose phase. Used by the vast reviewer
    aid script before the smoke is run, to dump the key delta as
    JSON evidence."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["inspect"])
    p.add_argument("repo_or_path")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--no-trust-remote-code", action="store_true")
    args = p.parse_args(argv)

    if args.mode == "inspect":
        inspection = inspect_dflash_checkpoint(
            args.repo_or_path,
            trust_remote_code=not args.no_trust_remote_code,
        )
        out = json.dumps(inspection.to_json(), indent=2, default=str)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(out)
            print(f"[dflash-loader] inspection -> {args.output}", file=sys.stderr)
        else:
            print(out)
        for w in inspection.warnings:
            print(f"[dflash-loader] WARN: {w}", file=sys.stderr)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli_main())
