"""HuggingFace model-cache pre-warm helpers.

The v0.3 runtime loads its verifier from the HuggingFace cache at
startup. Without a pre-warmed cache a first-run download blocks the
server boot for 1-10 minutes (depending on bandwidth) with no
progress feedback — a poor first-time-user experience.

This module exposes:

  * :data:`HF_CACHE_DEFAULT` — canonical cache root.
  * :func:`cache_dir_for_model` — the directory a given HF model id
    lands in under the cache.
  * :func:`is_model_in_cache` — fast read-only check; no network.
  * :func:`snapshot_size_bytes` — total bytes resident on disk for a
    cached model (informational).
  * :func:`prewarm_model_id` — explicit download with progress;
    raises on failure rather than silently re-trying.

These helpers are platform-neutral (no torch / mlx imports) so they
run quickly during the gRPC server's pre-flight check and don't
trigger the lazy-loaded backend dependencies. The CLI driver lives
at ``scripts/kakeya_prewarm.py``.

Per ADR 0008 §9: this is verifier-independent infrastructure;
exercised by Linux unit tests against synthetic cache directories.
The full HF download path is exercised by the integration suite +
the Mac M4 reviewer aid (``scripts/review_pr_g5_on_mac.sh``).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


HF_CACHE_DEFAULT = Path(
    os.environ.get(
        "HF_HUB_CACHE",
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    )
)
"""Default HuggingFace cache root.

Resolution order (matches transformers / huggingface_hub):
  1. ``$HF_HUB_CACHE`` if set.
  2. ``$HF_HOME`` if set (cache lives under ``hub/`` subdir).
  3. ``~/.cache/huggingface`` (cache lives under ``hub/`` subdir).

The directory we actually look in is :data:`HF_CACHE_DEFAULT` / ``hub``
when this module's helpers convert a model id to a path; ``HF_HOME``
historically pointed at the root, ``HF_HUB_CACHE`` at the ``hub`` dir
directly. We normalize by always appending ``hub`` for default-cache
inspection unless the caller already passed a directory ending in
``hub`` or ``models--*``.
"""


def _hub_root(cache_root: Path) -> Path:
    """Return the ``hub/`` subdirectory of a cache root.

    Keeps callers from caring whether they passed ``HF_HOME``-style
    or ``HF_HUB_CACHE``-style. Idempotent: passing a ``hub``-suffixed
    path returns it unchanged.
    """
    cache_root = Path(cache_root)
    if cache_root.name == "hub":
        return cache_root
    return cache_root / "hub"


def cache_dir_for_model(
    model_id: str, *, cache_root: Optional[Path] = None,
) -> Path:
    """Return the directory under the HF cache where a model id lands.

    HF caches use a ``models--<owner>--<repo>`` directory naming
    scheme. This helper computes the path WITHOUT any I/O — it does
    NOT check whether the directory exists. Pair with
    :func:`is_model_in_cache` for the existence check.
    """
    if "/" not in model_id:
        raise ValueError(
            f"model_id must be 'owner/repo' shape, got {model_id!r}"
        )
    flat = "models--" + model_id.replace("/", "--")
    root = _hub_root(cache_root or HF_CACHE_DEFAULT)
    return root / flat


def is_model_in_cache(
    model_id: str, *, cache_root: Optional[Path] = None,
) -> bool:
    """Read-only check: is the model already cached on disk?

    Only checks for the existence of the model's cache directory and
    that it contains at least one snapshot. Does NOT validate that
    the snapshot is complete / consistent — a partial download leaves
    a directory tree in place. The pre-warm CLI (:func:`prewarm_model_id`)
    is the canonical source of "fully downloaded".
    """
    cache_dir = cache_dir_for_model(model_id, cache_root=cache_root)
    if not cache_dir.is_dir():
        return False
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(snapshots.iterdir())


def snapshot_size_bytes(
    model_id: str, *, cache_root: Optional[Path] = None,
) -> int:
    """Total bytes resident on disk for a cached model.

    Sums the size of every regular file under the model's cache
    directory. Returns 0 if the model isn't in cache. Fast for
    typical model directory sizes (Qwen3-0.6B = ~12 files, ~1.2 GB).
    """
    cache_dir = cache_dir_for_model(model_id, cache_root=cache_root)
    if not cache_dir.is_dir():
        return 0
    total = 0
    for entry in cache_dir.rglob("*"):
        # Resolve symlinks; HF cache uses symlinks heavily under
        # snapshots/<rev>/. Skip dangling symlinks rather than
        # raising — partial downloads can leave them behind.
        try:
            if entry.is_symlink():
                target = entry.resolve(strict=False)
                if target.is_file():
                    total += target.stat().st_size
            elif entry.is_file():
                total += entry.stat().st_size
        except OSError:  # pragma: no cover - filesystem races
            continue
    return total


@dataclass(frozen=True)
class PrewarmStatus:
    """Result of a :func:`prewarm_model_id` call."""

    model_id: str
    cache_dir: Path
    snapshot_bytes: int
    was_already_cached: bool

    def human(self) -> str:
        action = "already cached" if self.was_already_cached else "downloaded"
        return (
            f"{self.model_id}: {action} at {self.cache_dir} "
            f"({self.snapshot_bytes / (1024 * 1024):.1f} MiB on disk)"
        )


def prewarm_model_id(
    model_id: str,
    *,
    cache_root: Optional[Path] = None,
    include_tokenizer: bool = True,
    progress_callback=None,
) -> PrewarmStatus:
    """Ensure a HuggingFace model + tokenizer are fully downloaded.

    Idempotent: returns ``was_already_cached=True`` immediately if the
    model is already on disk. Otherwise runs ``snapshot_download``
    (via huggingface_hub, the standard tool) which surfaces a
    progress bar by default and uses HF Hub's resume-friendly
    chunked downloads.

    ``include_tokenizer=False`` skips the tokenizer download — useful
    for inference-only workflows that already have the tokenizer
    elsewhere. Default True because v0.3's verifier-side code needs
    both weights and tokenizer config.

    The ``progress_callback`` parameter is reserved for future use;
    huggingface_hub's standard tqdm bar is what users see today.

    Raises on download failure (network error, permission denied,
    disk full); does NOT silently fall back to "best effort".
    """
    del progress_callback  # reserved for v0.4

    if is_model_in_cache(model_id, cache_root=cache_root):
        return PrewarmStatus(
            model_id=model_id,
            cache_dir=cache_dir_for_model(model_id, cache_root=cache_root),
            snapshot_bytes=snapshot_size_bytes(
                model_id, cache_root=cache_root,
            ),
            was_already_cached=True,
        )

    # Lazy import: keep the module's top-level import surface tiny
    # (huggingface_hub itself is fine but pulls in transitive deps).
    from huggingface_hub import snapshot_download

    download_kwargs = {"repo_id": model_id}
    if cache_root is not None:
        download_kwargs["cache_dir"] = str(_hub_root(cache_root))
    # The default `allow_patterns` is None which downloads everything;
    # if the caller wants only weights, they can post-filter. v0.3
    # downloads the full snapshot — tokenizer + weights + config in
    # one call. Removing files saves <50 MB on Qwen3-0.6B; not worth
    # the API complexity here.
    if not include_tokenizer:
        download_kwargs["ignore_patterns"] = [
            "tokenizer*",
            "vocab*",
            "merges*",
            "*.txt",  # tokenizer.json variants
        ]

    snapshot_download(**download_kwargs)

    return PrewarmStatus(
        model_id=model_id,
        cache_dir=cache_dir_for_model(model_id, cache_root=cache_root),
        snapshot_bytes=snapshot_size_bytes(
            model_id, cache_root=cache_root,
        ),
        was_already_cached=False,
    )


def assert_cached_or_raise(
    model_id: str,
    *,
    cache_root: Optional[Path] = None,
    prewarm_command_hint: str = (
        "python3 scripts/kakeya_prewarm.py --verifier-id {model_id}"
    ),
) -> None:
    """Pre-flight assertion: raise with a friendly message if missing.

    Used by ``scripts/start_grpc_runtime_server.py`` to fail fast on
    a cold cache rather than silently triggering a 5 GB download
    inside the server boot path. The error message points at the
    prewarm CLI; substitute ``{model_id}`` is filled in for clarity.
    """
    if is_model_in_cache(model_id, cache_root=cache_root):
        return
    cache_dir = cache_dir_for_model(model_id, cache_root=cache_root)
    hint = prewarm_command_hint.format(model_id=model_id)
    raise FileNotFoundError(
        f"HF cache miss for {model_id!r} (looked in {cache_dir}).\n"
        f"Pre-warm the cache before starting the server:\n"
        f"    {hint}\n"
        f"This avoids blocking server boot on a multi-GB download "
        f"with no progress feedback."
    )


def free_disk_bytes(path: Optional[Path] = None) -> int:
    """Best-effort free-disk-bytes for the cache filesystem.

    Useful for the prewarm CLI to give the user a "this won't fit"
    error before starting the download instead of after. Returns 0
    if the path doesn't exist or stat fails.
    """
    target = Path(path or HF_CACHE_DEFAULT)
    if not target.exists():
        target = target.parent
    try:
        return shutil.disk_usage(target).free
    except OSError:  # pragma: no cover - filesystem-dependent
        return 0
