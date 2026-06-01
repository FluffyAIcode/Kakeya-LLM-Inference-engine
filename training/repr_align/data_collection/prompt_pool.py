"""Prompt-pool composition + filtering (ADR 0004 §2.1).

Multi-domain prompt sampling with quality filters and dedup. This
module is the *contract layer* — production data collection hands
this object a stream of raw domain prompts and gets back a
sampled, filtered, deduped, quota-respecting list ready for rollout.

The downstream rollout worker (PR 2/3 of this work line) does not
care where the prompts came from; it only needs the
:class:`Prompt` records this module emits.

Filters are intentionally pluggable so production can swap in
``fasttext`` / ``langdetect`` / ``datasketch.MinHashLSH`` without
touching the pool logic. Each filter is a ``Protocol`` with one
default reference implementation that has no dependencies beyond
the stdlib + numpy, so unit tests run against real concrete classes
(no mocks per project rule).
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence


_TOKEN_RE = re.compile(r"\S+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Prompt:
    """One candidate prompt awaiting verifier rollout."""

    prompt_id: str
    text: str
    domain: str
    language: str = ""
    n_tokens: int = 0
    dedup_signature: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must be non-empty")
        if not self.text:
            raise ValueError("text must be non-empty")
        if not self.domain:
            raise ValueError("domain must be non-empty")


@dataclass(frozen=True)
class DomainQuota:
    """One domain's share of the final pool (ADR 0004 §2.1 table)."""

    name: str
    share: float
    source_tag: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not (0.0 < self.share <= 1.0):
            raise ValueError(f"share must be in (0, 1], got {self.share}")


# ---------------------------------------------------------------------------
# Filter protocols + reference implementations
# ---------------------------------------------------------------------------


class LanguageDetector(Protocol):  # pragma: no cover - protocol body
    """Returns an ISO 639-1 language tag for a text, or ``"und"``."""

    def __call__(self, text: str) -> str: ...


class CharRatioLanguageDetector:
    """Deterministic language tagger using CJK char ratio.

    Reference implementation that does not pull in fasttext. Tags
    text with ratio >= ``cjk_threshold`` of CJK characters as
    ``zh``; otherwise ``en``. Production swaps in a real detector
    via the :class:`LanguageDetector` protocol.

    Deterministic by design — same input always returns the same
    tag — so unit tests don't need fixtures.
    """

    def __init__(self, *, cjk_threshold: float = 0.3) -> None:
        if not (0.0 < cjk_threshold <= 1.0):
            raise ValueError(
                f"cjk_threshold must be in (0, 1], got {cjk_threshold}"
            )
        self._cjk_threshold = cjk_threshold

    def __call__(self, text: str) -> str:
        if not text:
            return "und"
        cjk = sum(1 for ch in text if _CJK_RE.match(ch))
        ratio = cjk / max(len(text), 1)
        if ratio >= self._cjk_threshold:
            return "zh"
        return "en"


class Deduper(Protocol):  # pragma: no cover - protocol body
    """Stateful deduper. ``observe`` returns False if a similar text
    has already been seen, True otherwise."""

    def observe(self, signature: tuple[int, ...]) -> bool: ...

    def signature_for(self, text: str) -> tuple[int, ...]: ...


class ShingleJaccardDeduper:
    """Reference deduper using k-shingles + Jaccard similarity.

    Each text is shingled into overlapping k-grams of words; the
    shingle set is hashed into a sorted tuple of int signatures. Two
    texts are duplicates if their signature-set Jaccard exceeds
    ``threshold``.

    O(N²) in the worst case, which is acceptable for the unit-test
    pool sizes (≤ 100 prompts) and for development-scale rollouts
    (≤ 10 k prompts). Production at 50 k+ prompts plugs in
    ``datasketch.MinHashLSH`` via the :class:`Deduper` protocol.

    Deterministic: signatures are SHA-256-derived integers, so two
    identical inputs always produce identical signatures.
    """

    def __init__(self, *, k: int = 5, threshold: float = 0.85) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self._k = k
        self._threshold = threshold
        self._seen: list[frozenset[int]] = []

    def signature_for(self, text: str) -> tuple[int, ...]:
        words = _TOKEN_RE.findall(text.lower())
        if len(words) < self._k:
            shingles = [" ".join(words)] if words else []
        else:
            shingles = [
                " ".join(words[i : i + self._k])
                for i in range(len(words) - self._k + 1)
            ]
        sigs = sorted({
            int.from_bytes(
                hashlib.sha256(s.encode("utf-8")).digest()[:8],
                "big",
            )
            for s in shingles
        })
        return tuple(sigs)

    def observe(self, signature: tuple[int, ...]) -> bool:
        sig_set = frozenset(signature)
        if not sig_set:
            # Empty signature (empty / sub-k text) is never a dup of
            # itself; pass through and let the length filter handle.
            self._seen.append(sig_set)
            return True
        for prev in self._seen:
            if not prev:
                continue
            inter = len(sig_set & prev)
            union = len(sig_set | prev)
            if union > 0 and (inter / union) >= self._threshold:
                return False
        self._seen.append(sig_set)
        return True


# ---------------------------------------------------------------------------
# Length filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LengthFilter:
    """Simple word-tokenized length gate (ADR 0004 §2.1 quality filter).

    Counts whitespace-separated tokens. Production replaces with a
    real tokenizer count via the per-verifier tokenizer at the
    rollout worker stage; this stage's filter is intentionally cheap
    and language-agnostic.
    """

    min_tokens: int
    max_tokens: int

    def __post_init__(self) -> None:
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be >= 0, got {self.min_tokens}")
        if self.max_tokens <= self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be > min_tokens "
                f"({self.min_tokens})"
            )

    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))

    def admits(self, text: str) -> bool:
        n = self.count_tokens(text)
        return self.min_tokens <= n <= self.max_tokens


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolConfig:
    """Top-level pool configuration."""

    quotas: tuple[DomainQuota, ...]
    target_size: int
    length_filter: LengthFilter
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.quotas:
            raise ValueError("quotas must be non-empty")
        if self.target_size <= 0:
            raise ValueError(f"target_size must be > 0, got {self.target_size}")
        share_sum = sum(q.share for q in self.quotas)
        if abs(share_sum - 1.0) > 1e-6:
            raise ValueError(
                f"quota shares must sum to 1.0 (±1e-6), got {share_sum}"
            )
        names = [q.name for q in self.quotas]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate quota name in {names}")


class PromptPool:
    """Multi-domain prompt pool builder (ADR 0004 §2.1).

    Usage::

        config = PoolConfig(
            quotas=(DomainQuota("chat_en", 0.5), DomainQuota("code", 0.5)),
            target_size=100,
            length_filter=LengthFilter(5, 4096),
        )
        pool = PromptPool(config=config)
        pool.register("chat_en", chat_en_iter)
        pool.register("code", code_iter)
        prompts = pool.build()  # length 100, deduped, filtered, quota-respected
    """

    def __init__(
        self,
        *,
        config: PoolConfig,
        language_detector: LanguageDetector | None = None,
        deduper: Deduper | None = None,
    ) -> None:
        self._config = config
        self._lang = language_detector or CharRatioLanguageDetector()
        self._deduper = deduper or ShingleJaccardDeduper()
        self._streams: dict[str, list[Prompt]] = {q.name: [] for q in config.quotas}
        self._registered: set[str] = set()

    @property
    def config(self) -> PoolConfig:
        return self._config

    def register(self, domain: str, prompts: Iterable[Prompt]) -> int:
        """Add raw prompts for one domain.

        Returns the number of prompts admitted by the length filter.
        Dedup runs at :meth:`build` time so cross-domain duplicates
        are handled.
        """
        if domain not in self._streams:
            raise KeyError(
                f"domain {domain!r} is not in the configured quotas; "
                f"known: {sorted(self._streams)}"
            )
        admitted = 0
        for p in prompts:
            if p.domain != domain:
                raise ValueError(
                    f"prompt {p.prompt_id!r} has domain {p.domain!r} but was "
                    f"registered under {domain!r}"
                )
            if not self._config.length_filter.admits(p.text):
                continue
            language = p.language or self._lang(p.text)
            n_tokens = p.n_tokens or self._config.length_filter.count_tokens(p.text)
            self._streams[domain].append(
                Prompt(
                    prompt_id=p.prompt_id,
                    text=p.text,
                    domain=p.domain,
                    language=language,
                    n_tokens=n_tokens,
                    dedup_signature=p.dedup_signature,
                )
            )
            admitted += 1
        self._registered.add(domain)
        return admitted

    def build(self) -> Sequence[Prompt]:
        """Sample, dedup and assemble the final prompt list.

        Sampling order: shuffle each domain's admitted list with a
        seeded RNG; greedily take prompts in domain order, calling
        the deduper; stop a domain once it hits its quota count.
        """
        missing = {q.name for q in self._config.quotas} - self._registered
        if missing:
            raise RuntimeError(
                f"build() called with unregistered domains: {sorted(missing)}"
            )

        rng = random.Random(self._config.seed)
        out: list[Prompt] = []
        for q in self._config.quotas:
            quota_count = max(1, int(round(q.share * self._config.target_size)))
            stream = list(self._streams[q.name])
            rng.shuffle(stream)
            taken = 0
            for p in stream:
                if taken >= quota_count:
                    break
                signature = p.dedup_signature or self._deduper.signature_for(p.text)
                if not self._deduper.observe(signature):
                    continue
                out.append(
                    Prompt(
                        prompt_id=p.prompt_id,
                        text=p.text,
                        domain=p.domain,
                        language=p.language,
                        n_tokens=p.n_tokens,
                        dedup_signature=signature,
                    )
                )
                taken += 1
            if taken < quota_count:
                raise RuntimeError(
                    f"domain {q.name!r} produced only {taken} prompts after "
                    f"dedup but quota requires {quota_count}; register more "
                    f"upstream prompts or relax the dedup threshold"
                )
        return tuple(out)


__all__ = [
    "Prompt",
    "DomainQuota",
    "PoolConfig",
    "PromptPool",
    "LengthFilter",
    "LanguageDetector",
    "CharRatioLanguageDetector",
    "Deduper",
    "ShingleJaccardDeduper",
]
