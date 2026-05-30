"""Unit tests for ``training.repr_align.data_collection.prompt_pool``."""

from __future__ import annotations

import pytest

from training.repr_align.data_collection.prompt_pool import (
    CharRatioLanguageDetector,
    DomainQuota,
    LengthFilter,
    PoolConfig,
    Prompt,
    PromptPool,
    ShingleJaccardDeduper,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_prompt_rejects_empty_id():
    with pytest.raises(ValueError, match="prompt_id"):
        Prompt(prompt_id="", text="t", domain="d")


def test_prompt_rejects_empty_text():
    with pytest.raises(ValueError, match="text"):
        Prompt(prompt_id="x", text="", domain="d")


def test_prompt_rejects_empty_domain():
    with pytest.raises(ValueError, match="domain"):
        Prompt(prompt_id="x", text="t", domain="")


# ---------------------------------------------------------------------------
# DomainQuota
# ---------------------------------------------------------------------------


def test_quota_rejects_zero_share():
    with pytest.raises(ValueError, match="share"):
        DomainQuota(name="x", share=0.0)


def test_quota_rejects_share_over_one():
    with pytest.raises(ValueError, match="share"):
        DomainQuota(name="x", share=1.5)


def test_quota_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        DomainQuota(name="", share=0.5)


# ---------------------------------------------------------------------------
# CharRatioLanguageDetector
# ---------------------------------------------------------------------------


def test_lang_detect_empty_returns_und():
    det = CharRatioLanguageDetector()
    assert det("") == "und"


def test_lang_detect_english():
    det = CharRatioLanguageDetector()
    assert det("hello world this is plain ascii") == "en"


def test_lang_detect_chinese():
    det = CharRatioLanguageDetector()
    assert det("你好世界") == "zh"


def test_lang_detect_threshold_invalid():
    with pytest.raises(ValueError, match="cjk_threshold"):
        CharRatioLanguageDetector(cjk_threshold=0.0)
    with pytest.raises(ValueError, match="cjk_threshold"):
        CharRatioLanguageDetector(cjk_threshold=1.5)


# ---------------------------------------------------------------------------
# ShingleJaccardDeduper
# ---------------------------------------------------------------------------


def test_dedup_observe_returns_true_for_first_occurrence():
    d = ShingleJaccardDeduper(k=3, threshold=0.85)
    sig = d.signature_for("the quick brown fox jumps over")
    assert d.observe(sig) is True


def test_dedup_rejects_near_duplicate():
    d = ShingleJaccardDeduper(k=3, threshold=0.5)
    sig_a = d.signature_for("the quick brown fox jumps over the lazy dog")
    assert d.observe(sig_a) is True
    # Slight variation, large overlap of shingles → blocked at low threshold
    sig_b = d.signature_for("the quick brown fox jumps over the lazy dog tail")
    assert d.observe(sig_b) is False


def test_dedup_admits_dissimilar_text():
    d = ShingleJaccardDeduper(k=3, threshold=0.85)
    a = d.signature_for("alpha bravo charlie delta echo")
    assert d.observe(a) is True
    b = d.signature_for("xray yankee zulu omega tango papa")
    assert d.observe(b) is True


def test_dedup_signature_for_short_text_below_k():
    d = ShingleJaccardDeduper(k=10)
    sig = d.signature_for("a b c")
    # Below-k branch: produces a single-element signature (the joined text)
    assert len(sig) == 1


def test_dedup_signature_for_empty_text():
    d = ShingleJaccardDeduper(k=3)
    sig = d.signature_for("")
    assert sig == ()
    # Empty signature passes through observe
    assert d.observe(sig) is True


def test_dedup_observe_skips_empty_prev():
    """Empty prior signature must not match anything (cover the
    `if not prev: continue` branch in observe)."""
    d = ShingleJaccardDeduper(k=3, threshold=0.5)
    d.observe(())  # seed with empty signature
    sig = d.signature_for("alpha bravo charlie delta echo foxtrot")
    assert d.observe(sig) is True


def test_dedup_rejects_bad_k():
    with pytest.raises(ValueError, match="k must"):
        ShingleJaccardDeduper(k=0)


def test_dedup_rejects_bad_threshold():
    with pytest.raises(ValueError, match="threshold"):
        ShingleJaccardDeduper(threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        ShingleJaccardDeduper(threshold=1.5)


# ---------------------------------------------------------------------------
# LengthFilter
# ---------------------------------------------------------------------------


def test_length_filter_admits_in_range():
    f = LengthFilter(min_tokens=2, max_tokens=10)
    assert f.admits("hello world here")
    assert f.count_tokens("hello world") == 2


def test_length_filter_rejects_too_short():
    f = LengthFilter(min_tokens=5, max_tokens=10)
    assert not f.admits("only two")


def test_length_filter_rejects_too_long():
    f = LengthFilter(min_tokens=1, max_tokens=3)
    assert not f.admits("a b c d e")


def test_length_filter_rejects_invalid_min_tokens():
    with pytest.raises(ValueError, match="min_tokens"):
        LengthFilter(min_tokens=-1, max_tokens=10)


def test_length_filter_rejects_invalid_range():
    with pytest.raises(ValueError, match="max_tokens"):
        LengthFilter(min_tokens=10, max_tokens=10)


# ---------------------------------------------------------------------------
# PoolConfig
# ---------------------------------------------------------------------------


def _make_filter() -> LengthFilter:
    return LengthFilter(min_tokens=1, max_tokens=100)


def test_pool_config_rejects_empty_quotas():
    with pytest.raises(ValueError, match="quotas"):
        PoolConfig(quotas=(), target_size=10, length_filter=_make_filter())


def test_pool_config_rejects_zero_target():
    with pytest.raises(ValueError, match="target_size"):
        PoolConfig(
            quotas=(DomainQuota("x", 1.0),),
            target_size=0,
            length_filter=_make_filter(),
        )


def test_pool_config_rejects_shares_not_summing_to_one():
    with pytest.raises(ValueError, match="quota shares"):
        PoolConfig(
            quotas=(DomainQuota("a", 0.6), DomainQuota("b", 0.6)),
            target_size=10,
            length_filter=_make_filter(),
        )


def test_pool_config_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        PoolConfig(
            quotas=(DomainQuota("a", 0.5), DomainQuota("a", 0.5)),
            target_size=10,
            length_filter=_make_filter(),
        )


# ---------------------------------------------------------------------------
# PromptPool — registration & build
# ---------------------------------------------------------------------------


def _make_prompts(domain: str, n: int, seed_text: str = "alpha bravo charlie") -> list[Prompt]:
    """Build n distinct prompts for one domain with mostly disjoint
    vocabulary so the deduper does not collapse them."""
    out: list[Prompt] = []
    for i in range(n):
        words = [f"{seed_text}_{i}", f"word{i*3}", f"item{i*7}", f"tag{i*11}", f"id{i*13}"]
        out.append(Prompt(
            prompt_id=f"{domain}_{i}",
            text=" ".join(words),
            domain=domain,
        ))
    return out


def test_pool_register_unknown_domain_raises():
    config = PoolConfig(
        quotas=(DomainQuota("a", 1.0),),
        target_size=2,
        length_filter=_make_filter(),
    )
    pool = PromptPool(config=config)
    with pytest.raises(KeyError):
        pool.register("not_a_domain", [])


def test_pool_register_with_domain_mismatch_raises():
    config = PoolConfig(
        quotas=(DomainQuota("a", 1.0),),
        target_size=2,
        length_filter=_make_filter(),
    )
    pool = PromptPool(config=config)
    bad = Prompt(prompt_id="x", text="text", domain="b")
    with pytest.raises(ValueError, match="prompt 'x'"):
        pool.register("a", [bad])


def test_pool_register_returns_admitted_count_after_length_filter():
    f = LengthFilter(min_tokens=2, max_tokens=10)
    config = PoolConfig(
        quotas=(DomainQuota("a", 1.0),),
        target_size=2,
        length_filter=f,
    )
    pool = PromptPool(config=config)
    prompts = [
        Prompt(prompt_id="ok1", text="alpha bravo", domain="a"),
        Prompt(prompt_id="bad_short", text="x", domain="a"),
        Prompt(prompt_id="ok2", text="charlie delta echo", domain="a"),
    ]
    n = pool.register("a", prompts)
    assert n == 2


def test_pool_build_respects_quota_and_dedups():
    config = PoolConfig(
        quotas=(DomainQuota("a", 0.5), DomainQuota("b", 0.5)),
        target_size=4,
        length_filter=LengthFilter(min_tokens=1, max_tokens=100),
        seed=123,
    )
    pool = PromptPool(config=config)
    pool.register("a", _make_prompts("a", 5, "alpha"))
    pool.register("b", _make_prompts("b", 5, "yankee"))
    prompts = pool.build()
    assert len(prompts) == 4
    # Quota: 2 from a, 2 from b
    counts = {"a": 0, "b": 0}
    for p in prompts:
        counts[p.domain] += 1
    assert counts == {"a": 2, "b": 2}


def test_pool_build_seeded_rng_is_reproducible():
    def _build():
        config = PoolConfig(
            quotas=(DomainQuota("a", 1.0),),
            target_size=3,
            length_filter=_make_filter(),
            seed=42,
        )
        pool = PromptPool(config=config)
        pool.register("a", _make_prompts("a", 10))
        return [p.prompt_id for p in pool.build()]

    assert _build() == _build()


def test_pool_build_raises_when_domain_unregistered():
    config = PoolConfig(
        quotas=(DomainQuota("a", 0.5), DomainQuota("b", 0.5)),
        target_size=4,
        length_filter=_make_filter(),
    )
    pool = PromptPool(config=config)
    pool.register("a", _make_prompts("a", 3))
    # b is never registered
    with pytest.raises(RuntimeError, match="unregistered"):
        pool.build()


def test_pool_build_raises_when_domain_underflow_after_dedup():
    config = PoolConfig(
        quotas=(DomainQuota("a", 1.0),),
        target_size=10,
        length_filter=_make_filter(),
    )
    pool = PromptPool(config=config)
    # Only 2 raw prompts, way less than the target → must error
    pool.register("a", _make_prompts("a", 2))
    with pytest.raises(RuntimeError, match="produced only"):
        pool.build()


def test_pool_build_uses_existing_dedup_signature_if_supplied():
    """If a Prompt already carries a dedup_signature the deduper
    should use it instead of recomputing — deterministic and lets
    upstream pipelines plug in MinHash without re-shingling here."""
    config = PoolConfig(
        quotas=(DomainQuota("a", 1.0),),
        target_size=1,
        length_filter=_make_filter(),
    )
    pool = PromptPool(config=config)
    p = Prompt(
        prompt_id="x",
        text="alpha bravo charlie delta echo",
        domain="a",
        dedup_signature=(1234567890,),
    )
    pool.register("a", [p])
    out = pool.build()
    assert out[0].dedup_signature == (1234567890,)


def test_pool_build_skips_duplicates_via_deduper():
    """Cover the ``if not deduper.observe(): continue`` branch in
    :meth:`PromptPool.build` using a deterministic deduper test
    double that rejects every other call. This is a real concrete
    class implementing the :class:`Deduper` protocol — not a mock.
    """

    class AlternatingRejectDeduper:
        """Accepts call 0, 2, 4, ...; rejects call 1, 3, 5, ..."""

        def __init__(self) -> None:
            self.calls = 0

        def signature_for(self, text: str) -> tuple[int, ...]:
            return (hash(text) & 0xFFFFFFFF,)

        def observe(self, signature: tuple[int, ...]) -> bool:
            self.calls += 1
            return self.calls % 2 == 1

    config = PoolConfig(
        quotas=(DomainQuota("a", 1.0),),
        target_size=2,
        length_filter=_make_filter(),
        seed=0,
    )
    pool = PromptPool(config=config, deduper=AlternatingRejectDeduper())
    pool.register("a", _make_prompts("a", 4))
    out = pool.build()
    # 4 raw prompts, deduper accepts 2 and rejects 2 → quota of 2 satisfied
    assert len(out) == 2


def test_pool_config_property_exposed():
    config = PoolConfig(
        quotas=(DomainQuota("a", 1.0),),
        target_size=1,
        length_filter=_make_filter(),
    )
    pool = PromptPool(config=config)
    assert pool.config is config
