"""Tests for ``training.repr_align``'s PEP 562 lazy-import policy.

The package's ``__init__.py`` exposes ``ReprAlignedSurgery`` and
``SurgeryConfig`` via ``__getattr__`` so that importing the
torch-free ``data_collection`` subpackage doesn't pay the heavy
torch + transformers import cost. These tests cover that
contract.
"""

from __future__ import annotations

import pytest


def test_lazy_attribute_resolves_repr_aligned_surgery():
    import training.repr_align as ra
    cls = ra.ReprAlignedSurgery
    assert cls.__name__ == "ReprAlignedSurgery"
    # Resolves to the real class from proposer_surgery
    from training.repr_align.proposer_surgery import (
        ReprAlignedSurgery as direct_cls,
    )
    assert cls is direct_cls


def test_lazy_attribute_resolves_surgery_config():
    import training.repr_align as ra
    cfg = ra.SurgeryConfig
    assert cfg.__name__ == "SurgeryConfig"


def test_lazy_attribute_unknown_name_raises_attribute_error():
    import training.repr_align as ra
    with pytest.raises(AttributeError, match="no attribute 'NotARealThing'"):
        _ = ra.NotARealThing


def test_dir_includes_lazy_public_symbols():
    import training.repr_align as ra
    listing = dir(ra)
    assert "ReprAlignedSurgery" in listing
    assert "SurgeryConfig" in listing
    # And the ordinary module attributes are still listed
    assert "__name__" in listing
