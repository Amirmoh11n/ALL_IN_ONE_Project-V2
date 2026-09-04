"""Tumor class registry tests."""

import pytest

from src.data.classes import TumorClasses


def test_canonical_order_and_count():
    assert TumorClasses.NAMES == ["glioma", "meningioma", "notumor", "pituitary"]
    assert TumorClasses.num_classes() == 4


def test_round_trip_name_index():
    for i, name in enumerate(TumorClasses.NAMES):
        assert TumorClasses.name_to_index(name) == i
        assert TumorClasses.index_to_name(i) == name


def test_unknown_class_raises():
    with pytest.raises(ValueError, match="Unknown class"):
        TumorClasses.name_to_index("adenoma")
