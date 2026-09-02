"""Tests for src/data/classes.py (TumorClasses)."""

import pytest

from src.data.classes import TumorClasses


def test_num_classes_is_four():
    assert TumorClasses.num_classes() == 4


def test_name_to_index_and_back_roundtrip():
    for expected_index, name in enumerate(TumorClasses.NAMES):
        assert TumorClasses.name_to_index(name) == expected_index
        assert TumorClasses.index_to_name(expected_index) == name


def test_unknown_class_name_raises():
    with pytest.raises(ValueError):
        TumorClasses.name_to_index("not_a_real_class")
