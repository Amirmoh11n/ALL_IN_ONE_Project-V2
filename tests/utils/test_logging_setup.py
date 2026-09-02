"""Tests for src/utils/logging_setup.py (configure_logging).

Note: logging.basicConfig is a no-op once the root logger already has handlers.
Tests therefore reset the root logger state before each call so the level
actually gets applied.
"""

import logging

from src.utils.logging_setup import configure_logging


def _reset_root_logger():
    """Clear handlers so basicConfig can take effect again."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)  # neutral starting point


def test_configure_logging_sets_info_level():
    _reset_root_logger()
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_sets_debug_level():
    _reset_root_logger()
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_sets_warning_level():
    _reset_root_logger()
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_is_case_insensitive():
    _reset_root_logger()
    configure_logging("info")
    assert logging.getLogger().level == logging.INFO

    _reset_root_logger()
    configure_logging("DeBuG")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_falls_back_to_info_for_unknown_level():
    _reset_root_logger()
    configure_logging("NOT_A_REAL_LEVEL")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_default_is_info():
    _reset_root_logger()
    configure_logging()
    assert logging.getLogger().level == logging.INFO
