"""
Configures the root logger's level from configs/config.yaml (logging.level),
so verbosity can be changed without touching source code.
"""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger to the given level.

    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR" (case-insensitive).
            Falls back to INFO if the value is not recognized.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
