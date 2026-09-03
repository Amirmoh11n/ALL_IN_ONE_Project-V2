"""Data-quality checks for MRI images: corrupt files, duplicates, size outliers."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

Sample = Tuple[Path, int]


@dataclass
class QualityReport:
    """Summary of dropped / kept samples after quality filtering."""

    kept: List[Sample] = field(default_factory=list)
    dropped_corrupt: int = 0
    dropped_duplicate: int = 0
    dropped_size: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "kept": len(self.kept),
            "dropped_corrupt": self.dropped_corrupt,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_size": self.dropped_size,
        }


class ImageQualityFilter:
    """Filter unreadable, duplicate, and tiny MRI files before splitting."""

    def __init__(
        self,
        drop_corrupt: bool = True,
        drop_duplicates: bool = True,
        min_width: int = 32,
        min_height: int = 32,
    ) -> None:
        self.drop_corrupt = drop_corrupt
        self.drop_duplicates = drop_duplicates
        self.min_width = min_width
        self.min_height = min_height

    def filter(self, samples: List[Sample]) -> QualityReport:
        report = QualityReport()
        seen_hashes: Dict[str, Path] = {}
        for path, label in samples:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    digest = hashlib.md5(image.tobytes()).hexdigest() if self.drop_duplicates else ""
            except (UnidentifiedImageError, OSError, ValueError):
                if self.drop_corrupt:
                    report.dropped_corrupt += 1
                    logger.warning("Dropping unreadable image: %s", path)
                    continue
                report.kept.append((path, label))
                continue

            if width < self.min_width or height < self.min_height:
                report.dropped_size += 1
                logger.warning("Dropping undersized image %s (%dx%d)", path, width, height)
                continue

            if self.drop_duplicates and digest in seen_hashes:
                report.dropped_duplicate += 1
                logger.info("Dropping duplicate of %s: %s", seen_hashes[digest], path)
                continue

            if digest:
                seen_hashes[digest] = path
            report.kept.append((path, label))

        logger.info("Quality filter: %s", report.to_dict())
        return report
