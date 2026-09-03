"""Train / validation splitting with optional patient-aware grouping.

The Nickparvar Brain Tumor MRI Dataset usually has no reliable patient IDs.
``split_strategy: auto`` tries to extract IDs from filenames; if coverage is
weak it falls back to a stratified image-level split (documented on the model card).
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.classes import TumorClasses

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_PATIENT_PATTERNS = (
    re.compile(r"(patient[_-]?\d+)", re.I),
    re.compile(r"(subj(?:ect)?[_-]?\d+)", re.I),
    re.compile(r"(pid[_-]?\d+)", re.I),
    re.compile(r"(P\d{3,})", re.I),
)

Sample = Tuple[Path, int]


@dataclass
class SplitResult:
    """Container for the outcome of a train/validation split."""

    train_samples: List[Sample] = field(default_factory=list)
    val_samples: List[Sample] = field(default_factory=list)
    mode: str = "stratified_image_level"


class PatientIdExtractor:
    """Best-effort patient identifier parser for MRI filenames."""

    def extract(self, path: Path) -> Optional[str]:
        stem = path.stem
        for pattern in _PATIENT_PATTERNS:
            match = pattern.search(stem)
            if match:
                return match.group(1).lower()
        return None


class DatasetSplitter:
    """Stratified image-level split, with optional patient-aware grouping."""

    def __init__(
        self,
        train_dir: Path,
        val_ratio: float,
        random_seed: int = 42,
        strategy: str = "auto",
        samples: Optional[List[Sample]] = None,
    ) -> None:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
        self.train_dir = Path(train_dir)
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.strategy = strategy
        self.samples = samples
        self.extractor = PatientIdExtractor()

    def split(self) -> SplitResult:
        rng = random.Random(self.random_seed)
        all_samples = self.samples if self.samples is not None else self._collect_samples()
        if self.strategy == "stratified_image_level":
            result = self._split_image_level(all_samples, rng)
            result.mode = "stratified_image_level"
            return result
        if self.strategy in {"auto", "patient_aware"}:
            ids = [self.extractor.extract(path) for path, _ in all_samples]
            n_with_id = sum(1 for pid in ids if pid)
            unique_real = len({pid for pid in ids if pid})
            grouped_enough = (
                n_with_id >= 0.5 * max(1, len(all_samples))
                and unique_real < 0.85 * n_with_id
                and unique_real >= 8
            )
            if grouped_enough:
                grouped = self._group_by_patient(all_samples)
                result = self._split_patient_aware(grouped, rng)
                result.mode = "patient_aware"
                logger.info("Using patient-aware split (%d patient IDs).", unique_real)
                return result
            if self.strategy == "patient_aware":
                logger.warning(
                    "Patient-aware requested but IDs are unreliable "
                    "(with_id=%d unique=%d / %d images); falling back to image-level.",
                    n_with_id,
                    unique_real,
                    len(all_samples),
                )
            result = self._split_image_level(all_samples, rng)
            result.mode = "stratified_image_level"
            logger.info("Split mode: stratified_image_level (no reliable patient IDs).")
            return result
        raise ValueError(f"Unknown split_strategy: {self.strategy}")

    def _collect_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        for class_name in TumorClasses.NAMES:
            class_dir = self.train_dir / class_name
            class_index = TumorClasses.name_to_index(class_name)
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in _IMAGE_SUFFIXES:
                    samples.append((path, class_index))
            if not any(s[1] == class_index for s in samples):
                logger.warning("No images found for class '%s' in %s", class_name, class_dir)
        return samples

    def _group_by_patient(self, samples: List[Sample]) -> Dict[str, List[Sample]]:
        grouped: Dict[str, List[Sample]] = defaultdict(list)
        for path, label in samples:
            pid = self.extractor.extract(path) or f"image::{path.resolve()}"
            grouped[pid].append((path, label))
        return grouped

    def _split_image_level(self, samples: List[Sample], rng: random.Random) -> SplitResult:
        by_class: Dict[int, List[Path]] = defaultdict(list)
        for path, label in samples:
            by_class[label].append(path)
        result = SplitResult()
        for class_index, paths in by_class.items():
            shuffled = paths.copy()
            rng.shuffle(shuffled)
            n_val = round(len(shuffled) * self.val_ratio)
            result.val_samples.extend((p, class_index) for p in shuffled[:n_val])
            result.train_samples.extend((p, class_index) for p in shuffled[n_val:])
        rng.shuffle(result.train_samples)
        rng.shuffle(result.val_samples)
        logger.info(
            "Split complete: %d train, %d val (val_ratio=%.2f, mode=image-level).",
            len(result.train_samples),
            len(result.val_samples),
            self.val_ratio,
        )
        return result

    def _split_patient_aware(
        self, grouped: Dict[str, List[Sample]], rng: random.Random
    ) -> SplitResult:
        # Assign each patient a majority class for stratification, then hold out patients.
        patients = list(grouped.items())
        rng.shuffle(patients)
        by_class: Dict[int, List[str]] = defaultdict(list)
        for pid, items in patients:
            labels = [label for _, label in items]
            majority = max(set(labels), key=labels.count)
            by_class[majority].append(pid)
        val_ids = set()
        for pids in by_class.values():
            n_val = max(1, round(len(pids) * self.val_ratio)) if pids else 0
            val_ids.update(pids[:n_val])
        result = SplitResult()
        for pid, items in grouped.items():
            if pid in val_ids:
                result.val_samples.extend(items)
            else:
                result.train_samples.extend(items)
        rng.shuffle(result.train_samples)
        rng.shuffle(result.val_samples)
        logger.info(
            "Split complete: %d train, %d val (patients val=%d, mode=patient-aware).",
            len(result.train_samples),
            len(result.val_samples),
            len(val_ids),
        )
        return result
