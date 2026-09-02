"""
Defines the 4 tumor class labels for the Brain Tumor MRI Dataset and provides
utilities to map between class names and integer indices.

Classes (fixed by the Nickparvar Brain Tumor MRI Dataset folder names):
    - glioma
    - meningioma
    - notumor
    - pituitary
"""

from typing import List


class TumorClasses:
    """Ordered list of tumor classes and name <-> index lookup helpers.

    The order below is the canonical order used everywhere in the project
    (dataset labels, confusion matrix axes, model output indices, etc.).
    Do not reorder without updating any already-trained checkpoints.
    """

    NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]

    @classmethod
    def name_to_index(cls, name: str) -> int:
        """Return the integer index for a given class name.

        Args:
            name: One of TumorClasses.NAMES.

        Raises:
            ValueError: If name is not a recognized class.
        """
        try:
            return cls.NAMES.index(name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown class name '{name}'. Expected one of {cls.NAMES}."
            ) from exc

    @classmethod
    def index_to_name(cls, index: int) -> str:
        """Return the class name for a given integer index."""
        return cls.NAMES[index]

    @classmethod
    def num_classes(cls) -> int:
        """Return the total number of classes."""
        return len(cls.NAMES)
