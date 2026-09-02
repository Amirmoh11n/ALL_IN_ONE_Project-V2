# src/data

Everything related to preparing data for the model.

- `classes.py` — the 4 tumor class labels (`TumorClasses`) and label/index mapping.
- `downloader.py` — `DatasetDownloader`: checks whether the dataset already exists under
  `data/raw/`; if not, downloads it via `kagglehub` and copies it into place.
- `splitter.py` — `DatasetSplitter`: stratified image-level split of the Training folder
  into train (~85%) and validation (~15%) subsets. Patient-level split is not used —
  this dataset merges 3 sources (figshare, SARTAJ, Br35H) with no patient IDs available.
- `augment.py` — `AugmentationFactory`: builds the training transform pipeline
  (flip, rotation, color jitter, ImageNet normalization) and the val/test pipeline
  (resize + normalization only, no augmentation).
- `dataset.py` — `BrainTumorDataset`: PyTorch `Dataset` wrapping a list of
  `(image_path, class_index)` samples, or built directly from a class-subfoldered
  directory (used for the Testing folder).
- `pipeline.py` — `DataPipeline`: facade that wires the above into ready-to-use
  train/val/test `DataLoader`s, driven entirely by `configs/config.yaml`.

All settings (paths, split ratio, image size, augmentation params, batch size, etc.)
live in `configs/config.yaml` — nothing here is hardcoded.
