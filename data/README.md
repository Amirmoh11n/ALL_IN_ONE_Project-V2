# data

Local storage for the raw Brain Tumor MRI Dataset (Nickparvar). Not part of source
control (see `.gitignore`) — `src/data/downloader.py` will auto-download it here via
`kagglehub` the first time the pipeline runs, if it isn't already present.

- `raw/Training/` — original Training folder (split into train/validation by
  `src/data/splitter.py`; ~85% train, ~15% validation for hyperparameter tuning).
- `raw/Testing/` — original Testing folder, kept untouched for final evaluation only.

## Kaggle authentication (required for auto-download)

`kagglehub` needs Kaggle API credentials to download the dataset. Either:
- place a `kaggle.json` API token at `~/.kaggle/kaggle.json`, or
- set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

If the dataset is already present under `raw/Training` and `raw/Testing` (e.g. placed
there manually), no download or credentials are needed — the pipeline detects and
reuses it.
