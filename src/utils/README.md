# src/utils

Shared helper utilities used across the project.

- `config_loader.py` — `ConfigLoader`: loads `configs/config.yaml` and exposes a
  dotted-path getter, e.g. `config.get("data.val_split", 0.15)`.
- `logging_setup.py` — `configure_logging(level)`: sets the root logger level from
  `configs/config.yaml` (`logging.level`), so verbosity is configurable without
  touching source code.
- `checkpoint.py` — `load_model_checkpoint(model, path, device)`: loads a saved
  checkpoint's weights into a model instance. Shared by evaluation and (later)
  inference, so the checkpoint format is defined in exactly one place.
