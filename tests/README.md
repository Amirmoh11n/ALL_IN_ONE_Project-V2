# tests

Unit/integration tests, organized into one subfolder per `src/` package (mirrors
the source tree), using synthetic (tiny, generated) images so tests run fast and
don't need the real dataset or network access.

```
tests/
├── conftest.py            # shared fixtures (synthetic Training/Testing folders),
│                           # applies to every subfolder below automatically
├── data/                   -> mirrors src/data/
│   ├── test_classes.py
│   ├── test_downloader.py
│   ├── test_splitter.py
│   ├── test_dataset.py
│   └── test_pipeline.py    # end-to-end DataPipeline test
├── models/                 -> mirrors src/models/
│   └── test_efficientnet.py
├── engine/                 -> mirrors src/engine/
│   └── test_trainer.py
├── metrics/                -> mirrors src/metrics/
│   └── test_metrics.py     # all 6 metrics, hand-computed expected values
├── evaluate/                -> mirrors src/evaluate/
│   └── test_evaluate.py
├── export/                 -> mirrors src/export/
│   └── test_export.py
├── inference/               -> mirrors src/inference/
│   └── test_inference.py
└── utils/                  -> mirrors src/utils/
    ├── test_config_loader.py
    └── test_checkpoint.py
```

Run everything:
```bash
pytest tests/ -v
```

Run just one package's tests:
```bash
pytest tests/data/ -v
pytest tests/engine/ -v
```
