# tests

Offline unit and API tests. They do **not** download the MRI dataset or ImageNet
weights unless a test explicitly builds a pretrained model.

| File | Covers |
| --- | --- |
| `test_classes.py` | 4-class registry |
| `test_config_loader.py` | YAML dotted keys, snapshots |
| `test_quality.py` | corrupt / tiny / duplicate drops |
| `test_splitter.py` | patient-id parse, stratified, auto fallback, patient-aware |
| `test_metrics.py` | accuracy, F1, recall, PPV/NPV, ECE, ROC-AUC |
| `test_calibration.py` | temperature scaling + ECE comparison |
| `test_evaluation_result.py` | JSON / markdown metric payload |
| `test_early_stopping.py` | monitor mode + class weights |
| `test_factory.py` | EfficientNet-B4 / B3 random init |
| `test_cli.py` | argparse surface |
| `test_schemas.py` | FastAPI response contracts |
| `test_prediction_result.py` | inference payload |
| `test_model_service.py` | softmax + preprocess |
| `test_api.py` | `/api/health`, `/api/model-info`, `/api/predict` |

```bash
PYTHONPATH=. pytest -q
```

See `docs/CI_CD.md` for the GitHub Actions pipeline.
