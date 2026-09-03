# Web Application — Brain Tumor MRI Classifier

Production-oriented web interface for the trained/exported EfficientNet-B4 classifier.

## Stack

- Frontend: semantic HTML, Tailwind CSS, vanilla JavaScript
- Backend: FastAPI + Uvicorn
- Inference: ONNX Runtime
- Model artifact: `artifacts/exports/brain_tumor_efficientnet_b4.onnx`

## Flow

```text
MRI image
   ↓
Browser upload / drag & drop
   ↓
FastAPI /api/predict
   ↓
Resize + ImageNet normalization
   ↓
ONNX Runtime
   ↓
Softmax probabilities
   ↓
Prediction + confidence + class distribution
   ↓
Interactive UI
```

## Run

From the repository root:

```bash
uv sync
uv run uvicorn webapplication.backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

If the ONNX export is not present, train and export first:

```bash
uv run brain-tumor train
uv run brain-tumor export
```

Then restart FastAPI so the model service loads the new artifact.

## Environment variables

- `MODEL_PATH`: custom ONNX path
- `HOST`: server host
- `PORT`: server port
- `MAX_UPLOAD_MB`: upload limit, default `10`

## API

### Health

`GET /api/health`

### Prediction

`POST /api/predict` with multipart form field `file`.

Swagger docs are available at `/docs`.

> Educational/research use only. The model is not a medical diagnosis system.
