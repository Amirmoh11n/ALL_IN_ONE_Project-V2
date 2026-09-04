# 🧠 Brain Tumor MRI Classification

<p align="center">
  <img width="300" height="300" alt="logo" src="https://github.com/user-attachments/assets/b20eea47-9411-461f-ab31-53d984bd9b41" ,alt="Brain Tumor MRI Classification Logo"/>
</p>

<p align="center">
  <strong>End-to-End Brain MRI Classification & Deployment Pipeline</strong>
</p>

<p align="center">
  A production-oriented PyTorch project for classifying brain MRI images into four categories using EfficientNet-B4 transfer learning, MLflow experiment tracking, validated model export, and a FastAPI + ONNX web inference service.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch\&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow\&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker\&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

</p>

---
## Web Application 
<img width="1080" height="608" alt="V2P22" src="https://github.com/user-attachments/assets/42b0134c-ba49-4753-ac8e-95ddc090fef2" />

---
## Overview

**Brain Tumor MRI Classification** is an end-to-end deep learning system designed to classify brain MRI images into four categories:

| Class        | Description      |
| ------------ | ---------------- |
| `glioma`     | Glioma tumor     |
| `meningioma` | Meningioma tumor |
| `notumor`    | No visible tumor |
| `pituitary`  | Pituitary tumor  |

The project focuses not only on model training, but on the complete machine-learning lifecycle:

```text
Dataset
   ↓
Data Validation & Preparation
   ↓
Stratified Train / Validation Split
   ↓
Data Augmentation
   ↓
EfficientNet-B4 Transfer Learning
   ↓
Weighted Cross-Entropy Training
   ↓
Validation & Early Stopping
   ↓
Best Model Checkpoint
   ↓
MLflow Experiment Tracking
   ↓
Independent Test Evaluation
   ↓
Model Export & Validation
   ↓
FastAPI + ONNX Inference
   ↓
Web Application / Cloud Deployment
```

---

## ✅ Status

| Component | V2.0 |
| --- | --- |
| EfficientNet-B4 (+ B3 fallback) | Done |
| Config-driven `val_f1_macro` checkpoint | Done |
| Auto patient-aware / stratified split | Done |
| Data quality filter | Done |
| Mixup/CutMix (off by default) | Done |
| Multi-seed (3) mean ± std | Done |
| Specificity, PPV/NPV, ECE, temperature | Done |
| Confusion + ROC plots | Done |
| Grad-CAM artifacts (not web) | Done |
| Stable ONNX (legacy exporter, no Lite) | Done |
| Batch CSV inference | Done |
| Web i18n EN / FA + disclaimer | Done |
| Docker serving image | Done |
| CI lint + unit tests | Done |
| Model card | `docs/MODEL_CARD.md` |

**Not a clinical device.** See the model card.

### Test metrics (fill after `brain-tumor evaluate`)

After evaluation, copy `artifacts/evaluation/metrics_table.md` here.

```text
uv run brain-tumor train
uv run brain-tumor evaluate
uv run brain-tumor export
uv run brain-tumor explain
uv run brain-tumor predict --image path/to/slice.jpg
uv run brain-tumor predict-batch --folder path/to/folder --output artifacts/evaluation/batch.csv
```

---


## ✨ Key Features

* 🧠 **EfficientNet-B4** with ImageNet transfer learning
* 📊 Stratified train/validation splitting
* ⚖️ Class-weighted `CrossEntropyLoss`
* 🚀 Adam optimizer
* 📉 `ReduceLROnPlateau` learning-rate scheduling
* ⏹️ Early stopping with best-checkpoint restoration
* 🔬 MLflow experiment tracking
* 📈 Comprehensive test evaluation
* 📦 Multiple deployment-oriented model exports
* 🔄 ONNX Runtime inference validation
* 🖼️ Single-image prediction CLI
* 🌐 FastAPI inference backend
* 💻 Browser-based web interface
* 🐳 Docker support
* ☁️ AWS ECS / Fargate deployment support(Not implement)
* 🔐 Optional private S3 model loading
* 🧪 Automated test suite
* ⚙️ Centralized YAML configuration
* 🧰 `uv`-based dependency management

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │   Kaggle / Local    │
                         │    MRI Dataset      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Pipeline     │
                         │                     │
                         │ • Validation        │
                         │ • Stratified Split  │
                         │ • Augmentation      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   EfficientNet-B4   │
                         │ ImageNet Transfer   │
                         │      Learning       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Trainer        │
                         │                     │
                         │ • Weighted CE       │
                         │ • Adam              │
                         │ • LR Scheduler      │
                         │ • AMP               │
                         │ • Early Stopping    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       Best Checkpoint        │
                    │       + MLflow Run            │
                    └──────────────┬───────────────┘
                                   │
                 ┌─────────────────┼──────────────────┐
                 │                 │                  │
                 ▼                 ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │   Evaluation   │ │     Export     │ │   Inference    │
        │                │ │                │ │                │
        │ Accuracy       │ │ TorchScript    │ │ Single MRI     │
        │ Precision      │ │ PyTorch Lite   │ │ Prediction     │
        │ Recall         │ │ ONNX           │ │                │
        │ F1             │ │                │ │                │
        │ ROC-AUC        │ │                │ │                │
        └────────────────┘ └───────┬────────┘ └───────┬────────┘
                                   │                  │
                                   ▼                  ▼
                             ┌─────────────┐   ┌──────────────┐
                             │ ONNX Runtime│   │   FastAPI    │
                             │ Validation   │   │ Web Service  │
                             └─────────────┘   └──────┬───────┘
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │ Web Browser   │
                                                └──────────────┘
```

---

# 📂 Project Structure

```text
.
├── configs/
│   └── config.yaml                  # Single source of truth for all hyperparameters
│
├── data/
│   └── raw/                         # Dataset (auto-downloaded via kagglehub if missing)
│       ├── Training/
│       └── Testing/
│
├── artifacts/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── history.json
│   ├── evaluation/
│   │   └── test_metrics.json
│   ├── exports/
│   │   ├── *_web.pt                 # TorchScript (web)
│   │   ├── *_gpu.pt                 # Frozen TorchScript
│   │   ├── *_mobile.ptl             # PyTorch Lite
│   │   └── *.onnx                   # ONNX
│   └── mlruns/                      # MLflow tracking
│
├── src/
│   ├── cli.py                       # Entry point: train / evaluate / export / predict
│   ├── data/                        # Acquisition, split, augment, Dataset, DataPipeline
│   ├── models/                      # EfficientNet-B4 classifier + factory
│   ├── engine/                      # Trainer (AMP, early stopping, LR scheduler, MLflow)
│   ├── metrics/                     # Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
│   ├── evaluate/                    # Independent test-set evaluation
│   ├── export/                      # TorchScript / Lite / ONNX exporters with validation
│   ├── inference/                   # Single-image InferencePipeline
│   └── utils/                       # ConfigLoader, checkpoint, logging
│
├── tests/                           # Full unit + integration suite (85 tests)
│   ├── data/  engine/  evaluate/  export/
│   ├── inference/  metrics/  models/  utils/  web/
│
├── webapplication/
│   ├── backend/                     # FastAPI + ONNX Runtime service
│   └── frontend/                    # Simple browser UI
│
├── docker/                          # Dockerfile(s)
├── aws/                             # Not Implement
├── scripts/
├── start_webapp.bash
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

---

# 🚀 Quick Start

This project uses **[uv](https://docs.astral.sh/uv/)** for environment and dependency management.

### 1. Clone

```bash
git clone https://github.com/Amirmoh11n/ALL_IN_ONE_Project-V2/
cd ALL_IN_ONE_Project-V2
```

### 2. Setup

```bash
chmod +x scripts/setup.bash scripts/run.bash
./scripts/setup.bash
```

The setup script:

* Creates the virtual environment
* Installs project dependencies
* Creates required artifact directories
* Prepares the project for training

---

# 📦 Dataset

The pipeline supports automatic dataset acquisition when the expected dataset directories are missing.

Expected structure:

```text
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

If the dataset already exists under `data/raw/`, Kaggle authentication is not required.

### Kaggle Authentication

Using `kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Or environment variables:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

---

# 🧪 Testing

Run the complete test suite:

```bash
uv run pytest tests -v
```

The suite currently contains **85 tests** covering data pipeline, model, trainer (including AMP / early-stopping / class weights / MLflow), metrics, evaluation, export (TorchScript, Lite, ONNX with validation), inference, and the FastAPI backend.

All tests pass on the current codebase.

---

# 🎯 Training

Run the complete training pipeline:

```bash
uv run brain-tumor train
```

For a quick smoke test:

```bash
uv run brain-tumor train --epochs 2
```

Training pipeline:

```text
Dataset
   ↓
Stratified Split
   ↓
Augmentation
   ↓
DataLoader
   ↓
EfficientNet-B4
   ↓
Weighted CrossEntropy
   ↓
Adam
   ↓
ReduceLROnPlateau
   ↓
Validation
   ↓
Early Stopping
   ↓
Best Checkpoint
```

Outputs:

```text
artifacts/checkpoints/
├── best_model.pt
└── history.json
```

---

# 📊 MLflow Experiment Tracking

MLflow is used to track experiments, parameters, metrics, and model-related artifacts.

Start the local MLflow server:

```bash
./scripts/run.bash mlflow
```

Then open the URL printed by MLflow.

Experiment configuration is available in:

```text
configs/config.yaml
```

---

# 🔬 Final Evaluation

The final evaluation is performed against the **untouched Testing split**:

```bash
uv run brain-tumor evaluate
```

Results:

```text
artifacts/evaluation/test_metrics.json
```

### Metrics

The evaluation report includes:

* Accuracy
* Macro Precision
* Macro Recall / Sensitivity
* Macro F1
* Macro ROC-AUC (OvR)
* Per-class Precision
* Per-class Recall
* Per-class F1
* Confusion Matrix

### Results

> Add the final test results here after the final training run.

Example:

```text
Accuracy:        XX.XX%
Macro Precision: XX.XX%
Macro Recall:    XX.XX%
Macro F1:        XX.XX%
ROC-AUC (OvR):   XX.XX%
```

A confusion matrix can also be added here:

```text
                Predicted
              G   M   N   P
Actual   G    ·   ·   ·   ·
         M    ·   ·   ·   ·
         N    ·   ·   ·   ·
         P    ·   ·   ·   ·
```

---

# 📦 Model Export

After successful training:

```bash
uv run brain-tumor export
```

The exporter generates:

| Artifact          | Purpose                          |
| ----------------- | -------------------------------- |
| `*_web.pt`        | TorchScript inference            |
| `*_gpu.pt`        | GPU-oriented TorchScript         |
| `*_mobile.ptl`    | PyTorch Lite / mobile deployment |
| `*.onnx`          | Cross-platform inference         |
| `*_manifest.json` | Export metadata                  |

### ONNX Validation

The ONNX artifact is automatically validated using:

* `onnx.checker`
* ONNX Runtime inference
* PyTorch vs ONNX output comparison
* Dynamic batch-size testing

This helps ensure that the exported model is not only generated successfully, but is also executable and numerically consistent with the original model.

---

# 🖼️ Single-Image Inference

Run inference on a single MRI:

```bash
uv run brain-tumor predict \
  --image /absolute/path/to/mri.jpg
```

Example:

```json
{
  "predicted_class": "glioma",
  "confidence": 0.973,
  "probabilities": {
    "glioma": 0.973,
    "meningioma": 0.012,
    "notumor": 0.004,
    "pituitary": 0.011
  }
}
```

---

# 🌐 Web Application

The project includes a deployment-oriented web inference application.

### Stack

```text
Browser
   │
   ▼
HTML + Tailwind CSS + JavaScript
   │
   ▼
FastAPI
   │
   ▼
ONNX Runtime
   │
   ▼
EfficientNet-B4 ONNX Model
```

### Features

* MRI image upload
* PNG / JPG / JPEG / WEBP / BMP support
* 10 MB default upload limit
* Server-side model execution
* ONNX Runtime inference
* Health endpoint
* Readiness endpoint
* Docker support
* Cloud deployment support
* Optional private S3 model loading

The ONNX model is **never sent to the browser**.

### Start locally

After training and exporting:

```bash
./start_webapp.bash
```

The application will be available at:

```text
http://127.0.0.1:8000
```

---

# ☁️ Cloud Deployment

The web inference service is designed to support containerized deployment on AWS.

Supported deployment architecture:

```text
                    ┌───────────────┐
                    │    Browser    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     AWS       │
                    │ Load Balancer │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ECS / Fargate │
                    │   FastAPI     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ONNX Runtime  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Model / S3    │
                    └───────────────┘
```

For AWS deployment instructions:

```text
aws/README.md
```

The application can optionally load the model from a private S3 location using:

```text
MODEL_S3_URI
```

---

# ⚙️ Configuration

Experiment and runtime settings are centralized in:

```text
configs/config.yaml
```

Main sections:

```yaml
data:
model:
training:
evaluation:
export:
artifacts:
tracking:
logging:
```

Training hyperparameters are intentionally kept outside the source code to improve:

* Reproducibility
* Experiment management
* Configuration portability
* MLflow integration

---

# 🛠️ CLI

The main CLI commands are:

```bash
# Train
uv run brain-tumor train

# Evaluate
uv run brain-tumor evaluate

# Export
uv run brain-tumor export

# Predict
uv run brain-tumor predict --image /path/to/image.jpg

# Tests
uv run pytest tests -v
```

Or use the project runner:

```bash
./scripts/run.bash train
./scripts/run.bash evaluate
./scripts/run.bash export
./scripts/run.bash predict --image /path/to/image.jpg
./scripts/run.bash test
./scripts/run.bash mlflow
```

---

# 🔁 Reproducible ML Lifecycle

```text
┌──────────────────────────────────────────┐
│              DATA INGESTION              │
│          Kaggle / Local Dataset          │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│           DATA PREPARATION               │
│      Validation + Stratified Split       │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              TRAINING                   │
│ EfficientNet-B4 + Transfer Learning     │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│         EXPERIMENT TRACKING              │
│                MLflow                    │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              EVALUATION                  │
│      Untouched Testing Dataset           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│               EXPORT                     │
│       TorchScript / PTL / ONNX           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              DEPLOYMENT                  │
│        FastAPI + ONNX Runtime            │
└──────────────────────────────────────────┘
```

---

# ⚠️ Medical & Dataset Limitations

This project is intended for **machine-learning engineering, experimentation, and research purposes**.

It is **not a medical diagnostic system** and model predictions must not be interpreted as a clinical diagnosis or used as a substitute for qualified medical professionals.

The final test set is kept separate from training and validation. However, the available dataset does not provide patient-level identifiers; therefore, the validation strategy is based on **image-level stratification rather than patient-level splitting**.

This limitation should be considered when interpreting model performance and generalization.

---

# 🔐 Security & Deployment Notes

For production deployment:

* Do not commit Kaggle credentials.
* Do not commit private cloud credentials.
* Keep production models in controlled storage.
* Prefer private S3 buckets for cloud-hosted models.
* Configure upload-size limits appropriately.
* Expose only required API endpoints.
* Use HTTPS in production.
* Keep dependencies updated.
* Do not expose the ONNX model directly to clients.

---

# 📜 License

This project is licensed under the **Apache License 2.0**.

You may use, modify, distribute, and use this project for commercial purposes under the terms and conditions of the license.

See [`LICENSE`](LICENSE) for the full license text.

---

# 👨‍💻 Author

**Amirmohammad Nashalji**

Computer Engineering Student | Machine Learning & AI Engineer

---

<p align="center">
  Built with PyTorch, MLflow, ONNX Runtime, FastAPI, Docker and ❤️
</p>
