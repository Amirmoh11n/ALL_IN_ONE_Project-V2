# 🧠 Brain Tumor MRI Classification

<p align="center">
  <img width="300" height="300" alt="Brain Tumor MRI Classification Logo" src="https://github.com/user-attachments/assets/b20eea47-9411-461f-ab31-53d984bd9b41"/>
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

## 🌐 Web Application




---

# 📌 Overview

**Brain Tumor MRI Classification** is an end-to-end deep learning system designed to classify brain MRI images into four categories:

| Class        | Description      |
| ------------ | ---------------- |
| `glioma`     | Glioma tumor     |
| `meningioma` | Meningioma tumor |
| `notumor`    | No visible tumor |
| `pituitary`  | Pituitary tumor  |

The project is designed around the complete machine-learning lifecycle:

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
Web Application
```

---

# 📊 Training Results

The final training run used **EfficientNet-B4** with transfer learning and monitored `val_f1_macro` for model selection.

Training reached **28 epochs**, after which **Early Stopping** was triggered.

The best checkpoint was obtained at **Epoch 23**:

| Metric              | Best Validation Result |
| ------------------- | ---------------------: |
| Validation Accuracy |             **98.77%** |
| Validation Macro F1 |             **98.80%** |
| Validation Loss     |             **0.0729** |
| Training Loss       |             **0.0042** |
| Learning Rate       |              `2.5e-05` |
| Best Epoch          |            **23 / 30** |
| Early Stopping      |           **Epoch 28** |

### Best Checkpoint

```text
artifacts/checkpoints/best_model.pt
```

The model checkpoint was selected using:

```text
monitor = val_f1_macro
best value = 0.9880
```

Training log:

```text
Epoch 23/30
train_loss=0.0042
val_loss=0.0729
val_acc=0.9877
val_f1=0.9880
monitor(val_f1_macro)=0.9880
lr=2.5e-05

Saved best checkpoint ->
artifacts/checkpoints/best_model.pt
```

> **Important:** The 98.80% Macro F1 and 98.77% Accuracy above are **validation metrics**. Final test-set metrics are reported separately after running `brain-tumor evaluate`.

---

# 📈 Final Test Evaluation

The final evaluation is performed against the **untouched Testing split**:

```bash
uv run brain-tumor evaluate
```

Results are stored in:

```text
artifacts/evaluation/test_metrics.json
```

### Test Results

> Run the evaluation command and replace the values below with the actual test metrics.

```text
Accuracy:        XX.XX%
Macro Precision: XX.XX%
Macro Recall:    XX.XX%
Macro F1:        XX.XX%
ROC-AUC (OvR):   XX.XX%
```

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
* Specificity
* PPV / NPV
* ECE / calibration metrics

---

# ✨ Key Features

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
* 📂 Batch inference with CSV output
* 🌐 FastAPI inference backend
* 💻 Browser-based web interface
* 🐳 Docker support
* ☁️ AWS deployment architecture
* 🔐 Optional private S3 model loading
* 🧪 Automated test suite
* ⚙️ Centralized YAML configuration
* 🧰 `uv`-based dependency management
* 🔥 AMP / mixed-precision training
* 🎯 Config-driven checkpoint selection
* 🧪 Model explainability with Grad-CAM

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
│   └── config.yaml
│
├── data/
│   └── raw/
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
│   │   ├── *_web.pt
│   │   ├── *_gpu.pt
│   │   ├── *_mobile.ptl
│   │   ├── *.onnx
│   │   └── *_manifest.json
│   └── mlruns/
│
├── src/
│   ├── cli.py
│   ├── data/
│   ├── models/
│   ├── engine/
│   ├── metrics/
│   ├── evaluate/
│   ├── export/
│   ├── inference/
│   └── utils/
│
├── tests/
├── webapplication/
│   ├── backend/
│   └── frontend/
│
├── docker/
├── aws/
├── scripts/
├── start_webapp.bash
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

---

# 🚀 Quick Start

This project uses **uv** for environment and dependency management.

### 1. Clone

```bash
git clone https://github.com/Amirmoh11n/ALL_IN_ONE_Project-V2.git
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

Expected dataset structure:

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

The project supports automatic dataset acquisition through `kagglehub` when the expected dataset is unavailable locally.

---

# 🧪 Testing

Run the complete test suite:

```bash
uv run pytest tests -v
```

The project contains **85 tests** covering:

* Data pipeline
* Model
* Training
* AMP
* Early stopping
* Class weighting
* MLflow
* Metrics
* Evaluation
* Export
* ONNX validation
* Inference
* FastAPI backend
* Web functionality

---

# 🎯 Training

Run the training pipeline:

```bash
uv run brain-tumor train
```

For a smoke test:

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

Output:

```text
artifacts/checkpoints/
├── best_model.pt
└── history.json
```

---

# 📊 MLflow Experiment Tracking

MLflow is used to track:

* Experiments
* Hyperparameters
* Training metrics
* Validation metrics
* Model artifacts

Start the local MLflow server:

```bash
./scripts/run.bash mlflow
```

Configuration:

```text
configs/config.yaml
```

---

# 🔬 Evaluation

Run:

```bash
uv run brain-tumor evaluate
```

The evaluation uses the untouched Testing split:

```text
Training
   ├── Train
   └── Validation

Testing
   └── Final Evaluation
```

Results:

```text
artifacts/evaluation/test_metrics.json
```

---

# 📦 Model Export

After training:

```bash
uv run brain-tumor export
```

Generated artifacts include:

| Artifact          | Purpose                  |
| ----------------- | ------------------------ |
| `*_web.pt`        | TorchScript inference    |
| `*_gpu.pt`        | GPU-oriented TorchScript |
| `*_mobile.ptl`    | Mobile deployment        |
| `*.onnx`          | Cross-platform inference |
| `*_manifest.json` | Export metadata          |

### ONNX Validation

The ONNX model is validated using:

* `onnx.checker`
* ONNX Runtime inference
* PyTorch vs ONNX output comparison
* Dynamic batch-size testing

This verifies that the exported model is executable and numerically consistent with the original PyTorch model.

---

# 🖼️ Single-Image Inference

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

# 📂 Batch Inference

Run inference over a complete folder:

```bash
uv run brain-tumor predict-batch \
  --folder path/to/folder \
  --output artifacts/evaluation/batch.csv
```

The resulting CSV can be used for further analysis and error inspection.

---

# 🌐 Web Application

The project includes a FastAPI + ONNX Runtime inference service.

### Technology Stack

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
EfficientNet-B4 ONNX
```

### Features

* MRI image upload
* JPG / JPEG / PNG / WEBP / BMP support
* Upload-size limitation
* Server-side inference
* ONNX Runtime
* Health endpoint
* Readiness endpoint
* Docker support
* Optional private S3 model loading
* EN / FA interface
* Medical disclaimer

The ONNX model is **never sent to the browser**.

### Start locally

```bash
./start_webapp.bash
```

Then open:

```text
http://127.0.0.1:8000
```

---

# ☁️ Cloud Deployment

The architecture is prepared for containerized deployment on AWS:

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

> AWS deployment is currently **not implemented**.

---

# ⚙️ Configuration

All major experiment and runtime settings are centralized in:

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

This improves:

* Reproducibility
* Experiment management
* Configuration portability
* MLflow integration

---

# 🛠️ CLI

```bash
# Train
uv run brain-tumor train

# Evaluate
uv run brain-tumor evaluate

# Export
uv run brain-tumor export

# Explainability
uv run brain-tumor explain

# Single prediction
uv run brain-tumor predict --image /path/to/image.jpg

# Batch prediction
uv run brain-tumor predict-batch \
  --folder /path/to/folder \
  --output artifacts/evaluation/batch.csv

# Tests
uv run pytest tests -v
```

Project runner:

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
│              TRAINING                    │
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
│        Untouched Testing Dataset         │
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

It is **not a medical diagnostic system**.

Model predictions must not be interpreted as a clinical diagnosis or used as a substitute for qualified medical professionals.

The final test set is kept separate from training and validation. However, the available dataset does not provide patient-level identifiers. Therefore, the validation strategy is based on **image-level stratification rather than patient-level splitting**.

This limitation should be considered when interpreting model performance and generalization.

---

# 🔐 Security & Deployment Notes

For production deployment:

* Do not commit Kaggle credentials.
* Do not commit cloud credentials.
* Keep production models in controlled storage.
* Prefer private S3 buckets for cloud-hosted models.
* Configure upload-size limits.
* Expose only required API endpoints.
* Use HTTPS in production.
* Keep dependencies updated.
* Do not expose the ONNX model directly to clients.

---

# 📌 Roadmap

* [x] EfficientNet-B4 training pipeline
* [x] Validation and early stopping
* [x] MLflow experiment tracking
* [x] Comprehensive evaluation
* [x] ONNX export and validation
* [x] FastAPI inference service
* [x] Web application
* [x] Docker support
* [x] Grad-CAM explainability
* [x] Batch inference
* [ ] AWS production deployment
* [ ] Automated model versioning
* [ ] CI/CD pipeline
* [ ] API authentication
* [ ] Production monitoring
* [ ] Inference latency benchmarking
* [ ] Model performance dashboard
* [ ] Patient-level evaluation when appropriate metadata becomes available

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
