from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "artifacts" / "exports"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(EXPORT_DIR / "brain_tumor_efficientnet_b4.onnx")))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
MODEL_S3_URI = os.getenv("MODEL_S3_URI", "")
