from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .config import ALLOWED_EXTENSIONS, CONFIG_PATH, CORS_ORIGINS, MAX_UPLOAD_MB, MODEL_PATH, MODEL_S3_URI
from .model_service import ONNXModelService
from .schemas import HealthResponse, PredictionResponse

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "webapplication" / "frontend"

app = FastAPI(
    title="NeuraMRI — Brain Tumor MRI Classifier",
    version="1.1.0",
    description="Server-side FastAPI inference API for the exported EfficientNet-B3 ONNX model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

service = ONNXModelService(MODEL_PATH, CONFIG_PATH, MODEL_S3_URI)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "public, max-age=3600"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health/live")
def live():
    return {"status": "alive"}


@app.get("/api/health/ready", response_model=HealthResponse)
def ready():
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.load_error or "Model is not ready")
    return health()


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if service.ready else "model_not_ready",
        model_ready=service.ready,
        model_format="ONNX",
        model_path=service.model_path.name,
        classes=service.classes,
        provider=service.provider,
        error=service.load_error,
    )


@app.get("/api/model-info")
def model_info():
    return {
        "ready": service.ready,
        "model": service.model_path.name,
        "input_size": service.image_size,
        "classes": service.classes,
        "provider": service.provider,
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported image type. Use PNG, JPG, JPEG, WEBP, or BMP.")
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image is too large. Maximum size is {MAX_UPLOAD_MB} MB.")
    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    try:
        return service.predict(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed. Check server logs.")
