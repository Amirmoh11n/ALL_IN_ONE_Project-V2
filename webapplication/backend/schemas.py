from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Probability(BaseModel):
    class_name: str
    value: float = Field(ge=0.0, le=1.0)
    percentage: float = Field(ge=0.0, le=100.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_percentage: float = Field(ge=0.0, le=100.0)
    probabilities: Dict[str, float]
    ranked_probabilities: List[Probability]
    model: str
    device: str
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_format: str
    model_path: str
    classes: List[str]
    provider: str
    error: Optional[str] = None
