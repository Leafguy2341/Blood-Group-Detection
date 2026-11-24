from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.feature_extraction import extract_image_features


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "best_model.joblib"
MODEL_PATH_ENV = "BLOOD_GROUP_MODEL_PATH"

logger = logging.getLogger("blood_group_api")


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_path: str


class PredictionResponse(BaseModel):
    filename: str
    predicted_blood_group: str
    confidence_score: float | None


class ModelBundle:
    def __init__(self, package: dict[str, Any], path: Path) -> None:
        self.package = package
        self.path = path
        self.model = package["model"]
        self.feature_columns = list(package["feature_columns"])
        self.classes = list(package.get("classes") or getattr(self.model, "classes_", []))
        self.model_name = str(package.get("model_name") or infer_estimator_name(self.model))


def resolve_model_path() -> Path:
    configured_path = os.getenv(MODEL_PATH_ENV)
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = APP_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"{MODEL_PATH_ENV} points to a missing file: {path}")
        return path

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    candidates = sorted(MODEL_DIR.rglob("*.joblib"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .joblib model packages found under {MODEL_DIR}")
    return candidates[0]


def validate_model_package(package: dict[str, Any], model_path: Path) -> None:
    required_keys = {"model", "feature_columns"}
    missing = required_keys - set(package.keys())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Model package {model_path.name} is missing required keys: {missing_text}")
    if not package["feature_columns"]:
        raise ValueError(f"Model package {model_path.name} has empty feature_columns.")


def patch_legacy_model_compatibility(model: Any) -> None:
    """Patch older sklearn pickles that miss attributes expected by newer sklearn."""
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            patch_legacy_model_compatibility(step)
        return

    if model.__class__.__name__ == "SimpleImputer" and not hasattr(model, "_fill_dtype"):
        model._fill_dtype = getattr(model, "_fit_dtype", None)


def infer_estimator_name(model: Any) -> str:
    if hasattr(model, "named_steps") and model.named_steps:
        return model.named_steps[list(model.named_steps.keys())[-1]].__class__.__name__
    return model.__class__.__name__


def load_model_bundle() -> ModelBundle:
    model_path = resolve_model_path()
    package = joblib.load(model_path)
    validate_model_package(package, model_path)
    patch_legacy_model_compatibility(package["model"])
    bundle = ModelBundle(package=package, path=model_path)
    logger.info("Loaded model %s from %s", bundle.model_name, bundle.path)
    return bundle


def get_model_bundle(app: FastAPI) -> ModelBundle:
    bundle = getattr(app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return bundle


def build_feature_frame(features: dict[str, float | int], feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([features])
    frame = frame.reindex(columns=feature_columns, fill_value=0.0)
    frame = frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return frame.astype(np.float64)


def confidence_for_prediction(model: Any, prediction: str, probabilities: np.ndarray, fallback_classes: list[Any]) -> float:
    row = np.asarray(probabilities, dtype=np.float64)
    if row.ndim == 2:
        row = row[0]

    classes = list(getattr(model, "classes_", []) or fallback_classes)
    class_labels = [str(label) for label in classes]
    if prediction in class_labels:
        return float(row[class_labels.index(prediction)])
    return float(row.max())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_bundle()
    yield


app = FastAPI(
    title="Blood Group Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = get_model_bundle(app)
    return HealthResponse(
        status="ok",
        model_name=bundle.model_name,
        model_path=str(bundle.path.relative_to(APP_DIR) if bundle.path.is_relative_to(APP_DIR) else bundle.path),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    bundle = get_model_bundle(app)
    filename = file.filename or "uploaded_image"

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        encoded = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a JPG, JPEG, or PNG image.")

        features = extract_image_features(image)
        input_frame = build_feature_frame(features, bundle.feature_columns)

        prediction = str(bundle.model.predict(input_frame)[0])
        if not hasattr(bundle.model, "predict_proba"):
            raise RuntimeError("Loaded model does not support predict_proba.")

        probabilities = bundle.model.predict_proba(input_frame)
        confidence = confidence_for_prediction(bundle.model, prediction, probabilities, bundle.classes)

        return PredictionResponse(
            filename=filename,
            predicted_blood_group=prediction,
            confidence_score=confidence,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc
