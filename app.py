import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional runtime dependency
    st = None


APP_TITLE = "Hệ thống nhận diện và dự đoán nhóm máu"
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MAX_IMAGE_SIDE = 1200
MASK_NAMES = [
    "anti_a_blue",
    "anti_b_yellow",
    "blood_red",
    "foreground",
    "reaction_candidate",
]


app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def configure_streamlit() -> None:
    if st is None:
        return
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )


configure_streamlit()


def find_model_files(model_dir: Path = MODEL_DIR) -> list[Path]:
    """Return all saved model packages, newest first."""
    if not model_dir.exists():
        return []
    return sorted(model_dir.rglob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


def cache_resource(*args: Any, **kwargs: Any):
    if st is None:
        def decorator(func):
            return func

        return decorator
    return st.cache_resource(*args, **kwargs)


@lru_cache(maxsize=4)
def load_model_package(model_path: str) -> tuple[dict, Path]:
    path = Path(model_path)
    package = joblib.load(path)
    validate_model_package(package, path)
    patch_legacy_model_compatibility(package["model"])
    return package, path


def validate_model_package(package: dict, model_path: Path) -> None:
    required_keys = {"model", "feature_columns", "target"}
    missing = required_keys - set(package.keys())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Model package {model_path.name} thiếu key: {missing_text}")
    if not package["feature_columns"]:
        raise ValueError(f"Model package {model_path.name} không có feature_columns.")

def patch_legacy_model_compatibility(model) -> None:
    """Patch legacy sklearn estimators loaded from older pickle formats."""
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            patch_legacy_model_compatibility(step)
        return

    if model.__class__.__name__ == "SimpleImputer" and not hasattr(model, "_fill_dtype"):
        # scikit-learn 1.8 expects this attribute on older pickles.
        model._fill_dtype = getattr(model, "_fit_dtype", None)


def entropy_from_values(values: np.ndarray, bins: int = 32, value_range=(0, 255)) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def normalized_histogram(values: np.ndarray, bins: int, value_range, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_hist_{i}": 0.0 for i in range(bins)}

    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = hist.sum()
    hist = hist.astype(np.float64) / total if total else np.zeros_like(hist, dtype=np.float64)
    return {f"{prefix}_hist_{i}": hist[i] for i in range(bins)}


def lbp_image(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << i)
    return lbp


def lbp_features(gray: np.ndarray, mask: np.ndarray, prefix: str, bins: int = 16) -> dict:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        feats = {f"{prefix}_lbp_hist_{i}": 0.0 for i in range(bins)}
        feats[f"{prefix}_lbp_entropy"] = 0.0
        return feats

    lbp = lbp_image(gray)
    inner_mask = mask[1:-1, 1:-1]
    values = lbp[inner_mask]
    feats = normalized_histogram(values, bins=bins, value_range=(0, 256), prefix=f"{prefix}_lbp")
    feats[f"{prefix}_lbp_entropy"] = entropy_from_values(values, bins=bins, value_range=(0, 256))
    return feats


def glcm_features(gray: np.ndarray, mask: np.ndarray, prefix: str, levels: int = 16) -> dict:
    empty_result = {
        f"{prefix}_glcm_contrast": 0.0,
        f"{prefix}_glcm_homogeneity": 0.0,
        f"{prefix}_glcm_energy": 0.0,
        f"{prefix}_glcm_entropy": 0.0,
    }
    if gray.shape[0] < 2 or gray.shape[1] < 2 or mask.sum() < 2:
        return empty_result

    quantized = np.clip((gray.astype(np.float32) / 256 * levels).astype(np.int32), 0, levels - 1)
    valid = mask[:, :-1] & mask[:, 1:]
    left = quantized[:, :-1][valid]
    right = quantized[:, 1:][valid]
    matrix = np.zeros((levels, levels), dtype=np.float64)

    for a, b in zip(left, right):
        matrix[a, b] += 1
        matrix[b, a] += 1

    total = matrix.sum()
    if total == 0:
        return empty_result

    p = matrix / total
    i, j = np.indices(p.shape)
    nonzero = p[p > 0]
    return {
        f"{prefix}_glcm_contrast": float(((i - j) ** 2 * p).sum()),
        f"{prefix}_glcm_homogeneity": float((p / (1 + np.abs(i - j))).sum()),
        f"{prefix}_glcm_energy": float(np.sqrt((p ** 2).sum())),
        f"{prefix}_glcm_entropy": float(-(nonzero * np.log2(nonzero)).sum()),
    }


def connected_component_features(mask: np.ndarray) -> dict:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    areas = [stats[label, cv2.CC_STAT_AREA] for label in range(1, num_labels)]
    if not areas:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
        }

    largest = max(areas)
    return {
        "component_count": len(areas),
        "largest_component_area": largest,
        "largest_component_ratio": largest / mask.size,
    }


def contour_shape_features(mask: np.ndarray, prefix: str) -> dict:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            f"{prefix}_contour_count": 0,
            f"{prefix}_max_contour_area_ratio": 0.0,
            f"{prefix}_max_contour_perimeter_ratio": 0.0,
            f"{prefix}_max_contour_circularity": 0.0,
        }

    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float64)
    perimeters = np.array([cv2.arcLength(c, True) for c in contours], dtype=np.float64)
    idx = int(np.argmax(areas))
    area = areas[idx]
    perimeter = perimeters[idx]
    circularity = (4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    h, w = mask.shape
    return {
        f"{prefix}_contour_count": len(contours),
        f"{prefix}_max_contour_area_ratio": area / mask.size,
        f"{prefix}_max_contour_perimeter_ratio": perimeter / (2 * (h + w)),
        f"{prefix}_max_contour_circularity": circularity,
    }


def clean_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 80) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def segment_image(rgb: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    anti_a_blue = (h >= 80) & (h <= 135) & (s >= 35) & (v >= 35)
    anti_b_yellow = (h >= 15) & (h <= 45) & (s >= 35) & (v >= 45)
    blood_red = (((h <= 12) | (h >= 165)) & (s >= 30) & (v >= 30))
    foreground = ((s >= 38) & (v >= 30) & (v <= 250)) | ((s >= 18) & (v <= 150))

    masks = {
        "anti_a_blue": clean_mask(anti_a_blue, kernel_size=5, min_area=120),
        "anti_b_yellow": clean_mask(anti_b_yellow, kernel_size=5, min_area=120),
        "blood_red": clean_mask(blood_red, kernel_size=5, min_area=120),
        "foreground": clean_mask(foreground, kernel_size=5, min_area=200),
    }
    masks["reaction_candidate"] = masks["foreground"] & (
        masks["blood_red"] | masks["anti_a_blue"] | masks["anti_b_yellow"]
    )
    return masks


def features_for_mask(rgb: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    values_rgb = rgb[mask]
    values_hsv = hsv[mask]
    gray_values = gray[mask]
    edge_values = edges[mask]
    grad_values = grad_mag[mask]

    result = {
        f"{prefix}_mask_present": int(mask.any()),
        f"{prefix}_area_ratio": float(mask.mean()),
    }
    result.update({f"{prefix}_{k}": v for k, v in connected_component_features(mask).items()})
    result.update(contour_shape_features(mask, prefix))

    if values_rgb.size == 0:
        for name in [
            "r_mean",
            "g_mean",
            "b_mean",
            "r_std",
            "g_std",
            "b_std",
            "h_mean",
            "s_mean",
            "v_mean",
            "h_std",
            "s_std",
            "v_std",
            "gray_mean",
            "gray_std",
            "edge_density",
            "gradient_mean",
            "gradient_std",
            "entropy",
        ]:
            result[f"{prefix}_{name}"] = 0.0
        result.update(normalized_histogram(np.array([]), bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
        result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
        result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
        result.update(lbp_features(gray, mask, prefix))
        result.update(glcm_features(gray, mask, prefix))
        return result

    result.update(
        {
            f"{prefix}_r_mean": values_rgb[:, 0].mean(),
            f"{prefix}_g_mean": values_rgb[:, 1].mean(),
            f"{prefix}_b_mean": values_rgb[:, 2].mean(),
            f"{prefix}_r_std": values_rgb[:, 0].std(),
            f"{prefix}_g_std": values_rgb[:, 1].std(),
            f"{prefix}_b_std": values_rgb[:, 2].std(),
            f"{prefix}_h_mean": values_hsv[:, 0].mean(),
            f"{prefix}_s_mean": values_hsv[:, 1].mean(),
            f"{prefix}_v_mean": values_hsv[:, 2].mean(),
            f"{prefix}_h_std": values_hsv[:, 0].std(),
            f"{prefix}_s_std": values_hsv[:, 1].std(),
            f"{prefix}_v_std": values_hsv[:, 2].std(),
            f"{prefix}_gray_mean": gray_values.mean(),
            f"{prefix}_gray_std": gray_values.std(),
            f"{prefix}_edge_density": float((edge_values > 0).mean()),
            f"{prefix}_gradient_mean": grad_values.mean(),
            f"{prefix}_gradient_std": grad_values.std(),
            f"{prefix}_entropy": entropy_from_values(gray_values),
        }
    )
    result.update(normalized_histogram(values_hsv[:, 0], bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
    result.update(normalized_histogram(values_hsv[:, 1], bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
    result.update(normalized_histogram(values_hsv[:, 2], bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
    result.update(lbp_features(gray, mask, prefix))
    result.update(glcm_features(gray, mask, prefix))
    return result


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh. Hãy thử file JPG, JPEG hoặc PNG khác.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_if_needed(rgb: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1:
        return rgb
    new_size = (int(w / scale), int(h / scale))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def extract_features_from_image(rgb: np.ndarray, expected_features: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rgb = resize_if_needed(rgb)
    features = {
        "resized_width": rgb.shape[1],
        "resized_height": rgb.shape[0],
    }

    masks = segment_image(rgb)
    for mask_name, mask in masks.items():
        features.update(features_for_mask(rgb, mask, mask_name))

    df_features = pd.DataFrame([features])
    for col in expected_features:
        if col not in df_features.columns:
            df_features[col] = 0.0
    return df_features[expected_features], masks


def model_label(path: Path) -> str:
    parent = path.parent.name.replace("_", " ")
    return f"{path.name} ({parent})"


def extract_model_timestamp(model_path: Path) -> str | None:
    match = re.search(r"(\d{8}_\d{6})", model_path.stem)
    return match.group(1) if match else None


def infer_model_name(model_path: Path, timestamp: str | None) -> str:
    stem = model_path.stem
    if timestamp:
        stem = stem.removesuffix(f"_{timestamp}")
    return stem.removeprefix("color_segmentation_blood_group_")


def accuracy_from_metrics_file(model_path: Path, model_name: str) -> float | None:
    timestamp = extract_model_timestamp(model_path)
    if not timestamp:
        return None

    metrics_files = sorted(model_path.parent.glob(f"*metrics_{timestamp}.csv"))
    for metrics_file in metrics_files:
        try:
            metrics_df = pd.read_csv(metrics_file)
        except Exception:
            continue

        if "model" not in metrics_df.columns or "test_accuracy" not in metrics_df.columns:
            continue

        matched_rows = metrics_df[metrics_df["model"].astype(str) == model_name]
        if not matched_rows.empty:
            return float(matched_rows.iloc[0]["test_accuracy"])

    return None


def accuracy_from_report_file(model_path: Path) -> float | None:
    timestamp = extract_model_timestamp(model_path)
    if not timestamp:
        return None

    report_files = sorted(model_path.parent.glob(f"*classification_report_{timestamp}.txt"))
    for report_file in report_files:
        try:
            report_text = report_file.read_text(encoding="utf-8")
        except Exception:
            continue

        match = re.search(r"^\s*accuracy\s+([0-9.]+)", report_text, flags=re.MULTILINE)
        if match:
            return float(match.group(1))

    return None


def get_model_accuracy(package: dict, model_path: Path) -> float | None:
    for key in ("test_accuracy", "accuracy"):
        if key in package:
            return float(package[key])

    metrics = package.get("metrics")
    if isinstance(metrics, dict):
        for key in ("test_accuracy", "accuracy"):
            if key in metrics:
                return float(metrics[key])

    timestamp = extract_model_timestamp(model_path)
    model_name = str(package.get("model_name") or infer_model_name(model_path, timestamp))
    return accuracy_from_metrics_file(model_path, model_name) or accuracy_from_report_file(model_path)


def prediction_probabilities(model, input_df: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(input_df), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(input_df), dtype=np.float64)
        if scores.ndim == 1:
            pos = 1.0 / (1.0 + np.exp(-scores))
            probabilities = np.column_stack([1.0 - pos, pos])
        else:
            shifted = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    else:
        return None

    if probabilities.ndim == 1:
        probabilities = probabilities[np.newaxis, :]
    return probabilities


def top_predictions(model, input_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame | None:
    probabilities = prediction_probabilities(model, input_df)
    if probabilities is None:
        return None

    probabilities = probabilities[0]
    classes = model.classes_
    order = np.argsort(probabilities)[::-1][:top_n]
    return pd.DataFrame(
        {
            "nhom_mau": classes[order],
            "do_tin_cay": probabilities[order],
        }
    )


def prediction_confidence(top_df: pd.DataFrame | None) -> float | None:
    if top_df is None or top_df.empty:
        return None
    return float(top_df.iloc[0]["do_tin_cay"])


def predict_from_image_bytes(
    file_bytes: bytes,
    model_paths: list[Path] | None = None,
) -> dict[str, Any]:
    rgb = decode_uploaded_image_bytes(file_bytes)
    if not model_paths:
        model_files = find_model_files()
        if not model_files:
            raise FileNotFoundError("Không tìm thấy file model .joblib trong thư mục models.")
        model_paths = [model_files[0]]

    results: list[dict[str, Any]] = []
    for model_path in model_paths:
        package, resolved_path = load_model_package(str(model_path))
        model = package["model"]
        feature_columns = package["feature_columns"]

        input_df, _ = extract_features_from_image(rgb, feature_columns)
        prediction = str(model.predict(input_df)[0])
        top_df = top_predictions(model, input_df)
        confidence = prediction_confidence(top_df)

        item: dict[str, Any] = {
            "prediction": prediction,
            "model": resolved_path.name,
            "confidence": confidence,
            "top_predictions": [],
        }
        if top_df is not None:
            item["top_predictions"] = [
                {
                    "nhom_mau": str(row["nhom_mau"]),
                    "do_tin_cay": float(row["do_tin_cay"]),
                }
                for _, row in top_df.iterrows()
            ]
        results.append(item)

    primary = results[0]
    return {
        "prediction": primary["prediction"],
        "model": primary["model"],
        "confidence": primary["confidence"],
        "top_predictions": primary["top_predictions"],
        "results": results,
    }


def decode_uploaded_image_bytes(file_bytes: bytes) -> np.ndarray:
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh. Hãy thử file JPG, JPEG hoặc PNG khác.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Blood Group Predictor</title>
        <style>
          :root { color-scheme: light; }
          body {
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
            color: #4a1d1d;
          }
          .container {
            max-width: 980px;
            margin: 0 auto;
            padding: 24px 16px 40px;
          }
          .card {
            background: rgba(255,255,255,0.95);
            border: 1px solid #f3c4c4;
            border-radius: 20px;
            box-shadow: 0 14px 40px rgba(165, 42, 42, 0.12);
            padding: 24px;
          }
          h1 { margin-top: 0; font-size: clamp(24px, 4vw, 34px); }
          .muted { color: #7a4c4c; line-height: 1.6; }
          .grid { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 18px; margin-top: 18px; }
          .panel { border: 1px solid #f0d7d7; border-radius: 16px; padding: 16px; background: #fffdfd; }
          .dropzone { border: 2px dashed #d97878; border-radius: 14px; padding: 16px; text-align: center; }
          input[type="file"] { display: block; margin: 12px auto; }
          button {
            border: 0; border-radius: 999px; padding: 12px 18px; background: linear-gradient(90deg, #c92a2a, #e85d5d); color: white; cursor: pointer; font-weight: 700;
          }
          button:disabled { opacity: 0.6; cursor: wait; }
          .preview-box { min-height: 220px; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 12px; background: #f9ecec; }
          .preview-box img { max-width: 100%; max-height: 280px; border-radius: 12px; }
          .result-card { background: linear-gradient(135deg, #fff4f4, #ffe3e3); border-left: 8px solid #c92a2a; border-radius: 16px; padding: 18px; }
          .result-title { font-size: 18px; font-weight: 800; color: #8e2020; }
          .result-value { font-size: 36px; font-weight: 900; color: #7f1d1d; margin-top: 6px; }
          .confidence { margin-top: 10px; font-weight: 700; }
          .list { padding-left: 18px; }
          .pill { display: inline-block; padding: 6px 10px; background: #ffe1e1; border-radius: 999px; font-size: 12px; font-weight: 700; color: #9b2c2c; margin-top: 10px; }
          @media (max-width: 760px) {
            .grid { grid-template-columns: 1fr; }
            .container { padding: 12px 10px 28px; }
            .card { padding: 16px; }
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="card">
            <h1>Hệ thống dự đoán nhóm máu</h1>
            <p class="muted">Upload ảnh khay thử nhóm máu để xem kết quả trực quan trên web hoặc điện thoại. Hệ thống sẽ hiển thị ảnh đã chọn, kết quả dự đoán và top các nhánh có xác suất cao.</p>
            <form id="upload-form" enctype="multipart/form-data">
              <div class="dropzone">
                <label for="file-input">Chọn ảnh JPG, JPEG hoặc PNG</label>
                <input id="file-input" type="file" name="file" accept="image/*" required />
                <div style="margin-top:12px; text-align:left;">
                  <label for="model-select">Chọn model (có thể chọn nhiều):</label>
                  <select id="model-select" multiple size="5" style="width:100%; padding:8px; border-radius:8px; border:1px solid #d97878;"></select>
                </div>
                <button id="submit-btn" type="submit" style="margin-top:12px;">Dự đoán</button>
              </div>
            </form>
            <div class="grid">
              <div class="panel">
                <h3>Ảnh đã chọn</h3>
                <div class="preview-box" id="preview-box">
                  <span class="muted">Chưa có ảnh nào được chọn</span>
                </div>
              </div>
              <div class="panel">
                <h3>Kết quả</h3>
                <div id="result-box" class="result-card">
                  <div class="result-title">Chờ upload ảnh...</div>
                  <div class="result-value">—</div>
                  <div class="muted">Kết quả chỉ phục vụ demo và không thay thế xét nghiệm y tế.</div>
                </div>
                <div id="top-results" style="margin-top: 12px;"></div>
              </div>
            </div>
          </div>
        </div>
        <script>
          const form = document.getElementById('upload-form');
          const fileInput = document.getElementById('file-input');
          const modelSelect = document.getElementById('model-select');
          const previewBox = document.getElementById('preview-box');
          const resultBox = document.getElementById('result-box');
          const topResults = document.getElementById('top-results');
          const submitBtn = document.getElementById('submit-btn');

          async function loadModelOptions() {
            try {
              const response = await fetch('/models');
              const data = await response.json();
              if (!response.ok) throw new Error(data.detail || 'Không tải được danh sách model');
              data.models.forEach((value) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                modelSelect.appendChild(option);
              });
              if (modelSelect.options.length > 0) {
                modelSelect.options[0].selected = true;
              }
            } catch (error) {
              modelSelect.innerHTML = '<option value="">Không tải được model</option>';
            }
          }

          loadModelOptions();

          fileInput.addEventListener('change', () => {
            const file = fileInput.files && fileInput.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (event) => {
              previewBox.innerHTML = `<img src="${event.target.result}" alt="preview" />`;
            };
            reader.readAsDataURL(file);
          });

          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const file = fileInput.files && fileInput.files[0];
            if (!file) {
              alert('Vui lòng chọn ảnh trước khi dự đoán.');
              return;
            }

            submitBtn.disabled = true;
            submitBtn.textContent = 'Đang xử lý...';
            resultBox.innerHTML = '<div class="result-title">Đang phân tích...</div><div class="result-value">⏳</div><div class="muted">Đang trích xuất đặc trưng và dự đoán nhóm máu...</div>';
            topResults.innerHTML = '';

            try {
              const selectedModels = Array.from(modelSelect.selectedOptions).map((option) => option.value);
              const formData = new FormData();
              formData.append('file', file);
              selectedModels.forEach((modelName) => formData.append('models', modelName));
              const response = await fetch('/predict', { method: 'POST', body: formData });
              const data = await response.json();
              if (!response.ok) throw new Error(data.detail || 'Lỗi dự đoán');

              const confidence = data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A';
              resultBox.innerHTML = `
                <div class="result-title">Kết quả dự đoán</div>
                <div class="result-value">${data.prediction || 'Không rõ'}</div>
                <div class="confidence">Độ tin cậy: ${confidence}</div>
                <div class="pill">Model: ${data.model || 'unknown'}</div>
              `;

              if (data.top_predictions && data.top_predictions.length) {
                topResults.innerHTML = `
                  <div class="panel" style="padding: 14px;">
                    <h4 style="margin-top: 0;">Top dự đoán</h4>
                    <ul class="list">
                      ${data.top_predictions.map(item => `<li><strong>${item.nhom_mau}</strong> — ${(item.do_tin_cay * 100).toFixed(1)}%</li>`).join('')}
                    </ul>
                  </div>
                `;
              }

              if (data.results && data.results.length > 1) {
                const summary = data.results.map((item) => `- ${item.model}: ${item.prediction} (${item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : 'N/A'})`).join('<br>');
                topResults.innerHTML += `<div class="panel" style="padding: 14px; margin-top: 10px;"><h4 style="margin-top: 0;">So sánh model</h4><div class="muted">${summary}</div></div>`;
              }
            } catch (error) {
              resultBox.innerHTML = `<div class="result-title">Đã xảy ra lỗi</div><div class="result-value">⚠️</div><div class="muted">${error.message}</div>`;
            } finally {
              submitBtn.disabled = false;
              submitBtn.textContent = 'Dự đoán';
            }
          });
        </script>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_TITLE}


@app.get("/models")
def list_models() -> dict[str, Any]:
    model_files = find_model_files()
    return {"models": [path.name for path in model_files]}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    models: list[str] | None = Form(default=None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên file ảnh.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File ảnh rỗng.")

        selected_model_paths: list[Path] = []
        if models:
            available_model_paths = {path.name: path for path in find_model_files()}
            for model_name in models:
                resolved_path = available_model_paths.get(model_name)
                if resolved_path is not None:
                    selected_model_paths.append(resolved_path)
            if not selected_model_paths:
                raise HTTPException(status_code=400, detail="Không tìm thấy model nào phù hợp.")
        return predict_from_image_bytes(contents, selected_model_paths or None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {exc}") from exc


def render_sidebar(model_files: list[Path]) -> tuple[Path, bool, bool]:
    if st is None:
        raise RuntimeError("Streamlit không khả dụng trong môi trường này.")
    st.sidebar.header("Cấu hình")
    selected_label = st.sidebar.selectbox(
        "Model sử dụng",
        options=[model_label(path) for path in model_files],
        index=0,
    )
    selected_path = model_files[[model_label(path) for path in model_files].index(selected_label)]
    show_masks = st.sidebar.checkbox("Hiển thị mask phân đoạn", value=True)
    show_features = st.sidebar.checkbox("Hiển thị bảng feature", value=False)
    return selected_path, show_masks, show_features


def render_model_info(package: dict, model_path: Path) -> None:
    feature_columns = package["feature_columns"]
    accuracy = get_model_accuracy(package, model_path)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Thông tin model")
    st.sidebar.write(f"Thuật toán: `{package.get('model_name', 'N/A')}`")
    st.sidebar.write(f"Target: `{package.get('target', 'N/A')}`")
    st.sidebar.write(f"Số feature: `{len(feature_columns)}`")
    if accuracy is not None:
        st.sidebar.metric("Test accuracy", f"{accuracy:.2%}")
    else:
        st.sidebar.write("Test accuracy: `N/A`")
    st.sidebar.write(f"File: `{model_path.name}`")
    st.sidebar.caption(str(model_path.parent))


def render_mask_debug(masks: dict[str, np.ndarray]) -> None:
    st.subheader("Mask phân đoạn")
    cols = st.columns(len(MASK_NAMES))
    for idx, mask_name in enumerate(MASK_NAMES):
        mask = masks[mask_name].astype(np.uint8) * 255
        cols[idx].image(mask, caption=mask_name, use_container_width=True)


def render_prediction_card(prediction: str, top_df: pd.DataFrame | None) -> None:
    confidence_text = ""
    confidence = prediction_confidence(top_df)
    if confidence is not None:
        confidence_text = f"<p>Độ tin cậy của lần dự đoán: <strong>{confidence:.2%}</strong></p>"

    st.markdown(
        f"""
        <div style="
            background:#fff5f5;
            border:1px solid #ffd6d6;
            border-left:8px solid #c92a2a;
            border-radius:8px;
            padding:24px;
            text-align:center;">
            <p style="margin:0;color:#862e2e;font-weight:700;letter-spacing:0;">
                KẾT QUẢ DỰ ĐOÁN NHÓM MÁU
            </p>
            <h1 style="margin:8px 0 4px;color:#7f1d1d;font-size:72px;line-height:1;">
                {prediction}
            </h1>
            {confidence_text}
            <p style="margin:0;color:#555;">
                Kết quả chỉ phục vụ mục đích học tập và demo, không thay thế xét nghiệm y tế.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit không khả dụng trong môi trường này.")
    st.title(APP_TITLE)
    st.caption("Demo nhận diện nhóm máu từ ảnh khay thử bằng xử lý ảnh và machine learning.")

    model_files = find_model_files()
    if not model_files:
        st.error("Không tìm thấy file .joblib nào trong thư mục models hoặc các thư mục con.")
        st.stop()

    selected_model_path, show_masks, show_features = render_sidebar(model_files)

    try:
        model_package, model_path = load_model_package(str(selected_model_path))
    except Exception as exc:
        st.error(f"Không tải được model: {exc}")
        st.stop()

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    render_model_info(model_package, model_path)

    st.info("Ứng dụng này là bản demo nghiên cứu. Không sử dụng kết quả để ra quyết định chẩn đoán hay điều trị.")

    uploaded_file = st.file_uploader(
        "Tải lên ảnh khay thử nhóm máu",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.write("Hãy tải lên ảnh rõ nét, ánh sáng ổn định và nhìn thấy đầy đủ các vùng phản ứng.")
        return

    try:
        rgb = decode_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.subheader("Ảnh đầu vào")
        st.image(rgb, use_container_width=True)

    with right_col:
        st.subheader("Sẵn sàng phân tích")
        st.write(f"Kích thước ảnh: `{rgb.shape[1]} x {rgb.shape[0]}` px")
        run_prediction = st.button("Phân tích và dự đoán", type="primary", use_container_width=True)

    if not run_prediction:
        return

    with st.spinner("Đang phân đoạn màu, trích xuất feature và dự đoán..."):
        try:
            input_df, masks = extract_features_from_image(rgb, feature_columns)
            prediction = str(model.predict(input_df)[0])
            top_df = top_predictions(model, input_df)
        except Exception as exc:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {exc}")
            return

    result_tab, mask_tab, feature_tab = st.tabs(["Kết quả", "Mask", "Feature"])
    with result_tab:
        render_prediction_card(prediction, top_df)
        confidence = prediction_confidence(top_df)
        if confidence is not None:
            st.metric("Độ tin cậy của lần dự đoán", f"{confidence:.2%}")
            st.caption("Điểm này là xác suất model gán cho kết quả vừa dự đoán, không phải accuracy kiểm thử tổng thể.")
        if top_df is not None:
            st.subheader("Top dự đoán")
            st.dataframe(
                top_df.assign(do_tin_cay=top_df["do_tin_cay"].map(lambda value: f"{value:.2%}")),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Model hiện tại không hỗ trợ predict_proba nên app không hiển thị độ tin cậy.")

    with mask_tab:
        if show_masks:
            render_mask_debug(masks)
        else:
            st.caption("Bật tùy chọn 'Hiển thị mask phân đoạn' trong sidebar để xem mask.")

    with feature_tab:
        if show_features:
            st.dataframe(input_df, use_container_width=True)
        else:
            st.caption("Bật tùy chọn 'Hiển thị bảng feature' trong sidebar để xem feature đầu vào model.")


if __name__ == "__main__":
    main()
