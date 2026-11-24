"""Image feature extraction shared by the API inference service.

The saved estimators in ``models/`` were trained on handcrafted tabular
features, not raw pixels. This module mirrors the notebook/app feature names
so inference data can be reindexed against a model package's feature columns.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter


MAX_IMAGE_SIDE = 1200
MASK_NAMES = [
    "anti_a_blue",
    "anti_b_yellow",
    "blood_red",
    "foreground",
    "reaction_candidate",
]

FFT_PATCH_SIZE = 64
FFT_N_BANDS = 8
DOG_SCALES = [(1.0, 1.6), (2.0, 3.2), (4.0, 6.4), (8.0, 12.8)]

MORPH_RADII = [2, 4, 8, 16]
RADIAL_BINS = 10
ANGULAR_BINS = 8


def extract_image_features(image: np.ndarray) -> dict[str, float | int]:
    """Extract all supported handcrafted features from an OpenCV BGR image."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image payload.")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected a color image with three channels.")

    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    return extract_image_features_from_rgb(rgb)


def extract_image_features_from_rgb(rgb: np.ndarray) -> dict[str, float | int]:
    """Extract color, texture, frequency, and morphology features from RGB data."""
    rgb = resize_if_needed(rgb)
    masks = segment_image(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    features: dict[str, float | int] = {
        "resized_width": int(rgb.shape[1]),
        "resized_height": int(rgb.shape[0]),
    }

    for mask_name in MASK_NAMES:
        mask = masks.get(mask_name, np.zeros(gray.shape, dtype=bool))
        features.update(features_for_mask(rgb, mask, mask_name))

    for mask_name in MASK_NAMES:
        mask = masks.get(mask_name, np.zeros(gray.shape, dtype=bool))
        features[f"{mask_name}_mask_present"] = int(mask.any())
        features.update(fft_ring_features(gray, mask, mask_name))
        features.update(dog_wavelet_features(gray, mask, mask_name))
        features.update(morph_profile_features(mask, mask_name))
        features.update(spatial_distribution_features(mask, mask_name))

    return normalize_feature_values(features)


def normalize_feature_values(features: dict) -> dict[str, float | int]:
    """Convert numpy scalars into Python scalars for pandas/API compatibility."""
    normalized = {}
    for key, value in features.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = 0.0
        normalized[key] = value
    return normalized


def resize_if_needed(rgb: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1:
        return rgb
    new_size = (int(w / scale), int(h / scale))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def clean_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 80) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def segment_image(rgb: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    masks = {
        "anti_a_blue": clean_mask((h >= 80) & (h <= 135) & (s >= 35) & (v >= 35), 5, 120),
        "anti_b_yellow": clean_mask((h >= 15) & (h <= 45) & (s >= 35) & (v >= 45), 5, 120),
        "blood_red": clean_mask(((h <= 12) | (h >= 165)) & (s >= 30) & (v >= 30), 5, 120),
        "foreground": clean_mask(((s >= 38) & (v >= 30) & (v <= 250)) | ((s >= 18) & (v <= 150)), 5, 200),
    }
    masks["reaction_candidate"] = masks["foreground"] & (
        masks["blood_red"] | masks["anti_a_blue"] | masks["anti_b_yellow"]
    )
    return masks


def entropy_from_values(values: np.ndarray, bins: int = 32, value_range=(0, 255)) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    probabilities = hist.astype(np.float64) / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalized_histogram(values: np.ndarray, bins: int, value_range, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_hist_{i}": 0.0 for i in range(bins)}

    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = hist.sum()
    hist = hist.astype(np.float64) / total if total else np.zeros_like(hist, dtype=np.float64)
    return {f"{prefix}_hist_{i}": float(hist[i]) for i in range(bins)}


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


def lbp_features(gray: np.ndarray, mask: np.ndarray, prefix: str, bins: int = 16) -> dict[str, float]:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        result = {f"{prefix}_lbp_hist_{i}": 0.0 for i in range(bins)}
        result[f"{prefix}_lbp_entropy"] = 0.0
        return result

    lbp = lbp_image(gray)
    values = lbp[mask[1:-1, 1:-1]]
    result = normalized_histogram(values, bins=bins, value_range=(0, 256), prefix=f"{prefix}_lbp")
    result[f"{prefix}_lbp_entropy"] = entropy_from_values(values, bins=bins, value_range=(0, 256))
    return result


def glcm_features(gray: np.ndarray, mask: np.ndarray, prefix: str, levels: int = 16) -> dict[str, float]:
    empty = {
        f"{prefix}_glcm_contrast": 0.0,
        f"{prefix}_glcm_homogeneity": 0.0,
        f"{prefix}_glcm_energy": 0.0,
        f"{prefix}_glcm_entropy": 0.0,
    }
    if gray.shape[0] < 2 or gray.shape[1] < 2 or mask.sum() < 2:
        return empty

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
        return empty

    probabilities = matrix / total
    i, j = np.indices(probabilities.shape)
    nonzero = probabilities[probabilities > 0]
    return {
        f"{prefix}_glcm_contrast": float(((i - j) ** 2 * probabilities).sum()),
        f"{prefix}_glcm_homogeneity": float((probabilities / (1 + np.abs(i - j))).sum()),
        f"{prefix}_glcm_energy": float(np.sqrt((probabilities**2).sum())),
        f"{prefix}_glcm_entropy": float(-(nonzero * np.log2(nonzero)).sum()),
    }


def connected_component_features(mask: np.ndarray) -> dict[str, float | int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [stats[label, cv2.CC_STAT_AREA] for label in range(1, num_labels)]
    if not areas:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
        }

    largest = int(max(areas))
    return {
        "component_count": len(areas),
        "largest_component_area": largest,
        "largest_component_ratio": float(largest / mask.size),
    }


def contour_shape_features(mask: np.ndarray, prefix: str) -> dict[str, float | int]:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            f"{prefix}_contour_count": 0,
            f"{prefix}_max_contour_area_ratio": 0.0,
            f"{prefix}_max_contour_perimeter_ratio": 0.0,
            f"{prefix}_max_contour_circularity": 0.0,
        }

    areas = np.array([cv2.contourArea(contour) for contour in contours], dtype=np.float64)
    perimeters = np.array([cv2.arcLength(contour, True) for contour in contours], dtype=np.float64)
    idx = int(np.argmax(areas))
    area = float(areas[idx])
    perimeter = float(perimeters[idx])
    circularity = (4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    h, w = mask.shape
    return {
        f"{prefix}_contour_count": len(contours),
        f"{prefix}_max_contour_area_ratio": area / mask.size,
        f"{prefix}_max_contour_perimeter_ratio": perimeter / (2 * (h + w)),
        f"{prefix}_max_contour_circularity": circularity,
    }


def features_for_mask(rgb: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float | int]:
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

    result: dict[str, float | int] = {
        f"{prefix}_mask_present": int(mask.any()),
        f"{prefix}_area_ratio": float(mask.mean()),
    }
    result.update({f"{prefix}_{k}": v for k, v in connected_component_features(mask).items()})
    result.update(contour_shape_features(mask, prefix))

    if values_rgb.size == 0:
        empty_names = [
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
        ]
        for name in empty_names:
            result[f"{prefix}_{name}"] = 0.0
        result.update(normalized_histogram(np.array([]), bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
        result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
        result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
        result.update(lbp_features(gray, mask, prefix))
        result.update(glcm_features(gray, mask, prefix))
        return result

    result.update(
        {
            f"{prefix}_r_mean": float(values_rgb[:, 0].mean()),
            f"{prefix}_g_mean": float(values_rgb[:, 1].mean()),
            f"{prefix}_b_mean": float(values_rgb[:, 2].mean()),
            f"{prefix}_r_std": float(values_rgb[:, 0].std()),
            f"{prefix}_g_std": float(values_rgb[:, 1].std()),
            f"{prefix}_b_std": float(values_rgb[:, 2].std()),
            f"{prefix}_h_mean": float(values_hsv[:, 0].mean()),
            f"{prefix}_s_mean": float(values_hsv[:, 1].mean()),
            f"{prefix}_v_mean": float(values_hsv[:, 2].mean()),
            f"{prefix}_h_std": float(values_hsv[:, 0].std()),
            f"{prefix}_s_std": float(values_hsv[:, 1].std()),
            f"{prefix}_v_std": float(values_hsv[:, 2].std()),
            f"{prefix}_gray_mean": float(gray_values.mean()),
            f"{prefix}_gray_std": float(gray_values.std()),
            f"{prefix}_edge_density": float((edge_values > 0).mean()),
            f"{prefix}_gradient_mean": float(grad_values.mean()),
            f"{prefix}_gradient_std": float(grad_values.std()),
            f"{prefix}_entropy": entropy_from_values(gray_values),
        }
    )
    result.update(normalized_histogram(values_hsv[:, 0], bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
    result.update(normalized_histogram(values_hsv[:, 1], bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
    result.update(normalized_histogram(values_hsv[:, 2], bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
    result.update(lbp_features(gray, mask, prefix))
    result.update(glcm_features(gray, mask, prefix))
    return result


def _patch_from_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    patch = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    return cv2.resize(patch, (FFT_PATCH_SIZE, FFT_PATCH_SIZE), interpolation=cv2.INTER_AREA)


def fft_ring_features(gray: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    empty = {f"{prefix}_fft_band{i}_energy": 0.0 for i in range(FFT_N_BANDS)}
    empty.update({f"{prefix}_fft_band{i}_ratio": 0.0 for i in range(FFT_N_BANDS)})
    for key in ["dc_ratio", "high_ratio", "spectral_centroid", "spectral_spread"]:
        empty[f"{prefix}_fft_{key}"] = 0.0

    patch = _patch_from_mask(gray, mask)
    if patch is None:
        return empty

    magnitude = np.abs(fftshift(fft2(patch)))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yi, xi = np.ogrid[:h, :w]
    radius_map = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
    max_radius = min(cx, cy)

    band_energy = []
    for i in range(FFT_N_BANDS):
        ring = magnitude[
            (radius_map >= i * max_radius / FFT_N_BANDS)
            & (radius_map < (i + 1) * max_radius / FFT_N_BANDS)
        ]
        band_energy.append(float(ring.mean()) if ring.size else 0.0)

    total = sum(band_energy) + 1e-9
    result = {f"{prefix}_fft_band{i}_energy": energy for i, energy in enumerate(band_energy)}
    result.update({f"{prefix}_fft_band{i}_ratio": energy / total for i, energy in enumerate(band_energy)})
    result[f"{prefix}_fft_dc_ratio"] = (band_energy[0] + band_energy[1]) / total
    result[f"{prefix}_fft_high_ratio"] = sum(band_energy[FFT_N_BANDS // 2 :]) / total

    flat_radius = radius_map.flatten()
    flat_magnitude = magnitude.flatten()
    weight_sum = flat_magnitude.sum() + 1e-9
    centroid = float(np.dot(flat_radius, flat_magnitude) / weight_sum)
    result[f"{prefix}_fft_spectral_centroid"] = centroid
    result[f"{prefix}_fft_spectral_spread"] = float(
        np.sqrt(np.dot((flat_radius - centroid) ** 2, flat_magnitude) / weight_sum)
    )
    return result


def dog_response(gray_f: np.ndarray, sigma_fine: float, sigma_coarse: float) -> np.ndarray:
    return gaussian_filter(gray_f, sigma_fine) - gaussian_filter(gray_f, sigma_coarse)


def dog_wavelet_features(gray: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    result = {}
    gray_f = gray.astype(np.float32)
    energies = []

    for sigma_fine, sigma_coarse in DOG_SCALES:
        tag = f"s{int(sigma_fine * 10):03d}"
        response = dog_response(gray_f, sigma_fine, sigma_coarse)
        region = response[mask] if mask.any() else np.array([])
        if region.size == 0:
            result[f"{prefix}_dog_{tag}_mean"] = 0.0
            result[f"{prefix}_dog_{tag}_std"] = 0.0
            result[f"{prefix}_dog_{tag}_energy"] = 0.0
            result[f"{prefix}_dog_{tag}_peak_ratio"] = 0.0
            result[f"{prefix}_dog_{tag}_neg_ratio"] = 0.0
            energies.append(0.0)
            continue

        std = float(region.std()) + 1e-9
        energy = float(np.mean(region**2))
        result[f"{prefix}_dog_{tag}_mean"] = float(region.mean())
        result[f"{prefix}_dog_{tag}_std"] = std
        result[f"{prefix}_dog_{tag}_energy"] = energy
        result[f"{prefix}_dog_{tag}_peak_ratio"] = float((np.abs(region) > std).mean())
        result[f"{prefix}_dog_{tag}_neg_ratio"] = float((region < 0).mean())
        energies.append(energy)

    result[f"{prefix}_dog_fine_coarse_ratio"] = (energies[0] + 1e-9) / (energies[-1] + 1e-9)
    return result


def morph_profile_features(mask: np.ndarray, prefix: str) -> dict[str, float]:
    result = {}
    mask_u8 = mask.astype(np.uint8)
    original_area = float(mask.sum()) + 1e-9
    image_area = float(mask.size)

    for radius in MORPH_RADII:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        eroded = cv2.erode(mask_u8, kernel).astype(bool)
        dilated = cv2.dilate(mask_u8, kernel).astype(bool)
        shell = dilated & ~eroded

        erode_area = float(eroded.sum())
        dilate_area = float(dilated.sum())
        result[f"{prefix}_erode_r{radius}_ratio"] = erode_area / image_area
        result[f"{prefix}_dilate_r{radius}_ratio"] = dilate_area / image_area
        result[f"{prefix}_shell_r{radius}_ratio"] = float(shell.sum()) / image_area
        result[f"{prefix}_thickness_score_r{radius}"] = erode_area / original_area
        result[f"{prefix}_fill_ratio_r{radius}"] = erode_area / (dilate_area + 1e-9)

    gradient_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, gradient_kernel).astype(bool)
    result[f"{prefix}_morph_gradient_ratio"] = float(gradient.sum()) / image_area

    smooth_kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, smooth_kernel).astype(bool)
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, smooth_kernel).astype(bool)
    result[f"{prefix}_open_close_diff"] = float(np.abs(closed.astype(int) - opened.astype(int)).sum()) / image_area
    return result


def spatial_distribution_features(mask: np.ndarray, prefix: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    h, w = mask.shape
    image_area = float(h * w)
    mask_u8 = mask.astype(np.uint8)

    moments = cv2.moments(mask_u8)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, value in enumerate(hu_moments):
        result[f"{prefix}_hu_{i}"] = float(math.copysign(math.log(abs(value) + 1e-9), value))

    m00 = moments["m00"] + 1e-9
    centroid_x = moments["m10"] / m00
    centroid_y = moments["m01"] / m00
    result[f"{prefix}_centroid_x_norm"] = float(centroid_x / w)
    result[f"{prefix}_centroid_y_norm"] = float(centroid_y / h)

    ys, xs = np.where(mask)
    if ys.size == 0:
        for bin_idx in range(RADIAL_BINS):
            result[f"{prefix}_radial_bin{bin_idx}"] = 0.0
        for bin_idx in range(ANGULAR_BINS):
            result[f"{prefix}_angular_bin{bin_idx}"] = 0.0
        result[f"{prefix}_radial_max_bin"] = 0
        result[f"{prefix}_radial_spread"] = 0.0
        result[f"{prefix}_angular_entropy"] = 0.0
        result[f"{prefix}_bbox_aspect_ratio"] = 0.0
        result[f"{prefix}_bbox_fill_ratio"] = 0.0
        result[f"{prefix}_bbox_area_norm"] = 0.0
        return result

    dx = xs - centroid_x
    dy = ys - centroid_y
    radial_distance = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx)) % 360

    radial_bins = np.linspace(0, radial_distance.max() + 1e-9, RADIAL_BINS + 1)
    radial_hist, _ = np.histogram(radial_distance, bins=radial_bins)
    radial_total = radial_hist.sum() + 1e-9
    for bin_idx, value in enumerate(radial_hist):
        result[f"{prefix}_radial_bin{bin_idx}"] = float(value / radial_total)
    result[f"{prefix}_radial_max_bin"] = int(np.argmax(radial_hist))
    result[f"{prefix}_radial_spread"] = float(radial_distance.std())

    angular_bins = np.linspace(0, 360, ANGULAR_BINS + 1)
    angular_hist, _ = np.histogram(angle, bins=angular_bins)
    angular_probabilities = angular_hist / (angular_hist.sum() + 1e-9)
    for bin_idx, value in enumerate(angular_probabilities):
        result[f"{prefix}_angular_bin{bin_idx}"] = float(value)
    nonzero = angular_probabilities[angular_probabilities > 0]
    result[f"{prefix}_angular_entropy"] = float(-(nonzero * np.log2(nonzero)).sum())

    bbox_height = float(ys.max() - ys.min() + 1)
    bbox_width = float(xs.max() - xs.min() + 1)
    result[f"{prefix}_bbox_aspect_ratio"] = bbox_height / (bbox_width + 1e-9)
    result[f"{prefix}_bbox_fill_ratio"] = float(mask.sum()) / (bbox_height * bbox_width + 1e-9)
    result[f"{prefix}_bbox_area_norm"] = (bbox_height * bbox_width) / image_area
    return result
