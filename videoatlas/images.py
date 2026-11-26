"""Headless still-image analysis and single-image pair evaluation.

Folder names are used only after inference, as an offline answer key for
evaluation. They never enter detection, embedding, or grouping.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np

from .analyzer import FaceSample

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in np.arange(.30, .951, .05))


@dataclass
class ImageResult:
    path: Path
    label: str | None
    width: int = 0
    height: int = 0
    faces: list[FaceSample] = field(default_factory=list)
    error: str | None = None


def iter_image_files(root: Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    """Image files below root in a stable order, skipping hidden files and folders."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Image folder does not exist: {root}")
    allowed = {extension.lower() for extension in extensions}
    files = [path for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() in allowed
             and not any(part.startswith(".") for part in path.relative_to(root).parts)]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def folder_label(path: Path, root: Path) -> str | None:
    """First-level folder name below root, or None for files directly in root."""
    parts = Path(path).relative_to(root).parts
    return parts[0] if len(parts) > 1 else None


def analyze_images(analyzer, paths: list[Path], root: Path, stop: Callable[[], bool] | None = None,
                   progress: Callable[[int, int], None] | None = None) -> Iterator[ImageResult]:
    """Run the analyzer on each image. Stops early, without a partial result, when stop() is true."""
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if stop is not None and stop():
            return
        result = ImageResult(Path(path), folder_label(path, root))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            result.error = "unreadable_image"
        else:
            result.height, result.width = image.shape[:2]
            faces = analyzer.analyze_image(image, stop)
            if faces is None:
                return
            result.faces = faces
        if progress is not None:
            progress(index, total)
        yield result


def primary_face(result: ImageResult) -> FaceSample | None:
    """Largest accepted face, assuming a labelled photo shows its subject largest."""
    accepted = [face for face in result.faces if face.embedding is not None]
    if not accepted:
        return None
    return max(accepted, key=lambda face: face.bounding_box[2] * face.bounding_box[3])


def update_label_stats(stats: dict[str, dict], result: ImageResult) -> None:
    entry = stats.setdefault(result.label or "(unlabelled)", {
        "images": 0, "unreadable": 0, "images_with_detection": 0, "images_with_accepted_face": 0,
        "faces_detected": 0, "faces_accepted": 0, "rejections": Counter()})
    entry["images"] += 1
    entry["unreadable"] += result.error is not None
    entry["images_with_detection"] += bool(result.faces)
    entry["images_with_accepted_face"] += primary_face(result) is not None
    entry["faces_detected"] += len(result.faces)
    entry["faces_accepted"] += sum(face.embedding is not None for face in result.faces)
    entry["rejections"].update(face.rejection_reason for face in result.faces if face.rejection_reason)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _auc(positives: np.ndarray, negatives: np.ndarray) -> float | None:
    """Probability that a same-person pair outscores a different-person pair (ties count half)."""
    if not positives.size or not negatives.size:
        return None
    ordered = np.sort(negatives)
    below = np.searchsorted(ordered, positives, side="left").astype(np.float64)
    at_most = np.searchsorted(ordered, positives, side="right").astype(np.float64)
    return float((below + (at_most - below) / 2).sum() / (positives.size * negatives.size))


def pair_report(labels: list[str], vectors: np.ndarray, thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> dict:
    """Cosine scores for every pair of labelled images, one face per image.

    This is score-only evidence for single images. It is not the application's
    multi-sample HIGH policy and not a cluster-level merge outcome.
    """
    vectors = np.asarray(vectors, np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(labels):
        raise ValueError("vectors must have one row per label")
    report = {"unit": "single-image face pairs (largest accepted face per labelled image)",
              "image_count": len(labels), "label_count": len(set(labels))}
    if len(labels) < 2:
        return {**report, "pair_count": 0, "positive_count": 0, "negative_count": 0,
                "auc": None, "max_negative_score": None, "recall_with_zero_false_positives": None, "thresholds": []}
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(vectors).all():
        raise ValueError("vectors must be finite and nonzero")
    vectors = vectors / norms
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    upper = np.triu_indices(len(labels), k=1)
    scores = (vectors @ vectors.T)[upper]
    same = codes[upper[0]] == codes[upper[1]]
    positives, negatives = np.sort(scores[same]), np.sort(scores[~same])
    rows = []
    for threshold in thresholds:
        tp = int(positives.size - np.searchsorted(positives, threshold, side="left"))
        fp = int(negatives.size - np.searchsorted(negatives, threshold, side="left"))
        fn, tn = int(positives.size) - tp, int(negatives.size) - fp
        rows.append({"threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                     "false_positive_rate": _ratio(fp, fp + tn)})
    max_negative = float(negatives[-1]) if negatives.size else None
    clean = int(positives.size - np.searchsorted(positives, max_negative, side="right")) if max_negative is not None else int(positives.size)
    return {**report, "pair_count": int(scores.size), "positive_count": int(positives.size), "negative_count": int(negatives.size),
            "auc": _auc(positives, negatives), "max_negative_score": max_negative,
            "recall_with_zero_false_positives": _ratio(clean, int(positives.size)),
            "positive_score_quantiles": _quantiles(positives), "negative_score_quantiles": _quantiles(negatives),
            "thresholds": rows}


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if not values.size:
        return None
    points = (.01, .05, .25, .5, .75, .95, .99)
    return {f"p{int(point * 100):02d}": float(value) for point, value in zip(points, np.quantile(values, points))}
