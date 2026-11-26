"""Still-photo face analysis with several recognition models.

A run folder holds the results of one analysis:

- ``faces.jsonl``: one line per photo, with every detected face, its geometry,
  and size measurements. Face IDs are unique within the run.
- ``embeddings/<model>.npz``: L2-normalized embeddings, one row per usable face,
  stored separately for each model so embedding spaces never mix.
- ``report.json``: settings, model identities and providers, counts, timing.

First-level folder names are an offline answer key used only after inference.
They never enter detection, embedding, or grouping.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable

import cv2
import numpy as np

from .embedders import OnnxEmbedder, blur_score
from .images import folder_label, iter_image_files

RUN_FILES = ("faces.jsonl", "report.json")
EMBEDDING_DIR = "embeddings"
SIZE_BINS = (0, 10, 15, 20, 25, 30, 40, 60, 80, 120, 1_000_000)
# Below this eye distance the five landmarks are too close together to align a face reliably.
HARD_MIN_EYE_DISTANCE = 6.0


@dataclass
class PhotoFace:
    id: str
    index: int
    box: tuple[int, int, int, int]
    landmarks: list[list[float]]
    detection_score: float
    eye_distance: float
    yaw_ratio: float | None
    roll: float | None
    usable: bool
    rejection_reason: str | None = None
    blur: float | None = None
    brightness: float | None = None

    def record(self, width: int, height: int) -> dict:
        x0, y0, x1, y1 = self.box
        return {"id": self.id, "index": self.index,
                "box": [round(x0 / width, 6), round(y0 / height, 6), round((x1 - x0) / width, 6), round((y1 - y0) / height, 6)],
                "box_px": list(self.box), "landmarks": [[round(v, 2) for v in point] for point in self.landmarks],
                "detection_score": round(self.detection_score, 5), "eye_distance": round(self.eye_distance, 2),
                "face_size": int(min(x1 - x0, y1 - y0)), "yaw_ratio": None if self.yaw_ratio is None else round(self.yaw_ratio, 4),
                "roll": None if self.roll is None else round(self.roll, 2), "blur": None if self.blur is None else round(self.blur, 2),
                "brightness": None if self.brightness is None else round(self.brightness, 2),
                "usable": self.usable, "rejection_reason": self.rejection_reason}


@dataclass
class PhotoResult:
    path: Path
    relative: str
    label: str | None
    sha256: str
    width: int = 0
    height: int = 0
    error: str | None = None
    faces: list[PhotoFace] = field(default_factory=list)

    def record(self) -> dict:
        usable = [face for face in self.faces if face.usable]
        primary = max(usable, key=lambda face: face.eye_distance).index if usable else None
        return {"path": self.relative, "label": self.label, "sha256": self.sha256, "width": self.width, "height": self.height,
                "error": self.error, "faces": [face.record(max(self.width, 1), max(self.height, 1)) for face in self.faces],
                "primary_face": primary}


def face_geometry(landmarks: np.ndarray) -> tuple[float, float | None, float | None, str | None]:
    """Eye distance, a yaw proxy (nose offset along the eye axis / eye distance), roll, and a rejection reason."""
    points = np.asarray(landmarks, np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return 0.0, None, None, "landmarks_invalid"
    eyes = points[1] - points[0]
    gap = float(np.linalg.norm(eyes))
    if gap < 1e-6:
        return 0.0, None, None, "landmarks_invalid"
    axis = eyes / gap
    middle = (points[0] + points[1]) / 2
    yaw = float(np.dot(points[2] - middle, axis)) / gap
    roll = float(np.degrees(np.arctan2(eyes[1], eyes[0])))
    down = np.asarray([-axis[1], axis[0]])
    if points[1, 0] <= points[0, 0] or np.dot(points[2] - middle, down) <= 0:
        return gap, yaw, roll, "landmark_geometry_invalid"
    if gap < HARD_MIN_EYE_DISTANCE:
        return gap, yaw, roll, "face_too_small"
    return gap, yaw, roll, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PhotoAnalyzer:
    """SCRFD detection plus one or more recognition models."""

    def __init__(self, detector, embedders: list[OnnxEmbedder]) -> None:
        if not embedders:
            raise ValueError("At least one embedding model is required")
        names = [embedder.spec.name + ("+flip" if embedder.spec.flip else "") for embedder in embedders]
        if len(set(names)) != len(names):
            raise ValueError("Each embedding model may appear only once")
        self.detector, self.embedders, self.names = detector, embedders, names

    def providers(self) -> dict[str, tuple[str, ...]]:
        result = {"scrfd": tuple(self.detector.runner.session.get_providers())}
        result.update({name: embedder.providers for name, embedder in zip(self.names, self.embedders)})
        return result

    def analyze(self, image: np.ndarray, photo_id: str) -> tuple[list[PhotoFace], dict[str, np.ndarray]]:
        """Detect every face, then embed each usable face with every model."""
        faces, crops = [], {name: [] for name in self.names}
        for index, detection in enumerate(self.detector.detect(image)):
            x0, y0, x1, y1 = detection.box
            gap, yaw, roll, reason = face_geometry(detection.landmarks)
            face = PhotoFace(f"{photo_id}f{index}", index, (int(x0), int(y0), int(x1), int(y1)),
                             np.asarray(detection.landmarks, np.float64).tolist(), float(detection.score), gap, yaw, roll, reason is None, reason)
            if face.usable:
                try:
                    aligned = [embedder.align(image, detection.landmarks) for embedder in self.embedders]
                except ValueError:
                    face.usable, face.rejection_reason = False, "alignment_failed"
                else:
                    if min(support for _, _, support in aligned) < .6:
                        face.usable, face.rejection_reason = False, "alignment_outside_frame"
                    else:
                        reference = next((crop for (crop, _, _), embedder in zip(aligned, self.embedders) if embedder.spec.input_size == 112), aligned[0][0])
                        face.blur = blur_score(reference)
                        face.brightness = float(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).mean())
                        for name, (crop, _, _) in zip(self.names, aligned):
                            crops[name].append(crop)
            faces.append(face)
        vectors = {name: embedder.embed(crops[name]) for name, embedder in zip(self.names, self.embedders)}
        return faces, vectors


def analyze_folder(analyzer: PhotoAnalyzer, root: Path, output: Path, paths: Iterable[Path] | None = None,
                   progress: Callable[[int, int], None] | None = None, extra_report: dict | None = None) -> dict:
    """Analyze every photo below root and write a run folder. Returns the report."""
    root, output = Path(root).resolve(), Path(output).resolve()
    existing = [name for name in RUN_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Output already exists in {output}: {', '.join(existing)}. Choose a new output folder.")
    paths = list(paths) if paths is not None else iter_image_files(root)
    if not paths:
        raise ValueError(f"No images found below {root}")
    (output / EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ids = {name: [] for name in analyzer.names}
    rows = {name: [] for name in analyzer.names}
    stats: dict[str, Counter] = {}
    size_histogram = Counter()
    with (output / "faces.jsonl").open("w", encoding="utf-8") as stream:
        for number, path in enumerate(paths, 1):
            relative = Path(path).resolve().relative_to(root).as_posix()
            result = PhotoResult(Path(path), relative, folder_label(Path(path).resolve(), root), sha256_file(path))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                result.error = "unreadable_image"
            else:
                result.height, result.width = image.shape[:2]
                faces, vectors = analyzer.analyze(image, f"p{number:06d}")
                result.faces = faces
                usable_ids = [face.id for face in faces if face.usable]
                for name in analyzer.names:
                    if len(usable_ids) != len(vectors[name]) and usable_ids:
                        raise RuntimeError("Embedding count does not match usable faces")
                    ids[name].extend(usable_ids)
                    if usable_ids:
                        rows[name].append(vectors[name])
            entry = stats.setdefault(result.label or "(unlabelled)", Counter())
            entry["images"] += 1
            entry["unreadable"] += result.error is not None
            entry["faces_detected"] += len(result.faces)
            entry["faces_usable"] += sum(face.usable for face in result.faces)
            entry["images_with_usable_face"] += any(face.usable for face in result.faces)
            entry["images_with_one_usable_face"] += sum(face.usable for face in result.faces) == 1
            for face in result.faces:
                if face.rejection_reason:
                    entry[f"rejected:{face.rejection_reason}"] += 1
                if face.usable:
                    size_histogram[_bin_label(face.eye_distance)] += 1
            stream.write(json.dumps(result.record(), ensure_ascii=False) + "\n")
            if progress is not None:
                progress(number, len(paths))
    for name, embedder in zip(analyzer.names, analyzer.embedders):
        matrix = np.concatenate(rows[name]) if rows[name] else np.zeros((0, 0), np.float32)
        np.savez_compressed(output / EMBEDDING_DIR / f"{name}.npz", face_ids=np.asarray(ids[name]), vectors=matrix.astype(np.float32),
                            key=np.asarray(embedder.key), sha256=np.asarray(embedder.sha256))
    elapsed = time.monotonic() - started
    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "format": "videoatlas-photo-run-v1",
              "root": str(root), "image_count": len(paths), "elapsed_seconds": round(elapsed, 2),
              "images_per_second": round(len(paths) / elapsed, 2) if elapsed > 0 else None,
              "providers": {name: list(values) for name, values in analyzer.providers().items()},
              "models": {name: {"key": embedder.key, "sha256": embedder.sha256, "file": str(embedder.path), "source": embedder.spec.source,
                                "license": embedder.spec.license, "preprocessing": embedder.spec.preprocessing}
                         for name, embedder in zip(analyzer.names, analyzer.embedders)},
              "usable_faces_by_eye_distance_px": {label: size_histogram[label] for label in _bin_labels()},
              "labels": {label: dict(sorted(counter.items())) for label, counter in sorted(stats.items())},
              "notes": ["Folder names are an offline answer key used only after inference.",
                        "Usable means the face could be aligned and embedded; grouping policies apply their own size and quality gates."]}
    report.update(extra_report or {})
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _bin_labels() -> list[str]:
    return [f"{low}-{high - 1}" if high < 1_000_000 else f"{low}+" for low, high in zip(SIZE_BINS, SIZE_BINS[1:])]


def _bin_label(value: float) -> str:
    for label, low, high in zip(_bin_labels(), SIZE_BINS, SIZE_BINS[1:]):
        if low <= value < high:
            return label
    return _bin_labels()[-1]


@dataclass
class RunData:
    """A run folder loaded for grouping or evaluation."""
    photos: list[dict]
    faces: list[dict]                 # usable faces, in a fixed order
    face_photo: np.ndarray            # photo index per usable face
    embeddings: dict[str, np.ndarray]  # model name -> [faces, dims], rows aligned with ``faces``
    keys: dict[str, str]
    report: dict

    @property
    def face_ids(self) -> list[str]:
        return [face["id"] for face in self.faces]


def load_run(folder: Path, models: Iterable[str] | None = None) -> RunData:
    folder = Path(folder)
    photos = [json.loads(line) for line in (folder / "faces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    available = sorted(path.stem for path in (folder / EMBEDDING_DIR).glob("*.npz"))
    names = list(models) if models is not None else available
    missing = [name for name in names if name not in available]
    if missing:
        raise FileNotFoundError(f"Run has no embeddings for: {', '.join(missing)} (available: {', '.join(available)})")
    faces, face_photo = [], []
    for index, photo in enumerate(photos):
        for face in photo["faces"]:
            if face.get("usable"):
                faces.append({**face, "photo": photo["path"]})
                face_photo.append(index)
    order = {face["id"]: row for row, face in enumerate(faces)}
    embeddings, keys = {}, {}
    for name in names:
        with np.load(folder / EMBEDDING_DIR / f"{name}.npz", allow_pickle=False) as data:
            ids, vectors = [str(value) for value in data["face_ids"]], np.asarray(data["vectors"], np.float32)
            keys[name] = str(data["key"])
        if set(ids) != set(order) or len(ids) != len(order):
            raise ValueError(f"Embeddings for {name} do not match the usable faces in faces.jsonl")
        matrix = np.zeros((len(faces), vectors.shape[1] if vectors.ndim == 2 else 0), np.float32)
        matrix[[order[identifier] for identifier in ids]] = vectors
        embeddings[name] = matrix
    return RunData(photos, faces, np.asarray(face_photo, np.int64), embeddings, keys, report)
