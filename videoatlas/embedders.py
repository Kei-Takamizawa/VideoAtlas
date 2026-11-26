"""Replaceable face-embedding adapters for still-photo analysis.

Each adapter states its model file, alignment template, colour order, and pixel
normalization. Its embedding key combines the model name, the artifact SHA-256,
the preprocessing identity, and the numerical profile, so embeddings from
different models, versions, preprocessing, or CPU/GPU profiles never share one
embedding space.

Licences: every recognition model listed here has weights restricted to
non-commercial or research use. VideoAtlas is a personal, non-commercial tool
(design specification, Section 2); redistribution would need a new review.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .model_adapters import OnnxRunner, align_crop, artifact_sha256

RESOURCES = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class EmbedderSpec:
    """How to prepare a face for one recognition model."""
    name: str
    filename: str
    input_size: int
    template: str            # "arcface112" (standard 5-point template) or "face01"
    channel_order: str       # "rgb" or "bgr"
    mean: tuple[float, float, float] | None  # pixel-space mean (0-255), None = detect InsightFace style
    std: tuple[float, float, float] | None
    source: str
    license: str
    sha256: str | None = None  # pinned when the official artifact hash is known
    flip: bool = False         # average the embeddings of the crop and its mirror image

    @property
    def preprocessing(self) -> str:
        normalization = "insightface-auto" if self.mean is None else "mean{}-std{}".format(
            "/".join(f"{value:g}" for value in self.mean), "/".join(f"{value:g}" for value in self.std))
        flip = "-flipavg" if self.flip else ""
        return f"{self.channel_order}{self.input_size}-{normalization}-scrfd-{self.template}-v1{flip}"


_IMAGENET_MEAN = (123.675, 116.28, 103.53)
_IMAGENET_STD = (58.395, 57.12, 57.375)
_HALF = (127.5, 127.5, 127.5)

EMBEDDERS: dict[str, EmbedderSpec] = {
    # JAPANESE FACE V1: RGB, ImageNet normalization, 224 px FACE01 chip approximation.
    "face01": EmbedderSpec("face01", "JAPANESE_FACE_V1.onnx", 224, "face01", "rgb", _IMAGENET_MEAN, _IMAGENET_STD,
                           "https://github.com/yKesamaru/FACE01_trained_models", "non-commercial",
                           "e7ca51f4bc85f73ddb830683ac6a09077909fa45a52b2bff41a9c6e8ff267e2f"),
    # InsightFace ArcFace models: RGB, (x - 127.5) / 127.5 unless the graph normalizes itself.
    "w600k-r50": EmbedderSpec("w600k-r50", "insightface/buffalo_l/w600k_r50.onnx", 112, "arcface112", "rgb", None, None,
                              "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                              "non-commercial research"),
    "glintr100": EmbedderSpec("glintr100", "insightface/antelopev2/glintr100.onnx", 112, "arcface112", "rgb", None, None,
                              "InsightFace antelopev2 model pack", "non-commercial research"),
    # LVFace (ByteDance, ICCV 2025): RGB, (x / 255 - 0.5) / 0.5 per its ONNX inference script.
    "lvface-b": EmbedderSpec("lvface-b", "lvface/LVFace-B_Glint360K.onnx", 112, "arcface112", "rgb", _HALF, _HALF,
                             "https://huggingface.co/bytedance-research/LVFace", "non-commercial"),
    # AdaFace (official PyTorch checkpoint exported by Scripts/export_adaface.py): BGR, (x / 255 - 0.5) / 0.5.
    "adaface-r50": EmbedderSpec("adaface-r50", "adaface_ir50_ms1mv2.onnx", 112, "arcface112", "bgr", _HALF, _HALF,
                                "https://github.com/mk-minchul/AdaFace", "non-commercial research"),
    "adaface-ir101": EmbedderSpec("adaface-ir101", "adaface_ir101_webface12m.onnx", 112, "arcface112", "bgr", _HALF, _HALF,
                                  "https://github.com/mk-minchul/AdaFace", "non-commercial research"),
}


def spec_with_flip(spec: EmbedderSpec, flip: bool) -> EmbedderSpec:
    return spec if spec.flip == flip else EmbedderSpec(**{**spec.__dict__, "flip": flip})


def resolve_spec(name: str) -> EmbedderSpec:
    """Look up a model by name; a "+flip" suffix enables mirror averaging."""
    base, _, suffix = name.partition("+")
    if base not in EMBEDDERS or suffix not in ("", "flip"):
        raise ValueError(f"Unknown embedding model: {name}. Known: {', '.join(sorted(EMBEDDERS))} (optionally with +flip)")
    return spec_with_flip(EMBEDDERS[base], suffix == "flip")


def insightface_normalization(path: Path) -> tuple[float, float]:
    """Replicate InsightFace's ArcFaceONNX rule: graphs that start with Sub and Mul
    nodes normalize internally (mean 0, std 1); otherwise use 127.5 / 127.5."""
    import onnx
    graph = onnx.load(str(path), load_external_data=False).graph
    names = [node.name for node in graph.node[:8]]
    has_sub = any(name.startswith(("Sub", "_minus")) for name in names)
    has_mul = any(name.startswith(("Mul", "_mul")) for name in names)
    return (0.0, 1.0) if has_sub and has_mul else (127.5, 127.5)


def face_tensor(crop_bgr: np.ndarray, spec: EmbedderSpec, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    size = spec.input_size
    if crop_bgr.shape != (size, size, 3) or crop_bgr.dtype != np.uint8:
        raise ValueError(f"{spec.name} crop must be BGR uint8 {size}x{size}")
    pixels = crop_bgr[:, :, ::-1] if spec.channel_order == "rgb" else crop_bgr
    normalized = (pixels.astype(np.float32) - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
        raise RuntimeError("Model produced invalid embeddings")
    return vectors / norms


class OnnxEmbedder:
    """One recognition model with model-specific alignment and normalization."""

    def __init__(self, spec: EmbedderSpec, options, model_dir: Path | None = None) -> None:
        self.spec = spec
        path = (Path(model_dir) if model_dir is not None else RESOURCES) / spec.filename
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model file for {spec.name} is missing: {path} (source: {spec.source})")
        self.path = path
        self.sha256 = artifact_sha256(path)
        if spec.sha256 is not None and self.sha256 != spec.sha256:
            raise ValueError(f"{spec.name} checksum mismatch: expected {spec.sha256}, found {self.sha256}")
        if spec.mean is None:
            mean, std = insightface_normalization(path)
            self.mean, self.std = np.full(3, mean, np.float32), np.full(3, std, np.float32)
        else:
            self.mean, self.std = np.asarray(spec.mean, np.float32), np.asarray(spec.std, np.float32)
        self.runner = OnnxRunner(path, options)
        shape = self.runner.input.shape
        if len(shape) != 4 or tuple(shape[1:]) != (3, spec.input_size, spec.input_size) or self.runner.input.type != "tensor(float)":
            raise ValueError(f"{spec.name} input must be float32 [N,3,{spec.input_size},{spec.input_size}], got {shape}")
        self.dynamic_batch = not isinstance(shape[0], int)
        providers = self.runner.session.get_providers()
        self.profile = ("cuda-fp32-notf32" if providers[0] == "CUDAExecutionProvider" else "cpu-fp32")
        normalization = f"mean{self.mean[0]:g}-std{self.std[0]:g}" if spec.mean is None else ""
        self.key = f"{spec.name}:{self.sha256[:16]}:{spec.preprocessing}{':' + normalization if normalization else ''}:{self.profile}"

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self.runner.session.get_providers())

    def align(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, float, float]:
        model = "face01" if self.spec.template == "face01" else "adaface"
        return align_crop(image_bgr, landmarks, model, "full")

    def embed(self, crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """L2-normalized embeddings, one row per crop."""
        if not crops:
            return np.zeros((0, 0), np.float32)
        tensors = [face_tensor(crop, self.spec, self.mean, self.std) for crop in crops]
        if self.spec.flip:
            tensors += [np.ascontiguousarray(tensor[:, :, ::-1]) for tensor in tensors]
        limit = batch_size if self.dynamic_batch else 1
        rows = []
        for offset in range(0, len(tensors), limit):
            batch = np.ascontiguousarray(np.stack(tensors[offset:offset + limit]), dtype=np.float32)
            output = np.asarray(self.runner.run(batch)[0], np.float32)
            rows.append(output.reshape(len(batch), -1))
        vectors = np.concatenate(rows)
        if self.spec.flip:
            half = len(crops)
            vectors = l2_normalize(vectors[:half]) + l2_normalize(vectors[half:])
        return l2_normalize(vectors)


def blur_score(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian on an aligned crop, so faces of every size share one scale."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
