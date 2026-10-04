"""Model-specific preprocessing and SCRFD inference, with explicit artifact identities.

Sources: deepinsight/insightface SCRFD model_zoo/scrfd.py;
mk-minchul/AdaFace inference.py and face_alignment/mtcnn_pytorch/src/align_trans.py;
yKesamaru/FACE01_DEV api.py at afec7ebac709f14224353e7f8b6539711899b1ff.
SCRFD provides eye centers, unlike dlib's eye corners. The FACE01 alignment below
approximates its chip geometry and has a new identity; it is not dlib-equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import threading
import os
import sys
import importlib.metadata

import cv2
import numpy as np

LOG = logging.getLogger(__name__)
_NVIDIA_DLL_HANDLES: dict[str, object] = {}
_NVIDIA_DLL_LOCK = threading.RLock()
FACE01_PREPROCESSING = "rgb224-imagenet-scrfd-similarity5-dlib-hybrid-geometry-v2"
ADAFACE_PREPROCESSING = "bgr112-mean0.5-std0.5-scrfd-similarity5-v1"
UPPER_REGION_PREPROCESSING = "upper-eyes-only-no-lower-face-v1"
SCRFD_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
ADAFACE_TEMPLATE = np.asarray([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
# dlib v19.24 interpolation.h: average its five-point eye-corner references,
# retain its nose reference, and use mouth corners 48/54 from its 68-point mean.
# This hybrid SCRFD template is an approximation, not dlib-equivalent alignment.
# https://github.com/davisking/dlib/blob/v19.24/dlib/image_transforms/interpolation.h
FACE01_TEMPLATE = (np.asarray([[.22733006172505, .2213958465045],
    [.7528139680048, .22123279628455], [.4901123135679, .6277975316475],
    [.254149, .780233], [.726104, .780233]], dtype=np.float32) + .1) / 1.2 * 224


def configure_windows_nvidia_libraries() -> None:
    """Expose pip-provided CUDA/cuDNN bins to Windows lazy sub-library loading.

    ORT's preload API loads core DLLs, while cuDNN can load additional engines
    later through the process PATH. Only installed distribution locations are
    added; no system configuration or guessed external directory is changed.
    https://onnxruntime.ai/docs/install/#cuda-and-cudnn
    """
    if sys.platform != "win32":
        return
    packages = {"nvidia-cuda-runtime-cu12":"cuda_runtime", "nvidia-cuda-nvrtc-cu12":"cuda_nvrtc",
                "nvidia-cudnn-cu12":"cudnn", "nvidia-cublas-cu12":"cublas", "nvidia-cufft-cu12":"cufft",
                "nvidia-curand-cu12":"curand", "nvidia-nvjitlink-cu12":"nvjitlink"}
    with _NVIDIA_DLL_LOCK:
        directories = []
        for distribution,module in packages.items():
            try:
                directory = Path(importlib.metadata.distribution(distribution).locate_file(f"nvidia/{module}/bin")).resolve()
                if not directory.is_dir():
                    continue
                value = str(directory)
                directories.append(value)
                if value not in _NVIDIA_DLL_HANDLES:
                    _NVIDIA_DLL_HANDLES[value] = os.add_dll_directory(value)
            except (importlib.metadata.PackageNotFoundError,OSError):
                continue
        present = {item.casefold() for item in os.environ.get("PATH", "").split(os.pathsep)}
        additions = [item for item in directories if item.casefold() not in present]
        if additions:
            os.environ["PATH"] = os.pathsep.join(additions+[os.environ.get("PATH", "")])


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def similarity_transform(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Fit a nonreflecting similarity transform using every provided landmark."""
    source, destination = np.asarray(source, np.float64), np.asarray(destination, np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("Invalid alignment landmarks")
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("Degenerate alignment landmarks")
    x, y = source.T
    system = np.zeros((len(source) * 2, 4))
    system[0::2] = np.column_stack((x, -y, np.ones(len(x)), np.zeros(len(x))))
    system[1::2] = np.column_stack((y, x, np.zeros(len(x)), np.ones(len(x))))
    a, b, tx, ty = np.linalg.lstsq(system, destination.reshape(-1), rcond=None)[0]
    if a*a + b*b <= 1e-12:
        raise ValueError("Degenerate alignment transform")
    return np.asarray([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def preprocessing_identity(model: str, region: str = "full") -> str:
    base = {"face01": FACE01_PREPROCESSING, "adaface": ADAFACE_PREPROCESSING}[model]
    if region not in {"full", "upper"}:
        raise ValueError("Unknown recognition region")
    return base if region == "full" else f"{base}:{UPPER_REGION_PREPROCESSING}"


def upper_visible_mask(shape: tuple[int, ...], eyes: np.ndarray) -> np.ndarray:
    """Keep eyes/forehead; discard pixels below the eye line and nasal root.

    The cutoff is one tenth of the eye gap below the eye centers. A central
    strip below the eye line also removes the nasal root. This deliberately
    removes lower ears and lower hair rather than retaining jaw information.
    """
    eyes = np.asarray(eyes, np.float64)[:2]
    if eyes.shape != (2, 2) or not np.isfinite(eyes).all():
        raise ValueError("Two finite eye landmarks are required")
    axis = eyes[1] - eyes[0]
    gap = float(np.linalg.norm(axis))
    if gap < 1:
        raise ValueError("Degenerate eye landmarks")
    axis /= gap
    down = np.asarray([-axis[1], axis[0]])
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    mid = eyes.mean(axis=0)
    vertical = (xx-mid[0])*down[0] + (yy-mid[1])*down[1]
    horizontal = (xx-mid[0])*axis[0] + (yy-mid[1])*axis[1]
    return (vertical < .10*gap) & ~((vertical >= 0) & (np.abs(horizontal) < .18*gap))


def align_crop(frame: np.ndarray, landmarks: np.ndarray, model: str, region: str = "full") -> tuple[np.ndarray, float, float]:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks[:2] if region == "upper" else landmarks).all():
        raise ValueError("Finite SCRFD landmarks are required")
    if model == "face01":
        destination, source, size = FACE01_TEMPLATE, landmarks, 224
    elif model == "adaface":
        destination, source, size = ADAFACE_TEMPLATE, landmarks, 112
    else:
        raise ValueError(f"Unknown alignment model: {model}")
    if region == "upper":
        # A synthetic third point is derived exclusively from the two eyes.
        # Nose/mouth coordinates do not participate in this transform.
        def eye_triangle(points):
            delta = points[1]-points[0]
            return np.asarray([points[0], points[1], points.mean(axis=0)+[-delta[1],delta[0]]], np.float32)
        source, destination = eye_triangle(source[:2]), eye_triangle(destination[:2])
        neutral = np.asarray([104,116,124] if model == "face01" else [128,128,128], np.uint8)
        visible = upper_visible_mask(frame.shape, landmarks[:2])
        frame = np.where(visible[...,None], frame, neutral).astype(np.uint8)
    elif region != "full":
        raise ValueError("Unknown recognition region")
    matrix = similarity_transform(source, destination)
    residual = float(np.sqrt(np.mean(np.sum((cv2.transform(source[None], matrix)[0] - destination)**2, axis=1)))) / size
    crop = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    support = cv2.warpAffine(np.ones(frame.shape[:2], np.uint8), matrix, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    if region == "upper":
        visible = upper_visible_mask(crop.shape, FACE01_TEMPLATE[:2] if model == "face01" else ADAFACE_TEMPLATE[:2])
        crop[~visible] = neutral
        return crop, residual, float(support[visible].mean())
    return crop, residual, float(support.mean())


def face01_tensor(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.shape != (224, 224, 3):
        raise ValueError("FACE01 crop must be BGR uint8 224x224")
    rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    normalized = (rgb - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def landmark_visibility_proxy(frame: np.ndarray, landmarks: np.ndarray) -> float:
    """Uncalibrated visibility proxy from five local facial landmark patches.

    Missing, saturated, or nearly uniform patches lower the score. This can
    detect some opaque covers, but textured masks/glasses can pass and ordinary
    low-texture skin can score poorly. It is not an occlusion classifier.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape not in {(5, 2), (2, 2)} or not np.isfinite(points).all():
        return 0.0
    radius = max(2, int(round(float(np.linalg.norm(points[1]-points[0])) * .12)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scores = []
    for x, y in points:
        x, y = int(round(float(x))), int(round(float(y)))
        left, right = max(0, x-radius), min(width, x+radius+1)
        top, bottom = max(0, y-radius), min(height, y+radius+1)
        if left >= right or top >= bottom:
            scores.append(0.0)
            continue
        patch = gray[top:bottom, left:right]
        coverage = patch.size / (2*radius+1)**2
        unsaturated = float(np.mean((patch > 8) & (patch < 247)))
        texture = min(1.0, max(0.0, (float(patch.std())-4)/12))
        scores.append(coverage * unsaturated * texture)
    return float(np.clip(np.mean(scores), 0, 1))


def adaface_tensor(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.shape != (112, 112, 3):
        raise ValueError("AdaFace crop must be BGR uint8 112x112")
    return np.ascontiguousarray(((crop_bgr.astype(np.float32) / 255.0 - .5) / .5).transpose(2, 0, 1))


class OnnxRunner:
    """Explicit CPU retry after GPU initialization or execution failure."""
    def __init__(self, path: Path, options) -> None:
        import onnxruntime as ort
        self.ort, self.path, self.options = ort, path, options
        if not path.is_file():
            raise FileNotFoundError(f"Model is missing: {path}")
        config = ort.SessionOptions()
        config.intra_op_num_threads, config.inter_op_num_threads = options.threads, 1
        self.config = config
        self.lock = threading.RLock()
        providers = ["CPUExecutionProvider"]
        if options.acceleration != "cpu":
            configure_windows_nvidia_libraries()
            if hasattr(ort,"preload_dlls"):
                ort.preload_dlls()
        if options.acceleration != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": options.gpu_device_id, "use_tf32": "0"}))
        try:
            self.session = ort.InferenceSession(str(path), sess_options=config, providers=providers)
        except Exception:
            LOG.exception("GPU model initialization failed; retrying on CPU")
            self.session = self.cpu_session()
        self.session.disable_fallback()
        self.input = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()

    def cpu_session(self):
        session = self.ort.InferenceSession(str(self.path), sess_options=self.config, providers=["CPUExecutionProvider"])
        session.disable_fallback()
        return session

    def run(self, tensor: np.ndarray):
        with self.lock:
            return self._run(tensor)

    def _run(self, tensor: np.ndarray):
        try:
            return self.session.run([item.name for item in self.outputs], {self.input.name: tensor})
        except Exception:
            if self.session.get_providers() == ["CPUExecutionProvider"]:
                raise
            LOG.exception("GPU model execution failed; retrying on CPU")
            self.session = self.cpu_session()
            return self.session.run([item.name for item in self.outputs], {self.input.name: tensor})


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    score: float
    landmarks: np.ndarray


class SCRFDDetector:
    def __init__(self, options) -> None:
        self.options = options
        path = options.detection_model_path.expanduser().resolve()
        self.sha256 = artifact_sha256(path)
        default_path = Path(__file__).parent / "resources" / "scrfd_10g_kps.onnx"
        if path == default_path.resolve() and self.sha256 != SCRFD_SHA256:
            raise ValueError("Pinned SCRFD model checksum mismatch")
        self.multiscale = options.recognition_region == "upper" and options.upper_multiscale_detection
        self.inference_sha256 = self.sha256
        if self.multiscale:
            path = self._dynamic_output_model(path, options)
            self.inference_sha256 = artifact_sha256(path)
        self.runner = OnnxRunner(path, options)
        shape = self.runner.input.shape
        if len(shape) != 4 or shape[1] != 3 or self.runner.input.type != "tensor(float)" or isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError("SCRFD requires float32 [1,3,H,W] input")
        count = len(self.runner.outputs)
        if count not in (9, 15):
            raise ValueError("SCRFD model must include five landmark heads (9 or 15 outputs)")
        self.strides = (8, 16, 32) if count == 9 else (8, 16, 32, 64, 128)
        self.anchors = 2 if count == 9 else 1
        self.size = (shape[3], shape[2]) if isinstance(shape[2], int) and isinstance(shape[3], int) else (options.detection_max_dimension, options.detection_max_dimension)
        self.fixed_input = all(isinstance(value,int) for value in shape[2:])
        if all(isinstance(value,int) for value in shape[2:]):
            if any(value < 128 or value % max(self.strides) for value in self.size):
                raise ValueError("SCRFD fixed input dimensions must be multiples of the largest stride")
        else:
            alignment = max(self.strides)
            self.size = tuple(max(128,int(value)//alignment*alignment) for value in self.size)

    @staticmethod
    def _dynamic_output_model(path: Path, options) -> Path:
        """Derive metadata only; retain source weights/nodes and record both hashes.

        The pinned SCRFD input is dynamic but output proposal counts are marked
        for 640 pixels. Make those counts symbolic before using other sizes.
        """
        import json
        import onnx
        from .recognition import default_config_path
        source_hash = artifact_sha256(path)
        root = (options.cache_dir or default_config_path().parent / "InferenceCache") / "DetectorModels"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"{source_hash}-dynamic-proposals-v1.onnx"
        manifest = destination.with_suffix(".json")
        if destination.is_file() and manifest.is_file():
            try:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                if metadata.get("source_sha256") == source_hash and metadata.get("derived_sha256") == artifact_sha256(destination):
                    return destination
            except (ValueError, OSError):
                pass
        model = onnx.load(str(path))
        for index, output in enumerate(model.graph.output):
            dimensions = output.type.tensor_type.shape.dim
            if len(dimensions) != 2:
                raise ValueError("SCRFD dynamic proposal export requires rank-two outputs")
            dimensions[0].ClearField("dim_value")
            dimensions[0].dim_param = f"proposals_{index}"
        onnx.checker.check_model(model)
        temporary = destination.with_suffix(f".{threading.get_ident()}.tmp")
        onnx.save(model, str(temporary))
        temporary.replace(destination)
        manifest.write_text(json.dumps({"source_sha256":source_hash,"derived_sha256":artifact_sha256(destination),
            "operation":"output proposal-count metadata only; weights and graph nodes unchanged","version":"dynamic-proposals-v1"}),encoding="utf-8")
        return destination

    def detect(self, frame: np.ndarray) -> list[Detection]:
        original_size = self.size
        if not getattr(self, "multiscale", False) or getattr(self, "fixed_input", False):
            return self._detect_once(frame)
        alignment = max(self.strides)
        sizes = [original_size]
        sizes += [tuple(max(128, min(4096, int(value*factor)//alignment*alignment)) for value in original_size)
                  for factor in (.5,1.5)]
        candidates = []
        try:
            for size in dict.fromkeys(sizes):
                self.size = size
                candidates.extend(self._detect_once(frame))
        finally:
            self.size = original_size
        selected = []
        for detection in sorted(candidates, key=lambda item: item.score, reverse=True):
            a = np.asarray(detection.box, np.float64)
            duplicate = False
            for existing in selected:
                b = np.asarray(existing.box, np.float64)
                intersection = float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
                union = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
                if intersection/max(union,1e-12) > .4:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(detection)
        return selected

    def _detect_once(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        tw, th = self.size
        scale = min(tw / width, th / height)
        nw, nh = max(1, int(width*scale)), max(1, int(height*scale))
        image = np.zeros((th, tw, 3), np.uint8)
        image[:nh, :nw] = cv2.resize(frame, (nw, nh))
        tensor = cv2.dnn.blobFromImage(image, 1/128.0, (tw, th), (127.5,127.5,127.5), swapRB=True)
        outputs = self.runner.run(tensor)
        return self.decode(outputs, scale, width, height)

    def decode(self, outputs, scale: float, width: int, height: int) -> list[Detection]:
        candidates = []
        count = len(self.strides)
        for index, stride in enumerate(self.strides):
            grid_h, grid_w = self.size[1] // stride, self.size[0] // stride
            centers = np.stack(np.mgrid[:grid_h, :grid_w][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centers = np.repeat(centers, self.anchors, axis=0)
            scores = np.asarray(outputs[index]).reshape(-1)
            boxes = np.asarray(outputs[index+count]).reshape(-1, 4)*stride
            points = np.asarray(outputs[index+2*count]).reshape(-1,5,2)*stride
            if len(scores) != len(centers) or len(boxes) != len(scores) or len(points) != len(scores) or not all(np.isfinite(item).all() for item in (scores, boxes, points)):
                raise ValueError("Invalid SCRFD output geometry")
            for row in np.flatnonzero(scores >= self.options.detection_threshold):
                if not 0 <= scores[row] <= 1 or np.any(boxes[row] < 0):
                    continue
                x, y = centers[row]
                l,t,r,b = boxes[row]
                box = np.asarray([x-l,y-t,x+r,y+b]) / scale
                box[[0,2]] = np.clip(box[[0,2]], 0, width)
                box[[1,3]] = np.clip(box[[1,3]], 0, height)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                candidates.append((box, float(scores[row]), (points[row]+centers[row])/scale))
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = []
        while candidates:
            box, score, landmarks = candidates.pop(0)
            selected.append(Detection(tuple(int(v) for v in box), score, landmarks))
            survivors = []
            for other in candidates:
                overlap = np.maximum(0, np.minimum(box[2:], other[0][2:]) - np.maximum(box[:2], other[0][:2]))
                intersection = float(np.prod(overlap))
                union = float(np.prod(box[2:]-box[:2]) + np.prod(other[0][2:]-other[0][:2]) - intersection)
                if intersection / max(union, 1e-12) <= .4:
                    survivors.append(other)
            candidates = survivors
        return selected


class AdaFaceAdapter:
    def __init__(self, path: Path, options) -> None:
        self.runner = OnnxRunner(path.expanduser().resolve(), options)
        shape = self.runner.input.shape
        if len(shape) != 4 or tuple(shape[1:]) != (3,112,112) or self.runner.input.type != "tensor(float)":
            raise ValueError("AdaFace must have float32 [N,3,112,112] input")
        if len(self.runner.outputs) != 1 or tuple(self.runner.outputs[0].shape[1:]) != (512,):
            raise ValueError("AdaFace export must have one 512-dimensional embedding output")
        # Export manifest is mandatory: model shape alone cannot establish model identity.
        import json
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        digest = artifact_sha256(path)
        if manifest.get("model_name") != "AdaFace" or manifest.get("onnx_sha256") != digest or manifest.get("preprocessing") != ADAFACE_PREPROCESSING or not manifest.get("checkpoint_sha256"):
            raise ValueError("AdaFace export manifest is missing or does not match the artifact")
        self.digest = digest
        self.region = getattr(options, "recognition_region", "full")

    @property
    def model_key(self) -> str:
        providers = ','.join(self.runner.session.get_providers())
        return f"adaface:{self.digest}:{preprocessing_identity('adaface', self.region)}:ort{self.runner.ort.__version__}:{providers}-fp32"

    def embed(self, crop: np.ndarray) -> list[float]:
        vector = np.asarray(self.runner.run(adaface_tensor(crop)[None])[0], np.float32)
        if vector.shape != (1,512) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
            raise ValueError("AdaFace returned an invalid embedding")
        return (vector[0] / np.linalg.norm(vector[0])).tolist()
