from __future__ import annotations

from dataclasses import dataclass, fields
import json
import os
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecognitionOptions:
    backend: str = "face01"
    model_path: Path = Path(__file__).resolve().parent / "resources" / "JAPANESE_FACE_V1.onnx"
    acceleration: str = "auto"
    precision: str = "accurate"
    recognition_region: str = "full"
    upper_multiscale_detection: bool = True
    detection_max_dimension: int = 640
    threads: int = min(4, max(1, os.cpu_count() or 1))
    min_face_size: int = 80
    gpu_device_id: int = 0
    cache_dir: Path | None = None
    tensorrt_dll_dir: Path | None = None
    detection_model_path: Path = Path(__file__).resolve().parent / "resources" / "scrfd_10g_kps.onnx"
    detection_threshold: float = 0.6
    min_quality: float = 0.65
    blur_threshold: float = 12.0
    pose_threshold: float = 0.48
    sample_interval: float = 0.5
    face_sample_interval: float = 0.2
    adaface_enabled: bool = False
    adaface_model_path: Path | None = None
    japanese_face_weight: float = 1.0
    adaface_weight: float = 1.0
    japanese_face_threshold: float = 0.80
    adaface_threshold: float = 0.80
    high_threshold: float = 0.80
    medium_threshold: float = 0.65
    match_threshold: float = 0.65
    min_samples: int = 3
    max_samples: int = 30
    auto_merge_enabled: bool = False
    recursive: bool = True
    extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
    log_level: str = "INFO"
    database_dir: Path | None = None

    def analysis_fingerprint(self) -> str:
        """Settings identity; artifact hashes are stored separately with embeddings."""
        excluded = {"cache_dir", "database_dir", "log_level", "recursive", "extensions", "tensorrt_dll_dir"}
        values = {key: str(value) if isinstance(value, Path) else value for key, value in self.__dict__.items() if key not in excluded}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, config_path: Path | None = None) -> "RecognitionOptions":
        defaults = cls()
        if config_path is None:
            return defaults
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Recognition config file does not exist: {path}")
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read recognition config {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError("Recognition config must be a JSON object")
        unknown = sorted(set(raw) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown recognition config key(s): {', '.join(unknown)}")
        values = dict(raw)
        if "extensions" in values:
            if not isinstance(values["extensions"], list):
                raise ValueError("extensions must be an array")
            values["extensions"] = tuple(values["extensions"])
        for name in ("model_path", "cache_dir", "tensorrt_dll_dir", "detection_model_path", "adaface_model_path", "database_dir"):
            if name in values and values[name] is not None:
                if not isinstance(values[name], str) or not values[name].strip():
                    raise ValueError(f"{name} must be a nonempty path string")
                configured_path = Path(values[name]).expanduser()
                values[name] = configured_path if configured_path.is_absolute() else path.parent / configured_path
        options = cls(**{**defaults.__dict__, **values})
        options.validate()
        return options

    def validate(self) -> None:
        if self.backend != "face01":
            raise ValueError("backend must be 'face01'")
        if self.acceleration not in {"auto", "cpu", "cuda", "tensorrt"}:
            raise ValueError("acceleration must be 'auto', 'cpu', 'cuda', or 'tensorrt'")
        if self.precision not in {"accurate", "balanced"}:
            raise ValueError("precision must be 'accurate' or 'balanced'")
        if self.recognition_region not in {"full", "upper"}:
            raise ValueError("recognition_region must be 'full' or 'upper'")
        ranges = {"threads": (1, 8), "detection_max_dimension": (320, 4096), "min_face_size": (40, 512), "gpu_device_id": (0, 255), "min_samples": (2, 50), "max_samples": (2, 100)}
        for name, (minimum, maximum) in ranges.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        if not isinstance(self.model_path, Path):
            raise ValueError("model_path must be a path")
        if not isinstance(self.detection_model_path, Path):
            raise ValueError("detection_model_path must be a path")
        for name in ("cache_dir", "tensorrt_dll_dir", "adaface_model_path", "database_dir"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a path or null")
        import math
        for name in ("detection_threshold", "min_quality", "high_threshold", "medium_threshold", "match_threshold", "japanese_face_threshold", "adaface_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("blur_threshold", "sample_interval", "face_sample_interval", "pose_threshold", "japanese_face_weight", "adaface_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.medium_threshold >= self.high_threshold:
            raise ValueError("medium_threshold must be below high_threshold")
        if self.max_samples < self.min_samples:
            raise ValueError("max_samples must be at least min_samples")
        for name in ("adaface_enabled", "auto_merge_enabled", "recursive", "upper_multiscale_detection"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.recognition_region == "upper" and self.auto_merge_enabled:
            raise ValueError("Upper-face thresholds are not calibrated. Keep automatic merging off and review matches.")
        if self.adaface_enabled and self.adaface_model_path is None:
            raise ValueError("adaface_model_path is required when AdaFace is enabled")
        if not isinstance(self.extensions, tuple) or not self.extensions or any(not isinstance(item, str) or not item.startswith(".") or not item[1:].isalnum() for item in self.extensions):
            raise ValueError("extensions must contain file extensions such as .mp4")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")


def default_config_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "VideoAtlasPython"
    return root / "recognition.json"
