from __future__ import annotations

from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecognitionOptions:
    backend: str = "face01"
    model_path: Path = Path(__file__).resolve().parent / "resources" / "JAPANESE_FACE_V1.onnx"
    acceleration: str = "auto"
    precision: str = "accurate"
    detection_max_dimension: int = 1280
    threads: int = min(4, max(1, os.cpu_count() or 1))
    min_face_size: int = 80
    gpu_device_id: int = 0
    cache_dir: Path | None = None
    tensorrt_dll_dir: Path | None = None

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
        for name in ("model_path", "cache_dir", "tensorrt_dll_dir"):
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
        ranges = {"threads": (1, 8), "detection_max_dimension": (320, 4096), "min_face_size": (40, 512), "gpu_device_id": (0, 255)}
        for name, (minimum, maximum) in ranges.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        if not isinstance(self.model_path, Path):
            raise ValueError("model_path must be a path")
        for name in ("cache_dir", "tensorrt_dll_dir"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a path or null")


def default_config_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "VideoAtlasPython"
    return root / "recognition.json"
