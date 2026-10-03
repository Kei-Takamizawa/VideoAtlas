"""Runtime configuration for local face recognition."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
import sys
from typing import Any


def _default_backend() -> str:
    if sys.platform == "darwin":
        return "openvino"
    return "face01"


@dataclass(frozen=True)
class RecognitionOptions:
    """Validated runtime settings; values can be overridden in a JSON file."""

    backend: str = field(default_factory=_default_backend)
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
        """Load optional JSON settings over platform defaults without network access."""
        defaults = cls()
        path = Path(config_path).expanduser() if config_path is not None else None
        if path is None or not path.is_file():
            if path is not None:
                raise FileNotFoundError(f"Recognition config file does not exist: {path}")
            return defaults
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read recognition config {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError("Recognition config must be a JSON object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown recognition config key(s): {', '.join(unknown)}")
        values = dict(raw)
        if "model_path" in values:
            values["model_path"] = Path(values["model_path"]).expanduser()
        if values.get("cache_dir") is not None:
            values["cache_dir"] = Path(values["cache_dir"]).expanduser()
        if values.get("tensorrt_dll_dir") is not None:
            values["tensorrt_dll_dir"] = Path(values["tensorrt_dll_dir"]).expanduser()
        options = cls(**{**defaults.__dict__, **values})
        options.validate()
        return options

    def validate(self) -> None:
        if self.backend not in {"face01", "openvino"}:
            raise ValueError("backend must be 'face01' or 'openvino'")
        if self.acceleration not in {"auto", "cpu", "cuda", "tensorrt"}:
            raise ValueError("acceleration must be 'auto', 'cpu', 'cuda', or 'tensorrt'")
        if self.precision not in {"accurate", "balanced"}:
            raise ValueError("precision must be 'accurate' or 'balanced'")
        if not 1 <= self.threads <= 8:
            raise ValueError("threads must be between 1 and 8")
        if not 320 <= self.detection_max_dimension <= 4096:
            raise ValueError("detection_max_dimension must be between 320 and 4096")
        if not 40 <= self.min_face_size <= 512:
            raise ValueError("min_face_size must be between 40 and 512")
        if self.gpu_device_id < 0:
            raise ValueError("gpu_device_id must be zero or greater")


def default_config_path() -> Path:
    """Return the optional per-user recognition settings file."""
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "VideoAtlasPython"
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "VideoAtlasPython"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "VideoAtlasPython"
    return root / "recognition.json"
