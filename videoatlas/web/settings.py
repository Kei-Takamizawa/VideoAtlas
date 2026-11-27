"""Settings for the browser GUI server."""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import os
from pathlib import Path
import sys

LOOPBACK_NAMES = frozenset({"localhost"})
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Folder for the GUI's database, access key, and future uploads."""
    configured = os.environ.get("VIDEOATLAS_DATA_DIR")
    return Path(configured).expanduser() if configured else Path.home() / "videoatlas-data"


def is_loopback(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class WebSettings:
    data_dir: Path = field(default_factory=default_data_dir)
    host: str = "127.0.0.1"
    port: int = 8765
    python: str = sys.executable
    repo_root: Path = REPO_ROOT

    def validate(self) -> None:
        if not is_loopback(self.host):
            raise ValueError(f"The GUI listens on a loopback address only; {self.host!r} is not one. "
                             "Use an SSH tunnel to reach it from another computer.")
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"The port must be between 1024 and 65535, not {self.port}.")

    @property
    def web_dir(self) -> Path:
        return Path(self.data_dir) / "web"

    @property
    def jobs_db(self) -> Path:
        return self.web_dir / "jobs.sqlite3"

    @property
    def key_file(self) -> Path:
        return self.web_dir / "access-key"
