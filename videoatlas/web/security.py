"""Access key and session handling for the browser GUI.

The access key is a second lock behind the SSH tunnel: other programs on the
viewing computer, and web pages open in its browser, can reach the forwarded
port, but they do not know the key.
"""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import os
from pathlib import Path
import secrets
import stat
import threading
import time

KEY_BYTES = 32
MIN_KEY_LENGTH = 43


def _write_key(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(key + "\n")
    os.replace(temporary, path)


def load_or_create_key(path: Path) -> str:
    """Read the access key, creating it with owner-only permissions when missing."""
    path = Path(path)
    if not path.exists():
        _write_key(path, secrets.token_urlsafe(KEY_BYTES))
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        path.chmod(0o600)
    key = path.read_text(encoding="utf-8").strip()
    if len(key) < MIN_KEY_LENGTH:
        raise ValueError(f"The access key in {path} is too short. Run 'python -m videoatlas.web rotate-key'.")
    return key


def rotate_key(path: Path) -> str:
    """Replace the access key. Existing sessions end when the server restarts."""
    key = secrets.token_urlsafe(KEY_BYTES)
    _write_key(Path(path), key)
    return key


def keys_match(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


@dataclass(frozen=True)
class Session:
    token: str
    csrf: str
    created: float


class SessionStore:
    """In-memory sessions. A server restart signs the user out."""

    def __init__(self, ttl_seconds: int = 7 * 24 * 3600):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        session = Session(secrets.token_urlsafe(KEY_BYTES), secrets.token_urlsafe(KEY_BYTES), time.monotonic())
        with self._lock:
            self._sessions[session.token] = session
        return session

    def get(self, token: str | None) -> Session | None:
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if time.monotonic() - session.created >= self.ttl_seconds:
                del self._sessions[token]
                return None
            return session

    def delete(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)
