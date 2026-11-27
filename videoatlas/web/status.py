"""Summaries shown on the home page."""
from __future__ import annotations

from pathlib import Path
import shutil
import time

from .jobs import Job

GPU_PROVIDERS = frozenset({"CUDAExecutionProvider", "TensorrtExecutionProvider"})


def gpu_summary(job: Job | None) -> dict:
    """Where each model runs, from the latest finished GPU check."""
    if job is None:
        return {"state": "unknown", "models": [], "error": None, "checked": None}
    checked = ago(job.finished_at)
    if job.status != "succeeded":
        return {"state": "failed", "models": [], "error": job.error or "The GPU check did not finish.", "checked": checked}
    providers = (job.result or {}).get("providers", {}) if isinstance(job.result, dict) else {}
    models = [(str(name), str(values[0]) if values else "none") for name, values in sorted(providers.items())]
    if not models:
        return {"state": "failed", "models": [], "error": "The GPU check reported no models.", "checked": checked}
    state = "ok" if all(provider in GPU_PROVIDERS for _, provider in models) else "cpu"
    return {"state": state, "models": models, "error": None, "checked": checked}


def disk_summary(path: Path) -> dict:
    """Free space on the disk that holds the data folder."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return {"path": str(path), "free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}


def ago(timestamp: float | None, now: float | None = None) -> str | None:
    if timestamp is None:
        return None
    seconds = max(0, int((now if now is not None else time.time()) - timestamp))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} days ago"


def progress_percent(job: Job) -> int | None:
    if not job.progress_total:
        return None
    return max(0, min(100, round(100 * (job.progress_done or 0) / job.progress_total)))
