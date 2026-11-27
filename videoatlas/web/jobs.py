"""Queued GUI jobs, run one at a time as separate processes.

Each job runs ``python -m videoatlas.cli ...`` in its own process group. A model
crash ends only that process, and one runner thread guarantees that two GPU jobs
never run together.
"""
from __future__ import annotations

from collections import deque
from contextlib import closing
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import threading
import time
from typing import Callable

from .settings import WebSettings

MAX_OUTPUT_BYTES = 1024 * 1024
LOG_LINES = 50
MAX_LINE_CHARS = 2000
TERMINATE_GRACE_SECONDS = 10.0
PROGRESS = re.compile(r"^(\d+)/(\d+)\b")
ACTIVE = ("queued", "running")
FINISHED = ("succeeded", "failed", "stopped", "interrupted")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    progress_done INTEGER,
    progress_total INTEGER,
    result TEXT,
    error TEXT,
    log_tail TEXT NOT NULL DEFAULT '',
    stop_requested INTEGER NOT NULL DEFAULT 0
)
"""


@dataclass(frozen=True)
class JobKind:
    name: str
    title: str
    argv: Callable[[dict], list[str]]
    json_result: bool


JOB_KINDS: dict[str, JobKind] = {
    "doctor": JobKind("doctor", "GPU check", lambda params: ["-m", "videoatlas.cli", "doctor", "--device", "cuda"], True),
}


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    params: dict
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    progress_done: int | None
    progress_total: int | None
    result: object
    error: str | None
    log_tail: str
    stop_requested: bool

    @property
    def active(self) -> bool:
        return self.status in ACTIVE


def _job(row: sqlite3.Row) -> Job:
    return Job(row["id"], row["kind"], json.loads(row["params"]), row["status"], row["created_at"], row["started_at"],
               row["finished_at"], row["progress_done"], row["progress_total"],
               json.loads(row["result"]) if row["result"] is not None else None, row["error"], row["log_tail"],
               bool(row["stop_requested"]))


class JobStore:
    """Jobs in a small SQLite database. Safe to use from several threads."""

    def __init__(self, path: Path, kinds: dict[str, JobKind] = JOB_KINDS):
        self.path = Path(path)
        self.kinds = kinds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _one(self, query: str, args: tuple = ()) -> Job | None:
        with closing(self._connect()) as connection:
            row = connection.execute(query, args).fetchone()
        return _job(row) if row is not None else None

    def _execute(self, query: str, args: tuple = ()) -> int:
        with closing(self._connect()) as connection:
            return connection.execute(query, args).rowcount

    def enqueue(self, kind: str, params: dict | None = None) -> int:
        if kind not in self.kinds:
            raise ValueError(f"Unknown job kind: {kind!r}")
        with closing(self._connect()) as connection:
            cursor = connection.execute("INSERT INTO jobs (kind, params, status, created_at) VALUES (?, ?, 'queued', ?)",
                                        (kind, json.dumps(params or {}, sort_keys=True), time.time()))
            return int(cursor.lastrowid)

    def enqueue_unique(self, kind: str, params: dict | None = None) -> int:
        """Queue a job unless the same kind is already queued or running."""
        if kind not in self.kinds:
            raise ValueError(f"Unknown job kind: {kind!r}")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT id FROM jobs WHERE kind = ? AND status IN ('queued', 'running') ORDER BY id LIMIT 1",
                                     (kind,)).fetchone()
            if row is not None:
                connection.execute("COMMIT")
                return int(row["id"])
            cursor = connection.execute("INSERT INTO jobs (kind, params, status, created_at) VALUES (?, ?, 'queued', ?)",
                                        (kind, json.dumps(params or {}, sort_keys=True), time.time()))
            connection.execute("COMMIT")
            return int(cursor.lastrowid)

    def get(self, job_id: int) -> Job | None:
        return self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))

    def recent(self, limit: int = 10) -> list[Job]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_job(row) for row in rows]

    def latest_finished(self, kind: str) -> Job | None:
        return self._one("SELECT * FROM jobs WHERE kind = ? AND status IN ('succeeded', 'failed') ORDER BY id DESC LIMIT 1", (kind,))

    def claim_next(self) -> Job | None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?", (time.time(), row["id"]))
            connection.execute("COMMIT")
        return self.get(int(row["id"]))

    def set_progress(self, job_id: int, done: int | None, total: int | None, log_tail: str) -> None:
        self._execute("UPDATE jobs SET progress_done = ?, progress_total = ?, log_tail = ? WHERE id = ?",
                      (done, total, log_tail, job_id))

    def finish(self, job_id: int, status: str, result: object = None, error: str | None = None,
               log_tail: str | None = None) -> None:
        if status not in FINISHED:
            raise ValueError(f"Not a finished status: {status!r}")
        self._execute("UPDATE jobs SET status = ?, finished_at = ?, result = ?, error = ?, log_tail = COALESCE(?, log_tail) WHERE id = ?",
                      (status, time.time(), json.dumps(result) if result is not None else None, error, log_tail, job_id))

    def request_stop(self, job_id: int) -> bool:
        """Cancel a queued job, or ask the runner to stop a running one."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] not in ACTIVE:
                connection.execute("COMMIT")
                return False
            if row["status"] == "queued":
                connection.execute("UPDATE jobs SET status = 'stopped', finished_at = ? WHERE id = ?", (time.time(), job_id))
            else:
                connection.execute("UPDATE jobs SET stop_requested = 1 WHERE id = ?", (job_id,))
            connection.execute("COMMIT")
            return True

    def mark_interrupted(self) -> int:
        return self._execute("UPDATE jobs SET status = 'interrupted', finished_at = ?, error = ? WHERE status = 'running'",
                             (time.time(), "The GUI server stopped while this job was running."))


class JobRunner:
    """Runs queued jobs one at a time in a background thread."""

    def __init__(self, store: JobStore, settings: WebSettings, kinds: dict[str, JobKind] = JOB_KINDS,
                 poll_seconds: float = 0.5):
        self.store = store
        self.settings = settings
        self.kinds = kinds
        self.poll_seconds = poll_seconds
        self._closing = threading.Event()
        self._record = True
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._lock: int | None = None

    def start(self) -> None:
        """Take the data folder's runner lock, mark jobs left running by a crash, and start."""
        if self._thread is not None:
            return
        descriptor = os.open(self.store.path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            raise RuntimeError("Another VideoAtlas GUI server is already running jobs from this data folder.") from None
        self._lock = descriptor
        self.store.mark_interrupted()
        self._thread = threading.Thread(target=self._loop, name="videoatlas-jobs", daemon=True)
        self._thread.start()

    def close(self, timeout: float = TERMINATE_GRACE_SECONDS + 5, record: bool = True) -> None:
        """Stop the runner and any running child. record=False leaves the database as a crash would."""
        self._record = record
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def _loop(self) -> None:
        while not self._closing.is_set():
            job = self.store.claim_next()
            if job is None:
                self._closing.wait(self.poll_seconds)
                continue
            try:
                self._run(job)
            except Exception as error:  # keep the runner alive for the next job
                self.store.finish(job.id, "failed", error=f"The GUI could not run this job: {error}")

    def _run(self, job: Job) -> None:
        kind = self.kinds[job.kind]
        environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen([self.settings.python, *kind.argv(job.params)], cwd=self.settings.repo_root,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, env=environment)
        self._process = process
        lines: deque[str] = deque(maxlen=LOG_LINES)
        progress: list[int | None] = [None, None]
        stdout = bytearray()
        overflow = threading.Event()

        def read_stderr() -> None:
            for raw in process.stderr:
                line = raw.decode("utf-8", "replace").rstrip()[:MAX_LINE_CHARS]
                lines.append(line)
                match = PROGRESS.match(line)
                if match:
                    progress[0], progress[1] = int(match.group(1)), int(match.group(2))
                self.store.set_progress(job.id, progress[0], progress[1], "\n".join(lines))

        def read_stdout() -> None:
            while chunk := process.stdout.read(65536):
                if len(stdout) + len(chunk) > MAX_OUTPUT_BYTES:
                    overflow.set()
                else:
                    stdout.extend(chunk)

        readers = [threading.Thread(target=read_stderr, daemon=True), threading.Thread(target=read_stdout, daemon=True)]
        for reader in readers:
            reader.start()
        stopped_by_user = False
        while process.poll() is None:
            if self._closing.is_set():
                break
            current = self.store.get(job.id)
            if current is not None and current.stop_requested:
                stopped_by_user = True
                break
            time.sleep(self.poll_seconds)
        if process.poll() is None:
            self._terminate(process)
        for reader in readers:
            reader.join(5)
        self._process = None
        tail = "\n".join(lines)
        if self._closing.is_set() and not stopped_by_user:
            if self._record:
                self.store.finish(job.id, "interrupted", error="The GUI server stopped while this job was running.", log_tail=tail)
            return
        if stopped_by_user:
            self.store.finish(job.id, "stopped", log_tail=tail)
        elif process.returncode != 0:
            errors = [line for line in lines if line.startswith("error:")]
            message = errors[-1] if errors else (lines[-1] if lines else f"The command exited with code {process.returncode}.")
            self.store.finish(job.id, "failed", error=message, log_tail=tail)
        elif overflow.is_set():
            self.store.finish(job.id, "failed", error="The command output was too large (over 1 MiB).", log_tail=tail)
        elif kind.json_result:
            try:
                result = json.loads(stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.store.finish(job.id, "failed", error="The command did not print valid JSON.", log_tail=tail)
            else:
                self.store.finish(job.id, "succeeded", result=result, log_tail=tail)
        else:
            self.store.finish(job.id, "succeeded", log_tail=tail)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        for sent, wait in ((signal.SIGTERM, TERMINATE_GRACE_SECONDS), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(process.pid, sent)
            except ProcessLookupError:
                return
            try:
                process.wait(wait)
                return
            except subprocess.TimeoutExpired:
                continue
