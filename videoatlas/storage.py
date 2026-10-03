from dataclasses import asdict, dataclass
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


@dataclass
class SourceFolder:
    id: str
    path: str


@dataclass
class VideoRecord:
    id: str
    source_id: str
    path: str
    name: str
    duration: float
    file_size: int
    modification_time: float
    state: str
    last_analyzed_second: float
    sample_interval: float
    error_message: str | None
    poster_path: str | None
    embedding_model: str | None = None


@dataclass
class FaceRecord:
    id: str
    video_id: str
    second: float
    bounding_box: tuple[float, float, float, float] | None
    person_id: str | None
    thumbnail_path: str
    embedding: list[float] | None
    manual_assignment: bool
    excluded: bool
    embedding_model: str | None = None
    quality: float | None = None
    rejection_reason: str | None = None


@dataclass
class PersonRecord:
    id: str
    name: str
    thumbnail_path: str | None


@dataclass
class LibrarySnapshot:
    sources: list[SourceFolder]
    videos: list[VideoRecord]
    faces: list[FaceRecord]
    people: list[PersonRecord]


class LibraryStoreError(RuntimeError):
    pass


class LibraryStore:
    def __init__(self, root: Path | None = None) -> None:
        base = self._default_root() if root is None else Path(root)
        base.mkdir(parents=True, exist_ok=True)
        self.database_path = base / "library.sqlite3"
        self._connection = sqlite3.connect(self.database_path, timeout=5.0)
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._create_tables()
        except Exception:
            self._connection.close()
            raise

    @staticmethod
    def _default_root() -> Path:
        return Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home() / "AppData" / "Local") / "VideoAtlasPython"

    def _create_tables(self) -> None:
        with self._connection:
            self._connection.execute("CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, payload TEXT NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS faces (id TEXT PRIMARY KEY NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE, person_id TEXT REFERENCES people(id) ON DELETE SET NULL, payload TEXT NOT NULL)")

    def load_snapshot(self) -> LibrarySnapshot:
        sources = [SourceFolder(**self._decode(row["payload"])) for row in self._rows("sources")]
        videos = [VideoRecord(**self._decode(row["payload"])) for row in self._rows("videos")]
        faces = [self._face_from_dict({**self._decode(row["payload"]), "person_id": row["person_id"]}) for row in self._rows("faces")]
        people = [PersonRecord(**self._decode(row["payload"])) for row in self._rows("people")]
        return LibrarySnapshot(sources=sources, videos=videos, faces=faces, people=people)

    def save_source(self, source: SourceFolder) -> None:
        self._upsert("sources", source.id, asdict(source))

    def save_video(self, video: VideoRecord) -> None:
        self._upsert("videos", video.id, asdict(video), parent_column="source_id", parent_id=video.source_id)

    def save_person(self, person: PersonRecord) -> None:
        self._upsert("people", person.id, asdict(person))

    def save_face(self, face: FaceRecord) -> None:
        self._upsert("faces", face.id, asdict(face), parent_column="video_id", parent_id=face.video_id, nullable_column="person_id", nullable_id=face.person_id)

    def save_batch(self, faces: list[FaceRecord], video: VideoRecord) -> None:
        if any(face.video_id != video.id for face in faces):
            raise ValueError("Every face must belong to the batch video")
        self.apply_face_batch(faces, video, [], [])

    def apply_face_batch(self, faces: list[FaceRecord], video: VideoRecord, people: list[PersonRecord], remove_face_ids: list[str]) -> None:
        if any(face.video_id != video.id for face in faces):
            raise ValueError("Every face must belong to the batch video")
        with self._connection:
            for face_id in remove_face_ids:
                self._connection.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            for person in people:
                person_payload = json.dumps(asdict(person), ensure_ascii=False, separators=(",", ":"))
                self._connection.execute("INSERT INTO people(id, payload) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (person.id, person_payload))
            for face in faces:
                face_payload = json.dumps(asdict(face), ensure_ascii=False, separators=(",", ":"))
                self._connection.execute("INSERT INTO faces(id, video_id, person_id, payload) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET video_id=excluded.video_id, person_id=excluded.person_id, payload=excluded.payload", (face.id, face.video_id, face.person_id, face_payload))
            video_payload = json.dumps(asdict(video), ensure_ascii=False, separators=(",", ":"))
            self._connection.execute("INSERT INTO videos(id, source_id, payload) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET source_id=excluded.source_id, payload=excluded.payload", (video.id, video.source_id, video_payload))

    def delete_video(self, identifier: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM videos WHERE id = ?", (identifier,))

    def delete_source(self, identifier: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM sources WHERE id = ?", (identifier,))

    def delete_person(self, identifier: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM people WHERE id = ?", (identifier,))

    def delete_face(self, identifier: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM faces WHERE id = ?", (identifier,))

    def clear_index(self) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM faces")
            self._connection.execute("DELETE FROM videos")
            self._connection.execute("DELETE FROM people")
            self._connection.execute("DELETE FROM sources")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "LibraryStore":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

    def _rows(self, table: str) -> list[sqlite3.Row]:
        columns = "payload, person_id" if table == "faces" else "payload"
        return self._connection.execute(f"SELECT {columns} FROM {table} ORDER BY rowid").fetchall()

    @staticmethod
    def _decode(payload: str) -> dict[str, Any]:
        return json.loads(payload)

    @staticmethod
    def _face_from_dict(values: dict[str, Any]) -> FaceRecord:
        box = values.get("bounding_box")
        values["bounding_box"] = tuple(box) if box is not None else None
        embedding = values.get("embedding")
        values["embedding"] = [float(value) for value in embedding] if embedding is not None else None
        return FaceRecord(**values)

    def _upsert(self, table: str, identifier: str, value: dict[str, Any], parent_column: str | None = None, parent_id: str | None = None, nullable_column: str | None = None, nullable_id: str | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if table == "faces":
            sql = "INSERT INTO faces(id, video_id, person_id, payload) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET video_id=excluded.video_id, person_id=excluded.person_id, payload=excluded.payload"
            parameters = (identifier, parent_id, nullable_id, payload)
        elif table == "videos":
            sql = "INSERT INTO videos(id, source_id, payload) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET source_id=excluded.source_id, payload=excluded.payload"
            parameters = (identifier, parent_id, payload)
        else:
            sql = f"INSERT INTO {table}(id, payload) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload"
            parameters = (identifier, payload)
        with self._connection:
            self._connection.execute(sql, parameters)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v"})


def list_video_files(folder: Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    results: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()]
        for filename in filenames:
            candidate = Path(current) / filename
            if candidate.is_symlink():
                continue
            if candidate.suffix.lower() in VIDEO_EXTENSIONS and candidate.is_file():
                results.append(candidate.resolve())
    return sorted(results, key=lambda path: str(path).casefold())
