from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceFolder:
    id: str
    path: str
    enabled: bool = True


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
    file_hash: str | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    analyzed_at: str | None = None
    analysis_version: str | None = None
    analysis_seconds: float = 0.0


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
    track_id: str | None = None
    frame_number: int | None = None
    detection_score: float | None = None
    landmarks: list[list[float]] | None = None
    pose: dict[str, float] | str | None = None
    aligned_path: str | None = None
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embedding is not None and self.embedding_model:
            self.embeddings.setdefault(self.embedding_model, list(self.embedding))


@dataclass
class PersonRecord:
    id: str
    name: str
    thumbnail_path: str | None
    representative_face_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class TrackRecord:
    id: str
    video_id: str
    start_time: float
    end_time: float
    person_id: str | None = None
    face_ids: list[str] = field(default_factory=list)
    representative_face_id: str | None = None
    embedding_samples: dict[str, list[list[float]]] = field(default_factory=dict)
    quality: float = 0.0
    sample_face_ids: list[str] = field(default_factory=list)
    mean_embeddings: dict[str, list[float]] = field(default_factory=dict)
    central_face_id: str | None = None
    pose_face_ids: dict[str, str] = field(default_factory=dict)
    review_required: bool = False
    sample_settings: str | None = None


@dataclass
class RelationRecord:
    track_a: str
    track_b: str
    relation: str
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.track_a == self.track_b:
            raise ValueError("A relation requires two distinct tracks")
        if self.relation not in {"confirmed_same", "confirmed_different", "pending", "deferred"}:
            raise ValueError("Unknown track relation")
        self.track_a, self.track_b = sorted((self.track_a, self.track_b))


@dataclass
class LibrarySnapshot:
    sources: list[SourceFolder]
    videos: list[VideoRecord]
    faces: list[FaceRecord]
    people: list[PersonRecord]
    tracks: list[TrackRecord] = field(default_factory=list)
    relations: list[RelationRecord] = field(default_factory=list)


class LibraryStoreError(RuntimeError):
    pass


class LibraryStore:
    """SQLite payload compatibility plus normalized, model-versioned embeddings.

    Track relation endpoints intentionally have no cascading foreign keys: human
    labels survive index refreshes, person deletion, merge and split operations.
    """
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
            self._migrate_embeddings()
            self._import_legacy_json(base / "library.json")
            self._migrate_person_timestamps()
            self._migrate_tracks()
        except Exception:
            self._connection.close()
            raise

    @staticmethod
    def _default_root() -> Path:
        return Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home() / "AppData" / "Local") / "VideoAtlasPython"

    def _create_tables(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version > 2:
            raise LibraryStoreError(f"Library schema {version} is newer than supported schema 2")
        with self._connection:
            for sql in (
                "CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS faces (id TEXT PRIMARY KEY NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE, person_id TEXT REFERENCES people(id) ON DELETE SET NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS tracks (id TEXT PRIMARY KEY NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE, person_id TEXT REFERENCES people(id) ON DELETE SET NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS relations (track_a TEXT NOT NULL, track_b TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(track_a,track_b))",
                "CREATE TABLE IF NOT EXISTS face_embeddings (face_id TEXT NOT NULL REFERENCES faces(id) ON DELETE CASCADE, model_name TEXT NOT NULL, model_version TEXT NOT NULL, embedding TEXT NOT NULL, PRIMARY KEY(face_id,model_name,model_version))",
                "CREATE TABLE IF NOT EXISTS analysis_settings (id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS tracks_person ON tracks(person_id)",
                "CREATE INDEX IF NOT EXISTS faces_video ON faces(video_id)",
            ):
                self._connection.execute(sql)
            self._connection.execute("PRAGMA user_version = 2")

    def _migrate_embeddings(self) -> None:
        with self._connection:
            for row in self._connection.execute("SELECT payload FROM faces WHERE id NOT IN (SELECT face_id FROM face_embeddings)").fetchall():
                self._write_embeddings(self._face_from_dict(self._decode(row["payload"])))

    def _import_legacy_json(self, path: Path) -> None:
        if not path.exists() or self._connection.execute("SELECT 1 FROM schema_meta WHERE key='legacy_json_processed'").fetchone():
            return
        if self._connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]:
            with self._connection:
                self._connection.execute("INSERT INTO schema_meta VALUES('legacy_json_processed','existing_index_retained')")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise LibraryStoreError("Legacy library must contain an object")
        with self._connection:
            for table, record_type in (("sources", SourceFolder), ("videos", VideoRecord), ("people", PersonRecord), ("faces", FaceRecord), ("tracks", TrackRecord)):
                for values in data.get(table, []):
                    record = self._face_from_dict(values) if table == "faces" else record_type(**values)
                    self._write_record(table, record)
            for values in data.get("relations", []):
                self._write_relation(RelationRecord(**values))
            self._connection.execute("INSERT INTO schema_meta VALUES('legacy_json_processed','imported')")

    def _migrate_tracks(self) -> None:
        """Attach legacy detections to tracks without changing their person IDs."""
        from collections import defaultdict
        from .tracking import build_tracks
        snapshot = self.load_snapshot()
        groups = defaultdict(list)
        for face in snapshot.faces:
            if not face.track_id and not face.excluded:
                groups[(face.video_id, face.person_id)].append(face)
        with self._connection:
            for faces in groups.values():
                for track in build_tracks(faces, [], []):
                    self._write_record("tracks", track)
                for face in faces:
                    self._write_record("faces", face)

    def _migrate_person_timestamps(self) -> None:
        with self._connection:
            for row in self._rows("people"):
                values = self._decode(row["payload"])
                if "created_at" not in values or "updated_at" not in values:
                    self._write_record("people", PersonRecord(**values))

    def load_snapshot(self) -> LibrarySnapshot:
        def records(table: str, record_type: type) -> list:
            result = []
            for row in self._rows(table):
                values = self._decode(row["payload"])
                if table in {"faces", "tracks"}:
                    values["person_id"] = row["person_id"]
                result.append(self._face_from_dict(values) if table == "faces" else record_type(**values))
            return result
        return LibrarySnapshot(records("sources", SourceFolder), records("videos", VideoRecord), records("faces", FaceRecord), records("people", PersonRecord), records("tracks", TrackRecord), records("relations", RelationRecord))

    def _write_record(self, table: str, record: Any) -> None:
        payload = json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":"))
        columns = ["id"]
        values = [record.id]
        if table == "videos":
            columns.append("source_id")
            values.append(record.source_id)
        if table in {"faces", "tracks"}:
            columns.extend(["video_id", "person_id"])
            values.extend([record.video_id, record.person_id])
        columns.append("payload")
        values.append(payload)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns[1:])
        self._connection.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) ON CONFLICT(id) DO UPDATE SET {updates}", values)
        if table == "faces":
            self._write_embeddings(record)

    def _write_embeddings(self, face: FaceRecord) -> None:
        self._connection.execute("DELETE FROM face_embeddings WHERE face_id=?", (face.id,))
        for key, vector in face.embeddings.items():
            name, separator, version = key.partition(":")
            self._connection.execute("INSERT INTO face_embeddings VALUES(?,?,?,?)", (face.id, name, version if separator else "legacy-unversioned", json.dumps(vector)))

    def _save(self, table: str, record: Any) -> None:
        with self._connection:
            self._write_record(table, record)

    def save_source(self, source: SourceFolder) -> None:
        self._save("sources", source)

    def save_video(self, video: VideoRecord) -> None:
        self._save("videos", video)

    def save_person(self, person: PersonRecord) -> None:
        self._save("people", person)

    def save_face(self, face: FaceRecord) -> None:
        self._save("faces", face)

    def save_track(self, track: TrackRecord) -> None:
        self._save("tracks", track)

    def _write_relation(self, relation: RelationRecord) -> None:
        self._connection.execute("INSERT INTO relations VALUES(?,?,?) ON CONFLICT(track_a,track_b) DO UPDATE SET payload=excluded.payload", (relation.track_a, relation.track_b, json.dumps(asdict(relation))))

    def save_relation(self, relation: RelationRecord) -> None:
        with self._connection:
            self._write_relation(relation)

    def resolve_relation(self, track_a: str, track_b: str, relation: str) -> RelationRecord:
        value = RelationRecord(track_a, track_b, relation)
        snapshot = self.load_snapshot()
        tracks = {track.id: track for track in snapshot.tracks}
        if track_a not in tracks or track_b not in tracks:
            raise ValueError("Both relation tracks must exist")
        old_relation = next((r for r in snapshot.relations if (r.track_a, r.track_b) == (value.track_a, value.track_b)), None)
        if old_relation:
            value.details = dict(old_relation.details)
            if old_relation.relation != relation:
                value.details["history"] = list(value.details.get("history", [])) + [{"relation": old_relation.relation, "updated_at": old_relation.updated_at}]
        left, right = tracks[track_a], tracks[track_b]
        original_source_id = right.person_id
        people = {person.id: person for person in snapshot.people}
        source_tracks = [track for track in snapshot.tracks if track.person_id == right.person_id] if right.person_id else [right]
        target_tracks = [track for track in snapshot.tracks if track.person_id == left.person_id] if left.person_id else [left]
        pairs = {tuple(sorted((a.id, b.id))) for a in source_tracks for b in target_tracks if a.id != b.id}
        if relation == "confirmed_same" and any((r.track_a, r.track_b) in pairs and r.relation == "confirmed_different" for r in snapshot.relations):
            raise ValueError("Merge conflicts with a confirmed different-person decision")
        with self._connection:
            if relation == "confirmed_same" and (left.person_id != right.person_id or left.person_id is None):
                target_id = left.person_id
                if target_id is None:
                    representative = next((face for face in snapshot.faces if face.id == left.representative_face_id), None)
                    person = PersonRecord(str(uuid4()), "", representative.thumbnail_path if representative else None, left.representative_face_id)
                    self._write_record("people", person)
                    target_id = person.id
                affected = {track.id for track in source_tracks + target_tracks}
                for track in snapshot.tracks:
                    if track.id in affected:
                        track.person_id = target_id
                        track.review_required = False
                        self._write_record("tracks", track)
                for face in snapshot.faces:
                    if face.track_id in affected:
                        face.person_id, face.manual_assignment = target_id, True
                        self._write_record("faces", face)
                for a, b in pairs:
                    self._write_relation(RelationRecord(a, b, "confirmed_same", {"origin": "human_review"}))
                referenced = {track.person_id for track in snapshot.tracks} | {face.person_id for face in snapshot.faces}
                if original_source_id and original_source_id not in referenced and original_source_id != target_id:
                    self._connection.execute("DELETE FROM people WHERE id=?", (original_source_id,))
                    snapshot.people = [person for person in snapshot.people if person.id != original_source_id]
            elif relation == "confirmed_different" and left.person_id is not None and left.person_id == right.person_id:
                representative = next((face for face in snapshot.faces if face.id == right.representative_face_id), None)
                person = PersonRecord(str(uuid4()), "", representative.thumbnail_path if representative else None, right.representative_face_id)
                self._write_record("people", person)
                old_person = right.person_id
                right.person_id = person.id
                self._write_record("tracks", right)
                for face in snapshot.faces:
                    if face.track_id == right.id or face.id in right.face_ids:
                        face.person_id, face.manual_assignment = person.id, True
                        self._write_record("faces", face)
                labelled = {(r.track_a, r.track_b) for r in snapshot.relations}
                for other in snapshot.tracks:
                    pair = tuple(sorted((right.id, other.id)))
                    if other.id != right.id and other.person_id == old_person and pair not in labelled:
                        self._write_relation(RelationRecord(*pair, "confirmed_different", {"origin": "human_review_split"}))
            self._repair_representatives(snapshot)
            self._write_relation(value)
        return value

    def save_analysis_settings(self, fingerprint: str, settings: dict[str, Any]) -> None:
        with self._connection:
            self._connection.execute("INSERT INTO analysis_settings(fingerprint,created_at,payload) VALUES(?,?,?)", (fingerprint, utc_now(), json.dumps(settings)))

    def load_analysis_settings(self) -> list[dict[str, Any]]:
        return [{"fingerprint": row["fingerprint"], "created_at": row["created_at"], "settings": json.loads(row["payload"])} for row in self._connection.execute("SELECT * FROM analysis_settings ORDER BY id")]

    def save_batch(self, faces: list[FaceRecord], video: VideoRecord) -> None:
        self.apply_face_batch(faces, video, [], [])

    def apply_face_batch(self, faces: list[FaceRecord], video: VideoRecord, people: list[PersonRecord], remove_face_ids: list[str], tracks: list[TrackRecord] | None = None, relations: list[RelationRecord] | None = None) -> None:
        if any(face.video_id != video.id for face in faces):
            raise ValueError("Every face must belong to the batch video")
        with self._connection:
            self._write_record("videos", video)
            for face_id in remove_face_ids:
                self._connection.execute("DELETE FROM faces WHERE id=?", (face_id,))
            for person in people:
                self._write_record("people", person)
            for face in faces:
                self._write_record("faces", face)
            for track in tracks or []:
                self._write_record("tracks", track)
            if tracks is not None:
                active_ids = {track.id for track in tracks if track.video_id == video.id and track.face_ids}
                for row in self._connection.execute("SELECT id FROM tracks WHERE video_id=?", (video.id,)).fetchall():
                    if row["id"] not in active_ids:
                        self._connection.execute("DELETE FROM tracks WHERE id=?", (row["id"],))
            for relation in relations or []:
                self._write_relation(relation)

    def apply_identity_result(self, faces: list[FaceRecord], people: list[PersonRecord], tracks: list[TrackRecord], relations: list[RelationRecord]) -> None:
        with self._connection:
            for person in people:
                self._write_record("people", person)
            for track in tracks:
                self._write_record("tracks", track)
            for face in faces:
                self._write_record("faces", face)
            for relation in relations:
                self._write_relation(relation)

    def merge_people(self, source_id: str, target_id: str) -> None:
        if source_id == target_id:
            return
        snapshot = self.load_snapshot()
        persons = {person.id: person for person in snapshot.people}
        if source_id not in persons or target_id not in persons:
            raise ValueError("Both people must exist")
        source_tracks = [track for track in snapshot.tracks if track.person_id == source_id]
        target_tracks = [track for track in snapshot.tracks if track.person_id == target_id]
        pairs = {tuple(sorted((a.id, b.id))) for a in source_tracks for b in target_tracks}
        if any((r.track_a, r.track_b) in pairs and r.relation == "confirmed_different" for r in snapshot.relations):
            raise ValueError("Merge conflicts with a confirmed different-person decision")
        with self._connection:
            for face in snapshot.faces:
                if face.person_id == source_id:
                    face.person_id, face.manual_assignment = target_id, True
                    self._write_record("faces", face)
            for track in source_tracks:
                track.person_id = target_id
                track.review_required = False
                self._write_record("tracks", track)
            for a, b in pairs:
                self._write_relation(RelationRecord(a, b, "confirmed_same", {"origin": "human_merge"}))
            persons[target_id].updated_at = utc_now()
            self._write_record("people", persons[target_id])
            self._repair_representatives(snapshot)
            self._connection.execute("DELETE FROM people WHERE id=?", (source_id,))

    def split_track(self, track_id: str, name: str = "") -> PersonRecord:
        snapshot = self.load_snapshot()
        track = next((track for track in snapshot.tracks if track.id == track_id), None)
        if track is None:
            raise ValueError("Unknown track")
        face = next((face for face in snapshot.faces if face.id == track.representative_face_id), None)
        person = PersonRecord(str(uuid4()), name, face.thumbnail_path if face else None, track.representative_face_id)
        previous_person = track.person_id
        labelled_pairs = {(r.track_a, r.track_b): r for r in snapshot.relations}
        with self._connection:
            self._write_record("people", person)
            track.person_id = person.id
            self._write_record("tracks", track)
            for face in snapshot.faces:
                if face.track_id == track_id or face.id in track.face_ids:
                    face.person_id, face.manual_assignment = person.id, True
                    self._write_record("faces", face)
            for other in snapshot.tracks:
                pair = tuple(sorted((track.id, other.id)))
                if other.id != track.id and other.person_id == previous_person:
                    previous = labelled_pairs.get(pair)
                    details = dict(previous.details) if previous else {}
                    if previous and previous.relation != "confirmed_different":
                        details["history"] = list(details.get("history", [])) + [{"relation": previous.relation, "updated_at": previous.updated_at}]
                    details["origin"] = "human_split"
                    self._write_relation(RelationRecord(*pair, "confirmed_different", details))
            self._repair_representatives(snapshot)
        return person

    def assign_track(self, track_id: str, person_id: str | None) -> None:
        snapshot = self.load_snapshot()
        track = next((track for track in snapshot.tracks if track.id == track_id), None)
        if track is None or (person_id is not None and not any(person.id == person_id for person in snapshot.people)):
            raise ValueError("Unknown track or person")
        peers = {item.id for item in snapshot.tracks if item.person_id == person_id and item.id != track_id} if person_id else set()
        if any(relation.relation == "confirmed_different" and track_id in (relation.track_a, relation.track_b) and peers.intersection((relation.track_a, relation.track_b)) for relation in snapshot.relations):
            raise ValueError("Assignment conflicts with a confirmed different-person decision")
        with self._connection:
            track.person_id = person_id
            self._write_record("tracks", track)
            for face in snapshot.faces:
                if face.track_id == track_id or face.id in track.face_ids:
                    face.person_id, face.manual_assignment = person_id, True
                    self._write_record("faces", face)
            relations = {(r.track_a, r.track_b): r for r in snapshot.relations}
            for peer_id in peers:
                pair = tuple(sorted((track_id, peer_id)))
                previous = relations.get(pair)
                details = dict(previous.details) if previous else {}
                if previous and previous.relation != "confirmed_same":
                    details["history"] = list(details.get("history", [])) + [{"relation": previous.relation, "updated_at": previous.updated_at}]
                details["origin"] = "human_assignment"
                self._write_relation(RelationRecord(*pair, "confirmed_same", details))
            self._repair_representatives(snapshot)

    def _repair_representatives(self, snapshot: LibrarySnapshot) -> None:
        faces = {face.id: face for face in snapshot.faces}
        for person in snapshot.people:
            current = faces.get(person.representative_face_id)
            if current and current.person_id == person.id and not current.excluded:
                continue
            candidates = [face for face in snapshot.faces if face.person_id == person.id and not face.excluded and face.embeddings]
            chosen = max(candidates, key=lambda face: face.quality or 0) if candidates else None
            values = chosen.id if chosen else None, chosen.thumbnail_path if chosen else None
            if values != (person.representative_face_id, person.thumbnail_path):
                person.representative_face_id, person.thumbnail_path = values
                person.updated_at = utc_now()
                self._write_record("people", person)

    def exclude_face(self, face_id: str) -> None:
        from .tracking import refresh_representatives
        snapshot = self.load_snapshot()
        face_map = {face.id: face for face in snapshot.faces}
        if face_id not in face_map:
            raise ValueError("Unknown face")
        face = face_map[face_id]
        face.excluded = face.manual_assignment = True
        with self._connection:
            self._write_record("faces", face)
            for track in snapshot.tracks:
                if face_id in track.face_ids:
                    refresh_representatives(track, face_map)
                    self._write_record("tracks", track)
            for person in snapshot.people:
                if person.representative_face_id == face_id:
                    alternatives = [candidate for candidate in snapshot.faces if candidate.person_id == person.id and not candidate.excluded]
                    representative = max(alternatives, key=lambda candidate: candidate.quality or 0) if alternatives else None
                    person.representative_face_id = representative.id if representative else None
                    person.thumbnail_path = representative.thumbnail_path if representative else None
                    person.updated_at = utc_now()
                    self._write_record("people", person)

    def set_representative(self, face_id: str) -> None:
        snapshot = self.load_snapshot()
        face = next((face for face in snapshot.faces if face.id == face_id and not face.excluded), None)
        person = next((person for person in snapshot.people if face and person.id == face.person_id), None)
        if face is None or person is None:
            raise ValueError("Representative must be an included face belonging to a person")
        person.representative_face_id, person.thumbnail_path = face.id, face.thumbnail_path
        person.updated_at = utc_now()
        with self._connection:
            self._write_record("people", person)
            for track in snapshot.tracks:
                if track.id == face.track_id:
                    track.representative_face_id = face.id
                    self._write_record("tracks", track)

    def _delete(self, table: str, identifier: str) -> None:
        with self._connection:
            self._connection.execute(f"DELETE FROM {table} WHERE id=?", (identifier,))

    def delete_video(self, identifier: str) -> None:
        self._delete("videos", identifier)

    def delete_source(self, identifier: str) -> None:
        self._delete("sources", identifier)

    def delete_person(self, identifier: str) -> None:
        self._delete("people", identifier)

    def delete_face(self, identifier: str) -> None:
        self._delete("faces", identifier)

    def clear_index(self) -> None:
        with self._connection:
            for table in ("relations", "faces", "tracks", "videos", "people", "sources", "analysis_settings"):
                self._connection.execute(f"DELETE FROM {table}")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "LibraryStore":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

    def _rows(self, table: str) -> list[sqlite3.Row]:
        columns = "payload,person_id" if table in {"faces", "tracks"} else "payload"
        return self._connection.execute(f"SELECT {columns} FROM {table} ORDER BY rowid").fetchall()

    @staticmethod
    def _decode(payload: str) -> dict[str, Any]:
        return json.loads(payload)

    @staticmethod
    def _face_from_dict(values: dict[str, Any]) -> FaceRecord:
        values = dict(values)
        if values.get("bounding_box") is not None:
            values["bounding_box"] = tuple(values["bounding_box"])
        return FaceRecord(**values)


VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"})


def list_video_files(folder: Path, recursive: bool = True, extensions: set[str] | frozenset[str] | None = None) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    allowed = {"." + ext.lower().lstrip(".") for ext in (extensions if extensions is not None else VIDEO_EXTENSIONS)}
    results: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if recursive and not (Path(current) / name).is_symlink()]
        for filename in filenames:
            candidate = Path(current) / filename
            if not candidate.is_symlink() and candidate.suffix.lower() in allowed and candidate.is_file():
                results.append(candidate.resolve())
    return sorted(results, key=lambda path: str(path).casefold())


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("Hash chunk size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
