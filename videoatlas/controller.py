from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Event
import time
from uuid import uuid4
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal, Slot
from .storage import FaceRecord, LibraryStore, PersonRecord, SourceFolder, VideoRecord, list_video_files


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AnalysisWorker(QThread):
    video_started = Signal(str)
    chunk_ready = Signal(str, object)
    progressed = Signal(str, float, float)
    video_error = Signal(str, str)
    fatal_error = Signal(str)
    model_ready = Signal(str, str)
    metadata_ready = Signal(str, object)
    runtime_changed = Signal(str, str)

    def __init__(self, videos, stop_event: Event, config_path: Path, cached_faces=None, anchor_times=None):
        super().__init__()
        self.videos = [replace(video) for video in videos]
        self.stop_event, self.config_path, self.cached_faces = stop_event, config_path, cached_faces
        self.anchor_times = anchor_times or {}
        self.chunk_saved, self.model_accepted = Event(), Event()

    def run(self):
        try:
            from .analyzer import VideoAnalyzer
            from .recognition import RecognitionOptions
            options = RecognitionOptions.load(self.config_path if self.config_path.is_file() else None)
            if options.cache_dir is None:
                options = replace(options, cache_dir=self.config_path.parent / "Aligned")
            analyzer = VideoAnalyzer(options=options)
        except Exception as error:
            for video in self.videos:
                self.video_error.emit(video.id, str(error))
            self.fatal_error.emit(str(error))
            return
        try:
            self.model_ready.emit(analyzer.embedding_model, analyzer.runtime_description)
            while not self.model_accepted.wait(0.1):
                if self.stop_event.is_set():
                    return
            for video in self.videos:
                if self.stop_event.is_set():
                    break
                self._run_video(analyzer, video)
        except Exception as error:
            self.fatal_error.emit(str(error))
        finally:
            analyzer.close()

    def _deliver(self, video_id, chunk):
        self.chunk_saved.clear()
        self.chunk_ready.emit(video_id, chunk)
        while not self.chunk_saved.wait(0.1):
            if self.stop_event.is_set():
                return

    def _run_video(self, analyzer, video):
        self.video_started.emit(video.id)
        started = time.monotonic()
        initial_runtime = analyzer.runtime_description
        try:
            if self.cached_faces is not None:
                from .analyzer import AnalysisChunk, FaceSample
                faces = [face for face in self.cached_faces if face.video_id == video.id]
                samples = [FaceSample(face.second, face.bounding_box, Path(face.thumbnail_path).read_bytes(),
                                      face.embedding, face.embedding_model, face.quality or 0.0, face.rejection_reason,
                                      frame_number=face.frame_number, detection_score=face.detection_score,
                                      landmarks=face.landmarks, pose=face.pose, aligned_path=face.aligned_path,
                                      embeddings=face.embeddings) for face in faces]
                recalculated = analyzer.recalculate_samples(samples, stop=self.stop_event.is_set)
                if not self.stop_event.is_set() and len(recalculated) == len(samples):
                    if analyzer.runtime_description != initial_runtime:
                        self.runtime_changed.emit(analyzer.embedding_model, analyzer.runtime_description)
                    self._deliver(video.id, AnalysisChunk(video.duration, None, recalculated, video.duration, False))
                elif not self.stop_event.is_set():
                    raise RuntimeError("Embedding recalculation returned incomplete cached results")
            else:
                import cv2
                capture = cv2.VideoCapture(video.path)
                try:
                    self.metadata_ready.emit(video.id, {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                       "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                       "fps": float(capture.get(cv2.CAP_PROP_FPS))})
                finally:
                    capture.release()
                with closing(analyzer.iter_chunks(Path(video.path), interval=video.sample_interval,
                             start=video.last_analyzed_second, stop=self.stop_event.is_set,
                             forced_times=self.anchor_times.get(video.id),
                             progress=lambda second, duration: self.progressed.emit(video.id, second, duration))) as chunks:
                    for chunk in chunks:
                        if analyzer.runtime_description != initial_runtime:
                            self.runtime_changed.emit(analyzer.embedding_model, analyzer.runtime_description)
                            initial_runtime = analyzer.runtime_description
                        self._deliver(video.id, chunk)
                        if self.stop_event.is_set() or chunk.cancelled:
                            break
        except Exception as error:
            self.video_error.emit(video.id, str(error))
        finally:
            self.metadata_ready.emit(video.id, {"elapsed": time.monotonic() - started})


class LibraryController(QObject):
    changed = Signal()
    status_changed = Signal()

    @staticmethod
    def resolve_data_root(explicit: Path | None = None):
        if explicit is not None:
            return explicit
        redirect = LibraryStore._default_root() / "store-location.json"
        return Path(json.loads(redirect.read_text(encoding="utf-8"))["path"]).expanduser() if redirect.is_file() else None

    def __init__(self, root: Path | None = None):
        super().__init__()
        self.store = LibraryStore(root)
        self.thumbnail_dir = self.store.database_path.parent / "Thumbnails"
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.recognition_config_path = self.store.database_path.parent / "recognition.json"
        self.log_path = self.store.database_path.parent / "analysis.log"
        self.logger = logging.getLogger(f"videoatlas.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.log_handler = RotatingFileHandler(self.log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self.log_handler)
        self.package_logger = logging.getLogger("videoatlas")
        self.package_logger.addHandler(self.log_handler)
        self.package_logger.setLevel(logging.INFO)
        self.runtime_description, self.active_embedding_model = "", None
        self.sources, self.videos, self.faces, self.people, self.tracks, self.relations = [], [], [], [], [], []
        self.status_text, self.progress, self.is_scanning = "", 0.0, False
        self.stop_event, self.worker = Event(), None
        self.refresh_after_scan, self.closing, self.scan_failure_message = False, False, None
        self.scan_total = self.scan_done = 0
        self._embedding_only = False
        self._reload()
        for video in self.videos:
            if video.state == "analyzing":
                video.state = "pending"
                self.store.save_video(video)

    def _options(self):
        from .recognition import RecognitionOptions
        return RecognitionOptions.load(self.recognition_config_path if self.recognition_config_path.is_file() else None)

    def _grouping_options(self):
        from .grouping import GroupingOptions
        options = self._options()
        return GroupingOptions(auto_merge_enabled=options.auto_merge_enabled,
            high_threshold=options.high_threshold, medium_threshold=options.medium_threshold,
            minimum_samples=options.min_samples, minimum_quality=options.min_quality,
            match_threshold=options.match_threshold, auxiliary_enabled=options.adaface_enabled,
            maximum_samples=options.max_samples,
            model_thresholds={"face01": options.japanese_face_threshold, "adaface": options.adaface_threshold},
            model_weights={"face01": options.japanese_face_weight, "adaface": options.adaface_weight}
                          if options.adaface_enabled else {"face01": options.japanese_face_weight})

    def _reload(self):
        snapshot = self.store.load_snapshot()
        self.sources, self.videos, self.faces, self.people = snapshot.sources, snapshot.videos, snapshot.faces, snapshot.people
        self.tracks, self.relations = snapshot.tracks, snapshot.relations
        self.changed.emit()

    def _status(self, message, progress=None):
        self.status_text = message
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        self.status_changed.emit()

    @property
    def pending_reviews(self):
        tracks = {track.id: track for track in self.tracks}
        return sorted((r for r in self.relations if r.relation in {"pending", "deferred"}
                       and r.track_a in tracks and r.track_b in tracks
                       and (tracks[r.track_a].person_id != tracks[r.track_b].person_id
                            or tracks[r.track_a].person_id is None)),
                      key=lambda r: (r.relation == "deferred", r.track_a, r.track_b))

    def display_name(self, person):
        return person.name.strip()

    def add_folder(self, path):
        folder = Path(path).expanduser().resolve()
        if not folder.is_dir():
            self._status("Choose a folder containing videos.")
            return
        if not any(Path(source.path) == folder for source in self.sources):
            self.store.save_source(SourceFolder(str(uuid4()), str(folder)))
            self._reload()
        self.refresh_library()

    def refresh_library(self):
        if self.closing:
            return
        if self.is_scanning:
            self.refresh_after_scan = True
            return
        try:
            options = self._options()
        except (OSError, ValueError) as error:
            self._status(f"Cannot load settings: {error}")
            return
        for source in list(self.sources):
            if not source.enabled:
                continue
            try:
                files = list_video_files(Path(source.path), recursive=options.recursive, extensions=options.extensions)
            except Exception as error:
                self._status(f"Cannot check {source.path}: {error}")
                self.logger.exception("Folder discovery failed: %s", source.path)
                continue
            found = {str(path) for path in files}
            for video in self.videos:
                if video.source_id == source.id and video.path not in found:
                    video.state, video.error_message = "missing", "The original video was not found in its registered folder."
                    self.store.save_video(video)
            for path in files:
                try:
                    info = path.stat()
                    current = next((video for video in self.videos if video.path == str(path) and video.state != "missing"), None)
                    if current and current.file_size == info.st_size and current.modification_time == info.st_mtime and current.state != "missing" and current.file_hash:
                        continue
                    digest = file_digest(path)
                    if current is not None and current.file_hash is not None and current.file_hash != digest:
                        # Different bytes are a different video. Preserve historical corrections
                        # under their original ID, never apply them to replacement content.
                        current.state = "missing"
                        current.error_message = "The original video content was replaced. Its reviewed index is retained separately."
                        self.store.save_video(current)
                        current = None
                    if current is None:
                        current = next((video for video in self.videos if video.file_hash == digest and
                                        (video.state == "missing" or not Path(video.path).is_file())), None)
                    if current is None:
                        if any(video.file_hash == digest for video in self.videos):
                            continue
                        current = VideoRecord(str(uuid4()), source.id, str(path), path.name, 0.0, info.st_size,
                                              info.st_mtime, "pending", 0.0, options.sample_interval, None, None, file_hash=digest)
                        self.videos.append(current)
                    else:
                        if current.state == "missing":
                            current.state = "ready" if current.analyzed_at else "pending"
                        current.path, current.name, current.source_id = str(path), path.name, source.id
                        current.file_size, current.modification_time, current.file_hash = info.st_size, info.st_mtime, digest
                        current.error_message = None
                    self.store.save_video(current)
                except Exception as error:
                    self._status(f"Cannot read information for {path.name}: {error}")
                    self.logger.exception("Video registration failed: %s", path)
        self._reload()
        self.resume_scanning()

    def resume_scanning(self):
        self._start_worker([video for video in self.videos if video.state == "pending"])

    def _start_worker(self, videos, cached_faces=None):
        if self.closing or self.is_scanning or not videos:
            return
        self.stop_event.clear()
        self.scan_failure_message = None
        self.scan_total, self.scan_done, self.is_scanning = len(videos), 0, True
        self._embedding_only = cached_faces is not None
        reviewed_ids = {identifier for relation in self.relations
                        if relation.relation in {"confirmed_same", "confirmed_different"}
                        for identifier in (relation.track_a, relation.track_b)}
        anchors = {video.id: sorted({face.second for face in self.faces if face.video_id == video.id
                   and (face.manual_assignment or face.track_id in reviewed_ids)}) for video in videos}
        self.worker = AnalysisWorker(videos, self.stop_event, self.recognition_config_path, cached_faces, anchors)
        for name in ("model_ready", "video_started", "chunk_ready", "progressed", "video_error", "fatal_error", "metadata_ready", "runtime_changed", "finished"):
            handler = {"chunk_ready": self._accept_chunk, "finished": self._scan_finished}.get(name, getattr(self, f"_{name}", None))
            getattr(self.worker, name).connect(handler)
        self._status(f"Checking the analysis model ({len(videos)} candidate videos)", 0.0)
        self.worker.start()

    @Slot(str, str)
    def _model_ready(self, model, runtime):
        try:
            self.active_embedding_model, self.runtime_description = model, runtime
            settings = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self._options()).items()}
            settings["primary_embedding_version"] = model
            settings["runtime"] = runtime
            self.store.save_analysis_settings(self._options().analysis_fingerprint(), settings)
            for worker_video in self.worker.videos:
                video = self.video_by_id(worker_video.id)
                if video:
                    if video.embedding_model != model and not self._embedding_only:
                        worker_video.last_analyzed_second = video.last_analyzed_second = 0.0
                    video.embedding_model = model
                    self.store.save_video(video)
            self.logger.info("Analysis runtime: %s; videos=%d", runtime, self.scan_total)
            self._status(f"{runtime}: analyzing {self.scan_total} videos.")
        except Exception as error:
            self.stop_event.set()
            self.scan_failure_message = f"Cannot save the embedding profile: {error}"
            self._status(self.scan_failure_message)
        finally:
            if self.worker:
                self.worker.model_accepted.set()

    @Slot(str, str)
    def _runtime_changed(self, model, runtime):
        self.active_embedding_model, self.runtime_description = model, runtime
        self.logger.warning("Analysis runtime changed: %s", runtime)
        self.status_changed.emit()

    def set_acceleration(self, value):
        return self.save_settings({"acceleration": value})

    def set_precision(self, value):
        return self.save_settings({"precision": value})

    def save_settings(self, values):
        if self.is_scanning:
            self._status("Pause analysis before changing settings.")
            return False
        temporary = self.recognition_config_path.with_name(f"recognition-{uuid4().hex}.tmp")
        try:
            updates = dict(values)
            folders, new_root = updates.pop("target_folders", None), updates.pop("data_root", None)
            for key in ("cache_dir", "tensorrt_dll_dir", "adaface_model_path", "database_dir"):
                if key in updates and isinstance(updates[key], str) and not updates[key].strip():
                    updates[key] = None
            current = json.loads(self.recognition_config_path.read_text(encoding="utf-8")) if self.recognition_config_path.exists() else {}
            current.update(updates)
            temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            from .recognition import RecognitionOptions
            options = RecognitionOptions.load(temporary)
            folder_paths = [Path(folder).expanduser().resolve() for folder in folders] if folders is not None else []
            if any(not folder.is_dir() for folder in folder_paths):
                raise ValueError("Every target folder must exist")
            temporary.replace(self.recognition_config_path)
            existing = {Path(source.path) for source in self.sources}
            if folders is not None:
                for source in self.sources:
                    source.enabled = Path(source.path) in folder_paths
                    self.store.save_source(source)
            for folder in folder_paths:
                if folder not in existing:
                    self.store.save_source(SourceFolder(str(uuid4()), str(folder)))
            self._reload()
            if new_root and Path(new_root).expanduser().resolve() != self.store.database_path.parent.resolve():
                destination = Path(new_root).expanduser().resolve()
                destination.mkdir(parents=True, exist_ok=True)
                default_root = LibraryStore._default_root()
                default_root.mkdir(parents=True, exist_ok=True)
                (default_root / "store-location.json").write_text(json.dumps({"path": str(destination)}, indent=2), encoding="utf-8")
                (destination / "recognition.json").write_text(self.recognition_config_path.read_text(encoding="utf-8"), encoding="utf-8")
                self._status("Settings saved. Restart to open the selected library directory. The current library remains at its existing path.")
            else:
                self._status("Settings saved. Choose Reanalyze all or Recalculate embeddings to update existing results.")
            self.logger.setLevel(options.log_level)
            self.package_logger.setLevel(options.log_level)
            self.runtime_description, self.active_embedding_model = "", None
            return True
        except (OSError, ValueError, TypeError) as error:
            self._status(f"Cannot save settings: {error}")
            return False
        finally:
            temporary.unlink(missing_ok=True)

    def pause_scanning(self):
        if self.is_scanning:
            self.stop_event.set()
            self._status("Pausing and saving partial results…")

    @Slot(str)
    def _video_started(self, video_id):
        video = self.video_by_id(video_id)
        if video:
            video.state = "analyzing"
            if not self._save_video(video):
                return
            self.logger.info("Video analysis started: id=%s path=%s", video.id, video.path)
            self._status(f"Analyzing: {video.name}")
            self.changed.emit()

    @Slot(str, object)
    def _metadata_ready(self, video_id, metadata):
        video = self.video_by_id(video_id)
        if video:
            for key in ("width", "height", "fps"):
                if key in metadata:
                    setattr(video, key, metadata[key])
            if "elapsed" in metadata:
                video.analysis_seconds = (video.analysis_seconds or 0.0) + metadata["elapsed"]
            self._save_video(video)

    @Slot(str, float, float)
    def _progressed(self, video_id, second, duration):
        fraction = (self.scan_done + max(0.0, min(1.0, second / max(duration, 1.0)))) / max(self.scan_total, 1)
        self._status(self.status_text, fraction)

    @Slot(str, object)
    def _accept_chunk(self, video_id, chunk):
        try:
            from .grouping import classify_analysis
            video = self.video_by_id(video_id)
            if video is None:
                return
            previous_persons = {track.id: track.person_id for track in self.tracks}
            result = classify_analysis(chunk.faces, replace(video, last_analyzed_second=0.0) if self._embedding_only else video,
                self.faces, self.people, self.thumbnail_dir, chunk.completed_second, chunk.duration,
                existing_tracks=self.tracks, relations=self.relations, options=self._grouping_options())
            video = replace(video, duration=chunk.duration, last_analyzed_second=chunk.completed_second,
                state="ready" if not chunk.cancelled and chunk.completed_second >= chunk.duration else "pending", error_message=None)
            video.embedding_model = self.active_embedding_model
            if video.state == "ready":
                video.analyzed_at, video.analysis_version = utc_now(), self._options().analysis_fingerprint()
            if video.poster_path is None and chunk.poster_jpeg:
                poster = self.thumbnail_dir / f"poster-{video.id}.jpg"
                poster.write_bytes(chunk.poster_jpeg)
                video.poster_path = str(poster)
            self.store.apply_face_batch(result.faces, video, result.people, result.removed_ids, tracks=result.tracks, relations=result.relations)
            self.logger.info("Analysis saved: video=%s faces=%d tracks=%d embeddings=%d reassigned_tracks=%d pending=%d completed=%.3f",
                video.id, len(result.faces), len(result.tracks), sum(len(face.embeddings) for face in result.faces),
                sum(t.id in previous_persons and t.person_id != previous_persons[t.id] for t in result.tracks),
                sum(r.relation == "pending" for r in result.relations), chunk.completed_second)
            self._reload()
            if video.state == "ready":
                self.scan_done += 1
        except Exception as error:
            self.logger.exception("Cannot persist analysis: %s", video_id)
            self.stop_event.set()
            self.scan_failure_message = f"Cannot save analysis results: {error}"
            self._video_error(video_id, self.scan_failure_message)
        finally:
            if self.worker:
                self.worker.chunk_saved.set()

    @Slot(str, str)
    def _video_error(self, video_id, message):
        video = self.video_by_id(video_id)
        if video:
            video.state, video.error_message = "failed", message
            self._save_video(video)
            self.scan_done += 1
        self.logger.error("Video analysis failed: %s: %s", video_id, message)
        self._status(f"Analysis failed: {video.name if video else video_id}: {message}")
        self.changed.emit()

    @Slot(str)
    def _fatal_error(self, message):
        self.scan_failure_message = message
        self.logger.error("Analysis stopped: %s", message)
        self._status(f"Cannot start analysis: {message}")

    def _save_video(self, video):
        try:
            self.store.save_video(video)
            return True
        except Exception as error:
            self.stop_event.set()
            self.scan_failure_message = f"Cannot save video state: {error}"
            self.logger.exception("Cannot save video state: %s", video.id)
            self._status(self.scan_failure_message)
            return False

    @Slot()
    def _scan_finished(self):
        was_paused = self.stop_event.is_set()
        for video in self.videos:
            if video.state == "analyzing":
                video.state = "pending"
                self.store.save_video(video)
        finished_worker, self.worker = self.worker, None
        if finished_worker and not self.closing:
            finished_worker.deleteLater()
        self.is_scanning = False
        self._reload()
        if self.scan_failure_message:
            self._status(self.scan_failure_message)
        elif was_paused:
            self._status("Paused.")
        else:
            failed = sum(video.state == "failed" for video in self.videos)
            self._status(f"Analysis finished. {failed} failed videos; {len(self.pending_reviews)} pairs await review.", 1.0)
        if self.refresh_after_scan and not self.closing:
            self.refresh_after_scan = False
            self.refresh_library()

    def video_by_id(self, video_id):
        return next((video for video in self.videos if video.id == video_id), None)

    def face_by_id(self, face_id):
        return next((face for face in self.faces if face.id == face_id), None)

    def rename_person(self, person_id, name):
        if self.is_scanning:
            return
        person = next((item for item in self.people if item.id == person_id), None)
        if person:
            person.name, person.updated_at = name.strip(), utc_now()
            self.store.save_person(person)
            self._reload()

    def merge_people(self, source_id, target_id):
        if not self.is_scanning and source_id != target_id:
            try:
                self.store.merge_people(source_id, target_id)
                self.logger.info("Person merge: person_merge_count=1 source=%s target=%s user_confirmed=True", source_id, target_id)
                self._reload()
            except (ValueError, KeyError) as error:
                self._status(f"Cannot merge people: {error}")

    def split_face(self, face_id):
        face = self.face_by_id(face_id)
        if face and face.track_id:
            self.split_track(face.track_id)
        elif face and not self.is_scanning:
            person = PersonRecord(str(uuid4()), "", face.thumbnail_path)
            self.store.save_person(person)
            face.person_id, face.manual_assignment = person.id, True
            self.store.save_face(face)
            self._reload()

    def split_track(self, track_id):
        if not self.is_scanning:
            try:
                self.store.split_track(track_id)
                self._reload()
            except (ValueError, KeyError) as error:
                self._status(f"Cannot split track: {error}")

    def assign_face(self, face_id, person_id):
        if self.is_scanning:
            return
        face = self.face_by_id(face_id)
        if face:
            try:
                if face.track_id:
                    self.store.assign_track(face.track_id, person_id)
                else:
                    face.person_id, face.manual_assignment, face.excluded = person_id, True, False
                    self.store.save_face(face)
                self._reload()
            except (ValueError, KeyError) as error:
                self._status(f"Cannot assign face: {error}")

    def exclude_face(self, face_id):
        if not self.is_scanning:
            self.store.exclude_face(face_id)
            self._reload()

    def set_representative(self, face_id):
        if not self.is_scanning:
            self.store.set_representative(face_id)
            self._reload()

    def resolve_review(self, track_a, track_b, decision):
        if self.is_scanning:
            self._status("Pause analysis before reviewing people.")
            return
        try:
            self.store.resolve_relation(track_a, track_b, decision)
            self._reload()
        except (ValueError, KeyError) as error:
            self._status(f"Cannot save review: {error}")

    def rescan_video(self, video_id, detailed=True):
        if self.is_scanning:
            return
        video = self.video_by_id(video_id)
        if video and Path(video.path).is_file():
            video.sample_interval = min(0.2, self._options().sample_interval) if detailed else self._options().sample_interval
            video.last_analyzed_second, video.analysis_seconds = 0.0, 0.0
            video.state, video.error_message = "pending", None
            self.store.save_video(video)
            self.changed.emit()
            self.resume_scanning()

    def reanalyze_all(self):
        if self.is_scanning:
            return
        options = self._options()
        for video in self.videos:
            if Path(video.path).is_file():
                video.state, video.last_analyzed_second, video.analysis_seconds = "pending", 0.0, 0.0
                video.sample_interval, video.error_message = options.sample_interval, None
                self.store.save_video(video)
        self._reload()
        self.resume_scanning()

    def recalculate_embeddings(self, video_id):
        if self.is_scanning:
            return
        video = self.video_by_id(video_id)
        faces = [face for face in self.faces if face.video_id == video_id]
        if video is None or not faces:
            self._status("No cached faces are available. Analyze this video first.")
            return
        if any((face.quality or 0.0) >= self._options().min_quality and not face.aligned_path for face in faces):
            self._status("Aligned crops are missing. Reanalyze this video to create its cache.")
            return
        self._start_worker([video], faces)

    def evaluation_report(self):
        from .grouping import compare_tracks
        from .evaluation import evaluate_pairs, optimize_thresholds
        tracks = {track.id: track for track in self.tracks}
        examples = []
        model_examples = []
        policy_results = []
        grouping_options = self._grouping_options()
        evidence_options = replace(grouping_options, auxiliary_enabled=True)
        for relation in self.relations:
            if relation.relation in {"confirmed_same", "confirmed_different"} and relation.track_a in tracks and relation.track_b in tracks:
                result = compare_tracks(tracks[relation.track_a], tracks[relation.track_b], grouping_options)
                examples.append({"same": relation.relation == "confirmed_same",
                                 "scores": {key: value["median"] for key, value in result["models"].items()}})
                evidence = compare_tracks(tracks[relation.track_a], tracks[relation.track_b], evidence_options)
                model_examples.append({"same": relation.relation == "confirmed_same",
                                       "scores": {key: value["median"] for key, value in evidence["models"].items()}})
                policy_results.append({"same": relation.relation == "confirmed_same",
                                       "score": 1.0 if result["decision"] == "HIGH" else 0.0})
        active_keys = {key for example in examples for key in example["scores"]}
        options = self._options()
        weights = {key: options.adaface_weight if key.startswith("adaface:") else options.japanese_face_weight
                   for key in active_keys}
        individual_models = {key: evaluate_pairs(model_examples,
            threshold=options.adaface_threshold if key.startswith("adaface:") else options.japanese_face_threshold,
            weights={key: 1.0}) for key in sorted({key for example in model_examples for key in example["scores"]})}
        return {"evaluation": evaluate_pairs(examples, threshold=options.high_threshold, weights=weights),
                "individual_models": individual_models,
                "prediction_rule": "Configured weighted model median cosine similarity >= HIGH threshold; model versions require complete compatible evidence.",
                "high_decision_policy": evaluate_pairs(policy_results, threshold=.5),
                "policy_note": "HIGH track-pair eligibility includes quality, sample count, disagreement, overlapping tracks, and unresolved reviewed-history guards. Cluster merges also enforce negative relations and all member comparisons.",
                "optimization": optimize_thresholds(model_examples),
                "note": "Track pairs from reviewed data only. Tuning results describe this labeled sample and are not independent accuracy validation."}

    def regroup_people(self):
        if self.is_scanning:
            self._status("Pause analysis before regrouping people.")
            return
        from .grouping import recluster_tracks
        result = recluster_tracks(self.faces, self.tracks, self.people, self.relations, self._grouping_options())
        self.store.apply_identity_result(result.faces, result.people, result.tracks, result.relations)
        self._reload()
        self._status(f"Regrouped cached tracks. {len(self.pending_reviews)} pairs await review.")

    def remove_source(self, source_id):
        if not self.is_scanning:
            for video in self.videos:
                if video.source_id == source_id:
                    self._remove_video_images(video.id)
            self.store.delete_source(source_id)
            self._reload()

    def clear_index(self):
        if not self.is_scanning:
            self.store.clear_index()
            for image in self.thumbnail_dir.glob("*.jpg"):
                image.unlink(missing_ok=True)
            self._reload()
            self._status("The index and generated images were deleted. Original videos remain.", 0.0)

    def _remove_video_images(self, video_id):
        paths = [Path(face.thumbnail_path) for face in self.faces if face.video_id == video_id]
        video = self.video_by_id(video_id)
        if video and video.poster_path:
            paths.append(Path(video.poster_path))
        for path in paths:
            if path.parent.resolve() == self.thumbnail_dir.resolve() and path.suffix.lower() == ".jpg":
                path.unlink(missing_ok=True)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.stop_event.set()
        if self.worker:
            while self.worker.isRunning():
                QCoreApplication.processEvents()
                self.worker.wait(50)
            QCoreApplication.processEvents()
        self.store.close()
        self.logger.removeHandler(self.log_handler)
        self.package_logger.removeHandler(self.log_handler)
        self.log_handler.close()
