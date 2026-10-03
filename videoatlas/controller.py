from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
from threading import Event
from uuid import uuid4
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal, Slot
from .grouping import classify_faces
from .storage import FaceRecord, LibraryStore, PersonRecord, SourceFolder, VideoRecord, list_video_files


class AnalysisWorker(QThread):
    video_started = Signal(str)
    chunk_ready = Signal(str, object)
    progressed = Signal(str, float, float)
    video_error = Signal(str, str)
    fatal_error = Signal(str)
    model_ready = Signal(str, str)

    def __init__(self, videos: list[VideoRecord], stop_event: Event, config_path: Path) -> None:
        super().__init__()
        self.videos = [replace(video) for video in videos]
        self.stop_event = stop_event
        self.config_path = config_path
        self.chunk_saved = Event()
        self.model_accepted = Event()

    def run(self) -> None:
        try:
            from .analyzer import VideoAnalyzer
            from .recognition import RecognitionOptions
            analyzer = VideoAnalyzer(options=RecognitionOptions.load(self.config_path if self.config_path.is_file() else None))
        except Exception as error:
            self.fatal_error.emit(str(error))
            return
        try:
            self.model_ready.emit(analyzer.embedding_model, analyzer.runtime_description)
            while not self.model_accepted.wait(0.1):
                if self.stop_event.is_set():
                    return
            self._run_videos(analyzer)
        except Exception as error:
            self.fatal_error.emit(str(error))
        finally:
            try:
                analyzer.close()
            except Exception as error:
                self.fatal_error.emit(f"Cannot release analysis resources: {error}")

    def _run_videos(self, analyzer: object) -> None:
        for video in self.videos:
            if self.stop_event.is_set():
                break
            if video.state == "ready" and video.embedding_model == analyzer.embedding_model:
                continue
            if video.state not in {"pending", "ready"}:
                continue
            if video.embedding_model != analyzer.embedding_model:
                video.last_analyzed_second = 0.0
                video.embedding_model = analyzer.embedding_model
            self.video_started.emit(video.id)
            try:
                with closing(analyzer.iter_chunks(Path(video.path), interval=video.sample_interval, start=video.last_analyzed_second, stop=self.stop_event.is_set, progress=lambda second, duration: self.progressed.emit(video.id, second, duration))) as chunks:
                    for chunk in chunks:
                        self.chunk_saved.clear()
                        self.chunk_ready.emit(video.id, chunk)
                        self.chunk_saved.wait()
                        if self.stop_event.is_set() or chunk.cancelled:
                            break
            except Exception as error:
                self.video_error.emit(video.id, str(error))


class LibraryController(QObject):
    changed = Signal()
    status_changed = Signal()

    def __init__(self, root: Path | None = None) -> None:
        super().__init__()
        self.store = LibraryStore(root)
        self.thumbnail_dir = self.store.database_path.parent / "Thumbnails"
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.recognition_config_path = self.store.database_path.parent / "recognition.json"
        self.runtime_description = ""
        self.active_embedding_model: str | None = None
        self.sources: list[SourceFolder] = []
        self.videos: list[VideoRecord] = []
        self.faces: list[FaceRecord] = []
        self.people: list[PersonRecord] = []
        self.status_text = ""
        self.progress = 0.0
        self.is_scanning = False
        self.stop_event = Event()
        self.worker: AnalysisWorker | None = None
        self.refresh_after_scan = False
        self.closing = False
        self.scan_failure_message: str | None = None
        self._reload()
        for video in self.videos:
            if video.state == "analyzing":
                video.state = "pending"
                self.store.save_video(video)

    def _reload(self) -> None:
        snapshot = self.store.load_snapshot()
        self.sources = snapshot.sources
        self.videos = snapshot.videos
        self.faces = snapshot.faces
        self.people = snapshot.people
        self.changed.emit()

    def _status(self, message: str, progress: float | None = None) -> None:
        self.status_text = message
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        self.status_changed.emit()

    def add_folder(self, path: Path) -> None:
        folder = Path(path).expanduser().resolve()
        if not folder.is_dir():
            self._status("Choose a folder containing videos.")
            return
        if not any(source.path == str(folder) for source in self.sources):
            source = SourceFolder(id=str(uuid4()), path=str(folder))
            self.store.save_source(source)
            self._reload()
        self.refresh_library()

    def refresh_library(self) -> None:
        if self.closing:
            return
        if self.is_scanning:
            self.refresh_after_scan = True
            return
        for source in list(self.sources):
            try:
                files = list_video_files(Path(source.path))
            except Exception as error:
                self._status(f"Cannot check {source.path}: {error}")
                continue
            found = {str(path) for path in files}
            for video in self.videos:
                if video.source_id != source.id:
                    continue
                if video.path not in found:
                    video.state = "missing"
                    video.error_message = "The original video was not found in its registered folder."
                    self.store.save_video(video)
            for path in files:
                try:
                    info = path.stat()
                except OSError as error:
                    self._status(f"Cannot read information for {path.name}: {error}")
                    continue
                current = next((video for video in self.videos if video.source_id == source.id and video.path == str(path)), None)
                if current is None:
                    current = VideoRecord(str(uuid4()), source.id, str(path), path.name, 0.0, info.st_size, info.st_mtime, "pending", 0.0, 2.0, None, None)
                    self.store.save_video(current)
                    self.videos.append(current)
                    continue
                changed = current.file_size != info.st_size or current.modification_time != info.st_mtime
                if changed:
                    current.last_analyzed_second = 0.0
                    current.duration = 0.0
                    current.poster_path = None
                if changed or current.state == "missing":
                    current.file_size = info.st_size
                    current.modification_time = info.st_mtime
                    current.state = "pending"
                    current.error_message = None
                    self.store.save_video(current)
        self._reload()
        self._remove_orphan_people()
        self.resume_scanning()

    def resume_scanning(self) -> None:
        if self.closing:
            return
        if self.is_scanning:
            return
        pending = [video for video in self.videos if video.state in {"pending", "ready"}]
        if not pending:
            return
        self.stop_event.clear()
        self.scan_failure_message = None
        self.scan_total = len(pending)
        self.scan_done = 0
        self.is_scanning = True
        self.worker = AnalysisWorker(pending, self.stop_event, self.recognition_config_path)
        self.worker.model_ready.connect(self._model_ready)
        self.worker.video_started.connect(self._video_started)
        self.worker.chunk_ready.connect(self._accept_chunk)
        self.worker.progressed.connect(self._progressed)
        self.worker.video_error.connect(self._video_error)
        self.worker.fatal_error.connect(self._fatal_error)
        self.worker.finished.connect(self._scan_finished)
        self._status(f"Checking the analysis model ({len(pending)} candidate videos)", 0.0)
        self.worker.start()

    @Slot(str, str)
    def _model_ready(self, model: str, runtime: str) -> None:
        try:
            self.active_embedding_model = model
            self.runtime_description = runtime
            count = 0
            migration_count = 0
            for video in self.videos:
                if video.state not in {"pending", "ready"}:
                    continue
                migration = video.embedding_model != model
                if video.state == "pending" or migration:
                    count += 1
                if migration:
                    migration_count += 1
                    updated = replace(video, last_analyzed_second=0.0, state="pending", embedding_model=model)
                    self.store.save_video(updated)
                    video.last_analyzed_second = 0.0
                    video.state = "pending"
                    video.embedding_model = model
            self.scan_total = count
            message = f"{runtime}: analyzing {count} videos."
            if migration_count:
                message += f" Restarting {migration_count} videos with a different embedding profile."
            self._status(message, 0.0 if count else 1.0)
            self.changed.emit()
        except Exception as error:
            self.stop_event.set()
            self.scan_failure_message = f"Cannot save the embedding profile: {error}"
            self._status(self.scan_failure_message)
        finally:
            if self.worker is not None:
                self.worker.model_accepted.set()

    def set_acceleration(self, acceleration: str) -> bool:
        if self.is_scanning:
            self._status("Pause analysis before changing acceleration settings.")
            return False
        if acceleration not in {"auto", "cuda", "tensorrt", "cpu"}:
            return False
        return self._save_recognition_setting("acceleration", acceleration)

    def set_precision(self, precision: str) -> bool:
        if self.is_scanning:
            self._status("Pause analysis before changing recognition settings.")
            return False
        if precision not in {"accurate", "balanced"}:
            return False
        return self._save_recognition_setting("precision", precision)

    def _save_recognition_setting(self, name: str, value: str) -> bool:
        temporary = self.recognition_config_path.with_name(f"recognition-{uuid4().hex}.tmp")
        try:
            values = json.loads(self.recognition_config_path.read_text(encoding="utf-8")) if self.recognition_config_path.exists() else {}
            if not isinstance(values, dict):
                raise ValueError("Recognition settings must be a JSON object")
            values[name] = value
            temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            from .recognition import RecognitionOptions
            RecognitionOptions.load(temporary)
            temporary.replace(self.recognition_config_path)
        except (OSError, ValueError, TypeError) as error:
            self._status(f"Cannot save recognition settings: {error}")
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        self.runtime_description = ""
        self.active_embedding_model = None
        self._status("Saved face analysis settings. They apply to the next analysis.")
        return True

    def pause_scanning(self) -> None:
        if not self.is_scanning:
            return
        self.stop_event.set()
        self._status("Pausing and saving partial results…")

    @Slot(str)
    def _video_started(self, video_id: str) -> None:
        video = self.video_by_id(video_id)
        if video is None:
            return
        updated = replace(video, state="analyzing")
        try:
            self.store.save_video(updated)
        except Exception as error:
            self.stop_event.set()
            self.scan_failure_message = f"Cannot save video state: {error}"
            self._status(self.scan_failure_message)
            return
        video.state = "analyzing"
        self._status(f"Analyzing: {video.name}")
        self.changed.emit()

    @Slot(str, float, float)
    def _progressed(self, video_id: str, second: float, duration: float) -> None:
        part = max(0.0, min(1.0, second / max(duration, 1.0)))
        fraction = (self.scan_done + part) / max(self.scan_total, 1)
        self._status(self.status_text, fraction)

    @Slot(str, object)
    def _accept_chunk(self, video_id: str, chunk: object) -> None:
        try:
            video = self.video_by_id(video_id)
            if video is None:
                return
            faces, people, removed_ids = classify_faces(chunk.faces, video, self.faces, self.people, self.thumbnail_dir, chunk.completed_second, chunk.duration)
            video = replace(video)
            video.duration = chunk.duration
            video.last_analyzed_second = chunk.completed_second
            video.state = "ready" if not chunk.cancelled and chunk.completed_second >= chunk.duration else "pending"
            video.error_message = None
            if video.poster_path is None and chunk.poster_jpeg:
                poster = self.thumbnail_dir / f"poster-{video.id}.jpg"
                poster.write_bytes(chunk.poster_jpeg)
                video.poster_path = str(poster)
            self.store.apply_face_batch(faces, video, people, removed_ids)
            self._reload()
            if video.state == "ready":
                self.scan_done += 1
        except Exception as error:
            self.stop_event.set()
            self.scan_failure_message = f"Cannot save analysis results: {error}"
            self._status(self.scan_failure_message)
        finally:
            if self.worker is not None:
                self.worker.chunk_saved.set()

    @Slot(str, str)
    def _video_error(self, video_id: str, message: str) -> None:
        video = self.video_by_id(video_id)
        if video is None:
            self._status(message)
            return
        updated = replace(video, state="failed", error_message=message)
        try:
            self.store.save_video(updated)
        except Exception as error:
            self.stop_event.set()
            self.scan_failure_message = f"Cannot save video state: {error}"
            self._status(self.scan_failure_message)
            return
        video.state = "failed"
        video.error_message = message
        self.scan_done += 1
        self._status(f"Analysis failed: {video.name}: {message}")
        self.changed.emit()

    @Slot(str)
    def _fatal_error(self, message: str) -> None:
        self.scan_failure_message = message
        self._status(f"Cannot start analysis: {message}")

    @Slot()
    def _scan_finished(self) -> None:
        was_paused = self.stop_event.is_set()
        try:
            for video in self.videos:
                if video.state == "analyzing":
                    self.store.save_video(replace(video, state="pending"))
                    video.state = "pending"
        except Exception as error:
            self.scan_failure_message = f"Cannot save interrupted video state: {error}"
        finished_worker = self.worker
        self.worker = None
        if finished_worker is not None and not self.closing:
            finished_worker.deleteLater()
        self.is_scanning = False
        if self.scan_failure_message is not None:
            self._status(self.scan_failure_message)
        elif was_paused:
            self._status("Paused.")
        elif not any(video.state == "pending" for video in self.videos):
            try:
                self._remove_orphan_people()
            except Exception as error:
                self.scan_failure_message = f"Cannot update the people index: {error}"
            failed_count = sum(video.state == "failed" for video in self.videos)
            message = f"Analysis finished. Reasons for {failed_count} failed videos are shown on their cards." if failed_count else "Analysis complete."
            self._status(self.scan_failure_message or message, 1.0)
        else:
            self._status("Analysis stopped. Check the status area.")
        self.changed.emit()
        if self.refresh_after_scan and not self.closing:
            self.refresh_after_scan = False
            self.refresh_library()

    def video_by_id(self, video_id: str) -> VideoRecord | None:
        return next((video for video in self.videos if video.id == video_id), None)

    def face_by_id(self, face_id: str) -> FaceRecord | None:
        return next((face for face in self.faces if face.id == face_id), None)

    def rename_person(self, person_id: str, name: str) -> None:
        if self.is_scanning:
            self._status("Pause analysis before editing people.")
            return
        person = next((item for item in self.people if item.id == person_id), None)
        if person is None or not name.strip():
            return
        person.name = name.strip()
        self.store.save_person(person)
        self.changed.emit()

    def merge_people(self, source_id: str, target_id: str) -> None:
        if self.is_scanning or source_id == target_id:
            return
        if not all(any(person.id == identifier for person in self.people) for identifier in (source_id, target_id)):
            return
        for face in self.faces:
            if face.person_id == source_id:
                face.person_id = target_id
                face.manual_assignment = True
                self.store.save_face(face)
        self.store.delete_person(source_id)
        self._reload()

    def split_face(self, face_id: str) -> None:
        if self.is_scanning:
            self._status("Pause analysis before editing faces.")
            return
        face = self.face_by_id(face_id)
        if face is None:
            return
        person = PersonRecord(str(uuid4()), "", face.thumbnail_path)
        self.store.save_person(person)
        face.person_id = person.id
        face.manual_assignment = True
        self.store.save_face(face)
        self._reload()

    def assign_face(self, face_id: str, person_id: str | None) -> None:
        if self.is_scanning:
            self._status("Pause analysis before editing faces.")
            return
        face = self.face_by_id(face_id)
        target_exists = person_id is None or any(person.id == person_id for person in self.people)
        if face is None or not target_exists:
            return
        face.person_id = person_id
        face.manual_assignment = True
        face.excluded = False
        self.store.save_face(face)
        self._reload()

    def exclude_face(self, face_id: str) -> None:
        if self.is_scanning:
            self._status("Pause analysis before editing faces.")
            return
        face = self.face_by_id(face_id)
        if face is None:
            return
        face.excluded = True
        face.manual_assignment = True
        self.store.save_face(face)
        self._reload()

    def rescan_video(self, video_id: str, detailed: bool = True) -> None:
        if self.is_scanning:
            return
        video = self.video_by_id(video_id)
        if video is None or not Path(video.path).is_file():
            return
        video.sample_interval = 0.5 if detailed else 2.0
        video.last_analyzed_second = 0.0
        video.state = "pending"
        video.error_message = None
        self.store.save_video(video)
        self.changed.emit()
        self.resume_scanning()

    def remove_source(self, source_id: str) -> None:
        if self.is_scanning:
            self._status("Pause analysis before removing a folder.")
            return
        video_ids = [video.id for video in self.videos if video.source_id == source_id]
        for video_id in video_ids:
            self._remove_video_images(video_id)
        self.store.delete_source(source_id)
        self._reload()
        self._remove_orphan_people()

    def clear_index(self) -> None:
        if self.is_scanning:
            self._status("Pause analysis before clearing the index.")
            return
        self.store.clear_index()
        for image in self.thumbnail_dir.glob("*.jpg"):
            image.unlink(missing_ok=True)
        self._reload()
        self._status("The index and generated images were deleted. Original videos remain.", 0.0)

    def _remove_video_images(self, video_id: str) -> None:
        paths = [Path(face.thumbnail_path) for face in self.faces if face.video_id == video_id]
        video = self.video_by_id(video_id)
        if video is not None and video.poster_path:
            paths.append(Path(video.poster_path))
        for path in paths:
            if path.parent == self.thumbnail_dir and path.suffix.lower() == ".jpg":
                path.unlink(missing_ok=True)

    def _remove_orphan_people(self) -> None:
        active_ids = {face.person_id for face in self.faces if face.person_id is not None}
        for person in self.people:
            if person.id not in active_ids and not person.name.strip():
                self.store.delete_person(person.id)
        self._reload()

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.stop_event.set()
        running = self.worker
        if running is not None:
            while running.isRunning():
                QCoreApplication.processEvents()
                running.wait(50)
            QCoreApplication.processEvents()
        self.store.close()
