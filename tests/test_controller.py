import json
import os
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from videoatlas.controller import AnalysisWorker, LibraryController
from videoatlas.storage import SourceFolder, RelationRecord, TrackRecord


@pytest.fixture
def controller(tmp_path):
    app = QApplication.instance() or QApplication([])
    model = LibraryController(tmp_path / "index")
    model.resume_scanning = lambda: None
    yield model
    model.close()


def register(model, folder):
    folder.mkdir()
    model.store.save_source(SourceFolder("source", str(folder)))
    model._reload()


def test_renaming_preserves_analysis_identity(controller, tmp_path):
    folder = tmp_path / "videos"
    register(controller, folder)
    path = folder / "old.mp4"
    path.write_bytes(b"one unchanged video")
    controller.refresh_library()
    video = controller.videos[0]
    original_id = video.id
    video.state, video.analyzed_at = "ready", "2026-10-04T00:00:00Z"
    video.last_analyzed_second, video.duration = 10.0, 10.0
    controller.store.save_video(video)
    path.rename(folder / "new.mp4")
    controller.refresh_library()
    assert len(controller.videos) == 1
    assert controller.videos[0].id == original_id
    assert controller.videos[0].name == "new.mp4"
    assert controller.videos[0].state == "ready"
    assert controller.videos[0].last_analyzed_second == 10.0


def test_changed_content_restarts_but_mtime_only_does_not(controller, tmp_path):
    folder = tmp_path / "videos"
    register(controller, folder)
    path = folder / "video.mkv"
    path.write_bytes(b"first video")
    controller.refresh_library()
    video = controller.videos[0]
    video.state, video.analyzed_at, video.last_analyzed_second = "ready", "date", 10.0
    video.modification_time -= 1
    controller.store.save_video(video)
    controller.refresh_library()
    assert controller.videos[0].state == "ready"
    path.write_bytes(b"new different content")
    controller.refresh_library()
    assert len(controller.videos) == 2
    assert controller.videos[0].state == "missing"
    assert controller.videos[1].state == "pending"
    assert controller.videos[1].last_analyzed_second == 0.0
    assert controller.videos[0].id != controller.videos[1].id
    controller.refresh_library()
    assert len(controller.videos) == 2


def test_settings_can_disable_folder_without_erasing_reviewed_index(controller, tmp_path):
    folder = tmp_path / "videos"
    register(controller, folder)
    (folder / "a.mp4").write_bytes(b"one video")
    controller.refresh_library()
    assert controller.save_settings({"target_folders": []})
    assert not controller.sources[0].enabled
    (folder / "b.mp4").write_bytes(b"new video")
    controller.refresh_library()
    assert len(controller.videos) == 1


def test_duplicate_content_shares_analysis(controller, tmp_path):
    folder = tmp_path / "videos"
    register(controller, folder)
    (folder / "a.mp4").write_bytes(b"same bytes")
    (folder / "b.mov").write_bytes(b"same bytes")
    controller.refresh_library()
    assert len(controller.videos) == 1


def test_invalid_settings_do_not_replace_saved_settings(controller):
    assert controller.save_settings({"high_threshold": .95, "medium_threshold": .70})
    before = controller.recognition_config_path.read_bytes()
    assert not controller.save_settings({"high_threshold": .40})
    assert controller.recognition_config_path.read_bytes() == before


def test_model_thresholds_persist_and_control_grouping(controller):
    assert controller.save_settings({"japanese_face_threshold": .91, "adaface_threshold": .86})
    assert controller._grouping_options().model_thresholds == {"face01": .91, "adaface": .86}
    before = controller.recognition_config_path.read_bytes()
    assert not controller.save_settings({"adaface_threshold": 1.1})
    assert controller.recognition_config_path.read_bytes() == before


def test_ready_videos_do_not_require_loading_models(controller):
    received = []
    controller._start_worker = lambda videos: received.extend(videos)
    controller.videos = [SimpleNamespace(state="ready"), SimpleNamespace(state="pending")]
    LibraryController.resume_scanning(controller)
    assert len(received) == 1 and received[0].state == "pending"


def test_missing_model_marks_every_affected_video_failed(controller, tmp_path, monkeypatch):
    from videoatlas import analyzer
    folder = tmp_path / "videos"
    register(controller, folder)
    (folder / "a.mp4").write_bytes(b"one")
    (folder / "b.mp4").write_bytes(b"two")
    controller.refresh_library()
    monkeypatch.setattr(analyzer, "VideoAnalyzer", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Missing model")))
    worker = AnalysisWorker(controller.videos, controller.stop_event, controller.recognition_config_path)
    worker.video_error.connect(controller._video_error)
    worker.run()
    assert {v.state for v in controller.videos} == {"failed"}
    assert all(v.error_message == "Missing model" for v in controller.videos)
    assert controller.store.load_snapshot().videos[0].state == "failed"


def test_empty_reviews_have_no_invented_accuracy(controller):
    report = controller.evaluation_report()
    assert report["evaluation"]["sample_count"] == 0
    assert report["evaluation"]["precision"] is None
    assert report["evaluation"]["false_positive_rate"] is None
    assert report["optimization"]["best_candidate"] is None


def test_pending_reviews_exclude_retired_endpoints_and_same_person(controller):
    controller.tracks = [TrackRecord("a", "video", 0, 1, "p1"),
                         TrackRecord("b", "video", 0, 1, "p2"),
                         TrackRecord("c", "video", 0, 1, "p1")]
    controller.relations = [RelationRecord("a", "b", "pending"),
                            RelationRecord("a", "retired", "pending"),
                            RelationRecord("a", "c", "deferred")]
    assert [(r.track_a, r.track_b) for r in controller.pending_reviews] == [("a", "b")]


def test_evaluation_uses_configured_weights_and_separate_high_guards(controller):
    assert controller.save_settings({"adaface_enabled": True, "adaface_model_path": "adaface.onnx"})
    left = TrackRecord("a", "v1", 0, 1, quality=.9,
                       embedding_samples={"face01:v1": [[1., 0.]] * 3, "adaface:v1": [[1., 0.]] * 3})
    right = TrackRecord("b", "v2", 0, 1, quality=.9,
                        embedding_samples={"face01:v1": [[.9, math.sqrt(1 - .9**2)]] * 3,
                                           "adaface:v1": [[.4, math.sqrt(1 - .4**2)]] * 3})
    controller.tracks = [left, right]
    controller.relations = [RelationRecord("a", "b", "confirmed_same")]
    assert controller.evaluation_report()["evaluation"]["fn"] == 1
    assert controller.save_settings({"japanese_face_weight": 9.0})
    controller.tracks = [left, right]
    controller.relations = [RelationRecord("a", "b", "confirmed_same")]
    report = controller.evaluation_report()
    assert report["evaluation"]["tp"] == 1
    assert report["high_decision_policy"]["fn"] == 1  # Disagreement still prevents HIGH.
    assert report["individual_models"]["face01:v1"]["tp"] == 1
    assert report["individual_models"]["adaface:v1"]["fn"] == 1
    assert controller.save_settings({"adaface_enabled": False})
    controller.tracks = [left, right]
    controller.relations = [RelationRecord("a", "b", "confirmed_same")]
    report = controller.evaluation_report()
    assert report["individual_models"]["adaface:v1"]["sample_count"] == 1
    assert any(candidate["weights"] == {"adaface:v1": 1.0} for candidate in report["optimization"]["candidates"])
