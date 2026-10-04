"""Optional genuine-model smoke tests; synthetic videos are not accuracy data."""
import json
import os
from pathlib import Path
import time

import cv2
import pytest
from PySide6.QtWidgets import QApplication

from videoatlas.controller import LibraryController
from videoatlas.storage import VideoRecord

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
RESOURCES = Path(__file__).resolve().parents[1] / "videoatlas" / "resources"


def await_idle(model, app, timeout=45):
    deadline = time.monotonic() + timeout
    while model.is_scanning and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert not model.is_scanning, model.status_text


def test_genuine_models_video_to_review_and_cached_recalculation(tmp_path):
    sample = RESOURCES / "model-smoke.jpg"
    auxiliary = RESOURCES / "adaface_ir50_ms1mv2.onnx"
    required = [sample, auxiliary, RESOURCES / "JAPANESE_FACE_V1.onnx", RESOURCES / "scrfd_10g_kps.onnx"]
    if not all(path.is_file() for path in required):
        pytest.skip("Optional model smoke needs official locally provisioned models and model-smoke.jpg")
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "videos"
    folder.mkdir()
    image = cv2.imread(str(sample))
    height, width = image.shape[:2]
    for name in ("a", "b"):
        # Change one pixel outside faces to give each video a distinct content identity.
        frame = image.copy()
        frame[0, 0] = 0 if name == "a" else 255
        writer = cv2.VideoWriter(str(folder / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
        assert writer.isOpened()
        for _ in range(15):
            writer.write(frame)
        writer.release()
    model = LibraryController(tmp_path / "index")
    try:
        assert model.save_settings({"acceleration": "cpu", "adaface_enabled": True,
                                    "adaface_model_path": str(auxiliary), "auto_merge_enabled": False})
        model.add_folder(folder)
        await_idle(model, app)
        assert len(model.videos) == 2
        assert all(video.state == "ready" for video in model.videos), model.status_text
        accepted = [face for face in model.faces if face.embedding]
        assert accepted and all(len(face.embeddings) == 2 for face in accepted)
        assert all(len(vector) == 512 for face in accepted for vector in face.embeddings.values())
        assert all(face.frame_number is not None and face.track_id for face in accepted)
        assert all(video.file_hash and video.width and video.height and video.fps and video.analysis_version for video in model.videos)
        assert model.store.load_analysis_settings()
        assert model.pending_reviews
        review = model.pending_reviews[0]
        model.resolve_review(review.track_a, review.track_b, "confirmed_same")
        tracks = {track.id: track for track in model.tracks}
        assert tracks[review.track_a].person_id == tracks[review.track_b].person_id
        decision = next(r for r in model.relations if (r.track_a, r.track_b) == (review.track_a, review.track_b))
        assert decision.relation == "confirmed_same"
        ids_before = {face.id for face in model.faces}
        model.recalculate_embeddings(model.videos[0].id)
        await_idle(model, app)
        assert all(video.state == "ready" for video in model.videos), model.status_text
        assert {face.id for face in model.faces} == ids_before
        assert any(r.relation == "confirmed_same" for r in model.relations)
        assert "faces=" in model.log_path.read_text(encoding="utf-8")
        model.regroup_people()
        tracks = {track.id: track for track in model.tracks}
        assert tracks[review.track_a].person_id == tracks[review.track_b].person_id
        model.resolve_review(review.track_a, review.track_b, "confirmed_different")
        assert model.save_settings({"auto_merge_enabled": True})
        model.rescan_video(model.videos[0].id, detailed=True)
        await_idle(model, app)
        assert model.videos[0].state == "ready", model.status_text
        tracks = {track.id: track for track in model.tracks}
        assert tracks[review.track_a].person_id != tracks[review.track_b].person_id
        label = next(r for r in model.relations if (r.track_a, r.track_b) == (review.track_a, review.track_b))
        assert label.relation == "confirmed_different"
        assert any(entry["relation"] == "confirmed_same" for entry in label.details["history"])
    finally:
        model.close()


def test_video_player_decodes_and_seeks_to_appearance(tmp_path):
    import numpy as np
    from PySide6.QtMultimedia import QMediaPlayer
    from videoatlas.ui import PlayerDialog
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "playback.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((96, 96, 3), index * 10, dtype=np.uint8))
    writer.release()
    model = LibraryController(tmp_path / "playback-index")
    video = VideoRecord("video", "source", str(path), path.name, 2.0, path.stat().st_size,
                        path.stat().st_mtime, "ready", 2.0, .5, None, None)
    dialog = PlayerDialog(model, video, .5)
    frames = []
    errors = []
    dialog.video_widget.videoSink().videoFrameChanged.connect(lambda frame: frames.append(frame.isValid()))
    dialog.player.errorOccurred.connect(lambda error, message: errors.append(message))
    try:
        deadline = time.monotonic() + 10
        while dialog.player.mediaStatus() != QMediaPlayer.MediaStatus.LoadedMedia and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert not errors, errors
        assert dialog.initial_position == 0
        assert dialog.player.position() == 500
        dialog._toggle_playback()
        while not any(frames) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert any(frames) and not errors
        dialog._toggle_playback()
        assert dialog.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
    finally:
        dialog.player.stop()
        dialog.close()
        app.processEvents()
        model.close()
