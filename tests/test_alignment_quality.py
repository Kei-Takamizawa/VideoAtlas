import numpy as np
import pytest
from videoatlas.model_adapters import FACE01_TEMPLATE, align_crop, similarity_transform, landmark_visibility_proxy


def test_primary_alignment_uses_mouth_corners_and_records_new_geometry():
    y, x = np.mgrid[:300, :300]
    frame = np.stack((x % 255, y % 255, (x+y) % 255), axis=-1).astype(np.uint8)
    points = FACE01_TEMPLATE.copy()
    points = points * 1.1 + 20
    crop, residual, support = align_crop(frame, points, "face01")
    assert crop.shape == (224, 224, 3) and residual < 1e-6 and support > .99
    perturbed = points.copy()
    perturbed[3:, 1] += 15
    changed, changed_residual, _ = align_crop(frame, perturbed, "face01")
    assert not np.array_equal(crop, changed)
    assert changed_residual > residual


def test_similarity_transform_rejects_degenerate_landmarks():
    with pytest.raises(ValueError, match="Degenerate"):
        similarity_transform(np.zeros((5, 2)), FACE01_TEMPLATE)


def test_opaque_landmark_cover_lowers_the_visibility_proxy():
    rng = np.random.default_rng(14)
    frame = rng.integers(50, 200, (224, 224, 3), dtype=np.uint8)
    uncovered = landmark_visibility_proxy(frame, FACE01_TEMPLATE)
    covered = frame.copy()
    for x, y in FACE01_TEMPLATE[3:]:
        covered[int(y)-20:int(y)+21, int(x)-20:int(x)+21] = 0
    hidden = landmark_visibility_proxy(covered, FACE01_TEMPLATE)
    assert 0 <= hidden < uncovered <= 1
    assert hidden < .7
    assert landmark_visibility_proxy(frame, np.full((5, 2), -100)) == 0


def test_roll_does_not_make_valid_mouth_geometry_fail():
    from videoatlas.analyzer import VideoAnalyzer
    from videoatlas.recognition import RecognitionOptions
    analyzer = object.__new__(VideoAnalyzer)
    analyzer.options = RecognitionOptions(acceleration="cpu")
    rng = np.random.default_rng(17)
    frame = rng.integers(30, 220, (512, 512, 3), dtype=np.uint8)
    points = FACE01_TEMPLATE + 144
    angle = np.pi / 3
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = (points-256) @ rotation.T + 256
    primary, auxiliary, quality, reason, pose = analyzer._prepare_face(frame, points, (60, 60, 452, 452), .99)
    assert reason is None and primary is not None and auxiliary is not None
    assert quality >= analyzer.options.min_quality and abs(pose["roll"]-60) < 1
