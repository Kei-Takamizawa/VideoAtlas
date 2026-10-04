import numpy as np
import pytest

from Scripts.evaluate_masked_faces import mask_target_lower_face, upper_context_descriptor
from videoatlas.model_adapters import align_crop, preprocessing_identity, upper_visible_mask


def test_opaque_lower_face_mask_covers_nose_mouth_jaw_and_stays_in_target_box():
    frame = np.full((240, 320, 3), 210, dtype=np.uint8)
    eyes = np.asarray([[145, 85], [175, 85]], dtype=np.float32)
    box = (115, 55, 205, 175)
    masked, hidden = mask_target_lower_face(frame, box, eyes)

    assert hidden[130, 160]
    assert hidden[165, 115]
    assert not hidden[75, 160]
    assert not hidden[180, 20]
    np.testing.assert_array_equal(masked[hidden], np.tile([128, 128, 128], (hidden.sum(), 1)))
    np.testing.assert_array_equal(masked[~hidden], frame[~hidden])


def test_upper_alignment_is_invariant_to_hidden_pixels_and_lower_landmarks():
    rng = np.random.default_rng(917)
    frame = rng.integers(10, 245, size=(300, 300, 3), dtype=np.uint8)
    points = np.asarray([[145, 105], [195, 105], [170, 138], [150, 170], [190, 170]], dtype=np.float32)
    visible = upper_visible_mask(frame.shape, points[:2])
    changed_hidden = frame.copy()
    changed_hidden[~visible] = [12, 231, 99]
    lower_corrupted = points.copy()
    lower_corrupted[2:] = np.nan

    crop_a, residual_a, support_a = align_crop(frame, points, "face01", "upper")
    crop_b, residual_b, support_b = align_crop(changed_hidden, lower_corrupted, "face01", "upper")

    np.testing.assert_array_equal(crop_a, crop_b)
    assert residual_a == pytest.approx(residual_b, abs=1e-12)
    assert residual_a < 1e-6
    assert support_a == pytest.approx(support_b, abs=1e-12)


def test_model_preprocessing_identity_separates_full_and_upper_regions():
    assert preprocessing_identity("face01", "full") != preprocessing_identity("face01", "upper")
    assert preprocessing_identity("adaface", "full") != preprocessing_identity("adaface", "upper")


def test_upper_context_descriptor_cannot_read_hidden_lower_face_pixels():
    rng = np.random.default_rng(918)
    frame = rng.integers(10, 245, size=(300, 300, 3), dtype=np.uint8)
    eyes = np.asarray([[145, 105], [195, 105]], dtype=np.float32)
    visible = upper_visible_mask(frame.shape, eyes)
    changed = frame.copy()
    changed[~visible] = [5, 250, 70]

    baseline = upper_context_descriptor(frame, eyes)
    perturbed = upper_context_descriptor(changed, eyes)

    np.testing.assert_array_equal(baseline["appearance"], perturbed["appearance"])
    assert baseline["shooting_geometry"] == perturbed["shooting_geometry"]
