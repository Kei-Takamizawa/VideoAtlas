"""Input ablation contracts, not masked-person accuracy measurements."""
from dataclasses import replace
import numpy as np
import pytest
from videoatlas.analyzer import VideoAnalyzer, FaceSample
from videoatlas.model_adapters import (FACE01_TEMPLATE, align_crop,
    upper_visible_mask, preprocessing_identity)
from videoatlas.recognition import RecognitionOptions
from videoatlas.tracking import pose_bucket


@pytest.mark.parametrize("model", ["face01", "adaface"])
def test_hidden_pixels_and_lower_landmarks_cannot_change_upper_input(model):
    rng = np.random.default_rng(91)
    frame = rng.integers(40,210,(400,400,3),dtype=np.uint8)
    points = FACE01_TEMPLATE + 80
    visible = upper_visible_mask(frame.shape, points[:2])
    changed = frame.copy()
    changed[~visible] = rng.integers(0,255,(np.count_nonzero(~visible),3),dtype=np.uint8)
    altered_points = points.copy()
    altered_points[2:] = np.nan
    original, _, _ = align_crop(frame, points, model, "upper")
    altered, _, _ = align_crop(changed, altered_points, model, "upper")
    assert np.array_equal(original, altered)
    assert preprocessing_identity(model, "full") != preprocessing_identity(model, "upper")


def test_upper_quality_ignores_hidden_pixels_and_missing_lower_geometry():
    analyzer = object.__new__(VideoAnalyzer)
    analyzer.options = RecognitionOptions(recognition_region="upper", acceleration="cpu")
    frame = np.random.default_rng(17).integers(30,220,(512,512,3),dtype=np.uint8)
    points = FACE01_TEMPLATE + 144
    visible = upper_visible_mask(frame.shape, points[:2])
    altered = frame.copy(); altered[~visible] = 0
    altered_points = points.copy(); altered_points[2:] = np.nan
    before = analyzer._prepare_face(frame,points,(60,60,452,452),.99)
    after = analyzer._prepare_face(altered,altered_points,(60,60,452,452),.99)
    assert before[3] is None and after[3] is None
    assert np.array_equal(before[0],after[0])
    assert before[2] == after[2]
    # Eye patch visibility must not read the hidden half either.
    assert before[4]["yaw"] is None and pose_bucket(before[4]) == "unknown"


def test_region_changes_analysis_identity_and_is_validated():
    options = RecognitionOptions()
    assert options.analysis_fingerprint() != replace(options,recognition_region="upper").analysis_fingerprint()
    with pytest.raises(ValueError,match="recognition_region"):
        replace(options,recognition_region="lower").validate()
    with pytest.raises(ValueError,match="not calibrated"):
        replace(options,recognition_region="upper",auto_merge_enabled=True).validate()


def test_full_face_cache_cannot_be_recalculated_as_upper_face(tmp_path):
    analyzer = object.__new__(VideoAnalyzer)
    analyzer.options = RecognitionOptions(cache_dir=tmp_path)
    analyzer._closed = False
    path = analyzer._cache_crops(np.full((224,224,3),128,np.uint8),np.full((112,112,3),128,np.uint8))
    sample = FaceSample(0, (0,0,1,1), b"", None, None, 1, None, aligned_path=path)
    analyzer.options = replace(analyzer.options,recognition_region="upper")
    with pytest.raises(ValueError,match="preprocessing"):
        analyzer._recalculate_samples([sample])
