"""Detector-only size probe for the offline masked-face cohort."""
from __future__ import annotations

import json
from pathlib import Path

from videoatlas.model_adapters import SCRFDDetector
from videoatlas.recognition import RecognitionOptions
from videoatlas.storage import LibraryStore
from Scripts.evaluate_masked_faces import (
    ARTIFACTS,
    ROOT,
    _open_frame,
    _pick_samples,
    _target_detection,
    mask_target_lower_face,
)


OUTPUT = ARTIFACTS / "mask-multiscale-probe.json"
SOURCE_MODEL = ROOT / "videoatlas" / "resources" / "scrfd_10g_kps.onnx"
SIZES = (320, 640, 960)


def run():
    store = LibraryStore(ARTIFACTS / "real-library-gpu")
    snapshot = store.load_snapshot()
    store.close()
    face_map = {face.id: face for face in snapshot.faces}
    track_map = {track.id: track for track in snapshot.tracks}
    videos = {video.id: video for video in snapshot.videos}
    manifest = json.loads((ARTIFACTS / "real-track-evaluation.json").read_text(encoding="utf-8"))
    cohort = []
    for row in manifest["selected_tracks"]:
        video = videos[row["video_id"]]
        track = track_map[row["track_id"]]
        cohort.append({"video_name": row["video_name"], "path": Path(video.path),
                       "samples": _pick_samples(track, face_map)})

    cache_dir = ARTIFACTS / "masked-detector-probe-cache"
    options = RecognitionOptions(acceleration="cuda", detection_max_dimension=640,
                                 detection_threshold=.6, recognition_region="upper",
                                 upper_multiscale_detection=True, adaface_enabled=False,
                                 cache_dir=cache_dir, detection_model_path=SOURCE_MODEL)
    # Reuse the app's provenance-checked metadata-only derivation, then disable
    # its automatic size pyramid while this script measures each size alone.
    derived_model = SCRFDDetector._dynamic_output_model(SOURCE_MODEL, options)
    options = RecognitionOptions(acceleration="cuda", detection_max_dimension=640,
                                 detection_threshold=.6, recognition_region="upper",
                                 upper_multiscale_detection=False, adaface_enabled=False,
                                 cache_dir=cache_dir, detection_model_path=derived_model)
    detector = SCRFDDetector(options)
    original_size = detector.size
    input_shape = detector.runner.input.shape
    if not (len(input_shape) == 4 and all(value in (None, "?") or isinstance(value, str) for value in input_shape[2:])):
        raise RuntimeError(f"SCRFD input is not dynamic; cannot safely change model input size: {input_shape}")

    outcomes = {size: set() for size in SIZES}
    sample_rows = []
    total = sum(len(video["samples"]) for video in cohort)
    try:
        for video_index, video in enumerate(cohort):
            for sample_index, sample in enumerate(video["samples"]):
                frame = _open_frame(video["path"], sample.frame_number)
                height, width = frame.shape[:2]
                nbox = sample.bounding_box
                box = (int(nbox[0] * width), int(nbox[1] * height),
                       int((nbox[0] + nbox[2]) * width), int((nbox[1] + nbox[3]) * height))
                detector.size = (640, 640)
                clean = _target_detection(detector.detect(frame), box)
                row = {"cohort_index": video_index, "sample_index": sample_index,
                       "clean_640_match": clean is not None, "masked_match_by_size": {str(size): False for size in SIZES}}
                if clean is not None:
                    masked, _hidden = mask_target_lower_face(frame, clean.box, clean.landmarks[:2])
                    for size in SIZES:
                        detector.size = (size, size)
                        candidate = _target_detection(detector.detect(masked), clean.box, clean.landmarks[:2])
                        hit = candidate is not None
                        row["masked_match_by_size"][str(size)] = hit
                        if hit:
                            outcomes[size].add((video_index, sample_index))
                sample_rows.append(row)
    finally:
        detector.size = original_size
        close = getattr(detector.runner, "close", None)
        if callable(close):
            close()

    union_hits = set().union(*outcomes.values())
    denominator = total
    result = {
        "purpose": "Evaluate recall recovery from SCRFD input scale alone after the same opaque lower-face mask, without recognition inference or threshold changes.",
        "requested_sizes": list(SIZES),
        "detection_threshold": .6,
        "model_input_shape": list(input_shape),
        "source_model": str(SOURCE_MODEL),
        "temporary_probe_model": str(derived_model),
        "temporary_model_change": "Used the app's provenance-checked copy, changing only output proposal-count metadata; original model weights and graph nodes were preserved.",
        "original_detector_size": list(original_size),
        "selected_videos": len(cohort),
        "target_samples": denominator,
        "clean_target_matches_at_640": sum(row["clean_640_match"] for row in sample_rows),
        "masked_hits": {str(size): {"hits": len(outcomes[size]), "denominator": denominator,
                                    "recall": len(outcomes[size]) / max(denominator, 1)} for size in SIZES},
        "union_across_sizes": {"hits": len(union_hits), "denominator": denominator,
                               "recall": len(union_hits) / max(denominator, 1),
                               "additional_over_640": len(union_hits - outcomes[640])},
        "marginal_hits_only_at_size": {str(size): len(outcomes[size] - set().union(*(outcomes[other] for other in SIZES if other != size))) for size in SIZES},
        "limitations": ["The size probe is detector recall only; no recognition embeddings or identity scores were computed.",
                        "Targets are the same 56 selected face samples used by the preceding ablation.",
                        "A detection counts only when it overlaps the saved target box and matches its eye center."],
        "samples": sample_rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    run()
