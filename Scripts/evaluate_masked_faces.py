"""Offline upper-face masking ablation on the manually selected video tracks.

Filename labels are read only after all frame inference is complete. This file
must remain an offline evaluator; none of its labels or scores enter the app.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import re
import statistics

import cv2
import numpy as np

from videoatlas.analyzer import VideoAnalyzer
from videoatlas.evaluation import evaluate_pairs, optimize_thresholds
from videoatlas.model_adapters import (
    adaface_tensor,
    face01_tensor,
    upper_visible_mask,
)
from videoatlas.recognition import RecognitionOptions
from videoatlas.storage import LibraryStore


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".test-artifacts"
SNAPSHOT = ARTIFACTS / "real-library-gpu"
GALLERY = ARTIFACTS / "real-track-gallery.json"
OUTPUT = ARTIFACTS / "masked-face-ablation.json"
CONTACT_SHEET = ARTIFACTS / "masked-face-ablation.png"
ADAFACE_MODEL = ROOT / "videoatlas" / "resources" / "adaface_ir50_ms1mv2.onnx"
MAX_SAMPLES_PER_TRACK = 3
MASK_BGR = np.asarray([128, 128, 128], dtype=np.uint8)


def mask_target_lower_face(frame: np.ndarray, box: tuple[int, int, int, int], eyes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Opaque-mask the selected target's nose/lower face inside an expanded box."""
    visible = upper_visible_mask(frame.shape, eyes)
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    pad_x = max(2, int(round((x1 - x0) * .08)))
    pad_y = max(2, int(round((y1 - y0) * .06)))
    x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
    target = np.zeros((height, width), dtype=bool)
    target[y0:y1, x0:x1] = True
    hidden = target & ~visible
    masked = frame.copy()
    masked[hidden] = MASK_BGR
    return masked, hidden


def upper_context_descriptor(frame: np.ndarray, eyes: np.ndarray) -> dict:
    """Visible upper-area colour/texture and scale cues, separate from identity embeddings."""
    eyes = np.asarray(eyes, dtype=np.float32)
    axis = eyes[1] - eyes[0]
    eye_gap = float(np.linalg.norm(axis))
    axis /= max(eye_gap, 1e-12)
    down = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    center = eyes.mean(axis=0)
    height, width = frame.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    dx, dy = xx - center[0], yy - center[1]
    horizontal = dx * axis[0] + dy * axis[1]
    vertical = dx * down[0] + dy * down[1]
    visible = upper_visible_mask(frame.shape, eyes)
    roi = (np.abs(horizontal) <= 1.45 * eye_gap) & (vertical >= -1.6 * eye_gap) & (vertical <= .12 * eye_gap)
    mask = visible & roi
    # Keep Sobel neighborhoods away from the synthetic boundary so texture
    # cannot encode pixels from the hidden lower-face area.
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not np.any(mask):
        raise ValueError("Selected face has no visible upper-region pixels")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Hue and saturation capture broad colour; grayscale gradients capture
    # coarse hair/skin texture without including the hidden lower face.
    colour = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [12, 4], [0, 180, 0, 256]).reshape(-1)
    colour = colour.astype(np.float32)
    colour /= max(float(np.linalg.norm(colour)), 1e-12)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    texture = cv2.calcHist([np.clip(mag, 0, 255).astype(np.uint8)], [0], mask.astype(np.uint8), [8], [0, 256]).reshape(-1).astype(np.float32)
    texture /= max(float(np.linalg.norm(texture)), 1e-12)
    return {
        "appearance": np.concatenate((colour, texture)).tolist(),
        "shooting_geometry": {
            "eye_gap_frame_diagonal": eye_gap / math.hypot(width, height),
        },
    }


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def _target_detection(detections, stored_box, stored_eyes=None):
    if not detections:
        return None
    if stored_eyes is None:
        best = max(detections, key=lambda item: _iou(item.box, stored_box))
        return best if _iou(best.box, stored_box) >= .10 else None
    reference = np.asarray(stored_eyes, dtype=np.float32)
    ref_gap = max(float(np.linalg.norm(reference[1] - reference[0])), 1.0)
    def rank(item):
        candidate = np.asarray(item.landmarks[:2], dtype=np.float32)
        eye_error = float(np.linalg.norm(candidate.mean(axis=0) - reference.mean(axis=0))) / ref_gap
        return eye_error, -_iou(item.box, stored_box), -item.score
    candidate = min(detections, key=rank)
    candidate_eyes = np.asarray(candidate.landmarks[:2], dtype=np.float32)
    if np.linalg.norm(candidate_eyes.mean(axis=0) - reference.mean(axis=0)) / ref_gap > .4 or _iou(candidate.box, stored_box) < .10:
        return None
    return candidate


def _pick_samples(track, faces_by_id):
    candidates = [faces_by_id[identifier] for identifier in (track.sample_face_ids or track.face_ids)
                  if identifier in faces_by_id and faces_by_id[identifier].frame_number is not None
                  and faces_by_id[identifier].bounding_box is not None]
    candidates.sort(key=lambda face: (-(face.quality or 0.0), face.second, face.id))
    chosen = []
    target_gap = min(1.0, max(.15, (track.end_time - track.start_time) / (MAX_SAMPLES_PER_TRACK + 1)))
    for face in candidates:
        if all(abs(face.second - previous.second) >= target_gap for previous in chosen):
            chosen.append(face)
        if len(chosen) == MAX_SAMPLES_PER_TRACK:
            break
    return sorted(chosen, key=lambda face: (face.second, face.id))


def _open_frame(path: Path, frame_number: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open benchmark video: {path}")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number):
        capture.release()
        raise RuntimeError(f"Cannot seek to benchmark frame {frame_number}: {path}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"Cannot read benchmark frame {frame_number}: {path}")
    return frame


def _infer(analyzer, frame, detection):
    crop, auxiliary, quality, reason, _pose = analyzer._prepare_face(frame, detection.landmarks, detection.box, detection.score)
    result = {"quality": quality, "reason": reason, "crop": crop, "auxiliary": auxiliary,
              "face01": None, "adaface": None}
    if crop is None:
        return result
    vectors = analyzer._infer_batch([face01_tensor(crop)])
    if vectors is None or len(vectors) != 1:
        raise RuntimeError("FACE01 inference returned no embedding")
    result["face01"] = vectors[0]
    result["adaface"] = analyzer._adaface.embed(auxiliary)
    return result


def _cosine(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else None


def _median_pairwise(left, right):
    scores = [_cosine(a, b) for a in left for b in right]
    scores = [score for score in scores if score is not None]
    return float(statistics.median(scores)) if scores else None


def _geometry_score(left, right):
    ratios = []
    for key in left.keys() & right.keys():
        a, b = float(left[key]), float(right[key])
        if a > 0 and b > 0:
            ratios.append(math.exp(-abs(math.log(a / b))))
    return float(statistics.mean(ratios)) if ratios else None


def _threshold_report(examples):
    train = [item for item in examples if not ("4.MP4" in item["left"] or "4.MP4" in item["right"])]
    heldout = [item for item in examples if item not in train]
    candidate_weights = [{key: 1.0} for key in sorted({key for item in train for key in item["scores"]})]
    families = sorted({key for item in train for key in item["scores"]})
    if {"face01", "adaface"}.issubset(families):
        candidate_weights += [{"face01": .5, "adaface": .5}]
        if {"face01", "adaface", "upper_context", "shooting_geometry"}.issubset(families):
            candidate_weights += [{"face01": .35, "adaface": .35, "upper_context": .20, "shooting_geometry": .10}]
    thresholds = [value / 100 for value in range(20, 101)] + [1.000001]
    training = optimize_thresholds(train, thresholds, candidate_weights)
    best = training["best_candidate"]
    holdout = evaluate_pairs(heldout, best["threshold"], best["weights"]) if best else None
    zero_fp_candidates = [candidate for candidate in training["candidates"]
                          if candidate["fp"] == 0 and candidate["sample_count"] == len(train)]
    zero_fp = max(zero_fp_candidates, key=lambda candidate: (candidate["recall"] or 0, candidate["threshold"])) if zero_fp_candidates else None
    return {
        "training": {key: value for key, value in training.items() if key != "candidates"},
        "training_candidate_count": len(training["candidates"]),
        "training_candidate_reports": training["candidates"],
        "heldout_at_training_selected_candidate": holdout,
        "heldout_at_training_selected_zero_fp_candidate": evaluate_pairs(heldout, zero_fp["threshold"], zero_fp["weights"]) if zero_fp else None,
        "training_zero_fp_candidate": zero_fp,
        "fixed_balanced_face_embedding_ensemble": {
            "weights": {"face01": .5, "adaface": .5},
            "training_at_0_8": evaluate_pairs(train, .8, {"face01": .5, "adaface": .5}),
            "heldout_at_0_8": evaluate_pairs(heldout, .8, {"face01": .5, "adaface": .5}),
        },
    }


def _split_examples(examples):
    train = [item for item in examples if not ("4.MP4" in item["left"] or "4.MP4" in item["right"])]
    heldout = [item for item in examples if item not in train]
    return train, heldout


def _single_family_report(examples, family):
    complete = [{"same": item["same"], "left": item["left"], "right": item["right"],
                 "scores": {family: item["scores"][family]}}
                for item in examples if family in item["scores"]]
    train, heldout = _split_examples(complete)
    thresholds = [value / 100 for value in range(20, 101)] + [1.000001]
    tuning = optimize_thresholds(train, thresholds, [{family: 1.0}])
    zero_fp = [candidate for candidate in tuning["candidates"]
               if candidate["fp"] == 0 and candidate["sample_count"] == len(train)]
    selected = max(zero_fp, key=lambda candidate: (candidate["recall"] or 0, candidate["threshold"])) if zero_fp else None
    return {
        "family": family,
        "total_complete_pairs": len(complete),
        "training_eligible_pairs": len(train),
        "heldout_eligible_pairs": len(heldout),
        "training_excluded_missing_or_masked_detection": 105 - len(train),
        "heldout_excluded_missing_or_masked_detection": 85 - len(heldout),
        "training_selected_zero_fp_candidate": _compact_candidate(selected),
        "heldout_at_training_selected_zero_fp_candidate": _compact_pair_metrics(evaluate_pairs(heldout, selected["threshold"], {family: 1.0})) if selected else None,
        "at_fixed_0_8": _compact_pair_metrics(evaluate_pairs(examples, .8, {family: 1.0})),
        "heldout_auc": evaluate_pairs(heldout, .5, {family: 1.0})["auc"],
    }


def _embedding_ensemble_report(examples):
    complete = [item for item in examples if all(key in item["scores"] for key in ("face01", "adaface"))]
    train, heldout = _split_examples(complete)
    weights = {"face01": .5, "adaface": .5}
    thresholds = [value / 100 for value in range(20, 101)] + [1.000001]
    tuning = optimize_thresholds(train, thresholds, [weights])
    zero_fp = [candidate for candidate in tuning["candidates"]
               if candidate["fp"] == 0 and candidate["sample_count"] == len(train)]
    selected = max(zero_fp, key=lambda candidate: (candidate["recall"] or 0, candidate["threshold"])) if zero_fp else None
    return {
        "weights": weights,
        "total_complete_pairs": len(complete),
        "training_eligible_pairs": len(train),
        "heldout_eligible_pairs": len(heldout),
        "training_excluded_missing_or_masked_detection": 105 - len(train),
        "heldout_excluded_missing_or_masked_detection": 85 - len(heldout),
        "training_selected_zero_fp_candidate": _compact_candidate(selected),
        "heldout_at_training_selected_zero_fp_candidate": _compact_pair_metrics(evaluate_pairs(heldout, selected["threshold"], weights)) if selected else None,
    }


def _context_candidate_report(examples):
    weights = {"face01": .35, "adaface": .35, "upper_context": .20, "shooting_geometry": .10}
    complete = [item for item in examples if all(key in item["scores"] for key in weights)]
    train, heldout = _split_examples(complete)
    thresholds = [value / 100 for value in range(20, 101)] + [1.000001]
    tuning = optimize_thresholds(train, thresholds, [weights])
    zero_fp = [candidate for candidate in tuning["candidates"]
               if candidate["fp"] == 0 and candidate["sample_count"] == len(train)]
    selected = max(zero_fp, key=lambda candidate: (candidate["recall"] or 0, candidate["threshold"])) if zero_fp else None
    return {
        "weights": weights,
        "total_complete_pairs": len(complete),
        "training_eligible_pairs": len(train),
        "heldout_eligible_pairs": len(heldout),
        "training_excluded_missing_or_masked_detection": 105 - len(train),
        "heldout_excluded_missing_or_masked_detection": 85 - len(heldout),
        "training_selected_zero_fp_candidate": _compact_candidate(selected),
        "heldout_at_training_selected_zero_fp_candidate": _compact_pair_metrics(evaluate_pairs(heldout, selected["threshold"], weights)) if selected else None,
        "warning": "Exploratory candidate includes upper appearance and shooting geometry; neither cue is a stable biometric.",
    }


def _score_distribution(examples):
    keys = sorted({key for item in examples for key in item["scores"]})
    result = {}
    for key in keys:
        result[key] = {}
        for label, rows in (("same", [item for item in examples if item["same"]]),
                            ("different", [item for item in examples if not item["same"]])):
            values = [item["scores"][key] for item in rows if key in item["scores"]]
            result[key][label] = {"pair_count": len(values), "median": statistics.median(values) if values else None,
                                  "minimum": min(values) if values else None, "maximum": max(values) if values else None}
    return result


def _compact_pair_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "roc"}


def _compact_candidate(candidate):
    if candidate is None:
        return None
    fields = ("threshold", "weights", "sample_count", "excluded_missing_evidence", "positive_count", "negative_count",
              "tp", "fp", "tn", "fn", "precision", "recall", "false_positive_rate", "false_discovery_fraction",
              "false_positive_rate_denominator", "false_discovery_denominator", "auc", "abstain_only", "deployable_threshold")
    return {key: candidate[key] for key in fields if key in candidate}


def run():
    if not SNAPSHOT.is_dir() or not GALLERY.is_file() or not ADAFACE_MODEL.is_file():
        raise FileNotFoundError("Required private benchmark artifacts or AdaFace ONNX export are missing")
    store = LibraryStore(SNAPSHOT)
    snapshot = store.load_snapshot()
    store.close()
    face_map = {face.id: face for face in snapshot.faces}
    track_map = {track.id: track for track in snapshot.tracks}
    videos = {video.id: video for video in snapshot.videos}
    if not (ARTIFACTS / "real-track-evaluation.json").is_file():
        raise FileNotFoundError("The previously reviewed offline cohort manifest is missing")
    prior = json.loads((ARTIFACTS / "real-track-evaluation.json").read_text(encoding="utf-8"))
    # The prior visual audit fixes the cohort membership (including the
    # companion exclusion) without a filename exception in this script.
    selected = prior["selected_tracks"]
    cohort = []
    for index, row in enumerate(selected):
        track = track_map[row["track_id"]]
        video = videos[row["video_id"]]
        path = Path(video.path)
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark video is missing: {path}")
        cohort.append({"index": index, "track": track, "video": video, "path": path,
                       "samples": _pick_samples(track, face_map), "video_name": row["video_name"]})
    if len(cohort) != 20:
        raise RuntimeError(f"Expected 20 manually selected tracks, got {len(cohort)}")

    options = dict(acceleration="cuda", precision="accurate", adaface_enabled=True,
                   adaface_model_path=ADAFACE_MODEL, min_quality=.65,
                   cache_dir=ARTIFACTS / "masked-face-cache", auto_merge_enabled=False)
    analyzers = {}
    try:
        for region in ("full", "upper"):
            analyzer = VideoAnalyzer(RecognitionOptions(recognition_region=region, **options))
            analyzers[region] = analyzer
            print(f"Loaded {region} inference providers: {analyzer.runtime_description}")

        rows = []
        contact_items = []
        detector_attempts = {"clean": 0, "masked": 0}
        detector_hits = {"clean": 0, "masked": 0}
        quality_attempts = defaultdict(int)
        quality_accepted = defaultdict(int)
        embedding_counts = defaultdict(int)
        for video in cohort:
            for sample in video["samples"]:
                frame = _open_frame(video["path"], sample.frame_number)
                stored_box_norm = sample.bounding_box
                h, w = frame.shape[:2]
                stored_box = (int(stored_box_norm[0] * w), int(stored_box_norm[1] * h),
                              int((stored_box_norm[0] + stored_box_norm[2]) * w),
                              int((stored_box_norm[1] + stored_box_norm[3]) * h))
                clean_detection = _target_detection(analyzers["full"]._detector.detect(frame), stored_box)
                detector_attempts["clean"] += 1
                clean_hit = clean_detection is not None
                detector_hits["clean"] += int(clean_hit)
                row = {"cohort_index": video["index"], "frame_number": sample.frame_number,
                       "second": sample.second, "clean_detector_hit": clean_hit,
                       "masked_detector_hit": False, "modes": {}}
                if clean_detection is None:
                    rows.append(row)
                    continue
                eyes = np.asarray(clean_detection.landmarks[:2], dtype=np.float32)
                masked_frame, hidden = mask_target_lower_face(frame, clean_detection.box, eyes)
                masked_detection = _target_detection(analyzers["upper"]._detector.detect(masked_frame), clean_detection.box, eyes)
                detector_attempts["masked"] += 1
                masked_hit = masked_detection is not None
                detector_hits["masked"] += int(masked_hit)
                row["masked_detector_hit"] = masked_hit
                clean_full = _infer(analyzers["full"], frame, clean_detection)
                clean_upper = _infer(analyzers["upper"], frame, clean_detection)
                modes = {"full_clean": clean_full, "upper_clean": clean_upper}
                if masked_detection is not None:
                    modes["upper_masked"] = _infer(analyzers["upper"], masked_frame, masked_detection)
                for mode, result in modes.items():
                    quality_attempts[mode] += 1
                    accepted = result["crop"] is not None
                    quality_accepted[mode] += int(accepted)
                    if accepted:
                        embedding_counts[mode + ":face01"] += 1
                        embedding_counts[mode + ":adaface"] += int(result["adaface"] is not None)
                        context_detection = masked_detection if mode == "upper_masked" else clean_detection
                        context_frame = masked_frame if mode == "upper_masked" else frame
                        context_eyes = np.asarray(context_detection.landmarks[:2], dtype=np.float32)
                        context = upper_context_descriptor(context_frame, context_eyes)
                        result["context"] = context
                        crop = result["crop"]
                        if mode == "full_clean":
                            crop = result["crop"]
                        thumb = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                        contact_items.append((video["index"], mode, thumb))
                    row["modes"][mode] = {"accepted": accepted, "quality": result["quality"],
                                           "reason": result["reason"], "face01": result["face01"],
                                           "adaface": result["adaface"], "context": result.get("context")}
                rows.append(row)

        # Human-provided filename labels are applied only after all model
        # inference is complete. The model calls above receive no names/labels.
        for video in cohort:
            video["subject"] = re.sub(r"\d+$", "", Path(video["video_name"]).stem)
        names = {video["index"]: video["video_name"] for video in cohort}
        labels = {video["index"]: video["subject"] for video in cohort}
        modes = ("full_clean", "upper_clean", "upper_masked")
        model_families = ("face01", "adaface")
        pair_rows = {mode: [] for mode in modes}
        raw_similarity = {mode: [] for mode in modes}
        for left, right in combinations(cohort, 2):
            by_left = [row for row in rows if row["cohort_index"] == left["index"]]
            by_right = [row for row in rows if row["cohort_index"] == right["index"]]
            same = labels[left["index"]] == labels[right["index"]]
            for mode in modes:
                left_items = [row["modes"][mode] for row in by_left if mode in row["modes"] and row["modes"][mode]["accepted"]]
                right_items = [row["modes"][mode] for row in by_right if mode in row["modes"] and row["modes"][mode]["accepted"]]
                scores = {}
                for family in model_families:
                    a = [item[family] for item in left_items if item.get(family) is not None]
                    b = [item[family] for item in right_items if item.get(family) is not None]
                    score = _median_pairwise(a, b) if a and b else None
                    if score is not None:
                        scores[family] = score
                # Context comes only from the visible eye-derived upper ROI;
                # in the masked mode, use the actual masked detector frame.
                left_context = [item["context"] for item in left_items if "context" in item]
                right_context = [item["context"] for item in right_items if "context" in item]
                if left_context and right_context:
                    scores["upper_context"] = _median_pairwise([item["appearance"] for item in left_context], [item["appearance"] for item in right_context])
                    g_left = {key: statistics.mean(item["shooting_geometry"][key] for item in left_context) for key in left_context[0]["shooting_geometry"]}
                    g_right = {key: statistics.mean(item["shooting_geometry"][key] for item in right_context) for key in right_context[0]["shooting_geometry"]}
                    scores["shooting_geometry"] = _geometry_score(g_left, g_right)
                pair = {"same": same, "scores": scores, "left": names[left["index"]], "right": names[right["index"]]}
                pair_rows[mode].append(pair)
                raw_similarity[mode].append({"left": names[left["index"]], "right": names[right["index"]], "same": same, "scores": scores})

        report = {
            "description": "Offline pseudo-mask ablation on one visually selected expected-subject track per video.",
            "cohort": {"video_count": len(cohort), "pair_count": sum(1 for _ in combinations(cohort, 2)),
                       "positive_pairs": sum(1 for a, b in combinations(cohort, 2) if labels[a["index"]] == labels[b["index"]]),
                       "negative_pairs": sum(1 for a, b in combinations(cohort, 2) if labels[a["index"]] != labels[b["index"]]),
                       "samples_per_track_max": MAX_SAMPLES_PER_TRACK,
                       "sample_count": sum(len(video["samples"]) for video in cohort)},
            "execution": {"requested_acceleration": "cuda", "runtime_by_region": {key: analyzer.runtime_description for key, analyzer in analyzers.items()},
                          "upper_multiscale_detection": analyzers["upper"].options.upper_multiscale_detection,
                          "upper_detector_scales": ["base", "0.5x", "1.5x"] if analyzers["upper"].options.upper_multiscale_detection else ["base"],
                          "selected_tracks": len(cohort)},
            "detector": {"clean": {"attempted": detector_attempts["clean"], "hits": detector_hits["clean"],
                                    "recall_over_all_selected_samples": detector_hits["clean"] / max(sum(len(v["samples"]) for v in cohort), 1)},
                          "masked": {"attempted": detector_attempts["masked"], "hits": detector_hits["masked"],
                                     "recall_over_all_selected_samples": detector_hits["masked"] / max(sum(len(v["samples"]) for v in cohort), 1)}},
            "quality_acceptance": {mode: {"attempted": quality_attempts[mode], "accepted": quality_accepted[mode],
                                          "accepted_over_all_selected_samples": quality_accepted[mode] / max(sum(len(v["samples"]) for v in cohort), 1),
                                          "accepted_given_detector_hit": quality_accepted[mode] / max(quality_attempts[mode], 1),
                                          "detector_hits": quality_attempts[mode],
                                          "target_sample_denominator": sum(len(v["samples"]) for v in cohort)} for mode in sorted(quality_attempts)},
            "genuine_embeddings": dict(embedding_counts),
            "conditions": {},
            "pair_scores": raw_similarity,
            "limitations": ["Filename answer key is applied only after all inference and is not an application input.",
                            "One manually selected track per supplied video; the cohort is small and non-independent.",
                            "The fourth clip in each subject's series is held out; pairwise split shares clips and is not subject-independent.",
                            "Scores and thresholds are offline ablation results, not production settings or general masked-face accuracy.",
                            "Upper appearance and shooting geometry are separate context cues, not stable biometric embeddings."],
        }
        for mode in modes:
            examples = pair_rows[mode]
            available = sorted({key for item in examples for key in item["scores"]})
            reports = {family: _single_family_report(examples, family) for family in available}
            per_video = []
            for video in cohort:
                video_rows = [item for item in rows if item["cohort_index"] == video["index"]]
                per_video.append({"cohort_index": video["index"],
                                  "selected_samples": len(video["samples"]),
                                  "clean_detection_hits": sum(item["clean_detector_hit"] for item in video_rows),
                                  "masked_detection_hits": sum(item["masked_detector_hit"] for item in video_rows),
                                  "accepted_embeddings": {condition: sum(item.get("modes", {}).get(condition, {}).get("accepted", False) for item in video_rows)
                                                          for condition in modes}})
            report["conditions"][mode] = {"available_score_families": available,
                                           "score_medians_by_pair_label": _score_distribution(examples),
                                           "score_evaluations": reports,
                                           "balanced_embedding_thresholds": _embedding_ensemble_report(examples),
                                           "context_added_candidate": _context_candidate_report(examples),
                                           "accepted_frames_per_video": per_video,
                                           "videos_with_at_least_3_accepted_embeddings": {
                                               condition: sum(item["accepted_embeddings"][condition] >= 3 for item in per_video)
                                               for condition in modes}}
        single_scale_path = ARTIFACTS / "masked-face-ablation-single-scale.json"
        if single_scale_path.is_file():
            single = json.loads(single_scale_path.read_text(encoding="utf-8"))
            old_rows = single["conditions"]["full_clean"]["accepted_frames_per_video"]
            new_rows = report["conditions"]["full_clean"]["accepted_frames_per_video"]
            old_counts = {item["cohort_index"]: item["accepted_embeddings"]["upper_masked"] for item in old_rows}
            new_counts = {item["cohort_index"]: item["accepted_embeddings"]["upper_masked"] for item in new_rows}
            deltas = [new_counts[index] - old_counts[index] for index in old_counts.keys() & new_counts.keys()]
            previous_pairs = single["conditions"]["upper_masked"]["balanced_embedding_thresholds"]
            current_pairs = report["conditions"]["upper_masked"]["balanced_embedding_thresholds"]
            previous_holdout = previous_pairs["heldout_at_training_selected_zero_fp_candidate"]
            current_holdout = current_pairs["heldout_at_training_selected_zero_fp_candidate"]
            report["single_scale_comparison"] = {
                "masked_quality_accepted_frames_before": single["quality_acceptance"]["upper_masked"]["accepted"],
                "masked_quality_accepted_frames_after": quality_accepted["upper_masked"],
                "per_video_net_additional_accepted_frames": sum(max(0, delta) for delta in deltas),
                "per_video_lost_acceptances": sum(max(0, -delta) for delta in deltas),
                "videos_with_more_accepted_frames": sum(delta > 0 for delta in deltas),
                "tracks_with_at_least_3_masked_embeddings_before": single["conditions"]["full_clean"]["videos_with_at_least_3_accepted_embeddings"]["upper_masked"],
                "tracks_with_at_least_3_masked_embeddings_after": report["conditions"]["full_clean"]["videos_with_at_least_3_accepted_embeddings"]["upper_masked"],
                "heldout_complete_pair_count_before": previous_pairs["heldout_eligible_pairs"],
                "heldout_complete_pair_count_after": current_pairs["heldout_eligible_pairs"],
                "heldout_positive_pairs_before": previous_holdout["positive_count"],
                "heldout_positive_pairs_after": current_holdout["positive_count"],
            }
        report["detector_provenance"] = _detector_provenance(analyzers["upper"])
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        _write_contact_sheet(contact_items)
        print(json.dumps({"output": str(OUTPUT), "contact_sheet": str(CONTACT_SHEET),
                          "cohort": report["cohort"], "detector": report["detector"],
                          "quality_acceptance": report["quality_acceptance"],
                          "conditions": {mode: {"families": value["available_score_families"],
                                               "score_medians": value["score_medians_by_pair_label"],
                                               "face01_zero_fp_holdout": _compact_pair_metrics(value["score_evaluations"]["face01"]["heldout_at_training_selected_zero_fp_candidate"]) if value["score_evaluations"].get("face01", {}).get("heldout_at_training_selected_zero_fp_candidate") else None,
                                               "adaface_zero_fp_holdout": _compact_pair_metrics(value["score_evaluations"]["adaface"]["heldout_at_training_selected_zero_fp_candidate"]) if value["score_evaluations"].get("adaface", {}).get("heldout_at_training_selected_zero_fp_candidate") else None,
                                               "balanced_zero_fp_holdout": _compact_pair_metrics(value["balanced_embedding_thresholds"]["heldout_at_training_selected_zero_fp_candidate"]) if value["balanced_embedding_thresholds"]["heldout_at_training_selected_zero_fp_candidate"] else None,
                                               "context_candidate_zero_fp_holdout": _compact_pair_metrics(value["context_added_candidate"]["heldout_at_training_selected_zero_fp_candidate"]) if value["context_added_candidate"]["heldout_at_training_selected_zero_fp_candidate"] else None,
                                               "videos_with_at_least_3": value["videos_with_at_least_3_accepted_embeddings"]}
                                         for mode, value in report["conditions"].items()}}, indent=2))
    finally:
        for analyzer in analyzers.values():
            analyzer.close()


def _write_contact_sheet(items):
    if not items:
        return
    cols, cell_w, cell_h = 6, 150, 158
    rows = math.ceil(len(items) / cols)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, np.uint8)
    for index, (cohort_index, mode, image) in enumerate(items):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        sheet[y:y + 128, x:x + 128] = image
        cv2.putText(sheet, f"track {cohort_index:02d}", (x + 2, y + 143), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, mode.replace("_", " ")[:21], (x + 2, y + 156), cv2.FONT_HERSHEY_SIMPLEX, .35, (20, 20, 20), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", sheet)
    if not ok:
        raise RuntimeError("Could not encode diagnostic contact sheet")
    CONTACT_SHEET.write_bytes(encoded.tobytes())


def _detector_provenance(analyzer):
    detector = analyzer._detector
    source = detector.options.detection_model_path.expanduser().resolve()
    derived = Path(detector.runner.path).expanduser().resolve()
    manifest_path = derived.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    return {
        "source_model": str(source),
        "source_sha256": detector.sha256,
        "inference_model": str(derived),
        "inference_sha256": detector.inference_sha256,
        "derived_manifest": manifest,
    }


if __name__ == "__main__":
    run()
