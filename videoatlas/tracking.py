"""Conservative spatial and embedding association, independent of model runtimes."""
from collections import defaultdict
import math
from statistics import mean
from uuid import uuid4
from .storage import FaceRecord, TrackRecord


def cosine(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    if not all(math.isfinite(value) for value in left + right):
        return None
    denominator = math.sqrt(math.fsum(value * value for value in left) * math.fsum(value * value for value in right))
    if denominator <= 0:
        return None
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right)) / denominator))


def overlap(left: tuple | None, right: tuple | None) -> float:
    if left is None or right is None:
        return 0.0
    intersection = max(0.0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0])) * max(0.0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1]))
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def association_score(previous: FaceRecord, current: FaceRecord, maximum_gap: float) -> float | None:
    gap = current.second - previous.second
    if gap <= 0 or gap > maximum_gap or previous.bounding_box is None or current.bounding_box is None:
        return None
    a, b = previous.bounding_box, current.bounding_box
    motion = math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2)
    scale = max(math.hypot(a[2], a[3]), math.hypot(b[2], b[3]), .01)
    iou = overlap(a, b)
    if iou < .15 and motion / scale > .45:
        return None
    scores = [score for key in previous.embeddings.keys() & current.embeddings.keys() if (score := cosine(previous.embeddings[key], current.embeddings[key])) is not None]
    # Spatial continuity alone cannot establish a person's track.
    if not scores or min(scores) < .75:
        return None
    return .5 * min(scores) + .35 * iou + .15 * max(0.0, 1 - motion / scale)


def build_tracks(faces: list[FaceRecord], existing_faces: list[FaceRecord], existing_tracks: list[TrackRecord], maximum_gap: float = 1.5, minimum_quality: float = .5) -> list[TrackRecord]:
    tracks = {track.id: track for track in existing_tracks}
    all_faces = {face.id: face for face in existing_faces}
    all_faces.update({face.id: face for face in faces})
    recent: dict[str, FaceRecord] = {}
    for face in existing_faces:
        if face.track_id and not face.excluded and (face.track_id not in recent or face.second > recent[face.track_id].second):
            recent[face.track_id] = face
    frames = defaultdict(list)
    for face in faces:
        frames[(face.video_id, round(face.second * 1000000))].append(face)
    touched = set()
    for (video_id, _), detections in sorted(frames.items(), key=lambda pair: pair[0]):
        occupied = set()
        candidates = {}
        for face in detections:
            if face.excluded:
                continue
            scores = sorted(((score, track_id) for track_id, previous in recent.items() if previous.video_id == video_id and (score := association_score(previous, face, maximum_gap)) is not None), reverse=True)
            if scores and (len(scores) == 1 or scores[0][0] - scores[1][0] >= .08):
                candidates[face.id] = scores[0]
        for face in detections:
            if face.excluded:
                continue
            track_id = face.track_id if face.track_id in tracks else None
            if track_id in occupied:
                track_id = None
            best = candidates.get(face.id)
            if track_id is None and best and best[1] not in occupied and not any(other.id != face.id and candidates.get(other.id, (-1, None))[1] == best[1] and candidates[other.id][0] >= best[0] - .08 for other in detections):
                track_id = best[1]
            if track_id is None:
                track_id = str(uuid4())
                tracks[track_id] = TrackRecord(track_id, face.video_id, face.second, face.second, face.person_id)
            track = tracks[track_id]
            face.track_id = track_id
            if face.manual_assignment:
                track.person_id = face.person_id
            track.start_time = min(track.start_time, face.second)
            track.end_time = max(track.end_time, face.second)
            if face.id not in track.face_ids:
                track.face_ids.append(face.id)
            occupied.add(track_id)
            touched.add(track_id)
            recent[track_id] = face
    for track_id in touched:
        refresh_representatives(tracks[track_id], all_faces, minimum_quality)
    return [tracks[track_id] for track_id in sorted(touched)]


def pose_bucket(pose: dict | str | None) -> str:
    if isinstance(pose, str):
        return pose.lower() if pose.lower() in {"left", "right", "frontal"} else "unknown"
    if isinstance(pose, dict) and "yaw" in pose:
        yaw = pose["yaw"]
        if not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
            return "unknown"
        return "left" if yaw < -15 else "right" if yaw > 15 else "frontal"
    return "unknown"


def refresh_representatives(track: TrackRecord, faces: dict[str, FaceRecord], minimum_quality: float = .5, limit: int = 30) -> None:
    track.sample_settings = f"{minimum_quality:.17g}:{limit}"
    track.face_ids = [identifier for identifier in track.face_ids if identifier in faces]
    if track.face_ids:
        track.start_time = min(faces[identifier].second for identifier in track.face_ids)
        track.end_time = max(faces[identifier].second for identifier in track.face_ids)
    candidates = sorted((faces[identifier] for identifier in track.face_ids if identifier in faces and not faces[identifier].excluded and faces[identifier].embeddings and (faces[identifier].quality or 0) >= minimum_quality), key=lambda face: (-(face.quality or 0), face.second))
    selected: list[FaceRecord] = []
    # Spacing and pose bins retain diversity; never replace all samples by a mean.
    for face in candidates:
        bucket = pose_bucket(face.pose)
        if face.frame_number is not None and any(previous.video_id == face.video_id and previous.frame_number == face.frame_number for previous in selected):
            continue
        if any(abs(face.second - previous.second) < .15 and bucket == pose_bucket(previous.pose) for previous in selected):
            continue
        duplicates = []
        for previous in selected:
            common = face.embeddings.keys() & previous.embeddings.keys()
            similarities = [cosine(face.embeddings[key], previous.embeddings[key]) for key in common]
            if similarities and all(score is not None and score > .999 for score in similarities) and bucket == pose_bucket(previous.pose):
                duplicates.append(previous)
        # Keep a small repeated-view baseline, not thirty indistinguishable frames.
        if len(duplicates) >= 3 or any(abs(face.second - previous.second) < .5 for previous in duplicates):
            continue
        selected.append(face)
        if len(selected) >= limit:
            break
    track.sample_face_ids = [face.id for face in selected]
    track.embedding_samples = {}
    for face in selected:
        for key, embedding in face.embeddings.items():
            if cosine(embedding, embedding) is not None:
                track.embedding_samples.setdefault(key, []).append(embedding)
    track.mean_embeddings = {}
    for key, vectors in track.embedding_samples.items():
        dimension = len(vectors[0])
        vectors = [vector for vector in vectors if len(vector) == dimension]
        vector = [mean(values) for values in zip(*vectors)]
        length = math.sqrt(sum(value * value for value in vector))
        if length:
            track.mean_embeddings[key] = [value / length for value in vector]
    track.representative_face_id = selected[0].id if selected else None
    track.quality = mean(face.quality or 0 for face in selected) if selected else 0
    track.pose_face_ids = {}
    for face in selected:
        track.pose_face_ids.setdefault(pose_bucket(face.pose), face.id)
    if selected and track.mean_embeddings:
        key = next(iter(track.mean_embeddings))
        compatible = [face for face in selected if key in face.embeddings]
        track.central_face_id = max(compatible, key=lambda face: cosine(face.embeddings[key], track.mean_embeddings[key]) if cosine(face.embeddings[key], track.mean_embeddings[key]) is not None else -1).id
    else:
        track.central_face_id = None
