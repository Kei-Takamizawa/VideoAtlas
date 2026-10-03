from collections import defaultdict
import math
from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING
from .storage import FaceRecord, PersonRecord, VideoRecord
if TYPE_CHECKING:
    from .analyzer import FaceSample
NO_MATCH = math.inf
MATCH_THRESHOLD = 0.18
MATCH_MARGIN = 0.08


def _distance(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return NO_MATCH
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_length = math.sqrt(math.fsum(value * value for value in left))
    right_length = math.sqrt(math.fsum(value * value for value in right))
    if not math.isfinite(dot) or not math.isfinite(left_length * right_length) or left_length == 0 or right_length == 0:
        return NO_MATCH
    return max(0.0, 1.0 - dot / (left_length * right_length))


def _overlap(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[0] + left[2], right[0] + right[2])
    y2 = min(left[1] + left[3], right[1] + right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left[2] * left[3] + right[2] * right[3] - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _add_representative(face: FaceRecord, representatives: dict[tuple[str, str], list[FaceRecord]]) -> None:
    if face.excluded or face.person_id is None or face.embedding is None or face.embedding_model is None:
        return
    examples = representatives.setdefault((face.person_id, face.embedding_model), [])
    if any(example.id == face.id for example in examples):
        return
    if not examples:
        examples.append(face)
        return
    novelty = min(_distance(face.embedding, example.embedding) for example in examples)
    if len(examples) >= 2 and novelty < 0.10:
        return
    if len(examples) < 8:
        examples.append(face)
        return
    separations = [(min(_distance(examples[index].embedding, other.embedding) for other_index, other in enumerate(examples) if other_index != index), index) for index in range(1, len(examples))]
    smallest_separation, redundant_index = min(separations)
    if novelty > smallest_separation + 0.03:
        examples[redundant_index] = face


def _temporal_score(previous: FaceRecord, sample: "FaceSample", maximum_gap: float) -> float:
    if previous.embedding_model is None or previous.embedding_model != sample.embedding_model:
        return NO_MATCH
    gap = sample.second - previous.second
    if gap <= 0 or gap > maximum_gap:
        return NO_MATCH
    overlap = _overlap(previous.bounding_box, sample.bounding_box)
    if overlap < 0.5:
        return NO_MATCH
    distance = _distance(previous.embedding, sample.embedding)
    if distance > MATCH_THRESHOLD:
        return NO_MATCH
    return distance + 0.2 * (1.0 - overlap) + 0.01 * gap / maximum_gap


def _temporal_person(sample: "FaceSample", peers: list["FaceSample"], recent: dict[str, FaceRecord], occupied: set[str], maximum_gap: float) -> str | None:
    if sample.embedding is None:
        return None
    candidates = sorted((_temporal_score(face, sample, maximum_gap), person_id) for person_id, face in recent.items() if person_id not in occupied)
    if not candidates or not math.isfinite(candidates[0][0]):
        return None
    best_score, best_id = candidates[0]
    if len(candidates) > 1 and candidates[1][0] - best_score < MATCH_MARGIN:
        return None
    if any(peer is not sample and _temporal_score(recent[best_id], peer, maximum_gap) <= best_score + MATCH_MARGIN for peer in peers):
        return None
    return best_id


def _cluster_person(sample: "FaceSample", representatives: dict[tuple[str, str], list[FaceRecord]], occupied: set[str], temporal_id: str | None, thumbnail_path: str, new_people: list[PersonRecord]) -> str | None:
    if sample.embedding is None or sample.embedding_model is None:
        return None
    candidates: list[tuple[float, str]] = []
    for (person_id, model), examples in representatives.items():
        if model != sample.embedding_model or person_id in occupied:
            continue
        distances = sorted(_distance(sample.embedding, face.embedding) for face in examples)
        if not distances or distances[0] > MATCH_THRESHOLD:
            continue
        if len(distances) < 2 or distances[1] > MATCH_THRESHOLD:
            continue
        candidates.append((distances[0], person_id))
    candidates.sort()
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < MATCH_MARGIN:
        return None
    if candidates:
        best_id = candidates[0][1]
        if temporal_id is not None and temporal_id != best_id:
            return None
        return best_id
    if temporal_id is not None:
        return temporal_id
    identifier = str(uuid4())
    new_people.append(PersonRecord(id=identifier, name="", thumbnail_path=thumbnail_path))
    return identifier


def classify_faces(samples: list["FaceSample"], video: VideoRecord, existing_faces: list[FaceRecord], people: list[PersonRecord], thumbnail_dir: Path, completed_second: float, duration: float) -> tuple[list[FaceRecord], list[PersonRecord], list[str]]:
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    previous = [face for face in existing_faces if face.video_id == video.id]
    lower = max(0.0, video.last_analyzed_second - 1e-6)
    upper = math.inf if completed_second >= duration else completed_second - 1e-6
    old_region = [face for face in previous if lower <= face.second < upper]
    named_person_ids = {person.id for person in people if person.name.strip()}
    removed_ids = [face.id for face in old_region if not face.manual_assignment and not face.excluded and face.person_id not in named_person_ids]
    removed = set(removed_ids)
    if not samples:
        return [], [], removed_ids
    previous_by_frame: dict[int, list[FaceRecord]] = defaultdict(list)
    for face in previous:
        previous_by_frame[round(face.second * 10)].append(face)
    existing_person_ids = {person.id for person in people}
    representatives: dict[tuple[str, str], list[FaceRecord]] = {}
    for face in existing_faces:
        if face.id not in removed and face.person_id in existing_person_ids and (face.video_id != video.id or face.second < video.last_analyzed_second):
            _add_representative(face, representatives)
    occupied_by_frame: dict[int, set[str]] = defaultdict(set)
    for face in old_region:
        if (face.manual_assignment or face.person_id in named_person_ids) and not face.excluded and face.person_id is not None:
            occupied_by_frame[round(face.second * 600)].add(face.person_id)
    recent: dict[str, FaceRecord] = {}
    maximum_gap = max(4.5, video.sample_interval * 1.5)
    for face in previous:
        if face.id not in removed and face.person_id in existing_person_ids and not face.excluded and face.bounding_box is not None and face.embedding is not None and face.second < video.last_analyzed_second and video.last_analyzed_second - face.second <= maximum_gap + 0.25:
            if face.person_id not in recent or face.second > recent[face.person_id].second:
                recent[face.person_id] = face
    samples_by_frame: dict[int, list[FaceSample]] = defaultdict(list)
    for sample in samples:
        samples_by_frame[round(sample.second * 600)].append(sample)
    new_faces: list[FaceRecord] = []
    new_people: list[PersonRecord] = []
    matched_old_ids: set[str] = set()
    for sample in sorted(samples, key=lambda item: item.second):
        frame_key = round(sample.second * 600)
        old_key = round(sample.second * 10)
        old_candidates = [face for key in range(old_key - 1, old_key + 2) for face in previous_by_frame[key] if face.id not in matched_old_ids and abs(face.second - sample.second) <= 0.1]
        matched = [(1.0 - _overlap(face.bounding_box, sample.bounding_box), face) for face in old_candidates if _overlap(face.bounding_box, sample.bounding_box) >= 0.3]
        matched.extend((1.0 + _distance(face.embedding, sample.embedding), face) for face in old_candidates if face.bounding_box is None and sample.embedding_model is not None and face.embedding_model == sample.embedding_model and _distance(face.embedding, sample.embedding) <= MATCH_THRESHOLD)
        reusable = min(matched, key=lambda item: (not (item[1].manual_assignment or item[1].excluded or item[1].person_id in named_person_ids), item[0], abs(item[1].second - sample.second)))[1] if matched else None
        identifier = reusable.id if reusable is not None else str(uuid4())
        if reusable is not None:
            matched_old_ids.add(reusable.id)
        image_path = thumbnail_dir / f"face-{identifier}.jpg"
        image_path.write_bytes(sample.thumbnail_jpeg)
        corrected = reusable is not None and (reusable.manual_assignment or reusable.excluded or reusable.person_id in named_person_ids)
        temporal_id = None if corrected else _temporal_person(sample, samples_by_frame[frame_key], recent, occupied_by_frame[frame_key], maximum_gap)
        person_id = reusable.person_id if corrected else _cluster_person(sample, representatives, occupied_by_frame[frame_key], temporal_id, str(image_path), new_people)
        face = FaceRecord(id=identifier, video_id=video.id, second=sample.second, bounding_box=sample.bounding_box, person_id=person_id, thumbnail_path=str(image_path), embedding=sample.embedding, manual_assignment=corrected, excluded=reusable.excluded if reusable is not None else False, embedding_model=sample.embedding_model, quality=sample.quality, rejection_reason=sample.rejection_reason)
        new_faces.append(face)
        if person_id is not None and not face.excluded:
            occupied_by_frame[frame_key].add(person_id)
            _add_representative(face, representatives)
            recent[person_id] = face
    return new_faces, new_people, removed_ids
