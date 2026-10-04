"""Track-based identity decisions. All default thresholds are uncalibrated."""
from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING
from uuid import uuid4
import numpy as np
from .storage import FaceRecord, PersonRecord, RelationRecord, TrackRecord, VideoRecord
from .tracking import build_tracks, cosine, overlap, refresh_representatives
if TYPE_CHECKING:
    from .analyzer import FaceSample

NO_MATCH = math.inf
MATCH_THRESHOLD = .18
MATCH_MARGIN = .08


@dataclass
class GroupingOptions:
    auto_merge_enabled: bool = False
    high_threshold: float = .90
    medium_threshold: float = .70
    minimum_samples: int = 3
    maximum_samples: int = 30
    minimum_quality: float = .5
    match_threshold: float = .75
    model_weights: dict[str, float] = field(default_factory=dict)
    model_thresholds: dict[str, float] = field(default_factory=dict)
    auxiliary_enabled: bool = False
    maximum_track_gap: float = 1.5


@dataclass
class ClassificationResult:
    faces: list[FaceRecord]
    people: list[PersonRecord]
    removed_ids: list[str]
    tracks: list[TrackRecord] = field(default_factory=list)
    relations: list[RelationRecord] = field(default_factory=list)


def _distance(left: list[float] | None, right: list[float] | None) -> float:
    score = cosine(left or [], right or [])
    return 1 - score if score is not None else NO_MATCH


_overlap = overlap


def similarity_statistics(left: list[list[float]], right: list[list[float]], match_threshold: float = .75, _cache: dict | None = None) -> dict:
    def normalized_groups(vectors):
        if _cache is not None and id(vectors) in _cache:
            stored, groups = _cache[id(vectors)]
            if stored is vectors:
                return groups
        groups = {}
        for vector in vectors:
            array = np.asarray(vector, dtype=np.float64)
            if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                continue
            norm = float(np.linalg.norm(array))
            if not np.isfinite(norm) or norm <= 0:
                continue
            groups.setdefault(len(array), []).append(array / norm)
        if _cache is not None:
            _cache[id(vectors)] = (vectors, groups)
        return groups
    left_groups, right_groups = normalized_groups(left), normalized_groups(right)
    dimensions = left_groups.keys() & right_groups.keys()
    comparisons = [np.clip(np.asarray(left_groups[dimension]) @ np.asarray(right_groups[dimension]).T, -1, 1).ravel()
                   for dimension in dimensions]
    if not comparisons:
        return {}
    scores = np.sort(np.concatenate(comparisons)).tolist()
    n = len(scores)
    return {"max": scores[-1], "mean": mean(scores), "median": median(scores), "top10_mean": mean(scores[-max(1, math.ceil(n * .1)):]), "top20_mean": mean(scores[-max(1, math.ceil(n * .2)):]), "lower_quantile": scores[math.floor((n - 1) * .1)], "match_rate": sum(score >= match_threshold for score in scores) / n, "pair_count": n, "sample_count_a": sum(len(left_groups[d]) for d in dimensions), "sample_count_b": sum(len(right_groups[d]) for d in dimensions), "embedding_dimensions": sorted(dimensions)}


def _primary(key: str) -> bool:
    return any(name in key.lower() for name in ("face01", "japanese"))


def compare_tracks(left: TrackRecord, right: TrackRecord, options: GroupingOptions | None = None, _cache: dict | None = None) -> dict:
    options = options or GroupingOptions()
    compatible_keys = left.embedding_samples.keys() & right.embedding_samples.keys()
    active_keys = {key for key in compatible_keys if _primary(key) or options.auxiliary_enabled and "adaface" in key.lower()}
    models = {key: similarity_statistics(left.embedding_samples[key], right.embedding_samples[key], options.match_threshold, _cache) for key in active_keys}
    models = {key: statistics for key, statistics in models.items() if statistics}
    reasons = []
    if left.review_required or right.review_required:
        reasons.append("Reanalysis track lineage requires human review")
    if left.video_id == right.video_id and max(left.start_time, right.start_time) <= min(left.end_time, right.end_time):
        reasons.append("Tracks overlap in the same video")
    if not any(_primary(key) for key in models):
        reasons.append("Primary model evidence is missing or model versions differ")
    if sum(_primary(key) for key in models) > 1 or sum("adaface" in key.lower() for key in models) > 1:
        reasons.append("Multiple versions of one model require separate evaluation")
    if options.auxiliary_enabled and not any("adaface" in key.lower() for key in models):
        reasons.append("Auxiliary model evidence is missing")
    if not models:
        return {"decision": "LOW", "score": None, "models": {}, "reasons": reasons or ["No compatible embeddings"]}
    weights = {key: options.model_weights.get(key, options.model_weights.get(key.split(":", 1)[0], 1.0)) for key in models}
    total = sum(max(0.0, value) for value in weights.values())
    score = sum(statistics["median"] * max(0.0, weights[key]) for key, statistics in models.items()) / total if total else None
    if score is None:
        return {"decision": "LOW", "score": None, "models": models, "reasons": reasons + ["No positive model weights"]}
    if any(min(statistics["sample_count_a"], statistics["sample_count_b"]) < options.minimum_samples for statistics in models.values()):
        reasons.append("Too few high-quality samples")
    if any(len(statistics["embedding_dimensions"]) != 1 for statistics in models.values()):
        reasons.append("One model identity contains inconsistent embedding dimensions")
    if min(left.quality, right.quality) < options.minimum_quality:
        reasons.append("Track sample quality is insufficient")
    if any(track.pose_face_ids and "frontal" not in track.pose_face_ids and "unknown" not in track.pose_face_ids for track in (left, right)):
        reasons.append("Only profile samples are available")
    high_per_model = []
    for key, statistics in models.items():
        family = "face01" if _primary(key) else "adaface" if "adaface" in key.lower() else key.split(":", 1)[0]
        threshold = options.model_thresholds.get(key, options.model_thresholds.get(family, options.high_threshold))
        statistics["high_threshold"] = threshold
        high_per_model.append(statistics["median"] >= threshold and statistics["lower_quantile"] >= options.match_threshold and statistics["match_rate"] >= .9)
    if any(high_per_model) and not all(high_per_model):
        reasons.append("Models disagree or similarity distributions are unstable")
    elif not all(high_per_model):
        reasons.append("Model-specific HIGH thresholds or distribution evidence are insufficient")
    decision = "HIGH" if score >= options.high_threshold and all(high_per_model) and not reasons else "MEDIUM" if score >= options.medium_threshold else "LOW"
    return {"decision": decision, "score": score, "models": models, "reasons": reasons}


def _cluster_tracks(faces: list[FaceRecord], tracks: list[TrackRecord], people: list[PersonRecord], relations: list[RelationRecord], options: GroupingOptions, preferred_ids: dict[str, str] | None = None, changed_track_ids: set[str] | None = None) -> ClassificationResult:
    # Embedding lists stay fixed during this call. This cache is discarded after
    # the batch, so edits and model/settings changes cannot reuse stale evidence.
    normalized_cache = {}
    persons = {person.id: person for person in people}
    new_people = []
    indexed_relations = {(relation.track_a, relation.track_b): relation for relation in relations}
    new_relations = []
    face_map = {face.id: face for face in faces}
    clusters: dict[str, list[TrackRecord]] = {}
    manual_unknown = {face.track_id for face in faces if face.manual_assignment and face.person_id is None}
    manual_tracks = {face.track_id for face in faces if face.manual_assignment}
    track_map = {track.id: track for track in tracks}
    for relation in relations:
        if relation.relation != "confirmed_different":
            continue
        left, right = track_map.get(relation.track_a), track_map.get(relation.track_b)
        if left and right and left.person_id is not None and left.person_id == right.person_id:
            if left.id in manual_tracks and right.id in manual_tracks:
                raise ValueError("Stored negative relation conflicts with manual track assignments")
            (right if right.id not in manual_tracks else left).person_id = None
    for track in tracks:
        if track.person_id in persons:
            clusters.setdefault(track.person_id, []).append(track)
    for track in tracks:
        if track.person_id in persons or track.id in manual_unknown:
            continue
        if not track.embedding_samples:
            track.person_id = None
            continue
        possible = []
        for person_id, members in list(clusters.items()):
            comparisons = []
            blocked = False
            for member in members:
                pair = tuple(sorted((track.id, member.id)))
                relation = indexed_relations.get(pair)
                if relation and relation.relation == "confirmed_different":
                    blocked = True
                    break
                comparison = compare_tracks(track, member, options, normalized_cache)
                if relation and relation.relation == "confirmed_same":
                    comparison = {**comparison, "decision": "HIGH", "human_confirmed": True}
                comparisons.append((member, comparison))
            if blocked or not comparisons:
                continue
            eligible = all(comparison["decision"] == "HIGH" for _, comparison in comparisons)
            human = all(comparison.get("human_confirmed", False) for _, comparison in comparisons)
            if eligible and (options.auto_merge_enabled or human):
                possible.append((person_id, comparisons))
            else:
                for member, comparison in comparisons:
                    pair = tuple(sorted((track.id, member.id)))
                    if comparison["decision"] in {"HIGH", "MEDIUM"} and pair not in indexed_relations:
                        relation = RelationRecord(*pair, "pending", comparison)
                        indexed_relations[pair] = relation
                        new_relations.append(relation)
        if len(possible) == 1:
            track.person_id = possible[0][0]
        else:
            representative = face_map.get(track.representative_face_id)
            preferred = (preferred_ids or {}).get(track.id)
            if preferred in persons and preferred not in clusters:
                person = persons[preferred]
                person.representative_face_id = track.representative_face_id
                person.thumbnail_path = representative.thumbnail_path if representative else person.thumbnail_path
            else:
                person = PersonRecord(str(uuid4()), "", representative.thumbnail_path if representative else None, track.representative_face_id)
            persons[person.id] = person
            new_people.append(person)
            track.person_id = person.id
            if len(possible) > 1:
                for _, comparisons in possible:
                    for member, comparison in comparisons:
                        pair = tuple(sorted((track.id, member.id)))
                        if pair not in indexed_relations:
                            relation = RelationRecord(*pair, "pending", {**comparison, "reasons": ["Multiple matching identities"]})
                            indexed_relations[pair] = relation
                            new_relations.append(relation)
        clusters.setdefault(track.person_id, []).append(track)
    track_map = {track.id: track for track in tracks}
    for face in faces:
        track = track_map.get(face.track_id)
        if track and not face.excluded and not face.manual_assignment:
            face.person_id = track.person_id
    for person_id, members in clusters.items():
        person = persons[person_id]
        current = face_map.get(person.representative_face_id)
        if current is not None and current.person_id == person_id and not current.excluded:
            continue
        eligible = [face_map[track.representative_face_id] for track in members if track.representative_face_id in face_map and not face_map[track.representative_face_id].excluded]
        representative = max(eligible, key=lambda face: face.quality or 0) if eligible else None
        old_values = person.representative_face_id, person.thumbnail_path
        person.representative_face_id = representative.id if representative else None
        person.thumbnail_path = representative.thumbnail_path if representative else None
        if old_values != (person.representative_face_id, person.thumbnail_path) and all(item.id != person.id for item in new_people):
            new_people.append(person)
    changed_relations = {(relation.track_a, relation.track_b): relation for relation in new_relations}
    for pair, relation in indexed_relations.items():
        if relation.relation in {"pending", "deferred"} and pair[0] in track_map and pair[1] in track_map:
            if changed_track_ids is not None and not changed_track_ids.intersection(pair):
                continue
            comparison = compare_tracks(track_map[pair[0]], track_map[pair[1]], options, normalized_cache)
            details = {**relation.details, **comparison}
            if details != relation.details:
                changed_relations[pair] = RelationRecord(*pair, relation.relation, details)
    return ClassificationResult(faces, new_people, [], tracks, list(changed_relations.values()))


def classify_analysis(samples: list["FaceSample"], video: VideoRecord, existing_faces: list[FaceRecord], people: list[PersonRecord], thumbnail_dir: Path, completed_second: float, duration: float, *, existing_tracks: list[TrackRecord] | None = None, relations: list[RelationRecord] | None = None, options: GroupingOptions | None = None) -> ClassificationResult:
    options = options or GroupingOptions()
    existing_tracks = existing_tracks or []
    relations = relations or []
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    lower = max(0, video.last_analyzed_second - 1e-6)
    upper = math.inf if completed_second >= duration else completed_second - 1e-6
    region = [face for face in existing_faces if face.video_id == video.id and lower <= face.second < upper]
    reviewed_track_ids = {endpoint for relation in relations if relation.relation in {"confirmed_same", "confirmed_different"} for endpoint in (relation.track_a, relation.track_b)}
    reviewed_track_ids.update(face.track_id for face in existing_faces if face.manual_assignment and face.track_id)
    named_person_ids = {person.id for person in people if person.name.strip()}
    named_track_ids = {track.id for track in existing_tracks if track.person_id in named_person_ids}
    protected_track_ids = reviewed_track_ids | named_track_ids
    reviewed_video = any(track.video_id == video.id and track.id in reviewed_track_ids for track in existing_tracks)
    reanalysis_requires_lineage = video.analyzed_at is not None and reviewed_video
    matched = set()
    faces = []
    uncertain_face_ids = set()
    def protected(face: FaceRecord) -> bool:
        return face.manual_assignment or face.excluded or face.track_id in protected_track_ids
    for sample in sorted(samples, key=lambda sample: sample.second):
        candidates = [face for face in region if face.id not in matched and abs(face.second - sample.second) <= (1e-5 if protected(face) else .1) and overlap(face.bounding_box, sample.bounding_box) >= .3]
        candidates.sort(key=lambda face: overlap(face.bounding_box, sample.bounding_box), reverse=True)
        reusable = candidates[0] if candidates else None
        uncertain = any(protected(face) and abs(face.second - sample.second) <= .1 and overlap(face.bounding_box, sample.bounding_box) >= .3 for face in region)
        if reusable is not None and protected(reusable):
            best_overlap = overlap(reusable.bounding_box, sample.bounding_box)
            ambiguous_old = len(candidates) > 1 and best_overlap - overlap(candidates[1].bounding_box, sample.bounding_box) < .15
            competing_sample = any(peer is not sample and abs(peer.second - reusable.second) <= 1e-5 and overlap(peer.bounding_box, reusable.bounding_box) >= .3 for peer in samples)
            if ambiguous_old or competing_sample:
                reusable = None
            else:
                uncertain = False
        identifier = reusable.id if reusable else str(uuid4())
        if reusable:
            matched.add(identifier)
        elif uncertain:
            uncertain_face_ids.add(identifier)
        image_path = thumbnail_dir / f"face-{identifier}.jpg"
        image_path.write_bytes(sample.thumbnail_jpeg)
        corrected = bool(reusable and (reusable.manual_assignment or reusable.excluded))
        keep_lineage = bool(reusable and (protected(reusable) or reusable.track_id in named_track_ids))
        faces.append(FaceRecord(id=identifier, video_id=video.id, second=sample.second, bounding_box=sample.bounding_box, person_id=reusable.person_id if corrected else None, thumbnail_path=str(image_path), embedding=sample.embedding, manual_assignment=corrected, excluded=reusable.excluded if reusable else False, embedding_model=sample.embedding_model, quality=sample.quality, rejection_reason=sample.rejection_reason, track_id=reusable.track_id if keep_lineage else None, frame_number=getattr(sample, "frame_number", None), detection_score=getattr(sample, "detection_score", None), landmarks=getattr(sample, "landmarks", None), pose=getattr(sample, "pose", None), aligned_path=getattr(sample, "aligned_path", None), embeddings=dict(getattr(sample, "embeddings", {}) or {})))
    removed = [face.id for face in region if face.id not in matched and not face.manual_assignment and not face.excluded and face.track_id not in protected_track_ids]
    remaining = [face for face in existing_faces if face.id not in set(removed) and face.id not in matched]
    past = [face for face in remaining if face.video_id != video.id or face.second < lower]
    touched = build_tracks(faces, [face for face in past if face.video_id == video.id],
        [track for track in existing_tracks if track.video_id == video.id], options.maximum_track_gap, options.minimum_quality)
    previous_track_ids = {track.id for track in existing_tracks}
    for track in touched:
        if track.id not in previous_track_ids and (reanalysis_requires_lineage or uncertain_face_ids.intersection(track.face_ids)):
            track.review_required = True
    track_map = {track.id: track for track in existing_tracks}
    track_map.update({track.id: track for track in touched})
    all_faces = remaining + faces
    face_map = {face.id: face for face in all_faces}
    changed_track_ids = {track.id for track in touched} | {face.track_id for face in region if face.id in set(removed)}
    changed_track_ids.update(face.track_id for face in region if face.id in matched and face.track_id not in protected_track_ids)
    sample_settings = f"{options.minimum_quality:.17g}:{options.maximum_samples}"
    for track in track_map.values():
        if track.id in changed_track_ids or track.sample_settings != sample_settings:
            track.face_ids = [identifier for identifier in track.face_ids if identifier in face_map and face_map[identifier].track_id == track.id]
            refresh_representatives(track, face_map, options.minimum_quality, options.maximum_samples)
            changed_track_ids.add(track.id)
    active_tracks = [track for track in track_map.values() if track.face_ids]
    result = _cluster_tracks(all_faces, active_tracks, people, relations, options, changed_track_ids=changed_track_ids)
    # Existing detections in a touched track share its person membership too.
    result.faces = [face for face in all_faces if face.video_id == video.id and (face.id in {new.id for new in faces} or face.track_id in {track.id for track in touched})]
    result.removed_ids = removed
    return result


def classify_faces(samples: list["FaceSample"], video: VideoRecord, existing_faces: list[FaceRecord], people: list[PersonRecord], thumbnail_dir: Path, completed_second: float, duration: float) -> tuple[list[FaceRecord], list[PersonRecord], list[str]]:
    result = classify_analysis(samples, video, existing_faces, people, thumbnail_dir, completed_second, duration)
    return result.faces, result.people, result.removed_ids


def recluster_tracks(faces: list[FaceRecord], tracks: list[TrackRecord], people: list[PersonRecord], relations: list[RelationRecord], options: GroupingOptions | None = None) -> ClassificationResult:
    """Re-evaluate cached tracks without image inference; preserve human decisions."""
    options = options or GroupingOptions()
    manual_tracks = {face.track_id for face in faces if face.manual_assignment}
    protected = {endpoint for relation in relations if relation.relation == "confirmed_same" for endpoint in (relation.track_a, relation.track_b)}
    preferred_ids = {track.id: track.person_id for track in tracks if track.person_id is not None}
    for track in tracks:
        refresh_representatives(track, {face.id: face for face in faces}, options.minimum_quality, options.maximum_samples)
        if track.id not in manual_tracks and track.id not in protected:
            track.person_id = None
    return _cluster_tracks(faces, tracks, people, relations, options, preferred_ids)
