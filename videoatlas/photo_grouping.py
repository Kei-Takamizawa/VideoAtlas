"""Conservative grouping of photo faces into people.

The policy has three steps. All thresholds are uncalibrated until an
evaluation on labelled photos chooses them (see ``photo_evaluation``).

1. Gate: faces that are too small or turned too far aside never start or join a
   group automatically. They go to the review list with suggested groups.
2. Cluster: constrained average-linkage clustering. Two clusters merge only when
   the mean similarity over *every* cross pair of faces reaches the merge
   threshold, so one close pair cannot chain two people together. Faces from the
   same photo are different people and never share a cluster (cannot-link), and
   neither do faces that the user marked as different people.
3. Attach: a face left alone may join a group when the mean of its top-k
   similarities to that group's faces reaches the attach threshold, and the best
   group leads the second-best group by a margin. Otherwise it is listed for
   review with its candidate groups.

Several models can be fused only through a per-model affine score normalization
and a weighted mean. Embeddings themselves are never mixed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .photos import RunData

UNGROUPED = -1


@dataclass(frozen=True)
class GroupingPolicy:
    models: tuple[str, ...]
    merge_threshold: float
    attach_threshold: float | None = None
    attach_margin: float = .05
    attach_top_k: int = 5
    review_threshold: float | None = None
    min_group_size: int = 2
    min_eye_distance: float = 0.0
    max_abs_yaw_ratio: float | None = None
    linkage: str = "average"
    weights: tuple[float, ...] | None = None
    # Score normalization per model: normalized = (cosine - offset) / scale.
    offsets: tuple[float, ...] | None = None
    scales: tuple[float, ...] | None = None
    calibration: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.models:
            raise ValueError("A grouping policy needs at least one model")
        count = len(self.models)
        for name in ("weights", "offsets", "scales"):
            value = getattr(self, name)
            if value is not None and len(value) != count:
                raise ValueError(f"{name} must have one value per model")
        if self.scales is not None and any(scale <= 0 for scale in self.scales):
            raise ValueError("scales must be positive")
        if self.linkage not in ("average", "complete"):
            raise ValueError("linkage must be average or complete")
        if self.min_group_size < 2 or self.attach_top_k < 1:
            raise ValueError("min_group_size must be at least 2 and attach_top_k at least 1")
        if not all(math.isfinite(value) for value in (self.merge_threshold, self.attach_margin, self.min_eye_distance)):
            raise ValueError("thresholds must be finite")

    def to_json(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict) -> "GroupingPolicy":
        values = dict(data)
        for name in ("models", "weights", "offsets", "scales"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        policy = cls(**values)
        policy.validate()
        return policy


def fused_similarity(embeddings: dict[str, np.ndarray], policy: GroupingPolicy, rows: np.ndarray | None = None) -> np.ndarray:
    """Weighted mean of per-model normalized cosine similarities (float32, square)."""
    total = None
    weights = policy.weights or (1.0,) * len(policy.models)
    offsets = policy.offsets or (0.0,) * len(policy.models)
    scales = policy.scales or (1.0,) * len(policy.models)
    for name, weight, offset, scale in zip(policy.models, weights, offsets, scales):
        vectors = embeddings[name] if rows is None else embeddings[name][rows]
        part = (vectors @ vectors.T - np.float32(offset)) * np.float32(weight / scale)
        total = part if total is None else total + part
    return (total / np.float32(sum(weights))).astype(np.float32, copy=False)


def same_photo_constraints(face_photo: np.ndarray) -> np.ndarray:
    """Boolean matrix: True where two different faces come from the same photo."""
    face_photo = np.asarray(face_photo)
    matrix = face_photo[:, None] == face_photo[None, :]
    np.fill_diagonal(matrix, False)
    return matrix


def agglomerate(similarity: np.ndarray, cannot_link: np.ndarray | None = None, linkage: str = "average",
                floor: float = -math.inf) -> list[tuple[int, int, float]]:
    """Constrained agglomerative clustering with the nearest-neighbour-chain algorithm.

    Returns merges as (kept representative, absorbed representative, similarity)
    for every merge whose linkage similarity is at least ``floor``. Average and
    complete linkage are reducible, and cannot-link pairs act as infinite
    distance, which keeps them reducible. Cutting the merges at any threshold
    at or above ``floor`` therefore gives the same clusters as running the
    clustering at that threshold.
    """
    values = np.array(similarity, dtype=np.float64, copy=True)
    count = values.shape[0]
    if values.shape != (count, count):
        raise ValueError("similarity must be square")
    blocked = np.zeros((count, count), bool) if cannot_link is None else np.array(cannot_link, dtype=bool, copy=True)
    sizes = np.ones(count, np.float64)
    active = np.ones(count, bool)
    merges: list[tuple[int, int, float]] = []
    chain: list[int] = []
    average = linkage == "average"
    if linkage not in ("average", "complete"):
        raise ValueError("linkage must be average or complete")

    def scores(index: int) -> np.ndarray:
        row = values[index] / (sizes[index] * sizes) if average else values[index].copy()
        row[~active | blocked[index]] = -math.inf
        row[index] = -math.inf
        return row

    while active.any():
        if not chain:
            chain.append(int(np.flatnonzero(active)[0]))
        current = chain[-1]
        row = scores(current)
        nearest = int(np.argmax(row))
        best = float(row[nearest])
        if not best >= floor:
            active[current] = False
            chain.pop()
            continue
        if len(chain) >= 2 and row[chain[-2]] >= best:
            nearest = chain[-2]
        if len(chain) >= 2 and nearest == chain[-2]:
            chain.pop()
            chain.pop()
            keep, drop = min(current, nearest), max(current, nearest)
            if average:
                values[keep] += values[drop]
                sizes[keep] += sizes[drop]
            else:
                np.minimum(values[keep], values[drop], out=values[keep])
            values[:, keep] = values[keep]
            blocked[keep] |= blocked[drop]
            blocked[:, keep] = blocked[keep]
            active[drop] = False
            merges.append((keep, drop, best))
        else:
            chain.append(nearest)
    return merges


def cut(merges: Iterable[tuple[int, int, float]], count: int, threshold: float) -> np.ndarray:
    """Cluster label per item after applying every merge at or above threshold."""
    parent = np.arange(count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for keep, drop, height in sorted(merges, key=lambda merge: -merge[2]):
        if height < threshold:
            break
        left, right = find(keep), find(drop)
        if left != right:
            parent[max(left, right)] = min(left, right)
    return np.asarray([find(index) for index in range(count)])


def top_k_means(similarity_rows: np.ndarray, members: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k largest similarities from each row to the given members."""
    block = similarity_rows[:, members]
    take = min(k, block.shape[1])
    if take == block.shape[1]:
        return block.mean(axis=1)
    return np.partition(block, -take, axis=1)[:, -take:].mean(axis=1)


@dataclass
class GroupingResult:
    policy: GroupingPolicy
    face_ids: list[str]
    labels: np.ndarray                     # group index per face, or UNGROUPED
    group_sizes: list[int]
    attached: np.ndarray                   # True for faces that joined a group in the attach step
    gated: np.ndarray                      # True for faces that the gate kept out of automatic grouping
    review: list[dict]                     # faces left for review, with candidate groups


def gate_mask(faces: list[dict], policy: GroupingPolicy) -> np.ndarray:
    eye = np.asarray([face["eye_distance"] for face in faces], np.float64)
    passed = eye >= policy.min_eye_distance
    if policy.max_abs_yaw_ratio is not None:
        yaw = np.asarray([abs(face["yaw_ratio"]) if face.get("yaw_ratio") is not None else math.inf for face in faces])
        passed &= yaw <= policy.max_abs_yaw_ratio
    return passed


def group_faces(run: RunData, policy: GroupingPolicy, cannot_link: np.ndarray | None = None,
                similarity: np.ndarray | None = None, merges: list | None = None) -> GroupingResult:
    """Group every usable face in a run. ``cannot_link`` may add user decisions."""
    policy.validate()
    count = len(run.faces)
    passed = gate_mask(run.faces, policy)
    similarity = fused_similarity(run.embeddings, policy) if similarity is None else similarity
    constraints = same_photo_constraints(run.face_photo)
    if cannot_link is not None:
        constraints |= cannot_link
    rows = np.flatnonzero(passed)
    if merges is None:
        merges = agglomerate(similarity[np.ix_(rows, rows)], constraints[np.ix_(rows, rows)], policy.linkage, policy.merge_threshold)
    local = cut(merges, len(rows), policy.merge_threshold)
    return finish_grouping(run, policy, similarity, constraints, rows, local)


def finish_grouping(run: RunData, policy: GroupingPolicy, similarity: np.ndarray, constraints: np.ndarray,
                    rows: np.ndarray, local_labels: np.ndarray) -> GroupingResult:
    count = len(run.faces)
    labels = np.full(count, UNGROUPED, np.int64)
    unique, sizes = np.unique(local_labels, return_counts=True)
    big = [value for value, size in sorted(zip(unique, sizes), key=lambda item: (-item[1], item[0])) if size >= policy.min_group_size]
    index_of = {value: number for number, value in enumerate(big)}
    for row, value in zip(rows, local_labels):
        if value in index_of:
            labels[row] = index_of[value]
    attached = np.zeros(count, bool)
    gated = np.ones(count, bool)
    gated[rows] = False
    members = [np.flatnonzero(labels == group) for group in range(len(big))]
    leftovers = np.flatnonzero(labels == UNGROUPED)
    review: list[dict] = []
    if members and len(leftovers):
        scores = np.stack([top_k_means(similarity[leftovers], group, policy.attach_top_k) for group in members], axis=1)
        for position, face in enumerate(leftovers):
            for group, group_members in enumerate(members):
                if constraints[face, group_members].any():
                    scores[position, group] = -math.inf
        if policy.attach_threshold is not None:
            ordered = sorted(range(len(leftovers)), key=lambda position: -float(np.max(scores[position])))
            for position in ordered:
                face = leftovers[position]
                if gated[face]:
                    continue
                row = scores[position]
                best_group = int(np.argmax(row))
                best = float(row[best_group])
                second = float(np.partition(row, -2)[-2]) if len(row) > 1 else -math.inf
                current = np.flatnonzero(labels == best_group)
                if best >= policy.attach_threshold and best - second >= policy.attach_margin and not constraints[face, current].any():
                    labels[face] = best_group
                    attached[face] = True
        threshold = policy.review_threshold if policy.review_threshold is not None else -math.inf
        for position, face in enumerate(leftovers):
            if labels[face] != UNGROUPED:
                continue
            row = scores[position]
            order = [int(group) for group in np.argsort(-row)[:3] if row[group] >= threshold and math.isfinite(row[group])]
            review.append({"face": run.faces[face]["id"], "photo": run.faces[face]["photo"], "gated": bool(gated[face]),
                           "candidates": [{"group": group, "score": round(float(row[group]), 4)} for group in order]})
    else:
        review = [{"face": run.faces[face]["id"], "photo": run.faces[face]["photo"], "gated": bool(gated[face]), "candidates": []}
                  for face in leftovers]
    sizes_out = [int(np.sum(labels == group)) for group in range(len(big))]
    return GroupingResult(policy, run.face_ids, labels, sizes_out, attached, gated, review)


def group_name(index: int) -> str:
    return f"g{index + 1:04d}"


def groups_document(run: RunData, result: GroupingResult, run_folder: str | None = None) -> dict:
    """The groups file: each group's faces and photos, and each photo's groups."""
    groups = []
    for index, size in enumerate(result.group_sizes):
        members = np.flatnonzero(result.labels == index)
        faces = [run.faces[row] for row in members]
        representative = max(faces, key=lambda face: (face["eye_distance"], face["detection_score"]))
        groups.append({"id": group_name(index), "size": size, "representative_face": representative["id"],
                       "faces": [face["id"] for face in faces], "photos": sorted({face["photo"] for face in faces})})
    photos = {}
    for photo in run.photos:
        photos[photo["path"]] = {"groups": [], "review_faces": [], "faces": len(photo["faces"])}
    for row, label in enumerate(result.labels):
        if label != UNGROUPED:
            entry = photos[run.faces[row]["photo"]]["groups"]
            if group_name(int(label)) not in entry:
                entry.append(group_name(int(label)))
    review = []
    for item in result.review:
        photos[item["photo"]]["review_faces"].append(item["face"])
        review.append({**item, "candidates": [{"group": group_name(candidate["group"]), "score": candidate["score"]}
                                              for candidate in item["candidates"]]})
    for entry in photos.values():
        entry["groups"].sort()
    return {"format": "videoatlas-photo-groups-v1", "run": run_folder, "policy": result.policy.to_json(),
            "embedding_keys": {name: run.keys[name] for name in result.policy.models},
            "group_count": len(groups), "groups": groups, "photos": photos, "review": review,
            "notes": ["A photo with several people appears in each of their groups.",
                      "Faces in the review list did not meet the automatic thresholds; candidates are suggestions only."]}


def write_groups(path: Path, document: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
