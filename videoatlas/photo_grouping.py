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
3. Cluster merge (optional): clusters of one person are often split by
   makeup, age, or lighting. Two clusters of at least ``cluster_min_size``
   faces merge when their mean cross similarity, divided by the geometric mean
   of their own mean internal similarities, reaches ``cluster_merge_threshold``.
   This compares how close two clusters are with how tight each one is.
4. Attach: a face left alone may join a group when the mean of its top-k
   similarities to that group's faces reaches the attach threshold, and the best
   group leads the second-best group by a margin. Otherwise it is listed for
   review with its candidate groups.
5. Suggestions: group pairs whose normalized score reaches
   ``suggest_threshold`` are listed for the user to confirm or reject.

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
    cluster_merge_threshold: float | None = None
    cluster_min_size: int = 3
    # Largest size of the smaller cluster in an automatic cluster merge. Two
    # larger clusters are never merged automatically; they can only become a
    # merge suggestion, so one wrong decision cannot join two whole people.
    cluster_auto_max_size: int | None = None
    suggest_threshold: float | None = None
    max_suggestions: int = 200
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
        if self.cluster_auto_max_size is not None and self.cluster_auto_max_size < 1:
            raise ValueError("cluster_auto_max_size must be at least 1")
        if self.cluster_min_size < 2 or self.max_suggestions < 0:
            raise ValueError("cluster_min_size must be at least 2 and max_suggestions not negative")
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
        vectors = np.asarray(embeddings[name] if rows is None else embeddings[name][rows], np.float32)
        part = vectors @ vectors.T  # in-place steps below keep at most two N x N arrays alive
        part -= np.float32(offset)
        part *= np.float32(weight / scale / sum(weights))
        if total is None:
            total = part
        else:
            total += part
            del part
    return total


def square_block(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """matrix[rows][:, rows]; the matrix itself (no copy) when rows are all rows in order."""
    if len(rows) == matrix.shape[0] and (len(rows) == 0 or (rows[0] == 0 and rows[-1] == len(rows) - 1)):
        return matrix
    return matrix[np.ix_(rows, rows)]


def same_photo_constraints(face_photo: np.ndarray) -> np.ndarray:
    """Boolean matrix: True where two different faces come from the same photo."""
    face_photo = np.asarray(face_photo)
    matrix = face_photo[:, None] == face_photo[None, :]
    np.fill_diagonal(matrix, False)
    return matrix


def agglomerate(similarity: np.ndarray, cannot_link: np.ndarray | None = None, linkage: str = "average",
                floor: float = -math.inf, overwrite: bool = False) -> list[tuple[int, int, float]]:
    """Constrained agglomerative clustering with the nearest-neighbour-chain algorithm.

    Returns merges as (kept representative, absorbed representative, similarity)
    for every merge whose linkage similarity is at least ``floor``. Average and
    complete linkage are reducible, and cannot-link pairs act as infinite
    distance, which keeps them reducible. Cutting the merges at any threshold
    at or above ``floor`` therefore gives the same clusters as running the
    clustering at that threshold. With ``overwrite`` the inputs are used as
    working memory when their types allow it; pass it only for temporary copies.
    """
    count = similarity.shape[0]
    # Sums of similarities; float32 halves memory for large runs (relative error ~1e-7).
    dtype = np.float64 if count <= 8192 else np.float32
    values = np.asarray(similarity, dtype) if overwrite else np.array(similarity, dtype, copy=True)
    if values.shape != (count, count):
        raise ValueError("similarity must be square")
    if cannot_link is None:
        blocked = np.zeros((count, count), bool)
    else:
        blocked = np.asarray(cannot_link, bool) if overwrite else np.array(cannot_link, bool, copy=True)
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


class ClusterSums:
    """Sums of pairwise similarities within and between clusters, for cluster-level scores.

    ``items`` selects the rows and columns of ``similarity`` that take part
    (default: all); ``labels`` gives one cluster label per item. Sums are
    built in row chunks, so no item-by-item copy of the matrix is made.
    """

    def __init__(self, similarity: np.ndarray, constraints: np.ndarray, labels: np.ndarray,
                 items: np.ndarray | None = None, chunk: int = 1024) -> None:
        labels = np.asarray(labels)
        items = np.arange(len(labels)) if items is None else np.asarray(items)
        values, inverse = np.unique(labels, return_inverse=True)
        inverse = inverse.reshape(-1)
        count = len(values)
        order = np.argsort(inverse, kind="stable")
        ordered = items[order]
        counts = np.bincount(inverse, minlength=count)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        partial = np.empty((len(ordered), count), np.float64)
        for begin in range(0, len(ordered), chunk):
            block = np.asarray(similarity[ordered[begin:begin + chunk]][:, ordered], np.float64)
            partial[begin:begin + chunk] = np.add.reduceat(block, starts, axis=1)
        self.cross = np.add.reduceat(partial, starts, axis=0) if len(ordered) else np.zeros((0, 0))
        del partial
        self.diagonal = np.bincount(inverse, weights=np.asarray(similarity[items, items], np.float64), minlength=count)
        position = np.full(similarity.shape[0], -1, np.int64)
        position[items] = inverse
        self.blocked = np.zeros((count, count), bool)
        for begin in range(0, len(items), chunk * 8):
            rows = items[begin:begin + chunk * 8]
            left, right = np.nonzero(constraints[rows])
            right_cluster = position[right]
            keep = right_cluster >= 0
            self.blocked[position[rows[left[keep]]], right_cluster[keep]] = True
        self.sizes = counts.astype(np.float64)
        self.values = values
        self.inverse = inverse

    def scores(self, min_size: int) -> np.ndarray:
        """Normalized score for every cluster pair; -inf where not allowed."""
        sizes = self.sizes
        pairs_inside = sizes * (sizes - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            within = (np.diag(self.cross) - self.diagonal) / pairs_inside
            cross_mean = self.cross / np.outer(sizes, sizes)
            scores = cross_mean / np.sqrt(np.outer(within, within))
        valid = (sizes >= min_size) & (within > 0)
        scores[~np.outer(valid, valid) | self.blocked | ~np.isfinite(scores)] = -math.inf
        np.fill_diagonal(scores, -math.inf)
        return scores


def merge_clusters(similarity: np.ndarray, constraints: np.ndarray, labels: np.ndarray, min_size: int,
                   floor: float | None, max_smaller: int | None = None) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Greedily merge the cluster pair with the highest normalized score, down to ``floor``.

    Returns the cluster index of every item and the merges in order. The order
    does not depend on the final threshold, so any threshold at or above floor
    is applied by taking the merges before the first one below it. Clusters
    smaller than ``min_size`` never take part, so only larger ones are scored.
    With ``max_smaller``, a pair whose smaller cluster has more items than
    that is never merged.
    """
    _, cluster_of, counts = np.unique(labels, return_inverse=True, return_counts=True)
    cluster_of = cluster_of.reshape(-1)
    merges: list[tuple[int, int, float]] = []
    big = np.flatnonzero(counts >= min_size)
    if floor is None or len(big) < 2:
        return cluster_of, merges
    items = np.flatnonzero(np.isin(cluster_of, big))
    sums = ClusterSums(similarity, constraints, cluster_of[items], items)
    ids = sums.values  # cluster index of each local cluster
    count = len(ids)
    active = np.ones(count, bool)
    scores = sums.scores(min_size)
    sizes = sums.sizes
    if max_smaller is not None:
        scores[np.minimum.outer(sizes, sizes) > max_smaller] = -math.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        within = (np.diag(sums.cross) - sums.diagonal) / (sizes * (sizes - 1))
    # Best partner of every cluster, so each step needs no search over all pairs.
    best_index = np.argmax(scores, axis=1)
    best_value = scores[np.arange(count), best_index]
    while True:
        left = int(np.argmax(best_value))
        right = int(best_index[left])
        best = float(best_value[left])
        if not best >= floor:
            break
        keep, drop = min(left, right), max(left, right)
        merges.append((int(ids[keep]), int(ids[drop]), best))
        sums.cross[keep] += sums.cross[drop]
        sums.cross[:, keep] += sums.cross[:, drop]
        sums.diagonal[keep] += sums.diagonal[drop]
        sizes[keep] += sizes[drop]
        sums.blocked[keep] |= sums.blocked[drop]
        sums.blocked[:, keep] = sums.blocked[keep]
        active[drop] = False
        sizes[drop] = 0
        sums.cross[drop] = 0
        sums.cross[:, drop] = 0
        # Only the merged cluster's scores change; update its row and column.
        within[keep] = (sums.cross[keep, keep] - sums.diagonal[keep]) / (sizes[keep] * (sizes[keep] - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            row = sums.cross[keep] / (sizes[keep] * sizes) / np.sqrt(within[keep] * within)
        allowed = active & (sizes >= min_size) & (within > 0) & ~sums.blocked[keep] & np.isfinite(row)
        if max_smaller is not None:
            allowed &= np.minimum(sizes[keep], sizes) <= max_smaller
        if not within[keep] > 0:
            allowed[:] = False
        row[~allowed] = -math.inf
        row[keep] = -math.inf
        scores[keep] = row
        scores[:, keep] = row
        scores[drop] = -math.inf
        scores[:, drop] = -math.inf
        best_value[drop] = -math.inf
        stale = (best_index == keep) | (best_index == drop)
        stale[keep] = True
        stale[drop] = False
        if stale.any():
            rows = np.flatnonzero(stale)
            best_index[rows] = np.argmax(scores[rows], axis=1)
            best_value[rows] = scores[rows, best_index[rows]]
        better = row > best_value
        better[keep] = False
        best_index[better] = keep
        best_value[better] = row[better]
    return cluster_of, merges


def apply_cluster_merges(cluster_of: np.ndarray, merges: list[tuple[int, int, float]], threshold: float | None) -> np.ndarray:
    """Cluster label per item after the merges that come before the first one below threshold."""
    parent = np.arange(int(cluster_of.max()) + 1 if len(cluster_of) else 0)
    if threshold is not None:
        for keep, drop, score in merges:
            if score < threshold:
                break
            parent[parent == drop] = keep
    return parent[cluster_of]


def stage_two(similarity: np.ndarray, constraints: np.ndarray, local_labels: np.ndarray, policy: "GroupingPolicy") -> np.ndarray:
    if policy.cluster_merge_threshold is None or not len(local_labels):
        return local_labels
    cluster_of, merges = merge_clusters(similarity, constraints, local_labels, policy.cluster_min_size, policy.cluster_merge_threshold,
                                        policy.cluster_auto_max_size)
    return apply_cluster_merges(cluster_of, merges, policy.cluster_merge_threshold)


def group_suggestions(similarity: np.ndarray, constraints: np.ndarray, labels: np.ndarray, threshold: float | None,
                      limit: int) -> list[dict]:
    """Group pairs that may be one person, strongest first, for the user to confirm."""
    grouped = np.flatnonzero(labels != UNGROUPED)
    if threshold is None or limit == 0 or not len(grouped):
        return []
    sums = ClusterSums(similarity, constraints, labels[grouped], grouped)
    scores = sums.scores(2)
    upper = np.triu(np.ones_like(scores, dtype=bool), k=1)
    left, right = np.nonzero(upper & (scores >= threshold))
    order = np.argsort(-scores[left, right], kind="stable")[:limit]
    return [{"groups": [int(sums.values[left[index]]), int(sums.values[right[index]])], "score": round(float(scores[left[index], right[index]]), 4)}
            for index in order]


@dataclass
class GroupingResult:
    policy: GroupingPolicy
    face_ids: list[str]
    labels: np.ndarray                     # group index per face, or UNGROUPED
    group_sizes: list[int]
    attached: np.ndarray                   # True for faces that joined a group in the attach step
    gated: np.ndarray                      # True for faces that the gate kept out of automatic grouping
    review: list[dict]                     # faces left for review, with candidate groups
    suggestions: list[dict] = field(default_factory=list)  # group pairs that may be one person


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
    passed = gate_mask(run.faces, policy)
    similarity = fused_similarity(run.embeddings, policy) if similarity is None else similarity
    constraints = same_photo_constraints(run.face_photo)
    if cannot_link is not None:
        constraints |= cannot_link
    rows = np.flatnonzero(passed)
    if merges is None:
        block = square_block(similarity, rows)
        merges = agglomerate(block, square_block(constraints, rows), policy.linkage, policy.merge_threshold,
                             overwrite=block is not similarity)
        del block
    local = cut(merges, len(rows), policy.merge_threshold)
    local = stage_two(square_block(similarity, rows), square_block(constraints, rows), local, policy)
    return finish_grouping(run, policy, similarity, constraints, rows, local)


def finish_grouping(run: RunData, policy: GroupingPolicy, similarity: np.ndarray, constraints: np.ndarray,
                    rows: np.ndarray, local_labels: np.ndarray, cache: dict | None = None,
                    review_candidates: bool = True) -> GroupingResult:
    """Form groups from cluster labels, then attach leftovers, list review faces, and suggest merges.

    ``cache`` may hold attach scores between calls that share every input
    except the attach and suggestion settings. ``review_candidates=False``
    lists review faces without candidate groups, which is faster.
    """
    count = len(run.faces)
    labels = np.full(count, UNGROUPED, np.int64)
    unique, inverse, sizes = np.unique(local_labels, return_inverse=True, return_counts=True)
    order = np.lexsort((unique, -sizes))  # largest first, then by cluster label
    big = order[sizes[order] >= policy.min_group_size]
    index_of = np.full(len(unique), UNGROUPED, np.int64)
    index_of[big] = np.arange(len(big))
    labels[rows] = index_of[inverse.reshape(-1)]
    group_count = len(big)
    attached = np.zeros(count, bool)
    gated = np.ones(count, bool)
    gated[rows] = False
    grouped = np.flatnonzero(labels != UNGROUPED)
    by_group = grouped[np.argsort(labels[grouped], kind="stable")]
    members = np.split(by_group, np.cumsum(np.bincount(labels[grouped], minlength=group_count))[:-1]) if group_count else []
    leftovers = np.flatnonzero(labels == UNGROUPED)
    scores = None
    if group_count and len(leftovers):
        if cache is not None and "scores" in cache:
            scores, group_blocked = cache["scores"], cache["group_blocked"].copy()
        else:
            leftover_rows = similarity[leftovers]
            scores = np.stack([top_k_means(leftover_rows, group, policy.attach_top_k) for group in members], axis=1)
            del leftover_rows
            # A group is closed to a face that is cannot-linked with any of its members.
            group_blocked = np.stack([constraints[group].any(axis=0) for group in members])
            scores[group_blocked[:, leftovers].T] = -math.inf
            if cache is not None:
                cache["scores"], cache["group_blocked"] = scores, group_blocked.copy()
        if policy.attach_threshold is not None:
            if scores.shape[1] > 1:
                top_two = np.partition(scores, -2, axis=1)[:, -2:]
                best_scores, second_scores = top_two[:, 1], top_two[:, 0]
            else:
                best_scores, second_scores = scores[:, 0], np.full(len(leftovers), -math.inf)
            best_groups = np.argmax(scores, axis=1)
            with np.errstate(invalid="ignore"):  # -inf minus -inf: no open group, never eligible
                lead = best_scores - second_scores
            eligible = (~gated[leftovers]) & (best_scores >= policy.attach_threshold) & (lead >= policy.attach_margin)
            for position in sorted(np.flatnonzero(eligible), key=lambda item: -float(best_scores[item])):
                face, group = leftovers[position], int(best_groups[position])
                # Two faces from one photo may both qualify; the stronger one takes the group.
                if group_blocked[group, face]:
                    continue
                labels[face] = group
                attached[face] = True
                group_blocked[group] |= constraints[face]
    remaining = np.flatnonzero(labels[leftovers] == UNGROUPED)
    candidates: list[list[dict]] = [[] for _ in remaining]
    if scores is not None and review_candidates and len(remaining):
        threshold = policy.review_threshold if policy.review_threshold is not None else -math.inf
        block = scores[remaining]
        take = min(3, block.shape[1])
        top = np.argpartition(-block, take - 1, axis=1)[:, :take] if block.shape[1] > take else np.tile(np.arange(take), (len(remaining), 1))
        top_scores = np.take_along_axis(block, top, axis=1)
        ranking = np.argsort(-top_scores, axis=1, kind="stable")
        top, top_scores = np.take_along_axis(top, ranking, axis=1), np.take_along_axis(top_scores, ranking, axis=1)
        for index in range(len(remaining)):
            candidates[index] = [{"group": int(group), "score": round(float(score), 4)} for group, score in zip(top[index], top_scores[index])
                                 if score >= threshold and math.isfinite(score)]
    review = [{"face": run.faces[face]["id"], "photo": run.faces[face]["photo"], "gated": bool(gated[face]), "candidates": candidates[index]}
              for index, face in enumerate(leftovers[remaining])]
    sizes_out = np.bincount(labels[labels != UNGROUPED], minlength=group_count).astype(int).tolist()
    suggestions = group_suggestions(similarity, constraints, labels, policy.suggest_threshold, policy.max_suggestions)
    return GroupingResult(policy, run.face_ids, labels, sizes_out, attached, gated, review, suggestions)


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
            "merge_suggestions": [{"groups": [group_name(item["groups"][0]), group_name(item["groups"][1])], "score": item["score"]}
                                  for item in result.suggestions],
            "notes": ["A photo with several people appears in each of their groups.",
                      "Faces in the review list did not meet the automatic thresholds; candidates are suggestions only.",
                      "Merge suggestions name group pairs that may be one person; only the user's confirmation merges them."]}


def write_groups(path: Path, document: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
