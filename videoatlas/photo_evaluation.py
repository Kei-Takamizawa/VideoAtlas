"""Accuracy measurement for photo grouping against a folder-per-person answer key.

Answer key rules
- A face gets its folder's label only when it is the single usable face in a
  labelled photo whose file content (SHA-256) is not also filed under another
  person. Faces in multi-face photos stay unlabelled, because the folder does
  not say which face is the subject.
- A labelled photo is "found" when any of its faces is in the person's group.

Error definitions (the user's two failure modes)
- Mixing: a labelled face inside a group whose majority label is a different
  person. Reported per face and as the number of groups with two or more people.
- Missing: a labelled photo of a person that does not appear in that person's
  main group (the group holding most of the person's labelled faces). Photos
  with no usable face, photos left for review, and photos in another group of
  the same person are counted separately so the cause is visible.

Thresholds chosen on the same people they are evaluated on look better than
they are. ``calibrate_leave_one_person_out`` chooses thresholds without the
person being evaluated, then counts only errors that involve that person.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np

from .photo_grouping import UNGROUPED, GroupingPolicy, GroupingResult, agglomerate, apply_cluster_merges, cut, finish_grouping, \
    fused_similarity, gate_mask, merge_clusters, same_photo_constraints
from .photos import RunData

FMR_TARGETS = (1e-2, 1e-3, 1e-4)
HISTOGRAM_BINS = 4000


# ---------------------------------------------------------------- answer key

def conflicting_hashes(run: RunData) -> set[str]:
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for photo in run.photos:
        if photo.get("label"):
            labels_by_hash[photo["sha256"]].add(photo["label"])
    return {digest for digest, labels in labels_by_hash.items() if len(labels) > 1}


def face_labels(run: RunData) -> np.ndarray:
    """Folder label per usable face, or None where the answer key is ambiguous."""
    conflicts = conflicting_hashes(run)
    usable_count = Counter(int(index) for index in run.face_photo)
    labels = []
    for face, photo_index in zip(run.faces, run.face_photo):
        photo = run.photos[int(photo_index)]
        clear = photo.get("label") and usable_count[int(photo_index)] == 1 and photo["sha256"] not in conflicts
        labels.append(photo["label"] if clear else None)
    return np.asarray(labels, dtype=object)


def photo_labels(run: RunData) -> dict[int, str]:
    conflicts = conflicting_hashes(run)
    return {index: photo["label"] for index, photo in enumerate(run.photos) if photo.get("label") and photo["sha256"] not in conflicts}


# ------------------------------------------------------- verification scores

def _scores_by_identity(vectors: np.ndarray, codes: np.ndarray, identities: int) -> dict[tuple[int, int], np.ndarray]:
    """Cosine scores for every unordered face pair, grouped by identity pair."""
    result = {}
    for left in range(identities):
        a = vectors[codes == left]
        for right in range(left, identities):
            b = vectors[codes == right]
            scores = a @ b.T
            if left == right:
                scores = scores[np.triu_indices(len(a), k=1)]
            result[(left, right)] = scores.ravel()
    return result


def _histogram(scores: np.ndarray) -> np.ndarray:
    return np.histogram(np.clip(scores, -1, 1), bins=HISTOGRAM_BINS, range=(-1, 1))[0].astype(np.float64)


def _curve_metrics(positive: np.ndarray, negative: np.ndarray) -> dict:
    """AUC, EER, and TAR at fixed false-match rates from score histograms."""
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return {"auc": None, "eer": None, **{f"tar_at_fmr_{target:g}": None for target in FMR_TARGETS}}
    # Scores at or above bin i: accept.
    tar = np.cumsum(positive[::-1])[::-1] / total_positive
    fmr = np.cumsum(negative[::-1])[::-1] / total_negative
    below = np.concatenate([[0.0], np.cumsum(negative)[:-1]])
    auc = float(np.sum(positive * (below + negative / 2)) / (total_positive * total_negative))
    edges = np.linspace(-1, 1, HISTOGRAM_BINS + 1)
    frr = 1 - tar
    crossing = int(np.argmin(np.abs(fmr - frr)))
    result = {"auc": auc, "eer": float((fmr[crossing] + frr[crossing]) / 2), "eer_threshold": float(edges[crossing])}
    for target in FMR_TARGETS:
        allowed = np.flatnonzero(fmr <= target)
        index = int(allowed[0]) if len(allowed) else None
        result[f"tar_at_fmr_{target:g}"] = float(tar[index]) if index is not None else 0.0
        result[f"threshold_at_fmr_{target:g}"] = float(edges[index]) if index is not None else None
    return result


def _keep_and_codes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.flatnonzero(labels != None)  # noqa: E711 - object array comparison
    names, codes = np.unique(labels[keep].astype(str), return_inverse=True)
    return keep, names, codes


def _pool(histograms: dict, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Positive and negative histograms for a multiset of people (pairs between copies of one person skipped)."""
    identities = len(counts)
    positive = [counts[i] * histograms[(i, i)] for i in range(identities) if counts[i]]
    negative = [counts[i] * counts[j] * histograms[(i, j)] for i, j in histograms if i != j and counts[i] and counts[j]]
    if not positive or not negative:
        return None
    return np.sum(positive, axis=0), np.sum(negative, axis=0)


def verification_report(run: RunData, model: str, labels: np.ndarray | None = None,
                        eye_bins: Iterable[float] = (0, 15, 20, 30, 40, math.inf)) -> dict:
    """Single-face pair separability for one model, overall and by face size."""
    labels = face_labels(run) if labels is None else labels
    keep, names, codes = _keep_and_codes(labels)
    vectors = run.embeddings[model][keep]
    eye = np.asarray([run.faces[row]["eye_distance"] for row in keep])
    histograms = {pair: _histogram(scores) for pair, scores in _scores_by_identity(vectors, codes, len(names)).items()}
    positive, negative = _pool(histograms, np.ones(len(names)))
    report = {"model": model, "key": run.keys.get(model), "faces": int(len(keep)), "people": len(names),
              "positive_pairs": int(positive.sum()), "negative_pairs": int(negative.sum()), **_curve_metrics(positive, negative)}
    scores = vectors @ vectors.T
    upper = np.triu_indices(len(keep), k=1)
    pair_scores = scores[upper]
    same = codes[upper[0]] == codes[upper[1]]
    smaller = np.minimum(eye[upper[0]], eye[upper[1]])
    strata = []
    bins = list(eye_bins)
    for low, high in zip(bins, bins[1:]):
        mask = (smaller >= low) & (smaller < high)
        pos, neg = _histogram(pair_scores[mask & same]), _histogram(pair_scores[mask & ~same])
        faces_in_bin = int(np.sum((eye >= low) & (eye < high)))
        strata.append({"smaller_eye_distance_px": [low, None if math.isinf(high) else high], "faces": faces_in_bin,
                       "positive_pairs": int(pos.sum()), "negative_pairs": int(neg.sum()), **_curve_metrics(pos, neg)})
    report["by_smaller_eye_distance"] = strata
    quantiles = (.01, .05, .25, .5, .75, .95, .99)
    for name, mask in (("positive", same), ("negative", ~same)):
        values = pair_scores[mask]
        report[f"{name}_score_quantiles"] = {f"p{int(q * 100):02d}": float(v) for q, v in zip(quantiles, np.quantile(values, quantiles))} if values.size else None
    people = []
    for code, name in enumerate(names):
        own = codes == code
        if own.sum() < 2:
            continue
        block = scores[np.ix_(own, own)][np.triu_indices(int(own.sum()), k=1)]
        other = scores[np.ix_(own, ~own)].max(axis=1)
        people.append({"person": name, "faces": int(own.sum()), "median_same_person_score": float(np.median(block)),
                       "median_best_other_person_score": float(np.median(other))})
    report["per_person"] = people
    return report


def compare_models(run: RunData, models: list[str], labels: np.ndarray | None = None, repetitions: int = 1000, seed: int = 0) -> dict:
    """Paired person-bootstrap comparison: every model sees the same resampled people.

    Faces are the single usable faces of labelled photos, so every model is
    compared on exactly the same faces only when the run analyzed them together.
    """
    labels = face_labels(run) if labels is None else labels
    keep, names, codes = _keep_and_codes(labels)
    histograms = {}
    for model in models:
        vectors = run.embeddings[model][keep]
        histograms[model] = {pair: _histogram(scores) for pair, scores in _scores_by_identity(vectors, codes, len(names)).items()}
    keys = ("auc", "eer", *(f"tar_at_fmr_{target:g}" for target in FMR_TARGETS))
    point = {model: {key: _curve_metrics(*_pool(histograms[model], np.ones(len(names))))[key] for key in keys} for model in models}
    generator = np.random.default_rng(seed)
    samples = {model: {key: [] for key in keys} for model in models}
    for _ in range(repetitions):
        counts = np.bincount(generator.integers(0, len(names), len(names)), minlength=len(names)).astype(np.float64)
        pooled = {model: _pool(histograms[model], counts) for model in models}
        if any(value is None for value in pooled.values()):
            continue
        for model in models:
            metrics = _curve_metrics(*pooled[model])
            for key in keys:
                samples[model][key].append(metrics[key])
    reference = models[0]
    summary = {"faces": int(len(keep)), "people": len(names), "repetitions": len(samples[reference]["auc"]), "seed": seed,
               "reference": reference, "models": {}}
    for model in models:
        entry = {}
        for key in keys:
            values = np.asarray(samples[model][key], np.float64)
            ref = np.asarray(samples[reference][key], np.float64)
            entry[key] = {"value": point[model][key],
                          "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values.size else None}
            if model != reference and values.size:
                difference = values - ref
                entry[key]["difference_vs_reference"] = point[model][key] - point[reference][key]
                entry[key]["difference_ci95"] = [float(np.quantile(difference, .025)), float(np.quantile(difference, .975))]
        summary["models"][model] = entry
    summary["note"] = ("People are resampled with replacement. With few people the intervals describe variation among "
                       "these people only, not a population.")
    return summary


# ------------------------------------------------------------ grouping metrics

REVIEW_BUDGETS = (0, 10, 25, 50, 100, 200)


def _majorities(group_of_face: np.ndarray, labels: np.ndarray) -> tuple[dict[int, Counter], dict[int, str]]:
    counters: dict[int, Counter] = defaultdict(Counter)
    for row, group in enumerate(group_of_face):
        if group != UNGROUPED and labels[row] is not None:
            counters[int(group)][labels[row]] += 1
    return counters, {group: counter.most_common(1)[0][0] for group, counter in counters.items()}


def _people_counts(run: RunData, group_of_face: np.ndarray, labels: np.ndarray, photos: dict[int, str],
                   focus: set[str], majority: dict[int, str], sizes: dict[int, int]) -> dict:
    """Where each labelled photo of each person ended up. Ties for the main group go to the lower group id."""
    photo_count = len(run.photos)
    face_photo = np.asarray(run.face_photo, np.int64)
    grouped = group_of_face != UNGROUPED
    has_face = np.bincount(face_photo, minlength=photo_count) > 0
    in_any = np.bincount(face_photo, weights=grouped, minlength=photo_count) > 0
    photo_ids = np.fromiter(photos.keys(), np.int64, len(photos))
    photo_people = np.asarray(list(photos.values()), dtype=object)
    per_person = {}
    for person in sorted(focus):
        values, counts = np.unique(group_of_face[grouped & (labels == person)], return_counts=True)
        main = int(values[np.argmax(counts)]) if len(values) else None
        person_groups = [int(group) for group in values if majority.get(int(group)) == person]
        in_main = np.bincount(face_photo, weights=group_of_face == main, minlength=photo_count) > 0 if main is not None \
            else np.zeros(photo_count, bool)
        in_own = np.bincount(face_photo, weights=np.isin(group_of_face, person_groups), minlength=photo_count) > 0
        mine = photo_ids[photo_people == person]
        no_face = ~has_face[mine]
        found = ~no_face & in_main[mine]
        other_own = ~no_face & ~found & in_own[mine]
        others = ~no_face & ~found & ~other_own & in_any[mine]
        counts_ = {"photos": int(len(mine)), "no_usable_face": int(no_face.sum()), "found_in_main_group": int(found.sum()),
                   "in_other_group_of_person": int(other_own.sum()), "only_in_other_people_groups": int(others.sum())}
        counts_["left_for_review"] = counts_["photos"] - sum(counts_[key] for key in counts_ if key != "photos")
        total = counts_["photos"]
        per_person[person] = {**counts_, "groups_led": len(person_groups), "main_group_size": sizes.get(main, 0) if main is not None else 0,
                              "missing_rate": _ratio(total - counts_["found_in_main_group"], total)}
    return per_person


def _missing_summary(per_person: dict) -> dict:
    total = sum(entry["photos"] for entry in per_person.values())
    found = sum(entry["found_in_main_group"] for entry in per_person.values())
    no_face = sum(entry["no_usable_face"] for entry in per_person.values())
    return {"photos": total, "found_in_main_group": found, "missing_rate": _ratio(total - found, total), "no_usable_face": no_face,
            "missing_rate_excluding_no_face": _ratio(total - no_face - found, total - no_face),
            "left_for_review": sum(entry["left_for_review"] for entry in per_person.values()),
            "in_other_group_of_person": sum(entry["in_other_group_of_person"] for entry in per_person.values()),
            "only_in_other_people_groups": sum(entry["only_in_other_people_groups"] for entry in per_person.values())}


def simulate_review(run: RunData, result: GroupingResult, labels: np.ndarray, photos: dict[int, str], focus: set[str],
                    budgets: Iterable[int] = REVIEW_BUDGETS) -> list[dict]:
    """A user who answers the first N merge suggestions correctly, using the answer key.

    A suggestion counts as correct when both groups are led by the same person.
    Rejected suggestions are wasted effort; accepted ones merge the groups.
    Faces in the review list are not counted here.
    """
    _, majority = _majorities(result.labels, labels)
    rows = []
    for budget in budgets:
        parent = {group: group for group in range(len(result.group_sizes))}

        def find(group: int) -> int:
            while parent[group] != group:
                group = parent[group]
            return group

        accepted = rejected = 0
        for item in result.suggestions[:budget]:
            left, right = item["groups"]
            if majority.get(left) is not None and majority.get(left) == majority.get(right):
                a, b = find(left), find(right)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                accepted += 1
            else:
                rejected += 1
        merged = np.asarray([find(int(group)) if group != UNGROUPED else UNGROUPED for group in result.labels])
        _, merged_majority = _majorities(merged, labels)
        sizes = Counter(int(group) for group in merged if group != UNGROUPED)
        summary = _missing_summary(_people_counts(run, merged, labels, photos, focus, merged_majority, sizes))
        rows.append({"budget": budget, "suggestions_answered": min(budget, len(result.suggestions)), "accepted": accepted,
                     "rejected": rejected, "photos": summary["photos"], "found_in_main_group": summary["found_in_main_group"],
                     "missing_rate": summary["missing_rate"], "in_other_group_of_person": summary["in_other_group_of_person"]})
    return rows


def grouping_metrics(run: RunData, result: GroupingResult, labels: np.ndarray | None = None,
                     people: Iterable[str] | None = None, review: bool = False, photos: dict[int, str] | None = None) -> dict:
    """Mixing and missing for a grouping result. ``people`` limits the people evaluated."""
    labels = face_labels(run) if labels is None else labels
    photos = photo_labels(run) if photos is None else photos
    focus = set(people) if people is not None else {label for label in photos.values()}
    counters, majority = _majorities(result.labels, labels)
    path_of = [face["photo"] for face in run.faces]
    mixed_faces = grouped_labelled = mixed_groups = 0
    mixed_detail = []
    for group, counter in counters.items():
        top = majority[group]
        if top not in focus and not (set(counter) - {top}) & focus:
            continue
        intruders = sum(count for label, count in counter.items() if label != top and (label in focus or top in focus))
        grouped_labelled += sum(count for label, count in counter.items() if label in focus or top in focus)
        mixed_faces += intruders
        if intruders:
            mixed_groups += 1
            rows = np.flatnonzero(result.labels == group)
            mixed_detail.append({"group": group, "size": result.group_sizes[group], "majority": top,
                                 "intruders": {label: count for label, count in counter.items() if label != top},
                                 "intruder_photos": sorted(path_of[row] for row in rows if labels[row] is not None and labels[row] != top)[:50]})
    sizes = {group: size for group, size in enumerate(result.group_sizes)}
    per_person = _people_counts(run, result.labels, labels, photos, focus, majority, sizes)
    metrics = {"groups": len(result.group_sizes), "grouped_faces": int(np.sum(result.labels != UNGROUPED)),
               "attached_faces": int(result.attached.sum()), "review_faces": len(result.review), "merge_suggestions": len(result.suggestions),
               "mixing": {"mixed_faces": mixed_faces, "grouped_labelled_faces": grouped_labelled,
                          "mixing_rate": _ratio(mixed_faces, grouped_labelled), "mixed_groups": mixed_groups,
                          "details": sorted(mixed_detail, key=lambda item: -sum(item["intruders"].values()))[:20]},
               "missing": {**_missing_summary(per_person), "mean_person_missing_rate": _mean([entry["missing_rate"] for entry in per_person.values()])},
               "people": per_person}
    if review and result.suggestions:
        metrics["review_simulation"] = simulate_review(run, result, labels, photos, focus)
    return metrics


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------- calibration

def _subset(run: RunData, rows: np.ndarray) -> RunData:
    photo_indexes = sorted({int(run.face_photo[row]) for row in rows})
    remap = {old: new for new, old in enumerate(photo_indexes)}
    return RunData([run.photos[index] for index in photo_indexes], [run.faces[row] for row in rows],
                   np.asarray([remap[int(run.face_photo[row])] for row in rows], np.int64),
                   {name: matrix[rows] for name, matrix in run.embeddings.items()}, run.keys, run.report)


@dataclass(frozen=True)
class SearchGrid:
    """Candidate settings. None in cluster or attach values means the step is off."""
    merge: tuple[float, ...]
    attach: tuple[float | None, ...] = (None,)
    eye: tuple[float, ...] = (0.0,)
    cluster: tuple[float | None, ...] = (None,)

    @property
    def merge_floor(self) -> float:
        return min(self.merge)

    @property
    def cluster_floor(self) -> float | None:
        values = [value for value in self.cluster if value is not None]
        return min(values) if values else None


class _Grid:
    """Caches each clustering step so that every grid setting is cheap to evaluate."""

    def __init__(self, run: RunData, base: GroupingPolicy, grid: SearchGrid) -> None:
        self.run, self.base, self.grid = run, base, grid
        self.similarity = fused_similarity(run.embeddings, base)
        self.constraints = same_photo_constraints(run.face_photo)
        self.trees: dict[float, tuple[np.ndarray, list]] = {}
        self.cluster_trees: dict[tuple[float, float], tuple[np.ndarray, list]] = {}
        self.cuts: dict[tuple[float, float], np.ndarray] = {}
        self.attach_scores: dict[tuple, dict] = {}
        self._block_eye: float | None = None
        self._block: tuple[np.ndarray, np.ndarray] | None = None

    def _rows(self, eye: float) -> tuple[np.ndarray, list]:
        if eye not in self.trees:
            rows = np.flatnonzero(gate_mask(self.run.faces, replace(self.base, min_eye_distance=eye)))
            merges = agglomerate(self.similarity[np.ix_(rows, rows)], self.constraints[np.ix_(rows, rows)], self.base.linkage, self.grid.merge_floor)
            self.trees[eye] = (rows, merges)
        return self.trees[eye]

    def _blocks(self, eye: float) -> tuple[np.ndarray, np.ndarray]:
        if self._block_eye != eye:  # keep one gated copy at a time; the grid loops over eye first
            rows, _ = self._rows(eye)
            self._block = (self.similarity[np.ix_(rows, rows)], self.constraints[np.ix_(rows, rows)])
            self._block_eye = eye
        return self._block

    def local_labels(self, eye: float, merge: float, cluster: float | None) -> tuple[np.ndarray, np.ndarray]:
        rows, merges = self._rows(eye)
        if (eye, merge) not in self.cuts:
            self.cuts[(eye, merge)] = cut(merges, len(rows), merge)
        local = self.cuts[(eye, merge)]
        if cluster is None:
            return rows, local
        key = (eye, merge)
        if key not in self.cluster_trees:
            similarity, constraints = self._blocks(eye)
            self.cluster_trees[key] = merge_clusters(similarity, constraints, local, self.base.cluster_min_size, self.grid.cluster_floor)
        cluster_of, cluster_merges = self.cluster_trees[key]
        return rows, apply_cluster_merges(cluster_of, cluster_merges, cluster)

    def result(self, policy: GroupingPolicy, suggestions: bool = True) -> GroupingResult:
        rows, local = self.local_labels(policy.min_eye_distance, policy.merge_threshold, policy.cluster_merge_threshold)
        if not suggestions:
            policy = replace(policy, suggest_threshold=None)
        key = (policy.min_eye_distance, policy.merge_threshold, policy.cluster_merge_threshold)
        if key not in self.attach_scores:
            self.attach_scores = {key: {}}  # the grid loops over attach last, so one entry is enough
        return finish_grouping(self.run, policy, self.similarity, self.constraints, rows, local, self.attach_scores[key],
                               review_candidates=suggestions)

    def policies(self):
        for eye in self.grid.eye:
            for merge in self.grid.merge:
                for cluster in self.grid.cluster:
                    for attach in self.grid.attach:
                        yield replace(self.base, min_eye_distance=eye, merge_threshold=merge, cluster_merge_threshold=cluster,
                                      attach_threshold=attach)


def _rates(metrics: dict) -> tuple[float, float]:
    mixing, missing = metrics["mixing"]["mixing_rate"], metrics["missing"]["missing_rate"]
    return (0.0 if mixing is None else mixing, 1.0 if missing is None else missing)


def _conservative(policy: GroupingPolicy) -> tuple:
    """Tie-breaker: prefer the stricter setting."""
    return (-policy.merge_threshold, -(policy.cluster_merge_threshold if policy.cluster_merge_threshold is not None else math.inf),
            -(policy.attach_threshold if policy.attach_threshold is not None else math.inf), -policy.min_eye_distance)


def evaluate_grid(run: RunData, base: GroupingPolicy, grid: SearchGrid, labels: np.ndarray | None = None) -> list[tuple[GroupingPolicy, dict]]:
    labels = face_labels(run) if labels is None else labels
    photos = photo_labels(run)
    cache = _Grid(run, base, grid)
    return [(policy, grouping_metrics(run, cache.result(policy, suggestions=False), labels, photos=photos)) for policy in cache.policies()]


def pick(table: list[tuple[GroupingPolicy, dict]], max_mixing_rate: float) -> tuple[GroupingPolicy | None, dict | None]:
    """Fewest missing photos among settings within the mixing limit; ties go to the stricter setting."""
    allowed = [(policy, metrics) for policy, metrics in table if _rates(metrics)[0] <= max_mixing_rate]
    if not allowed:
        return None, None
    return min(allowed, key=lambda item: (_rates(item[1])[1], _rates(item[1])[0], _conservative(item[0])))


def choose_policy(run: RunData, base: GroupingPolicy, grid: SearchGrid, max_mixing_rate: float,
                  labels: np.ndarray | None = None) -> tuple[GroupingPolicy | None, dict | None, list[dict]]:
    table = evaluate_grid(run, base, grid, labels)
    policy, metrics = pick(table, max_mixing_rate)
    return policy, metrics, [table_row(policy_, metrics_) for policy_, metrics_ in table]


def table_row(policy: GroupingPolicy, metrics: dict) -> dict:
    mixing, missing = _rates(metrics)
    return {"min_eye_distance": policy.min_eye_distance, "merge_threshold": policy.merge_threshold,
            "cluster_merge_threshold": policy.cluster_merge_threshold, "attach_threshold": policy.attach_threshold,
            "mixing_rate": mixing, "mixed_groups": metrics["mixing"]["mixed_groups"], "missing_rate": missing, "groups": metrics["groups"]}


def _setting(policy: GroupingPolicy) -> dict:
    return {"min_eye_distance": policy.min_eye_distance, "merge_threshold": policy.merge_threshold,
            "cluster_merge_threshold": policy.cluster_merge_threshold, "attach_threshold": policy.attach_threshold}


def _fold_base(run: RunData, base: GroupingPolicy) -> GroupingPolicy:
    """Score normalization for fused models comes only from the people used for choosing."""
    if len(base.models) < 2:
        return base
    offsets, scales = impostor_normalization(run, base.models)
    return replace(base, offsets=offsets, scales=scales)


_HELD_OUT_COUNTS = ("photos", "found_in_main_group", "no_usable_face", "left_for_review", "in_other_group_of_person",
                    "only_in_other_people_groups")


def _review_totals(rows: list[dict]) -> list[dict]:
    """Sum the review simulation of several folds, budget by budget."""
    by_budget: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        for key in ("suggestions_answered", "accepted", "rejected", "photos", "found_in_main_group"):
            by_budget[row["budget"]][key] += row[key]
    return [{"budget": budget, **counts, "missing_rate": _ratio(counts["photos"] - counts["found_in_main_group"], counts["photos"])}
            for budget, counts in sorted(by_budget.items())]


def calibrate_leave_one_person_out(run: RunData, base: GroupingPolicy, grid: SearchGrid, max_mixing_rates: Iterable[float]) -> dict:
    """Choose settings without each person in turn, then measure errors that involve that person.

    Every mixing limit in ``max_mixing_rates`` is evaluated from the same grid,
    so the trade-off between mixing and missing is visible. The held-out
    person's photos are grouped together with everyone else's, as in real use.
    """
    labels = face_labels(run)
    photos = photo_labels(run)
    people = sorted(set(photos.values()))
    targets = sorted(set(max_mixing_rates))
    full_cache: dict[tuple, _Grid] = {}
    folds: dict[float, list[dict]] = {target: [] for target in targets}
    for person in people:
        train_rows = np.flatnonzero(np.asarray([photos.get(int(index)) != person for index in run.face_photo]))
        train = _subset(run, train_rows)
        fold_base = _fold_base(train, base)
        table = evaluate_grid(train, fold_base, grid)
        key = (fold_base.offsets, fold_base.scales)
        if key not in full_cache:
            full_cache[key] = _Grid(run, fold_base, grid)
        for target in targets:
            policy, train_metrics = pick(table, target)
            if policy is None:
                folds[target].append({"held_out": person, "policy": None, "note": "No setting met the mixing limit on the other people."})
                continue
            test = grouping_metrics(run, full_cache[key].result(policy), labels, people=[person], review=True)
            folds[target].append({"held_out": person, "policy": _setting(policy),
                                  "train_mixing_rate": train_metrics["mixing"]["mixing_rate"],
                                  "train_missing_rate": train_metrics["missing"]["missing_rate"],
                                  "mixed_faces": test["mixing"]["mixed_faces"],
                                  "grouped_labelled_faces": test["mixing"]["grouped_labelled_faces"],
                                  "mixed_groups": test["mixing"]["mixed_groups"], "mixing_details": test["mixing"]["details"],
                                  "person": test["people"][person], "merge_suggestions": test["merge_suggestions"],
                                  "review_simulation": test.get("review_simulation") or []})
    results = {}
    for target in targets:
        measured = [fold for fold in folds[target] if fold["policy"] is not None]
        totals = Counter()
        for fold in measured:
            totals["mixed_faces"] += fold["mixed_faces"]
            totals["grouped_labelled_faces"] += fold["grouped_labelled_faces"]
            for name in _HELD_OUT_COUNTS:
                totals[name] += fold["person"][name]
        held_out = {"mixed_faces": totals["mixed_faces"], "grouped_labelled_faces": totals["grouped_labelled_faces"],
                    "mixing_rate": _ratio(totals["mixed_faces"], totals["grouped_labelled_faces"]),
                    **{name: totals[name] for name in _HELD_OUT_COUNTS},
                    "missing_rate": _ratio(totals["photos"] - totals["found_in_main_group"], totals["photos"]),
                    "folds_without_setting": len(folds[target]) - len(measured),
                    "after_review": _review_totals([row for fold in measured for row in fold["review_simulation"]])}
        results[str(target)] = {"folds": folds[target], "held_out": held_out}
    return {"method": "leave-one-person-out: settings chosen on the other people; errors counted only where the held-out person is involved",
            "targets": results}


def impostor_normalization(run: RunData, models: Iterable[str], labels: np.ndarray | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-model mean and standard deviation of different-person scores, for score fusion."""
    labels = face_labels(run) if labels is None else labels
    keep = np.flatnonzero(labels != None)  # noqa: E711
    codes = np.unique(labels[keep].astype(str), return_inverse=True)[1]
    upper = np.triu_indices(len(keep), k=1)
    different = codes[upper[0]] != codes[upper[1]]
    offsets, scales = [], []
    for model in models:
        vectors = run.embeddings[model][keep]
        scores = (vectors @ vectors.T)[upper][different]
        offsets.append(float(scores.mean()))
        scales.append(float(scores.std()) or 1.0)
    return tuple(offsets), tuple(scales)


def add_fused_embedding(run: RunData, models: list[str]) -> str:
    """Add a combined embedding whose dot product ranks pairs like the fused grouping score.

    Each model's unit vector is scaled by the square root of a weight
    proportional to 1 / (spread of its different-person scores), and the
    parts are joined. The dot product is then a weighted mean of cosines,
    which orders pairs exactly like ``fused_similarity`` with equal weights,
    so AUC and TAR at a false-match rate match; thresholds do not. The spread
    is measured on all labelled people, which flatters the result a little.
    """
    name = ",".join(models)
    _, scales = impostor_normalization(run, models)
    weights = np.asarray([1.0 / scale for scale in scales])
    weights /= weights.sum()
    run.embeddings[name] = np.concatenate([run.embeddings[model] * np.float32(math.sqrt(weight))
                                           for model, weight in zip(models, weights)], axis=1).astype(np.float32)
    run.keys[name] = "fused(" + ";".join(f"{run.keys.get(model)}*{weight:.4f}" for model, weight in zip(models, weights)) + ")"
    return name


# ------------------------------------------------- known-people identification

def identification_report(run: RunData, model: str, enroll_sizes: Iterable[int] = (3, 5, 10, 20), repeats: int = 5,
                          top_m: int = 3, margin: float = .05, thresholds: Iterable[float] | None = None,
                          max_wrong_rate: float = .001, seed: int = 0) -> dict:
    """Open-set identification against a small gallery of named example photos.

    This measures the alternative to unsupervised grouping: the user names a few
    photos per person, and each new face is matched to the closest person, or
    to nobody. Every person takes a turn as an unknown person who has no
    gallery, so accepting any of their faces counts as a wrong assignment.
    Gallery and query photos come from the same folders, so near-duplicate
    photos can make the results optimistic.
    """
    labels = face_labels(run)
    keep, names, codes = _keep_and_codes(labels)
    vectors = run.embeddings[model][keep]
    similarity = vectors @ vectors.T
    thresholds = list(thresholds) if thresholds is not None else [round(value, 3) for value in np.arange(.10, .901, .025)]
    generator = np.random.default_rng(seed)
    members = [np.flatnonzero(codes == code) for code in range(len(names))]
    sizes = {}
    for size in enroll_sizes:
        counts = np.zeros((len(thresholds), 4), np.int64)  # known correct, known wrong, known total, unknown accepted
        unknown_total = 0
        for _ in range(repeats):
            gallery = [generator.choice(indexes, size, replace=False) if len(indexes) > size else None for indexes in members]
            if any(item is None for item in gallery):
                continue
            in_gallery = np.zeros(len(keep), bool)
            for item in gallery:
                in_gallery[item] = True
            queries = np.flatnonzero(~in_gallery)
            take = min(top_m, size)
            person_scores = np.stack([np.partition(similarity[np.ix_(queries, item)], -take, axis=1)[:, -take:].mean(axis=1)
                                      for item in gallery], axis=1)
            truth = codes[queries]
            for unknown in range(len(names)):
                scores = person_scores.copy()
                scores[:, unknown] = -math.inf
                order = np.argsort(-scores, axis=1)
                best_person = order[:, 0]
                best = scores[np.arange(len(queries)), best_person]
                second = scores[np.arange(len(queries)), order[:, 1]] if scores.shape[1] > 2 else np.full(len(queries), -math.inf)
                is_unknown = truth == unknown
                unknown_total += int(is_unknown.sum())
                for row, threshold in enumerate(thresholds):
                    accepted = (best >= threshold) & (best - second >= margin)
                    counts[row, 0] += int(np.sum(accepted & ~is_unknown & (best_person == truth)))
                    counts[row, 1] += int(np.sum(accepted & ~is_unknown & (best_person != truth)))
                    counts[row, 2] += int(np.sum(~is_unknown))
                    counts[row, 3] += int(np.sum(accepted & is_unknown))
        table = []
        for threshold, (correct, wrong, known, unknown_accepted) in zip(thresholds, counts):
            total = known + unknown_total
            table.append({"threshold": threshold, "correct_rate": _ratio(correct, known),
                          "wrong_rate": _ratio(wrong + unknown_accepted, total),
                          "known_wrong": int(wrong), "unknown_accepted": int(unknown_accepted), "queries": int(total)})
        safe = [row for row in table if row["wrong_rate"] is not None and row["wrong_rate"] <= max_wrong_rate]
        sizes[str(size)] = {"best_within_limit": max(safe, key=lambda row: row["correct_rate"] or 0) if safe else None, "table": table}
    return {"model": model, "people": len(names), "faces": int(len(keep)), "repeats": repeats, "top_m": top_m, "margin": margin,
            "max_wrong_rate": max_wrong_rate, "enroll_sizes": sizes,
            "note": "Threshold chosen on the same people it is measured on, and gallery photos may be near-duplicates of queries: optimistic."}
