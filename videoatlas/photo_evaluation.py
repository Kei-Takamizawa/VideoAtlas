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

from .photo_grouping import UNGROUPED, GroupingPolicy, GroupingResult, add_leftover_groups, agglomerate, apply_cluster_merges, cut, \
    attach_to_new_groups, existing_group_scores, finish_grouping, fused_similarity, gate_mask, group_faces, group_means, \
    initial_groups, known_cluster_scores, leftover_merges, mean_ratios, merge_clusters, same_photo_constraints, square_block, top_k_means
from .photos import RunData

FMR_TARGETS = (1e-2, 1e-3, 1e-4)
HISTOGRAM_BINS = 4000


# ---------------------------------------------------------------- answer key

def _known_label(photo: dict) -> str | None:
    """Folder label, including labels hidden from tuning (they still mark shared photos)."""
    return photo.get("label") or photo.get("hidden_label")


def conflicting_hashes(run: RunData) -> set[str]:
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for photo in run.photos:
        if _known_label(photo):
            labels_by_hash[photo["sha256"]].add(_known_label(photo))
    return {digest for digest, labels in labels_by_hash.items() if len(labels) > 1}


def repeated_photos(run: RunData) -> set[int]:
    """Indexes of photos whose exact file already appeared earlier in the run."""
    seen: set[str] = set()
    repeats = set()
    for index, photo in enumerate(run.photos):
        if photo["sha256"] in seen:
            repeats.add(index)
        seen.add(photo["sha256"])
    return repeats


def hide_labels(run: RunData, people: Iterable[str]) -> RunData:
    """Hide these people's folder labels from tuning; their photos stay in the collection."""
    hidden = set(people)
    unknown = hidden - {_known_label(photo) for photo in run.photos}
    if unknown:
        raise ValueError(f"No labelled photos for: {', '.join(sorted(unknown))}")
    for photo in run.photos:
        if photo.get("label") in hidden:
            photo["hidden_label"], photo["label"] = photo["label"], None
    return run


def face_labels(run: RunData) -> np.ndarray:
    """Folder label per usable face, or None where the answer key is ambiguous.

    A face is labelled only when it is the single usable face of a labelled
    photo, the same file is not filed under two people, and it is the first
    copy of that file.
    """
    conflicts = conflicting_hashes(run)
    repeats = repeated_photos(run)
    usable_count = Counter(int(index) for index in run.face_photo)
    labels = []
    for face, photo_index in zip(run.faces, run.face_photo):
        photo = run.photos[int(photo_index)]
        clear = (photo.get("label") and usable_count[int(photo_index)] == 1 and photo["sha256"] not in conflicts
                 and int(photo_index) not in repeats)
        labels.append(photo["label"] if clear else None)
    return np.asarray(labels, dtype=object)


def photo_labels(run: RunData) -> dict[int, str]:
    conflicts = conflicting_hashes(run)
    repeats = repeated_photos(run)
    return {index: photo["label"] for index, photo in enumerate(run.photos)
            if photo.get("label") and photo["sha256"] not in conflicts and index not in repeats}


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

REVIEW_BUDGETS = (0, 10, 25, 50, 100, 200, 300, 500)


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
    new_groups = set(result.leftover_groups)
    new_summary = Counter()
    for group, counter in counters.items():
        top = majority[group]
        if top not in focus and not (set(counter) - {top}) & focus:
            continue
        intruders = sum(count for label, count in counter.items() if label != top and (label in focus or top in focus))
        grouped_labelled += sum(count for label, count in counter.items() if label in focus or top in focus)
        mixed_faces += intruders
        if group in new_groups:
            new_summary["mixed_faces"] += intruders
            new_summary["mixed_groups"] += bool(intruders)
            new_summary["led_by_evaluated_people"] += top in focus
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
    if new_groups:
        metrics["leftover_groups"] = {"groups": len(new_groups), "faces": int(np.isin(result.labels, sorted(new_groups)).sum()),
                                      **{name: int(new_summary[name]) for name in ("led_by_evaluated_people", "mixed_groups", "mixed_faces")}}
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
    """Candidate settings. None in cluster or attach values means the step is off.

    ``margin`` and ``ratio`` apply only when attach is on; an empty margin
    tuple keeps the base policy's margin, and None in ratio turns that check off.
    ``cluster_ratio`` is the cluster merge's whole-cluster check (None: off).
    """
    merge: tuple[float, ...]
    attach: tuple[float | None, ...] = (None,)
    eye: tuple[float, ...] = (0.0,)
    cluster: tuple[float | None, ...] = (None,)
    margin: tuple[float, ...] = ()
    ratio: tuple[float | None, ...] = (None,)
    cluster_ratio: tuple[float | None, ...] = ()  # empty: the base policy's cluster_min_ratio

    @property
    def merge_floor(self) -> float:
        return min(self.merge)

    @property
    def cluster_floor(self) -> float | None:
        values = [value for value in self.cluster if value is not None]
        return min(values) if values else None


class _Grid:
    """Caches each clustering step so that every grid setting is cheap to evaluate."""

    def __init__(self, run: RunData, base: GroupingPolicy, grid: SearchGrid, newcomers_per_person: int = 200) -> None:
        self.run, self.base, self.grid = run, base, grid
        self.similarity = fused_similarity(run.embeddings, base)
        self.constraints = same_photo_constraints(run.face_photo)
        self.labels = face_labels(run)
        self.newcomer_faces = newcomer_sample(self.labels, newcomers_per_person)
        self.newcomer_cache: dict[tuple, dict] = {}
        self.trees: dict[float, tuple[np.ndarray, list]] = {}
        self.cluster_trees: dict[tuple[float, float], tuple[np.ndarray, list]] = {}
        self.cuts: dict[tuple[float, float], np.ndarray] = {}
        self.attach_scores: dict[tuple, dict] = {}
        self._block_eye: float | None = None
        self._block: tuple[np.ndarray, np.ndarray] | None = None

    def _rows(self, eye: float) -> tuple[np.ndarray, list]:
        if eye not in self.trees:
            rows = np.flatnonzero(gate_mask(self.run.faces, replace(self.base, min_eye_distance=eye)))
            block = square_block(self.similarity, rows)  # a temporary copy unless every face passes the gate
            merges = agglomerate(block, square_block(self.constraints, rows), self.base.linkage, self.grid.merge_floor,
                                 overwrite=block is not self.similarity)
            del block
            self.trees[eye] = (rows, merges)
        return self.trees[eye]

    def _blocks(self, eye: float) -> tuple[np.ndarray, np.ndarray]:
        if self._block_eye != eye:  # keep one gated copy at a time; the grid loops over eye first
            rows, _ = self._rows(eye)
            self._block = (square_block(self.similarity, rows), square_block(self.constraints, rows))
            self._block_eye = eye
        return self._block

    def local_labels(self, eye: float, merge: float, cluster: float | None,
                     cluster_ratio: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        rows, merges = self._rows(eye)
        if (eye, merge) not in self.cuts:
            self.cuts[(eye, merge)] = cut(merges, len(rows), merge)
        local = self.cuts[(eye, merge)]
        if cluster is None:
            return rows, local
        key = (eye, merge, cluster_ratio)
        if key not in self.cluster_trees:
            similarity, constraints = self._blocks(eye)
            self.cluster_trees[key] = merge_clusters(similarity, constraints, local, self.base.cluster_min_size, self.grid.cluster_floor,
                                                     self.base.cluster_auto_max_size, cluster_ratio)
        cluster_of, cluster_merges = self.cluster_trees[key]
        return rows, apply_cluster_merges(cluster_of, cluster_merges, cluster)

    @staticmethod
    def _structure(policy: GroupingPolicy) -> tuple:
        return (policy.min_eye_distance, policy.merge_threshold, policy.cluster_merge_threshold, policy.cluster_min_ratio)

    def result(self, policy: GroupingPolicy, suggestions: bool = True) -> GroupingResult:
        rows, local = self.local_labels(*self._structure(policy))
        if not suggestions:
            policy = replace(policy, suggest_threshold=None)
        key = self._structure(policy)
        if key not in self.attach_scores:
            self.attach_scores = {key: {}}  # the grid loops over attach last, so one entry is enough
        return finish_grouping(self.run, policy, self.similarity, self.constraints, rows, local, self.attach_scores[key],
                               review_candidates=suggestions)

    def newcomer(self, policy: GroupingPolicy) -> dict:
        """Share of sampled labelled faces that the attach step would put into another person's group
        if their own person had no group yet (a stand-in for people seen for the first time)."""
        key = self._structure(policy)
        if key not in self.newcomer_cache:
            self.newcomer_cache = {key: self._newcomer_scores(*key)}  # the grid loops over attach settings last
        data = self.newcomer_cache[key]
        accepted = np.zeros(len(data["person"]), bool)
        if policy.attach_threshold is not None:
            accepted = data["eligible"] & (data["best"] >= policy.attach_threshold) & (data["lead"] >= policy.attach_margin)
            if policy.attach_min_ratio is not None:
                accepted &= data["ratio"] >= policy.attach_min_ratio
        rates = [float(accepted[data["person"] == person].mean()) for person in np.unique(data["person"])]
        return {"faces": int(len(accepted)), "accepted": int(accepted.sum()), "rate": _ratio(int(accepted.sum()), len(accepted)),
                "mean_person_rate": _mean(rates), "max_person_rate": max(rates) if rates else None}

    def _newcomer_scores(self, eye: float, merge: float, cluster: float | None, cluster_ratio: float | None) -> dict:
        faces = self.newcomer_faces
        rows, local = self.local_labels(eye, merge, cluster, cluster_ratio)
        group_of, members = initial_groups(len(self.run.faces), rows, local, self.base.min_group_size)
        person = self.labels[faces]
        _, led_by = _majorities(group_of, self.labels)
        scores = np.full((len(faces), len(members)), -math.inf)
        take_k = self.base.attach_top_k
        for index, group in enumerate(members):
            block = np.array(self.similarity[np.ix_(faces, group)], np.float64)
            inside = np.flatnonzero(group_of[faces] == index)
            if len(inside):  # a face never counts towards its own score
                block[inside, np.searchsorted(group, faces[inside])] = -math.inf
            take = min(take_k, len(group))
            top = np.partition(block, -take, axis=1)[:, -take:]
            values = top.mean(axis=1)
            if len(inside) and take == len(group):
                values[inside] = np.sort(top[inside], axis=1)[:, 1:].mean(axis=1) if take > 1 else -math.inf
            scores[:, index] = values
            scores[person == led_by.get(index), index] = -math.inf  # the person's own groups do not exist yet
        if len(members):
            order = np.argsort(-scores, axis=1)
            best_group = order[:, 0]
            best = scores[np.arange(len(faces)), best_group]
            second = scores[np.arange(len(faces)), order[:, 1]] if len(members) > 1 else np.full(len(faces), -math.inf)
            ratios = mean_ratios(*group_means(self.similarity, faces, members))[np.arange(len(faces)), best_group]
        else:
            best = second = ratios = np.full(len(faces), -math.inf)
        with np.errstate(invalid="ignore"):
            lead = best - second
        lead[np.isnan(lead)] = -math.inf
        eligible = np.isin(faces, rows) & np.isfinite(best)
        return {"person": person.astype(str), "best": best, "lead": lead, "ratio": ratios, "eligible": eligible}

    def policies(self):
        margins = self.grid.margin or (self.base.attach_margin,)
        cluster_ratios = self.grid.cluster_ratio or (self.base.cluster_min_ratio,)
        for eye in self.grid.eye:
            for merge in self.grid.merge:
                for cluster_ratio in cluster_ratios:
                    for cluster in self.grid.cluster:
                        if cluster is None and cluster_ratio != cluster_ratios[0]:
                            continue  # the ratio only matters when the cluster merge is on
                        common = dict(min_eye_distance=eye, merge_threshold=merge, cluster_merge_threshold=cluster,
                                      cluster_min_ratio=cluster_ratio if cluster is not None else None)
                        for attach in self.grid.attach:
                            if attach is None:
                                yield replace(self.base, **common, attach_threshold=None)
                                continue
                            for margin in margins:
                                for ratio in self.grid.ratio:
                                    yield replace(self.base, **common, attach_threshold=attach, attach_margin=margin,
                                                  attach_min_ratio=ratio)


def newcomer_sample(labels: np.ndarray, per_person: int, seed: int = 0) -> np.ndarray:
    """Up to ``per_person`` answer-key faces of each labelled person, so every person weighs about the same."""
    generator = np.random.default_rng(seed)
    chosen = []
    for person in sorted({label for label in labels if label is not None}):
        rows = np.flatnonzero(labels == person)
        chosen.append(np.sort(generator.choice(rows, size=min(per_person, len(rows)), replace=False)))
    return np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, np.int64)


def _rates(metrics: dict) -> tuple[float, float]:
    mixing, missing = metrics["mixing"]["mixing_rate"], metrics["missing"]["missing_rate"]
    return (0.0 if mixing is None else mixing, 1.0 if missing is None else missing)


def _conservative(policy: GroupingPolicy) -> tuple:
    """Tie-breaker: prefer the stricter setting."""
    return (-policy.merge_threshold, -(policy.cluster_merge_threshold if policy.cluster_merge_threshold is not None else math.inf),
            -(policy.attach_threshold if policy.attach_threshold is not None else math.inf),
            -(policy.attach_min_ratio if policy.attach_min_ratio is not None else -math.inf), -policy.attach_margin,
            -(policy.cluster_min_ratio if policy.cluster_min_ratio is not None else -math.inf),
            -policy.min_eye_distance)


def evaluate_grid(run: RunData, base: GroupingPolicy, grid: SearchGrid, labels: np.ndarray | None = None) -> list[tuple[GroupingPolicy, dict]]:
    labels = face_labels(run) if labels is None else labels
    photos = photo_labels(run)
    cache = _Grid(run, base, grid)
    table = []
    for policy in cache.policies():
        metrics = grouping_metrics(run, cache.result(policy, suggestions=False), labels, photos=photos)
        metrics["newcomer"] = cache.newcomer(policy)
        table.append((policy, metrics))
    return table


def _newcomer_rate(metrics: dict) -> float:
    entry = metrics.get("newcomer")
    return 0.0 if not entry or entry["rate"] is None else entry["rate"]


def pick(table: list[tuple[GroupingPolicy, dict]], max_mixing_rate: float,
         max_newcomer_rate: float | None = None) -> tuple[GroupingPolicy | None, dict | None]:
    """Fewest missing photos among settings within the mixing limit (and the newcomer limit, if given);
    ties go to the stricter setting."""
    allowed = [(policy, metrics) for policy, metrics in table if _rates(metrics)[0] <= max_mixing_rate
               and (max_newcomer_rate is None or _newcomer_rate(metrics) <= max_newcomer_rate)]
    if not allowed:
        return None, None
    return min(allowed, key=lambda item: (_rates(item[1])[1], _rates(item[1])[0], _conservative(item[0])))


def choose_policy(run: RunData, base: GroupingPolicy, grid: SearchGrid, max_mixing_rate: float,
                  labels: np.ndarray | None = None, max_newcomer_rate: float | None = None) -> tuple[GroupingPolicy | None, dict | None, list[dict]]:
    table = evaluate_grid(run, base, grid, labels)
    policy, metrics = pick(table, max_mixing_rate, max_newcomer_rate)
    return policy, metrics, [table_row(policy_, metrics_) for policy_, metrics_ in table]


def table_row(policy: GroupingPolicy, metrics: dict) -> dict:
    mixing, missing = _rates(metrics)
    return {**_setting(policy), "mixing_rate": mixing, "mixed_groups": metrics["mixing"]["mixed_groups"], "missing_rate": missing,
            "newcomer_rate": (metrics.get("newcomer") or {}).get("rate"), "groups": metrics["groups"]}


def _setting(policy: GroupingPolicy) -> dict:
    return {"min_eye_distance": policy.min_eye_distance, "merge_threshold": policy.merge_threshold,
            "cluster_merge_threshold": policy.cluster_merge_threshold, "attach_threshold": policy.attach_threshold,
            "attach_margin": policy.attach_margin, "attach_min_ratio": policy.attach_min_ratio,
            "cluster_min_ratio": policy.cluster_min_ratio}


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


def calibrate_leave_one_person_out(run: RunData, base: GroupingPolicy, grid: SearchGrid, max_mixing_rates: Iterable[float],
                                   max_newcomer_rate: float | None = None) -> dict:
    """Choose settings without each person in turn, then measure errors that involve that person.

    Every mixing limit in ``max_mixing_rates`` is evaluated from the same grid,
    so the trade-off between mixing and missing is visible. The held-out
    person's photos are grouped together with everyone else's, as in real use.
    """
    labels = face_labels(run)
    photos = photo_labels(run)
    people = sorted(set(photos.values()))
    targets = sorted(set(max_mixing_rates))
    # One full-run grid at a time: each holds an N x N matrix (about 2.5 GB at 25,000 faces).
    full: _Grid | None = None
    full_key: tuple | None = None
    folds: dict[float, list[dict]] = {target: [] for target in targets}
    for person in people:
        train_rows = np.flatnonzero(np.asarray([photos.get(int(index)) != person for index in run.face_photo]))
        train = _subset(run, train_rows)
        fold_base = _fold_base(train, base)
        key = (fold_base.offsets, fold_base.scales)
        if key != full_key:
            full = None  # free it before the training grid is built
        table = evaluate_grid(train, fold_base, grid)
        del train
        if full is None:
            full, full_key = _Grid(run, fold_base, grid), key
        for target in targets:
            policy, train_metrics = pick(table, target, max_newcomer_rate)
            if policy is None:
                folds[target].append({"held_out": person, "policy": None, "note": "No setting met the limits on the other people."})
                continue
            test = grouping_metrics(run, full.result(policy), labels, people=[person], review=True)
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


# ---------------------------------------------------------------- open-set check

STRANGER_OUTCOMES = ("not_grouped", "own_group", "other_person_group", "unlabelled_group")


def _quantiles(values: list[float]) -> dict:
    data = np.asarray([value for value in values if value is not None and math.isfinite(value)], np.float64)
    if not data.size:
        return {}
    return {**{f"p{int(q * 100):02d}": round(float(np.quantile(data, q)), 4) for q in (.01, .1, .5, .9, .99)},
            "max": round(float(data.max()), 4), "count": int(data.size)}


def leftover_setting(item) -> tuple[float, int, float | None, bool]:
    """(threshold, minimum size, known-group score limit or None, attach to new groups) from 2 to 4 items."""
    threshold, min_size, cap, attach = (*tuple(item), None, False)[:4]
    if int(min_size) < 2:
        raise ValueError("leftover minimum sizes must be at least 2")
    return float(threshold), int(min_size), None if cap is None else float(cap), bool(attach)


def _policy_setting(policy: GroupingPolicy) -> list[tuple[float, int, float | None, bool]]:
    if policy.leftover_merge_threshold is None:
        return []
    return [leftover_setting((policy.leftover_merge_threshold, policy.leftover_min_size, policy.leftover_max_known_score,
                              policy.leftover_attach))]


def _apply_leftover_settings(result: GroupingResult, similarity: np.ndarray, face_photo: np.ndarray,
                             settings: list[tuple], linkage: str, policy: GroupingPolicy | None = None) -> dict:
    """Group labels (and new groups) for each leftover setting, from one grouping without the leftover step.

    Settings that attach to new groups use the attach rules of ``policy``.
    """
    leftover = np.flatnonzero((result.labels == UNGROUPED) & ~result.gated)
    if not settings:
        return {}
    leftover_similarity = square_block(similarity, leftover)
    merges = leftover_merges(leftover_similarity, same_photo_constraints(np.asarray(face_photo)[leftover]),
                             np.arange(len(leftover)), min(setting[0] for setting in settings), linkage)
    del leftover_similarity
    count = len(result.group_sizes)
    known = existing_group_scores(similarity, result.labels, leftover, count) \
        if any(setting[2] is not None for setting in settings) else None
    face_photo = np.asarray(face_photo)

    def linked(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return face_photo[left][:, None] == face_photo[right][None, :]

    applied = {}
    for setting in settings:
        labels, new_groups = add_leftover_groups(result.labels, leftover, merges, setting[0], setting[1], count, setting[2],
                                                 similarity, known)
        if setting[3]:
            if policy is None:
                raise ValueError("attaching to new groups needs the policy's attach rules")
            labels = attach_to_new_groups(similarity, linked, labels, leftover, new_groups, policy)[0]
        applied[setting] = (labels, new_groups)
    return applied


class _LikelyPerson:
    """Estimated person for unlabelled faces: the labelled person with the highest mean score, by a clear lead.

    It uses the same models as the grouping, so it is a rough estimate, not an answer key.
    """

    def __init__(self, run: RunData, policy: GroupingPolicy, key: np.ndarray, lead: float = 1.0) -> None:
        self.people = sorted({label for label in key if label is not None})
        weights = policy.weights or (1.0,) * len(policy.models)
        offsets = policy.offsets or (0.0,) * len(policy.models)
        scales = policy.scales or (1.0,) * len(policy.models)
        self.parts = [(run.embeddings[name], np.stack([run.embeddings[name][key == person].mean(axis=0) for person in self.people]),
                       offset, weight / scale / sum(weights)) for name, weight, offset, scale in zip(policy.models, weights, offsets, scales)]
        self.lead = lead

    def __call__(self, rows: np.ndarray) -> list[str | None]:
        if not len(rows) or not self.people:
            return [None] * len(rows)
        total = sum(factor * (np.asarray(vectors[rows], np.float64) @ means.T - offset) for vectors, means, offset, factor in self.parts)
        if total.shape[1] == 1:
            return [self.people[0]] * len(rows)
        top = np.sort(total, axis=1)[:, -2:]
        best = np.argmax(total, axis=1)
        return [self.people[int(index)] if gap >= self.lead else None for index, gap in zip(best, top[:, 1] - top[:, 0])]


def _stranger_outcomes(labels: np.ndarray, own: dict[int, str], key: np.ndarray) -> tuple[dict[int, list[int]], dict[int, tuple]]:
    """Each kept face's outcome, the majority label of its group, and its group; also each group's members."""
    members_of: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(labels != UNGROUPED):
        members_of[int(labels[index])].append(int(index))
    outcomes = {}
    for index, person in own.items():
        group = int(labels[index])
        others = [] if group == UNGROUPED else [member for member in members_of[group] if own.get(member) != person]
        if group == UNGROUPED:
            outcome, majority = "not_grouped", None
        elif not others:
            outcome, majority = "own_group", person
        else:
            named = Counter(key[member] for member in others if key[member] is not None)
            majority = named.most_common(1)[0][0] if named else None
            outcome = "other_person_group" if majority is not None else "unlabelled_group"
        outcomes[index] = (outcome, majority, group)
    return members_of, outcomes


def _outcome_rates(counts: Counter) -> dict:
    kept = sum(counts.values())
    return {"kept_faces": kept, **{name: counts[name] for name in STRANGER_OUTCOMES},
            "own_group_rate": _ratio(counts["own_group"], kept),
            "wrong_rate": _ratio(counts["other_person_group"] + counts["unlabelled_group"], kept),
            "other_person_rate": _ratio(counts["other_person_group"], kept)}


def stranger_test(run: RunData, policy: GroupingPolicy, people: Iterable[str], keep_sizes: Iterable[int] = (1, 3),
                  repeats: int = 3, batch: int = 5, seed: int = 0,
                  leftover_grid: Iterable[tuple] = (), likely_lead: float = 1.0) -> dict:
    """How often a person with only a few photos is put into a group of other people.

    Real collections keep adding people who appear in only one or a few
    photos. For each repeat and keep size, the named people are split into
    batches. In a batch, each person keeps ``keep`` random answer-key photos,
    and all their other photos (and every copy of those files) are removed;
    the rest of the collection is unchanged. A kept face is placed correctly
    when it is left out of the groups (review) or its group holds only that
    person's kept faces. Any other placement counts as wrong. This is
    conservative: a group of unlabelled faces can be the same person seen in
    another person's group photo.

    ``leftover_grid`` lists (threshold, minimum size[, known-group score
    limit]) settings for new groups from leftover faces. Each is scored on the
    same batches, by outcome counts only, after the policy's other steps.

    Each wrong placement lists who else is in the group. Unlabelled faces get
    an estimated person (``_LikelyPerson``, lead ``likely_lead``), which shows
    whether a group of unlabelled faces is probably the newcomer seen in other
    photos. That estimate uses the same models, so it is not an answer key.
    """
    focus = sorted(set(people))
    grid = [leftover_setting(item) for item in leftover_grid]
    unknown = set(focus) - {_known_label(photo) for photo in run.photos}
    if unknown:
        raise ValueError(f"No labelled photos for: {', '.join(sorted(unknown))}")
    conflicts, repeated = conflicting_hashes(run), repeated_photos(run)
    usable = Counter(int(index) for index in run.face_photo)
    face_hash = np.asarray([run.photos[int(index)]["sha256"] for index in run.face_photo], dtype=object)
    # Answer-key label per face. Labels hidden from tuning stay hidden: those
    # faces count as unlabelled, so a placement with them is still wrong, but
    # whose faces they are is never read.
    key = np.full(len(run.faces), None, dtype=object)
    candidates: dict[str, list[int]] = defaultdict(list)
    hashes: dict[str, set[str]] = defaultdict(set)
    for photo in run.photos:
        if _known_label(photo) in focus:
            hashes[_known_label(photo)].add(photo["sha256"])
    for row, index in enumerate(run.face_photo):
        photo = run.photos[int(index)]
        if usable[int(index)] == 1 and photo["sha256"] not in conflicts and int(index) not in repeated and _known_label(photo):
            key[row] = photo.get("label")
            if _known_label(photo) in focus:
                candidates[_known_label(photo)].append(row)
    full = fused_similarity(run.embeddings, policy)
    generator = np.random.default_rng(seed)
    sizes = sorted(set(int(size) for size in keep_sizes))
    if not sizes or sizes[0] < 1 or batch < 1 or repeats < 1:
        raise ValueError("keep sizes, batch and repeats must be at least 1")
    own_setting = _policy_setting(policy)
    base_policy = replace(policy, leftover_merge_threshold=None) if grid else policy
    likely = _LikelyPerson(run, policy, key, likely_lead)
    report = {"method": stranger_test.__doc__.split("\n\n")[0].strip(), "people": focus, "repeats": repeats, "batch": batch,
              "seed": seed, "attach_threshold": policy.attach_threshold, "leftover_merge_threshold": policy.leftover_merge_threshold,
              "leftover_min_size": policy.leftover_min_size, "leftover_max_known_score": policy.leftover_max_known_score,
              "keep": {}}
    for size in sizes:
        counts = Counter()
        grid_counts = {setting: Counter() for setting in grid}
        details, best_scores, leads, ratios = [], [], [], []
        for _ in range(repeats):
            order = [focus[index] for index in generator.permutation(len(focus))]
            for start in range(0, len(order), batch):
                chunk = order[start:start + batch]
                kept: dict[int, str] = {}
                for person in chunk:
                    pool = candidates.get(person, [])
                    for row in generator.choice(pool, size=min(size, len(pool)), replace=False) if pool else []:
                        kept[int(row)] = person
                removed = set().union(*(hashes[person] for person in chunk))
                rows = np.asarray(sorted(set(np.flatnonzero([digest not in removed for digest in face_hash]).tolist()) | set(kept)))
                position = {int(row): index for index, row in enumerate(rows)}
                sub = _subset(run, rows)
                similarity = square_block(full, rows)
                result = group_faces(sub, base_policy, similarity=similarity)
                labels = result.labels
                sub_key = key[rows]
                own = {position[row]: person for row, person in kept.items()}
                base_groups = len(result.group_sizes) - len(result.leftover_groups)
                if grid:
                    for setting, (new_labels, _) in _apply_leftover_settings(result, similarity, sub.face_photo, grid + own_setting,
                                                                             policy.linkage, policy).items():
                        if setting in grid_counts:
                            grid_counts[setting].update(outcome for outcome, _, _ in _stranger_outcomes(new_labels, own, sub_key)[1].values())
                        if setting in own_setting:
                            labels = new_labels
                members_of, outcomes = _stranger_outcomes(labels, own, sub_key)
                group_ids = sorted(members_of)
                kept_positions = np.asarray(sorted(own), np.int64)
                ratio_of = mean_ratios(*group_means(similarity, kept_positions, [np.asarray(members_of[group]) for group in group_ids])) \
                    if group_ids and len(kept_positions) else None
                for index, person in own.items():
                    outcome, majority, group = outcomes[index]
                    counts[outcome] += 1
                    row = similarity[[index]]
                    scores = sorted(((float(top_k_means(row, np.asarray([member for member in members if member != index]),
                                                        policy.attach_top_k)[0]), group_index)
                                     for group_index, members in members_of.items()
                                     if any(own.get(member) != person for member in members if member != index)), reverse=True)
                    best = scores[0] if scores else (None, None)
                    lead = ratio = None
                    if best[0] is not None:
                        best_scores.append(best[0])
                        lead = best[0] - scores[1][0] if len(scores) > 1 else math.inf
                        ratio = float(ratio_of[int(np.searchsorted(kept_positions, index)), group_ids.index(best[1])])
                        leads.append(lead)
                        ratios.append(ratio)
                    if outcome in ("other_person_group", "unlabelled_group"):
                        others = [member for member in members_of[group] if own.get(member) != person]
                        unlabelled = [member for member in others if sub_key[member] is None]
                        guesses = Counter(likely(rows[np.asarray(unlabelled, np.int64)]))
                        entry = {"person": person, "outcome": outcome, "group_majority": majority, "group_size": len(members_of[group]),
                                 "attached": bool(result.attached[index]), "from_leftovers": group >= base_groups,
                                 "same_person_faces": len(members_of[group]) - len(others),
                                 "labelled_others": dict(Counter(sub_key[member] for member in others
                                                                 if sub_key[member] is not None).most_common(3)),
                                 "unlabelled_others": len(unlabelled), "unlabelled_likely_same_person": guesses.get(person, 0),
                                 "unlabelled_likely_people": {str(name): count for name, count in guesses.most_common(3)},
                                 "best_other_group_score": None if best[0] is None else round(best[0], 4),
                                 "lead": None if lead is None else round(lead, 4),
                                 "mean_ratio": None if ratio is None else round(ratio, 4)}
                        if group >= base_groups and base_groups:
                            existing = np.where(labels < base_groups, labels, UNGROUPED)
                            group_rows = np.asarray(members_of[group], np.int64)
                            scores = known_cluster_scores(similarity, group_rows,
                                                          existing_group_scores(similarity, existing, group_rows, base_groups))
                            nearest = int(np.argmax(scores))
                            entry["known_score"] = round(float(scores[nearest]), 4)
                            entry["nearest_group_majority"] = Counter(sub_key[member] for member in np.flatnonzero(existing == nearest)
                                                                      if sub_key[member] is not None).most_common(1)[0][0] \
                                if any(sub_key[member] is not None for member in np.flatnonzero(existing == nearest)) else None
                        details.append(entry)
                del similarity, result, sub, labels
        values = np.asarray(best_scores)
        report["keep"][str(size)] = {**_outcome_rates(counts),
                                     "wrong_by_attach": sum(1 for item in details if item["attached"]),
                                     "wrong_in_new_groups": sum(1 for item in details if item["from_leftovers"]),
                                     "unlabelled_group_likely_same_person": sum(
                                         1 for item in details if item["outcome"] == "unlabelled_group"
                                         and item["unlabelled_likely_same_person"] * 2 > item["unlabelled_others"]),
                                     "best_score_in_another_group": _quantiles(best_scores),
                                     "lead_over_second_group": _quantiles(leads),
                                     "mean_ratio_to_best_group": _quantiles(ratios),
                                     "above_attach_threshold": None if policy.attach_threshold is None or not values.size
                                     else _ratio(int(np.sum(values >= policy.attach_threshold)), int(values.size)),
                                     "wrong": details}
        if grid:
            report["keep"][str(size)]["leftover_grid"] = [{"leftover_merge_threshold": setting[0], "leftover_min_size": setting[1],
                                                           "leftover_max_known_score": setting[2], "leftover_attach": setting[3],
                                                           **_outcome_rates(grid_counts[setting])}
                                                          for setting in grid]
    return report


def leftover_sweep(run: RunData, policy: GroupingPolicy, settings: Iterable[tuple[float, int]],
                   people: Iterable[str] | None = None) -> list[dict]:
    """Mixing and missing for several new-group-from-leftover settings on one grouping.

    The policy's other steps run once without merge suggestions. The first row
    has no new groups; each further row adds the new groups of one
    (threshold, minimum size) setting. Labels hidden with ``hide_labels`` stay
    hidden.
    """
    settings = [leftover_setting(item) for item in settings]
    base_policy = replace(policy, leftover_merge_threshold=None, suggest_threshold=None)
    similarity = fused_similarity(run.embeddings, policy)
    base = group_faces(run, base_policy, similarity=similarity)
    labels, photos = face_labels(run), photo_labels(run)
    leftover = np.flatnonzero((base.labels == UNGROUPED) & ~base.gated)
    applied = _apply_leftover_settings(base, similarity, run.face_photo, settings, policy.linkage, policy)
    del similarity
    rows = []
    for setting in [None] + settings:
        result = base
        if setting is not None:
            new_labels, new_groups = applied[setting]
            grouped = {run.faces[face]["id"] for face in leftover[new_labels[leftover] != UNGROUPED]}
            result = replace(base, policy=replace(base_policy, leftover_merge_threshold=setting[0], leftover_min_size=setting[1],
                                                  leftover_max_known_score=setting[2], leftover_attach=setting[3]),
                             labels=new_labels, leftover_groups=new_groups,
                             group_sizes=np.bincount(new_labels[new_labels != UNGROUPED], minlength=len(base.group_sizes) + len(new_groups))
                             .astype(int).tolist(),
                             review=[item for item in base.review if item["face"] not in grouped])
        metrics = grouping_metrics(run, result, labels, people, photos=photos)
        rows.append({"leftover_merge_threshold": None if setting is None else setting[0],
                     "leftover_min_size": None if setting is None else setting[1],
                     "leftover_max_known_score": None if setting is None else setting[2],
                     "leftover_attach": None if setting is None else setting[3],
                     "groups": metrics["groups"], "review_faces": metrics["review_faces"],
                     "mixed_faces": metrics["mixing"]["mixed_faces"], "mixing_rate": metrics["mixing"]["mixing_rate"],
                     "missing_rate": metrics["missing"]["missing_rate"],
                     "in_other_group_of_person": metrics["missing"]["in_other_group_of_person"],
                     "left_for_review": metrics["missing"]["left_for_review"],
                     "leftover_groups": metrics.get("leftover_groups", {"groups": 0, "faces": 0}),
                     "mixed_groups": [{**{key: value for key, value in item.items() if key != "intruder_photos"},
                                       "from_leftovers": item["group"] in set(result.leftover_groups)}
                                      for item in metrics["mixing"]["details"]]})
    return rows


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
