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
from dataclasses import replace
import math
from typing import Iterable

import numpy as np

from .photo_grouping import UNGROUPED, GroupingPolicy, GroupingResult, agglomerate, cut, finish_grouping, \
    fused_similarity, gate_mask, same_photo_constraints
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

def grouping_metrics(run: RunData, result: GroupingResult, labels: np.ndarray | None = None,
                     people: Iterable[str] | None = None) -> dict:
    """Mixing and missing for a grouping result. ``people`` limits the people evaluated."""
    labels = face_labels(run) if labels is None else labels
    photos = photo_labels(run)
    focus = set(people) if people is not None else {label for label in photos.values()}
    group_labels: dict[int, Counter] = defaultdict(Counter)
    for row, group in enumerate(result.labels):
        if group != UNGROUPED and labels[row] is not None:
            group_labels[int(group)][labels[row]] += 1
    majority = {group: counter.most_common(1)[0][0] for group, counter in group_labels.items()}
    mixed_faces = grouped_labelled = 0
    mixed_groups = 0
    mixed_detail = []
    for group, counter in group_labels.items():
        top = majority[group]
        involved = {label for label in counter if label != top}
        relevant = (top in focus) or bool(involved & focus)
        if not relevant:
            continue
        intruders = sum(count for label, count in counter.items() if label != top and (label in focus or top in focus))
        grouped_labelled += sum(count for label, count in counter.items() if label in focus or top in focus)
        mixed_faces += intruders
        if intruders:
            mixed_groups += 1
            mixed_detail.append({"group": group, "size": result.group_sizes[group], "majority": top,
                                 "intruders": {label: count for label, count in counter.items() if label != top}})
    faces_by_photo: dict[int, list[int]] = defaultdict(list)
    for row, photo in enumerate(run.face_photo):
        faces_by_photo[int(photo)].append(row)
    per_person = {}
    for person in sorted(focus):
        own_groups = Counter(int(group) for row, group in enumerate(result.labels) if group != UNGROUPED and labels[row] == person)
        main = own_groups.most_common(1)[0][0] if own_groups else None
        person_groups = {group for group in own_groups if majority.get(group) == person}
        counts = Counter({key: 0 for key in ("photos", "no_usable_face", "found_in_main_group", "in_other_group_of_person",
                                             "only_in_other_people_groups", "left_for_review")})
        for photo_index, label in photos.items():
            if label != person:
                continue
            counts["photos"] += 1
            rows = faces_by_photo.get(photo_index, [])
            groups = {int(result.labels[row]) for row in rows if result.labels[row] != UNGROUPED}
            review = any(result.labels[row] == UNGROUPED for row in rows)
            if not rows:
                counts["no_usable_face"] += 1
            elif main is not None and main in groups:
                counts["found_in_main_group"] += 1
            elif groups & person_groups:
                counts["in_other_group_of_person"] += 1
            elif groups:
                counts["only_in_other_people_groups"] += 1
            elif review:
                counts["left_for_review"] += 1
        total = counts["photos"]
        per_person[person] = {**counts, "groups_led": len(person_groups),
                              "main_group_size": result.group_sizes[main] if main is not None else 0,
                              "missing_rate": _ratio(total - counts["found_in_main_group"], total),
                              "missing_rate_excluding_no_face": _ratio(total - counts["no_usable_face"] - counts["found_in_main_group"],
                                                                        total - counts["no_usable_face"])}
    total_photos = sum(entry["photos"] for entry in per_person.values())
    found = sum(entry["found_in_main_group"] for entry in per_person.values())
    no_face = sum(entry["no_usable_face"] for entry in per_person.values())
    return {"groups": len(result.group_sizes), "grouped_faces": int(np.sum(result.labels != UNGROUPED)),
            "attached_faces": int(result.attached.sum()), "review_faces": len(result.review),
            "mixing": {"mixed_faces": mixed_faces, "grouped_labelled_faces": grouped_labelled,
                       "mixing_rate": _ratio(mixed_faces, grouped_labelled), "mixed_groups": mixed_groups,
                       "details": sorted(mixed_detail, key=lambda item: -sum(item["intruders"].values()))[:20]},
            "missing": {"photos": total_photos, "found_in_main_group": found, "missing_rate": _ratio(total_photos - found, total_photos),
                        "no_usable_face": no_face,
                        "missing_rate_excluding_no_face": _ratio(total_photos - no_face - found, total_photos - no_face),
                        "mean_person_missing_rate": _mean([entry["missing_rate"] for entry in per_person.values()])},
            "people": per_person}


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


class _Grid:
    """Clusters every gate setting once, then cuts the merge tree at each threshold."""

    def __init__(self, run: RunData, base: GroupingPolicy) -> None:
        self.run, self.base = run, base
        self.similarity = fused_similarity(run.embeddings, base)
        self.constraints = same_photo_constraints(run.face_photo)
        self.trees: dict[float, tuple[np.ndarray, list]] = {}

    def results(self, min_eye: float, merge_values: Iterable[float], attach_values: Iterable[float | None]):
        policy = replace(self.base, min_eye_distance=min_eye)
        if min_eye not in self.trees:
            rows = np.flatnonzero(gate_mask(self.run.faces, policy))
            floor = min(merge_values)
            merges = agglomerate(self.similarity[np.ix_(rows, rows)], self.constraints[np.ix_(rows, rows)], policy.linkage, floor)
            self.trees[min_eye] = (rows, merges)
        rows, merges = self.trees[min_eye]
        for merge in merge_values:
            local = cut(merges, len(rows), merge)
            for attach in attach_values:
                candidate = replace(policy, merge_threshold=merge, attach_threshold=attach)
                yield candidate, finish_grouping(self.run, candidate, self.similarity, self.constraints, rows, local)


def _score(metrics: dict) -> tuple[float, float]:
    mixing, missing = metrics["mixing"]["mixing_rate"], metrics["missing"]["missing_rate"]
    return (0.0 if mixing is None else mixing, 1.0 if missing is None else missing)


def choose_policy(run: RunData, base: GroupingPolicy, merge_values, attach_values, eye_values,
                  max_mixing_rate: float, labels: np.ndarray | None = None) -> tuple[GroupingPolicy | None, dict | None, list[dict]]:
    """The policy with the fewest missing photos among those within the mixing limit."""
    labels = face_labels(run) if labels is None else labels
    grid = _Grid(run, base)
    best, best_metrics, best_rank, table = None, None, None, []
    for eye in eye_values:
        for policy, result in grid.results(eye, merge_values, attach_values):
            metrics = grouping_metrics(run, result, labels)
            mixing, missing = _score(metrics)
            table.append({"min_eye_distance": eye, "merge_threshold": policy.merge_threshold, "attach_threshold": policy.attach_threshold,
                          "mixing_rate": mixing, "mixed_groups": metrics["mixing"]["mixed_groups"], "missing_rate": missing,
                          "groups": metrics["groups"]})
            if mixing <= max_mixing_rate:
                rank = (missing, mixing, -policy.merge_threshold, -(policy.attach_threshold or math.inf))
                if best_metrics is None or rank < best_rank:
                    best, best_metrics, best_rank = policy, metrics, rank
    return best, best_metrics, table


def calibrate_leave_one_person_out(run: RunData, base: GroupingPolicy, merge_values, attach_values, eye_values,
                                   max_mixing_rate: float) -> dict:
    """Choose thresholds without each person in turn, then measure errors that involve that person."""
    labels = face_labels(run)
    photos = photo_labels(run)
    people = sorted(set(photos.values()))
    folds = []
    totals = Counter()
    for person in people:
        train_rows = np.flatnonzero(np.asarray([photos.get(int(index)) != person for index in run.face_photo]))
        train = _subset(run, train_rows)
        policy, train_metrics, _ = choose_policy(train, base, merge_values, attach_values, eye_values, max_mixing_rate)
        if policy is None:
            folds.append({"held_out": person, "policy": None, "note": "No setting met the mixing limit on the other people."})
            continue
        test = grouping_metrics(run, _group(run, policy), labels, people=[person])
        entry = test["people"][person]
        folds.append({"held_out": person, "policy": {"min_eye_distance": policy.min_eye_distance, "merge_threshold": policy.merge_threshold,
                                                      "attach_threshold": policy.attach_threshold},
                      "train_mixing_rate": train_metrics["mixing"]["mixing_rate"], "train_missing_rate": train_metrics["missing"]["missing_rate"],
                      "held_out_mixed_faces": test["mixing"]["mixed_faces"], "held_out_grouped_labelled_faces": test["mixing"]["grouped_labelled_faces"],
                      "held_out_mixed_groups": test["mixing"]["mixed_groups"], "held_out_mixing_details": test["mixing"]["details"],
                      "held_out_person": entry})
        totals["mixed_faces"] += test["mixing"]["mixed_faces"]
        totals["grouped_labelled_faces"] += test["mixing"]["grouped_labelled_faces"]
        totals["photos"] += entry["photos"]
        totals["found"] += entry["found_in_main_group"]
        totals["no_face"] += entry["no_usable_face"]
        totals["review"] += entry.get("left_for_review", 0)
        totals["other_group_of_person"] += entry.get("in_other_group_of_person", 0)
        totals["only_in_other_people_groups"] += entry.get("only_in_other_people_groups", 0)
    return {"method": "leave-one-person-out: thresholds chosen on the other people; errors counted only where the held-out person is involved",
            "max_mixing_rate": max_mixing_rate, "folds": folds,
            "held_out": {"mixed_faces": totals["mixed_faces"], "grouped_labelled_faces": totals["grouped_labelled_faces"],
                         "mixing_rate": _ratio(totals["mixed_faces"], totals["grouped_labelled_faces"]),
                         "photos": totals["photos"], "found_in_main_group": totals["found"],
                         "missing_rate": _ratio(totals["photos"] - totals["found"], totals["photos"]),
                         "no_usable_face": totals["no_face"], "left_for_review": totals["review"],
                         "in_other_group_of_person": totals["other_group_of_person"],
                         "only_in_other_people_groups": totals["only_in_other_people_groups"]}}


def _group(run: RunData, policy: GroupingPolicy) -> GroupingResult:
    from .photo_grouping import group_faces
    return group_faces(run, policy)


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
