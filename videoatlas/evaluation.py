"""Track-pair evaluation and candidate tuning, never automatic settings changes."""
import math
from typing import Iterable
from .grouping import GroupingOptions, compare_tracks
from .storage import RelationRecord, TrackRecord


def _model_family(key: str) -> str:
    if "face01" in key.lower() or "japanese" in key.lower():
        return "face01"
    return key.split(":", 1)[0].lower()


def labelled_examples(tracks: list[TrackRecord], relations: list[RelationRecord], options: GroupingOptions | None = None) -> list[dict]:
    by_id = {track.id: track for track in tracks}
    examples = []
    for relation in relations:
        if relation.relation not in {"confirmed_same", "confirmed_different"} or relation.track_a not in by_id or relation.track_b not in by_id:
            continue
        comparison = compare_tracks(by_id[relation.track_a], by_id[relation.track_b], options)
        scores = {key: statistics["median"] for key, statistics in comparison["models"].items()}
        if scores:
            examples.append({"track_a": relation.track_a, "track_b": relation.track_b, "same": relation.relation == "confirmed_same", "scores": scores})
    return examples


def _score(example: dict, weights: dict[str, float] | None = None) -> float | None:
    if "scores" not in example:
        score = example.get("score")
        return float(score) if score is not None and math.isfinite(float(score)) else None
    scores = example["scores"]
    active = weights or {key: 1.0 for key in scores}
    # Candidate ensembles use the same complete evidence cohort; missing models
    # are excluded rather than silently renormalizing an incomplete ensemble.
    if any(key not in scores for key, weight in active.items() if weight > 0):
        return None
    total = sum(weight for weight in active.values() if weight > 0)
    if total <= 0 or any(not math.isfinite(scores[key]) for key, weight in active.items() if weight > 0):
        return None
    return sum(scores[key] * weight for key, weight in active.items() if weight > 0) / total


def evaluate_pairs(examples: Iterable[dict], threshold: float, weights: dict[str, float] | None = None) -> dict:
    examples = list(examples)
    values = [(bool(example["same"]), score) for example in examples if (score := _score(example, weights)) is not None]
    tp = sum(same and score >= threshold for same, score in values)
    fp = sum(not same and score >= threshold for same, score in values)
    tn = sum(not same and score < threshold for same, score in values)
    fn = sum(same and score < threshold for same, score in values)
    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None
    positives = [score for same, score in values if same]
    negatives = [score for same, score in values if not same]
    auc = sum(1 if positive > negative else .5 if positive == negative else 0 for positive in positives for negative in negatives) / (len(positives) * len(negatives)) if positives and negatives else None
    roc = []
    # Infinity is represented by None so saved reports remain strict JSON.
    for cutoff in [math.inf] + sorted({score for _, score in values}, reverse=True):
        roc.append({"threshold": None if math.isinf(cutoff) else cutoff, "false_positive_rate": ratio(sum(score >= cutoff for score in negatives), len(negatives)), "true_positive_rate": ratio(sum(score >= cutoff for score in positives), len(positives))})
    return {"unit": "human-labelled track pairs", "threshold": threshold, "weights": weights or {}, "sample_count": len(values), "excluded_missing_evidence": len(examples) - len(values), "positive_count": tp + fn, "negative_count": fp + tn, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "false_positive_rate": ratio(fp, fp + tn), "false_discovery_fraction": ratio(fp, tp + fp), "false_positive_rate_denominator": fp + tn, "false_discovery_denominator": tp + fp, "roc": roc, "auc": auc}


def optimize_thresholds(examples: list[dict], thresholds: list[float] | None = None, weight_candidates: list[dict[str, float]] | None = None) -> dict:
    """Return training-cohort candidate reports, not validated deployment advice.

    Lexicographic objective minimizes FP, then FN. A held-out evaluation is needed
    before adopting settings. No numeric accuracy target is inferred.
    """
    keys = sorted({key for example in examples for key in example.get("scores", {})})
    if weight_candidates is None:
        weight_candidates = [{key: 1.0} for key in keys]
        if len(keys) == 2 and _model_family(keys[0]) != _model_family(keys[1]):
            weight_candidates += [{keys[0]: weight, keys[1]: 1 - weight} for weight in (.25, .5, .75)]
        if not keys:
            weight_candidates = [{}]
    for weights in weight_candidates:
        families = [_model_family(key) for key, weight in weights.items() if weight > 0]
        if len(set(families)) != len(families):
            raise ValueError("Evaluate model versions separately; do not ensemble two versions of one model")
    if thresholds is None:
        thresholds = [value / 100 for value in range(50, 100, 5)] + [1.000001]
    candidates = [evaluate_pairs(examples, threshold, weights) for weights in weight_candidates for threshold in thresholds]
    for candidate in candidates:
        candidate["abstain_only"] = candidate["threshold"] > 1.0
        candidate["deployable_threshold"] = 0 <= candidate["threshold"] <= 1.0
    supported = [candidate for candidate in candidates if candidate["positive_count"] and candidate["negative_count"]]
    # Exclusion counts remain visible, and smaller cohorts cannot appear better
    # merely by dropping difficult examples from the comparison.
    if supported:
        largest = max(candidate["sample_count"] for candidate in supported)
        comparable = [candidate for candidate in supported if candidate["sample_count"] == largest]
        best = min(comparable, key=lambda candidate: (candidate["fp"], candidate["fn"], -candidate["threshold"]))
    else:
        best = None
    return {"candidate_only": True, "evaluation_split": "training (human-reviewed pairs; selection bias possible)", "requires_held_out_validation": True, "objective": "minimize false-positive count, then false-negative count on the largest complete-evidence cohort", "input_pair_count": len(examples), "candidates": candidates, "best_candidate": best}
