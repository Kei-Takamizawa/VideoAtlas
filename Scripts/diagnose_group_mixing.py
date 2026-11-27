"""Show how faces of other people reached one person's group, step by step.

    python Scripts/diagnose_group_mixing.py RUN --policy POLICY --person NAME [--output FILE]

The grouping is repeated with the policy. For the largest group led by NAME,
the script lists every cluster merge that built it (sizes, answer-key people on
each side, the fused score, the whole-cluster ratio, and the same scores for
each model alone), and the faces that joined it in the attach step. It prints
names and numbers only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoatlas.photo_evaluation import face_labels  # noqa: E402
from videoatlas.photo_grouping import (UNGROUPED, GroupingPolicy, agglomerate, cut, finish_grouping, fused_similarity,  # noqa: E402
                                       gate_mask, merge_clusters, same_photo_constraints, square_block)
from videoatlas.photos import load_run  # noqa: E402


def _people(labels: np.ndarray, items: np.ndarray) -> dict:
    return dict(Counter(label for label in labels[items] if label is not None).most_common())


def _pair_scores(vectors: np.ndarray, offset: float, scale: float, left: np.ndarray, right: np.ndarray) -> dict:
    """Normalized cluster score and whole-cluster ratio for one model."""
    def mean_within(items: np.ndarray) -> float:
        block = (vectors[items] @ vectors[items].T - offset) / scale
        return float((block.sum() - np.trace(block)) / (len(items) * (len(items) - 1)))

    cross = float(((vectors[left] @ vectors[right].T - offset) / scale).mean())
    within_left, within_right = mean_within(left), mean_within(right)
    larger = within_left if len(left) >= len(right) else within_right
    score = cross / np.sqrt(within_left * within_right) if within_left > 0 and within_right > 0 else None
    return {"cross_mean": round(cross, 4), "score": None if score is None else round(float(score), 4),
            "ratio_to_larger": round(cross / larger, 4) if larger > 0 else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--person", required=True)
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    parser.add_argument("--table", action="store_true", help="print one line per merge instead of JSON")
    args = parser.parse_args()
    policy = GroupingPolicy.from_json(json.loads(args.policy.read_text(encoding="utf-8")))
    run = load_run(args.run, policy.models)
    labels = face_labels(run)
    similarity = fused_similarity(run.embeddings, policy)
    constraints = same_photo_constraints(run.face_photo)
    rows = np.flatnonzero(gate_mask(run.faces, policy))
    block = square_block(similarity, rows)
    merges = agglomerate(block, square_block(constraints, rows), policy.linkage, policy.merge_threshold,
                         overwrite=block is not similarity)
    del block
    local = cut(merges, len(rows), policy.merge_threshold)
    cluster_of, cluster_merges = merge_clusters(square_block(similarity, rows), square_block(constraints, rows), local,
                                                policy.cluster_min_size, policy.cluster_merge_threshold,
                                                policy.cluster_auto_max_size, policy.cluster_min_ratio)
    applied = [merge for merge in cluster_merges
               if policy.cluster_merge_threshold is not None and merge[2] >= policy.cluster_merge_threshold]
    parent = np.arange(int(cluster_of.max()) + 1 if len(cluster_of) else 0)

    def find(cluster: int) -> int:
        while parent[cluster] != cluster:
            parent[cluster] = parent[parent[cluster]]
            cluster = parent[cluster]
        return int(cluster)

    for keep, drop, _ in applied:
        parent[find(drop)] = find(keep)
    final_local = np.asarray([find(int(cluster)) for cluster in cluster_of])
    result = finish_grouping(run, policy, similarity, constraints, rows, final_local)
    led = Counter(int(group) for row, group in enumerate(result.labels) if group != UNGROUPED and labels[row] == args.person)
    if not led:
        print(json.dumps({"person": args.person, "note": "no group holds this person's answer-key faces"}))
        return 0
    target = led.most_common(1)[0][0]
    target_rows = np.flatnonzero(result.labels == target)
    position = np.full(len(run.faces), -1, np.int64)
    position[rows] = np.arange(len(rows))
    target_roots = {int(final_local[position[row]]) for row in target_rows if position[row] >= 0 and not result.attached[row]}
    members = {int(cluster): rows[np.flatnonzero(cluster_of == cluster)] for cluster in np.unique(cluster_of)}
    offsets = policy.offsets or (0.0,) * len(policy.models)
    scales = policy.scales or (1.0,) * len(policy.models)
    events = []
    for order, (keep, drop, score) in enumerate(applied):
        left, right = members[keep], members.pop(drop)
        members[keep] = np.concatenate([left, right])
        if find(keep) not in target_roots:
            continue
        cross = float(similarity[np.ix_(left, right)].mean())
        within = []
        for items in (left, right):
            block = np.asarray(similarity[np.ix_(items, items)], np.float64)
            within.append(float((block.sum() - np.trace(block)) / (len(items) * (len(items) - 1))))
        larger = within[0] if len(left) >= len(right) else within[1]
        events.append({"order": order, "score": round(score, 4), "sizes": [int(len(left)), int(len(right))],
                       "people": [_people(labels, left), _people(labels, right)],
                       "fused_ratio_to_larger": round(cross / larger, 4) if larger > 0 else None,
                       "models": {name: _pair_scores(run.embeddings[name], offset, scale, left, right)
                                  for name, offset, scale in zip(policy.models, offsets, scales)}})
    attached = target_rows[result.attached[target_rows]]
    report = {"person": args.person, "group_size": int(len(target_rows)), "people_in_group": _people(labels, target_rows),
              "first_step_clusters": len(target_roots) and int(sum(1 for cluster in np.unique(cluster_of) if find(int(cluster)) in target_roots)),
              "cluster_merges_into_group": events,
              "attached_faces": {"count": int(len(attached)), "people": _people(labels, attached)},
              "policy": {key: value for key, value in policy.to_json().items() if key not in ("offsets", "scales", "calibration")}}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if not args.table:
        print(text)
        return 0
    print(f"{args.person}: group of {report['group_size']} faces, answer-key people {report['people_in_group']}, "
          f"{report['first_step_clusters']} first-step clusters, attached {report['attached_faces']}")
    print("order | sizes | people (kept side / joining side) | fused score | fused ratio | " + " | ".join(
        f"{name} score/ratio" for name in policy.models))
    for event in events:
        print(f"{event['order']} | {event['sizes'][0]}+{event['sizes'][1]} | {event['people'][0]} / {event['people'][1]} | "
              f"{event['score']} | {event['fused_ratio_to_larger']} | " + " | ".join(
                  f"{values['score']}/{values['ratio_to_larger']}" for values in event["models"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
