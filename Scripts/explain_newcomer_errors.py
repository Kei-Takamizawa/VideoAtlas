"""Explain where newcomers' faces went in the newcomer test, and what the receiving groups hold.

    python Scripts/explain_newcomer_errors.py --strangers FILE [--keep 30]
    python Scripts/explain_newcomer_errors.py --run RUN --policy POLICY --groups GROUPS --people Juun,Yuha
    python Scripts/explain_newcomer_errors.py --run RUN --policy POLICY --remove-people X,Y --people Juun,Yuha

1. ``--strangers``: from a newcomer test report (``photos strangers`` output,
   or the sweep script's output), the wrong placements in another person's
   group, by newcomer and receiving group.
2. ``--groups``: for each named person's main group in a groups file, the
   labelled people in it and the estimated person of its unlabelled faces.
3. ``--remove-people``: the grouping is repeated without every photo of those
   people (all copies of their files), except ``--keep`` random answer-key
   photos of each (``--keep-photos``), as in the newcomer test. For each named person's main
   group it shows the same composition, split by how faces joined (clustering
   or the attach step). For each removed person it shows where their kept
   faces and their estimated faces in other folders' group photos went.

The estimated person is the labelled person with the highest mean score by a
clear lead (``LikelyPerson``). It uses the same models as the grouping, so it
is a rough estimate, not an answer key. The script prints names and numbers
only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoatlas.photo_evaluation import LikelyPerson, _known_label, _subset, face_labels, led_by_newcomer  # noqa: E402
from videoatlas.photo_grouping import UNGROUPED, GroupingPolicy, group_faces  # noqa: E402
from videoatlas.photos import load_run  # noqa: E402


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _median(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return round(float(statistics.median(values)), 4) if values else None


def explain_strangers(path: Path, keep: list[str] | None) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    report = report.get("newcomers", report)
    for size, entry in report["keep"].items():
        if keep and size not in keep:
            continue
        rows = defaultdict(list)
        for item in entry["wrong"]:
            if item["outcome"] == "other_person_group":
                rows[(item["person"], item["group_majority"])].append(item)
        other = [item for item in entry["wrong"] if item["outcome"] == "other_person_group"]
        led = sum(1 for item in other if led_by_newcomer(item))
        kept = entry["kept_faces"]
        print(f"keep {size}: kept faces {kept}; in another person's group by labelled faces {len(other)} "
              f"({100 * len(other) / kept:.1f}%), of them in a group with more of the newcomer's kept faces than of any "
              f"labelled person {led}; left in a group led by another person {len(other) - led} ({100 * (len(other) - led) / kept:.1f}%)")
        print(f"keep {size}: newcomer | receiving group (labelled majority) | faces | led by newcomer | attached | median group size | "
              "median unlabelled faces in the group | median of them likely the newcomer | median mean ratio | "
              "median best score | median lead")
        for (person, majority), items in sorted(rows.items(), key=lambda pair: -len(pair[1])):
            print(f"  {person} | {majority} | {len(items)} | {sum(led_by_newcomer(item) for item in items)} | "
                  f"{sum(item['attached'] for item in items)} | "
                  f"{_median([item['group_size'] for item in items])} | {_median([item.get('unlabelled_others') for item in items])} | "
                  f"{_median([item.get('unlabelled_likely_same_person') for item in items])} | "
                  f"{_median([item.get('mean_ratio') for item in items])} | {_median([item.get('best_other_group_score') for item in items])} | "
                  f"{_median([item.get('lead') for item in items])}")


def _composition(name: str, members: np.ndarray, labels: np.ndarray, guesses: dict[int, str | None], attached: np.ndarray | None) -> None:
    labelled = Counter(labels[row] for row in members if labels[row] is not None)
    unlabelled = [row for row in members if labels[row] is None]
    likely = Counter(guesses[row] for row in unlabelled)
    print(f"{name}: group of {len(members)} faces; labelled {dict(labelled.most_common(6))}; unlabelled {len(unlabelled)}, "
          f"estimated {dict((str(key), value) for key, value in likely.most_common(6))}")
    if attached is not None:
        for flag, label in ((False, "joined by clustering"), (True, "joined by the attach step")):
            part = [row for row in members if bool(attached[row]) == flag]
            print(f"  {label}: {len(part)} faces; labelled {dict(Counter(labels[row] for row in part if labels[row] is not None).most_common(4))}; "
                  f"unlabelled estimated {dict((str(key), value) for key, value in Counter(guesses[row] for row in part if labels[row] is None).most_common(4))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strangers", type=Path)
    parser.add_argument("--keep", type=_names)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--remove-people", type=_names)
    parser.add_argument("--people", type=_names, default=[])
    parser.add_argument("--lead", type=float, default=1.0, help="lead needed for an estimated person")
    parser.add_argument("--keep-photos", type=int, default=0, help="answer-key photos kept per removed person")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.strangers:
        explain_strangers(args.strangers, args.keep)
    if not (args.groups or args.remove_people):
        return 0
    if not (args.run and args.policy and args.people):
        parser.error("--groups and --remove-people need --run, --policy and --people")
    policy = GroupingPolicy.from_json(json.loads(args.policy.read_text(encoding="utf-8")))
    run = load_run(args.run, policy.models)
    labels = face_labels(run)
    estimate = LikelyPerson(run, policy, labels, args.lead)
    if args.groups:
        document = json.loads(args.groups.read_text(encoding="utf-8"))
        row_of = {face_id: row for row, face_id in enumerate(run.face_ids)}
        groups = [np.asarray([row_of[face] for face in group["faces"]], np.int64) for group in document["groups"]]
        for person in args.people:
            counts = [int(np.sum(labels[group] == person)) for group in groups]
            if not counts or max(counts) == 0:
                print(f"{person}: no group holds this person's labelled faces")
                continue
            members = groups[int(np.argmax(counts))]
            unlabelled = members[[labels[row] is None for row in members]]
            guesses = dict(zip(unlabelled.tolist(), estimate(unlabelled)))
            _composition(person, members, labels, guesses, None)
    if args.remove_people:
        removed = set(args.remove_people)
        hashes = {photo["sha256"] for photo in run.photos if _known_label(photo) in removed}
        if not hashes:
            parser.error("no labelled photos for --remove-people")
        generator = np.random.default_rng(args.seed)
        kept: dict[int, str] = {}
        for person in sorted(removed):
            pool = np.flatnonzero(labels == person)
            for row in generator.choice(pool, size=min(args.keep_photos, len(pool)), replace=False) if len(pool) else []:
                kept[int(row)] = person
        rows = np.asarray(sorted({row for row, index in enumerate(run.face_photo) if run.photos[int(index)]["sha256"] not in hashes}
                                 | set(kept)))
        sub = _subset(run, rows)
        result = group_faces(sub, policy)
        sub_labels = labels[rows].copy()
        position = {int(row): index for index, row in enumerate(rows)}
        for row in kept:
            sub_labels[position[row]] = None   # kept faces are newcomers: no label in the composition
        print(f"Without {', '.join(sorted(removed))} (keeping {args.keep_photos} photos each): {len(rows)} of {len(run.faces)} faces, "
              f"{len(result.group_sizes)} groups")
        unlabelled_all = np.flatnonzero([label is None for label in sub_labels])
        guess_all = dict(zip(unlabelled_all.tolist(), estimate(rows[unlabelled_all])))
        new_groups = set(result.leftover_groups)

        def describe(group: int) -> str:
            members = np.flatnonzero(result.labels == group)
            labelled = Counter(sub_labels[row] for row in members if sub_labels[row] is not None)
            newcomers = Counter(kept[int(rows[row])] for row in members if int(rows[row]) in kept)
            estimated = Counter(guess_all[row] for row in members if sub_labels[row] is None and int(rows[row]) not in kept)
            return (f"group of {len(members)}{' (from leftovers)' if group in new_groups else ''}: labelled {dict(labelled.most_common(3))}, "
                    f"kept newcomer faces {dict(newcomers.most_common(3))}, other unlabelled estimated "
                    f"{dict((str(key), value) for key, value in estimated.most_common(3))}")

        for person in sorted(removed):
            for kind, faces in (("kept faces", [position[row] for row, name in kept.items() if name == person]),
                                ("estimated faces in other photos", [row for row, guess in guess_all.items()
                                                                    if guess == person and int(rows[row]) not in kept])):
                placed = Counter(int(result.labels[row]) for row in faces)
                print(f"{person}, {kind}: {len(faces)}; not grouped {placed.pop(UNGROUPED, 0)}")
                for group, count in placed.most_common(5):
                    print(f"  {count} in {describe(group)}")
        for person in args.people:
            values, counts = np.unique(result.labels[(result.labels != UNGROUPED) & (sub_labels == person)], return_counts=True)
            if not len(values):
                print(f"{person}: no group holds this person's labelled faces")
                continue
            members = np.flatnonzero(result.labels == values[np.argmax(counts)])
            unlabelled = members[[sub_labels[row] is None for row in members]]
            guesses = dict(zip(unlabelled.tolist(), estimate(rows[unlabelled])))
            _composition(person, members, sub_labels, guesses, result.attached)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
