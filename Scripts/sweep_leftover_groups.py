"""Compare settings for new groups formed from leftover faces.

    python Scripts/sweep_leftover_groups.py RUN --policy POLICY --people A,B,... \
        --thresholds 5,6,7 --min-sizes 2,3 [--known-caps none,0.5] [--policy-leftover 6:2:0.5] \
        [--exclude-people X,Y] [--keep 3,10,30] [--output FILE]

Two measurements for every (threshold, minimum size, known-group score limit) setting:

1. Known people: mixing and missing for ``--people`` on the whole collection,
   with the labels of ``--exclude-people`` hidden.
2. Newcomers: the ``photos strangers`` test on ``--people``; how often a
   person who keeps only a few photos gets a group of their own, and how
   often their faces land in a group with other people. ``--policy-leftover``
   sets the policy's own leftover setting; the "policy" rows and the detailed
   list of wrong placements use it.

It prints names and numbers only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoatlas.photo_evaluation import hide_labels, leftover_sweep, stranger_test  # noqa: E402
from videoatlas.photo_grouping import GroupingPolicy  # noqa: E402
from videoatlas.photos import load_run  # noqa: E402


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _setting(row: dict) -> str:
    if row["leftover_merge_threshold"] is None:
        return "off | - | -"
    return f"{row['leftover_merge_threshold']} | {row['leftover_min_size']} | {row['leftover_max_known_score']}"


def _rates(entry: dict) -> str:
    kept = entry["kept_faces"] or 1
    return "/".join(f"{100 * entry[name] / kept:.1f}" for name in ("own_group", "other_person_group", "unlabelled_group"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--people", type=_names, required=True, help="people to measure (never the test people while tuning)")
    parser.add_argument("--exclude-people", type=_names, default=[], help="people whose labels are hidden")
    parser.add_argument("--thresholds", type=lambda value: [float(item) for item in _names(value)], required=True)
    parser.add_argument("--min-sizes", type=lambda value: [int(item) for item in _names(value)], default=[3])
    parser.add_argument("--known-caps", type=lambda value: [None if item.lower() == "none" else float(item) for item in _names(value)],
                        default=[None], help="known-group score limits; none means no limit")
    parser.add_argument("--policy-leftover", help="THRESHOLD:MIN_SIZE[:KNOWN_CAP] for the policy's own leftover step")
    parser.add_argument("--keep", type=lambda value: [int(item) for item in _names(value)], default=[3, 10, 30])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-strangers", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = GroupingPolicy.from_json(json.loads(args.policy.read_text(encoding="utf-8")))
    if args.policy_leftover:
        parts = args.policy_leftover.split(":")
        policy = replace(policy, leftover_merge_threshold=float(parts[0]), leftover_min_size=int(parts[1]),
                         leftover_max_known_score=float(parts[2]) if len(parts) > 2 and parts[2].lower() != "none" else None)
        policy.validate()
    settings = list(product(args.thresholds, args.min_sizes, args.known_caps))
    run = load_run(args.run, policy.models)
    if args.exclude_people:
        hide_labels(run, args.exclude_people)
    report = {"run": str(args.run), "policy": str(args.policy), "policy_leftover": args.policy_leftover,
              "people": args.people, "excluded_people": args.exclude_people,
              "known_people": leftover_sweep(run, policy, settings, args.people)}
    if args.skip_strangers:
        print("Known people: threshold | min size | known cap | groups | new groups (faces) | mixed faces (in new groups) | mixing | "
              "missing | in another group of the person | left for review")
        for row in report["known_people"]:
            new = row["leftover_groups"]
            print(f"{_setting(row)} | {row['groups']} | {new['groups']} ({new['faces']}) | "
                  f"{row['mixed_faces']} ({new.get('mixed_faces', 0)}) | {row['mixing_rate']:.5f} | {row['missing_rate']:.4f} | "
                  f"{row['in_other_group_of_person']} | {row['left_for_review']}")
    else:
        report["newcomers"] = stranger_test(run, policy, args.people, args.keep, args.repeats, args.batch, args.seed, settings)
        keep = report["newcomers"]["keep"]
        print("One row per setting. Known people: new groups (faces), mixed faces (in new groups), photos in another group "
              "of the person, photos left for review. Newcomers, per photos kept: own group / another person's group / "
              "unlabelled group, in % of kept faces.")
        print("threshold | min size | known cap | new groups | mixed | other own group | review | "
              + " | ".join(f"keep {size}" for size in keep))
        grid_rows = {size: {_setting(row): row for row in entry["leftover_grid"]} for size, entry in keep.items()}
        for row in report["known_people"]:
            new = row["leftover_groups"]
            newcomer = " | ".join("-" if row["leftover_merge_threshold"] is None and args.policy_leftover else
                                  _rates(keep[size] if row["leftover_merge_threshold"] is None else grid_rows[size][_setting(row)])
                                  for size in keep)
            print(f"{_setting(row)} | {new['groups']} ({new['faces']}) | {row['mixed_faces']} ({new.get('mixed_faces', 0)}) | "
                  f"{row['in_other_group_of_person']} | {row['left_for_review']} | {newcomer}")
        if args.policy_leftover:
            print(f"policy with leftover setting {args.policy_leftover}: " + " | ".join(_rates(keep[size]) for size in keep))
        for size, entry in keep.items():
            print(f"keep {size}, policy: wrong placements {entry['other_person_group'] + entry['unlabelled_group']}, of them in new groups "
                  f"{entry['wrong_in_new_groups']}; unlabelled-group placements whose unlabelled faces are likely the same person "
                  f"{entry['unlabelled_group_likely_same_person']} of {entry['unlabelled_group']}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_plain), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
