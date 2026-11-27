"""Compare settings for new groups formed from leftover faces.

    python Scripts/sweep_leftover_groups.py RUN --policy POLICY --people A,B,... \
        --thresholds 5,6,7 --min-sizes 2,3 [--exclude-people X,Y] [--keep 3,10,30] [--output FILE]

Two measurements for every (threshold, minimum size) setting:

1. Known people: mixing and missing for ``--people`` on the whole collection,
   with the labels of ``--exclude-people`` hidden.
2. Newcomers: the ``photos strangers`` test on ``--people``; how often a
   person who keeps only a few photos gets a group of their own, and how
   often their faces land in a group with other people.

It prints names and numbers only.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--people", type=_names, required=True, help="people to measure (never the test people while tuning)")
    parser.add_argument("--exclude-people", type=_names, default=[], help="people whose labels are hidden")
    parser.add_argument("--thresholds", type=lambda value: [float(item) for item in _names(value)], required=True)
    parser.add_argument("--min-sizes", type=lambda value: [int(item) for item in _names(value)], default=[3])
    parser.add_argument("--keep", type=lambda value: [int(item) for item in _names(value)], default=[3, 10, 30])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-strangers", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = GroupingPolicy.from_json(json.loads(args.policy.read_text(encoding="utf-8")))
    settings = list(product(args.thresholds, args.min_sizes))
    run = load_run(args.run, policy.models)
    if args.exclude_people:
        hide_labels(run, args.exclude_people)
    report = {"run": str(args.run), "policy": str(args.policy), "people": args.people, "excluded_people": args.exclude_people,
              "known_people": leftover_sweep(run, policy, settings, args.people)}
    print("Known people: threshold | min size | groups | new groups (faces) | mixed faces (in new groups) | mixing | "
          "missing | in another group of the person | left for review")
    for row in report["known_people"]:
        new = row["leftover_groups"]
        setting = "off | -" if row["leftover_merge_threshold"] is None else f"{row['leftover_merge_threshold']} | {row['leftover_min_size']}"
        print(f"{setting} | {row['groups']} | {new['groups']} ({new['faces']}) | "
              f"{row['mixed_faces']} ({new.get('mixed_faces', 0)}) | {row['mixing_rate']:.5f} | {row['missing_rate']:.4f} | "
              f"{row['in_other_group_of_person']} | {row['left_for_review']}")
    if not args.skip_strangers:
        report["newcomers"] = stranger_test(run, policy, args.people, args.keep, args.repeats, args.batch, args.seed, settings)
        print("Newcomers: keep | threshold | min size | own group | another person's group | unlabelled group | not grouped")
        for size, entry in report["newcomers"]["keep"].items():
            print(f"{size} | policy | - | {entry['own_group']} | {entry['other_person_group']} | {entry['unlabelled_group']} | {entry['not_grouped']}")
            for row in entry["leftover_grid"]:
                print(f"{size} | {row['leftover_merge_threshold']} | {row['leftover_min_size']} | {row['own_group']} | "
                      f"{row['other_person_group']} | {row['unlabelled_group']} | {row['not_grouped']}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_plain), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
