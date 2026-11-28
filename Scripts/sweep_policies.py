"""Compare variants of one grouping policy on labelled people.

    python Scripts/sweep_policies.py RUN --policy BASE --people A,B,C \
        --set leftover_join_threshold=none,2.0,3.0 --set leftover_join_min_ratio=none,0.5 \
        --variant "attach_min_ratio=0.5 leftover_join_threshold=2.0" \
        [--newcomer-keep 1,3,10,30 --repeats 3 --batch 5 --seed 0] [--output FILE] [--policy-dir DIR]

Each ``--set name=v1,v2`` gives values for one policy field (``none`` means
None); every combination is one variant. Each ``--variant`` adds one variant
with the listed changes. The base policy is always the first row. The fused scores and the first clustering step are computed once,
so the variants may differ in any setting except the models, their
normalization, and the gate (``merge_threshold`` may differ).

For each variant the script prints mixing and missing for the given people,
the causes of the missing photos, and with ``--newcomer-keep`` the newcomer
test on the same batches for every variant. Use development people only when
the output guides a change. The script prints names and numbers only.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoatlas.photo_evaluation import face_labels, grouping_metrics, led_by_newcomer, photo_labels, stranger_test  # noqa: E402
from videoatlas.photo_grouping import (GroupingPolicy, agglomerate, fused_similarity, gate_mask, group_faces,  # noqa: E402
                                       same_photo_constraints, square_block)
from videoatlas.photos import load_run  # noqa: E402

FIXED = ("models", "weights", "offsets", "scales", "min_eye_distance", "max_abs_yaw_ratio", "linkage", "calibration")


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _value(text: str):
    text = text.strip()
    if text.lower() == "none":
        return None
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    number = float(text)
    return int(number) if number.is_integer() and "." not in text else number


def _setting(text: str) -> tuple[str, list]:
    name, _, values = text.partition("=")
    known = {field.name for field in fields(GroupingPolicy)}
    if name not in known or name in FIXED:
        raise argparse.ArgumentTypeError(f"cannot vary {name!r}")
    return name, [_value(value) for value in values.split(",")]


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def _variant(text: str) -> dict:
    changes = {}
    for item in text.split():
        name, values = _setting(item)
        if len(values) != 1:
            raise argparse.ArgumentTypeError(f"one value per field in --variant: {item!r}")
        changes[name] = values[0]
    return changes


def variants(base: GroupingPolicy, settings: list[tuple[str, list]], extra: list[dict] = ()) -> list[tuple[dict, GroupingPolicy]]:
    rows = [({}, base)]
    names = [name for name, _ in settings]
    combinations = [dict(zip(names, values)) for values in itertools.product(*(values for _, values in settings))] if settings else []
    for changes in combinations + list(extra):
        policy = replace(base, **changes)
        policy.validate()
        if policy != base and all(policy != existing for _, existing in rows):
            rows.append((changes, policy))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--people", type=_names, required=True)
    parser.add_argument("--set", type=_setting, action="append", default=[], dest="settings")
    parser.add_argument("--variant", type=_variant, action="append", default=[], dest="extra")
    parser.add_argument("--newcomer-keep", type=lambda value: [int(item) for item in _names(value)], default=[])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write all results as JSON")
    parser.add_argument("--policy-dir", type=Path, help="write each variant's policy file here")
    args = parser.parse_args()

    base = GroupingPolicy.from_json(json.loads(args.policy.read_text(encoding="utf-8")))
    rows = variants(base, args.settings, args.extra)
    run = load_run(args.run, base.models)
    labels, photos = face_labels(run), photo_labels(run)
    started = time.monotonic()
    similarity = fused_similarity(run.embeddings, base)
    passed = np.flatnonzero(gate_mask(run.faces, base))
    floor = min(policy.merge_threshold for _, policy in rows)
    block = square_block(similarity, passed)
    merges = agglomerate(block, square_block(same_photo_constraints(run.face_photo), passed), base.linkage, floor,
                         overwrite=block is not similarity)
    del block
    results = []
    for index, (changes, policy) in enumerate(rows):
        result = group_faces(run, policy, similarity=similarity, merges=merges)
        metrics = grouping_metrics(run, result, labels, args.people, photos=photos)
        missing = metrics["missing"]
        results.append({"variant": index, "changes": changes, "groups": metrics["groups"],
                        "new_groups": len(result.leftover_groups), "joined_or_attached_faces": metrics["attached_faces"],
                        "review_faces": metrics["review_faces"], "mixed_faces": metrics["mixing"]["mixed_faces"],
                        "mixing_rate": metrics["mixing"]["mixing_rate"], "mixed_groups": metrics["mixing"]["mixed_groups"],
                        "missing_rate": missing["missing_rate"], "photos": missing["photos"],
                        **{key: missing[key] for key in ("no_usable_face", "in_other_group_of_person", "left_for_review",
                                                         "only_in_other_people_groups")},
                        "mixing_details": [{key: value for key, value in item.items() if key != "intruder_photos"}
                                           for item in metrics["mixing"]["details"]]})
        if args.policy_dir:
            args.policy_dir.mkdir(parents=True, exist_ok=True)
            (args.policy_dir / f"variant-{index}.json").write_text(json.dumps(policy.to_json(), indent=1), encoding="utf-8")
    del similarity, merges
    grouping_seconds = round(time.monotonic() - started, 1)

    for index, (_, policy) in enumerate(rows):
        if not args.newcomer_keep:
            break
        report = stranger_test(run, policy, args.people, args.newcomer_keep, args.repeats, args.batch, args.seed)
        results[index]["newcomers"] = {
            size: {**{key: entry[key] for key in ("kept_faces", "not_grouped", "own_group", "other_person_group", "unlabelled_group",
                                                  "own_group_rate", "other_person_group_led_by_newcomer",
                                                  "in_group_led_by_another_person_rate", "wrong_in_new_groups")},
                   "wrong": [item for item in entry["wrong"] if item["outcome"] == "other_person_group" and not led_by_newcomer(item)][:30]}
            for size, entry in report["keep"].items()}

    if args.output:
        args.output.write_text(json.dumps({"run": str(args.run), "policy": str(args.policy), "people": args.people,
                                           "grouping_seconds": grouping_seconds, "results": results}, indent=1), encoding="utf-8")

    print(f"{len(args.people)} people, {results[0]['photos']} labelled photos; grouping of {len(rows)} variants: {grouping_seconds} s")
    print("variant | changes | groups | new groups | mixed faces | mixing | missing | fragment | review | other groups | no face")
    for row in results:
        changes = ", ".join(f"{key}={value}" for key, value in row["changes"].items()) or "base"
        print(f"{row['variant']} | {changes} | {row['groups']} | {row['new_groups']} | {row['mixed_faces']} ({row['mixed_groups']} groups) | "
              f"{_percent(row['mixing_rate'])} | {_percent(row['missing_rate'])} | {row['in_other_group_of_person']} | "
              f"{row['left_for_review']} | {row['only_in_other_people_groups']} | {row['no_usable_face']}")
    if args.newcomer_keep:
        print("\nNewcomers: share of kept photos in their own group / left in a group led by another person, per photos kept")
        print("variant | " + " | ".join(str(size) for size in args.newcomer_keep))
        for row in results:
            cells = [f"{_percent(entry['own_group_rate'])} / {_percent(entry['in_group_led_by_another_person_rate'])}"
                     for entry in row["newcomers"].values()]
            print(f"{row['variant']} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
