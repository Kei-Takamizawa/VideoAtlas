"""Show where the missing photos of some people went under one grouping policy.

    python Scripts/explain_missing.py RUN --policy POLICY --people A,B,C [--output FILE]

A photo is missing when it is not in the person's main group. Each labelled
photo gets one cause: no usable face, in another group of the same person (a
fragment), its only face left for review, its only face in another person's
group, or a group photo whose faces are in none of the person's groups.

For fragments, the script shows their sizes, their merge-suggestion score to
the main group, and what a rule that joins each group to its best-scoring
partner would bring in and get wrong. For review photos, it shows whether the
first candidate group is the person's main group. It also repeats the merge
suggestion simulation (a user who answers the first N suggestions correctly).

Use development people only when the output guides a change. The script
prints names and numbers only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoatlas.photo_evaluation import MISSING_CAUSES, face_labels, missing_breakdown, photo_labels, simulate_review  # noqa: E402
from videoatlas.photo_grouping import GroupingPolicy, fused_similarity, group_faces  # noqa: E402
from videoatlas.photos import load_run  # noqa: E402


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--people", type=_names, required=True, help="comma-separated people to explain")
    parser.add_argument("--join-thresholds", type=_floats, default=(0.0, 0.1, 0.15, 0.2, 0.25, 0.3))
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    policy = GroupingPolicy.from_json(json.loads(args.policy.read_text(encoding="utf-8")))
    run = load_run(args.run, policy.models)
    similarity = fused_similarity(run.embeddings, policy)
    result = group_faces(run, policy, similarity=similarity)
    labels, photos = face_labels(run), photo_labels(run)
    report = missing_breakdown(run, result, similarity, args.people, labels, photos, args.join_thresholds)
    del similarity
    report["review_simulation"] = simulate_review(run, result, labels, photos, set(args.people))
    report["policy"] = str(args.policy)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    total = report["photos"]
    found = report["causes"]["found_in_main_group"]["photos"]
    missing = total - found
    print(f"{len(args.people)} people, {total} labelled photos, {missing} missing ({_percent(missing / total if total else None)})")
    print("cause | photos | share of all photos | share of missing")
    for cause in MISSING_CAUSES:
        if cause == "found_in_main_group":
            continue
        count = report["causes"][cause]["photos"]
        print(f"{cause} | {count} | {_percent(count / total if total else None)} | {_percent(count / missing if missing else None)}")

    fragments = report["fragments"]
    print(f"\nFragments: {fragments['groups']} groups hold {fragments['missing_photos']} missing photos; "
          f"{fragments['photos_with_pair_in_suggestions']} of them are in a group paired with the main group in the merge suggestions")
    print("by fragment size: " + ", ".join(f"{key}: {value}" for key, value in fragments["photos_by_size"].items()))
    print("by score to the main group: " + ", ".join(f"{key}: {value}" for key, value in fragments["photos_by_score_to_main"].items()))
    print("largest fragments: person | size | from leftovers | missing photos | score to main | main is best partner | "
          "best score to another person | suggestion rank")
    for entry in fragments["largest"][:15]:
        print(f"  {entry['person']} | {entry['size']} | {entry['from_leftovers']} | {entry['missing_photos']} | {entry.get('score_to_main')} | "
              f"{entry.get('main_is_best_partner')} | {entry.get('best_score_to_another_person')} | {entry.get('suggestion_rank')}")

    print("\nJoin each group to its best-scoring partner (smaller group joins), groups led by these people only:")
    print("threshold | same person | different people | with a group without labelled faces | photos into main groups")
    for row in report["fragment_joins"]:
        print(f"{row['threshold']} | {row['joins_same_person']} | {row['joins_different_people']} | "
              f"{row['joins_with_unlabelled_group']} | {row['photos_into_main_group']}")

    print("\nSingle-face review photos, by the first candidate's score:")
    print("score band | first candidate | photos")
    for row in report["review_faces"]:
        print(f"{row['band']} | {row['first_candidate']} | {row['photos']}")

    print("\nMerge suggestions answered correctly by a user:")
    print("budget | answered | accepted | rejected | missing")
    for row in report["review_simulation"]:
        print(f"{row['budget']} | {row['suggestions_answered']} | {row['accepted']} | {row['rejected']} | {_percent(row['missing_rate'])}")

    print("\nperson | photos | missing | fragment | review | other person | group photo | no face | groups led")
    for person, entry in sorted(report["per_person"].items(), key=lambda item: -(item[1]["missing_rate"] or 0)):
        print(f"{person} | {entry['photos']} | {_percent(entry['missing_rate'])} | {entry['in_own_fragment']} | {entry['single_face_in_review']} | "
              f"{entry['single_face_in_other_person_group']} | {entry['group_photo_not_found']} | {entry['no_usable_face']} | {entry['groups_led']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
