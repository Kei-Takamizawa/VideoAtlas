"""Print a short digest of a photo evaluation run (verify.json and calibration-*.json).

    python Scripts/summarize_photo_evaluation.py RUN_FOLDER [--intruders]

The digest holds counts, rates, thresholds, and person folder names only.
With --intruders it also lists the photo paths of faces that were grouped
with another person in the held-out measurement, so that a person can check
whether the label or the grouping is wrong.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def num(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(folder: str, intruders: bool = False) -> int:
    run = Path(folder)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    print(f"Run {run}: {report['image_count']} photos, {report['elapsed_seconds']} s, providers {report['providers']}")
    print(f"Usable faces by eye distance (px): {report['usable_faces_by_eye_distance_px']}")
    verify_path = run / "verify.json"
    if verify_path.is_file():
        verify = json.loads(verify_path.read_text(encoding="utf-8"))
        comparison = verify["comparison"]
        print(f"\nSingle-face pairs: {comparison['faces']} faces, {comparison['people']} people, reference {comparison['reference']}")
        for name, entry in verify["models"].items():
            row = comparison["models"][name]
            tar = row["tar_at_fmr_0.001"]
            line = (f"- {name}: AUC {num(entry['auc'])} EER {pct(entry['eer'])} TAR@FMR1e-3 {pct(entry['tar_at_fmr_0.001'])} "
                    f"[{pct(tar['ci95'][0])}, {pct(tar['ci95'][1])}] thr {num(entry.get('threshold_at_fmr_0.001'))} "
                    f"TAR@1e-4 {pct(entry['tar_at_fmr_0.0001'])}")
            if "difference_ci95" in tar:
                line += f" | vs ref TAR@1e-3 {pct(tar['difference_vs_reference'])} CI [{pct(tar['difference_ci95'][0])}, {pct(tar['difference_ci95'][1])}]"
            print(line)
            print("    by smaller eye distance: " + "; ".join(
                f"{s['smaller_eye_distance_px'][0]}-{s['smaller_eye_distance_px'][1] or ''}px n={s['faces']} AUC {num(s['auc'])} TAR@1e-3 {pct(s['tar_at_fmr_0.001'])}"
                for s in entry["by_smaller_eye_distance"]))
            print("    per person (median same / median best other): " + "; ".join(
                f"{p['person']} {num(p['median_same_person_score'], 2)}/{num(p['median_best_other_person_score'], 2)}" for p in entry["per_person"]))
    for path in sorted(run.glob("calibration-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        grid = data["grid"]
        print(f"\nCalibration {path.stem}: {grid['settings']} settings, {data['elapsed_seconds']} s "
              f"(merge {grid['merge'][0]}..{grid['merge'][-1]}, attach {grid['attach']}, eye {grid['min_eye_distance']}, "
              f"cluster merge {grid['cluster_merge']})")
        held_out = data["held_out"] or {"targets": {}}
        if data.get("excluded_people"):
            print(f"  kept out of tuning: {', '.join(data['excluded_people'])}")
        print("  held-out estimate by mixing limit (limit applied on the other people):" if data["held_out"]
              else "  held-out estimate: skipped (--skip-held-out)")
        for target, entry in held_out["targets"].items():
            held = entry["held_out"]
            review = "; ".join(f"{row['budget']}: {pct(row['missing_rate'])}" for row in held["after_review"] if row["budget"])
            print(f"  - limit {pct(float(target))}: mixing {held['mixed_faces']}/{held['grouped_labelled_faces']} = {pct(held['mixing_rate'])}; "
                  f"missing {held['photos'] - held['found_in_main_group']}/{held['photos']} = {pct(held['missing_rate'])} "
                  f"(no usable face {held['no_usable_face']}, review {held['left_for_review']}, other group of same person "
                  f"{held['in_other_group_of_person']}, only in other people's groups {held['only_in_other_people_groups']})"
                  + (f"; folds without a setting {held['folds_without_setting']}" if held["folds_without_setting"] else ""))
            if review:
                print(f"      missing after answering the first N merge suggestions: {review}")
        target = str(float(data["policy_target"]))
        print(f"  folds at limit {pct(float(target))}:")
        for fold in held_out["targets"].get(target, {}).get("folds", []):
            policy = fold.get("policy")
            if policy is None:
                print(f"  - {fold['held_out']}: no setting met the limit")
                continue
            person = fold["person"]
            print(f"  - {fold['held_out']}: eye>={policy['min_eye_distance']} merge {policy['merge_threshold']} "
                  f"cluster {policy['cluster_merge_threshold']} attach {policy['attach_threshold']} "
                  f"| mixed {fold['mixed_faces']}/{fold['grouped_labelled_faces']} in {fold['mixed_groups']} groups "
                  f"| missing {pct(person['missing_rate'])} of {person['photos']} (groups led {person['groups_led']}, "
                  f"main group {person['main_group_size']}, suggestions {fold['merge_suggestions']})")
            for detail in fold["mixing_details"][:3]:
                print(f"      mixed group size {detail['size']} led by {detail['majority']}: intruders {detail['intruders']}")
                if intruders:
                    for photo in detail.get("intruder_photos", []):
                        print(f"        {photo}")
        for target, chosen in data["all_people"]["targets"].items():
            policy, metrics = chosen["policy"], chosen["metrics"]
            if not policy:
                continue
            print(f"  all people, limit {pct(float(target))} (optimistic): eye>={policy['min_eye_distance']} merge {policy['merge_threshold']} "
                  f"cluster {policy['cluster_merge_threshold']} attach {policy['attach_threshold']} | groups {metrics['groups']}, "
                  f"review faces {metrics['review_faces']}, mixing {pct(metrics['mixing']['mixing_rate'])}, "
                  f"missing {pct(metrics['missing']['missing_rate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], "--intruders" in sys.argv[2:]))
