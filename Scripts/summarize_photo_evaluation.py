"""Print a short digest of a photo evaluation run (verify.json and calibration-*.json).

    python Scripts/summarize_photo_evaluation.py RUN_FOLDER

The digest holds counts, rates, thresholds, and person folder names only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def num(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(folder: str) -> int:
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
        held = data["held_out"]["held_out"]
        print(f"\nCalibration {path.stem} (max mixing {pct(data['max_mixing_rate'])}, {data['elapsed_seconds']} s)")
        print(f"  held-out: mixing {held['mixed_faces']}/{held['grouped_labelled_faces']} = {pct(held['mixing_rate'])}; "
              f"missing {held['photos'] - held['found_in_main_group']}/{held['photos']} = {pct(held['missing_rate'])} "
              f"(no usable face {held['no_usable_face']}, review {held['left_for_review']}, other group of same person "
              f"{held['in_other_group_of_person']}, only in other people's groups {held['only_in_other_people_groups']})")
        for fold in data["held_out"]["folds"]:
            policy = fold.get("policy")
            if policy is None:
                print(f"  - {fold['held_out']}: no setting met the limit")
                continue
            person = fold["held_out_person"]
            print(f"  - {fold['held_out']}: eye>={policy['min_eye_distance']} merge {policy['merge_threshold']} attach {policy['attach_threshold']} "
                  f"| mixed {fold['held_out_mixed_faces']}/{fold['held_out_grouped_labelled_faces']} in {fold['held_out_mixed_groups']} groups "
                  f"| missing {pct(person['missing_rate'])} of {person['photos']} (groups led {person['groups_led']}, main group {person['main_group_size']})")
            for detail in fold["held_out_mixing_details"][:3]:
                print(f"      mixed group size {detail['size']} led by {detail['majority']}: intruders {detail['intruders']}")
        chosen = data["all_people"]
        if chosen["policy"]:
            metrics = chosen["metrics"]
            policy = chosen["policy"]
            print(f"  all people (optimistic): eye>={policy['min_eye_distance']} merge {policy['merge_threshold']} attach {policy['attach_threshold']} | "
                  f"groups {metrics['groups']}, review faces {metrics['review_faces']}, mixing {pct(metrics['mixing']['mixing_rate'])}, "
                  f"missing {pct(metrics['missing']['missing_rate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
