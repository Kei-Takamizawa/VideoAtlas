"""``python -m videoatlas.cli photos ...``: analyze, measure, calibrate, and group photos."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np


def _models(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("list at least one model")
    return names


def _grid(value: str) -> list[float]:
    """Either comma-separated values or start:stop:step (stop included)."""
    if ":" in value:
        start, stop, step = (float(part) for part in value.split(":"))
        if step <= 0 or stop < start:
            raise argparse.ArgumentTypeError("grid needs start <= stop and a positive step")
        count = int(round((stop - start) / step)) + 1
        return [round(start + step * index, 6) for index in range(count)]
    return [float(item) for item in value.split(",") if item.strip()]


def _optional_grid(value: str) -> list[float | None]:
    """Like a grid; the item "none" means the step is off."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    keep_none = any(item.lower() == "none" for item in items)
    rest = ",".join(item for item in items if item.lower() != "none")
    return ([None] if keep_none else []) + (_grid(rest) if rest else [])


def _write_json(path: Path, data: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def cmd_analyze(args) -> int:
    from .cli import build_options, require_device
    from .embedders import OnnxEmbedder, resolve_spec
    from .images import iter_image_files
    from .model_adapters import SCRFDDetector
    from .photos import PhotoAnalyzer, analyze_folder
    root = args.root.expanduser().resolve()
    paths = iter_image_files(root)
    if args.limit is not None:
        paths = paths[:args.limit]
    options = build_options(args.config, args.device, args.cache_dir)
    options = replace(options, detection_threshold=args.detection_threshold)
    options.validate()
    detector = SCRFDDetector(options)
    embedders = [OnnxEmbedder(resolve_spec(name), options, args.model_dir) for name in args.models]
    analyzer = PhotoAnalyzer(detector, embedders)
    require_device(_ProviderView(analyzer), args.device)
    started = time.monotonic()

    def progress(index: int, total: int) -> None:
        if index % 200 == 0 or index == total:
            rate = index / max(time.monotonic() - started, 1e-9)
            print(f"{index}/{total} photos ({rate:.1f}/s)", file=sys.stderr, flush=True)

    report = analyze_folder(analyzer, root, args.output.expanduser().resolve(), paths, progress,
                            {"device": args.device, "detection_threshold": args.detection_threshold})
    print(json.dumps({key: report[key] for key in ("image_count", "elapsed_seconds", "images_per_second", "providers",
                                                    "usable_faces_by_eye_distance_px")}, indent=2, ensure_ascii=False))
    return 0


class _ProviderView:
    """Adapts PhotoAnalyzer to the provider check used by the other commands."""
    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer

    def execution_providers(self):
        return self.analyzer.providers()


def cmd_verify(args) -> int:
    from .photo_evaluation import add_fused_embedding, compare_models, verification_report
    from .photos import load_run
    fused = args.fuse or []
    needed = None if args.models is None else sorted(set(args.models) | {model for group in fused for model in group})
    run = load_run(args.run, needed)
    names = list(args.models or sorted(run.embeddings))
    names += [add_fused_embedding(run, group) for group in fused]
    result = {"run": str(args.run), "models": {name: verification_report(run, name) for name in names},
              "comparison": compare_models(run, names, repetitions=args.bootstrap, seed=args.seed),
              "notes": ["Single-face pairs from labelled photos whose file is not filed under two people.",
                        "Score-only evidence; grouping accuracy is measured by the calibrate command."]}
    _write_json(args.output, result)
    rows = {name: {key: result["models"][name].get(key) for key in ("auc", "eer", "tar_at_fmr_0.001", "threshold_at_fmr_0.001")}
            for name in names}
    print(json.dumps(rows, indent=2))
    return 0


def _base_policy(args, run):
    from .photo_evaluation import impostor_normalization
    from .photo_grouping import GroupingPolicy
    offsets = scales = None
    if len(args.models) > 1:
        offsets, scales = impostor_normalization(run, args.models)
    return GroupingPolicy(tuple(args.models), merge_threshold=0.0, attach_threshold=None, attach_margin=args.attach_margin,
                          attach_top_k=args.top_k, min_group_size=args.min_group_size, linkage=args.linkage,
                          cluster_min_size=args.cluster_min_size, suggest_threshold=args.suggest_threshold,
                          max_suggestions=args.max_suggestions, offsets=offsets, scales=scales)


def _brief(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return {**{key: metrics[key] for key in ("groups", "grouped_faces", "attached_faces", "review_faces", "merge_suggestions")},
            "mixing": {key: value for key, value in metrics["mixing"].items() if key != "details"},
            "missing": metrics["missing"], "people": metrics["people"]}


def cmd_calibrate(args) -> int:
    from .photo_evaluation import SearchGrid, calibrate_leave_one_person_out, evaluate_grid, pick, table_row
    from .photos import load_run
    run = load_run(args.run, args.models)
    base = _base_policy(args, run)
    grid = SearchGrid(tuple(args.merge_grid), tuple(args.attach_grid), tuple(args.eye_grid), tuple(args.cluster_grid))
    targets = sorted(set(args.targets) | {args.policy_target})
    started = time.monotonic()
    held_out = calibrate_leave_one_person_out(run, base, grid, targets)
    table = evaluate_grid(run, base, grid)
    all_people = {}
    for target in targets:
        policy, metrics = pick(table, target)
        all_people[str(target)] = {"policy": policy.to_json() if policy else None, "metrics": _brief(metrics)}
    result = {"run": str(args.run), "models": args.models, "policy_target": args.policy_target,
              "grid": {"merge": grid.merge, "attach": grid.attach, "min_eye_distance": grid.eye, "cluster_merge": grid.cluster,
                       "settings": len(table)},
              "held_out": held_out,
              "all_people": {"targets": all_people,
                             "note": "Chosen and measured on the same people: optimistic. Use the held_out numbers as the estimate."},
              "table": [table_row(policy, metrics) for policy, metrics in table],
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    policy, _ = pick(table, args.policy_target)
    if policy is not None and args.policy_output:
        calibrated = replace(policy, review_threshold=args.review_threshold,
                             calibration={"run": str(args.run), "max_mixing_rate": args.policy_target,
                                          "held_out": held_out["targets"][str(args.policy_target)]["held_out"]})
        _write_json(args.policy_output, calibrated.to_json())
    _write_json(args.output, result)
    digest = {target: {key: entry["held_out"][key] for key in ("mixing_rate", "missing_rate", "folds_without_setting")}
              for target, entry in held_out["targets"].items()}
    print(json.dumps({"held_out": digest, "policy": policy.to_json() if policy else None}, indent=2, default=_json_default))
    return 0


def cmd_identify(args) -> int:
    from .photo_evaluation import identification_report
    from .photos import load_run
    run = load_run(args.run, args.models)
    result = {name: identification_report(run, name, args.enroll, args.repeats, max_wrong_rate=args.max_wrong) for name in args.models}
    _write_json(args.output, result)
    print(json.dumps({name: {size: entry["best_within_limit"] for size, entry in data["enroll_sizes"].items()}
                      for name, data in result.items()}, indent=2))
    return 0


def cmd_group(args) -> int:
    from .photo_evaluation import grouping_metrics
    from .photo_grouping import GroupingPolicy, group_faces, groups_document, write_groups
    from .photos import load_run
    policy = GroupingPolicy.from_json(json.loads(Path(args.policy).read_text(encoding="utf-8")))
    run = load_run(args.run, policy.models)
    result = group_faces(run, policy)
    document = groups_document(run, result, str(args.run))
    if any(photo.get("label") for photo in run.photos):
        document["evaluation"] = grouping_metrics(run, result)
    write_groups(args.output, document)
    summary = {"groups": document["group_count"], "review_faces": len(document["review"])}
    if "evaluation" in document:
        summary.update(mixing=document["evaluation"]["mixing"]["mixing_rate"], missing=document["evaluation"]["missing"]["missing_rate"])
    print(json.dumps(summary, indent=2))
    return 0


def add_photo_commands(commands, common) -> None:
    photos = commands.add_parser("photos", help="analyze, measure, calibrate, and group photos by person")
    actions = photos.add_subparsers(dest="photo_command", required=True)

    analyze = actions.add_parser("analyze", parents=[common], help="detect and embed every face with one or more models")
    analyze.add_argument("root", type=Path, help="photo folder, optionally with one subfolder per person")
    analyze.add_argument("--output", type=Path, required=True, help="new run folder")
    analyze.add_argument("--models", type=_models, default=["face01"], help="comma-separated model names; add +flip for mirror averaging")
    analyze.add_argument("--model-dir", type=Path, help="folder that holds the model files (default: videoatlas/resources)")
    analyze.add_argument("--detection-threshold", type=float, default=0.5)
    analyze.add_argument("--limit", type=int)
    analyze.set_defaults(handler=cmd_analyze)

    verify = actions.add_parser("verify", help="compare models on single-face pairs from labelled photos")
    verify.add_argument("run", type=Path)
    verify.add_argument("--models", type=_models)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--fuse", type=_models, action="append",
                        help="also measure the fused score of these comma-separated models (repeatable)")
    verify.add_argument("--bootstrap", type=int, default=1000)
    verify.add_argument("--seed", type=int, default=0)
    verify.set_defaults(handler=cmd_verify)

    calibrate = actions.add_parser("calibrate", help="choose thresholds with leave-one-person-out evaluation")
    calibrate.add_argument("run", type=Path)
    calibrate.add_argument("--models", type=_models, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--policy-output", type=Path, help="write the policy chosen on all people here")
    calibrate.add_argument("--targets", type=_grid, default=[0.0, 0.0005, 0.001, 0.002, 0.005, 0.01],
                           help="mixing limits to evaluate; each gives one held-out estimate")
    calibrate.add_argument("--policy-target", type=float, default=0.002, help="mixing limit for the written policy")
    calibrate.add_argument("--merge-grid", type=_grid, default=_grid("0.20:0.70:0.025"))
    calibrate.add_argument("--attach-grid", type=_optional_grid, default=[None, *_grid("0.25:0.60:0.05")])
    calibrate.add_argument("--eye-grid", type=_grid, default=[0.0, 16.0])
    calibrate.add_argument("--cluster-grid", type=_optional_grid, default=[None],
                           help="cluster merge thresholds (normalized score); none turns the step off")
    calibrate.add_argument("--cluster-min-size", type=int, default=3)
    calibrate.add_argument("--attach-margin", type=float, default=0.05)
    calibrate.add_argument("--top-k", type=int, default=5)
    calibrate.add_argument("--min-group-size", type=int, default=2)
    calibrate.add_argument("--linkage", choices=("average", "complete"), default="average")
    calibrate.add_argument("--suggest-threshold", type=float,
                           help="lowest normalized score at which two groups are suggested as one person")
    calibrate.add_argument("--max-suggestions", type=int, default=200)
    calibrate.add_argument("--review-threshold", type=float, help="lowest score at which a group is suggested for a review face")
    calibrate.set_defaults(handler=cmd_calibrate)

    identify = actions.add_parser("identify", help="measure matching against a few named example photos per person")
    identify.add_argument("run", type=Path)
    identify.add_argument("--models", type=_models, required=True)
    identify.add_argument("--output", type=Path, required=True)
    identify.add_argument("--enroll", type=lambda value: [int(item) for item in value.split(",")], default=[3, 5, 10, 20])
    identify.add_argument("--repeats", type=int, default=5)
    identify.add_argument("--max-wrong", type=float, default=0.001)
    identify.set_defaults(handler=cmd_identify)

    group = actions.add_parser("group", help="group a run with a policy file and write the groups file")
    group.add_argument("run", type=Path)
    group.add_argument("--policy", type=Path, required=True)
    group.add_argument("--output", type=Path, required=True)
    group.set_defaults(handler=cmd_group)
