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


def _attach_grid(value: str) -> list[float | None]:
    return [None] if value.strip().lower() == "none" else _grid(value)


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
    from .photo_evaluation import compare_models, verification_report
    from .photos import load_run
    run = load_run(args.run, args.models)
    names = args.models or sorted(run.embeddings)
    result = {"run": str(args.run), "models": {name: verification_report(run, name) for name in names},
              "comparison": compare_models(run, names, repetitions=args.bootstrap, seed=args.seed),
              "notes": ["Single-face pairs from labelled photos whose file is not filed under two people.",
                        "Score-only evidence; grouping accuracy is measured by the calibrate command."]}
    _write_json(args.output, result)
    rows = {name: {key: result["models"][name].get(key) for key in ("auc", "eer", "tar_at_fmr_0.001", "threshold_at_fmr_0.001")}
            for name in names}
    print(json.dumps(rows, indent=2))
    return 0


def _base_policy(args, run) -> "GroupingPolicy":
    from .photo_evaluation import impostor_normalization
    from .photo_grouping import GroupingPolicy
    offsets = scales = None
    if len(args.models) > 1:
        offsets, scales = impostor_normalization(run, args.models)
    return GroupingPolicy(tuple(args.models), merge_threshold=0.0, attach_threshold=None, attach_margin=args.attach_margin,
                          attach_top_k=args.top_k, min_group_size=args.min_group_size, linkage=args.linkage,
                          offsets=offsets, scales=scales)


def cmd_calibrate(args) -> int:
    from .photo_evaluation import calibrate_leave_one_person_out, choose_policy
    from .photos import load_run
    run = load_run(args.run, args.models)
    base = _base_policy(args, run)
    started = time.monotonic()
    held_out = calibrate_leave_one_person_out(run, base, args.merge_grid, args.attach_grid, args.eye_grid, args.max_mixing)
    policy, metrics, table = choose_policy(run, base, args.merge_grid, args.attach_grid, args.eye_grid, args.max_mixing)
    result = {"run": str(args.run), "models": args.models, "max_mixing_rate": args.max_mixing,
              "grid": {"merge": args.merge_grid, "attach": args.attach_grid, "min_eye_distance": args.eye_grid},
              "held_out": held_out,
              "all_people": {"policy": policy.to_json() if policy else None, "metrics": metrics,
                             "note": "Chosen and measured on the same people: optimistic. Use the held_out numbers as the estimate."},
              "table": table, "elapsed_seconds": round(time.monotonic() - started, 1)}
    if policy is not None and args.policy_output:
        calibrated = replace(policy, review_threshold=args.review_threshold,
                             calibration={"run": str(args.run), "max_mixing_rate": args.max_mixing,
                                          "held_out": held_out["held_out"]})
        _write_json(args.policy_output, calibrated.to_json())
    _write_json(args.output, result)
    print(json.dumps({"held_out": held_out["held_out"], "policy": result["all_people"]["policy"]}, indent=2, default=_json_default))
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
    verify.add_argument("--bootstrap", type=int, default=1000)
    verify.add_argument("--seed", type=int, default=0)
    verify.set_defaults(handler=cmd_verify)

    for name, handler, helptext in (("calibrate", cmd_calibrate, "choose thresholds with leave-one-person-out evaluation"),):
        calibrate = actions.add_parser(name, help=helptext)
        calibrate.add_argument("run", type=Path)
        calibrate.add_argument("--models", type=_models, required=True)
        calibrate.add_argument("--output", type=Path, required=True)
        calibrate.add_argument("--policy-output", type=Path, help="write the policy chosen on all people here")
        calibrate.add_argument("--max-mixing", type=float, default=0.002, help="highest allowed share of mixed faces on the people used for choosing")
        calibrate.add_argument("--merge-grid", type=_grid, default=_grid("0.20:0.70:0.025"))
        calibrate.add_argument("--attach-grid", type=_attach_grid, default=_grid("0.25:0.80:0.05"))
        calibrate.add_argument("--eye-grid", type=_grid, default=[0.0, 12.0, 16.0, 20.0, 25.0])
        calibrate.add_argument("--attach-margin", type=float, default=0.05)
        calibrate.add_argument("--top-k", type=int, default=5)
        calibrate.add_argument("--min-group-size", type=int, default=2)
        calibrate.add_argument("--linkage", choices=("average", "complete"), default="average")
        calibrate.add_argument("--review-threshold", type=float, help="lowest score at which a group is suggested for a review face")
        calibrate.set_defaults(handler=handler)

    group = actions.add_parser("group", help="group a run with a policy file and write the groups file")
    group.add_argument("run", type=Path)
    group.add_argument("--policy", type=Path, required=True)
    group.add_argument("--output", type=Path, required=True)
    group.set_defaults(handler=cmd_group)
