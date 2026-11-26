"""Headless VideoAtlas commands for machines without a display.

    python -m videoatlas.cli doctor [--device cuda|auto|cpu]
    python -m videoatlas.cli images ROOT --output DIR [--device cuda|auto|cpu]
    python -m videoatlas.cli photos analyze|verify|calibrate|group ...

``--device cuda`` (the default) stops with an error when any model cannot run on
the GPU, instead of silently using the CPU. ``auto`` allows a logged CPU fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

from .recognition import RecognitionOptions, default_config_path

DEVICES = ("cuda", "auto", "cpu")
OUTPUT_FILES = ("faces.jsonl", "embeddings.npz", "report.json")


def build_options(config: Path | None, device: str, cache_dir: Path | None = None) -> RecognitionOptions:
    if config is None:
        default = default_config_path()
        config = default if default.is_file() else None
    options = RecognitionOptions.load(config)
    changes: dict[str, object] = {"acceleration": device}
    if cache_dir is not None:
        changes["cache_dir"] = Path(cache_dir).expanduser().resolve()
    options = replace(options, **changes)
    options.validate()
    return options


def open_analyzer(options: RecognitionOptions):
    from .analyzer import VideoAnalyzer
    return VideoAnalyzer(options)


def require_device(analyzer, device: str) -> None:
    """Confirm every loaded model registered a GPU provider when the GPU is required."""
    if device != "cuda":
        return
    on_cpu = [name for name, providers in analyzer.execution_providers().items() if not providers or providers[0] == "CPUExecutionProvider"]
    if on_cpu:
        raise RuntimeError("GPU was requested, but these models run on the CPU: " + ", ".join(on_cpu))


def describe_runtime(analyzer) -> dict:
    return {"providers": {name: list(values) for name, values in analyzer.execution_providers().items()},
            "embedding_model": analyzer.embedding_model, "runtime": analyzer.runtime_description}


def cmd_doctor(args) -> int:
    analyzer = open_analyzer(build_options(args.config, args.device, args.cache_dir))
    try:
        require_device(analyzer, args.device)
        print(json.dumps(describe_runtime(analyzer), indent=2, ensure_ascii=False))
    finally:
        analyzer.close()
    return 0


def _face_record(face) -> dict:
    return {"box": [round(float(value), 6) for value in face.bounding_box], "detection_score": face.detection_score,
            "quality": face.quality, "rejection_reason": face.rejection_reason, "pose": face.pose,
            "accepted": face.embedding is not None}


def cmd_images(args) -> int:
    from .images import analyze_images, iter_image_files, pair_report, primary_face, update_label_stats
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Output already exists in {output}: {', '.join(existing)}. Choose a new --output folder.")
    paths = iter_image_files(root)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        paths = paths[:args.limit]
    if not paths:
        raise ValueError(f"No images found below {root}")
    output.mkdir(parents=True, exist_ok=True)
    options = build_options(args.config, args.device, args.cache_dir or output / "aligned-cache")
    analyzer = open_analyzer(options)
    started = time.monotonic()
    labels, vectors, sources, stats = [], [], [], {}
    try:
        require_device(analyzer, args.device)
        runtime = describe_runtime(analyzer)

        def progress(index: int, total: int) -> None:
            if index % 100 == 0 or index == total:
                elapsed = time.monotonic() - started
                print(f"{index}/{total} images ({index / elapsed:.1f} images/s)", file=sys.stderr, flush=True)

        with (output / "faces.jsonl").open("w", encoding="utf-8") as stream:
            for result in analyze_images(analyzer, paths, root, progress=progress):
                relative = result.path.relative_to(root).as_posix()
                primary = primary_face(result)
                update_label_stats(stats, result)
                stream.write(json.dumps({"path": relative, "label": result.label, "width": result.width, "height": result.height,
                                         "error": result.error, "faces": [_face_record(face) for face in result.faces],
                                         "primary_face": result.faces.index(primary) if primary is not None else None},
                                        ensure_ascii=False) + "\n")
                if primary is not None and result.label is not None:
                    labels.append(result.label)
                    vectors.append(primary.embedding)
                    sources.append(relative)
    finally:
        analyzer.close()
    elapsed = time.monotonic() - started
    matrix = np.asarray(vectors, np.float32) if vectors else np.zeros((0, 0), np.float32)
    np.savez_compressed(output / "embeddings.npz", vectors=matrix, labels=np.asarray(labels), paths=np.asarray(sources),
                        embedding_model=np.asarray(runtime["embedding_model"]))
    pairs = pair_report(labels, matrix)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "image_count": len(paths),
              "elapsed_seconds": round(elapsed, 2), "images_per_second": round(len(paths) / elapsed, 2) if elapsed > 0 else None,
              "device": args.device, "runtime": runtime,
              "options": {name: getattr(options, name) for name in ("acceleration", "precision", "recognition_region", "detection_threshold",
                                                                     "min_face_size", "min_quality", "blur_threshold", "pose_threshold")},
              "labels": {label: {**entry, "rejections": dict(entry["rejections"])} for label, entry in sorted(stats.items())},
              "pairs": pairs,
              "notes": ["Folder names are an offline answer key used only after inference.",
                        "Pair scores compare single images and are not the multi-sample HIGH grouping policy."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    accepted = sum(entry["images_with_accepted_face"] for entry in stats.values())
    print(json.dumps({"images": len(paths), "images_with_accepted_face": accepted, "pairs": pairs["pair_count"],
                      "auc": pairs["auc"], "recall_with_zero_false_positives": pairs["recall_with_zero_false_positives"],
                      "elapsed_seconds": report["elapsed_seconds"], "providers": runtime["providers"],
                      "output": str(output)}, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m videoatlas.cli", description="Headless VideoAtlas processing.")
    parser.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", choices=DEVICES, default="cuda",
                        help="cuda (default): fail unless every model runs on the GPU; auto: allow a logged CPU fallback; cpu: CPU only")
    common.add_argument("--config", type=Path, help="recognition.json with analysis settings")
    common.add_argument("--cache-dir", type=Path, help="folder for aligned-face and detector caches")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", parents=[common], help="load the models and report where each one runs")
    images = commands.add_parser("images", parents=[common], help="analyze a photo folder; first-level folder names label people for evaluation")
    images.add_argument("root", type=Path, help="photo folder, optionally with one subfolder per person")
    images.add_argument("--output", type=Path, required=True, help="new folder for faces.jsonl, embeddings.npz, and report.json")
    images.add_argument("--limit", type=int, help="analyze only the first N images")
    from .photo_cli import add_photo_commands
    add_photo_commands(commands, common)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    handlers = {"doctor": cmd_doctor, "images": cmd_images}
    try:
        return args.handler(args) if args.command == "photos" else handlers[args.command](args)
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError, NotADirectoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
