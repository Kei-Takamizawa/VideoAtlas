"""Download candidate face-recognition models for photo grouping from their official sources.

    python Scripts/provision_photo_models.py --models w600k-r50,glintr100,lvface-b

Only ONNX files are downloaded; nothing is unpickled or executed. Each file's
SHA-256 is printed and written to ``videoatlas/resources/photo-models.json`` so
that a later run can confirm it uses the same artifacts.

Licences: the InsightFace model packs and LVFace weights are for non-commercial
research use. VideoAtlas is a personal, non-commercial tool.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "videoatlas" / "resources"
SCRFD_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
INSIGHTFACE_RELEASE = "https://github.com/deepinsight/insightface/releases/download/v0.7/{}.zip"
LVFACE_REPOSITORY = "bytedance-research/LVFace"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "videoatlas-model-provisioning"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(data: bytes, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return sha256(data)


def zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    matches = [item for item in archive.namelist() if Path(item).name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} in the archive, found {matches}")
    return archive.read(matches[0])


def insightface_pack(pack: str, member: str, destination: Path) -> dict:
    url = INSIGHTFACE_RELEASE.format(pack)
    archive = zipfile.ZipFile(io.BytesIO(fetch(url)))
    names = [Path(item).name for item in archive.namelist()]
    provenance = "not checked"
    if "det_10g.onnx" in names:
        detector = sha256(zip_member(archive, "det_10g.onnx"))
        if detector != SCRFD_SHA256:
            raise RuntimeError(f"{pack}: det_10g.onnx hash {detector} differs from the pinned SCRFD hash; refusing the archive")
        provenance = "archive also contains the pinned SCRFD det_10g.onnx"
    digest = save(zip_member(archive, member), destination)
    return {"url": url, "member": member, "sha256": digest, "provenance": provenance, "archive_files": sorted(names)}


def lvface(variant: str, destination: Path) -> dict:
    listing = json.loads(fetch(f"https://huggingface.co/api/models/{LVFACE_REPOSITORY}"))
    revision = listing.get("sha")
    files = [item["rfilename"] for item in listing.get("siblings", [])]
    wanted = [name for name in files if name.lower().endswith(".onnx") and f"lvface-{variant}".lower() in name.lower()]
    if len(wanted) != 1:
        raise RuntimeError(f"Could not pick one LVFace-{variant.upper()} ONNX file from: {files}")
    url = f"https://huggingface.co/{LVFACE_REPOSITORY}/resolve/{revision}/{wanted[0]}"
    digest = save(fetch(url), destination)
    return {"url": url, "revision": revision, "sha256": digest, "repository_files": files}


PROVIDERS = {
    "w600k-r50": lambda: insightface_pack("buffalo_l", "w600k_r50.onnx", RESOURCES / "insightface" / "buffalo_l" / "w600k_r50.onnx"),
    "glintr100": lambda: insightface_pack("antelopev2", "glintr100.onnx", RESOURCES / "insightface" / "antelopev2" / "glintr100.onnx"),
    "lvface-b": lambda: lvface("b", RESOURCES / "lvface" / "LVFace-B_Glint360K.onnx"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="w600k-r50,glintr100,lvface-b")
    args = parser.parse_args(argv)
    manifest_path = RESOURCES / "photo-models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    failures = 0
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        if name not in PROVIDERS:
            print(f"{name}: unknown model", file=sys.stderr)
            failures += 1
            continue
        try:
            manifest[name] = PROVIDERS[name]()
            print(f"{name}: {manifest[name]['sha256']}")
        except Exception as error:  # report each model separately
            print(f"{name}: failed: {error}", file=sys.stderr)
            failures += 1
    RESOURCES.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
