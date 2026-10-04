"""Download and export the pinned official AdaFace R50 MS1MV2 checkpoint locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

REVISION = "c60eaa786a42c03444f3df7096dbaf9d57ae010d"
BASE = f"https://raw.githubusercontent.com/mk-minchul/AdaFace/{REVISION}"
SOURCE_HASHES = {
    "net.py": "b4db4eb0174a385fd29e5f616391b50d443f455990c8b88dcab1f8021af8ba4c",
    "LICENSE": "95b6e493eb9dba27f2150304e790ae254bab18d1611f4d6e2ade28fa3a271583",
    "README.md": "c3e65e47bb06a6387287bb76efe0fcf835ffbec466c305eb0a9744cc362ff18a",
}
CHECKPOINT_SHA256 = "234da6ce931f821e0a8e920063dc2091d8122ea46285c159d2f25d3f762fecb3"
CHECKPOINT_SIZE = 700286703
GOOGLE_DRIVE_ID = "1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""):
            checksum.update(block)
    return checksum.hexdigest()


def provision(work_dir: Path, checkpoint: Path | None = None) -> tuple[Path, Path]:
    source_dir = work_dir / "source"
    source_dir.mkdir(parents=True,exist_ok=True)
    for name,expected in SOURCE_HASHES.items():
        path = source_dir/name
        if path.is_file() and digest(path) == expected:
            continue
        with urlopen(f"{BASE}/{name}",timeout=60) as response:
            data = response.read(1024*1024)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Official AdaFace source checksum mismatch: {name}")
        path.write_bytes(data)
    path = checkpoint or work_dir / "adaface_ir50_ms1mv2.ckpt"
    if not path.is_file():
        import gdown
        temporary = path.with_suffix(".download")
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            gdown.download(id=GOOGLE_DRIVE_ID,output=str(temporary),quiet=False)
            if not temporary.is_file() or temporary.stat().st_size != CHECKPOINT_SIZE or digest(temporary) != CHECKPOINT_SHA256:
                raise RuntimeError("Official AdaFace checkpoint download was incomplete or its checksum did not match")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size != CHECKPOINT_SIZE or digest(path) != CHECKPOINT_SHA256:
        raise RuntimeError("Local AdaFace checkpoint does not match the pinned official R50 MS1MV2 artifact")
    return source_dir,path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]/"videoatlas"/"resources"
    parser.add_argument("--work-dir",type=Path,default=default_root/"adaface-provisioning")
    parser.add_argument("--output",type=Path,default=default_root/"adaface_ir50_ms1mv2.onnx")
    parser.add_argument("--checkpoint",type=Path,help="Reuse an existing official checkpoint instead of downloading 700 MB")
    args = parser.parse_args()
    source,checkpoint = provision(args.work_dir,args.checkpoint)
    subprocess.run([sys.executable,str(Path(__file__).with_name("export_adaface.py")),"--source-dir",str(source),
                    "--checkpoint",str(checkpoint),"--architecture","ir_50","--output",str(args.output)],check=True)
    manifest_path = args.output.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"source_revision":REVISION,"checkpoint_url":f"https://drive.google.com/file/d/{GOOGLE_DRIVE_ID}/view",
                     "source_license_url":f"{BASE}/LICENSE","source_license_sha256":SOURCE_HASHES["LICENSE"],
                     "source_readme_sha256":SOURCE_HASHES["README.md"]})
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    args.output.with_name("ADAFACE-LICENSE.txt").write_bytes((source/"LICENSE").read_bytes())
    print(f"Official AdaFace source revision: {REVISION}; code terms: {BASE}/LICENSE")
    print("AdaFace remains opt-in; configure adaface_enabled and adaface_model_path after reviewing the source's model terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
