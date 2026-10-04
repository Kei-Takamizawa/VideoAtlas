"""Download pinned local face models without importing or running the app."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import zipfile
from urllib.request import urlopen

REVISION = "afec7ebac709f14224353e7f8b6539711899b1ff"
BASE = f"https://raw.githubusercontent.com/yKesamaru/FACE01_DEV/{REVISION}"
ASSETS = (
    ("JAPANESE_FACE_V1.onnx", f"{BASE}/face01lib/models/JAPANESE_FACE_V1.onnx", 26083027,
     "e7ca51f4bc85f73ddb830683ac6a09077909fa45a52b2bff41a9c6e8ff267e2f"),
)
SCRFD_RELEASE = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
SCRFD_ARCHIVE_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
SCRFD_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"


def download_scrfd(destination: Path) -> None:
    """Extract only the SCRFD detector from the official, pinned model release."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
        archive = Path(folder) / "buffalo_l.zip"
        download(SCRFD_RELEASE, archive, 288621354, SCRFD_ARCHIVE_SHA256)
        with zipfile.ZipFile(archive) as bundle:
            info = bundle.getinfo("det_10g.onnx")
            if info.file_size != 16923827:
                raise RuntimeError("SCRFD artifact size mismatch")
            data = bundle.read(info)
        if hashlib.sha256(data).hexdigest() != SCRFD_SHA256:
            raise RuntimeError("SCRFD SHA-256 mismatch")
        temporary = Path(folder) / "scrfd.onnx"
        temporary.write_bytes(data)
        temporary.replace(destination)


def download(url: str, destination: Path, size: int | None = None, digest: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            checksum = hashlib.sha256()
            written = 0
            with urlopen(url, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    checksum.update(block)
                    written += len(block)
                    if size is not None and written > size:
                        raise RuntimeError(f"Unexpected model size: {destination.name}")
            if size is not None and written != size:
                raise RuntimeError(f"Incomplete model download: {destination.name}")
            if digest is not None and checksum.hexdigest() != digest:
                raise RuntimeError(f"SHA-256 mismatch: {destination.name}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "videoatlas" / "resources")
    arguments = parser.parse_args()
    print(f"FACE01 model terms: {BASE}/LICENSE/LICENSE")
    print("These third-party weights are not covered by VideoAtlas's MIT license.")
    print("InsightFace pretrained weights are restricted to non-commercial research: https://github.com/deepinsight/insightface#license")
    download(f"{BASE}/LICENSE/LICENSE", arguments.output_dir / "FACE01-LICENSE.txt")
    for name, url, size, digest in ASSETS:
        download(url, arguments.output_dir / name, size, digest)
        print(f"Saved {name}: {size:,} bytes")
    download_scrfd(arguments.output_dir / "scrfd_10g_kps.onnx")
    print("Saved SCRFD-10G with five landmarks: 16,923,827 bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
