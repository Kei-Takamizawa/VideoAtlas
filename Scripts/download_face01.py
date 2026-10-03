"""Download pinned local face models without importing or running the app."""

# ダウンロード先や引数を扱う標準ライブラリだけを使います。
import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

# 研究開発用に確認した FACE01_DEV の版を固定します。
REVISION = "afec7ebac709f14224353e7f8b6539711899b1ff"
BASE = f"https://raw.githubusercontent.com/yKesamaru/FACE01_DEV/{REVISION}"
# モデルごとの出所・容量・SHA-256を固定し、別の重みへのすり替わりを防ぎます。
ASSETS = (
    ("JAPANESE_FACE_V1.onnx", f"{BASE}/face01lib/models/JAPANESE_FACE_V1.onnx", 26083027,
     "e7ca51f4bc85f73ddb830683ac6a09077909fa45a52b2bff41a9c6e8ff267e2f"),
    ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 3758596,
     "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
)


# 途中のダウンロードを完成品に見せず、完全なファイルだけを置き換えます。
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


# macOS・WindowsのどちらでもPythonだけで初回取得を行えます。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "videoatlas" / "resources")
    arguments = parser.parse_args()
    print(f"FACE01 model terms: {BASE}/LICENSE/LICENSE")
    print("These third-party weights are not covered by VideoAtlas's MIT license.")
    download(f"{BASE}/LICENSE/LICENSE", arguments.output_dir / "FACE01-LICENSE.txt")
    for name, url, size, digest in ASSETS:
        download(url, arguments.output_dir / name, size, digest)
        print(f"Saved {name}: {size:,} bytes")
    return 0


# 直接指定されたときだけ取得処理を始めます。
if __name__ == "__main__":
    raise SystemExit(main())
