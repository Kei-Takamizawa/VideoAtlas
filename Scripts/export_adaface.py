"""Export a locally downloaded official AdaFace checkpoint to an audited ONNX artifact.

Requires torch, onnx and onnxruntime in the export environment. Download the
official repository and a checkpoint from https://github.com/mk-minchul/AdaFace.
No checkpoint or pretrained weights are included with VideoAtlas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", choices=("ir_18", "ir_50", "ir_100"), default="ir_50")
    parser.add_argument("--output", type=Path, default=Path("videoatlas/resources/adaface.onnx"))
    args = parser.parse_args()
    source, checkpoint = args.source_dir.resolve(), args.checkpoint.resolve()
    if not (source/"net.py").is_file() or not checkpoint.is_file():
        parser.error("Provide the official AdaFace source directory and a local checkpoint")
    import torch
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(source))
    import net
    model = net.build_model(args.architecture)
    # The official inference.py strips the Lightning model. prefix.
    class CheckpointMetadata:
        """Inert substitute for the old Lightning callback metadata key."""
        pass

    # Official 2022 Lightning checkpoints reference this callback class in
    # training metadata. An inert marker keeps weights-only loading enabled;
    # no Lightning callback implementation or pickle execution is required.
    with torch.serialization.safe_globals([(CheckpointMetadata, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    state = {key[6:]: value for key,value in state.items() if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()

    class EmbeddingsOnly(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, image):
            embeddings, _norms = self.backbone(image)
            return embeddings

    wrapper = EmbeddingsOnly(model).eval()
    sample = torch.linspace(-1,1,3*112*112).reshape(1,3,112,112)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        reference = wrapper(sample).numpy()
        torch.onnx.export(wrapper,sample,str(args.output),input_names=["images"],output_names=["embeddings"],
                          dynamic_axes={"images":{0:"batch"},"embeddings":{0:"batch"}},opset_version=17,dynamo=False)
    session = ort.InferenceSession(str(args.output),providers=["CPUExecutionProvider"])
    actual = session.run(["embeddings"],{"images":sample.numpy()})[0]
    if actual.shape != (1,512) or not np.isfinite(actual).all():
        raise RuntimeError("Export produced invalid embeddings")
    np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=1e-4)
    manifest = {"model_name":"AdaFace","architecture":args.architecture,"checkpoint_sha256":sha256(checkpoint),
                "onnx_sha256":sha256(args.output),"source_net_sha256":sha256(source/"net.py"),
                "preprocessing":"bgr112-mean0.5-std0.5-scrfd-similarity5-v1",
                "source_url":"https://github.com/mk-minchul/AdaFace","export_max_abs_error":float(np.max(np.abs(actual-reference)))}
    args.output.with_suffix(".json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Saved {args.output}; CPU export parity maximum absolute error: {manifest['export_max_abs_error']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
