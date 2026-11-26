from collections import OrderedDict
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videoatlas import analyzer as analyzer_module
from videoatlas import cli
from videoatlas.analyzer import FaceSample, VideoAnalyzer, _SessionState
from videoatlas.images import analyze_images, folder_label, iter_image_files, pair_report, primary_face, ImageResult
from videoatlas.model_adapters import OnnxRunner
from videoatlas.recognition import RecognitionOptions


def face(box, embedding=None, reason=None):
    return FaceSample(0.0, box, b"", embedding, "fake" if embedding is not None else None, .9 if embedding is not None else 0., reason)


def test_image_files_are_filtered_sorted_and_labelled(tmp_path):
    for relative in ("B/2.JPG", "A/1.webp", "A/.hidden/3.jpg", ".cache/4.png", "top.png", "A/notes.txt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    files = iter_image_files(tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["A/1.webp", "B/2.JPG", "top.png"]
    assert [folder_label(path, tmp_path) for path in files] == ["A", "B", None]
    with pytest.raises(NotADirectoryError):
        iter_image_files(tmp_path / "missing")


def test_analyze_images_reports_unreadable_files_and_stops_cleanly(tmp_path):
    (tmp_path / "A").mkdir()
    assert cv2.imwrite(str(tmp_path / "A" / "ok.png"), np.full((20, 30, 3), 90, np.uint8))
    (tmp_path / "A" / "broken.jpg").write_bytes(b"not an image")
    seen = []
    fake = SimpleNamespace(analyze_image=lambda image, stop: seen.append(image.shape) or [face((0, 0, .5, .5), [1., 0.])])
    results = list(analyze_images(fake, iter_image_files(tmp_path), tmp_path))
    assert [(result.path.name, result.error, len(result.faces)) for result in results] == [("broken.jpg", "unreadable_image", 0), ("ok.png", None, 1)]
    assert seen == [(20, 30, 3)] and results[1].width == 30 and results[1].height == 20
    stopped = SimpleNamespace(analyze_image=lambda image, stop: None)
    assert list(analyze_images(stopped, [tmp_path / "A" / "ok.png"], tmp_path)) == []


def test_primary_face_is_the_largest_accepted_face():
    result = ImageResult(Path("a.png"), "A", faces=[face((0, 0, .9, .9), None, "pose_yaw_out_of_range"),
                                                    face((0, 0, .2, .2), [1., 0.]), face((0, 0, .3, .4), [0., 1.])])
    assert primary_face(result).embedding == [0., 1.]
    assert primary_face(ImageResult(Path("b.png"), "A", faces=[face((0, 0, .5, .5), None, "blur_too_high")])) is None


def test_pair_report_counts_thresholds_auc_and_zero_false_positive_recall():
    labels = ["a", "a", "b", "b"]
    vectors = np.asarray([[1, 0], [.8, .6], [0, 1], [.6, .8]], np.float32)
    report = pair_report(labels, vectors, thresholds=[.7])
    # positives: .8, .8; negatives: 0, .6, .6, .96
    assert (report["pair_count"], report["positive_count"], report["negative_count"]) == (6, 2, 4)
    row = report["thresholds"][0]
    assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (2, 1, 3, 0)
    assert row["precision"] == pytest.approx(2 / 3) and row["false_positive_rate"] == pytest.approx(.25)
    assert report["auc"] == pytest.approx(.75)
    assert report["max_negative_score"] == pytest.approx(.96)
    assert report["recall_with_zero_false_positives"] == 0
    assert pair_report(["a"], np.ones((1, 2)))["pair_count"] == 0
    with pytest.raises(ValueError):
        pair_report(["a", "b"], np.zeros((2, 2)))


class FakeAnalyzer:
    def __init__(self, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        self.providers, self.closed = providers, False
        self.embedding_model, self.runtime_description = "face01:fake", "fake runtime"

    def execution_providers(self):
        return {"face01": self.providers, "scrfd": self.providers}

    def analyze_image(self, image, stop=None):
        value = float(image[0, 0, 0])
        return [face((0, 0, .5, .5), [1., value / 255.])]

    def close(self):
        self.closed = True


def test_cli_doctor_fails_when_gpu_is_required_but_models_run_on_cpu(monkeypatch, capsys):
    fake = FakeAnalyzer(("CPUExecutionProvider",))
    monkeypatch.setattr(cli, "open_analyzer", lambda options: fake)
    assert cli.main(["doctor", "--device", "cuda"]) == 1
    assert "GPU was requested" in capsys.readouterr().err and fake.closed
    assert cli.main(["doctor", "--device", "auto"]) == 0


def test_cli_images_writes_outputs_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    root = tmp_path / "photos"
    for label, values in {"A": (10, 12, 14), "B": (240, 238)}.items():
        (root / label).mkdir(parents=True)
        for index, value in enumerate(values):
            assert cv2.imwrite(str(root / label / f"{index}.png"), np.full((8, 8, 3), value, np.uint8))
    options_seen = []
    monkeypatch.setattr(cli, "open_analyzer", lambda options: options_seen.append(options) or FakeAnalyzer())
    output = tmp_path / "run"
    assert cli.main(["images", str(root), "--output", str(output), "--device", "cuda"]) == 0
    assert options_seen[0].acceleration == "cuda" and options_seen[0].cache_dir == output / "aligned-cache"
    lines = [json.loads(line) for line in (output / "faces.jsonl").read_text().splitlines()]
    assert [line["path"] for line in lines] == ["A/0.png", "A/1.png", "A/2.png", "B/0.png", "B/1.png"]
    assert all(line["primary_face"] == 0 and line["faces"][0]["accepted"] for line in lines)
    report = json.loads((output / "report.json").read_text())
    assert report["image_count"] == 5 and report["pairs"]["positive_count"] == 4 and report["pairs"]["negative_count"] == 6
    assert report["labels"]["A"]["images_with_accepted_face"] == 3
    saved = np.load(output / "embeddings.npz")
    assert saved["vectors"].shape == (5, 2) and list(saved["labels"]) == ["A", "A", "A", "B", "B"]
    capsys.readouterr()
    assert cli.main(["images", str(root), "--output", str(output)]) == 1
    assert "already exists" in capsys.readouterr().err


def fake_onnxruntime(available, registered):
    class Session:
        def __init__(self, path, sess_options=None, providers=None):
            self.providers = list(registered)
        def get_providers(self):
            return self.providers
        def disable_fallback(self):
            pass
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 8, 8], type="tensor(float)")]
        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=[1, 2])]
    return SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session,
                           get_available_providers=lambda: list(available), preload_dlls=lambda: None)


@pytest.mark.parametrize("available,registered,message", [
    (["CPUExecutionProvider"], ["CPUExecutionProvider"], "no CUDA provider"),
    (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"], "registered only the CPU provider"),
])
def test_runner_never_uses_cpu_when_gpu_is_required(tmp_path, monkeypatch, available, registered, message):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(available, registered))
    with pytest.raises(RuntimeError, match=message):
        OnnxRunner(model, replace(RecognitionOptions(), acceleration="cuda"))


def test_runner_auto_fallback_to_cpu_is_logged(tmp_path, monkeypatch, caplog):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(["CPUExecutionProvider"], ["CPUExecutionProvider"]))
    with caplog.at_level(logging.WARNING):
        runner = OnnxRunner(model, RecognitionOptions())
    assert runner.session.get_providers() == ["CPUExecutionProvider"]
    assert "CUDA is unavailable; model.onnx runs on the CPU" in caplog.text


def test_runner_gpu_execution_failure_is_fatal_when_gpu_is_required():
    runner = OnnxRunner.__new__(OnnxRunner)
    runner.session = SimpleNamespace(run=lambda *args: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
                                     get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    runner.input, runner.outputs, runner.lock, runner.require_gpu = SimpleNamespace(name="i"), [SimpleNamespace(name="o")], threading.RLock(), True
    runner.cpu_session = lambda: pytest.fail("must not retry on the CPU")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        runner.run(np.zeros(1))


def face01_cpu_ort(caplog_marker=None):
    model_input = SimpleNamespace(name="images", shape=[1, 3, 224, 224], type="tensor(float)")
    model_output = SimpleNamespace(name="embedding", shape=[1, 512], type="tensor(float)")
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def get_inputs(self):
            return [model_input]
        def get_outputs(self):
            return [model_output]
        def disable_fallback(self):
            pass
        def get_providers(self):
            return ["CPUExecutionProvider"]
        def get_provider_options(self):
            return {}
    return SimpleNamespace(SessionOptions=lambda: SimpleNamespace(), InferenceSession=Session, __version__="test", __file__="/nonexistent/onnxruntime/__init__.py",
                           get_available_providers=lambda: ["CPUExecutionProvider"], preload_dlls=lambda: None,
                           GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99))


@pytest.mark.parametrize("acceleration", ["cuda", "auto"])
def test_face01_session_requires_gpu_or_logs_cpu_fallback(monkeypatch, caplog, acceleration):
    monkeypatch.setitem(sys.modules, "onnxruntime", face01_cpu_ort())
    monkeypatch.setattr(analyzer_module, "_gpu_identity", lambda device_id: None)
    monkeypatch.setattr(analyzer_module, "_SESSION_CACHE", OrderedDict())
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.options = replace(RecognitionOptions(), acceleration=acceleration)
    analyzer.model_path, analyzer.model_sha256, analyzer._dll_libraries = Path("fake.onnx"), "artifact", []
    if acceleration == "cuda":
        with pytest.raises(RuntimeError, match="could not start on the GPU"):
            analyzer._load_session()
    else:
        with caplog.at_level(logging.WARNING):
            state, report = analyzer._load_session()
        assert state.providers == ("CPUExecutionProvider",)
        assert "FACE01 runs on the CPU although acceleration is auto" in caplog.text


def test_face01_gpu_execution_failure_is_fatal_when_gpu_is_required():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.options, analyzer._closed = replace(RecognitionOptions(), acceleration="cuda"), False
    model_input = SimpleNamespace(name="images", shape=[1, 3, 224, 224], type="tensor(float)")
    model_output = SimpleNamespace(name="embedding", shape=[1, 512], type="tensor(float)")
    def allocate(*args):
        raise RuntimeError("CUDA out of memory")
    analyzer._ort = SimpleNamespace(OrtValue=SimpleNamespace(ortvalue_from_shape_and_type=allocate))
    analyzer._state = _SessionState(None, model_input, model_output, False, ("CUDAExecutionProvider", "CPUExecutionProvider"), "gpu", threading.Lock())
    analyzer._input, analyzer._output, analyzer._dynamic_batch, analyzer._gpu_buffers = model_input, model_output, False, OrderedDict()
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        analyzer._infer_batch([np.zeros((3, 224, 224), np.float32)])
    assert analyzer._state.providers[0] == "CUDAExecutionProvider"


def test_linux_gpu_library_identity_reads_mapped_cuda_libraries(tmp_path, monkeypatch):
    for name in ("libcudart.so.12", "libcudnn.so.9", "libcudnn_ops.so.9", "libfoo.so"):
        (tmp_path / name).write_bytes(name.encode())
    maps = tmp_path / "maps"
    maps.write_text(f"7f00-7f01 r-xp 00000000 00:00 1 {tmp_path / 'libcudart.so.12'}\n"
                    f"7f01-7f02 r-xp 00000000 00:00 2 {tmp_path / 'libfoo.so'}\n"
                    "7f02-7f03 rw-p 00000000 00:00 0 [heap]\n")
    monkeypatch.setattr(analyzer_module, "_PROC_MAPS", maps)
    names = [Path(path).name for path, digest in analyzer_module._loaded_gpu_libraries_linux()]
    assert names == ["libcudart.so.12", "libcudnn.so.9", "libcudnn_ops.so.9"]


def test_analyze_image_validates_input_and_releases_the_lifecycle_flag():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._closed, analyzer._active, analyzer._lifecycle_lock = False, False, threading.RLock()
    calls = []
    analyzer._process_frame = lambda frame, second, stop, poster: (calls.append((frame.shape, second, poster)) or [face((0, 0, 1, 1), [1.])], None)
    assert len(analyzer.analyze_image(np.zeros((4, 5, 3), np.uint8))) == 1
    assert calls == [((4, 5, 3), 0.0, False)] and not analyzer._active
    for bad in (np.zeros((4, 5), np.uint8), np.zeros((4, 5, 3), np.float32), "image"):
        with pytest.raises(ValueError):
            analyzer.analyze_image(bad)
    analyzer._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        analyzer.analyze_image(np.zeros((4, 5, 3), np.uint8))
