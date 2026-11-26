import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videoatlas import cli
from videoatlas.embedders import EMBEDDERS, EmbedderSpec, OnnxEmbedder, face_tensor, insightface_normalization, resolve_spec
from videoatlas.photo_evaluation import (calibrate_leave_one_person_out, choose_policy, compare_models, face_labels,
                                         grouping_metrics, verification_report, _curve_metrics, _histogram)
from videoatlas.photo_grouping import (UNGROUPED, GroupingPolicy, agglomerate, cut, group_faces, groups_document,
                                       same_photo_constraints)
from videoatlas.photos import PhotoAnalyzer, RunData, analyze_folder, face_geometry, load_run


def unit(rows):
    rows = np.asarray(rows, np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def make_run(photo_faces, vectors, labels=None, hashes=None, eye=30.0, model="m"):
    """photo_faces: number of usable faces per photo; vectors: one row per face in order."""
    photos, faces, face_photo = [], [], []
    row = 0
    for index, count in enumerate(photo_faces):
        label = labels[index] if labels else None
        path = f"{label or 'x'}/{index}.jpg"
        entries = []
        for position in range(count):
            entry = {"id": f"p{index}f{position}", "index": position, "usable": True, "eye_distance": eye if np.isscalar(eye) else eye[row],
                     "detection_score": .9, "yaw_ratio": 0.0}
            entries.append(entry)
            faces.append({**entry, "photo": path})
            face_photo.append(index)
            row += 1
        photos.append({"path": path, "label": label, "sha256": hashes[index] if hashes else f"h{index}", "faces": entries})
    return RunData(photos, faces, np.asarray(face_photo), {model: unit(vectors)}, {model: f"{model}:key"}, {})


def naive_average_linkage(similarity, blocked, threshold):
    clusters = [{index} for index in range(len(similarity))]
    while True:
        best, pair = -math.inf, None
        for (a, left), (b, right) in itertools.combinations(enumerate(clusters), 2):
            if any(blocked[i, j] for i in left for j in right):
                continue
            score = np.mean([similarity[i, j] for i in left for j in right])
            if score > best:
                best, pair = score, (a, b)
        if pair is None or best < threshold:
            return sorted(sorted(cluster) for cluster in clusters)
        a, b = pair
        clusters[a] |= clusters[b]
        del clusters[b]


def partition(labels):
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return sorted(groups.values())


@pytest.mark.parametrize("seed", range(6))
def test_average_linkage_matches_brute_force_with_constraints(seed):
    generator = np.random.default_rng(seed)
    vectors = unit(generator.normal(size=(14, 4)))
    similarity = vectors @ vectors.T
    blocked = np.zeros((14, 14), bool)
    for i, j in generator.choice(14, size=(4, 2)):
        if i != j:
            blocked[i, j] = blocked[j, i] = True
    merges = agglomerate(similarity, blocked, "average", floor=-.2)
    for threshold in (-.2, 0.0, .2, .4, .6):
        assert partition(cut(merges, 14, threshold)) == naive_average_linkage(similarity, blocked, threshold)


def test_complete_linkage_requires_every_pair_above_threshold():
    generator = np.random.default_rng(3)
    vectors = unit(generator.normal(size=(20, 3)))
    similarity = vectors @ vectors.T
    labels = cut(agglomerate(similarity, None, "complete", .3), 20, .3)
    for group in partition(labels):
        for i, j in itertools.combinations(group, 2):
            assert similarity[i, j] >= .3 - 1e-9


def test_average_linkage_does_not_chain_two_people_through_one_bridge():
    # A~B and B~C are close, A~C is far: single linkage would chain all three.
    similarity = np.asarray([[1, .9, .1], [.9, 1, .8], [.1, .8, 1]])
    labels = cut(agglomerate(similarity, None, "average", .7), 3, .7)
    assert partition(labels) == [[0, 1], [2]]


def test_faces_from_one_photo_never_share_a_group():
    vectors = [[1, 0], [1, .01], [1, .02], [0, 1]]
    run = make_run([2, 1, 1], vectors)
    result = group_faces(run, GroupingPolicy(("m",), merge_threshold=.5, attach_threshold=.5))
    assert result.labels[0] != result.labels[1]
    assert same_photo_constraints(run.face_photo)[0, 1]


def test_user_negative_decisions_are_respected():
    run = make_run([1, 1, 1], [[1, 0], [1, .01], [1, .02]])
    blocked = np.zeros((3, 3), bool)
    blocked[0, 2] = blocked[2, 0] = True
    result = group_faces(run, GroupingPolicy(("m",), merge_threshold=.5, attach_threshold=.5), cannot_link=blocked)
    assert result.labels[0] != result.labels[2] or result.labels[0] == UNGROUPED


def test_attach_needs_threshold_and_margin():
    a = [[1, 0, 0]] * 3
    b = [[0, 1, 0]] * 3
    lonely = [[.7, .7, .14]]  # equally close to both groups
    clear = [[.95, .1, .3]]
    run = make_run([1] * 8, a + b + lonely + clear)
    policy = GroupingPolicy(("m",), merge_threshold=.99, attach_threshold=.8, attach_margin=.05, min_group_size=3)
    result = group_faces(run, policy)
    assert result.labels[6] == UNGROUPED and result.labels[7] == result.labels[0]
    assert result.attached[7] and not result.attached[6]
    review = {item["face"]: item for item in result.review}
    assert [candidate["group"] for candidate in review["p6f0"]["candidates"]][:2] in ([0, 1], [1, 0])


def test_small_faces_are_gated_to_review():
    vectors = [[1, 0]] * 4
    run = make_run([1] * 4, vectors, eye=[40, 40, 40, 8])
    result = group_faces(run, GroupingPolicy(("m",), merge_threshold=.5, attach_threshold=.5, min_eye_distance=12))
    assert result.labels[3] == UNGROUPED and result.gated[3]
    assert {item["face"] for item in result.review} == {"p3f0"}


def test_answer_key_skips_multi_face_photos_and_files_under_two_people():
    run = make_run([1, 2, 1, 1], [[1, 0], [1, 0], [0, 1], [1, 0], [1, 0]], labels=["A", "A", "B", "A"], hashes=["h0", "h1", "dup", "dup"])
    assert list(face_labels(run)) == ["A", None, None, None, None]


def test_grouping_metrics_count_mixing_and_missing():
    # People A (3 photos) and B (2 photos); one B face sits in A's group; one A photo is alone.
    vectors = [[1, 0, 0], [1, 0, 0], [1, 0, .01], [0, 1, 0], [0, 0, 1]]
    run = make_run([1] * 5, vectors, labels=["A", "A", "A", "B", "B"])
    result = group_faces(run, GroupingPolicy(("m",), merge_threshold=.99))
    result.labels[:] = [0, 0, UNGROUPED, 0, UNGROUPED]
    result.group_sizes = [3]
    metrics = grouping_metrics(run, result)
    assert metrics["mixing"]["mixed_faces"] == 1 and metrics["mixing"]["grouped_labelled_faces"] == 3
    assert metrics["mixing"]["mixed_groups"] == 1
    assert metrics["people"]["A"]["found_in_main_group"] == 2 and metrics["people"]["A"]["missing_rate"] == pytest.approx(1 / 3)
    assert metrics["missing"]["photos"] == 5 and metrics["missing"]["found_in_main_group"] == 3


def synthetic_people(people=5, photos=12, spread=.35, seed=0):
    generator = np.random.default_rng(seed)
    centers = unit(generator.normal(size=(people, 32)))
    vectors, labels = [], []
    for person in range(people):
        for _ in range(photos):
            vectors.append(centers[person] + generator.normal(scale=spread / np.sqrt(32), size=32))
            labels.append(chr(65 + person))
    return make_run([1] * len(labels), vectors, labels=labels)


def test_leave_one_person_out_calibration_reports_held_out_errors():
    run = synthetic_people()
    base = GroupingPolicy(("m",), merge_threshold=0.0)
    report = calibrate_leave_one_person_out(run, base, [.3, .5, .7], [None, .6], [0.0], max_mixing_rate=0.0)
    assert len(report["folds"]) == 5
    assert report["held_out"]["mixed_faces"] == 0
    assert report["held_out"]["photos"] == 60 and report["held_out"]["missing_rate"] < .2
    policy, metrics, table = choose_policy(run, base, [.3, .5, .7], [None], [0.0], max_mixing_rate=0.0)
    assert policy is not None and metrics["mixing"]["mixed_faces"] == 0 and len(table) == 3


def test_curve_metrics_and_model_comparison():
    perfect = _curve_metrics(_histogram(np.asarray([.8, .9])), _histogram(np.asarray([.1, .2, .3])))
    assert perfect["auc"] == 1 and perfect["tar_at_fmr_0.001"] == 1
    run = synthetic_people(people=4, photos=6)
    noisy = run.embeddings["m"] + np.random.default_rng(1).normal(scale=.4, size=run.embeddings["m"].shape).astype(np.float32)
    run.embeddings["noisy"] = unit(noisy)
    report = verification_report(run, "m")
    assert report["people"] == 4 and report["positive_pairs"] == 4 * 15 and report["auc"] > .95
    comparison = compare_models(run, ["m", "noisy"], repetitions=50)
    assert comparison["models"]["noisy"]["auc"]["difference_vs_reference"] < 0
    assert comparison["models"]["noisy"]["auc"]["difference_ci95"][0] <= comparison["models"]["noisy"]["auc"]["difference_ci95"][1]


def test_groups_document_lists_each_photo_under_every_group():
    vectors = [[1, 0], [0, 1], [1, .01], [.01, 1]]
    run = make_run([2, 2], vectors)
    result = group_faces(run, GroupingPolicy(("m",), merge_threshold=.9))
    document = groups_document(run, result)
    assert document["group_count"] == 2
    assert all(len(entry["groups"]) == 2 for entry in document["photos"].values())
    json.dumps(document)


def test_policy_round_trip_and_validation():
    policy = GroupingPolicy(("a", "b"), merge_threshold=.4, offsets=(.1, .2), scales=(.1, .1))
    assert GroupingPolicy.from_json(json.loads(json.dumps(policy.to_json()))) == policy
    with pytest.raises(ValueError):
        GroupingPolicy(("a",), merge_threshold=.4, scales=(1.0, 2.0)).validate()


def test_face_geometry_rejects_tiny_and_inverted_faces():
    good = np.asarray([[30, 40], [60, 40], [45, 55], [35, 70], [55, 70]], float)
    assert face_geometry(good)[3] is None and face_geometry(good)[0] == pytest.approx(30)
    assert face_geometry(good / 10)[3] == "face_too_small"
    assert face_geometry(good[[1, 0, 2, 3, 4]])[3] == "landmark_geometry_invalid"


def test_face_tensor_follows_channel_order_and_normalization():
    crop = np.zeros((112, 112, 3), np.uint8)
    crop[..., 0] = 255  # blue in BGR
    rgb = EmbedderSpec("x", "x.onnx", 112, "arcface112", "rgb", (127.5,) * 3, (127.5,) * 3, "", "")
    bgr = EmbedderSpec("x", "x.onnx", 112, "arcface112", "bgr", (127.5,) * 3, (127.5,) * 3, "", "")
    mean, std = np.full(3, 127.5, np.float32), np.full(3, 127.5, np.float32)
    assert face_tensor(crop, rgb, mean, std)[2].mean() == pytest.approx(1) and face_tensor(crop, rgb, mean, std)[0].mean() == pytest.approx(-1)
    assert face_tensor(crop, bgr, mean, std)[0].mean() == pytest.approx(1)
    assert resolve_spec("w600k-r50+flip").flip and not EMBEDDERS["w600k-r50"].flip
    with pytest.raises(ValueError):
        resolve_spec("unknown")


def _tiny_embedding_model(path: Path, normalize_inside: bool = False) -> None:
    import onnx
    from onnx import TensorProto, helper
    weights = np.random.default_rng(0).normal(size=(3 * 112 * 112, 8)).astype(np.float32)
    nodes, inputs = [], "input"
    initializers = [helper.make_tensor("w", TensorProto.FLOAT, weights.shape, weights.ravel()),
                    helper.make_tensor("shape", TensorProto.INT64, [2], [-1, 3 * 112 * 112])]
    if normalize_inside:
        initializers += [helper.make_tensor("m", TensorProto.FLOAT, [], [127.5]), helper.make_tensor("s", TensorProto.FLOAT, [], [.0078125])]
        nodes += [helper.make_node("Sub", ["input", "m"], ["sub"], name="Sub_0"), helper.make_node("Mul", ["sub", "s"], ["mul"], name="Mul_1")]
        inputs = "mul"
    nodes += [helper.make_node("Reshape", [inputs, "shape"], ["flat"], name="reshape"), helper.make_node("MatMul", ["flat", "w"], ["output"], name="matmul")]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 112, 112])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 8])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_onnx_embedder_keys_and_normalizes(tmp_path):
    _tiny_embedding_model(tmp_path / "plain.onnx")
    _tiny_embedding_model(tmp_path / "inside.onnx", normalize_inside=True)
    assert insightface_normalization(tmp_path / "plain.onnx") == (127.5, 127.5)
    assert insightface_normalization(tmp_path / "inside.onnx") == (0.0, 1.0)
    options = SimpleNamespace(acceleration="cpu", threads=1, gpu_device_id=0)
    spec = EmbedderSpec("tiny", "plain.onnx", 112, "arcface112", "rgb", None, None, "test", "test")
    embedder = OnnxEmbedder(spec, options, tmp_path)
    crops = [np.random.default_rng(index).integers(0, 255, (112, 112, 3), dtype=np.uint8) for index in range(3)]
    vectors = embedder.embed(crops, batch_size=2)
    assert vectors.shape == (3, 8) and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert embedder.key.startswith("tiny:") and embedder.key.endswith(":cpu-fp32") and "mean127.5-std127.5" in embedder.key
    flipped = OnnxEmbedder(EmbedderSpec(**{**spec.__dict__, "flip": True}), options, tmp_path)
    assert flipped.key != embedder.key and flipped.embed(crops).shape == (3, 8)
    with pytest.raises(ValueError):
        OnnxEmbedder(EmbedderSpec(**{**spec.__dict__, "sha256": "0" * 64}), options, tmp_path)


class FakeDetector:
    runner = SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))

    def detect(self, image):
        height, width = image.shape[:2]
        points = np.asarray([[.35, .4], [.65, .4], [.5, .55], [.4, .7], [.6, .7]]) * [width, height]
        return [SimpleNamespace(box=(int(width * .2), int(height * .2), int(width * .8), int(height * .9)), score=.95, landmarks=points)]


class FakeEmbedder:
    def __init__(self):
        self.spec = SimpleNamespace(name="fake", flip=False, input_size=112, source="test", license="test", preprocessing="fake-v1")
        self.key, self.sha256, self.path, self.providers = "fake:key", "0" * 64, Path("fake.onnx"), ("CPUExecutionProvider",)

    def align(self, image, landmarks):
        return cv2.resize(image, (112, 112)), 0.0, 1.0

    def embed(self, crops):
        return unit([[crop[..., 2].mean() + 1, crop[..., 1].mean() + 1, 1.0] for crop in crops]) if crops else np.zeros((0, 0), np.float32)


def test_analyze_folder_then_calibrate_and_group_from_the_command_line(tmp_path):
    root = tmp_path / "photos"
    colours = {"A": (0, 0, 250), "B": (0, 250, 0)}
    for person, colour in colours.items():
        (root / person).mkdir(parents=True)
        for index in range(4):
            image = np.full((160, 120, 3), colour, np.uint8)
            image[:4, :4] = index
            assert cv2.imwrite(str(root / person / f"{index}.png"), image)
    (root / "A" / "broken.jpg").write_bytes(b"not an image")
    run_folder = tmp_path / "run"
    report = analyze_folder(PhotoAnalyzer(FakeDetector(), [FakeEmbedder()]), root, run_folder)
    assert report["image_count"] == 9 and report["labels"]["A"]["unreadable"] == 1
    with pytest.raises(FileExistsError):
        analyze_folder(PhotoAnalyzer(FakeDetector(), [FakeEmbedder()]), root, run_folder)
    run = load_run(run_folder)
    assert len(run.faces) == 8 and run.embeddings["fake"].shape == (8, 3)
    assert cli.main(["photos", "verify", str(run_folder), "--models", "fake", "--output", str(tmp_path / "verify.json"), "--bootstrap", "20"]) == 0
    assert cli.main(["photos", "calibrate", str(run_folder), "--models", "fake", "--output", str(tmp_path / "calibration.json"),
                     "--policy-output", str(tmp_path / "policy.json"), "--merge-grid", "0.5:0.9:0.2", "--attach-grid", "none",
                     "--eye-grid", "0", "--max-mixing", "0"]) == 0
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert calibration["held_out"]["held_out"]["mixed_faces"] == 0
    assert cli.main(["photos", "group", str(run_folder), "--policy", str(tmp_path / "policy.json"), "--output", str(tmp_path / "groups.json")]) == 0
    groups = json.loads((tmp_path / "groups.json").read_text())
    assert groups["group_count"] == 2 and groups["evaluation"]["mixing"]["mixed_faces"] == 0
    assert groups["photos"]["A/broken.jpg"] == {"groups": [], "review_faces": [], "faces": 0}
