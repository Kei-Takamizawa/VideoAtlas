import itertools
import json
import math
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videoatlas import cli
from videoatlas.embedders import EMBEDDERS, EmbedderSpec, OnnxEmbedder, face_tensor, insightface_normalization, resolve_spec
from videoatlas.photo_evaluation import (SearchGrid, add_fused_embedding, calibrate_leave_one_person_out, choose_policy,
                                         compare_models, face_labels, grouping_metrics, simulate_review, verification_report,
                                         _curve_metrics, _histogram, photo_labels)
from videoatlas.photo_grouping import (UNGROUPED, GroupingPolicy, agglomerate, apply_cluster_merges, cut, group_faces,
                                       groups_document, merge_clusters, same_photo_constraints)
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
    report = calibrate_leave_one_person_out(run, base, SearchGrid((.3, .5, .7), (None, .6)), [0.0, 0.01])
    assert set(report["targets"]) == {"0.0", "0.01"}
    strict = report["targets"]["0.0"]
    assert len(strict["folds"]) == 5 and strict["held_out"]["folds_without_setting"] == 0
    assert strict["held_out"]["mixed_faces"] == 0
    assert strict["held_out"]["photos"] == 60 and strict["held_out"]["missing_rate"] < .2
    policy, metrics, table = choose_policy(run, base, SearchGrid((.3, .5, .7)), 0.0)
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
    with pytest.raises(ValueError):
        analyze_folder(PhotoAnalyzer(FakeDetector(), [FakeEmbedder()]), root, root / "run")
    report = analyze_folder(PhotoAnalyzer(FakeDetector(), [FakeEmbedder()]), root, run_folder)
    assert report["image_count"] == 9 and report["labels"]["A"]["unreadable"] == 1
    with pytest.raises(FileExistsError):
        analyze_folder(PhotoAnalyzer(FakeDetector(), [FakeEmbedder()]), root, run_folder)
    run = load_run(run_folder)
    assert len(run.faces) == 8 and run.embeddings["fake"].shape == (8, 3)
    assert cli.main(["photos", "verify", str(run_folder), "--models", "fake", "--output", str(tmp_path / "verify.json"), "--bootstrap", "20"]) == 0
    assert cli.main(["photos", "calibrate", str(run_folder), "--models", "fake", "--output", str(tmp_path / "calibration.json"),
                     "--policy-output", str(tmp_path / "policy.json"), "--merge-grid", "0.5:0.9:0.2", "--attach-grid", "none",
                     "--eye-grid", "0", "--cluster-grid", "none,0.8", "--targets", "0", "--policy-target", "0",
                     "--suggest-threshold", "0.5"]) == 0
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert calibration["held_out"]["targets"]["0.0"]["held_out"]["mixed_faces"] == 0
    assert calibration["grid"]["settings"] == 6
    assert json.loads((tmp_path / "policy.json").read_text())["calibration"]["max_mixing_rate"] == 0
    assert cli.main(["photos", "verify", str(run_folder), "--models", "fake", "--fuse", "fake,fake", "--output", str(tmp_path / "verify2.json"),
                     "--bootstrap", "5"]) == 0
    assert "fake,fake" in json.loads((tmp_path / "verify2.json").read_text())["models"]
    assert cli.main(["photos", "group", str(run_folder), "--policy", str(tmp_path / "policy.json"), "--output", str(tmp_path / "groups.json"),
                     "--people", "B"]) == 0
    groups = json.loads((tmp_path / "groups.json").read_text())
    assert groups["group_count"] == 2 and groups["evaluation"]["mixing"]["mixed_faces"] == 0
    assert groups["evaluation_people"]["focus"] == ["B"] and groups["evaluation_people"]["missing"]["photos"] == 4
    assert cli.main(["photos", "calibrate", str(run_folder), "--models", "fake", "--output", str(tmp_path / "calibration-b.json"),
                     "--merge-grid", "0.5:0.9:0.2", "--attach-grid", "none", "--eye-grid", "0", "--targets", "0", "--policy-target", "0",
                     "--exclude-people", "B"]) == 0
    calibration_b = json.loads((tmp_path / "calibration-b.json").read_text())
    assert calibration_b["excluded_people"] == ["B"] and list(calibration_b["held_out"]["targets"]["0.0"]["folds"][0].values())[0] == "A"
    assert len(calibration_b["held_out"]["targets"]["0.0"]["folds"]) == 1
    assert cli.main(["photos", "calibrate", str(run_folder), "--models", "fake", "--output", str(tmp_path / "calibration-c.json"),
                     "--policy-output", str(tmp_path / "policy-c.json"), "--merge-grid", "0.5:0.9:0.2", "--attach-grid", "none",
                     "--eye-grid", "0", "--targets", "0", "--policy-target", "0", "--exclude-people", "B", "--skip-held-out"]) == 0
    assert json.loads((tmp_path / "calibration-c.json").read_text())["held_out"] is None
    assert json.loads((tmp_path / "policy-c.json").read_text())["calibration"]["excluded_people"] == ["B"]
    assert cli.main(["photos", "strangers", str(run_folder), "--policy", str(tmp_path / "policy.json"), "--people", "B", "--keep", "1",
                     "--repeats", "2", "--output", str(tmp_path / "strangers.json")]) == 0
    strangers = json.loads((tmp_path / "strangers.json").read_text())
    assert strangers["people"] == ["B"] and strangers["keep"]["1"]["kept_faces"] == 2
    assert groups["photos"]["A/broken.jpg"] == {"groups": [], "review_faces": [], "faces": 0}


def test_two_faces_from_one_photo_cannot_both_attach_to_one_group():
    group = [[1, 0, 0]] * 3
    twins = [[.97, .1, 0], [.95, .2, 0]]
    run = make_run([1, 1, 1, 2], group + twins)
    result = group_faces(run, GroupingPolicy(("m",), merge_threshold=.999, attach_threshold=.8, min_group_size=3))
    assert sorted(result.labels[3:].tolist()) == [UNGROUPED, 0]
    assert result.labels[3] == 0  # the closer face wins


def test_identification_counts_unknown_people_as_wrong_when_accepted():
    from videoatlas.photo_evaluation import identification_report
    run = synthetic_people(people=4, photos=12, spread=.3)
    report = identification_report(run, "m", enroll_sizes=(3,), repeats=2, thresholds=[-1.0, .5, .99])
    table = report["enroll_sizes"]["3"]["table"]
    loose, strict = table[0], table[-1]
    assert loose["unknown_accepted"] > 0 and loose["wrong_rate"] > .2
    assert strict["unknown_accepted"] == 0 and strict["known_wrong"] == 0
    assert report["enroll_sizes"]["3"]["best_within_limit"]["wrong_rate"] <= .001


def two_looks_and_a_stranger(seed=0):
    """Person A appears in two looks that average linkage keeps apart; B is another person."""
    generator = np.random.default_rng(seed)
    dims = 64
    base_a = generator.normal(size=dims)
    shift = generator.normal(size=dims)
    look1 = base_a + 0.5 * shift
    look2 = base_a - 0.5 * shift
    other = generator.normal(size=dims)
    vectors, labels = [], []
    for center, label, count in ((look1, "A", 8), (look2, "A", 8), (other, "B", 8)):
        for _ in range(count):
            vectors.append(center + generator.normal(scale=.45, size=dims))
            labels.append(label)
    return make_run([1] * len(labels), vectors, labels=labels)


def test_cluster_merge_joins_two_looks_of_one_person_but_not_a_stranger():
    run = two_looks_and_a_stranger()
    similarity = run.embeddings["m"] @ run.embeddings["m"].T
    within_look = similarity[:8, :8][np.triu_indices(8, 1)].mean()
    across_looks = similarity[:8, 8:16].mean()
    to_stranger = similarity[:16, 16:].mean()
    assert within_look > across_looks > to_stranger
    merge = float((within_look + across_looks) / 2)
    first = group_faces(run, GroupingPolicy(("m",), merge_threshold=merge))
    assert first.labels[0] != first.labels[8]  # the looks stay apart in the first step
    score = across_looks / within_look
    second = group_faces(run, GroupingPolicy(("m",), merge_threshold=merge, cluster_merge_threshold=float(score) - .05))
    assert len(set(second.labels[:16].tolist())) == 1
    assert second.labels[16] != second.labels[0]
    assert grouping_metrics(run, second)["mixing"]["mixed_faces"] == 0


def test_cluster_merge_respects_cannot_link_and_prefix_cut():
    run = two_looks_and_a_stranger()
    similarity = (run.embeddings["m"] @ run.embeddings["m"].T).astype(np.float32)
    labels = np.asarray([0] * 8 + [8] * 8 + [16] * 8)
    blocked = np.zeros_like(similarity, bool)
    blocked[0, 8] = blocked[8, 0] = True
    cluster_of, merges = merge_clusters(similarity, blocked, labels, 3, -1.0)
    joined = apply_cluster_merges(cluster_of, merges, -1.0)
    assert joined[0] != joined[8]  # a user decision between the looks blocks the merge
    cluster_of, merges = merge_clusters(similarity, np.zeros_like(blocked), labels, 3, -1.0)
    assert [merge[:2] for merge in merges][0] == (0, 1)
    assert len(set(apply_cluster_merges(cluster_of, merges, merges[0][2]).tolist())) == 2
    assert len(set(apply_cluster_merges(cluster_of, merges, None).tolist())) == 3
    small = merge_clusters(similarity, np.zeros_like(blocked), labels, 9, -1.0)[1]
    assert small == []  # clusters below the minimum size never merge


def test_merge_suggestions_and_review_simulation():
    run = two_looks_and_a_stranger()
    similarity = run.embeddings["m"] @ run.embeddings["m"].T
    within_look = similarity[:8, :8][np.triu_indices(8, 1)].mean()
    across_looks = similarity[:8, 8:16].mean()
    policy = GroupingPolicy(("m",), merge_threshold=float((within_look + across_looks) / 2), suggest_threshold=-1.0)
    result = group_faces(run, policy)
    assert len(result.suggestions) == 3
    first = result.suggestions[0]["groups"]
    assert {result.labels[0], result.labels[8]} == set(first)
    document = groups_document(run, result)
    assert len(document["merge_suggestions"]) == 3 and document["merge_suggestions"][0]["groups"][0].startswith("g")
    labels = face_labels(run)
    rows = simulate_review(run, result, labels, photo_labels(run), {"A"}, budgets=(0, 1, 3))
    assert rows[0]["missing_rate"] == .5 and rows[1]["missing_rate"] == 0
    assert rows[1]["accepted"] == 1 and rows[2]["rejected"] == 2
    metrics = grouping_metrics(run, result, review=True)
    assert metrics["merge_suggestions"] == 3 and metrics["review_simulation"][0]["budget"] == 0


def test_fused_embedding_orders_pairs_like_fused_similarity():
    from videoatlas.photo_grouping import fused_similarity
    from videoatlas.photo_evaluation import impostor_normalization
    run = synthetic_people(people=4, photos=6)
    noisy = run.embeddings["m"] + np.random.default_rng(2).normal(scale=.5, size=run.embeddings["m"].shape).astype(np.float32)
    run.embeddings["n"] = unit(noisy)
    run.keys["n"] = "n:key"
    name = add_fused_embedding(run, ["m", "n"])
    combined = run.embeddings[name] @ run.embeddings[name].T
    offsets, scales = impostor_normalization(run, ["m", "n"])
    fused = fused_similarity(run.embeddings, GroupingPolicy(("m", "n"), 0.0, offsets=offsets, scales=scales))
    upper = np.triu_indices(len(combined), 1)
    slope, intercept = np.polyfit(combined[upper], fused[upper], 1)
    assert slope > 0 and np.allclose(combined[upper] * slope + intercept, fused[upper], atol=1e-4)
    assert np.allclose(np.diag(combined), 1, atol=1e-5)


def test_cluster_sums_match_brute_force():
    from videoatlas.photo_grouping import ClusterSums
    generator = np.random.default_rng(5)
    similarity = generator.normal(size=(30, 30)).astype(np.float32)
    similarity = (similarity + similarity.T) / 2
    labels = generator.integers(0, 5, size=30) * 7
    blocked = np.zeros((30, 30), bool)
    blocked[2, 9] = blocked[9, 2] = True
    sums = ClusterSums(similarity, blocked, labels)
    for a, left in enumerate(sums.values):
        for b, right in enumerate(sums.values):
            block = similarity[np.ix_(labels == left, labels == right)]
            assert math.isclose(sums.cross[a, b], float(block.astype(np.float64).sum()), rel_tol=1e-5, abs_tol=1e-4)
    assert sums.blocked[list(sums.values).index(labels[2]), list(sums.values).index(labels[9])]
    assert np.allclose(sums.diagonal, [np.diag(similarity)[labels == value].sum() for value in sums.values], atol=1e-5)


def reference_merge_clusters(similarity, constraints, labels, min_size, floor, cap=None):
    """Recompute every cluster pair score after each merge (slow, obviously correct)."""
    from videoatlas.photo_grouping import ClusterSums
    _, cluster_of, counts = np.unique(labels, return_inverse=True, return_counts=True)
    big = np.flatnonzero(counts >= min_size)
    items = np.flatnonzero(np.isin(cluster_of, big))
    sums = ClusterSums(similarity[np.ix_(items, items)], constraints[np.ix_(items, items)], cluster_of[items])
    count, active, merges = len(sums.values), np.ones(len(sums.values), bool), []
    while count > 1:
        scores = sums.scores(min_size)
        scores[~active] = -math.inf
        scores[:, ~active] = -math.inf
        if cap is not None:
            scores[np.minimum.outer(sums.sizes, sums.sizes) > cap] = -math.inf
        left, right = divmod(int(np.argmax(scores)), count)
        if not scores[left, right] >= floor:
            break
        keep, drop = min(left, right), max(left, right)
        merges.append((int(sums.values[keep]), int(sums.values[drop]), float(scores[left, right])))
        sums.cross[keep] += sums.cross[drop]
        sums.cross[:, keep] += sums.cross[:, drop]
        sums.diagonal[keep] += sums.diagonal[drop]
        sums.sizes[keep] += sums.sizes[drop]
        sums.blocked[keep] |= sums.blocked[drop]
        sums.blocked[:, keep] = sums.blocked[keep]
        active[drop] = False
        sums.sizes[drop] = 0
        sums.cross[drop] = 0
        sums.cross[:, drop] = 0
    return merges


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_cluster_merge_matches_full_recompute(seed):
    generator = np.random.default_rng(seed)
    vectors = unit(generator.normal(size=(120, 16)) + np.repeat(generator.normal(size=(6, 16)) * 1.5, 20, axis=0))
    similarity = (vectors @ vectors.T).astype(np.float32)
    labels = generator.integers(0, 40, size=120)
    blocked = np.zeros((120, 120), bool)
    for a, b in generator.integers(0, 120, size=(15, 2)):
        if a != b:
            blocked[a, b] = blocked[b, a] = True
    for min_size, cap in ((2, None), (3, None), (2, 6)):
        _, merges = merge_clusters(similarity, blocked, labels, min_size, -0.5, cap)
        expected = reference_merge_clusters(similarity, blocked, labels, min_size, -0.5, cap)
        assert [merge[:2] for merge in merges] == [merge[:2] for merge in expected]
        assert np.allclose([merge[2] for merge in merges], [merge[2] for merge in expected], atol=1e-9)
        assert len(merges) > 5


def test_two_large_clusters_become_a_suggestion_not_an_automatic_merge():
    run = two_looks_and_a_stranger()
    similarity = run.embeddings["m"] @ run.embeddings["m"].T
    within_look = similarity[:8, :8][np.triu_indices(8, 1)].mean()
    across_looks = similarity[:8, 8:16].mean()
    merge = float((within_look + across_looks) / 2)
    cluster = float(across_looks / within_look) - .05
    free = group_faces(run, GroupingPolicy(("m",), merge_threshold=merge, cluster_merge_threshold=cluster))
    assert free.labels[0] == free.labels[8]
    capped = group_faces(run, GroupingPolicy(("m",), merge_threshold=merge, cluster_merge_threshold=cluster,
                                             cluster_auto_max_size=7, suggest_threshold=cluster))
    assert capped.labels[0] != capped.labels[8]  # both looks have 8 faces, above the cap
    assert set(capped.suggestions[0]["groups"]) == {capped.labels[0], capped.labels[8]}
    roomy = group_faces(run, GroupingPolicy(("m",), merge_threshold=merge, cluster_merge_threshold=cluster, cluster_auto_max_size=8))
    assert roomy.labels[0] == roomy.labels[8]
    with pytest.raises(ValueError):
        GroupingPolicy(("m",), 0.5, cluster_auto_max_size=0).validate()


def look_alikes(seed=0, per_person=40):
    """Two similar people: identity differs only along low-variance directions, nuisance is large."""
    generator = np.random.default_rng(seed)
    dims = 32
    base = np.zeros(dims)
    base[0] = 3.0
    offset = np.zeros(dims)
    offset[1] = 0.35  # identity difference, small
    vectors, labels = [], []
    for label, center in (("A", base + offset), ("B", base - offset)):
        for _ in range(per_person):
            nuisance = np.zeros(dims)
            nuisance[2:10] = generator.normal(scale=1.0, size=8)  # pose, light: large, shared by both people
            noise = generator.normal(scale=0.05, size=dims)
            vectors.append(center + nuisance + noise)
            labels.append(label)
    return make_run([1] * len(labels), vectors, labels=labels)


def test_within_class_whitening_separates_look_alikes():
    from videoatlas.photo_backend import within_class_whitening
    run = look_alikes()
    labels = np.asarray([0] * 40 + [1] * 40)
    before = verification_report(run, "m")["auc"]
    whitening = within_class_whitening(run.embeddings["m"], labels, min_size=10, shrinkage=0.05)
    run.embeddings["w"] = whitening.apply(run.embeddings["m"])
    run.keys["w"] = "w:key"
    after = verification_report(run, "w")["auc"]
    assert before < .8 and after > .95
    assert np.allclose(np.linalg.norm(run.embeddings["w"], axis=1), 1, atol=1e-5)
    assert whitening.info["groups_used"] == 2 and len(whitening.digest) == 16
    again = within_class_whitening(run.embeddings["m"], labels, min_size=10, shrinkage=0.05)
    assert again.digest == whitening.digest
    with pytest.raises(ValueError):
        within_class_whitening(run.embeddings["m"], np.asarray([0] * 80), min_size=10)


def test_adapt_command_writes_a_separate_embedding_space(tmp_path):
    run_folder = tmp_path / "run"
    (run_folder / "embeddings").mkdir(parents=True)
    run = two_looks_and_a_stranger()
    lines = [json.dumps({**photo, "faces": [{**face, "usable": True} for face in photo["faces"]]}) for photo in run.photos]
    (run_folder / "faces.jsonl").write_text("\n".join(lines))
    (run_folder / "report.json").write_text("{}")
    np.savez_compressed(run_folder / "embeddings" / "m.npz", face_ids=np.asarray(run.face_ids), vectors=run.embeddings["m"],
                        key=np.asarray("m:key"), sha256=np.asarray("abc"))
    command = ["photos", "adapt", str(run_folder), "--policy", str(tmp_path / "policy.json"), "--min-size", "5"]
    (tmp_path / "policy.json").write_text(json.dumps(GroupingPolicy(("m",), merge_threshold=-1.0).to_json()))
    assert cli.main(command) != 0  # one group only: no within-person spread across people
    similarity = run.embeddings["m"] @ run.embeddings["m"].T
    merge = float((similarity[:8, :8][np.triu_indices(8, 1)].mean() + similarity[:8, 8:16].mean()) / 2)
    (tmp_path / "policy.json").write_text(json.dumps(GroupingPolicy(("m",), merge_threshold=merge).to_json()))
    assert cli.main(command) == 0
    adapted = load_run(run_folder, ["m", "m-wccn"])
    assert adapted.keys["m-wccn"].startswith("m:key|wccn:")
    assert adapted.embeddings["m-wccn"].shape == run.embeddings["m"].shape
    assert (run_folder / "transforms" / "m-wccn.npz").is_file() and (run_folder / "adapt-wccn.json").is_file()
    assert cli.main(command) != 0  # refuses to overwrite
    assert cli.main(command + ["--overwrite"]) == 0


def test_hidden_people_and_repeated_files_leave_the_answer_key():
    from videoatlas.photo_evaluation import hide_labels
    vectors = [[1, 0, 0], [1, .1, 0], [0, 1, 0], [0, 1, .1], [0, 0, 1], [1, .05, 0]]
    run = make_run([1] * 6, vectors, labels=["A", "A", "B", "B", "C", "A"], hashes=["h0", "h1", "h2", "h3", "h2", "h0"])
    # h2 is filed under B and C (a shared photo); h0 appears twice under A (a repeated file).
    assert face_labels(run).tolist() == ["A", "A", None, "B", None, None]
    assert set(photo_labels(run)) == {0, 1, 3}
    hide_labels(run, ["C"])
    assert face_labels(run).tolist() == ["A", "A", None, "B", None, None]  # the shared photo stays out
    assert run.photos[4]["label"] is None and run.photos[4]["hidden_label"] == "C"
    with pytest.raises(ValueError):
        hide_labels(run, ["Nobody"])


def test_stranger_test_catches_newcomers_absorbed_by_a_loose_attach():
    from videoatlas.photo_evaluation import stranger_test
    generator = np.random.default_rng(3)
    centers = unit(generator.normal(size=(4, 32)))
    labels = [person for person in "ABCD" for _ in range(12)]
    vectors = np.concatenate([centers[index] + 0.25 * generator.normal(size=(12, 32)) for index in range(4)])
    run = make_run([1] * 48, vectors, labels=labels)
    similarity = run.embeddings["m"] @ run.embeddings["m"].T
    within = similarity[:12, :12][np.triu_indices(12, 1)].mean()
    strict = GroupingPolicy(("m",), merge_threshold=float(within) - .05)
    loose = replace(strict, attach_threshold=-1.0, attach_margin=0.0)
    safe = stranger_test(run, strict, ["A", "B"], keep_sizes=(1, 2), repeats=2, batch=2, seed=1)
    assert safe["keep"]["1"]["kept_faces"] == 4 and safe["keep"]["1"]["not_grouped"] == 4
    assert safe["keep"]["1"]["wrong_rate"] == 0
    assert safe["keep"]["2"]["own_group"] + safe["keep"]["2"]["not_grouped"] == 8
    absorbed = stranger_test(run, loose, ["A"], keep_sizes=(1,), repeats=3, batch=1, seed=1)
    entry = absorbed["keep"]["1"]
    assert entry["kept_faces"] == 3 and entry["other_person_group"] == 3 and entry["wrong_rate"] == 1.0
    assert entry["wrong_by_attach"] == 3 and entry["above_attach_threshold"] == 1.0
    assert {item["group_majority"] for item in entry["wrong"]} <= {"B", "C", "D"}
    with pytest.raises(ValueError):
        stranger_test(run, strict, ["Nobody"])
