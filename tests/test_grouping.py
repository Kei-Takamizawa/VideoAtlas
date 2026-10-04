import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from videoatlas.grouping import GroupingOptions, classify_analysis, compare_tracks, recluster_tracks, similarity_statistics
from videoatlas.storage import FaceRecord, PersonRecord, RelationRecord, TrackRecord, VideoRecord
from videoatlas.tracking import build_tracks, refresh_representatives


def detection(identifier, second, x=0, vector=None, video_id="v"):
    return FaceRecord(identifier, video_id, second, (x, 0, .2, .2), None, identifier + ".jpg", vector or [1., 0.], False, False, "face01:v1", .9)


def track(identifier, angle=0, video_id=None, person_id=None):
    vector = [math.cos(angle), math.sin(angle)]
    return TrackRecord(identifier, video_id or identifier, 0, 2, person_id, [], embedding_samples={"face01:v1": [vector] * 3}, quality=.9)


class GroupingTests(unittest.TestCase):
    def test_tracking_requires_spatial_and_embedding_evidence(self):
        faces = [detection("a", 0), detection("b", .5, .01), detection("c", 1, .7)]
        tracks = build_tracks(faces, [], [])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(faces[0].track_id, faces[1].track_id)
        self.assertNotEqual(faces[1].track_id, faces[2].track_id)

    def test_same_frame_two_people_never_share_a_track(self):
        faces = [detection("a", 0), detection("b", 0, .3), detection("c", .5, .01), detection("d", .5, .31)]
        tracks = build_tracks(faces, [], [])
        self.assertEqual(len(tracks), 2)
        self.assertNotEqual(faces[2].track_id, faces[3].track_id)

    def test_maximum_outlier_cannot_establish_high_match(self):
        left = track("a")
        right = track("b")
        right.embedding_samples["face01:v1"] = [[1., 0.], [0., 1.], [0., 1.]]
        comparison = compare_tracks(left, right)
        self.assertEqual(comparison["models"]["face01:v1"]["max"], 1)
        self.assertEqual(comparison["decision"], "LOW")

    def test_versions_auxiliary_disagreement_and_overlap_block_high(self):
        left, right = track("a"), track("b")
        right.embedding_samples = {"face01:v2": [[1., 0.]] * 3}
        self.assertEqual(compare_tracks(left, right)["decision"], "LOW")
        right.embedding_samples = {"face01:v1": [[1., 0.]] * 3, "adaface:v1": [[0., 1.]] * 3}
        left.embedding_samples["adaface:v1"] = [[1., 0.]] * 3
        self.assertNotEqual(compare_tracks(left, right, GroupingOptions(auxiliary_enabled=True))["decision"], "HIGH")
        right.embedding_samples = left.embedding_samples
        right.video_id = left.video_id
        self.assertEqual(compare_tracks(left, right)["decision"], "MEDIUM")
        left.embedding_samples = right.embedding_samples = {"adaface:v1": [[1., 0.]] * 3}
        self.assertNotEqual(compare_tracks(left, right)["decision"], "HIGH")

    def test_complete_link_prevents_transitive_merge(self):
        # A-B and B-C exceed .9; A-C is below .9.
        tracks = [track("a", 0), track("b", .35), track("c", .70)]
        faces = [detection(t.id, 0, video_id=t.video_id) for t in tracks]
        for t, f in zip(tracks, faces):
            f.track_id = t.id
            f.embeddings = {"face01:v1": t.embedding_samples["face01:v1"][0]}
            t.face_ids = [f.id]
        # Avoid refreshing cached samples down to one sample in this unit test.
        from videoatlas.grouping import _cluster_tracks
        result = _cluster_tracks(faces, tracks, [], [], GroupingOptions(auto_merge_enabled=True))
        self.assertEqual(len(result.people), 2)
        self.assertEqual(tracks[0].person_id, tracks[1].person_id)
        self.assertNotEqual(tracks[0].person_id, tracks[2].person_id)

    def test_each_model_threshold_blocks_a_high_weighted_score(self):
        left, right = track("a"), track("b", math.acos(.95))
        left.embedding_samples["adaface:v1"] = [[1., 0.]] * 3
        right.embedding_samples["adaface:v1"] = [[.85, math.sqrt(1 - .85 ** 2)]] * 3
        options = GroupingOptions(auxiliary_enabled=True, high_threshold=.8,
            model_thresholds={"face01": .9, "adaface": .9})
        comparison = compare_tracks(left, right, options)
        self.assertGreater(comparison["score"], .8)
        self.assertEqual(comparison["decision"], "MEDIUM")
        options.model_thresholds["adaface"] = .8
        self.assertEqual(compare_tracks(left, right, options)["decision"], "HIGH")
        options.model_thresholds["face01"] = .99
        self.assertEqual(compare_tracks(left, right, options)["decision"], "MEDIUM")

    def test_default_requires_review_and_negative_constraints_block(self):
        from videoatlas.grouping import _cluster_tracks
        left, right = track("a"), track("b")
        result = _cluster_tracks([], [left, right], [], [], GroupingOptions())
        self.assertEqual(len(result.people), 2)
        self.assertEqual(result.relations[0].relation, "pending")
        left, right = track("a"), track("b")
        result = _cluster_tracks([], [left, right], [], [RelationRecord("a", "b", "confirmed_different")], GroupingOptions(auto_merge_enabled=True))
        self.assertEqual(len(result.people), 2)

    def test_reanalysis_reuses_labelled_track_and_face(self):
        old = detection("f", 0)
        old.track_id, old.person_id, old.manual_assignment = "stable", "p", True
        existing = TrackRecord("stable", "v", 0, 0, "p", ["f"], "f")
        sample = SimpleNamespace(second=0, bounding_box=old.bounding_box, thumbnail_jpeg=b"jpeg", embedding=[1., 0.], embedding_model="face01:v1", quality=.9, rejection_reason=None)
        video = VideoRecord("v", "s", "v.mp4", "v", 1, 1, 0, "new", 0, .5, None, None)
        with tempfile.TemporaryDirectory() as directory:
            result = classify_analysis([sample], video, [old], [PersonRecord("p", "", None)], Path(directory), 1, 1, existing_tracks=[existing])
        self.assertEqual(result.faces[0].id, "f")
        self.assertEqual(result.faces[0].track_id, "stable")
        self.assertEqual(result.faces[0].person_id, "p")
        self.assertTrue(result.faces[0].manual_assignment)

    def test_reanalysis_rebuilds_unreviewed_track_and_prunes_stale_membership(self):
        old = detection("f", 0)
        old.track_id, old.person_id = "inferred-old", "p"
        existing = TrackRecord("inferred-old", "v", 0, 0, "p", ["f"], "f")
        sample = SimpleNamespace(second=0, bounding_box=old.bounding_box, thumbnail_jpeg=b"jpeg", embedding=[1., 0.], embedding_model="face01:v1", quality=.9, rejection_reason=None)
        video = VideoRecord("v", "s", "v.mp4", "v", 1, 1, 0, "new", 0, .5, None, None, analyzed_at="previous-run")
        with tempfile.TemporaryDirectory() as directory:
            result = classify_analysis([sample], video, [old], [PersonRecord("p", "", None)], Path(directory), 1, 1, existing_tracks=[existing])
        rebuilt = result.faces[0]
        self.assertEqual(rebuilt.id, "f")
        self.assertNotEqual(rebuilt.track_id, "inferred-old")
        self.assertEqual(existing.face_ids, [])
        self.assertFalse(any(track.id == "inferred-old" for track in result.tracks))
        self.assertTrue(any(track.id == rebuilt.track_id and track.face_ids == ["f"] for track in result.tracks))

    def test_reanalysis_preserves_named_identity_track_anchor(self):
        old = detection("f", 0)
        old.track_id, old.person_id = "named-track", "p"
        existing = TrackRecord("named-track", "v", 0, 0, "p", ["f"], "f")
        sample = SimpleNamespace(second=0, bounding_box=old.bounding_box, thumbnail_jpeg=b"jpeg", embedding=[1., 0.], embedding_model="face01:v1", quality=.9, rejection_reason=None)
        video = VideoRecord("v", "s", "v.mp4", "v", 1, 1, 0, "new", 0, .5, None, None, analyzed_at="previous-run")
        with tempfile.TemporaryDirectory() as directory:
            result = classify_analysis([sample], video, [old], [PersonRecord("p", "Alice", None)], Path(directory), 1, 1, existing_tracks=[existing])
        self.assertEqual(result.faces[0].track_id, "named-track")
        self.assertTrue(any(track.id == "named-track" for track in result.tracks))

    def test_match_rate_denominator_is_pair_comparisons(self):
        statistics = similarity_statistics([[1, 0], [0, 1]], [[1, 0], [0, 1]])
        self.assertEqual(statistics["pair_count"], 4)
        self.assertEqual(statistics["match_rate"], .5)

    def test_vectorized_statistics_match_scalar_cosine_and_exclude_bad_evidence(self):
        from videoatlas.tracking import cosine
        rng = np.random.default_rng(21)
        left, right = rng.normal(size=(5, 512)).tolist(), rng.normal(size=(6, 512)).tolist()
        expected = sorted(cosine(a, b) for a in left for b in right)
        statistics = similarity_statistics(left + [[float("nan")]], right + [[0] * 512], 0)
        self.assertEqual(statistics["pair_count"], 30)
        self.assertEqual((statistics["sample_count_a"], statistics["sample_count_b"]), (5, 6))
        self.assertAlmostEqual(statistics["max"], expected[-1], places=12)
        self.assertAlmostEqual(statistics["mean"], sum(expected)/30, places=12)
        self.assertEqual(statistics["match_rate"], sum(score >= 0 for score in expected)/30)

    def test_redundant_samples_are_capped_and_pose_strings_work(self):
        faces = [detection(str(index), index * .5) for index in range(60)]
        for face in faces:
            face.pose = "frontal"
        representative = TrackRecord("a", "v", -100, 100, face_ids=[face.id for face in faces])
        refresh_representatives(representative, {face.id: face for face in faces})
        self.assertEqual(len(representative.sample_face_ids), 3)
        self.assertEqual(representative.start_time, 0)
        self.assertEqual(representative.end_time, 29.5)
        self.assertIn("frontal", representative.pose_face_ids)

    def test_rejected_faces_do_not_create_people(self):
        sample = SimpleNamespace(second=0, bounding_box=(0, 0, .2, .2), thumbnail_jpeg=b"jpeg", embedding=None, embedding_model=None, quality=.1, rejection_reason="blur")
        video = VideoRecord("v", "s", "v.mp4", "v", 1, 1, 0, "new", 0, .5, None, None)
        with tempfile.TemporaryDirectory() as directory:
            result = classify_analysis([sample], video, [], [], Path(directory), 1, 1)
        self.assertFalse(result.people)
        self.assertIsNone(result.faces[0].person_id)
        self.assertEqual(len(result.tracks), 1)

    def test_one_native_frame_cannot_supply_multiple_evidence_samples(self):
        faces = [detection(str(index), index * .6, vector=[1., index * .1]) for index in range(3)]
        for face, pose in zip(faces, ("left", "frontal", "right")):
            face.frame_number = 7
            face.pose = pose
        representative = TrackRecord("t", "v", 0, 2, face_ids=[face.id for face in faces])
        refresh_representatives(representative, {face.id: face for face in faces})
        self.assertEqual(len(representative.sample_face_ids), 1)

    def test_negative_constraint_on_any_cluster_member_blocks_addition(self):
        from videoatlas.grouping import _cluster_tracks
        a, b, c = track("a", person_id="p"), track("b", person_id="p"), track("c")
        result = _cluster_tracks([], [a, b, c], [PersonRecord("p", "", None)], [RelationRecord("b", "c", "confirmed_different")], GroupingOptions(auto_merge_enabled=True))
        self.assertNotEqual(c.person_id, "p")

    def test_recluster_splits_inferred_cluster_without_erasing_name(self):
        faces = [detection("a", 0, video_id="a"), detection("b", 0, vector=[0., 1.], video_id="b")]
        tracks = []
        for face in faces:
            face.track_id, face.person_id = face.id, "p"
            tracks.append(TrackRecord(face.id, face.video_id, 0, 0, "p", [face.id]))
        result = recluster_tracks(faces, tracks, [PersonRecord("p", "Alice", None)], [], GroupingOptions(auto_merge_enabled=True))
        self.assertNotEqual(tracks[0].person_id, tracks[1].person_id)
        self.assertEqual(tracks[0].person_id, "p")
        self.assertTrue(any(person.id == "p" and person.name == "Alice" for person in result.people))

    def test_deferred_review_evidence_refresh_preserves_status(self):
        from videoatlas.grouping import _cluster_tracks
        a, b = track("a", person_id="p"), track("b", person_id="q")
        relation = RelationRecord("a", "b", "deferred", {"score": 0.2})
        result = _cluster_tracks([], [a, b], [PersonRecord("p", "", None), PersonRecord("q", "", None)], [relation], GroupingOptions())
        self.assertEqual(result.relations[0].relation, "deferred")
        self.assertEqual(result.relations[0].details["score"], 1.0)

    def test_changed_rescan_sampling_preserves_constraints_and_blocks_new_lineage(self):
        old_faces = [detection("old" + str(index), second) for index, second in enumerate((0, .6, 1.2))]
        right_faces = [detection("right" + str(index), second, video_id="other") for index, second in enumerate((0, .6, 1.2))]
        for face in old_faces:
            face.track_id, face.person_id = "old-track", "p"
        for face in right_faces:
            face.track_id, face.person_id = "right-track", "q"
        tracks = [TrackRecord("old-track", "v", 0, 1.2, "p", [face.id for face in old_faces]), TrackRecord("right-track", "other", 0, 1.2, "q", [face.id for face in right_faces])]
        samples = [SimpleNamespace(second=second, bounding_box=(0, 0, .2, .2), thumbnail_jpeg=b"jpeg", embedding=[1., 0.], embedding_model="face01:v1", quality=.9, rejection_reason=None) for second in (.2, .8, 1.4)]
        video = VideoRecord("v", "s", "v.mp4", "v", 2, 1, 0, "new", 0, .5, None, None, analyzed_at="previous-run")
        relation = RelationRecord("old-track", "right-track", "confirmed_different")
        with tempfile.TemporaryDirectory() as directory:
            result = classify_analysis(samples, video, old_faces + right_faces, [PersonRecord("p", "", None), PersonRecord("q", "", None)], Path(directory), 2, 2, existing_tracks=tracks, relations=[relation], options=GroupingOptions(auto_merge_enabled=True))
        self.assertFalse(result.removed_ids)
        self.assertTrue(any(track.id == "old-track" for track in result.tracks))
        new_track = next(track for track in result.tracks if track.id not in {"old-track", "right-track"})
        self.assertTrue(new_track.review_required)
        self.assertNotEqual(new_track.person_id, "q")
        comparison = compare_tracks(new_track, result.tracks[1])
        self.assertNotEqual(comparison["decision"], "HIGH")

    def test_nearby_sample_does_not_steal_manual_timestamp_anchor(self):
        old = detection("anchored", .5)
        old.track_id, old.person_id, old.manual_assignment = "stable", "p", True
        track_record = TrackRecord("stable", "v", .5, .5, "p", [old.id])
        samples = [SimpleNamespace(second=second, bounding_box=old.bounding_box, thumbnail_jpeg=b"jpeg", embedding=[1., 0.], embedding_model="face01:v1", quality=.9, rejection_reason=None) for second in (.4, .5)]
        video = VideoRecord("v", "s", "v.mp4", "v", 1, 1, 0, "new", 0, .5, None, None, analyzed_at="previous")
        with tempfile.TemporaryDirectory() as directory:
            result = classify_analysis(samples, video, [old], [PersonRecord("p", "", None)], Path(directory), 1, 1, existing_tracks=[track_record])
        anchored = next(face for face in result.faces if face.id == "anchored")
        self.assertEqual(anchored.second, .5)
        self.assertTrue(anchored.manual_assignment)
        early = next(face for face in result.faces if face.second == .4)
        self.assertNotEqual(early.id, "anchored")
        self.assertFalse(early.manual_assignment)

    def test_overlapping_competing_samples_cannot_transplant_manual_label(self):
        old = detection("anchored", .5)
        old.track_id, old.person_id, old.manual_assignment = "stable", "p", True
        track_record = TrackRecord("stable", "v", .5, .5, "p", [old.id])
        samples = [SimpleNamespace(second=.5, bounding_box=(x, 0, .2, .2), thumbnail_jpeg=b"jpeg", embedding=[1., 0.], embedding_model="face01:v1", quality=.9, rejection_reason=None) for x in (0, .01)]
        video = VideoRecord("v", "s", "v.mp4", "v", 1, 1, 0, "new", 0, .5, None, None, analyzed_at="previous")
        with tempfile.TemporaryDirectory() as directory:
            result = classify_analysis(samples, video, [old], [PersonRecord("p", "", None)], Path(directory), 1, 1, existing_tracks=[track_record])
        self.assertFalse(result.removed_ids)
        self.assertTrue(all(face.id != "anchored" and not face.manual_assignment for face in result.faces))
        self.assertTrue(all(track.review_required for track in result.tracks if track.id != "stable"))
        self.assertEqual(old.person_id, "p")


if __name__ == "__main__":
    unittest.main()
