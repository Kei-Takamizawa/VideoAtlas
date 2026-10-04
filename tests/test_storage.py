import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from videoatlas.storage import FaceRecord, LibraryStore, PersonRecord, RelationRecord, SourceFolder, TrackRecord, VideoRecord, file_hash, list_video_files


def video(identifier="v"):
    return VideoRecord(identifier, "s", "movie.mp4", "movie.mp4", 10, 100, 0, "new", 0, .5, None, None)


def face(identifier="f", person_id="p", track_id="t"):
    return FaceRecord(identifier, "v", 1, (0, 0, .2, .2), person_id, "thumb.jpg", [1., 0.], False, False, "face01:v1", .9, track_id=track_id)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = LibraryStore(self.root)
        self.store.save_source(SourceFolder("s", "source"))
        self.store.save_video(video())
        for identifier in ("p", "q"):
            self.store.save_person(PersonRecord(identifier, "", None))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def populate(self):
        self.store.save_face(face())
        self.store.save_track(TrackRecord("t", "v", 1, 1, "p", ["f"], "f"))
        self.store.save_face(face("g", "q", "u"))
        self.store.save_track(TrackRecord("u", "v", 2, 2, "q", ["g"], "g"))

    def test_embedding_versions_remain_separate(self):
        record = face()
        record.embeddings["face01:v2"] = [0., 1.]
        record.embeddings["adaface:hash:preprocess"] = [1., 1.]
        self.store.save_face(record)
        keys = {(row[0], row[1]) for row in self.store._connection.execute("SELECT model_name,model_version FROM face_embeddings")}
        self.assertEqual(keys, {("face01", "v1"), ("face01", "v2"), ("adaface", "hash:preprocess")})
        self.assertEqual(self.store.load_snapshot().faces[0].embeddings, record.embeddings)

    def test_atomic_batch_rollback(self):
        self.store.save_face(face())
        bad = face("bad", "missing")
        changed = video()
        changed.state = "complete"
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.apply_face_batch([bad], changed, [], ["f"])
        snapshot = self.store.load_snapshot()
        self.assertEqual(snapshot.faces[0].id, "f")
        self.assertEqual(snapshot.videos[0].state, "new")

    def test_negative_relation_blocks_merge_and_assignment(self):
        self.populate()
        self.store.resolve_relation("t", "u", "confirmed_different")
        with self.assertRaises(ValueError):
            self.store.merge_people("p", "q")
        with self.assertRaises(ValueError):
            self.store.assign_track("t", "q")
        self.assertEqual(len(self.store.load_snapshot().people), 2)

    def test_merge_split_preserves_durable_labels(self):
        self.populate()
        self.store.merge_people("p", "q")
        self.assertEqual(self.store.load_snapshot().relations[0].relation, "confirmed_same")
        new_person = self.store.split_track("t", "separate")
        snapshot = self.store.load_snapshot()
        self.assertEqual(snapshot.relations[0].relation, "confirmed_different")
        self.assertEqual(snapshot.relations[0].details["history"][0]["relation"], "confirmed_same")
        self.assertEqual(next(f for f in snapshot.faces if f.id == "f").person_id, new_person.id)
        self.store.delete_video("v")
        self.assertEqual(len(self.store.load_snapshot().relations), 1)

    def test_exclusion_updates_representatives(self):
        self.populate()
        self.store.set_representative("f")
        self.store.exclude_face("f")
        snapshot = self.store.load_snapshot()
        self.assertTrue(next(f for f in snapshot.faces if f.id == "f").excluded)
        self.assertIsNone(next(p for p in snapshot.people if p.id == "p").representative_face_id)
        self.assertNotIn("f", next(t for t in snapshot.tracks if t.id == "t").sample_face_ids)

    def test_old_json_payload_migration_is_lossless(self):
        self.store.close()
        connection = sqlite3.connect(self.root / "library.sqlite3")
        old_face = {"id": "old", "video_id": "v", "second": 0, "bounding_box": [0, 0, .2, .2], "person_id": "p", "thumbnail_path": "old.jpg", "embedding": [1, 0], "manual_assignment": True, "excluded": False, "embedding_model": "face01:old"}
        connection.execute("INSERT INTO faces VALUES(?,?,?,?)", ("old", "v", "p", json.dumps(old_face)))
        connection.commit()
        connection.close()
        self.store = LibraryStore(self.root)
        snapshot = self.store.load_snapshot()
        self.assertEqual(snapshot.faces[0].person_id, "p")
        self.assertTrue(snapshot.faces[0].manual_assignment)
        self.assertTrue(snapshot.faces[0].track_id)
        self.assertEqual(snapshot.faces[0].embeddings["face01:old"], [1, 0])
        self.assertEqual(snapshot.videos[0].width, 0)

    def test_settings_history_and_formats(self):
        self.store.save_analysis_settings("a", {"threshold": .9})
        self.store.save_analysis_settings("b", {"threshold": .8})
        self.assertEqual(len(self.store.load_analysis_settings()), 2)
        directory = self.root / "media"
        directory.mkdir()
        sub = directory / "sub"
        sub.mkdir()
        for extension in ("mp4", "mov", "avi", "mkv", "m4v", "webm"):
            (directory / ("movie." + extension)).write_bytes(b"content")
        (sub / "another.mp4").write_bytes(b"content")
        self.assertEqual(len(list_video_files(directory)), 7)
        self.assertEqual(len(list_video_files(directory, recursive=False)), 6)
        self.assertEqual(file_hash(directory / "movie.mp4"), file_hash(sub / "another.mp4"))

    def test_review_same_then_different_changes_membership_and_keeps_history(self):
        self.populate()
        self.store.resolve_relation("t", "u", "confirmed_same")
        snapshot = self.store.load_snapshot()
        self.assertEqual(len({track.person_id for track in snapshot.tracks}), 1)
        self.store.resolve_relation("t", "u", "confirmed_different")
        snapshot = self.store.load_snapshot()
        self.assertEqual(len({track.person_id for track in snapshot.tracks}), 2)
        self.assertEqual(snapshot.relations[0].details["history"][0]["relation"], "confirmed_same")

    def test_splitting_chosen_representative_repairs_old_person_image(self):
        self.populate()
        self.store.merge_people("p", "q")
        self.store.set_representative("f")
        self.store.split_track("t")
        old_person = next(person for person in self.store.load_snapshot().people if person.id == "q")
        self.assertEqual(old_person.representative_face_id, "g")

    def test_empty_tracks_retire_without_deleting_labels(self):
        self.populate()
        self.store.resolve_relation("t", "u", "pending")
        self.store.apply_face_batch([], video(), [], ["f", "g"], tracks=[])
        snapshot = self.store.load_snapshot()
        self.assertFalse(snapshot.tracks)
        self.assertEqual(len(snapshot.relations), 1)

    def test_legacy_json_import_does_not_resurrect_after_clear(self):
        legacy_root = self.root / "legacy"
        legacy_root.mkdir()
        data = {"sources": [{"id": "s", "path": "legacy"}], "videos": [video().__dict__], "people": [{"id": "p", "name": "Person", "thumbnail_path": None}], "faces": [face().__dict__]}
        # Legacy JSON has no track identifiers yet.
        data["faces"][0].pop("track_id")
        path = legacy_root / "library.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        original = path.read_bytes()
        with LibraryStore(legacy_root) as store:
            self.assertEqual(store.load_snapshot().faces[0].person_id, "p")
            store.clear_index()
        with LibraryStore(legacy_root) as store:
            self.assertFalse(store.load_snapshot().sources)
            self.assertFalse(store.load_snapshot().faces)
        self.assertEqual(path.read_bytes(), original)

    def test_future_schema_is_rejected_without_downgrade(self):
        future_root = self.root / "future"
        future_root.mkdir()
        connection = sqlite3.connect(future_root / "library.sqlite3")
        connection.execute("PRAGMA user_version=99")
        connection.close()
        with self.assertRaises(RuntimeError):
            LibraryStore(future_root)
        connection = sqlite3.connect(future_root / "library.sqlite3")
        self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)
        connection.close()


if __name__ == "__main__":
    unittest.main()
