"""Offscreen GUI contract tests; no recognition models or playback required."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication, QDialog, QLabel, QLineEdit, QPushButton, QScrollArea, QSplitter
except ImportError:
    QApplication = None

if QApplication is not None:
    from videoatlas.ui import MainWindow, SettingsDialog

    class FakeController(QObject):
        changed = Signal()
        status_changed = Signal()
        def __init__(self, root):
            super().__init__()
            self.data_dir = root
            self.recognition_config_path = root / "recognition.json"
            self.sources = []
            self.videos = []
            self.faces = []
            self.people = []
            self.tracks = []
            self.pending_reviews = []
            self.is_scanning = False
            self.status_text = ""
            self.runtime_description = ""
            self.progress = 0
            self.decisions = []
            self.saved = None
            self.renames = []
        def close(self): pass
        def resolve_review(self, *args): self.decisions.append(args)
        def rename_person(self, person_id, name):
            self.renames.append((person_id, name))
            person = next(item for item in self.people if item.id == person_id)
            person.name = name
            self.changed.emit()
        def save_settings(self, values):
            if self.is_scanning: return False
            self.saved = values
            return True

@unittest.skipIf(QApplication is None, "PySide6 is unavailable")
class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.model = FakeController(Path(self.directory.name))
        self.window = MainWindow(self.model)
    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.directory.cleanup()
    def review(self):
        self.model.people = [SimpleNamespace(id="p", name="Alice", thumbnail_path=None)]
        self.model.tracks = [SimpleNamespace(id=identifier, video_id="v", person_id="p", start_time=12, end_time=25,
            representative_face_id="f", face_ids=["f"]) for identifier in ("a", "b")]
        self.model.videos = [SimpleNamespace(id="v", name="sample.mp4", path="C:/missing.mp4")]
        self.model.faces = [SimpleNamespace(id="f", thumbnail_path="C:/missing-face.jpg")]
        self.model.pending_reviews = [SimpleNamespace(track_a="a", track_b="b", relation="pending",
            details={"decision":"MEDIUM", "score":.625, "reasons":["Model disagreement"], "models":{"Japanese Face": {"median":.85,"sample_count_a":3,"sample_count_b":4,"pair_count":12},"AdaFace": {"median":.4}}})]
        self.window._select_section("review")
    def test_review_evidence_and_dispatch(self):
        self.review()
        text = "\n".join(item.text() for item in self.window.findChildren(QLabel))
        self.assertIn("MEDIUM", text)
        self.assertIn("Model disagreement", text)
        self.assertIn("Japanese Face", text)
        self.assertIn("Median similarity: 0.850", text)
        self.assertIn("Samples: 3 / 4", text)
        self.assertIn("Track a", text)
        self.assertIn("Track b", text)
        self.assertIn("sample.mp4", text)
        self.assertEqual([shortcut.key().toString() for shortcut in self.window.review_shortcuts], ["S", "D", "L"])
        self.window.review_buttons[0].click()
        self.assertEqual(self.model.decisions, [("a", "b", "confirmed_same")])
    def test_review_cards_do_not_overlap(self):
        self.review()
        self.window.show()
        self.app.processEvents()
        labels = [item for item in self.window.findChildren(QLabel) if item.isVisible()]
        for metadata in [item for item in labels if item.text().startswith("Track a") or item.text().startswith("Track b")]:
            picture = next(item for item in metadata.parentWidget().findChildren(QLabel)
                           if item.text() == "Representative face unavailable")
            self.assertLessEqual(picture.geometry().bottom(), metadata.geometry().top())
        self.assertTrue(all(button.isVisible() for button in self.window.review_buttons))

    def test_analysis_blocks_review_writes(self):
        self.review()
        self.model.is_scanning = True
        self.window._update_status()
        self.assertTrue(all(not item.isEnabled() for item in self.window.review_buttons))
        self.assertTrue(all(not item.isEnabled() for item in self.window.review_shortcuts))
        self.window._resolve_review("confirmed_different")
        self.assertEqual(self.model.decisions, [])
    def test_missing_video_does_not_open_replaced_path(self):
        replaced = Path(self.directory.name) / "replaced.mp4"
        replaced.write_bytes(b"replacement content")
        video = SimpleNamespace(state="missing", path=str(replaced))
        with patch("videoatlas.ui.show_warning") as warning, patch("videoatlas.ui.PlayerDialog") as player:
            self.window._open_player(video)
        warning.assert_called_once()
        player.assert_not_called()

    def test_settings_collect_types_and_controller_save(self):
        dialog = SettingsDialog(self.model)
        values = dialog.values()
        self.assertFalse(values["auto_merge_enabled"])
        self.assertEqual(values["min_face_size"], 80)
        self.assertIn(".mp4", values["extensions"])
        dialog.fields["target_folders"].setPlainText("C:/Videos\nC:/Other")
        dialog._save()
        self.assertEqual(self.model.saved["target_folders"], ["C:/Videos", "C:/Other"])

    def test_complete_settings_save_with_blank_optional_paths(self):
        from videoatlas.controller import LibraryController
        from videoatlas.recognition import RecognitionOptions
        model = LibraryController(Path(self.directory.name) / "settings-index")
        try:
            dialog = SettingsDialog(model)
            dialog.fields["cache_dir"].clear()
            dialog.fields["adaface_model_path"].clear()
            values = dialog.values()
            self.assertIsNone(values["cache_dir"])
            self.assertIsNone(values["adaface_model_path"])
            dialog._save()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted, model.status_text)
            saved = RecognitionOptions.load(model.recognition_config_path)
            self.assertIsNone(saved.cache_dir)
            self.assertIsNone(saved.adaface_model_path)
            self.assertFalse(saved.adaface_enabled)
        finally:
            model.close()
    def test_people_three_panes_with_track_metadata(self):
        self.model.people = [SimpleNamespace(id="person", name="", thumbnail_path=None, created_at="2026-10-03")]
        self.model.tracks = [SimpleNamespace(id="private-track-uuid", video_id="v", person_id="person", start_time=12, end_time=25,
                                             face_ids=["private-face-uuid"], mean_embeddings={}, quality=.9)]
        self.model.videos = [SimpleNamespace(id="v", name="sample.mp4", path="C:/sample.mp4", duration=60, state="ready")]
        self.model.faces = [SimpleNamespace(id="private-face-uuid", video_id="v", person_id="person", second=12, excluded=False,
                                            thumbnail_path="", quality=.92, rejection_reason=None)]
        self.window.person_id = "person"
        self.window._select_section("people")
        self.assertEqual(self.window.findChildren(QSplitter)[0].count(), 3)
        self.assertEqual({item.objectName() for item in self.window.findChildren(QScrollArea)
                          if item.objectName().startswith("PeoplePaneScroll")},
                         {"PeoplePaneScrollLeft", "PeoplePaneScrollDetails", "PeoplePaneScrollVideos"})
        labels = "\n".join(item.text() for item in self.window.findChildren(QLabel))
        self.assertNotIn("Person 0001", labels)
        self.assertNotIn("Unnamed", labels)
        self.assertNotIn("private-track-uuid", labels)
        self.assertNotIn("private-face-uuid", labels)
        self.assertIn("0:12", labels)
        self.assertIn("0:25", labels)
        self.assertIn("quality 0.900", labels)
        visible_text = labels + "\n" + "\n".join(item.toolTip() for item in self.window.findChildren(QPushButton))
        self.assertNotIn("private-track-uuid", visible_text)
        self.assertNotIn("private-face-uuid", visible_text)
        self.assertTrue(any("sample.mp4" in item.toolTip() and "0:12" in item.toolTip()
                            for item in self.window.findChildren(QPushButton)))

    def test_people_list_paginates_face_tiles_by_sixty(self):
        self.model.people = [SimpleNamespace(id=f"person-{index}", name="", thumbnail_path=None) for index in range(61)]
        self.window._select_section("people")
        self.assertIsNotNone(self.window.findChild(QPushButton, "person-face-person-59"))
        self.assertIsNone(self.window.findChild(QPushButton, "person-face-person-60"))
        self.window.findChild(QPushButton, "ShowMorePeople").click()
        self.assertIsNotNone(self.window.findChild(QPushButton, "person-face-person-60"))

    def test_each_person_shows_shared_and_only_their_unique_videos(self):
        self.model.people = [SimpleNamespace(id="person-a", name="", thumbnail_path=None),
                             SimpleNamespace(id="person-b", name="", thumbnail_path=None)]
        self.model.videos = [SimpleNamespace(id=identifier, name=name, path=f"C:/{name}", duration=60, state="ready")
                             for identifier, name in (("shared", "shared.mp4"), ("a-only", "a-only.mp4"), ("b-only", "b-only.mp4"))]
        self.model.faces = [SimpleNamespace(id="face-a-shared", video_id="shared", person_id="person-a", second=2, excluded=False, thumbnail_path=""),
                            SimpleNamespace(id="face-a-only", video_id="a-only", person_id="person-a", second=4, excluded=False, thumbnail_path=""),
                            SimpleNamespace(id="face-b-shared", video_id="shared", person_id="person-b", second=8, excluded=False, thumbnail_path=""),
                            SimpleNamespace(id="face-b-only", video_id="b-only", person_id="person-b", second=10, excluded=False, thumbnail_path="")]
        self.window._select_section("people")

        self.window.findChild(QPushButton, "person-face-person-a").click()
        self.app.processEvents()
        person_a_pane = self.window.findChildren(QSplitter)[-1].widget(2)
        person_a_videos = {button.text().splitlines()[0] for button in person_a_pane.findChildren(QPushButton)
                           if button.text().splitlines() and button.text().splitlines()[0].endswith(".mp4")}
        self.assertEqual(person_a_videos, {"shared.mp4", "a-only.mp4"})

        self.window.findChild(QPushButton, "person-face-person-b").click()
        self.app.processEvents()
        person_b_pane = self.window.findChildren(QSplitter)[-1].widget(2)
        person_b_videos = {button.text().splitlines()[0] for button in person_b_pane.findChildren(QPushButton)
                           if button.text().splitlines() and button.text().splitlines()[0].endswith(".mp4")}
        self.assertEqual(person_b_videos, {"shared.mp4", "b-only.mp4"})

    def test_face_click_selects_person_and_shows_associated_videos(self):
        self.model.people = [SimpleNamespace(id="person-a", name="", thumbnail_path=None)]
        self.model.videos = [SimpleNamespace(id="video-a", name="holiday.mp4", path="C:/holiday.mp4", duration=120, state="ready")]
        self.model.tracks = [SimpleNamespace(id="track-a", video_id="video-a", person_id="person-a", start_time=3, end_time=9, face_ids=["face-a"])]
        self.model.faces = [SimpleNamespace(id="face-a", video_id="video-a", person_id="person-a", second=3, excluded=False, thumbnail_path="")]
        self.window._select_section("people")

        self.window.findChild(QPushButton, "person-face-person-a").click()

        self.assertEqual(self.window.person_id, "person-a")
        buttons = [item.text() for item in self.window.findChildren(QPushButton)]
        self.assertTrue(any("holiday.mp4" in text and "ready" in text for text in buttons))
        self.assertTrue(self.window.findChild(QLineEdit, "PersonNameEdit").text() == "")

    def test_inline_name_save_uses_controller_rename(self):
        self.model.people = [SimpleNamespace(id="person-a", name="", thumbnail_path=None)]
        self.window.person_id = "person-a"
        self.window._select_section("people")

        self.window.findChild(QLineEdit, "PersonNameEdit").setText("Aiko")
        save_button = next(button for button in self.window.findChildren(QPushButton) if button.text() == "Save name")
        save_button.click()

        self.assertEqual(self.model.renames, [("person-a", "Aiko")])
        labels = "\n".join(item.text() for item in self.window.findChildren(QLabel))
        self.assertIn("Aiko", labels)

    def test_inline_name_save_dispatches_blank_to_clear_optional_name(self):
        self.model.people = [SimpleNamespace(id="person-a", name="Aiko", thumbnail_path=None)]
        self.window.person_id = "person-a"
        self.window._select_section("people")
        self.window.findChild(QLineEdit, "PersonNameEdit").clear()
        next(button for button in self.window.findChildren(QPushButton) if button.text() == "Save name").click()
        self.assertEqual(self.model.renames, [("person-a", "")])

    def test_track_details_show_compatible_representative_cosine_similarity(self):
        model_key = "face01:secret-digest:preprocess:runtime"
        ada_key = "adaface:secret-digest:preprocess:runtime"
        self.model.people = [SimpleNamespace(id="person-a", name="", thumbnail_path=None, representative_face_id="rep-face")]
        self.model.tracks = [
            SimpleNamespace(id="private-reference-track", video_id="v1", person_id="person-a", start_time=0, end_time=3,
                            face_ids=["rep-face"], representative_face_id="rep-face",
                            mean_embeddings={model_key: [1, 0], ada_key: [1, 0]}, quality=.9),
            SimpleNamespace(id="private-other-track", video_id="v2", person_id="person-a", start_time=5, end_time=8,
                            face_ids=["other-face"], representative_face_id="other-face",
                            mean_embeddings={model_key: [.8, .6], ada_key: [.8, .6]}, quality=.8),
        ]
        self.window.person_id = "person-a"
        self.window._select_section("people")
        summaries = [item.text() for item in self.window.findChildren(QLabel) if item.objectName() == "TrackSimilaritySummary"]
        self.assertIn("Selected representative track", summaries)
        self.assertTrue(any("Representative cosine similarity" in text and "JAPANESE FACE V1: 0.800" in text
                            and "AdaFace: 0.800" in text for text in summaries))
        self.assertNotIn("secret-digest", "\n".join(item.text() for item in self.window.findChildren(QLabel)))
        self.assertNotIn("private-other-track", "\n".join(item.text() for item in self.window.findChildren(QLabel)))

    def test_people_and_menus_do_not_render_automatic_aliases(self):
        from videoatlas.ui import face_menu
        self.model.people = [SimpleNamespace(id="person-a", name="", thumbnail_path=None),
                             SimpleNamespace(id="person-b", name="", thumbnail_path=None)]
        self.model.faces = [SimpleNamespace(id="face-a", video_id="video-a", person_id="person-a", second=0, excluded=False,
                                            thumbnail_path="", track_id="track-a")]
        self.window._select_person("person-a")
        text = "\n".join(item.text() for item in self.window.findChildren(QLabel))
        self.assertNotIn("Person 0001", text)
        self.assertNotIn("Unnamed", text)
        menu = face_menu(self.window, self.model, self.model.faces[0])
        actions = menu.actions()
        move_menu = next(action.menu() for action in actions if action.menu() is not None and action.text() == "Move to another person")
        self.assertEqual(move_menu.actions()[0].text(), "0 videos · 0 tracks")
        self.assertTrue(move_menu.actions()[0].icon().isNull())
        self.assertNotIn("Person 0001", " ".join(action.text() for action in move_menu.actions()))
        self.assertNotIn("Unnamed", " ".join(action.text() for action in move_menu.actions()))

    def test_upper_region_setting_has_stable_value_and_explains_reanalysis(self):
        dialog = SettingsDialog(self.model)
        control = dialog.fields["recognition_region"]
        self.assertEqual(control.currentData(), "full")
        control.setCurrentIndex(control.findData("upper"))
        self.assertEqual(dialog.values()["recognition_region"], "upper")
        self.assertIn("reanalysis", control.toolTip())
        dialog.close()

    def test_obsolete_inferred_people_are_hidden_without_removing_history(self):
        self.model.people = [SimpleNamespace(id="active", name="", thumbnail_path=None, representative_face_id="face"),
                             SimpleNamespace(id="old", name="", thumbnail_path=None, representative_face_id="face")]
        self.model.faces = [SimpleNamespace(id="face", video_id="v", person_id="active", excluded=False, thumbnail_path="")]
        self.window._select_section("people")
        self.assertIsNotNone(self.window.findChild(QPushButton,"person-face-active"))
        self.assertIsNone(self.window.findChild(QPushButton,"person-face-old"))
        self.assertEqual(len(self.model.people),2)

if __name__ == "__main__": unittest.main()
