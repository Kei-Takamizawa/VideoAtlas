import argparse
import json
from pathlib import Path
import sys
from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QIcon, QPixmap, QShortcut, QKeySequence
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QApplication, QCheckBox, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLayout, QSpinBox, QTextEdit, QTabWidget, QSplitter, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget
from .controller import LibraryController
from .localization import load_language, localize_message, set_language, tr
from .storage import FaceRecord, PersonRecord, VideoRecord
from .tracking import cosine
from .recognition import RecognitionOptions
STYLE = """
QWidget { background: #f5f7fb; color: #17243a; font-family: 'Segoe UI'; font-size: 13px; }
QFrame#Sidebar { background: #15233b; border: none; }
QFrame#Sidebar QLabel { background: transparent; color: #e8f0ff; }
QFrame#Sidebar QPushButton { background: transparent; color: #c8d5e9; text-align: left; border: none; border-radius: 9px; padding: 10px 12px; }
QFrame#Sidebar QPushButton:hover { background: #233b5e; }
QFrame#Sidebar QPushButton:checked { background: #315687; color: white; font-weight: 700; }
QFrame#Sidebar QPushButton#AddFolder { background: #2b79ca; color: white; text-align: center; font-weight: 700; }
QFrame#Sidebar QPushButton#ClearIndex { color: #aebdd2; }
QPushButton { background: #e8eef8; border: 1px solid #d6e0ef; border-radius: 9px; padding: 8px 13px; }
QPushButton:hover { background: #dae7f8; }
QPushButton:disabled { color: #929eac; background: #eef1f5; }
QPushButton#Primary { background: #226fc1; color: white; border: none; font-weight: 700; }
QPushButton#Primary:hover { background: #185b9f; }
QLineEdit { background: white; border: 1px solid #d4deec; border-radius: 9px; padding: 8px 12px; }
QFrame#Card, QFrame#Empty, QFrame#Status { background: white; border: 1px solid #e0e7f1; border-radius: 15px; }
QFrame#Card QLabel, QFrame#Empty QLabel, QFrame#Status QLabel { background: transparent; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QProgressBar { background: #e8eff8; border: none; border-radius: 5px; height: 8px; text-align: center; }
QProgressBar::chunk { background: #2b79ca; border-radius: 5px; }
"""


def time_label(seconds: float) -> str:
    value = max(0, int(seconds))
    if value >= 3600:
        return f"{value // 3600}:{(value // 60) % 60:02d}:{value % 60:02d}"
    return f"{value // 60}:{value % 60:02d}"


def image_icon(path: str | None, width: int, height: int) -> QIcon:
    if not path or not Path(path).is_file():
        return QIcon()
    picture = QPixmap(path)
    if picture.isNull():
        return QIcon()
    scaled = picture.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(scaled)


def clear_layout(layout: QVBoxLayout | QGridLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def visible_people(model: LibraryController) -> list:
    active = {face.person_id for face in model.faces if not getattr(face, "excluded", False)}
    # Keep saved names/history available, but do not display obsolete inferred
    # groups whose representative moved to another track during reanalysis.
    return [person for person in model.people if person.id in active or person.name.strip()
            or not getattr(person, "representative_face_id", None)]


def face_menu(parent: QWidget, model: LibraryController, face: FaceRecord) -> QMenu:
    menu = QMenu(parent)
    representative = menu.addAction(tr("Use as representative face"))
    representative.triggered.connect(lambda: model.set_representative(face.id))
    split = menu.addAction(tr("Split into a new person"))
    split.triggered.connect(lambda: model.split_track(face.track_id) if getattr(face, "track_id", None) else model.split_face(face.id))
    unclassify = menu.addAction(tr("Move to unclassified"))
    unclassify.triggered.connect(lambda: model.assign_face(face.id, None))
    move = menu.addMenu(tr("Move to another person"))
    for person in visible_people(model):
        if person.id != face.person_id:
            action = move.addAction(image_icon(person_thumbnail_path(model, person), 32, 32), person.name.strip() or person_summary(model, person))
            action.setToolTip(person_tooltip(model, person))
            action.triggered.connect(lambda checked=False, target=person.id: model.assign_face(face.id, target))
    exclude = menu.addAction(tr("Exclude this face"))
    exclude.triggered.connect(lambda: model.exclude_face(face.id))
    menu.setEnabled(not model.is_scanning)
    return menu


def person_tracks(model: LibraryController, person: PersonRecord) -> list:
    return [track for track in getattr(model, "tracks", []) if track.person_id == person.id]


def person_video_ids(model: LibraryController, person: PersonRecord) -> set[str]:
    ids = {track.video_id for track in person_tracks(model, person)}
    ids.update(face.video_id for face in model.faces if face.person_id == person.id and not face.excluded)
    return ids


def person_thumbnail_path(model: LibraryController, person: PersonRecord) -> str | None:
    if person.thumbnail_path:
        return person.thumbnail_path
    face_id = getattr(person, "representative_face_id", None)
    face = next((item for item in model.faces if item.id == face_id), None)
    return face.thumbnail_path if face else None


def person_summary(model: LibraryController, person: PersonRecord) -> str:
    tracks = person_tracks(model, person)
    return f"{len(person_video_ids(model, person))} videos · {len(tracks)} tracks"


def person_tooltip(model: LibraryController, person: PersonRecord) -> str:
    names = [video.name for video in model.videos if video.id in person_video_ids(model, person)]
    return "Videos: " + (", ".join(names) if names else "none") + f"\n{person_summary(model, person)}"


def person_pane_scroll(content: QWidget, object_name: str) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setObjectName(object_name)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(content)
    return scroll


def track_similarity_summary(reference, candidate) -> str:
    reference_embeddings = getattr(reference, "mean_embeddings", {}) or {}
    candidate_embeddings = getattr(candidate, "mean_embeddings", {}) or {}
    scores = []
    used_names = set()
    for key in sorted(reference_embeddings.keys() & candidate_embeddings.keys()):
        score = cosine(reference_embeddings[key], candidate_embeddings[key])
        if score is None:
            continue
        prefix = key.split(":", 1)[0].casefold()
        if prefix in {"face01", "japanese_face", "japanese-face"}:
            model_name = "JAPANESE FACE V1"
        elif prefix == "adaface":
            model_name = "AdaFace"
        else:
            model_name = "Other model"
        if model_name not in used_names:
            scores.append((model_name, score))
            used_names.add(model_name)
        if len(scores) >= 3:
            break
    if not scores:
        return "Representative cosine similarity (not an accuracy score): unavailable (no compatible track embeddings)"
    return "Representative cosine similarity (not an accuracy score) · " + " · ".join(f"{name}: {score:.3f}" for name, score in scores)


def confirm_deletion(parent: QWidget, title: str, message: str) -> bool:
    dialog = QMessageBox(QMessageBox.Question, title, message, QMessageBox.Yes | QMessageBox.No, parent)
    dialog.button(QMessageBox.Yes).setText(tr("Delete"))
    dialog.button(QMessageBox.No).setText(tr("Cancel"))
    dialog.setDefaultButton(QMessageBox.No)
    return dialog.exec() == QMessageBox.Yes


def show_warning(parent: QWidget, title: str, message: str) -> None:
    dialog = QMessageBox(QMessageBox.Warning, title, message, QMessageBox.Ok, parent)
    dialog.button(QMessageBox.Ok).setText(tr("Close"))
    dialog.exec()


class PlayerDialog(QDialog):
    def __init__(self, model: LibraryController, video: VideoRecord, second: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.video = video
        self.setWindowTitle(f"VideoAtlas — {video.name}")
        self.resize(960, 680)
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel(video.name)
        title.setStyleSheet("font-size: 19px; font-weight: 700;")
        header.addWidget(title)
        header.addStretch()
        rescan = QPushButton(tr("Rescan in detail"))
        rescan.clicked.connect(lambda: self.model.rescan_video(video.id, True))
        rescan.setEnabled(not model.is_scanning)
        header.addWidget(rescan)
        close_button = QPushButton(tr("Close"))
        close_button.clicked.connect(self.accept)
        header.addWidget(close_button)
        root.addLayout(header)
        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumHeight(340)
        self.video_widget.setStyleSheet("background: #101827;")
        root.addWidget(self.video_widget, 1)
        self.audio = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        self.player.setSource(QUrl.fromLocalFile(video.path))
        controls = QHBoxLayout()
        self.play_button = QPushButton(tr("▶ Play"))
        self.play_button.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_button)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, int(video.duration * 1000)))
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(self.slider.setMaximum)
        self.player.positionChanged.connect(self.slider.setValue)
        controls.addWidget(self.slider, 1)
        self.position_label = QLabel(f"{time_label(second)} / {time_label(video.duration)}")
        self.player.positionChanged.connect(self._position_changed)
        controls.addWidget(self.position_label)
        root.addLayout(controls)
        heading = QLabel(tr("People in this video"))
        heading.setStyleSheet("font-weight: 700; font-size: 15px;")
        root.addWidget(heading)
        scroller = QScrollArea()
        scroller.setFixedHeight(126)
        scroller.setWidgetResizable(True)
        face_host = QWidget()
        self.face_row = QHBoxLayout(face_host)
        scroller.setWidget(face_host)
        root.addWidget(scroller)
        self.initial_position = max(0, int(second * 1000))
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self._render_faces()
        self.model.changed.connect(self._render_faces)

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia and self.initial_position > 0:
            self.player.setPosition(self.initial_position)
            self.initial_position = 0

    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_button.setText(tr("▶ Play"))
        else:
            self.player.play()
            self.play_button.setText(tr("Ⅱ Pause"))

    def _position_changed(self, milliseconds: int) -> None:
        self.position_label.setText(f"{time_label(milliseconds / 1000)} / {time_label(self.video.duration)}")

    def _render_faces(self) -> None:
        clear_layout(self.face_row)
        faces = sorted((face for face in self.model.faces if face.video_id == self.video.id and not face.excluded), key=lambda face: face.second)
        if not faces:
            message = tr("No faces were detected. You can run a detailed rescan.") if self.video.state == "ready" else tr("No people have been detected yet.")
            self.face_row.addWidget(QLabel(message))
        for face in faces:
            button = QPushButton(time_label(face.second))
            button.setIcon(image_icon(face.thumbnail_path, 72, 72))
            button.setIconSize(QSize(72, 72))
            button.setFixedSize(90, 92)
            button.clicked.connect(lambda checked=False, moment=face.second: self.player.setPosition(int(moment * 1000)))
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            button.customContextMenuRequested.connect(lambda position, current=face, item=button: face_menu(item, self.model, current).exec(item.mapToGlobal(position)))
            self.face_row.addWidget(button)
        self.face_row.addStretch()

    def closeEvent(self, event: object) -> None:
        self.player.stop()
        super().closeEvent(event)


class SettingsDialog(QDialog):
    """Edit persisted settings; the controller validates and enforces idle writes."""
    def __init__(self, model: LibraryController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Analysis settings")
        self.resize(660, 650)
        root = QVBoxLayout(self)
        warning = QLabel("Thresholds are uncalibrated. Automatic merging is off by default. "
                         "Only HIGH decisions are eligible; model disagreement requires review. "
                         "Folders removed here keep their indexed results.")
        warning.setWordWrap(True)
        root.addWidget(warning)
        self.fields: dict[str, QWidget] = {}
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        try:
            path = model.recognition_config_path
            options = RecognitionOptions.load(path if path.is_file() else None)
        except (OSError, ValueError, TypeError) as error:
            options = RecognitionOptions()
            warning.setText(f"Settings could not be loaded: {error}. Fix the paths/settings below.")
        values = dict(vars(options))
        if values["adaface_model_path"] is None:
            bundled_adaface = Path(__file__).resolve().parent / "resources" / "adaface_ir50_ms1mv2.onnx"
            if bundled_adaface.is_file():
                values["adaface_model_path"] = bundled_adaface
        definitions = {
            "Video": [("target_folders", "Target folders (one per line)", "folders", "\n".join(x.path for x in model.sources if getattr(x, "enabled", True))),
                      ("recursive", "Include subfolders", "bool", True),
                      ("extensions", "Extensions (comma separated)", "extensions", "mp4,mov,avi,mkv,m4v,webm"),
                      ("sample_interval", "Frame interval (seconds)", "float", .5),
                      ("face_sample_interval", "Interval while faces are present (seconds)", "float", .2)],
            "Detection / quality": [("detection_model_path", "SCRFD model path", "text", ""),
                      ("detection_threshold", "Detection confidence", "float", .5),
                      ("min_face_size", "Minimum face size (pixels)", "int", 80),
                      ("min_quality", "Minimum quality", "float", .45),
                      ("blur_threshold", "Minimum blur variance", "float", 20),
                      ("pose_threshold", "Maximum landmark pose asymmetry", "float", .48)],
            "Recognition": [("model_path", "JAPANESE FACE V1 path", "text", ""),
                      ("recognition_region", "Recognition region", "region", "full"),
                      ("adaface_enabled", "Enable AdaFace", "bool", False),
                      ("adaface_model_path", "AdaFace model path", "text", ""),
                      ("japanese_face_threshold", "Japanese Face HIGH median similarity", "float", .8),
                      ("adaface_threshold", "AdaFace HIGH median similarity", "float", .8),
                      ("japanese_face_weight", "Japanese Face score weight", "float", .7),
                      ("adaface_weight", "AdaFace score weight", "float", .3),
                      ("match_threshold", "Pair match similarity", "float", .65),
                      ("high_threshold", "HIGH similarity threshold", "float", .8),
                      ("medium_threshold", "MEDIUM similarity threshold", "float", .65),
                      ("min_samples", "Minimum samples per track", "int", 3),
                      ("max_samples", "Maximum samples per track", "int", 30),
                      ("auto_merge_enabled", "Enable HIGH automatic merging", "bool", False)],
            "Runtime / storage": [("acceleration", "Acceleration", "acceleration", "auto"),
                      ("cache_dir", "Cache directory (blank = default)", "text", ""),
                      ("data_root", "Data directory (restart may be required)", "text", str(getattr(options, "database_dir", None) or model.recognition_config_path.parent)),
                      ("log_level", "Log level", "loglevel", "INFO")],
        }
        self.kinds = {}
        for title, definitions_list in definitions.items():
            host = QWidget()
            page_layout = QVBoxLayout(host)
            form = QFormLayout()
            page_layout.addLayout(form)
            page_layout.addStretch()
            form.setFormAlignment(Qt.AlignTop)
            for key, label, kind, fallback in definitions_list:
                value = values.get(key, fallback)
                if kind == "bool":
                    control = QCheckBox(); control.setChecked(bool(value))
                elif kind in {"float", "int"}:
                    control = QDoubleSpinBox() if kind == "float" else QSpinBox()
                    control.setRange(0, 100000)
                    if kind == "float":
                        control.setDecimals(3); control.setSingleStep(.05)
                    control.setValue(float(value) if kind == "float" else int(value))
                elif kind == "region":
                    control = QComboBox()
                    control.addItem("Full face", "full")
                    control.addItem("Upper face (masks, experimental)", "upper")
                    control.setCurrentIndex(max(0, control.findData(value)))
                    control.setToolTip("Upper mode aligns using eyes only and excludes nose, mouth and jaw. Switching regions requires video reanalysis; caches and scores remain separate. Upper-face matches require review; automatic merging is unavailable until thresholds are calibrated.")
                elif kind in {"acceleration", "loglevel"}:
                    control = QComboBox()
                    control.addItems(["auto", "cpu", "cuda", "tensorrt"] if kind == "acceleration" else ["DEBUG", "INFO", "WARNING", "ERROR"])
                    control.setCurrentText(str(value))
                elif kind == "folders":
                    control = QTextEdit(); control.setPlainText(str(value)); control.setMaximumHeight(100)
                else:
                    control = QLineEdit(",".join(value) if kind == "extensions" and not isinstance(value, str) else str(value or ""))
                self.fields[key] = control
                self.kinds[key] = kind
                form.addRow(label, control)
            self.tabs.addTab(host, title)
        self.error_label = QLabel(); self.error_label.setWordWrap(True)
        root.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        buttons.button(QDialogButtonBox.Save).setEnabled(not model.is_scanning)

    def values(self) -> dict:
        result = {}
        for key, control in self.fields.items():
            kind = self.kinds[key]
            if kind == "bool": value = control.isChecked()
            elif kind in {"int", "float"}: value = control.value()
            elif kind == "region": value = control.currentData()
            elif kind in {"acceleration", "loglevel"}: value = control.currentText()
            elif kind == "folders": value = [line.strip() for line in control.toPlainText().splitlines() if line.strip()]
            elif kind == "extensions": value = ["." + item.strip().lower().lstrip(".") for item in control.text().split(",") if item.strip()]
            else:
                value = control.text().strip()
                if key in {"cache_dir", "adaface_model_path"} and not value:
                    value = None
            result[key] = value
        return result

    def _save(self) -> None:
        try:
            if self.model.save_settings(self.values()):
                self.accept()
            else:
                self.error_label.setText(self.model.status_text or "Settings were not saved. Pause analysis and check the values.")
        except (OSError, ValueError, TypeError) as error:
            self.error_label.setText(str(error))


class TextDialog(QDialog):
    def __init__(self, title: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 560)
        root = QVBoxLayout(self)
        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(text)
        root.addWidget(viewer)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        root.addWidget(close)


class MainWindow(QMainWindow):
    def __init__(self, model: LibraryController) -> None:
        super().__init__()
        load_language()
        self.model = model
        self.section = "all"
        self.source_id: str | None = None
        self.person_id: str | None = None
        self.show_unclassified = False
        self.person_tile_limit = 60
        self.visible_moments: dict[str, int] = {}
        self.review_pair: tuple[str, str] | None = None
        self.review_buttons: list[QPushButton] = []
        self.review_shortcuts = []
        for key, decision in (("S", "confirmed_same"), ("D", "confirmed_different"), ("L", "deferred")):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda current=decision: self._resolve_review(current))
            self.review_shortcuts.append(shortcut)
        self.setWindowTitle("VideoAtlas Python")
        self.resize(1120, 740)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)
        outer = QWidget()
        self.setCentralWidget(outer)
        columns = QHBoxLayout(outer)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(0)
        sidebar = self._build_sidebar()
        columns.addWidget(sidebar)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(30, 26, 30, 26)
        self.content_layout.setSpacing(18)
        self._build_header()
        self._build_status()
        self._build_acceleration_settings()
        self.scroller = QScrollArea()
        self.scroller.setWidgetResizable(True)
        self.list_host = QWidget()
        self.list_layout = QVBoxLayout(self.list_host)
        self.list_layout.setSizeConstraint(QLayout.SetMinimumSize)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.scroller.setWidget(self.list_host)
        self.content_layout.addWidget(self.scroller, 1)
        self.content_layout.addWidget(self.status_frame)
        columns.addWidget(content, 1)
        self.model.changed.connect(self._render)
        self.model.status_changed.connect(self._update_status)
        self._render()
        self._update_status()

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(245)
        self.side_layout = QVBoxLayout(sidebar)
        self.side_layout.setContentsMargins(18, 24, 18, 18)
        self.side_layout.setSpacing(7)
        brand = QLabel("◧  VideoAtlas")
        brand.setStyleSheet("font-size: 22px; font-weight: 800; color: white;")
        self.side_layout.addWidget(brand)
        self.caption = QLabel(tr("Local video library"))
        self.caption.setStyleSheet("color: #9eb4d4; font-size: 11px;")
        self.side_layout.addWidget(self.caption)
        self.side_layout.addSpacing(22)
        self.all_button = QPushButton(tr("▣  All videos"))
        self.all_button.setCheckable(True)
        self.all_button.clicked.connect(lambda: self._select_section("all"))
        self.side_layout.addWidget(self.all_button)
        self.people_button = QPushButton(tr("♙  People"))
        self.people_button.setCheckable(True)
        self.people_button.clicked.connect(lambda: self._select_section("people"))
        self.side_layout.addWidget(self.people_button)
        self.review_button = QPushButton("Review queue")
        self.review_button.setCheckable(True)
        self.review_button.clicked.connect(lambda: self._select_section("review"))
        self.side_layout.addWidget(self.review_button)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self._open_settings)
        self.side_layout.addWidget(self.settings_button)
        self.reanalyze_button = QPushButton("Reanalyze all videos")
        self.reanalyze_button.clicked.connect(lambda: self.model.reanalyze_all())
        self.side_layout.addWidget(self.reanalyze_button)
        self.regroup_button = QPushButton("Regroup cached tracks")
        self.regroup_button.clicked.connect(lambda: self.model.regroup_people())
        self.side_layout.addWidget(self.regroup_button)
        logs = QPushButton("Analysis logs")
        logs.clicked.connect(self._open_logs)
        self.side_layout.addWidget(logs)
        evaluation = QPushButton("Accuracy evaluation")
        evaluation.clicked.connect(self._open_evaluation)
        self.side_layout.addWidget(evaluation)
        self.folder_heading = QLabel(tr("Library folders"))
        self.folder_heading.setStyleSheet("color: #99afce; font-size: 11px; font-weight: 700; margin-top: 18px;")
        self.side_layout.addWidget(self.folder_heading)
        self.folder_layout = QVBoxLayout()
        self.side_layout.addLayout(self.folder_layout)
        self.add_button = QPushButton(tr("＋  Add folder"))
        self.add_button.setObjectName("AddFolder")
        self.add_button.clicked.connect(self._choose_folder)
        self.side_layout.addWidget(self.add_button)
        self.side_layout.addStretch()
        self.language_label = QLabel(tr("Language"))
        self.language_label.setStyleSheet("color: #99afce; font-size: 11px; font-weight: 700;")
        self.side_layout.addWidget(self.language_label)
        self.language_combo = QComboBox()
        self.language_combo.addItem(tr("English"), "en")
        self.language_combo.addItem(tr("Japanese"), "ja")
        self.language_combo.setCurrentIndex(0 if load_language() == "en" else 1)
        self.language_combo.currentIndexChanged.connect(self._change_language)
        self.side_layout.addWidget(self.language_combo)
        self.clear_button = QPushButton(tr("⌫  Clear index…"))
        self.clear_button.setObjectName("ClearIndex")
        self.clear_button.clicked.connect(self._confirm_clear)
        self.side_layout.addWidget(self.clear_button)
        return sidebar

    def _build_header(self) -> None:
        header = QHBoxLayout()
        self.title = QLabel(tr("All videos"))
        self.title.setStyleSheet("font-size: 29px; font-weight: 800;")
        header.addWidget(self.title)
        header.addStretch()
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Search video name or path"))
        self.search.setFixedWidth(225)
        self.search.textChanged.connect(self._render)
        header.addWidget(self.search)
        self.action_button = QPushButton(tr("Refresh folders"))
        self.action_button.setObjectName("Primary")
        self.action_button.clicked.connect(self._main_action)
        header.addWidget(self.action_button)
        self.content_layout.addLayout(header)

    def _build_status(self) -> None:
        self.status_frame = QFrame()
        self.status_frame.setObjectName("Status")
        status_layout = QVBoxLayout(self.status_frame)
        status_layout.setContentsMargins(14, 12, 14, 12)
        self.status_label = QLabel("")
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        status_layout.addWidget(self.progress_bar)
        # Status remains at the bottom of the main screen.


    def _build_acceleration_settings(self) -> None:
        self.configuration_error: str | None = None
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        choices = QHBoxLayout()
        self.acceleration_label = QLabel(tr("Face analysis acceleration"))
        choices.addWidget(self.acceleration_label)
        self.acceleration_combo = QComboBox()
        for text, value in (("Automatic", "auto"), ("TensorRT", "tensorrt"), ("CUDA", "cuda"), ("CPU", "cpu")):
            self.acceleration_combo.addItem(tr(text), value)
        choices.addWidget(self.acceleration_combo)
        self.precision_combo = QComboBox()
        self.precision_combo.addItem(tr("Accurate (FP32, TF32 off)"), "accurate")
        self.precision_combo.addItem(tr("Balanced (TensorRT FP16 / CUDA TF32)"), "balanced")
        choices.addWidget(self.precision_combo)
        choices.addStretch()
        layout.addLayout(choices)
        self.runtime_label = QLabel(tr("The active runtime appears when analysis starts."))
        self.runtime_label.setWordWrap(True)
        layout.addWidget(self.runtime_label)
        self.policy_label = QLabel()
        self.policy_label.setWordWrap(True)
        layout.addWidget(self.policy_label)
        self.content_layout.addWidget(panel)
        try:
            path = self.model.recognition_config_path
            options = RecognitionOptions.load(path if path.is_file() else None)
            self.acceleration_combo.setCurrentIndex(max(0, self.acceleration_combo.findData(options.acceleration)))
            self.precision_combo.setCurrentIndex(max(0, self.precision_combo.findData(options.precision)))
        except (OSError, ValueError, TypeError) as error:
            self.configuration_error = str(error)
        self.acceleration_combo.currentIndexChanged.connect(self._change_acceleration)
        self.precision_combo.currentIndexChanged.connect(self._change_precision)
        self._update_acceleration_policy()

    def _change_acceleration(self, index: int) -> None:
        value = self.acceleration_combo.itemData(index)
        if value in {"auto", "tensorrt", "cuda", "cpu"}:
            if not self.model.set_acceleration(value):
                self._restore_recognition_controls()
            else:
                self.configuration_error = None
            self._update_acceleration_policy()

    def _change_precision(self, index: int) -> None:
        value = self.precision_combo.itemData(index)
        if value in {"accurate", "balanced"}:
            if not self.model.set_precision(value):
                self._restore_recognition_controls()
            else:
                self.configuration_error = None
            self._update_acceleration_policy()

    def _restore_recognition_controls(self) -> None:
        try:
            path = self.model.recognition_config_path
            options = RecognitionOptions.load(path if path.is_file() else None)
            self.configuration_error = None
        except (OSError, ValueError, TypeError) as error:
            self.configuration_error = str(error)
            options = RecognitionOptions()
        for control, value in ((self.acceleration_combo, options.acceleration), (self.precision_combo, options.precision)):
            control.blockSignals(True)
            control.setCurrentIndex(max(0, control.findData(value)))
            control.blockSignals(False)

    def _update_acceleration_policy(self) -> None:
        accurate = self.precision_combo.currentData() == "accurate"
        self.acceleration_combo.setItemText(0, tr("Auto (CUDA FP32 → CPU)" if accurate else "Auto (TensorRT FP16 → CUDA TF32 → CPU)"))
        if self.acceleration_combo.currentData() == "cpu":
            policy = "CPU FP32."
        elif accurate:
            policy = "Accurate: CUDA FP32 with TF32 disabled, then CPU. TensorRT requests use this fallback."
        else:
            policy = "Balanced: TensorRT FP16, then CUDA with TF32 enabled, then CPU. CUDA requests skip TensorRT."
        self.policy_label.setText(tr(policy))
        self.runtime_label.setText(self.model.runtime_description or self.configuration_error or tr("The active runtime appears when analysis starts."))

    def _select_section(self, section: str, source_id: str | None = None) -> None:
        self.section = section
        self.source_id = source_id
        self._render()

    def _change_language(self, index: int) -> None:
        language = self.language_combo.itemData(index)
        if language not in {"ja", "en"}:
            return
        set_language(language)
        self.caption.setText(tr("Local video library"))
        self.all_button.setText(tr("▣  All videos"))
        self.people_button.setText(tr("♙  People"))
        self.folder_heading.setText(tr("Library folders"))
        self.add_button.setText(tr("＋  Add folder"))
        self.clear_button.setText(tr("⌫  Clear index…"))
        self.language_label.setText(tr("Language"))
        self.language_combo.setItemText(0, tr("English"))
        self.language_combo.setItemText(1, tr("Japanese"))
        self.search.setPlaceholderText(tr("Search video name or path"))
        if hasattr(self, "acceleration_combo"):
            self.acceleration_label.setText(tr("Face analysis acceleration"))
            self.precision_combo.setItemText(0, tr("Accurate (FP32, TF32 off)"))
            self.precision_combo.setItemText(1, tr("Balanced (TensorRT FP16 / CUDA TF32)"))
            self._update_acceleration_policy()
        self._render()

    def _render(self) -> None:
        clear_layout(self.folder_layout)
        for source in self.model.sources:
            button = QPushButton(f"▱  {Path(source.path).name}" + (" (inactive)" if not getattr(source, "enabled", True) else ""))
            button.setToolTip(source.path)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, identifier=source.id: self._select_section("folder", identifier))
            button.setChecked(self.section == "folder" and self.source_id == source.id)
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            button.customContextMenuRequested.connect(lambda point, identifier=source.id, item=button: self._folder_menu(item, point, identifier))
            self.folder_layout.addWidget(button)
        self.all_button.setChecked(self.section == "all")
        self.people_button.setChecked(self.section == "people")
        self.clear_button.setEnabled(not self.model.is_scanning)
        clear_layout(self.list_layout)
        self.review_pair = None
        self.review_buttons = []
        self.review_button.setChecked(self.section == "review")
        self.review_button.setText(f"Review queue ({len(getattr(self.model, 'pending_reviews', []))})")
        for shortcut in self.review_shortcuts:
            shortcut.setEnabled(self.section == "review" and not self.model.is_scanning)
        if self.section == "review":
            self.title.setText("Review queue")
            self.search.hide()
            self._render_reviews()
        elif self.section == "people":
            self.title.setText(tr("People"))
            self.search.hide()
            self._render_people()
        else:
            self.search.show()
            source = next((item for item in self.model.sources if item.id == self.source_id), None)
            self.title.setText(Path(source.path).name if self.section == "folder" and source is not None else tr("All videos"))
            self._render_videos()
        self._update_status()

    def _render_videos(self) -> None:
        needle = self.search.text().strip().casefold()
        videos = [video for video in self.model.videos if (self.section != "folder" or video.source_id == self.source_id) and (not needle or needle in video.name.casefold() or needle in video.path.casefold())]
        if not videos:
            title = tr("No results found") if needle else tr("No videos yet")
            description = tr("Try a different search term.") if needle else tr("Add a folder or drag a video folder here.")
            self._show_empty(title, description, not needle)
            return
        grid = QGridLayout()
        grid.setSpacing(17)
        for index, video in enumerate(sorted(videos, key=lambda item: item.name.casefold())):
            card = self._video_card(video)
            grid.addWidget(card, index // 3, index % 3)
        grid.setColumnStretch(3, 1)
        self.list_layout.addLayout(grid)
        self.list_layout.addStretch()

    def _video_card(self, video: VideoRecord) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card.setFixedWidth(235)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        picture = QLabel("▶")
        picture.setFixedSize(210, 118)
        picture.setStyleSheet("background: #dfe8f4; border-radius: 10px; color: #5c7fae; font-size: 34px;")
        picture.setAlignment(Qt.AlignCenter)
        if video.poster_path and Path(video.poster_path).is_file():
            pixmap = QPixmap(video.poster_path)
            if not pixmap.isNull():
                picture.setPixmap(pixmap.scaled(210, 118, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        layout.addWidget(picture)
        name = QLabel(video.name)
        name.setWordWrap(True)
        name.setStyleSheet("font-weight: 700;")
        layout.addWidget(name)
        state_names = {"pending": tr("Pending"), "analyzing": tr("Analyzing"), "ready": tr("Ready"), "missing": tr("File missing"), "failed": tr("Failed")}
        meta = QLabel(f"{time_label(video.duration)}   ·   {state_names.get(video.state, video.state)}")
        person_count = len({face.person_id for face in self.model.faces if face.video_id == video.id and face.person_id and not face.excluded})
        layout.addWidget(QLabel(f"{person_count} people · Analysis time: {getattr(video, 'analysis_seconds', 0):.1f}s\nAnalyzed: {getattr(video, 'analyzed_at', None) or 'Not recorded'}"))
        meta.setToolTip(video.path)
        meta.setStyleSheet("color: #718198; font-size: 11px;")
        layout.addWidget(meta)
        if video.error_message:
            reason = QLabel(localize_message(video.error_message))
            reason.setWordWrap(True)
            reason.setStyleSheet("color: #a4512b; font-size: 11px;")
            layout.addWidget(reason)
        layout.addStretch()
        open_button = QPushButton(tr("Open video"))
        open_button.clicked.connect(lambda checked=False, item=video: self._open_player(item))
        open_button.setEnabled(video.state != "missing" and Path(video.path).is_file())
        layout.addWidget(open_button)
        return card

    def _render_people(self) -> None:
        panes = QSplitter(Qt.Horizontal)
        people_host = QWidget()
        people_layout = QVBoxLayout(people_host)
        people_layout.setContentsMargins(0, 0, 0, 0)
        tracks = getattr(self.model, "tracks", [])
        people = visible_people(self.model)
        for person in people[:self.person_tile_limit]:
            tile = QFrame()
            tile.setObjectName("PersonTile")
            tile.setStyleSheet("QFrame#PersonTile { background: white; border: 2px solid %s; border-radius: 12px; }" % ("#2b79ca" if person.id == self.person_id else "#e0e7f1"))
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(8, 8, 8, 8)
            face_button = QPushButton()
            face_button.setObjectName(f"person-face-{person.id}")
            face_button.setIcon(image_icon(person_thumbnail_path(self.model, person), 132, 132))
            face_button.setIconSize(QSize(132, 132))
            face_button.setFixedSize(144, 144)
            face_button.setToolTip(person_tooltip(self.model, person))
            face_button.clicked.connect(lambda checked=False, identifier=person.id: self._select_person(identifier))
            face_button.setContextMenuPolicy(Qt.CustomContextMenu)
            face_button.customContextMenuRequested.connect(lambda point, current=person, item=face_button: self._person_menu(item, point, current))
            tile_layout.addWidget(face_button, 0, Qt.AlignHCenter)
            if person.name.strip():
                name = QLabel(person.name.strip())
                name.setAlignment(Qt.AlignCenter)
                name.setWordWrap(True)
                name.setObjectName("AssignedPersonName")
                tile_layout.addWidget(name)
            summary = QLabel(person_summary(self.model, person))
            summary.setAlignment(Qt.AlignCenter)
            tile_layout.addWidget(summary)
            people_layout.addWidget(tile)
        if len(people) > self.person_tile_limit:
            more_people = QPushButton(tr("Show 60 more people"))
            more_people.setObjectName("ShowMorePeople")
            more_people.clicked.connect(self._more_people)
            people_layout.addWidget(more_people)
        unknown = QPushButton("Unclassified faces")
        unknown.clicked.connect(self._select_unclassified)
        people_layout.addWidget(unknown)
        people_layout.addStretch()
        panes.addWidget(person_pane_scroll(people_host, "PeoplePaneScrollLeft"))
        details_host = QWidget()
        details_layout = QVBoxLayout(details_host)
        videos_host = QWidget()
        videos_layout = QVBoxLayout(videos_host)
        panes.addWidget(person_pane_scroll(details_host, "PeoplePaneScrollDetails"))
        panes.addWidget(person_pane_scroll(videos_host, "PeoplePaneScrollVideos"))
        panes.setStretchFactor(1, 2); panes.setStretchFactor(2, 1)
        self.list_layout.addWidget(panes)
        original = self.list_layout
        self.list_layout = details_layout
        self._render_person_results()
        details_layout.addStretch()
        self.list_layout = original
        videos_layout.addWidget(QLabel("Videos featuring this person"))
        selected_faces = [face for face in self.model.faces if not face.excluded and face.person_id == self.person_id]
        video_ids = {face.video_id for face in selected_faces}
        video_ids.update(track.video_id for track in tracks if track.person_id == self.person_id)
        for video in self.model.videos:
            if video.id in video_ids:
                button = QPushButton(f"{video.name}\n{time_label(video.duration)} · {video.state}")
                button.setToolTip(video.path)
                button.clicked.connect(lambda checked=False, item=video: self._open_player(item))
                videos_layout.addWidget(button)
                recalc = QPushButton("Recalculate embeddings")
                recalc.setEnabled(not self.model.is_scanning)
                recalc.clicked.connect(lambda checked=False, identifier=video.id: self.model.recalculate_embeddings(identifier))
                videos_layout.addWidget(recalc)
        videos_layout.addStretch()

    def _select_person(self, person_id: str) -> None:
        self.person_id = person_id
        self.show_unclassified = False
        self.visible_moments = {}
        self._render()

    def _select_unclassified(self) -> None:
        self.person_id = None
        self.show_unclassified = True
        self.visible_moments = {}
        self._render()

    def _render_person_results(self) -> None:
        person = next((item for item in self.model.people if item.id == self.person_id), None)
        if self.show_unclassified:
            heading = QLabel(tr("Unclassified faces"))
            heading.setStyleSheet("font-size: 20px; font-weight: 700; margin-top: 18px;")
            self.list_layout.addWidget(heading)
        if person:
            if person.name.strip():
                heading = QLabel(person.name.strip())
                heading.setStyleSheet("font-size: 20px; font-weight: 700; margin-top: 18px;")
                self.list_layout.addWidget(heading)
            name_row = QHBoxLayout()
            self.person_name_edit = QLineEdit(person.name.strip())
            self.person_name_edit.setObjectName("PersonNameEdit")
            self.person_name_edit.setPlaceholderText("Optional name")
            self.person_name_edit.setMaximumWidth(280)
            self.person_name_save = QPushButton("Save name")
            self.person_name_save.setEnabled(not self.model.is_scanning)
            self.person_name_save.clicked.connect(lambda checked=False, identifier=person.id: self._save_person_name(identifier))
            name_row.addWidget(self.person_name_edit)
            name_row.addWidget(self.person_name_save)
            name_row.addStretch()
            self.list_layout.addLayout(name_row)
            thumbnail = person_thumbnail_path(self.model, person)
            if thumbnail and Path(thumbnail).is_file():
                representative = QLabel()
                representative.setPixmap(QPixmap(thumbnail).scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.list_layout.addWidget(representative)
        own_tracks = sorted((track for track in getattr(self.model, "tracks", [])
                             if not self.show_unclassified and track.person_id == self.person_id),
                            key=lambda track: (track.start_time, track.end_time, track.video_id))
        reference_track = None
        if own_tracks:
            representative_face_id = getattr(person, "representative_face_id", None) if person else None
            reference_track = next((track for track in own_tracks
                                    if representative_face_id and (track.representative_face_id == representative_face_id
                                                                   or representative_face_id in track.face_ids)),
                                   own_tracks[0])
        for track_number, track in enumerate(own_tracks, start=1):
            face_count = len(getattr(track, "face_ids", getattr(track, "sample_face_ids", [])))
            text = QLabel(f"Track {track_number} · {time_label(track.start_time)}–{time_label(track.end_time)} · "
                          f"{face_count} faces · quality {getattr(track, 'quality', 0):.3f}")
            text.setObjectName("PersonTrackSummary")
            text.setWordWrap(True)
            self.list_layout.addWidget(text)
            if track is reference_track:
                similarity_text = QLabel("Selected representative track")
            else:
                similarity_text = QLabel(track_similarity_summary(reference_track, track))
            similarity_text.setObjectName("TrackSimilaritySummary")
            similarity_text.setWordWrap(True)
            self.list_layout.addWidget(similarity_text)
            split = QPushButton("Split this track into a new person")
            split.setEnabled(not self.model.is_scanning)
            split.clicked.connect(lambda checked=False, identifier=track.id: self.model.split_track(identifier))
            self.list_layout.addWidget(split)
        matches = [face for face in self.model.faces if not face.excluded and ((self.show_unclassified and face.person_id is None) or (not self.show_unclassified and face.person_id == self.person_id))]
        if not matches:
            self.list_layout.addWidget(QLabel(tr("No matching detection times.")))
            return
        for video in sorted(self.model.videos, key=lambda item: item.name.casefold()):
            moments = sorted((face for face in matches if face.video_id == video.id), key=lambda face: face.second)
            if not moments:
                continue
            video_heading = QLabel(tr("{name}  ·  {count} detections", name=video.name, count=len(moments)))
            video_heading.setStyleSheet("font-weight: 700; margin-top: 8px;")
            self.list_layout.addWidget(video_heading)
            limit = self.visible_moments.get(video.id, 60)
            row = QGridLayout()
            for index, face in enumerate(moments[:limit]):
                button = QPushButton(time_label(face.second))
                button.setToolTip(f"{video.name} · {time_label(face.second)}\nQuality: {getattr(face, 'quality', None)} · {getattr(face, 'rejection_reason', None) or 'accepted'}")
                button.setIcon(image_icon(face.thumbnail_path, 46, 46))
                button.setIconSize(QSize(46, 46))
                button.clicked.connect(lambda checked=False, item=video, second=face.second: self._open_player(item, second))
                button.setContextMenuPolicy(Qt.CustomContextMenu)
                button.customContextMenuRequested.connect(lambda point, current=face, item=button: face_menu(item, self.model, current).exec(item.mapToGlobal(point)))
                row.addWidget(button, index // 4, index % 4)
            self.list_layout.addLayout(row)
            if len(moments) > limit:
                more = QPushButton(tr("Show 60 more"))
                more.clicked.connect(lambda checked=False, identifier=video.id: self._more_moments(identifier))
                self.list_layout.addWidget(more)

    def _save_person_name(self, person_id: str) -> None:
        if self.model.is_scanning or not hasattr(self, "person_name_edit"):
            return
        value = self.person_name_edit.text().strip()
        self.model.rename_person(person_id, value)

    def _more_moments(self, video_id: str) -> None:
        self.visible_moments[video_id] = self.visible_moments.get(video_id, 60) + 60
        self._render()

    def _show_empty(self, title: str, description: str, show_add: bool) -> None:
        panel = QFrame()
        panel.setObjectName("Empty")
        panel.setMinimumHeight(270)
        layout = QVBoxLayout(panel)
        layout.addStretch()
        symbol = QLabel("▣")
        symbol.setStyleSheet("font-size: 38px; color: #7da3d1;")
        symbol.setAlignment(Qt.AlignCenter)
        layout.addWidget(symbol)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        heading.setAlignment(Qt.AlignCenter)
        layout.addWidget(heading)
        details = QLabel(description)
        details.setStyleSheet("color: #6d7d92;")
        details.setAlignment(Qt.AlignCenter)
        layout.addWidget(details)
        if show_add:
            action = QPushButton(tr("Add folder"))
            action.setObjectName("Primary")
            action.setFixedWidth(150)
            action.clicked.connect(self._choose_folder)
            layout.addWidget(action, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.list_layout.addWidget(panel)

    def _choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, tr("Choose a video folder"))
        if chosen:
            self.model.add_folder(Path(chosen))

    def _main_action(self) -> None:
        if self.model.is_scanning:
            self.model.pause_scanning()
        else:
            self.model.refresh_library()

    def _update_status(self) -> None:
        self.status_frame.setVisible(self.model.is_scanning or bool(self.model.status_text))
        self.status_label.setText(localize_message(self.model.status_text))
        self.progress_bar.setValue(int(self.model.progress * 100))
        self.action_button.setText(tr("Pause") if self.model.is_scanning else tr("Refresh folders"))
        self.clear_button.setEnabled(not self.model.is_scanning)
        self.settings_button.setEnabled(not self.model.is_scanning)
        self.reanalyze_button.setEnabled(not self.model.is_scanning)
        self.regroup_button.setEnabled(not self.model.is_scanning)
        for button in self.review_buttons:
            button.setEnabled(not self.model.is_scanning)
        for shortcut in self.review_shortcuts:
            shortcut.setEnabled(self.section == "review" and not self.model.is_scanning)
        if hasattr(self, "acceleration_combo"):
            self.acceleration_combo.setEnabled(not self.model.is_scanning)
            self.precision_combo.setEnabled(not self.model.is_scanning)
            if self.model.runtime_description:
                self.runtime_label.setText(localize_message(self.model.runtime_description))

    def _render_reviews(self) -> None:
        queue = list(getattr(self.model, "pending_reviews", []))
        self.review_button.setText(f"Review queue ({len(queue)})")
        if not queue:
            self._show_empty("No pending reviews", "Uncertain track pairs will appear here after analysis.", False)
            return
        relation = queue[0]
        self.review_pair = (relation.track_a, relation.track_b)
        self.list_layout.addWidget(QLabel(f"{len(queue)} reviews remaining · {relation.relation}"))
        pair = QHBoxLayout()
        for track_id in self.review_pair:
            track = next((item for item in getattr(self.model, "tracks", []) if item.id == track_id), None)
            panel = QFrame(); panel.setObjectName("Card")
            layout = QVBoxLayout(panel)
            if track is None:
                layout.addWidget(QLabel(f"Track {track_id} is unavailable. Refresh the library."))
            else:
                person = next((item for item in self.model.people if item.id == track.person_id), None)
                if person and person.name.strip():
                    layout.addWidget(QLabel(person.name.strip()))
                face = next((item for item in self.model.faces if item.id == track.representative_face_id), None)
                picture = QLabel("Representative face unavailable")
                picture.setFixedHeight(120)
                picture.setAlignment(Qt.AlignCenter)
                if face and Path(face.thumbnail_path).is_file():
                    pixmap = QPixmap(face.thumbnail_path)
                    if not pixmap.isNull(): picture.setPixmap(pixmap.scaled(112, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                layout.addWidget(picture)
                video = next((item for item in self.model.videos if item.id == track.video_id), None)
                label = QLabel(f"Track {track.id}\n{video.name if video else track.video_id}\n{time_label(track.start_time)}–{time_label(track.end_time)}")
                label.setWordWrap(True)
                if video: label.setToolTip(video.path)
                layout.addWidget(label)
                if video:
                    play = QPushButton("Play appearance")
                    play.clicked.connect(lambda checked=False, item=video, second=track.start_time: self._open_player(item, second))
                    layout.addWidget(play)
            pair.addWidget(panel)
        self.list_layout.addLayout(pair)
        details = relation.details or {}
        confidence = details.get("confidence", details.get("decision", details.get("level", "Not recorded")))
        reason = details.get("reason") or "; ".join(details.get("reasons", [])) or "No reason recorded"
        decision_label = QLabel(f"Overall decision: {confidence}\nReason: {reason}")
        decision_label.setWordWrap(True)
        self.list_layout.addWidget(decision_label)
        scores = details.get("model_scores", details.get("models", details.get("scores", {})))
        summaries = []
        for model_key, evidence in scores.items():
            if isinstance(evidence, dict):
                median = evidence.get("median")
                median_text = f"{median:.3f}" if isinstance(median, (float, int)) else "unavailable"
                summaries.append(f"{model_key}\nMedian similarity: {median_text} | "
                                 f"Samples: {evidence.get('sample_count_a', '?')} / {evidence.get('sample_count_b', '?')} | "
                                 f"Pair comparisons: {evidence.get('pair_count', '?')}\n"
                                 f"Match rate: {evidence.get('match_rate', 'unavailable')} | "
                                 f"Lower-end similarity: {evidence.get('lower_quantile', 'unavailable')}")
            else:
                summaries.append(f"{model_key}: {evidence}")
        if not summaries:
            summaries.append("Japanese Face score: unavailable\nAdaFace score: unavailable")
        overall_score = details.get("score")
        if isinstance(overall_score, (float, int)):
            summaries.insert(0, f"Overall similarity: {overall_score:.3f}")
        score_label = QLabel("Per-model evidence (cosine similarity):\n" + "\n\n".join(summaries))
        score_label.setWordWrap(True); score_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.list_layout.addWidget(score_label)
        evidence = QTextEdit(); evidence.setReadOnly(True)
        evidence.setPlainText(json.dumps(details, indent=2, default=str))
        evidence.setMaximumHeight(70)
        self.list_layout.addWidget(evidence)
        actions = QHBoxLayout()
        for label, decision in (("Same person (S)", "confirmed_same"), ("Different people (D)", "confirmed_different"), ("Review later (L)", "deferred")):
            button = QPushButton(label)
            button.setEnabled(not self.model.is_scanning)
            button.clicked.connect(lambda checked=False, value=decision: self._resolve_review(value))
            self.review_buttons.append(button); actions.addWidget(button)
        self.list_layout.insertLayout(1, actions)
        self.list_layout.addStretch()

    def _resolve_review(self, decision: str) -> None:
        if self.section == "review" and self.review_pair and not self.model.is_scanning:
            self.model.resolve_review(*self.review_pair, decision)

    def _open_settings(self) -> None:
        SettingsDialog(self.model, self).exec()
        self._restore_recognition_controls()
        self._render()

    def _open_logs(self) -> None:
        path = Path(self.model.log_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[-200000:] if path.is_file() else "No analysis log has been created yet."
        except OSError as error:
            text = f"Cannot read {path}: {error}"
        TextDialog("Analysis logs", f"{path}\n\n{text}", self).exec()

    def _open_evaluation(self) -> None:
        report = self.model.evaluation_report()
        text = ("Evaluation unit: reviewed track pairs.\n"
                "False-positive rate = FP / (FP + TN).\n"
                "False merges among predicted matches = FP / (TP + FP).\n"
                "Precision = TP / (TP + FP); recall = TP / (TP + FN).\n"
                "Threshold candidates are advisory and do not activate tuning.\n\n"
                + json.dumps(report, indent=2, default=str))
        TextDialog("Accuracy evaluation", text, self).exec()

    def _folder_menu(self, button: QPushButton, point: object, source_id: str) -> None:
        if self.model.is_scanning:
            return
        menu = QMenu(button)
        remove = menu.addAction(tr("Remove this folder"))
        remove.triggered.connect(lambda: self._confirm_remove_source(source_id))
        menu.exec(button.mapToGlobal(point))

    def _confirm_remove_source(self, source_id: str) -> None:
        answer = confirm_deletion(self, tr("Remove folder"), tr("Delete this folder's index and generated images? Original videos will remain."))
        if answer:
            self.model.remove_source(source_id)
            self._select_section("all")

    def _person_menu(self, button: QPushButton, point: object, person: PersonRecord) -> None:
        if self.model.is_scanning:
            return
        menu = QMenu(button)
        rename = menu.addAction(tr("Name or rename"))
        rename.triggered.connect(lambda: self._rename_person(person))
        merge = menu.addMenu(tr("Merge into another person"))
        for candidate in visible_people(self.model):
            if candidate.id != person.id:
                action = merge.addAction(image_icon(person_thumbnail_path(self.model, candidate), 32, 32), candidate.name.strip() or person_summary(self.model, candidate))
                action.setToolTip(person_tooltip(self.model, candidate))
                action.triggered.connect(lambda checked=False, target=candidate.id: self.model.merge_people(person.id, target))
        menu.exec(button.mapToGlobal(point))

    def _rename_person(self, person: PersonRecord) -> None:
        self.person_id = person.id
        self.show_unclassified = False
        self.section = "people"
        self._render()

    def _more_people(self) -> None:
        self.person_tile_limit += 60
        self._render()

    def _open_player(self, video: VideoRecord, second: float = 0.0) -> None:
        if video.state == "missing" or not Path(video.path).is_file():
            show_warning(self, tr("Cannot open video"), tr("The original video is missing. Refresh the folders."))
            return
        dialog = PlayerDialog(self.model, video, second, self)
        dialog.exec()

    def _confirm_clear(self) -> None:
        answer = confirm_deletion(self, tr("Clear index"), tr("Delete registered folders, analysis results, and generated face images? Original videos will remain."))
        if answer:
            self.model.clear_index()
            self._select_section("all")

    def dragEnterEvent(self, event: object) -> None:
        if any(Path(url.toLocalFile()).is_dir() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: object) -> None:
        folders = [Path(url.toLocalFile()) for url in event.mimeData().urls() if Path(url.toLocalFile()).is_dir()]
        for folder in folders:
            self.model.add_folder(folder)
        event.acceptProposedAction()

    def closeEvent(self, event: object) -> None:
        self.model.close()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="VideoAtlas Python")
    parser.add_argument("--data-dir", type=Path, default=None, help=tr("Index storage directory for the Python version"))
    arguments, qt_arguments = parser.parse_known_args()
    application = QApplication([sys.argv[0], *qt_arguments])
    model = LibraryController(LibraryController.resolve_data_root(arguments.data_dir))
    window = MainWindow(model)
    window.show()
    return application.exec()
