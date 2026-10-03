import argparse
from pathlib import Path
import sys
from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget
from .controller import LibraryController
from .localization import load_language, localize_message, set_language, tr
from .storage import FaceRecord, PersonRecord, VideoRecord
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
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def face_menu(parent: QWidget, model: LibraryController, face: FaceRecord) -> QMenu:
    menu = QMenu(parent)
    split = menu.addAction(tr("Split into a new person"))
    split.triggered.connect(lambda: model.split_face(face.id))
    unclassify = menu.addAction(tr("Move to unclassified"))
    unclassify.triggered.connect(lambda: model.assign_face(face.id, None))
    move = menu.addMenu(tr("Move to another person"))
    for person in model.people:
        if person.id != face.person_id:
            label = person.name or tr("Unnamed person")
            action = move.addAction(label)
            action.triggered.connect(lambda checked=False, target=person.id: model.assign_face(face.id, target))
    exclude = menu.addAction(tr("Exclude this face"))
    exclude.triggered.connect(lambda: model.exclude_face(face.id))
    menu.setEnabled(not model.is_scanning)
    return menu


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


class MainWindow(QMainWindow):
    def __init__(self, model: LibraryController) -> None:
        super().__init__()
        load_language()
        self.model = model
        self.section = "all"
        self.source_id: str | None = None
        self.person_id: str | None = None
        self.show_unclassified = False
        self.visible_moments: dict[str, int] = {}
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
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.scroller.setWidget(self.list_host)
        self.content_layout.addWidget(self.scroller, 1)
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
        self.content_layout.addWidget(self.status_frame)

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
            button = QPushButton(f"▱  {Path(source.path).name}")
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
        if self.section == "people":
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
        open_button.setEnabled(Path(video.path).is_file())
        layout.addWidget(open_button)
        return card

    def _render_people(self) -> None:
        if any(video.state in {"pending", "analyzing"} for video in self.model.videos):
            note = QLabel(tr("Some videos are being analyzed or are pending. Results will appear here as they become available."))
            note.setStyleSheet("color: #718198;")
            self.list_layout.addWidget(note)
        grid = QGridLayout()
        grid.setSpacing(12)
        for index, person in enumerate(self.model.people):
            tile = QPushButton(person.name or tr("Face photo"))
            tile.setIcon(image_icon(person.thumbnail_path, 72, 72))
            tile.setIconSize(QSize(72, 72))
            tile.setFixedSize(190, 95)
            tile.clicked.connect(lambda checked=False, identifier=person.id: self._select_person(identifier))
            tile.setContextMenuPolicy(Qt.CustomContextMenu)
            tile.customContextMenuRequested.connect(lambda point, current=person, item=tile: self._person_menu(item, point, current))
            grid.addWidget(tile, index // 4, index % 4)
        unclassified_count = sum(face.person_id is None and not face.excluded for face in self.model.faces)
        unknown = QPushButton(tr("Unclassified  {count}", count=unclassified_count))
        unknown.setFixedSize(190, 95)
        unknown.clicked.connect(self._select_unclassified)
        grid.addWidget(unknown, len(self.model.people) // 4, len(self.model.people) % 4)
        self.list_layout.addLayout(grid)
        if not self.model.people and unclassified_count == 0:
            self._show_empty(tr("No people found yet"), tr("Add a video folder to find detected faces here."), True)
        elif self.person_id is not None or self.show_unclassified:
            self._render_person_results()
        self.list_layout.addStretch()

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
        heading_text = tr("Unclassified faces") if self.show_unclassified else (person.name or tr("Selected person")) if person is not None else tr("Selected person")
        heading = QLabel(heading_text)
        heading.setStyleSheet("font-size: 20px; font-weight: 700; margin-top: 18px;")
        self.list_layout.addWidget(heading)
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
        if hasattr(self, "acceleration_combo"):
            self.acceleration_combo.setEnabled(not self.model.is_scanning)
            self.precision_combo.setEnabled(not self.model.is_scanning)
            if self.model.runtime_description:
                self.runtime_label.setText(localize_message(self.model.runtime_description))

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
        for candidate in self.model.people:
            if candidate.id != person.id:
                action = merge.addAction(candidate.name or tr("Unnamed person"))
                action.triggered.connect(lambda checked=False, target=candidate.id: self.model.merge_people(person.id, target))
        menu.exec(button.mapToGlobal(point))

    def _rename_person(self, person: PersonRecord) -> None:
        value, accepted = QInputDialog.getText(self, tr("Person name"), tr("Enter a display name"), text=person.name)
        if accepted and value.strip():
            self.model.rename_person(person.id, value)

    def _open_player(self, video: VideoRecord, second: float = 0.0) -> None:
        if not Path(video.path).is_file():
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
    model = LibraryController(arguments.data_dir)
    window = MainWindow(model)
    window.show()
    return application.exec()
