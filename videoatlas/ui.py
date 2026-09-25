# このファイルはPython版VideoAtlasのmacOS向け画面を定義します。
"""PySide6 で作る VideoAtlas のライブラリ画面。"""

# 起動時の保存先指定を読み取ります。
import argparse
# 登録するフォルダや画像のパスを扱います。
from pathlib import Path
# 実行引数をQtへ渡します。
import sys

# Qt の座標、サイズ、URLと定数を読み込みます。
from PySide6.QtCore import Qt, QSize, QUrl
# 画像表示に使うQt型を読み込みます。
from PySide6.QtGui import QIcon, QPixmap
# 動画の音声と再生を担当するQt部品を読み込みます。
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
# 再生画面に映像を表示するQt部品を読み込みます。
from PySide6.QtMultimediaWidgets import QVideoWidget
# ボタン、レイアウト、ダイアログをまとめて読み込みます。
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget

# 画面の操作を保存と解析へつなぐモデルを読み込みます。
from .controller import LibraryController
# 画面文言と言語の保存処理を読み込みます。
from .localization import load_language, localize_message, set_language, tr
# 顔、人物、動画の型を画面表示に使います。
from .storage import FaceRecord, PersonRecord, VideoRecord


# 画面全体の色、余白、文字を一箇所で定義します。
STYLE = """
/* 画面全体の背景、文字色、和文フォントを指定します。 */
QWidget { background: #f5f7fb; color: #17243a; font-family: 'Hiragino Sans', 'Helvetica Neue'; font-size: 13px; }
/* 左の案内欄を濃紺で表示します。 */
QFrame#Sidebar { background: #15233b; border: none; }
/* 案内欄内の説明文を明るくします。 */
QFrame#Sidebar QLabel { background: transparent; color: #e8f0ff; }
/* 案内先ボタンの通常状態を設定します。 */
QFrame#Sidebar QPushButton { background: transparent; color: #c8d5e9; text-align: left; border: none; border-radius: 9px; padding: 10px 12px; }
/* マウスを合わせた案内先を強調します。 */
QFrame#Sidebar QPushButton:hover { background: #233b5e; }
/* 現在選択中の案内先を表示します。 */
QFrame#Sidebar QPushButton:checked { background: #315687; color: white; font-weight: 700; }
/* フォルダ追加の主要操作に青色を使います。 */
QFrame#Sidebar QPushButton#AddFolder { background: #2b79ca; color: white; text-align: center; font-weight: 700; }
/* 索引消去の操作を控えめな色にします。 */
QFrame#Sidebar QPushButton#ClearIndex { color: #aebdd2; }
/* 一般の操作ボタンの形をそろえます。 */
QPushButton { background: #e8eef8; border: 1px solid #d6e0ef; border-radius: 9px; padding: 8px 13px; }
/* ボタンにマウスを乗せたことを示します。 */
QPushButton:hover { background: #dae7f8; }
/* 実行できない操作を淡色で表示します。 */
QPushButton:disabled { color: #929eac; background: #eef1f5; }
/* 重要な操作ボタンを青色にします。 */
QPushButton#Primary { background: #226fc1; color: white; border: none; font-weight: 700; }
/* 重要な操作へマウスを乗せたことを示します。 */
QPushButton#Primary:hover { background: #185b9f; }
/* 検索文字を入力する欄の枠を作ります。 */
QLineEdit { background: white; border: 1px solid #d4deec; border-radius: 9px; padding: 8px 12px; }
/* 動画カードと案内カードに白い背景を付けます。 */
QFrame#Card, QFrame#Empty, QFrame#Status { background: white; border: 1px solid #e0e7f1; border-radius: 15px; }
/* カード内の文字が背景色の帯を作らないよう透明にします。 */
QFrame#Card QLabel, QFrame#Empty QLabel, QFrame#Status QLabel { background: transparent; }
/* 一覧をスクロールする枠線を隠します。 */
QScrollArea { border: none; background: transparent; }
/* スクロール領域の内部も背景になじませます。 */
QScrollArea > QWidget > QWidget { background: transparent; }
/* 進捗バーの後ろ側を淡い青にします。 */
QProgressBar { background: #e8eff8; border: none; border-radius: 5px; height: 8px; text-align: center; }
/* 完了した進捗部分を青色にします。 */
QProgressBar::chunk { background: #2b79ca; border-radius: 5px; }
"""


# 秒数を動画画面で見やすい時間へ変換します。
def time_label(seconds: float) -> str:
    # 負数や端数を画面に出さないよう整数へ直します。
    value = max(0, int(seconds))
    # 一時間以上なら時・分・秒を表示します。
    if value >= 3600:
        # 時間をゼロ埋めした表示にします。
        return f"{value // 3600}:{(value // 60) % 60:02d}:{value % 60:02d}"
    # 一時間未満は分と秒で十分です。
    return f"{value // 60}:{value % 60:02d}"


# 保存済み画像を指定サイズへ縮めてアイコンにします。
def image_icon(path: str | None, width: int, height: int) -> QIcon:
    # 未生成の画像は空アイコンにします。
    if not path or not Path(path).is_file():
        # 画像のない状態を呼び出し側が描けるようにします。
        return QIcon()
    # JPEG画像をQtへ読み込みます。
    picture = QPixmap(path)
    # 壊れた画像を無理に表示しません。
    if picture.isNull():
        # 空アイコンを返します。
        return QIcon()
    # 縦横比を保って枠へ収めます。
    scaled = picture.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    # ボタンに設定できるアイコンへ変換します。
    return QIcon(scaled)


# 子要素の部品を削除し、一覧を新しいデータで描き直します。
def clear_layout(layout: QVBoxLayout | QGridLayout) -> None:
    # レイアウト内の要素がなくなるまで繰り返します。
    while layout.count():
        # 先頭の配置要素を取り外します。
        item = layout.takeAt(0)
        # 要素に入っていた部品を取得します。
        widget = item.widget()
        # 部品であればQtの後処理へ渡します。
        if widget is not None:
            # 古いカードを画面から削除します。
            widget.deleteLater()
        # 内側のレイアウトがあれば同じように空にします。
        elif item.layout() is not None:
            # 子レイアウト内の表示も消します。
            clear_layout(item.layout())


# 一つの顔の操作メニューをどの画面からも使えるようにします。
def face_menu(parent: QWidget, model: LibraryController, face: FaceRecord) -> QMenu:
    # 顔操作をまとめるメニューを作ります。
    menu = QMenu(parent)
    # 一つの顔を独立人物にする項目です。
    split = menu.addAction(tr("新しい人物に分割"))
    # 押されたとき顔IDを使って分割します。
    split.triggered.connect(lambda: model.split_face(face.id))
    # 顔を未分類へ戻す項目です。
    unclassify = menu.addAction(tr("未分類へ戻す"))
    # 押されたとき割当を空にします。
    unclassify.triggered.connect(lambda: model.assign_face(face.id, None))
    # 既存人物へ移す候補をまとめます。
    move = menu.addMenu(tr("別の人物へ移動"))
    # 現在の人物以外を移動先として並べます。
    for person in model.people:
        # 自分自身への移動は不要です。
        if person.id != face.person_id:
            # 名前がなければ画像で見分ける操作と説明します。
            label = person.name or tr("名前なしの人物")
            # メニューの項目を追加します。
            action = move.addAction(label)
            # ループ変数の値を固定して対象を取り違えません。
            action.triggered.connect(lambda checked=False, target=person.id: model.assign_face(face.id, target))
    # 誤検出を一覧から隠す項目です。
    exclude = menu.addAction(tr("この顔を除外"))
    # 押されたとき顔を除外します。
    exclude.triggered.connect(lambda: model.exclude_face(face.id))
    # 解析中は保存済み割当を触れないようにします。
    menu.setEnabled(not model.is_scanning)
    # 呼び出し側で表示できるメニューを返します。
    return menu


# 削除前の確認ダイアログを選択中の言語で表示します。
def confirm_deletion(parent: QWidget, title: str, message: str) -> bool:
    # 利用者が対象を判断できるタイトルと説明を設定します。
    dialog = QMessageBox(QMessageBox.Question, title, message, QMessageBox.Yes | QMessageBox.No, parent)
    # 明示的な削除操作へラベルを変えます。
    dialog.button(QMessageBox.Yes).setText(tr("削除"))
    # 誤操作を避ける中止操作の名前を設定します。
    dialog.button(QMessageBox.No).setText(tr("キャンセル"))
    # 何もしない操作を既定の選択にします。
    dialog.setDefaultButton(QMessageBox.No)
    # 利用者が削除を選んだ場合だけ真を返します。
    return dialog.exec() == QMessageBox.Yes


# 動画を開けない理由を現在の言語で知らせます。
def show_warning(parent: QWidget, title: str, message: str) -> None:
    # 警告の見出しと原因を設定します。
    dialog = QMessageBox(QMessageBox.Warning, title, message, QMessageBox.Ok, parent)
    # ダイアログを閉じる操作名も現在の言語にそろえます。
    dialog.button(QMessageBox.Ok).setText(tr("閉じる"))
    # 利用者が確認するまで原因を表示します。
    dialog.exec()


# 元動画を再生して検出時刻へ移動する画面です。
class PlayerDialog(QDialog):
    # 動画と再生開始時刻からダイアログを作ります。
    def __init__(self, model: LibraryController, video: VideoRecord, second: float = 0.0, parent: QWidget | None = None) -> None:
        # Qtダイアログの基本機能を初期化します。
        super().__init__(parent)
        # 顔の一覧更新に使うモデルを保持します。
        self.model = model
        # 再生する動画を保持します。
        self.video = video
        # 動画ファイル名をウィンドウ名へ設定します。
        self.setWindowTitle(f"VideoAtlas — {video.name}")
        # 動画と顔を見やすい初期サイズにします。
        self.resize(960, 680)
        # 上から下へ内容を並べます。
        root = QVBoxLayout(self)
        # 見出しと操作を横に並べます。
        header = QHBoxLayout()
        # 動画名を大きく表示します。
        title = QLabel(video.name)
        # 再生対象を見分けやすくします。
        title.setStyleSheet("font-size: 19px; font-weight: 700;")
        # 見出しを左へ追加します。
        header.addWidget(title)
        # 右へ操作を寄せます。
        header.addStretch()
        # 詳細解析の操作を作ります。
        rescan = QPushButton(tr("詳しく再解析"))
        # 0.5秒間隔の再解析を依頼します。
        rescan.clicked.connect(lambda: self.model.rescan_video(video.id, True))
        # 解析中は再解析の重複実行を避けます。
        rescan.setEnabled(not model.is_scanning)
        # 右上へ再解析ボタンを置きます。
        header.addWidget(rescan)
        # 再生画面を閉じるボタンを作ります。
        close_button = QPushButton(tr("閉じる"))
        # 押したらダイアログを閉じます。
        close_button.clicked.connect(self.accept)
        # 右上へ閉じるボタンを置きます。
        header.addWidget(close_button)
        # 見出し行を画面へ追加します。
        root.addLayout(header)
        # Qtの映像表示部品を作ります。
        self.video_widget = QVideoWidget(self)
        # 動画の映像領域に最低限の高さを与えます。
        self.video_widget.setMinimumHeight(340)
        # 黒い背景上に映像を表示します。
        self.video_widget.setStyleSheet("background: #101827;")
        # 映像表示部品を画面へ追加します。
        root.addWidget(self.video_widget, 1)
        # 音声出力を作ります。
        self.audio = QAudioOutput(self)
        # 再生制御部品を作ります。
        self.player = QMediaPlayer(self)
        # 再生音声を出力へ接続します。
        self.player.setAudioOutput(self.audio)
        # 再生映像を画面へ接続します。
        self.player.setVideoOutput(self.video_widget)
        # 選ばれた動画ファイルを再生対象へ設定します。
        self.player.setSource(QUrl.fromLocalFile(video.path))
        # 再生位置とボタンを並べる行です。
        controls = QHBoxLayout()
        # 利用者が再生を開始するボタンです。
        self.play_button = QPushButton(tr("▶ 再生"))
        # ボタンが押されたら再生状態を切り替えます。
        self.play_button.clicked.connect(self._toggle_playback)
        # ボタンを操作行へ追加します。
        controls.addWidget(self.play_button)
        # 動画内の位置を示すスライダーです。
        self.slider = QSlider(Qt.Horizontal)
        # 位置をミリ秒で指定できるようにします。
        self.slider.setRange(0, max(0, int(video.duration * 1000)))
        # 利用者が動かしたら動画を指定位置へ移します。
        self.slider.sliderMoved.connect(self.player.setPosition)
        # 動画の読み込み後に正しい全長を反映します。
        self.player.durationChanged.connect(self.slider.setMaximum)
        # 再生中の位置を画面へ反映します。
        self.player.positionChanged.connect(self.slider.setValue)
        # 横幅を位置操作に多く使います。
        controls.addWidget(self.slider, 1)
        # 現在位置と総時間の表示です。
        self.position_label = QLabel(f"{time_label(second)} / {time_label(video.duration)}")
        # 現在位置の更新を受け取ります。
        self.player.positionChanged.connect(self._position_changed)
        # 操作行へ時間表示を追加します。
        controls.addWidget(self.position_label)
        # 操作行を画面へ追加します。
        root.addLayout(controls)
        # 検出顔の見出しを表示します。
        heading = QLabel(tr("この動画に登場する人物"))
        # 見出しを少し大きくします。
        heading.setStyleSheet("font-weight: 700; font-size: 15px;")
        # 顔一覧の上へ置きます。
        root.addWidget(heading)
        # 顔写真を横方向へスクロールできる枠を作ります。
        scroller = QScrollArea()
        # 枠の高さをそろえます。
        scroller.setFixedHeight(126)
        # 中身のサイズに合わせて横スクロールします。
        scroller.setWidgetResizable(True)
        # 顔ボタンの容器を作ります。
        face_host = QWidget()
        # 顔ボタンを左から並べます。
        self.face_row = QHBoxLayout(face_host)
        # スクロール枠に顔ボタンを置きます。
        scroller.setWidget(face_host)
        # 顔一覧を画面へ追加します。
        root.addWidget(scroller)
        # 再生位置の初期値をミリ秒へ直します。
        self.initial_position = max(0, int(second * 1000))
        # 動画の読み込み後に検出時刻へ移動します。
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        # 最初の顔一覧を表示します。
        self._render_faces()
        # 顔の手動操作後にも一覧を更新します。
        self.model.changed.connect(self._render_faces)

    # 読み込み済みの動画を検出時刻から表示します。
    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        # 動画が読み込まれた時だけ初期移動をします。
        if status == QMediaPlayer.MediaStatus.LoadedMedia and self.initial_position > 0:
            # 自動再生せず初期位置へシークします。
            self.player.setPosition(self.initial_position)
            # 二度シークしないよう初期指定を消します。
            self.initial_position = 0

    # 再生と一時停止を切り替えます。
    def _toggle_playback(self) -> None:
        # 再生中なら一時停止します。
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            # 現在位置を保持して停止します。
            self.player.pause()
            # 次の操作名を再生にします。
            self.play_button.setText(tr("▶ 再生"))
        # 一時停止中なら再生します。
        else:
            # 現在位置から再生を始めます。
            self.player.play()
            # 次の操作名を一時停止にします。
            self.play_button.setText(tr("Ⅱ 一時停止"))

    # 現在位置を人が読める秒数へ変換します。
    def _position_changed(self, milliseconds: int) -> None:
        # 再生位置と動画の長さを併記します。
        self.position_label.setText(f"{time_label(milliseconds / 1000)} / {time_label(self.video.duration)}")

    # この動画で検出した顔を時刻順で並べます。
    def _render_faces(self) -> None:
        # 古い顔ボタンを削除します。
        clear_layout(self.face_row)
        # 除外していない顔だけを対象にします。
        faces = sorted((face for face in self.model.faces if face.video_id == self.video.id and not face.excluded), key=lambda face: face.second)
        # 未検出のときは断定しない案内を出します。
        if not faces:
            # 解析状態に応じて説明文を選びます。
            message = tr("顔が検出されませんでした。必要なら詳しく再解析できます。") if self.video.state == "ready" else tr("人物の検出結果はまだありません。")
            # 案内文を顔一覧へ置きます。
            self.face_row.addWidget(QLabel(message))
        # 検出顔を時刻とともに表示します。
        for face in faces:
            # 画像を押すと動画内のその時刻へ移動します。
            button = QPushButton(time_label(face.second))
            # 顔画像があればボタンへ表示します。
            button.setIcon(image_icon(face.thumbnail_path, 72, 72))
            # 顔画像の枠を設定します。
            button.setIconSize(QSize(72, 72))
            # 時刻が見える高さへそろえます。
            button.setFixedSize(90, 92)
            # 押した顔の時刻を再生位置にします。
            button.clicked.connect(lambda checked=False, moment=face.second: self.player.setPosition(int(moment * 1000)))
            # 右クリックで手動修正メニューを出します。
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            # この顔を固定して操作メニューを表示します。
            button.customContextMenuRequested.connect(lambda position, current=face, item=button: face_menu(item, self.model, current).exec(item.mapToGlobal(position)))
            # 横並びに顔ボタンを追加します。
            self.face_row.addWidget(button)
        # 余った幅を右側へ寄せます。
        self.face_row.addStretch()

    # ダイアログを閉じる際に動画の音も止めます。
    def closeEvent(self, event: object) -> None:
        # 再生を停止します。
        self.player.stop()
        # Qt の通常の終了処理へ進みます。
        super().closeEvent(event)


# 登録動画と人物を切り替える主ウィンドウです。
class MainWindow(QMainWindow):
    # 保存モデルを受け取って初期画面を組み立てます。
    def __init__(self, model: LibraryController) -> None:
        # Qt のウィンドウ機能を初期化します。
        super().__init__()
        # ウィンドウ内の部品を作る前に保存済みの言語を読みます。
        load_language()
        # 全操作で同じライブラリ状態を使います。
        self.model = model
        # 初期状態は全動画を表示します。
        self.section = "all"
        # フォルダ選択時の登録元IDです。
        self.source_id: str | None = None
        # 最後に選んだ人物IDです。
        self.person_id: str | None = None
        # 未分類だけを選んだか示します。
        self.show_unclassified = False
        # 人物別の時刻表示数を動画ごとに保持します。
        self.visible_moments: dict[str, int] = {}
        # ウィンドウの見出しを設定します。
        self.setWindowTitle("VideoAtlas Python")
        # 空のライブラリが見やすいサイズにします。
        self.resize(1120, 740)
        # フォルダのドロップを受け取ります。
        self.setAcceptDrops(True)
        # カラーパレットと部品の見た目を適用します。
        self.setStyleSheet(STYLE)
        # 全体の左右分割を作ります。
        outer = QWidget()
        # 中央部品として設定します。
        self.setCentralWidget(outer)
        # 左の案内と右の一覧を並べます。
        columns = QHBoxLayout(outer)
        # 外周の余白をなくします。
        columns.setContentsMargins(0, 0, 0, 0)
        # 左右の間隔をなくします。
        columns.setSpacing(0)
        # サイドバーを作ります。
        sidebar = self._build_sidebar()
        # 固定幅のサイドバーを左へ置きます。
        columns.addWidget(sidebar)
        # 右側の内容領域を作ります。
        content = QWidget()
        # 右側の縦方向レイアウトを作ります。
        self.content_layout = QVBoxLayout(content)
        # 上下左右の余白を設定します。
        self.content_layout.setContentsMargins(30, 26, 30, 26)
        # 各領域の間隔を設定します。
        self.content_layout.setSpacing(18)
        # 画面の見出しと操作列を追加します。
        self._build_header()
        # 解析状況の表示欄を作ります。
        self._build_status()
        # 一覧を配置するスクロール枠を作ります。
        self.scroller = QScrollArea()
        # ウィンドウの幅に合わせて中身を広げます。
        self.scroller.setWidgetResizable(True)
        # 動画と人物に共通の内容容器です。
        self.list_host = QWidget()
        # 中身を縦に積み上げます。
        self.list_layout = QVBoxLayout(self.list_host)
        # 内側の余白を消します。
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        # スクロール枠へ内容容器を入れます。
        self.scroller.setWidget(self.list_host)
        # 一覧を残りの高さへ広げます。
        self.content_layout.addWidget(self.scroller, 1)
        # 右側の画面を残りの幅へ広げます。
        columns.addWidget(content, 1)
        # 保存データが更新されたら画面を描き直します。
        self.model.changed.connect(self._render)
        # 状態文や進捗が更新されたら表示を直します。
        self.model.status_changed.connect(self._update_status)
        # 空ライブラリを含む初期画面を表示します。
        self._render()
        # 初期状態の解析欄を設定します。
        self._update_status()

    # サイドバーの移動先と登録操作を作ります。
    def _build_sidebar(self) -> QFrame:
        # 暗い背景のサイドバーを作ります。
        sidebar = QFrame()
        # 見た目を指定する名前を付けます。
        sidebar.setObjectName("Sidebar")
        # 内容に安定した幅を与えます。
        sidebar.setFixedWidth(245)
        # 縦に項目を並べます。
        self.side_layout = QVBoxLayout(sidebar)
        # サイドバーの余白を指定します。
        self.side_layout.setContentsMargins(18, 24, 18, 18)
        # 項目間の余白を指定します。
        self.side_layout.setSpacing(7)
        # アプリ名を表示します。
        brand = QLabel("◧  VideoAtlas")
        # 目立つ文字サイズにします。
        brand.setStyleSheet("font-size: 22px; font-weight: 800; color: white;")
        # サイドバーの先頭へ置きます。
        self.side_layout.addWidget(brand)
        # Python版の短い説明です。
        self.caption = QLabel(tr("ローカル動画ライブラリ"))
        # 補助文字を少し淡くします。
        self.caption.setStyleSheet("color: #9eb4d4; font-size: 11px;")
        # ブランド名の下へ置きます。
        self.side_layout.addWidget(self.caption)
        # ナビゲーション前に少し空けます。
        self.side_layout.addSpacing(22)
        # 全動画へ戻るボタンを作ります。
        self.all_button = QPushButton(tr("▣  すべての動画"))
        # 現在位置を示せるボタンにします。
        self.all_button.setCheckable(True)
        # 押すと動画一覧を選びます。
        self.all_button.clicked.connect(lambda: self._select_section("all"))
        # サイドバーへ追加します。
        self.side_layout.addWidget(self.all_button)
        # 人物画面へ移るボタンを作ります。
        self.people_button = QPushButton(tr("♙  人物"))
        # 選択状態を表示できるようにします。
        self.people_button.setCheckable(True)
        # 押すと人物画面を開きます。
        self.people_button.clicked.connect(lambda: self._select_section("people"))
        # サイドバーへ追加します。
        self.side_layout.addWidget(self.people_button)
        # 登録フォルダの見出しを作ります。
        self.folder_heading = QLabel(tr("ライブラリフォルダ"))
        # 読みやすい補助色を設定します。
        self.folder_heading.setStyleSheet("color: #99afce; font-size: 11px; font-weight: 700; margin-top: 18px;")
        # フォルダ一覧の上へ置きます。
        self.side_layout.addWidget(self.folder_heading)
        # 登録済みフォルダの操作を置くレイアウトです。
        self.folder_layout = QVBoxLayout()
        # サイドバーにフォルダ一覧を組み込みます。
        self.side_layout.addLayout(self.folder_layout)
        # 新しいフォルダを選ぶボタンです。
        self.add_button = QPushButton(tr("＋  フォルダを追加"))
        # 目立つ色を使うため名前を付けます。
        self.add_button.setObjectName("AddFolder")
        # ファイル選択ダイアログを開きます。
        self.add_button.clicked.connect(self._choose_folder)
        # フォルダ一覧の下へ置きます。
        self.side_layout.addWidget(self.add_button)
        # 下の削除操作まで空きを取ります。
        self.side_layout.addStretch()
        # 言語選択の見出しをサイドバーに置きます。
        self.language_label = QLabel(tr("言語"))
        # 補助項目として見出しの色を設定します。
        self.language_label.setStyleSheet("color: #99afce; font-size: 11px; font-weight: 700;")
        # 言語見出しを選択欄の直前に置きます。
        self.side_layout.addWidget(self.language_label)
        # 日本語と英語を直接選べる欄を作ります。
        self.language_combo = QComboBox()
        # 選択肢には表示名と保存用の言語コードを持たせます。
        self.language_combo.addItem(tr("日本語"), "ja")
        # 二つ目の選択肢に英語を加えます。
        self.language_combo.addItem(tr("英語"), "en")
        # 保存済みの言語を初期選択にします。
        self.language_combo.setCurrentIndex(0 if load_language() == "ja" else 1)
        # 利用者が選択したら画面の文言を更新します。
        self.language_combo.currentIndexChanged.connect(self._change_language)
        # 言語選択欄をサイドバーに追加します。
        self.side_layout.addWidget(self.language_combo)
        # 索引だけを消すボタンを作ります。
        self.clear_button = QPushButton(tr("⌫  インデックスを消去…"))
        # 補助的な破壊操作として色を設定します。
        self.clear_button.setObjectName("ClearIndex")
        # 確認画面を通して削除します。
        self.clear_button.clicked.connect(self._confirm_clear)
        # サイドバーの末尾へ置きます。
        self.side_layout.addWidget(self.clear_button)
        # 組み立てたサイドバーを返します。
        return sidebar

    # 画面の見出し、検索、更新操作を作ります。
    def _build_header(self) -> None:
        # 見出し行を作ります。
        header = QHBoxLayout()
        # 左側の画面タイトルです。
        self.title = QLabel(tr("すべての動画"))
        # 選択先が分かる大きな文字にします。
        self.title.setStyleSheet("font-size: 29px; font-weight: 800;")
        # 見出しを左へ置きます。
        header.addWidget(self.title)
        # 残りの操作を右へ寄せます。
        header.addStretch()
        # 動画名とパスの検索欄です。
        self.search = QLineEdit()
        # 入力前の説明を設定します。
        self.search.setPlaceholderText(tr("動画名・パスを検索"))
        # 一定の幅を与えます。
        self.search.setFixedWidth(225)
        # 入力ごとに動画カードを絞り込みます。
        self.search.textChanged.connect(self._render)
        # 検索欄を見出し行へ追加します。
        header.addWidget(self.search)
        # 更新または一時停止の共通ボタンです。
        self.action_button = QPushButton(tr("フォルダを更新"))
        # 主要操作の色を指定します。
        self.action_button.setObjectName("Primary")
        # 状態に応じた操作を実行します。
        self.action_button.clicked.connect(self._main_action)
        # 見出し行の右端へ置きます。
        header.addWidget(self.action_button)
        # 完成した見出しを画面へ追加します。
        self.content_layout.addLayout(header)

    # 動画解析の状態カードを作ります。
    def _build_status(self) -> None:
        # 白い状態カードを作ります。
        self.status_frame = QFrame()
        # 外観用の名前を付けます。
        self.status_frame.setObjectName("Status")
        # 状態文と進捗バーを縦に置きます。
        status_layout = QVBoxLayout(self.status_frame)
        # 状態カードの内側余白を設定します。
        status_layout.setContentsMargins(14, 12, 14, 12)
        # 現在の処理を説明する文字です。
        self.status_label = QLabel("")
        # 状態欄へ追加します。
        status_layout.addWidget(self.status_label)
        # 0から100の進捗バーを作ります。
        self.progress_bar = QProgressBar()
        # バー内の数字は小さく表示しません。
        self.progress_bar.setTextVisible(False)
        # 進捗バーを状態欄へ追加します。
        status_layout.addWidget(self.progress_bar)
        # 状態欄を見出しの下へ追加します。
        self.content_layout.addWidget(self.status_frame)

    # サイドバーの選択を変更します。
    def _select_section(self, section: str, source_id: str | None = None) -> None:
        # 全動画、人物、登録フォルダのいずれかを保存します。
        self.section = section
        # フォルダIDがあれば対象を記録します。
        self.source_id = source_id
        # 新しい選択先を描きます。
        self._render()

    # 利用者の言語選択を保存し、既存の固定文と一覧を描き直します。
    def _change_language(self, index: int) -> None:
        # 選択肢に結び付けた保存用コードを取得します。
        language = self.language_combo.itemData(index)
        # 言語が読み取れない場合は何も変えません。
        if language not in {"ja", "en"}:
            # 不正な選択値では画面を更新しません。
            return
        # 次回起動時にも同じ言語になるよう保存します。
        set_language(language)
        # サイドバーの固定文を選択した言語へ変えます。
        self.caption.setText(tr("ローカル動画ライブラリ"))
        # 全動画の操作名を更新します。
        self.all_button.setText(tr("▣  すべての動画"))
        # 人物画面の操作名を更新します。
        self.people_button.setText(tr("♙  人物"))
        # フォルダ一覧の見出しを更新します。
        self.folder_heading.setText(tr("ライブラリフォルダ"))
        # フォルダ追加の操作名を更新します。
        self.add_button.setText(tr("＋  フォルダを追加"))
        # 索引消去の操作名を更新します。
        self.clear_button.setText(tr("⌫  インデックスを消去…"))
        # 言語選択の見出しを更新します。
        self.language_label.setText(tr("言語"))
        # 選択肢の見せ方だけを更新し、各項目の言語コードは残します。
        self.language_combo.setItemText(0, tr("日本語"))
        # 英語側の表示名も更新します。
        self.language_combo.setItemText(1, tr("英語"))
        # 検索欄の説明を更新します。
        self.search.setPlaceholderText(tr("動画名・パスを検索"))
        # カードや人物結果を選択した言語で作り直します。
        self._render()

    # 最新データからサイドバーと一覧を作り直します。
    def _render(self) -> None:
        # 古いフォルダ項目を消します。
        clear_layout(self.folder_layout)
        # 登録済みフォルダごとに項目を追加します。
        for source in self.model.sources:
            # パスの最後の名前を表示します。
            button = QPushButton(f"▱  {Path(source.path).name}")
            # 長いパスをツールチップで確認できます。
            button.setToolTip(source.path)
            # 選択状態を表示できるようにします。
            button.setCheckable(True)
            # このフォルダの動画を表示します。
            button.clicked.connect(lambda checked=False, identifier=source.id: self._select_section("folder", identifier))
            # 選択中のフォルダだけ強調します。
            button.setChecked(self.section == "folder" and self.source_id == source.id)
            # 右クリックで登録解除を選べるようにします。
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            # フォルダIDを固定して削除メニューを表示します。
            button.customContextMenuRequested.connect(lambda point, identifier=source.id, item=button: self._folder_menu(item, point, identifier))
            # サイドバーのフォルダ一覧へ追加します。
            self.folder_layout.addWidget(button)
        # 全動画の選択状態を反映します。
        self.all_button.setChecked(self.section == "all")
        # 人物画面の選択状態を反映します。
        self.people_button.setChecked(self.section == "people")
        # 索引削除は解析中に押せないようにします。
        self.clear_button.setEnabled(not self.model.is_scanning)
        # 古い内容画面をすべて消します。
        clear_layout(self.list_layout)
        # 選択先が人物なら人物写真と結果を表示します。
        if self.section == "people":
            # 人物画面の見出しにします。
            self.title.setText(tr("人物"))
            # 動画検索欄は人物画面で隠します。
            self.search.hide()
            # 人物一覧と選択結果を作ります。
            self._render_people()
        # 全動画またはフォルダを選んだ場合です。
        else:
            # 検索欄を再表示します。
            self.search.show()
            # フォルダを選んだ場合は登録名を表示します。
            source = next((item for item in self.model.sources if item.id == self.source_id), None)
            # フォルダ名または全動画を見出しにします。
            self.title.setText(Path(source.path).name if self.section == "folder" and source is not None else tr("すべての動画"))
            # 動画カードを検索条件で表示します。
            self._render_videos()
        # 最後に進捗欄とボタンを更新します。
        self._update_status()

    # 動画カードを一覧へ配置します。
    def _render_videos(self) -> None:
        # 入力欄の検索語を小文字へ変えます。
        needle = self.search.text().strip().casefold()
        # フォルダと検索語の両方に合う動画を選びます。
        videos = [video for video in self.model.videos if (self.section != "folder" or video.source_id == self.source_id) and (not needle or needle in video.name.casefold() or needle in video.path.casefold())]
        # 表示できる動画がなければ操作案内を出します。
        if not videos:
            # 検索中なら検索結果の案内にします。
            title = tr("見つかりませんでした") if needle else tr("動画がありません")
            # 検索語がない場合はフォルダ追加方法を示します。
            description = tr("検索語を変えてください。") if needle else tr("フォルダを追加するか、ここへ動画フォルダをドラッグしてください。")
            # 空状態のカードを追加します。
            self._show_empty(title, description, not needle)
            # 動画カードの処理を終えます。
            return
        # 動画を左から三列で配置します。
        grid = QGridLayout()
        # カード同士の間隔を設定します。
        grid.setSpacing(17)
        # ファイル名の順で各動画を配置します。
        for index, video in enumerate(sorted(videos, key=lambda item: item.name.casefold())):
            # 一件の画像、状態、再生操作を作ります。
            card = self._video_card(video)
            # 三列の行と列へ置きます。
            grid.addWidget(card, index // 3, index % 3)
        # カードが左詰めになるよう残りの列を伸ばします。
        grid.setColumnStretch(3, 1)
        # 一覧にカード配置を追加します。
        self.list_layout.addLayout(grid)
        # カード数が少なくても上へ寄せます。
        self.list_layout.addStretch()

    # 一件の動画カードを作ります。
    def _video_card(self, video: VideoRecord) -> QFrame:
        # 白いカード枠を作ります。
        card = QFrame()
        # 共通カードの見た目を適用します。
        card.setObjectName("Card")
        # 三列で並べるため幅をそろえます。
        card.setFixedWidth(235)
        # 画像と文字を上下に置きます。
        layout = QVBoxLayout(card)
        # 内側の余白を設定します。
        layout.setContentsMargins(12, 12, 12, 12)
        # ポスター画像を置くラベルです。
        picture = QLabel("▶")
        # ポスター領域の大きさを設定します。
        picture.setFixedSize(210, 118)
        # 画像がなくても見やすい背景にします。
        picture.setStyleSheet("background: #dfe8f4; border-radius: 10px; color: #5c7fae; font-size: 34px;")
        # 代替文字を中央に寄せます。
        picture.setAlignment(Qt.AlignCenter)
        # 保存済みの画像があれば表示します。
        if video.poster_path and Path(video.poster_path).is_file():
            # JPEGを読み込みます。
            pixmap = QPixmap(video.poster_path)
            # 読み取れた画像だけ枠へ縮めます。
            if not pixmap.isNull():
                # 縦横比を保ったポスターへ変更します。
                picture.setPixmap(pixmap.scaled(210, 118, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        # ポスターをカードへ追加します。
        layout.addWidget(picture)
        # ファイル名を表示します。
        name = QLabel(video.name)
        # 長い名前を複数行にします。
        name.setWordWrap(True)
        # ファイル名を強調します。
        name.setStyleSheet("font-weight: 700;")
        # カードへ名前を追加します。
        layout.addWidget(name)
        # 動画の長さと解析状態を説明します。
        state_names = {"pending": tr("待機中"), "analyzing": tr("解析中"), "ready": tr("完了"), "missing": tr("ファイルなし"), "failed": tr("失敗")}
        # 秒数と状態を一行で表示します。
        meta = QLabel(f"{time_label(video.duration)}   ·   {state_names.get(video.state, video.state)}")
        # 補助文字を淡くします。
        meta.setStyleSheet("color: #718198; font-size: 11px;")
        # カードへ状態を追加します。
        layout.addWidget(meta)
        # 解析エラーがあれば理由を表示します。
        if video.error_message:
            # エラーを読みやすい文字列にします。
            reason = QLabel(localize_message(video.error_message))
            # 長い文章は折り返します。
            reason.setWordWrap(True)
            # エラーだけ赤みのある色にします。
            reason.setStyleSheet("color: #a4512b; font-size: 11px;")
            # カードへ理由を追加します。
            layout.addWidget(reason)
        # 再生操作を下へ寄せます。
        layout.addStretch()
        # この動画を開く操作です。
        open_button = QPushButton(tr("動画を開く"))
        # クリック時に動画IDを固定して再生します。
        open_button.clicked.connect(lambda checked=False, item=video: self._open_player(item))
        # ファイル不在なら再生を押せないようにします。
        open_button.setEnabled(Path(video.path).is_file())
        # カード末尾へ再生操作を追加します。
        layout.addWidget(open_button)
        # 完成したカードを返します。
        return card

    # 人物写真とその人が映った動画を表示します。
    def _render_people(self) -> None:
        # 解析待ち動画があれば状態を説明します。
        if any(video.state in {"pending", "analyzing"} for video in self.model.videos):
            # 結果が増える可能性を利用者に伝えます。
            note = QLabel(tr("解析中・解析待ちの動画があります。結果は順次ここへ追加されます。"))
            # 補助文字の色を指定します。
            note.setStyleSheet("color: #718198;")
            # 人物一覧の上へ追加します。
            self.list_layout.addWidget(note)
        # 未分類を含めて人物タイルを置くレイアウトです。
        grid = QGridLayout()
        # タイル間に適度な余白を作ります。
        grid.setSpacing(12)
        # 現在の人物を順番に表示します。
        for index, person in enumerate(self.model.people):
            # 写真と名前のボタンを作ります。
            tile = QPushButton(person.name or tr("顔写真"))
            # 代表画像を表示します。
            tile.setIcon(image_icon(person.thumbnail_path, 72, 72))
            # 画像の最大サイズを設定します。
            tile.setIconSize(QSize(72, 72))
            # 人物タイルの大きさをそろえます。
            tile.setFixedSize(190, 95)
            # 押すとこの人物の動画と時刻へ絞ります。
            tile.clicked.connect(lambda checked=False, identifier=person.id: self._select_person(identifier))
            # 右クリックの名前変更・統合を有効にします。
            tile.setContextMenuPolicy(Qt.CustomContextMenu)
            # 人物IDを固定してメニューを開きます。
            tile.customContextMenuRequested.connect(lambda point, current=person, item=tile: self._person_menu(item, point, current))
            # 四列で写真を並べます。
            grid.addWidget(tile, index // 4, index % 4)
        # 未分類顔の数を数えます。
        unclassified_count = sum(face.person_id is None and not face.excluded for face in self.model.faces)
        # 未分類のタイルを作ります。
        unknown = QPushButton(tr("未分類  {count}件", count=unclassified_count))
        # 他の写真と同じ大きさにします。
        unknown.setFixedSize(190, 95)
        # 押すと未分類顔を表示します。
        unknown.clicked.connect(self._select_unclassified)
        # 人物の後ろへ配置します。
        grid.addWidget(unknown, len(self.model.people) // 4, len(self.model.people) % 4)
        # タイル配置を画面へ追加します。
        self.list_layout.addLayout(grid)
        # 人物が一件もないときの案内を表示します。
        if not self.model.people and unclassified_count == 0:
            # 動画を登録する操作へ誘導します。
            self._show_empty(tr("人物はまだ見つかっていません"), tr("動画フォルダを追加すると、見つかった顔をここで探せます。"), True)
        # 既に人物を選んだ場合は関連動画を表示します。
        elif self.person_id is not None or self.show_unclassified:
            # 動画別の顔検出時刻を作ります。
            self._render_person_results()
        # 内容が少なくても上に寄せます。
        self.list_layout.addStretch()

    # 人物タイルを選んだとき結果を作り直します。
    def _select_person(self, person_id: str) -> None:
        # 選択した人物を保存します。
        self.person_id = person_id
        # 未分類の選択を解除します。
        self.show_unclassified = False
        # 検出時刻の表示上限を初期値へ戻します。
        self.visible_moments = {}
        # 人物画面を描き直します。
        self._render()

    # 未分類の顔だけを選びます。
    def _select_unclassified(self) -> None:
        # 特定人物の選択を解除します。
        self.person_id = None
        # 未分類状態を選びます。
        self.show_unclassified = True
        # 検出時刻の表示上限を初期値へ戻します。
        self.visible_moments = {}
        # 人物画面を描き直します。
        self._render()

    # 選択した人物の動画と検出時刻を表示します。
    def _render_person_results(self) -> None:
        # 現在の選択を人が読める文字にします。
        person = next((item for item in self.model.people if item.id == self.person_id), None)
        # 名前なしの人物は写真選択として説明します。
        heading_text = tr("未分類の顔") if self.show_unclassified else (person.name or tr("選択した人物")) if person is not None else tr("選択した人物")
        # 結果の見出しを作ります。
        heading = QLabel(heading_text)
        # 動画別結果との区切りを付けます。
        heading.setStyleSheet("font-size: 20px; font-weight: 700; margin-top: 18px;")
        # 一覧へ見出しを追加します。
        self.list_layout.addWidget(heading)
        # 選択に一致する有効な顔だけを抽出します。
        matches = [face for face in self.model.faces if not face.excluded and ((self.show_unclassified and face.person_id is None) or (not self.show_unclassified and face.person_id == self.person_id))]
        # 顔がなければ理由を表示します。
        if not matches:
            # 対象人物の検出結果がないことを表示します。
            self.list_layout.addWidget(QLabel(tr("該当する検出時刻はありません。")))
            # 空結果の描画を終えます。
            return
        # 関連動画を名前順に並べます。
        for video in sorted(self.model.videos, key=lambda item: item.name.casefold()):
            # この動画に属する顔を時刻順にします。
            moments = sorted((face for face in matches if face.video_id == video.id), key=lambda face: face.second)
            # 顔がない動画は表示しません。
            if not moments:
                # 次の動画を見ます。
                continue
            # 動画名と検出数を一つの行で表示します。
            video_heading = QLabel(tr("{name}  ·  {count}件", name=video.name, count=len(moments)))
            # 動画の区切りを強調します。
            video_heading.setStyleSheet("font-weight: 700; margin-top: 8px;")
            # 見出しを画面へ追加します。
            self.list_layout.addWidget(video_heading)
            # 最初は一動画につき60件を表示します。
            limit = self.visible_moments.get(video.id, 60)
            # 時刻ボタンを横方向へ折り返して並べます。
            row = QGridLayout()
            # 最大表示件数まで顔を追加します。
            for index, face in enumerate(moments[:limit]):
                # 検出時刻を操作名にします。
                button = QPushButton(time_label(face.second))
                # 顔画像を一緒に表示します。
                button.setIcon(image_icon(face.thumbnail_path, 46, 46))
                # 小さな画像枠を指定します。
                button.setIconSize(QSize(46, 46))
                # 押すと動画のその時刻を開きます。
                button.clicked.connect(lambda checked=False, item=video, second=face.second: self._open_player(item, second))
                # 顔単位の右クリック操作を設定します。
                button.setContextMenuPolicy(Qt.CustomContextMenu)
                # 顔IDを固定してメニューを開きます。
                button.customContextMenuRequested.connect(lambda point, current=face, item=button: face_menu(item, self.model, current).exec(item.mapToGlobal(point)))
                # 四列で時刻を配置します。
                row.addWidget(button, index // 4, index % 4)
            # この動画の検出時刻を画面に追加します。
            self.list_layout.addLayout(row)
            # まだ顔がある場合は追加の60件を表示できます。
            if len(moments) > limit:
                # 次の時刻を表示する操作を作ります。
                more = QPushButton(tr("さらに60件表示"))
                # 対象動画を固定して上限を増やします。
                more.clicked.connect(lambda checked=False, identifier=video.id: self._more_moments(identifier))
                # 追加表示ボタンを画面に置きます。
                self.list_layout.addWidget(more)

    # 一動画の表示件数を60件増やします。
    def _more_moments(self, video_id: str) -> None:
        # 今の表示上限へ60件を足します。
        self.visible_moments[video_id] = self.visible_moments.get(video_id, 60) + 60
        # 増えた時刻を画面へ表示します。
        self._render()

    # 空の一覧に次の操作を示すカードを置きます。
    def _show_empty(self, title: str, description: str, show_add: bool) -> None:
        # 白い案内カードを作ります。
        panel = QFrame()
        # 共通の空状態の外観を指定します。
        panel.setObjectName("Empty")
        # カードの最低高さを指定します。
        panel.setMinimumHeight(270)
        # 案内を縦方向に中央へ置きます。
        layout = QVBoxLayout(panel)
        # 説明の上下へ空きを作ります。
        layout.addStretch()
        # 簡単なフィルム記号を表示します。
        symbol = QLabel("▣")
        # 大きな淡い記号にします。
        symbol.setStyleSheet("font-size: 38px; color: #7da3d1;")
        # 記号を中央へ寄せます。
        symbol.setAlignment(Qt.AlignCenter)
        # カードへ追加します。
        layout.addWidget(symbol)
        # 空状態の見出しを作ります。
        heading = QLabel(title)
        # 目立つ太字にします。
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        # 見出しを中央へ寄せます。
        heading.setAlignment(Qt.AlignCenter)
        # カードへ追加します。
        layout.addWidget(heading)
        # 操作方法の説明を作ります。
        details = QLabel(description)
        # 補助色と折り返しを設定します。
        details.setStyleSheet("color: #6d7d92;")
        # テキストを中央へ寄せます。
        details.setAlignment(Qt.AlignCenter)
        # カードへ追加します。
        layout.addWidget(details)
        # フォルダ登録が有効ならボタンを表示します。
        if show_add:
            # 登録を始めるボタンです。
            action = QPushButton(tr("フォルダを追加"))
            # 主要操作の色を使います。
            action.setObjectName("Primary")
            # 不必要に長いボタンを避けます。
            action.setFixedWidth(150)
            # ファイル選択画面を開きます。
            action.clicked.connect(self._choose_folder)
            # ボタンを中央に配置します。
            layout.addWidget(action, alignment=Qt.AlignCenter)
        # 下側にも余白を入れます。
        layout.addStretch()
        # 空状態を画面へ追加します。
        self.list_layout.addWidget(panel)

    # フォルダ選択ダイアログから登録します。
    def _choose_folder(self) -> None:
        # macOSの標準ダイアログでフォルダを選びます。
        chosen = QFileDialog.getExistingDirectory(self, tr("動画フォルダを選択"))
        # キャンセルなら登録しません。
        if chosen:
            # 選んだ場所をライブラリへ登録します。
            self.model.add_folder(Path(chosen))

    # 更新ボタンを状態に応じて一時停止へ切り替えます。
    def _main_action(self) -> None:
        # 解析中なら次の区間で停止します。
        if self.model.is_scanning:
            # 停止指示を送ります。
            self.model.pause_scanning()
        # 待機中なら登録フォルダを再走査します。
        else:
            # 新規・変更動画と中断動画を処理します。
            self.model.refresh_library()

    # 現在の解析状態を上部カードへ反映します。
    def _update_status(self) -> None:
        # 状態がある時だけカードを表示します。
        self.status_frame.setVisible(self.model.is_scanning or bool(self.model.status_text))
        # 解析内容の文章を設定します。
        self.status_label.setText(localize_message(self.model.status_text))
        # 0から100までの整数へ変換します。
        self.progress_bar.setValue(int(self.model.progress * 100))
        # 実行中だけ一時停止にします。
        self.action_button.setText(tr("一時停止") if self.model.is_scanning else tr("フォルダを更新"))
        # 実行中の索引消去を止めます。
        self.clear_button.setEnabled(not self.model.is_scanning)

    # 元フォルダを登録解除するメニューです。
    def _folder_menu(self, button: QPushButton, point: object, source_id: str) -> None:
        # 解析中は関連索引を変更しません。
        if self.model.is_scanning:
            # 右クリック操作を表示せず戻ります。
            return
        # フォルダ専用のメニューを作ります。
        menu = QMenu(button)
        # 登録解除の操作を追加します。
        remove = menu.addAction(tr("このフォルダを登録解除"))
        # 押されたら確認してから削除します。
        remove.triggered.connect(lambda: self._confirm_remove_source(source_id))
        # 右クリック位置にメニューを出します。
        menu.exec(button.mapToGlobal(point))

    # 登録元フォルダを外す前に影響を説明します。
    def _confirm_remove_source(self, source_id: str) -> None:
        # 元動画は消えず索引だけ消えることを示します。
        answer = confirm_deletion(self, tr("フォルダを登録解除"), tr("このフォルダの索引と生成画像を削除します。元動画は削除しません。続けますか？"))
        # 明示的に選ばれた場合だけ削除します。
        if answer:
            # フォルダの索引を削除します。
            self.model.remove_source(source_id)
            # 残った全動画の画面へ戻します。
            self._select_section("all")

    # 人物を右クリックしたとき手動操作を表示します。
    def _person_menu(self, button: QPushButton, point: object, person: PersonRecord) -> None:
        # 解析中は自動分類と同時に編集しません。
        if self.model.is_scanning:
            # メニューを出さず戻ります。
            return
        # 対象人物の操作メニューを作ります。
        menu = QMenu(button)
        # 表示名を変更する操作を追加します。
        rename = menu.addAction(tr("名前を付ける・変更する"))
        # 入力画面から名前を保存します。
        rename.triggered.connect(lambda: self._rename_person(person))
        # 別の人物へ統合する候補を並べます。
        merge = menu.addMenu(tr("別の人物へ統合"))
        # 統合先となり得る人物を調べます。
        for candidate in self.model.people:
            # 同じ人物への統合は表示しません。
            if candidate.id != person.id:
                # 名前がない場合も操作できるようにします。
                action = merge.addAction(candidate.name or tr("名前なしの人物"))
                # 対象IDを固定して手動統合します。
                action.triggered.connect(lambda checked=False, target=candidate.id: self.model.merge_people(person.id, target))
        # 人物写真の位置にメニューを出します。
        menu.exec(button.mapToGlobal(point))

    # 人物の名前入力画面を表示します。
    def _rename_person(self, person: PersonRecord) -> None:
        # 現在の名前を初期値として入力欄へ示します。
        value, accepted = QInputDialog.getText(self, tr("人物名"), tr("表示する名前を入力"), text=person.name)
        # 利用者が確定した名前だけ保存します。
        if accepted and value.strip():
            # モデルへ手動名称を保存します。
            self.model.rename_person(person.id, value)

    # 動画再生画面を指定時刻で開きます。
    def _open_player(self, video: VideoRecord, second: float = 0.0) -> None:
        # 消えた元ファイルを開こうとしません。
        if not Path(video.path).is_file():
            # 利用者へ欠落を知らせます。
            show_warning(self, tr("動画を開けません"), tr("元動画が見つかりません。フォルダを更新してください。"))
            # 再生画面を作らず終えます。
            return
        # 現在の動画と時刻をプレイヤーへ渡します。
        dialog = PlayerDialog(self.model, video, second, self)
        # 利用者が閉じるまで再生画面を表示します。
        dialog.exec()

    # インデックス全消去の対象を確認します。
    def _confirm_clear(self) -> None:
        # 元動画を残すことと、顔画像を消すことを表示します。
        answer = confirm_deletion(self, tr("インデックスを消去"), tr("登録フォルダ、解析結果、生成した顔画像を削除します。元動画は削除しません。続けますか？"))
        # はっきり削除を選んだ場合だけ実行します。
        if answer:
            # Python版専用索引を消します。
            self.model.clear_index()
            # 空の全動画画面へ戻します。
            self._select_section("all")

    # ファイルマネージャーからのフォルダドロップを受け取ります。
    def dragEnterEvent(self, event: object) -> None:
        # ドロップ内容にフォルダがあれば受け入れます。
        if any(Path(url.toLocalFile()).is_dir() for url in event.mimeData().urls()):
            # フォルダを追加できることを示します。
            event.acceptProposedAction()
        # ファイルだけなら動画フォルダの選択を待ちます。
        else:
            # このドロップは受け入れません。
            event.ignore()

    # 落とされたフォルダを順に登録します。
    def dropEvent(self, event: object) -> None:
        # URLをローカルのパスへ変えます。
        folders = [Path(url.toLocalFile()) for url in event.mimeData().urls() if Path(url.toLocalFile()).is_dir()]
        # 有効なフォルダをライブラリへ追加します。
        for folder in folders:
            # フォルダを登録し、動画を更新します。
            self.model.add_folder(folder)
        # macOSへドロップ処理の完了を伝えます。
        event.acceptProposedAction()

    # ウィンドウ終了時に解析とSQLiteを閉じます。
    def closeEvent(self, event: object) -> None:
        # 安全な区間で解析を止め、保存先を閉じます。
        self.model.close()
        # Qt の通常の閉じる処理へ進みます。
        super().closeEvent(event)


# Pythonから画面を起動する入口です。
def main() -> int:
    # データ保存先を上書きできる起動引数を定義します。
    parser = argparse.ArgumentParser(description="VideoAtlas Python")
    # スクリーンショットなどで別索引を使うための指定です。
    parser.add_argument("--data-dir", type=Path, default=None, help=tr("Python版の索引保存先"))
    # 既知の引数とQtに渡す残りを分けます。
    arguments, qt_arguments = parser.parse_known_args()
    # QtのmacOSアプリケーションを作ります。
    application = QApplication([sys.argv[0], *qt_arguments])
    # Python版専用の保存先を開きます。
    model = LibraryController(arguments.data_dir)
    # ウィンドウを作ります。
    window = MainWindow(model)
    # 利用者へ初期画面を表示します。
    window.show()
    # ウィンドウが閉じるまでQtイベントを処理します。
    return application.exec()
