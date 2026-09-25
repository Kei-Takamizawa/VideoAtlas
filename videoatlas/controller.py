# このファイルは画面、動画解析、保存処理をつなぎます。
"""VideoAtlas の画面用状態とバックグラウンド解析。"""

# 既存レコードを安全に複製するために使います。
from dataclasses import replace
# 動画ファイルと保存先の場所を扱います。
from pathlib import Path
# 解析停止と保存完了の通知をスレッド間で共有します。
from threading import Event
# 新しいレコードへ重複しない識別子を付けます。
from uuid import uuid4

# Qt のシグナルとバックグラウンドスレッドを使います。
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal, Slot

# 動画解析の結果を人物に慎重に割り当てる関数です。
from .grouping import classify_faces, reconcile_groups
# SQLite に保存するデータ型とファイル列挙を読み込みます。
from .storage import FaceRecord, LibraryStore, PersonRecord, SourceFolder, VideoRecord, list_video_files


# Python ファイルから同梱の OpenVINO モデルを探します。
MODEL_XML = Path(__file__).resolve().parents[1] / "Sources" / "VideoAtlas" / "Resources" / "Models" / "face-reidentification-retail-0095.xml"


# 動画を画面とは別のスレッドで順番に解析します。
class AnalysisWorker(QThread):
    # 動画を読み始めたことを画面へ伝えます。
    video_started = Signal(str)
    # 60 フレームまでの解析済み結果を画面へ渡します。
    chunk_ready = Signal(str, object)
    # 動画の進み具合を画面へ渡します。
    progressed = Signal(str, float, float)
    # 一件の解析失敗を他の動画から分けて伝えます。
    video_error = Signal(str, str)
    # 必要な解析エンジンを開けない失敗を伝えます。
    fatal_error = Signal(str)

    # 解析対象と停止指示を保存します。
    def __init__(self, videos: list[VideoRecord], stop_event: Event) -> None:
        # Qt スレッドの内部状態を初期化します。
        super().__init__()
        # 画面側の書き換えに影響されない動画の写しを保持します。
        self.videos = [replace(video) for video in videos]
        # 一時停止を受け取る共有イベントを保持します。
        self.stop_event = stop_event
        # 一区間の保存完了を待つイベントを用意します。
        self.chunk_saved = Event()

    # 一つずつ動画を開き、区間単位の結果を送ります。
    def run(self) -> None:
        # モデルや処理ライブラリを一度だけ準備します。
        try:
            # 解析モジュールを動画処理の時だけ読み込みます。
            from .analyzer import VideoAnalyzer
            # 顔特徴量用の OpenVINO モデルを渡します。
            analyzer = VideoAnalyzer(MODEL_XML)
        # 依存ライブラリが不足した場合は未処理動画を残します。
        except Exception as error:
            # 画面に必要な追加設定を説明します。
            self.fatal_error.emit(str(error))
            # 準備できない状態で動画へ進みません。
            return
        # 登録済みの未処理動画を順に扱います。
        for video in self.videos:
            # 一時停止されたら次の動画は開きません。
            if self.stop_event.is_set():
                # 完了処理へ進みます。
                break
            # 一件を開始したことを通知します。
            self.video_started.emit(video.id)
            # 動画ごとの失敗を次の動画へ持ち越さないようにします。
            try:
                # 保存済みの秒数から最大60フレーム単位で処理します。
                chunks = analyzer.iter_chunks(Path(video.path), interval=video.sample_interval, start=video.last_analyzed_second, stop=self.stop_event.is_set, progress=lambda second, duration: self.progressed.emit(video.id, second, duration))
                # 次の区間を得るまで解析スレッドで待ちます。
                for chunk in chunks:
                    # 次の結果と取り違えないよう完了フラグを消します。
                    self.chunk_saved.clear()
                    # 画面スレッドへ今回の区間を渡します。
                    self.chunk_ready.emit(video.id, chunk)
                    # SQLite の保存が終わるまで次の区間を始めません。
                    self.chunk_saved.wait()
                    # 保存エラーや一時停止なら次の区間を読みません。
                    if self.stop_event.is_set() or chunk.cancelled:
                        # この動画を続けず待機状態へ戻します。
                        break
            # 壊れた動画など一件の問題は理由を画面に残します。
            except Exception as error:
                # この動画の失敗として通知します。
                self.video_error.emit(video.id, str(error))


# 画面に表示するライブラリの現在値を管理します。
class LibraryController(QObject):
    # 一覧や人物を描き直す必要があることを伝えます。
    changed = Signal()
    # 進捗や状態文を描き直す必要があることを伝えます。
    status_changed = Signal()

    # 保存済み索引を読み込み、前回の途中状態を戻します。
    def __init__(self, root: Path | None = None) -> None:
        # QObject のシグナル機能を準備します。
        super().__init__()
        # Python 版専用の SQLite 保存先を開きます。
        self.store = LibraryStore(root)
        # 画像はデータベースと同じ保存先の下に生成します。
        self.thumbnail_dir = self.store.database_path.parent / "Thumbnails"
        # 生成画像を置くフォルダを作ります。
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        # 登録フォルダの画面用一覧です。
        self.sources: list[SourceFolder] = []
        # 動画の画面用一覧です。
        self.videos: list[VideoRecord] = []
        # 顔の画面用一覧です。
        self.faces: list[FaceRecord] = []
        # 人物の画面用一覧です。
        self.people: list[PersonRecord] = []
        # 解析中に説明する短い日本語です。
        self.status_text = ""
        # 全動画を通じた進捗割合です。
        self.progress = 0.0
        # 解析スレッドが稼働中か示します。
        self.is_scanning = False
        # 一時停止要求を保持します。
        self.stop_event = Event()
        # 動画解析スレッドを後から参照できるようにします。
        self.worker: AnalysisWorker | None = None
        # 走査中に頼まれたフォルダ更新を記憶します。
        self.refresh_after_scan = False
        # 終了処理中に新しい解析を開始しないための印です。
        self.closing = False
        # 解析エンジンの準備失敗を完了表示まで保持します。
        self.scan_failure_message: str | None = None
        # 現在の保存内容を画面用一覧へ読み込みます。
        self._reload()
        # 強制終了時に解析中だったレコードを再開待ちへ戻します。
        for video in self.videos:
            # 前回の進捗を保ったまま状態だけ変更します。
            if video.state == "analyzing":
                # 解析待ちへ戻します。
                video.state = "pending"
                # 修正した状態を保存します。
                self.store.save_video(video)

    # 保存済み四種類のレコードを読み直します。
    def _reload(self) -> None:
        # 一回の読み取りでレコードの組を取得します。
        snapshot = self.store.load_snapshot()
        # 登録フォルダを置き換えます。
        self.sources = snapshot.sources
        # 動画を置き換えます。
        self.videos = snapshot.videos
        # 顔を置き換えます。
        self.faces = snapshot.faces
        # 人物を置き換えます。
        self.people = snapshot.people
        # 画面へ変更を通知します。
        self.changed.emit()

    # 説明文と進捗をまとめて画面へ伝えます。
    def _status(self, message: str, progress: float | None = None) -> None:
        # 新しい説明文を保存します。
        self.status_text = message
        # 進捗が指定された場合だけ0から1へ収めます。
        if progress is not None:
            # 不正な表示にならない範囲で割合を更新します。
            self.progress = max(0.0, min(1.0, progress))
        # 状態欄を描き直します。
        self.status_changed.emit()

    # フォルダを登録し、その配下の動画を探します。
    def add_folder(self, path: Path) -> None:
        # 画面やドロップで渡された文字列を絶対パスにします。
        folder = Path(path).expanduser().resolve()
        # 動画ファイルそのものや存在しない場所は登録しません。
        if not folder.is_dir():
            # 利用者が選び直せるよう理由を表示します。
            self._status("動画を含むフォルダを選択してください。")
            # 不正な入力の処理を終えます。
            return
        # 同じフォルダがあれば二重登録を避けます。
        if not any(source.path == str(folder) for source in self.sources):
            # 新しいフォルダの保存レコードを作ります。
            source = SourceFolder(id=str(uuid4()), path=str(folder))
            # 再起動後も使えるようSQLiteへ保存します。
            self.store.save_source(source)
            # サイドバーにも反映します。
            self._reload()
        # 新規・変更動画を確認します。
        self.refresh_library()

    # 登録フォルダを調べて動画の追加、変更、欠落を反映します。
    def refresh_library(self) -> None:
        # アプリを閉じる途中なら新しい走査を始めません。
        if self.closing:
            # 保存先を閉じる処理を優先します。
            return
        # 解析中は終了後に一度だけ実行します。
        if self.is_scanning:
            # 更新依頼を保持します。
            self.refresh_after_scan = True
            # 現在の解析との競合を避けます。
            return
        # フォルダごとに失敗を分離します。
        for source in list(self.sources):
            # 読み取れないフォルダは他のフォルダまで止めません。
            try:
                # シンボリックリンクをたどらず動画を列挙します。
                files = list_video_files(Path(source.path))
            # 権限失効や移動を画面へ表示します。
            except Exception as error:
                # 元動画が消えたと決めつけず理由だけ表示します。
                self._status(f"{source.path} を確認できません：{error}")
                # 次のフォルダを確認します。
                continue
            # 今見つかったパスを比較用の集合にします。
            found = {str(path) for path in files}
            # このフォルダに保存されている既存動画を順番に調べます。
            for video in self.videos:
                # 他のフォルダの動画には触れません。
                if video.source_id != source.id:
                    # 次の動画へ進みます。
                    continue
                # ファイルが見つからなければ欠落と理由を記録します。
                if video.path not in found:
                    # 元ファイル不在を状態に反映します。
                    video.state = "missing"
                    # 見つからない場所を説明します。
                    video.error_message = "登録フォルダ内で元動画が見つかりません。"
                    # 変更を保存します。
                    self.store.save_video(video)
            # 見つかった動画を追加または更新します。
            for path in files:
                # 列挙後に消えたファイルもあるため属性取得を個別に扱います。
                try:
                    # ファイルのサイズと更新日時を取得します。
                    info = path.stat()
                # このファイルだけの読取失敗を表示します。
                except OSError as error:
                    # 他の動画は続けられるよう説明を残します。
                    self._status(f"{path.name} の情報を読めません：{error}")
                    # 次のファイルへ進みます。
                    continue
                # 同じ登録元とパスの動画を探します。
                current = next((video for video in self.videos if video.source_id == source.id and video.path == str(path)), None)
                # 初めて見つかった動画を待機状態で作ります。
                if current is None:
                    # 解析前でも一覧へ置ける情報を作ります。
                    current = VideoRecord(str(uuid4()), source.id, str(path), path.name, 0.0, info.st_size, info.st_mtime, "pending", 0.0, 2.0, None, None)
                    # 新しい動画を保存します。
                    self.store.save_video(current)
                    # 同じ更新中に重複登録しないよう一覧へ追加します。
                    self.videos.append(current)
                    # 次のファイルへ進みます。
                    continue
                # 内容が変わったかサイズと更新日時で判定します。
                changed = current.file_size != info.st_size or current.modification_time != info.st_mtime
                # 内容が変わった場合は以前の顔とポスターを破棄します。
                if changed:
                    # 旧動画から生成した画像を安全な保存先だけで削除します。
                    self._remove_video_images(current.id)
                    # 保存済みの顔を動画IDで削除します。
                    for face in [face for face in self.faces if face.video_id == current.id]:
                        # 旧内容の顔レコードを削除します。
                        self.store.delete_face(face.id)
                    # 解析を先頭からやり直します。
                    current.last_analyzed_second = 0.0
                    # 古い長さを表示しません。
                    current.duration = 0.0
                    # 古い代表画像への参照を消します。
                    current.poster_path = None
                # 変更後または欠落復帰後は解析待ちにします。
                if changed or current.state == "missing":
                    # 新しいファイル情報を保存します。
                    current.file_size = info.st_size
                    # 更新時刻も新しい値にします。
                    current.modification_time = info.st_mtime
                    # 次の解析対象にします。
                    current.state = "pending"
                    # 前回のエラー説明を消します。
                    current.error_message = None
                    # 更新後のレコードをSQLiteへ保存します。
                    self.store.save_video(current)
        # 更新結果を一度読み直します。
        self._reload()
        # 参照されなくなった人物を人物一覧から除きます。
        self._remove_orphan_people()
        # 解析待ち動画があれば作業を始めます。
        self.resume_scanning()

    # 解析待ち動画をバックグラウンドで処理します。
    def resume_scanning(self) -> None:
        # 終了処理中には新たな解析を開始しません。
        if self.closing:
            # 現在の終了処理へ戻ります。
            return
        # 同時に二つの解析を動かしません。
        if self.is_scanning:
            # 現在の解析を続けます。
            return
        # 対象になる動画を現在の順番で選びます。
        pending = [video for video in self.videos if video.state == "pending"]
        # 待機動画がなければ画面を変えません。
        if not pending:
            # 作業を始めずに戻ります。
            return
        # 前回の一時停止指示を解除します。
        self.stop_event.clear()
        # 前回のエンジン失敗を新しい処理へ持ち込みません。
        self.scan_failure_message = None
        # 進捗計算に使う対象件数を保存します。
        self.scan_total = len(pending)
        # 進捗計算に使う完了件数を初期化します。
        self.scan_done = 0
        # 解析中の表示に切り替えます。
        self.is_scanning = True
        # 解析スレッドを作ります。
        self.worker = AnalysisWorker(pending, self.stop_event)
        # 動画開始通知を画面スレッドで処理します。
        self.worker.video_started.connect(self._video_started)
        # 一区間の結果を画面スレッドで保存します。
        self.worker.chunk_ready.connect(self._accept_chunk)
        # 動画内の進み具合を表示します。
        self.worker.progressed.connect(self._progressed)
        # 一件の失敗を状態へ反映します。
        self.worker.video_error.connect(self._video_error)
        # 解析器の準備失敗を表示します。
        self.worker.fatal_error.connect(self._fatal_error)
        # 全件処理後に表示を戻します。
        self.worker.finished.connect(self._scan_finished)
        # 最初の状態文を表示します。
        self._status(f"解析を開始します（{len(pending)}件）", 0.0)
        # Qt のバックグラウンド処理を開始します。
        self.worker.start()

    # 停止指示を次の安全な保存区間で反映します。
    def pause_scanning(self) -> None:
        # 解析中でなければ操作は不要です。
        if not self.is_scanning:
            # 何も変更せず終えます。
            return
        # 解析スレッドへ停止を知らせます。
        self.stop_event.set()
        # 保存が済むまで画面に途中状態を知らせます。
        self._status("一時停止して途中結果を保存しています…")

    # 指定動画の解析開始を画面に反映します。
    @Slot(str)
    def _video_started(self, video_id: str) -> None:
        # 動画IDに一致する画面用レコードを探します。
        video = self.video_by_id(video_id)
        # 削除済み動画の通知は無視します。
        if video is None:
            # 状態を変更せず戻ります。
            return
        # 現在の動画を解析中にします。
        video.state = "analyzing"
        # 再起動後にも中断を見つけられるよう保存します。
        self.store.save_video(video)
        # 動画名を進捗欄に表示します。
        self._status(f"解析中：{video.name}")
        # カードの状態表示を更新します。
        self.changed.emit()

    # 動画の進捗を全件に対する割合へ変換します。
    @Slot(str, float, float)
    def _progressed(self, video_id: str, second: float, duration: float) -> None:
        # 長さが0でもゼロ除算しない割合を作ります。
        part = max(0.0, min(1.0, second / max(duration, 1.0)))
        # 完了済み件数と現在位置を合わせます。
        fraction = (self.scan_done + part) / max(self.scan_total, 1)
        # 現在の説明文を残して進捗だけ更新します。
        self._status(self.status_text, fraction)

    # 保存区間の顔、人物、再開位置を一括で保存します。
    @Slot(str, object)
    def _accept_chunk(self, video_id: str, chunk: object) -> None:
        # 保存処理が失敗してもワーカースレッドを待たせ続けません。
        try:
            # 対象動画が残っているか確認します。
            video = self.video_by_id(video_id)
            # 既に削除された動画なら結果を捨てます。
            if video is None:
                # 保存する対象がありません。
                return
            # 新しい検出顔を既存の顔・人物へ慎重に分類します。
            faces, people, removed_ids = classify_faces(chunk.faces, video, self.faces, self.people, self.thumbnail_dir, chunk.completed_second, chunk.duration)
            # 動画の総秒数を新しいメディア値へ更新します。
            video.duration = chunk.duration
            # 次回の再開位置を保存します。
            video.last_analyzed_second = chunk.completed_second
            # 全体を読み終えた時だけ完了にします。
            video.state = "ready" if not chunk.cancelled and chunk.completed_second >= chunk.duration else "pending"
            # 成功した区間では古いエラーを消します。
            video.error_message = None
            # 最初の画像だけをポスターとして保存します。
            if video.poster_path is None and chunk.poster_jpeg:
                # 動画IDで安全な画像ファイル名を作ります。
                poster = self.thumbnail_dir / f"poster-{video.id}.jpg"
                # JPEGのバイト列をディスクへ書きます。
                poster.write_bytes(chunk.poster_jpeg)
                # 動画レコードから画像を参照します。
                video.poster_path = str(poster)
            # 顔の置換、人物作成、再開位置を一つのトランザクションで保存します。
            self.store.apply_face_batch(faces, video, people, removed_ids)
            # 確定した内容を画面へ読み込み直します。
            self._reload()
            # 一件が終わったときだけ全件進捗を増やします。
            if video.state == "ready":
                # 次の動画を含める進捗へ更新します。
                self.scan_done += 1
        # 保存や分類に失敗した場合は動画一件の失敗にします。
        except Exception as error:
            # 後続の区間を処理せず整合性を保ちます。
            self.stop_event.set()
            # この動画の状態と理由を更新します。
            self._video_error(video_id, f"解析結果を保存できません：{error}")
        # 正常時も失敗時も解析スレッドを再開させます。
        finally:
            # ワーカーが次の区間を処理できるよう通知します。
            if self.worker is not None:
                # 保存待ちのイベントを解除します。
                self.worker.chunk_saved.set()

    # 一件の失敗を他の動画と区別して保存します。
    @Slot(str, str)
    def _video_error(self, video_id: str, message: str) -> None:
        # 失敗した対象動画を探します。
        video = self.video_by_id(video_id)
        # 既に削除された動画には変更を加えません。
        if video is None:
            # 画面への説明だけで終えます。
            self._status(message)
            # 後続処理を省きます。
            return
        # 状態を明示的な失敗にします。
        video.state = "failed"
        # カードで読めるよう理由を残します。
        video.error_message = message
        # 再起動後にも理由を残します。
        self.store.save_video(video)
        # 処理済み件数に一件加えます。
        self.scan_done += 1
        # 失敗した動画名を表示します。
        self._status(f"解析失敗：{video.name}：{message}")
        # カードを描き直します。
        self.changed.emit()

    # 解析エンジンの設定失敗を表示します。
    @Slot(str)
    def _fatal_error(self, message: str) -> None:
        # 完了シグナル後にも正しい理由を表示できるよう保存します。
        self.scan_failure_message = message
        # 依存パッケージやモデルの問題を具体的に表示します。
        self._status(f"解析を開始できません：{message}")

    # 全件処理後に一時停止か完了を反映します。
    @Slot()
    def _scan_finished(self) -> None:
        # 停止指示があったか結果表示に使います。
        was_paused = self.stop_event.is_set()
        # 解析中の動画が残っていれば再開待ちへ戻します。
        for video in self.videos:
            # 強制終了も再開できる状態へそろえます。
            if video.state == "analyzing":
                # 次回の開始位置を残して待機状態にします。
                video.state = "pending"
                # 待機状態を保存します。
                self.store.save_video(video)
        # Qt の解析スレッドへの参照を外します。
        self.worker = None
        # 画面上の実行中フラグを解除します。
        self.is_scanning = False
        # 解析エンジンを開けない場合は詳しい原因を優先します。
        if self.scan_failure_message is not None:
            # 失敗後も再設定できるよう待機動画は残します。
            self._status(f"解析を開始できません：{self.scan_failure_message}")
        # 一時停止の文言を優先します。
        elif was_paused:
            # ユーザーの停止操作に合わせた説明です。
            self._status("一時停止しました。")
        # 未処理がなく、複数動画が完了した場合だけ慎重な照合をします。
        elif not any(video.state == "pending" for video in self.videos):
            # 異なる動画の人物グループを安全側の条件で調べます。
            self._reconcile_people()
            # 失敗した動画の件数を集めます。
            failed_count = sum(video.state == "failed" for video in self.videos)
            # 完了後も失敗理由がカードにあることを知らせます。
            message = f"解析終了：{failed_count}件の失敗理由を動画カードに表示しています。" if failed_count else "解析が完了しました。"
            # 最終状態を画面に表示します。
            self._status(message, 1.0)
        # 未処理が残っている場合は再開可能と表示します。
        else:
            # 終了した理由がエラー欄から分かるようにします。
            self._status("解析が止まりました。状態欄を確認してください。")
        # 全画面を最新の状態へ更新します。
        self.changed.emit()
        # 解析中に追加されたフォルダがあれば更新します。
        if self.refresh_after_scan and not self.closing:
            # 一度だけ再走査するためフラグを消します。
            self.refresh_after_scan = False
            # 追加フォルダの動画を確認します。
            self.refresh_library()

    # 安全な候補だけ別動画の人物グループをまとめます。
    def _reconcile_people(self) -> None:
        # 二件未満の準備済み動画に照合対象はありません。
        if sum(video.state == "ready" for video in self.videos) < 2:
            # 不要な比較を行いません。
            return
        # 手動修正と同時出現を守った人物IDの対応を求めます。
        mapping = reconcile_groups(self.faces, self.videos, self.people)
        # 統合元の人物が存在する組だけ更新します。
        for source_id, target_id in mapping.items():
            # 統合元の全顔を対象にします。
            for face in self.faces:
                # 関係ない顔の割当は変更しません。
                if face.person_id == source_id:
                    # 自動統合先へ割り当てます。
                    face.person_id = target_id
                    # 更新した顔を永続化します。
                    self.store.save_face(face)
            # 参照がなくなった元人物を削除します。
            self.store.delete_person(source_id)
        # 変更があればデータベースから再読込します。
        if mapping:
            # 人物タイルと顔一覧を更新します。
            self._reload()

    # 動画IDから現在のレコードを返します。
    def video_by_id(self, video_id: str) -> VideoRecord | None:
        # 一致する最初の動画を取得します。
        return next((video for video in self.videos if video.id == video_id), None)

    # 顔IDから現在のレコードを返します。
    def face_by_id(self, face_id: str) -> FaceRecord | None:
        # 一致する最初の顔を取得します。
        return next((face for face in self.faces if face.id == face_id), None)

    # 人物名を手動で変更します。
    def rename_person(self, person_id: str, name: str) -> None:
        # 解析中は自動分類との競合を避けます。
        if self.is_scanning:
            # 停止後の操作を案内します。
            self._status("解析を一時停止してから人物を編集してください。")
            # 変更せず戻ります。
            return
        # IDで人物を探します。
        person = next((item for item in self.people if item.id == person_id), None)
        # 空白名や存在しない人物は変更しません。
        if person is None or not name.strip():
            # 既存の名前を保持します。
            return
        # 前後の空白を除いて名前を保存します。
        person.name = name.strip()
        # データベースへ更新を反映します。
        self.store.save_person(person)
        # 人物タイルの表示を更新します。
        self.changed.emit()

    # 二つの人物グループを利用者の判断で統合します。
    def merge_people(self, source_id: str, target_id: str) -> None:
        # 解析中の編集や自分自身への統合を防ぎます。
        if self.is_scanning or source_id == target_id:
            # データを変更しません。
            return
        # 両方の人物が存在することを確認します。
        if not all(any(person.id == identifier for person in self.people) for identifier in (source_id, target_id)):
            # 古い画面選択からの操作を無視します。
            return
        # 元人物に属する顔を一件ずつ更新します。
        for face in self.faces:
            # 統合元以外には触れません。
            if face.person_id == source_id:
                # 統合先へ割り当てます。
                face.person_id = target_id
                # 利用者の手動判断を保護します。
                face.manual_assignment = True
                # 変更をSQLiteへ保存します。
                self.store.save_face(face)
        # 統合元の人物情報を除きます。
        self.store.delete_person(source_id)
        # 画面を最新の保存内容へ戻します。
        self._reload()

    # 一つの顔を新しい人物として分割します。
    def split_face(self, face_id: str) -> None:
        # 解析中は割当を変更しません。
        if self.is_scanning:
            # 一時停止を案内します。
            self._status("解析を一時停止してから顔を編集してください。")
            # 処理を終えます。
            return
        # 変更する顔を探します。
        face = self.face_by_id(face_id)
        # 顔が存在しなければ何もしません。
        if face is None:
            # 選択の変更を待ちます。
            return
        # 名前なしの新人物を作ります。
        person = PersonRecord(str(uuid4()), "", face.thumbnail_path)
        # 顔より先に人物を保存します。
        self.store.save_person(person)
        # 新しい人物へ顔を移します。
        face.person_id = person.id
        # 利用者が決めた分類として固定します。
        face.manual_assignment = True
        # 顔側の変更を保存します。
        self.store.save_face(face)
        # 最新の人物タイルを表示します。
        self._reload()

    # 顔を既存人物または未分類へ移します。
    def assign_face(self, face_id: str, person_id: str | None) -> None:
        # 解析中の編集を止めます。
        if self.is_scanning:
            # 一時停止後に操作するよう伝えます。
            self._status("解析を一時停止してから顔を編集してください。")
            # 割当を変えません。
            return
        # 対象顔を探します。
        face = self.face_by_id(face_id)
        # 変更先人物が実在するか確かめます。
        target_exists = person_id is None or any(person.id == person_id for person in self.people)
        # 古い画面選択なら変更しません。
        if face is None or not target_exists:
            # 入力を無視します。
            return
        # 指定された人物IDまたは未分類へ変えます。
        face.person_id = person_id
        # 手動修正として保護します。
        face.manual_assignment = True
        # 一覧で再表示できる状態へ戻します。
        face.excluded = False
        # SQLite に保存します。
        self.store.save_face(face)
        # 人物と顔の表示を更新します。
        self._reload()

    # 間違った検出を人物一覧から除外します。
    def exclude_face(self, face_id: str) -> None:
        # 解析中には編集しません。
        if self.is_scanning:
            # 操作できる時点を伝えます。
            self._status("解析を一時停止してから顔を編集してください。")
            # 現在の割当を保持します。
            return
        # 除外する顔を探します。
        face = self.face_by_id(face_id)
        # 存在しない顔は変更しません。
        if face is None:
            # 処理を終えます。
            return
        # 顔を画面と照合対象から除きます。
        face.excluded = True
        # 利用者の操作であることを記録します。
        face.manual_assignment = True
        # 保存済み値を更新します。
        self.store.save_face(face)
        # 画面を更新します。
        self._reload()

    # 詳細または通常の間隔で指定動画を先頭から調べ直します。
    def rescan_video(self, video_id: str, detailed: bool = True) -> None:
        # 実行中の再解析は現在の作業と競合します。
        if self.is_scanning:
            # 現在の解析を優先します。
            return
        # 動画を探します。
        video = self.video_by_id(video_id)
        # 元動画のないレコードは対象にしません。
        if video is None or not Path(video.path).is_file():
            # 選択先を変えず戻ります。
            return
        # 詳細なら0.5秒、通常なら2秒にします。
        video.sample_interval = 0.5 if detailed else 2.0
        # 旧手動修正との対応を再解析で引き継げるよう顔は残します。
        video.last_analyzed_second = 0.0
        # 先頭から再解析する状態にします。
        video.state = "pending"
        # 古い失敗理由を消します。
        video.error_message = None
        # 再解析予定をSQLiteに保存します。
        self.store.save_video(video)
        # カードを更新します。
        self.changed.emit()
        # 対象動画の再解析を始めます。
        self.resume_scanning()

    # 指定フォルダの索引だけを削除します。
    def remove_source(self, source_id: str) -> None:
        # 解析中の削除で保存先が変わらないようにします。
        if self.is_scanning:
            # 利用者へ停止を案内します。
            self._status("解析を一時停止してからフォルダを削除してください。")
            # 変更せず戻ります。
            return
        # このフォルダの動画IDを集めます。
        video_ids = [video.id for video in self.videos if video.source_id == source_id]
        # 関連する生成画像だけ先に消します。
        for video_id in video_ids:
            # 顔画像とポスターを安全な場所から削除します。
            self._remove_video_images(video_id)
        # SQLite の外部キー連鎖で動画と顔を除きます。
        self.store.delete_source(source_id)
        # 残ったレコードを画面へ反映します。
        self._reload()
        # 使われなくなった人物を除きます。
        self._remove_orphan_people()

    # 索引と生成画像だけを消し、元動画は残します。
    def clear_index(self) -> None:
        # 解析中に削除すると保存スレッドと競合します。
        if self.is_scanning:
            # 一時停止後に実行するよう伝えます。
            self._status("解析を一時停止してから索引を消去してください。")
            # 現在のデータは残します。
            return
        # 登録情報と顔のレコードをSQLiteから消します。
        self.store.clear_index()
        # このアプリが作ったJPEGだけを画像フォルダから削除します。
        for image in self.thumbnail_dir.glob("*.jpg"):
            # 対象はアプリ専用ディレクトリ内のファイルです。
            image.unlink(missing_ok=True)
        # 空になった索引を画面へ読み直します。
        self._reload()
        # 利用者へ削除範囲を説明します。
        self._status("索引と生成画像を消去しました。元動画は残っています。", 0.0)

    # 一件の動画から作った顔写真とポスターを安全に削除します。
    def _remove_video_images(self, video_id: str) -> None:
        # 削除対象の画像パスを収集します。
        paths = [Path(face.thumbnail_path) for face in self.faces if face.video_id == video_id]
        # この動画の代表画像も候補にします。
        video = self.video_by_id(video_id)
        # 代表画像があれば一覧へ追加します。
        if video is not None and video.poster_path:
            # ポスター画像のパスを候補へ加えます。
            paths.append(Path(video.poster_path))
        # 外部ファイルを誤って削除しないよう場所を確認します。
        for path in paths:
            # アプリ専用画像フォルダ直下の画像だけ削除します。
            if path.parent == self.thumbnail_dir and path.suffix.lower() == ".jpg":
                # 既に消えていても処理を続けます。
                path.unlink(missing_ok=True)

    # 有効な顔が一件も参照しない人物を除きます。
    def _remove_orphan_people(self) -> None:
        # 現在使われる人物IDを集合にします。
        active_ids = {face.person_id for face in self.faces if face.person_id is not None}
        # 参照がない人物だけを削除します。
        for person in self.people:
            # 顔から参照される人物は残します。
            if person.id not in active_ids:
                # 孤立した人物情報を削除します。
                self.store.delete_person(person.id)
        # 人物削除後の画面を読み直します。
        self._reload()

    # ウィンドウ終了時に接続とスレッドを片付けます。
    def close(self) -> None:
        # 終了待ちの間は新しい動画解析を起動しません。
        self.closing = True
        # 走査中なら次の保存可能な位置で止めます。
        self.stop_event.set()
        # 現在のスレッド参照を終了待ちの間だけ固定します。
        running = self.worker
        # 解析スレッドがあれば画面の保存通知も処理しながら待ちます。
        if running is not None:
            # 解析中のQtシグナルを処理できるよう短時間ずつ待ちます。
            while running.isRunning():
                # 保存区間のシグナルを画面スレッドで処理します。
                QCoreApplication.processEvents()
                # 50ミリ秒だけスレッド終了を待ちます。
                running.wait(50)
            # 最後の終了通知もSQLiteを閉じる前に反映します。
            QCoreApplication.processEvents()
        # SQLite のハンドルを閉じます。
        self.store.close()
