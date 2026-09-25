# このモジュールはVideoAtlasのメタデータ保存と動画ファイル列挙を担当します。
"""SQLite persistence for the Python VideoAtlas application."""

# 型付きデータクラスを使うための標準ライブラリです。
from dataclasses import asdict, dataclass
# JSONは顔特徴量などの可変長データを保存するために使います。
import json
# os.walkはシンボリックリンクを追跡しない再帰列挙に使います。
import os
# sqlite3はアプリ内蔵のSQLiteデータベース接続を提供します。
import sqlite3
# pathlibはパスの結合とファイル判定を読みやすくします。
from pathlib import Path
# 型注釈でリストや任意値を明示します。
from typing import Any


# 登録フォルダのIDとパスを保持します。
@dataclass
class SourceFolder:
    # フォルダを一意に識別する文字列IDです。
    id: str
    # 登録したフォルダのパスです。
    path: str


# 動画解析の状態とファイル情報を保持します。
@dataclass
class VideoRecord:
    # 動画レコードの一意なIDです。
    id: str
    # 動画を登録したフォルダのIDです。
    source_id: str
    # 動画ファイルの完全なパスです。
    path: str
    # 画面に表示する動画名です。
    name: str
    # 動画の長さを秒で表します。
    duration: float
    # 動画ファイルのサイズをバイトで表します。
    file_size: int
    # ファイルの更新時刻をUnix秒で表します。
    modification_time: float
    # pending、analyzing、ready、missing、failedのいずれかです。
    state: str
    # 最後に解析した動画位置を秒で表します。
    last_analyzed_second: float
    # 解析時に使ったサンプル間隔を秒で表します。
    sample_interval: float
    # 解析エラーなどの説明です。
    error_message: str | None
    # 生成済みポスター画像のパスです。
    poster_path: str | None


# 動画から見つかった顔と人物割当を保持します。
@dataclass
class FaceRecord:
    # 顔レコードの一意なIDです。
    id: str
    # 顔が見つかった動画のIDです。
    video_id: str
    # 動画内で顔が見つかった時刻を秒で表します。
    second: float
    # x、y、幅、高さの正規化またはピクセル座標です。
    bounding_box: tuple[float, float, float, float] | None
    # 顔を割り当てた人物のIDです。
    person_id: str | None
    # 顔サムネイル画像のパスです。
    thumbnail_path: str
    # 顔特徴量の数値列です。
    embedding: list[float] | None
    # 人が割当を手動変更したかを表します。
    manual_assignment: bool
    # 顔を人物一覧や集計から除外するかを表します。
    excluded: bool


# 人物の名前と代表画像を保持します。
@dataclass
class PersonRecord:
    # 人物レコードの一意なIDです。
    id: str
    # 人物の表示名です。
    name: str
    # 代表サムネイル画像のパスです。
    thumbnail_path: str | None


# データベースから読み出した全レコードをまとめます。
@dataclass
class LibrarySnapshot:
    # 登録フォルダの一覧です。
    sources: list[SourceFolder]
    # 登録動画の一覧です。
    videos: list[VideoRecord]
    # 検出済み顔の一覧です。
    faces: list[FaceRecord]
    # 人物の一覧です。
    people: list[PersonRecord]


# SQLiteに固有の処理エラーを表します。
class LibraryStoreError(RuntimeError):
    # 呼び出し側が保存層の失敗を識別できるようにします。
    pass


# SQLiteを使ってライブラリデータを読み書きします。
class LibraryStore:
    # 保存先を受け取り、接続とテーブルを準備します。
    def __init__(self, root: Path | None = None) -> None:
        # root省略時はユーザーのApplication Support配下を使います。
        base = Path.home() / "Library" / "Application Support" / "VideoAtlasPython" if root is None else Path(root)
        # 保存先ディレクトリがなければ作成します。
        base.mkdir(parents=True, exist_ok=True)
        # データベースファイルを保存先ディレクトリ内に決めます。
        self.database_path = base / "library.sqlite3"
        # SQLite接続を開き、別接続の書き込み完了を最大5秒待ちます。
        self._connection = sqlite3.connect(self.database_path, timeout=5.0)
        # 行を列番号でなく列名でも参照できるようにします。
        self._connection.row_factory = sqlite3.Row
        # 外部キーのCASCADE削除を有効にします。
        self._connection.execute("PRAGMA foreign_keys = ON")
        # 読み込みと書き込みの競合を抑えるWALモードを有効にします。
        self._connection.execute("PRAGMA journal_mode = WAL")
        # 必要なテーブルが存在する状態を作ります。
        self._create_tables()

    # テーブル定義を一度に作成します。
    def _create_tables(self) -> None:
        # 接続をまとめて確定またはロールバックする文脈にします。
        with self._connection:
            # 登録フォルダの基本情報を保存するテーブルを作ります。
            self._connection.execute("CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)")
            # source削除時に動画も削除されるよう外部キーを設定します。
            self._connection.execute("CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, payload TEXT NOT NULL)")
            # 人物情報を保存するテーブルを作ります。
            self._connection.execute("CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)")
            # 動画削除時に顔を消し、人物削除時は割当だけを解除します。
            self._connection.execute("CREATE TABLE IF NOT EXISTS faces (id TEXT PRIMARY KEY NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE, person_id TEXT REFERENCES people(id) ON DELETE SET NULL, payload TEXT NOT NULL)")

    # 全テーブルを読み込み、型付きスナップショットとして返します。
    def load_snapshot(self) -> LibrarySnapshot:
        # 登録フォルダ行をデータクラスに復元します。
        sources = [SourceFolder(**self._decode(row["payload"])) for row in self._rows("sources")]
        # 動画行をデータクラスに復元します。
        videos = [VideoRecord(**self._decode(row["payload"])) for row in self._rows("videos")]
        # 顔行のJSON値を顔データ型に合わせて復元します。
        faces = [self._face_from_dict(self._decode(row["payload"])) for row in self._rows("faces")]
        # 人物行をデータクラスに復元します。
        people = [PersonRecord(**self._decode(row["payload"])) for row in self._rows("people")]
        # 4種類のレコードを1つの結果にまとめます。
        return LibrarySnapshot(sources=sources, videos=videos, faces=faces, people=people)

    # 登録フォルダを追加または更新します。
    def save_source(self, source: SourceFolder) -> None:
        # 親レコードを持たないsource行をupsertします。
        self._upsert("sources", source.id, asdict(source))

    # 動画情報を追加または更新します。
    def save_video(self, video: VideoRecord) -> None:
        # source_id外部キーと動画JSONを同時に保存します。
        self._upsert("videos", video.id, asdict(video), parent_column="source_id", parent_id=video.source_id)

    # 人物情報を追加または更新します。
    def save_person(self, person: PersonRecord) -> None:
        # 親レコードを持たないpeople行をupsertします。
        self._upsert("people", person.id, asdict(person))

    # 顔情報を追加または更新します。
    def save_face(self, face: FaceRecord) -> None:
        # video_idと任意のperson_idを外部キー列として保存します。
        self._upsert("faces", face.id, asdict(face), parent_column="video_id", parent_id=face.video_id, nullable_column="person_id", nullable_id=face.person_id)

    # 顔の一括保存と動画進捗の更新を1トランザクションにします。
    def save_batch(self, faces: list[FaceRecord], video: VideoRecord) -> None:
        # 他動画の顔が混ざったバッチは保存前に拒否します。
        if any(face.video_id != video.id for face in faces):
            # 原因が分かる例外を出して誤った関連付けを防ぎます。
            raise ValueError("すべての顔のvideo_idはvideo.idと一致する必要があります")
        # 内部で直接SQLを使う一括保存処理へ委譲して原子性を保ちます。
        self.apply_face_batch(faces, video, [], [])

    # 人物、顔、動画進捗を一つのSQLiteトランザクションで更新します。
    def apply_face_batch(self, faces: list[FaceRecord], video: VideoRecord, people: list[PersonRecord], remove_face_ids: list[str]) -> None:
        # 他動画の顔が混ざったバッチは保存前に拒否します。
        if any(face.video_id != video.id for face in faces):
            # 対象動画の取り違えを明示します。
            raise ValueError("すべての顔のvideo_idはvideo.idと一致する必要があります")
        # すべてのSQLを同じ接続トランザクション内で実行します。
        with self._connection:
            # 今回再解析した範囲の古い顔IDだけを削除します。
            for face_id in remove_face_ids:
                # プレースホルダーで顔IDを安全に指定します。
                self._connection.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            # 新しい顔が参照する人物を先に保存します。
            for person in people:
                # 人物データをJSONへ変換します。
                person_payload = json.dumps(asdict(person), ensure_ascii=False, separators=(",", ":"))
                # 人物レコードを追加または置き換えます。
                self._connection.execute("INSERT INTO people(id, payload) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (person.id, person_payload))
            # 今回見つけた顔を順番に保存します。
            for face in faces:
                # 顔データをJSONへ変換します。
                face_payload = json.dumps(asdict(face), ensure_ascii=False, separators=(",", ":"))
                # 顔本体と外部キー列をまとめて追加または置き換えます。
                self._connection.execute("INSERT INTO faces(id, video_id, person_id, payload) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET video_id=excluded.video_id, person_id=excluded.person_id, payload=excluded.payload", (face.id, face.video_id, face.person_id, face_payload))
            # 顔バッチと同じ確定点で動画の進捗情報をJSONへ変換します。
            video_payload = json.dumps(asdict(video), ensure_ascii=False, separators=(",", ":"))
            # 動画の解析状態と再開位置を追加または更新します。
            self._connection.execute("INSERT INTO videos(id, source_id, payload) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET source_id=excluded.source_id, payload=excluded.payload", (video.id, video.source_id, video_payload))

    # 動画と関連する顔を外部キー連鎖で削除します。
    def delete_video(self, identifier: str) -> None:
        # 1つの接続トランザクションで動画を削除します。
        with self._connection:
            # faces.video_idのCASCADEにより関連顔も削除されます。
            self._connection.execute("DELETE FROM videos WHERE id = ?", (identifier,))

    # 登録フォルダを削除し、外部キーCASCADEで動画と顔も削除します。
    def delete_source(self, identifier: str) -> None:
        # フォルダ削除を接続トランザクションとして実行します。
        with self._connection:
            # videos.source_idとfaces.video_idのCASCADEを適用します。
            self._connection.execute("DELETE FROM sources WHERE id = ?", (identifier,))

    # 人物を削除し、顔側の人物参照を解除します。
    def delete_person(self, identifier: str) -> None:
        # 1つの接続トランザクションで人物を削除します。
        with self._connection:
            # faces.person_idのSET NULLにより顔データは残ります。
            self._connection.execute("DELETE FROM people WHERE id = ?", (identifier,))

    # 顔を1件削除します。
    def delete_face(self, identifier: str) -> None:
        # 1つの接続トランザクションで顔を削除します。
        with self._connection:
            # IDをパラメーターとして渡して対象行だけを消します。
            self._connection.execute("DELETE FROM faces WHERE id = ?", (identifier,))

    # ライブラリに登録されたフォルダ、動画、顔、人物をすべて削除します。
    def clear_index(self) -> None:
        # 全テーブル削除を1つのトランザクションでまとめます。
        with self._connection:
            # 顔を先に消して動画や人物への参照を解放します。
            self._connection.execute("DELETE FROM faces")
            # 動画を消してsourceへの参照を解放します。
            self._connection.execute("DELETE FROM videos")
            # 人物一覧を空にします。
            self._connection.execute("DELETE FROM people")
            # 登録フォルダ一覧を空にします。
            self._connection.execute("DELETE FROM sources")

    # データベース接続を明示的に閉じます。
    def close(self) -> None:
        # SQLiteのファイルハンドルを解放します。
        self._connection.close()

    # with構文の終了時にデータベース接続を閉じます。
    def __enter__(self) -> "LibraryStore":
        # このインスタンス自身をwithブロックへ渡します。
        return self

    # with構文の終了時に接続を閉じます。
    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        # 成功・失敗のどちらでも接続を解放します。
        self.close()

    # 指定テーブルの保存行を追加順に取得します。
    def _rows(self, table: str) -> list[sqlite3.Row]:
        # 内部呼び出し限定のテーブル名でpayload列を読むSQLを実行します。
        return self._connection.execute(f"SELECT payload FROM {table} ORDER BY rowid").fetchall()

    # 保存済みJSON文字列をPython値へ変換します。
    @staticmethod
    def _decode(payload: str) -> dict[str, Any]:
        # JSONの壊れや型不整合は標準例外として呼び出し元へ伝えます。
        return json.loads(payload)

    # 顔JSON内の矩形配列をタプルに戻して顔レコードを作ります。
    @staticmethod
    def _face_from_dict(values: dict[str, Any]) -> FaceRecord:
        # JSON配列として復元された矩形をイミュータブルな4要素タプルにします。
        box = values.get("bounding_box")
        # 矩形値がない場合はNoneのままにします。
        values["bounding_box"] = tuple(box) if box is not None else None
        # JSONの数値配列をfloat特徴量として明示します。
        embedding = values.get("embedding")
        # 特徴量がある場合だけfloat列へ変換します。
        values["embedding"] = [float(value) for value in embedding] if embedding is not None else None
        # 変換済み値から顔レコードを組み立てます。
        return FaceRecord(**values)

    # JSON本文と外部キー列を安全なパラメーター付きSQLで保存します。
    def _upsert(self, table: str, identifier: str, value: dict[str, Any], parent_column: str | None = None, parent_id: str | None = None, nullable_column: str | None = None, nullable_id: str | None = None) -> None:
        # JSONをUnicodeを保ったまま安定した文字列に変換します。
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        # テーブルごとのSQL列構成を選びます。
        if table == "faces":
            # 顔テーブルは動画IDとNULL可の人物IDも保存します。
            sql = "INSERT INTO faces(id, video_id, person_id, payload) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET video_id=excluded.video_id, person_id=excluded.person_id, payload=excluded.payload"
            # SQLの値はプレースホルダーに分離して渡します。
            parameters = (identifier, parent_id, nullable_id, payload)
        # 動画テーブルには登録元フォルダIDを保存します。
        elif table == "videos":
            # source_id外部キーと本文を同時にupsertします。
            sql = "INSERT INTO videos(id, source_id, payload) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET source_id=excluded.source_id, payload=excluded.payload"
            # SQLへ渡す値をタプルにまとめます。
            parameters = (identifier, parent_id, payload)
        # sourceとpeopleはIDと本文だけを持ちます。
        else:
            # 内部で定めたテーブル名を使い2列をupsertします。
            sql = f"INSERT INTO {table}(id, payload) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload"
            # SQLのIDと本文をパラメーター化します。
            parameters = (identifier, payload)
        # 1行の追加または更新を接続トランザクションで実行します。
        with self._connection:
            # 値をSQL文字列に埋め込まず安全にバインドします。
            self._connection.execute(sql, parameters)


# 対応する拡張子を小文字で固定して照合します。
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v"})


# フォルダ以下からMP4、MOV、M4Vファイルを再帰列挙します。
def list_video_files(folder: Path) -> list[Path]:
    # 入力をPath型に揃えます。
    root = Path(folder)
    # 存在しない場所やフォルダ以外を明示的なエラーにします。
    if not root.is_dir():
        # 呼び出し側が不正な入力を識別できるようにします。
        raise NotADirectoryError(str(root))
    # 見つかった動画パスを蓄積します。
    results: list[Path] = []
    # followlinks=Falseでシンボリックリンクのディレクトリへ入りません。
    for current, directories, filenames in os.walk(root, followlinks=False):
        # ルート直下を含めてリンクディレクトリを明示的に除外します。
        directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()]
        # 現在のディレクトリにあるファイル名を順番に確認します。
        for filename in filenames:
            # 拡張子を大文字小文字に依存せず比較します。
            candidate = Path(current) / filename
            # 動画のシンボリックリンク自体も対象外にします。
            if candidate.is_symlink():
                # シンボリックリンクはスキップします。
                continue
            # 指定された拡張子だけを列挙します。
            if candidate.suffix.lower() in VIDEO_EXTENSIONS and candidate.is_file():
                # 実ファイルの絶対パスを結果に追加します。
                results.append(candidate.resolve())
    # 呼び出しごとに安定した順序となるようパス順で返します。
    return sorted(results, key=lambda path: str(path).casefold())
