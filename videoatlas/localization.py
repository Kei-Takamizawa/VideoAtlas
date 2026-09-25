# このファイルは画面文言と解析メッセージの日本語・英語表示を管理します。
"""Small, persistent Japanese and English catalog for the current Python app."""

# 可変部分を含む日本語の状態文を英語へ置き換えるために使います。
import re
# 言語の選択をアプリの設定へ保存するために使います。
from PySide6.QtCore import QSettings

# 日本語の原文をキーにして対応する英語を一か所で管理します。
ENGLISH: dict[str, str] = {
    # 「新しい人物に分割」を英語でも表示できるようにします。
    "新しい人物に分割": "Split into a new person",
    # 「未分類へ戻す」を英語でも表示できるようにします。
    "未分類へ戻す": "Move to unclassified",
    # 「別の人物へ移動」を英語でも表示できるようにします。
    "別の人物へ移動": "Move to another person",
    # 「名前なしの人物」を英語でも表示できるようにします。
    "名前なしの人物": "Unnamed person",
    # 「この顔を除外」を英語でも表示できるようにします。
    "この顔を除外": "Exclude this face",
    # 「詳しく再解析」を英語でも表示できるようにします。
    "詳しく再解析": "Rescan in detail",
    # 「閉じる」を英語でも表示できるようにします。
    "閉じる": "Close",
    # 「▶ 再生」を英語でも表示できるようにします。
    "▶ 再生": "▶ Play",
    # 「Ⅱ 一時停止」を英語でも表示できるようにします。
    "Ⅱ 一時停止": "Ⅱ Pause",
    # 「この動画に登場する人物」を英語でも表示できるようにします。
    "この動画に登場する人物": "People in this video",
    # 「顔が検出されませんでした。必要なら詳しく再解析できます。」を英語でも表示できるようにします。
    "顔が検出されませんでした。必要なら詳しく再解析できます。": "No faces were detected. You can run a detailed rescan.",
    # 「人物の検出結果はまだありません。」を英語でも表示できるようにします。
    "人物の検出結果はまだありません。": "No people have been detected yet.",
    # 「ローカル動画ライブラリ」を英語でも表示できるようにします。
    "ローカル動画ライブラリ": "Local video library",
    # 「▣  すべての動画」を英語でも表示できるようにします。
    "▣  すべての動画": "▣  All videos",
    # 「♙  人物」を英語でも表示できるようにします。
    "♙  人物": "♙  People",
    # 「ライブラリフォルダ」を英語でも表示できるようにします。
    "ライブラリフォルダ": "Library folders",
    # 「＋  フォルダを追加」を英語でも表示できるようにします。
    "＋  フォルダを追加": "＋  Add folder",
    # 「⌫  インデックスを消去…」を英語でも表示できるようにします。
    "⌫  インデックスを消去…": "⌫  Clear index…",
    # 「すべての動画」を英語でも表示できるようにします。
    "すべての動画": "All videos",
    # 「人物」を英語でも表示できるようにします。
    "人物": "People",
    # 「動画名・パスを検索」を英語でも表示できるようにします。
    "動画名・パスを検索": "Search video name or path",
    # 「フォルダを更新」を英語でも表示できるようにします。
    "フォルダを更新": "Refresh folders",
    # 「一時停止」を英語でも表示できるようにします。
    "一時停止": "Pause",
    # 「見つかりませんでした」を英語でも表示できるようにします。
    "見つかりませんでした": "No results found",
    # 「動画がありません」を英語でも表示できるようにします。
    "動画がありません": "No videos yet",
    # 「検索語を変えてください。」を英語でも表示できるようにします。
    "検索語を変えてください。": "Try a different search term.",
    # 「フォルダを追加するか、ここへ動画フォルダをドラッグしてください。」を英語でも表示できるようにします。
    "フォルダを追加するか、ここへ動画フォルダをドラッグしてください。": "Add a folder or drag a video folder here.",
    # 「待機中」を英語でも表示できるようにします。
    "待機中": "Pending",
    # 「解析中」を英語でも表示できるようにします。
    "解析中": "Analyzing",
    # 「完了」を英語でも表示できるようにします。
    "完了": "Ready",
    # 「ファイルなし」を英語でも表示できるようにします。
    "ファイルなし": "File missing",
    # 「失敗」を英語でも表示できるようにします。
    "失敗": "Failed",
    # 「動画を開く」を英語でも表示できるようにします。
    "動画を開く": "Open video",
    # 「解析中・解析待ちの動画があります。結果は順次ここへ追加されます。」を英語でも表示できるようにします。
    "解析中・解析待ちの動画があります。結果は順次ここへ追加されます。": "Some videos are being analyzed or are pending. Results will appear here as they become available.",
    # 「顔写真」を英語でも表示できるようにします。
    "顔写真": "Face photo",
    # 「未分類  {count}件」を英語でも表示できるようにします。
    "未分類  {count}件": "Unclassified  {count}",
    # 「人物はまだ見つかっていません」を英語でも表示できるようにします。
    "人物はまだ見つかっていません": "No people found yet",
    # 「動画フォルダを追加すると、見つかった顔をここで探せます。」を英語でも表示できるようにします。
    "動画フォルダを追加すると、見つかった顔をここで探せます。": "Add a video folder to find detected faces here.",
    # 「未分類の顔」を英語でも表示できるようにします。
    "未分類の顔": "Unclassified faces",
    # 「選択した人物」を英語でも表示できるようにします。
    "選択した人物": "Selected person",
    # 「該当する検出時刻はありません。」を英語でも表示できるようにします。
    "該当する検出時刻はありません。": "No matching detection times.",
    # 「{name}  ·  {count}件」を英語でも表示できるようにします。
    "{name}  ·  {count}件": "{name}  ·  {count} detections",
    # 「さらに60件表示」を英語でも表示できるようにします。
    "さらに60件表示": "Show 60 more",
    # 「フォルダを追加」を英語でも表示できるようにします。
    "フォルダを追加": "Add folder",
    # 「動画フォルダを選択」を英語でも表示できるようにします。
    "動画フォルダを選択": "Choose a video folder",
    # 「このフォルダを登録解除」を英語でも表示できるようにします。
    "このフォルダを登録解除": "Remove this folder",
    # 「フォルダを登録解除」を英語でも表示できるようにします。
    "フォルダを登録解除": "Remove folder",
    # 「このフォルダの索引と生成画像を削除します。元動画は削除しません。続けますか？」を英語でも表示できるようにします。
    "このフォルダの索引と生成画像を削除します。元動画は削除しません。続けますか？": "Delete this folder's index and generated images? Original videos will remain.",
    # 「名前を付ける・変更する」を英語でも表示できるようにします。
    "名前を付ける・変更する": "Name or rename",
    # 「別の人物へ統合」を英語でも表示できるようにします。
    "別の人物へ統合": "Merge into another person",
    # 「人物名」を英語でも表示できるようにします。
    "人物名": "Person name",
    # 「表示する名前を入力」を英語でも表示できるようにします。
    "表示する名前を入力": "Enter a display name",
    # 「動画を開けません」を英語でも表示できるようにします。
    "動画を開けません": "Cannot open video",
    # 「元動画が見つかりません。フォルダを更新してください。」を英語でも表示できるようにします。
    "元動画が見つかりません。フォルダを更新してください。": "The original video is missing. Refresh the folders.",
    # 「インデックスを消去」を英語でも表示できるようにします。
    "インデックスを消去": "Clear index",
    # 「登録フォルダ、解析結果、生成した顔画像を削除します。元動画は削除しません。続けますか？」を英語でも表示できるようにします。
    "登録フォルダ、解析結果、生成した顔画像を削除します。元動画は削除しません。続けますか？": "Delete registered folders, analysis results, and generated face images? Original videos will remain.",
    # 「Python版の索引保存先」を英語でも表示できるようにします。
    "Python版の索引保存先": "Index storage directory for the Python version",
    # 「言語」を英語でも表示できるようにします。
    "言語": "Language",
    # 「日本語」を英語でも表示できるようにします。
    "日本語": "Japanese",
    # 「英語」を英語でも表示できるようにします。
    "英語": "English",
    # 「削除」を英語でも表示できるようにします。
    "削除": "Delete",
    # 「キャンセル」を英語でも表示できるようにします。
    "キャンセル": "Cancel",
    # 「動画を含むフォルダを選択してください。」を英語でも表示できるようにします。
    "動画を含むフォルダを選択してください。": "Choose a folder containing videos.",
    # 「登録フォルダ内で元動画が見つかりません。」を英語でも表示できるようにします。
    "登録フォルダ内で元動画が見つかりません。": "The original video was not found in its registered folder.",
    # 「一時停止して途中結果を保存しています…」を英語でも表示できるようにします。
    "一時停止して途中結果を保存しています…": "Pausing and saving partial results…",
    # 「一時停止しました。」を英語でも表示できるようにします。
    "一時停止しました。": "Paused.",
    # 「解析を一時停止してから人物を編集してください。」を英語でも表示できるようにします。
    "解析を一時停止してから人物を編集してください。": "Pause analysis before editing people.",
    # 「解析を一時停止してから顔を編集してください。」を英語でも表示できるようにします。
    "解析を一時停止してから顔を編集してください。": "Pause analysis before editing faces.",
    # 「解析を一時停止してからフォルダを削除してください。」を英語でも表示できるようにします。
    "解析を一時停止してからフォルダを削除してください。": "Pause analysis before removing a folder.",
    # 「解析を一時停止してから索引を消去してください。」を英語でも表示できるようにします。
    "解析を一時停止してから索引を消去してください。": "Pause analysis before clearing the index.",
    # 「解析が完了しました。」を英語でも表示できるようにします。
    "解析が完了しました。": "Analysis complete.",
    # 「解析が止まりました。状態欄を確認してください。」を英語でも表示できるようにします。
    "解析が止まりました。状態欄を確認してください。": "Analysis stopped. Check the status area.",
    # 「索引と生成画像を消去しました。元動画は残っています。」を英語でも表示できるようにします。
    "索引と生成画像を消去しました。元動画は残っています。": "The index and generated images were deleted. Original videos remain.",
}

# 変数を含む既存の状態文は日本語のまま保存されるため表示時に翻訳します。
MESSAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^解析を開始します（(\d+)件）$", "Starting analysis ({0} videos)"),
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^解析中：(.*)$", "Analyzing: {0}"),
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^解析失敗：(.*?)：(.*)$", "Analysis failed: {0}: {1}"),
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^解析を開始できません：(.*)$", "Cannot start analysis: {0}"),
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^解析結果を保存できません：(.*)$", "Cannot save analysis results: {0}"),
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^解析終了：(\d+)件の失敗理由を動画カードに表示しています。$", "Analysis finished. Reasons for {0} failed videos are shown on their cards."),
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^(.*?) を確認できません：(.*)$", "Cannot check {0}: {1}"),
    # 文章の中の件数や名前を残したまま英語へ変換します。
    (r"^(.*?) の情報を読めません：(.*)$", "Cannot read information for {0}: {1}"),
)

# 解析器が返す既知の英語エラーを日本語画面でも読めるようにします。
ENGINE_ERRORS: tuple[tuple[str, str], ...] = (
    # モデルの保存場所が欠けている場合の英語エラーを説明します。
    (r"OpenVINO model files are missing: (.*?) and (.*)", r"OpenVINOのモデルファイルがありません：\1 と \2"),
    # 顔の特徴量を計算するパッケージがない場合を説明します。
    (r"Python package 'openvino' is required for face embeddings", "顔の特徴量には Python パッケージ openvino が必要です"),
    # 顔検出器を読み込めない場合に対象の場所を残します。
    (r"OpenCV face detector could not be loaded: (.*)", r"OpenCVの顔検出器を読み込めません：\1"),
    # 顔の目印を調べるパッケージがない場合を説明します。
    (r"Python package 'mediapipe' is required for face landmarks", "顔の目印の検出には Python パッケージ mediapipe が必要です"),
    # 顔の目印を調べるモデルがない場合に場所を残します。
    (r"MediaPipe FaceLandmarker task model is missing: (.*)", r"MediaPipeの顔検出モデルがありません：\1"),
    # 不正な解析間隔が渡された場合を説明します。
    (r"interval must be a finite number greater than zero", "解析間隔は0より大きい有限の数で指定してください"),
    # 不正な開始時刻が渡された場合を説明します。
    (r"start must be a finite number greater than or equal to zero", "開始時刻は0以上の有限の数で指定してください"),
    # 終了時刻が開始時刻より前の場合を説明します。
    (r"end must be greater than start", "終了時刻は開始時刻より後にしてください"),
    # 動画ストリームを読み取れない場合に元ファイルを示します。
    (r"Cannot read a valid video stream: (.*)", r"有効な動画ストリームを読み取れません：\1"),
    # 指定した時刻のフレームを取り出せない場合を説明します。
    (r"Could not extract frame at (.*?) from (.*)", r"\1 のフレームを \2 から取り出せません"),
    # JPEG画像の符号化に失敗した場合を説明します。
    (r"OpenCV could not encode a JPEG image", "OpenCVでJPEG画像を作成できません"),
    # 空画像を保存できない場合を説明します。
    (r"Cannot encode an empty image", "空の画像は保存できません"),
    # 解析結果の特徴量が想定の長さと違う場合を説明します。
    (r"Expected 256 embedding values, got (\d+)", r"顔の特徴量は256個必要ですが、\1個でした"),
)

# 設定がまだない利用者には従来どおり日本語を表示します。
_language = "ja"


# 保存された言語の値を読み、使える二言語だけを受け入れます。
def load_language() -> str:
    # アプリ固有の保存領域から前回の選択を読みます。
    saved = QSettings("VideoAtlas", "VideoAtlas").value("ui_language", "ja")
    # 不明な値であれば安全な初期言語へ戻します。
    set_language(saved if saved in {"ja", "en"} else "ja", persist=False)
    # 画面で選択状態を設定できるよう値を返します。
    return _language


# 現在の表示言語を切り替え、必要なら次回起動用に保存します。
def set_language(language: str, persist: bool = True) -> None:
    # 想定外の言語コードは翻訳漏れを起こすため拒否します。
    if language not in {"ja", "en"}:
        # 呼び出し側が誤りを直せるよう明示します。
        raise ValueError(f"Unsupported language: {language}")
    # 翻訳関数が参照する現在値を書き換えます。
    global _language
    # 選ばれた言語を保持します。
    _language = language
    # 利用者が画面で変更したときだけ保存します。
    if persist:
        # 次回起動時も同じ言語を使います。
        QSettings("VideoAtlas", "VideoAtlas").setValue("ui_language", language)


# 日本語の原文を選択中の言語へ変え、名前や件数を差し込みます。
def tr(text: str, **values: object) -> str:
    # 英語が選択されている場合だけ対応する翻訳を探します。
    template = ENGLISH.get(text, text) if _language == "en" else text
    # 件数やファイル名の変数があれば両言語で同じように埋めます。
    return template.format(**values) if values else template


# 日本語で保持する解析状態や保存済みエラーを現在の言語で表示します。
def localize_message(message: str) -> str:
    # 日本語表示では元の解析メッセージをそのまま使います。
    if _language == "ja":
        # 解析器が英語で返した既知の原因を日本語へ変えます。
        for pattern, replacement in ENGINE_ERRORS:
            # 状態文の前半と元ファイルの名前は保持します。
            message = re.sub(pattern, replacement, message)
        # 利用者に表示できる日本語の文章を返します。
        return message
    # 固定文は通常の翻訳表で処理します。
    if message in ENGLISH:
        # 完全一致した英語を返します。
        return ENGLISH[message]
    # 可変部分を持つ既知の状態文を順番に照合します。
    for pattern, template in MESSAGE_PATTERNS:
        # 日本語の原文に合うパターンを探します。
        match = re.fullmatch(pattern, message)
        # 合致した場合はファイル名や件数を残して英語にします。
        if match is not None:
            # 原因の文字列は元の内容を保持します。
            return template.format(*match.groups())
    # 未知のライブラリエラーは内容を失わないようそのまま表示します。
    return message
