import re
from PySide6.QtCore import QSettings
JAPANESE: dict[str, str] = {
    "Face analysis acceleration": "顔解析の高速化",
    "Automatic": "自動",
    "Auto (CUDA FP32 → CPU)": "自動（CUDA FP32 → CPU）",
    "Auto (TensorRT FP16 → CUDA TF32 → CPU)": "自動（TensorRT FP16 → CUDA TF32 → CPU）",
    "Accurate (FP32, TF32 off)": "精度優先（FP32、TF32無効）",
    "Balanced (TensorRT FP16 / CUDA TF32)": "バランス（TensorRT FP16 / CUDA TF32）",
    "CPU FP32.": "CPU FP32。",
    "Accurate: CUDA FP32 with TF32 disabled, then CPU. TensorRT requests use this fallback.": "精度優先：TF32無効のCUDA FP32、次にCPU。TensorRT指定時もこの順で代替します。",
    "Balanced: TensorRT FP16, then CUDA with TF32 enabled, then CPU. CUDA requests skip TensorRT.": "バランス：TensorRT FP16、次にTF32有効のCUDA、最後にCPU。CUDA指定時はTensorRTを使いません。",
    "The active runtime appears when analysis starts.": "使用方式は解析開始時に表示します。",
    "Saved face analysis settings. They apply to the next analysis.": "顔解析の高速化設定を保存しました。次の解析から適用します。",
    "Split into a new person": "新しい人物に分割",
    "Move to unclassified": "未分類へ戻す",
    "Move to another person": "別の人物へ移動",
    "Unnamed person": "名前なしの人物",
    "Exclude this face": "この顔を除外",
    "Rescan in detail": "詳しく再解析",
    "Close": "閉じる",
    "▶ Play": "▶ 再生",
    "Ⅱ Pause": "Ⅱ 一時停止",
    "People in this video": "この動画に登場する人物",
    "No faces were detected. You can run a detailed rescan.": "顔が検出されませんでした。必要なら詳しく再解析できます。",
    "No people have been detected yet.": "人物の検出結果はまだありません。",
    "Local video library": "ローカル動画ライブラリ",
    "▣  All videos": "▣  すべての動画",
    "♙  People": "♙  人物",
    "Library folders": "ライブラリフォルダ",
    "＋  Add folder": "＋  フォルダを追加",
    "⌫  Clear index…": "⌫  インデックスを消去…",
    "All videos": "すべての動画",
    "People": "人物",
    "Search video name or path": "動画名・パスを検索",
    "Refresh folders": "フォルダを更新",
    "Pause": "一時停止",
    "No results found": "見つかりませんでした",
    "No videos yet": "動画がありません",
    "Try a different search term.": "検索語を変えてください。",
    "Add a folder or drag a video folder here.": "フォルダを追加するか、ここへ動画フォルダをドラッグしてください。",
    "Pending": "待機中",
    "Analyzing": "解析中",
    "Ready": "完了",
    "File missing": "ファイルなし",
    "Failed": "失敗",
    "Open video": "動画を開く",
    "Some videos are being analyzed or are pending. Results will appear here as they become available.": "解析中・解析待ちの動画があります。結果は順次ここへ追加されます。",
    "Face photo": "顔写真",
    "Unclassified  {count}": "未分類  {count}件",
    "No people found yet": "人物はまだ見つかっていません",
    "Add a video folder to find detected faces here.": "動画フォルダを追加すると、見つかった顔をここで探せます。",
    "Unclassified faces": "未分類の顔",
    "Selected person": "選択した人物",
    "No matching detection times.": "該当する検出時刻はありません。",
    "{name}  ·  {count} detections": "{name}  ·  {count}件",
    "Show 60 more": "さらに60件表示",
    "Add folder": "フォルダを追加",
    "Choose a video folder": "動画フォルダを選択",
    "Remove this folder": "このフォルダを登録解除",
    "Remove folder": "フォルダを登録解除",
    "Delete this folder's index and generated images? Original videos will remain.": "このフォルダの索引と生成画像を削除します。元動画は削除しません。続けますか？",
    "Name or rename": "名前を付ける・変更する",
    "Merge into another person": "別の人物へ統合",
    "Person name": "人物名",
    "Enter a display name": "表示する名前を入力",
    "Cannot open video": "動画を開けません",
    "The original video is missing. Refresh the folders.": "元動画が見つかりません。フォルダを更新してください。",
    "Clear index": "インデックスを消去",
    "Delete registered folders, analysis results, and generated face images? Original videos will remain.": "登録フォルダ、解析結果、生成した顔画像を削除します。元動画は削除しません。続けますか？",
    "Index storage directory for the Python version": "Python版の索引保存先",
    "Language": "言語",
    "Japanese": "日本語",
    "English": "英語",
    "Delete": "削除",
    "Cancel": "キャンセル",
    "Choose a folder containing videos.": "動画を含むフォルダを選択してください。",
    "The original video was not found in its registered folder.": "登録フォルダ内で元動画が見つかりません。",
    "Pausing and saving partial results…": "一時停止して途中結果を保存しています…",
    "Paused.": "一時停止しました。",
    "Pause analysis before editing people.": "解析を一時停止してから人物を編集してください。",
    "Pause analysis before editing faces.": "解析を一時停止してから顔を編集してください。",
    "Pause analysis before removing a folder.": "解析を一時停止してからフォルダを削除してください。",
    "Pause analysis before clearing the index.": "解析を一時停止してから索引を消去してください。",
    "Analysis complete.": "解析が完了しました。",
    "Analysis stopped. Check the status area.": "解析が止まりました。状態欄を確認してください。",
    "The index and generated images were deleted. Original videos remain.": "索引と生成画像を消去しました。元動画は残っています。",
}
MESSAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"Checking the analysis model \((\d+) candidate videos\)", "解析モデルを確認しています（候補{0}件）"),
    (r"Analyzing: (.*)", "解析中：{0}"),
    (r"Analysis failed: (.*)", "解析失敗：{0}"),
    (r"Cannot start analysis: (.*)", "解析を開始できません：{0}"),
    (r"Cannot save analysis results: (.*)", "解析結果を保存できません：{0}"),
    (r"Analysis finished. Reasons for (\d+) failed videos are shown on their cards.", "解析終了：{0}件の失敗理由を動画カードに表示しています。"),
    (r"Cannot check (.*)", "確認できません：{0}"),
    (r"Cannot read information for (.*)", "情報を読めません：{0}"),
)
ENGINE_ERRORS: tuple[tuple[str, str], ...] = (
    (r"OpenCV face detector could not be loaded: (.*)", r"OpenCVの顔検出器を読み込めません：\1"),
    (r"Python package 'mediapipe' is required for face landmarks", "顔の目印の検出には Python パッケージ mediapipe が必要です"),
    (r"MediaPipe FaceLandmarker task model is missing: (.*)", r"MediaPipeの顔検出モデルがありません：\1"),
    (r"interval must be a finite number greater than zero", "解析間隔は0より大きい有限の数で指定してください"),
    (r"start must be a finite number greater than or equal to zero", "開始時刻は0以上の有限の数で指定してください"),
    (r"end must be greater than start", "終了時刻は開始時刻より後にしてください"),
    (r"Cannot read a valid video stream: (.*)", r"有効な動画ストリームを読み取れません：\1"),
    (r"Could not extract frame at (.*?) from (.*)", r"\1 のフレームを \2 から取り出せません"),
    (r"OpenCV could not encode a JPEG image", "OpenCVでJPEG画像を作成できません"),
    (r"Cannot encode an empty image", "空の画像は保存できません"),
)
_language = "en"


def load_language() -> str:
    saved = QSettings("VideoAtlas", "VideoAtlas").value("ui_language", "en")
    set_language(saved if saved in {"ja", "en"} else "en", persist=False)
    return _language


def set_language(language: str, persist: bool = True) -> None:
    if language not in {"ja", "en"}:
        raise ValueError(f"Unsupported language: {language}")
    global _language
    _language = language
    if persist:
        QSettings("VideoAtlas", "VideoAtlas").setValue("ui_language", language)


def tr(text: str, **values: object) -> str:
    template = JAPANESE.get(text, text) if _language == "ja" else text
    return template.format(**values) if values else template


def localize_message(message: str) -> str:
    if _language == "ja":
        if message in JAPANESE:
            return JAPANESE[message]
        for pattern, template in MESSAGE_PATTERNS:
            match = re.fullmatch(pattern, message)
            if match is not None:
                return template.format(*match.groups())
        for pattern, replacement in ENGINE_ERRORS:
            message = re.sub(pattern, replacement, message)
        return message
    for english, japanese in JAPANESE.items():
        if message == japanese:
            return english
    return message
