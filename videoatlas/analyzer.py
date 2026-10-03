# 型注釈で呼び出し側の入力と戻り値を明確にします。
from __future__ import annotations
# データクラスを短く定義する標準機能を読み込みます。
from dataclasses import dataclass
# OpenVINO モデルの配置先を扱うために Path を読み込みます。
from pathlib import Path
# 進捗・停止コールバックの型を表すために Callable を読み込みます。
from typing import Callable, Iterator
# JPEG 化や顔画像の画素処理に NumPy を使います。
import numpy as np
# 動画フレーム取得と Haar 顔検出に OpenCV を使います。
import cv2
import hashlib
import os
import sys
import threading

from .recognition import RecognitionOptions, default_config_path

# 一つの顔の検出情報と任意の照合特徴量を保持します。
@dataclass
class FaceSample:
    # 顔が見つかった動画内の秒数を保持します。
    second: float
    # 顔領域をフレーム幅・高さで割った x, y, 幅, 高さで保持します。
    bounding_box: tuple[float, float, float, float]
    # 顔領域の JPEG データを保持します。
    thumbnail_jpeg: bytes
    # 有効な整列画像から計算できた場合にだけ特徴量を保持します。
    embedding: list[float] | None
    # 特徴量を作ったモデルと前処理の識別子です。
    embedding_model: str | None = None
    # 画像品質を0～1で表します。
    quality: float = 1.0
    # 特徴量を作れなかった場合の理由です。
    rejection_reason: str | None = None

# 最大 60 サンプル分の処理結果と再開位置を保持します。
@dataclass
class AnalysisChunk:
    # 動画全体の長さを秒で保持します。
    duration: float
    # 動画の先頭から作った JPEG ポスターを保持します。
    poster_jpeg: bytes | None
    # このチャンク内で検出した顔を保持します。
    faces: list[FaceSample]
    # 次回再開するときの動画内秒数を保持します。
    completed_second: float
    # 停止コールバックにより中断されたかを保持します。
    cancelled: bool

# 動画からフレーム、顔領域、顔特徴量を順番に生成します。
class _LegacyVideoAnalyzer:
    # 通常解析のフレーム長辺上限をピクセルで定義します。
    _MAX_DIMENSION = 1280
    # ひとつの永続化単位に含めるサンプル数を定義します。
    _CHUNK_SIZE = 60

    # OpenVINO モデルを読み込み、利用可能性を明示的に確認します。
    def __init__(self, model_xml: Path):
        # XML の隣に対応する重みファイルがあるかを確認します。
        model_bin = model_xml.with_suffix(".bin")
        # XML と BIN のどちらかが欠けていれば曖昧に続行しません。
        if not model_xml.is_file() or not model_bin.is_file():
            # 不足しているモデルの配置先を利用者に伝えます。
            raise FileNotFoundError(f"OpenVINO model files are missing: {model_xml} and {model_bin}")
        # Python OpenVINO がなければ埋め込み計算不能として明示します。
        try:
            # OpenVINO の Python API を遅延して読み込みます。
            from openvino import Core
        # 依存不足をモデル未使用のまま成功扱いにしないため捕捉します。
        except ImportError as error:
            # 必要なパッケージ名と状況を例外に含めます。
            raise RuntimeError("Python package 'openvino' is required for face embeddings") from error
        # 推論エンジンを作成します。
        self._core = Core()
        # 指定された XML と BIN を OpenVINO IR として読み込みます。
        model = self._core.read_model(model=str(model_xml), weights=str(model_bin))
        # モデルを CPU 用にコンパイルします。
        self._compiled = self._core.compile_model(model, "CPU")
        # 入力テンソル情報を後の推論で再利用します。
        self._input = self._compiled.input(0)
        # 出力テンソル情報を後の推論で再利用します。
        self._output = self._compiled.output(0)
        # OpenCV に付属する Haar 正面顔検出器のパスを取得します。
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        # Haar 検出器を初期化します。
        self._detector = cv2.CascadeClassifier(str(cascade_path))
        # 検出器データが欠落または破損していないか確認します。
        if self._detector.empty():
            # 顔検出器の読込失敗を明示的に返します。
            raise RuntimeError(f"OpenCV face detector could not be loaded: {cascade_path}")
        # MediaPipe Tasks の顔ランドマーク機能を遅延して読み込みます。
        try:
            # MediaPipe の基礎 API を読み込みます。
            import mediapipe as mp
        # MediaPipe が未導入なら検出を黙って代替しません。
        except ImportError as error:
            # 必要な依存名を明示して初期化を失敗させます。
            raise RuntimeError("Python package 'mediapipe' is required for face landmarks") from error
        # 指定された公式 Tasks モデルの配置先を決定します。
        task_path = Path(__file__).parent / "resources" / "face_landmarker.task"
        # モデルファイルが用意されていることを確認します。
        if not task_path.is_file():
            # モデルの入手先と必要なファイル名を案内します。
            raise FileNotFoundError(f"MediaPipe FaceLandmarker task model is missing: {task_path}")
        # 単一画像モードで使用するオプションを作成します。
        options = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)), running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=20)
        # 公式 MediaPipe Tasks FaceLandmarker を初期化します。
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        # MediaPipe を後で RGB Image に包むため参照を保持します。
        self._mp = mp

    # 指定間隔でサンプルし、最大 60 サンプルごとに再開可能な結果を返します。
    def iter_chunks(self, video_path: Path, interval: float, start: float = 0, end: float | None = None, stop: Callable[[], bool] | None = None, progress: Callable[[float, float], None] | None = None) -> Iterator[AnalysisChunk]:
        # サンプル間隔が正の有限値であることを確認します。
        if not np.isfinite(interval) or interval <= 0:
            # 不正な間隔では処理を始めず理由を返します。
            raise ValueError("interval must be a finite number greater than zero")
        # 開始位置が有限かつゼロ以上であることを確認します。
        if not np.isfinite(start) or start < 0:
            # 不正な再開時刻を拒否します。
            raise ValueError("start must be a finite number greater than or equal to zero")
        # OpenCV で動画ファイルを開きます。
        capture = cv2.VideoCapture(str(video_path))
        # コンテナのメタデータからフレーム数とフレームレートを取得します。
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # 動画の時間基準となるフレームレートを取得します。
        fps = capture.get(cv2.CAP_PROP_FPS)
        # 動画時間が計算可能であることとファイルが開けたことを確認します。
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
            # 資料を解放し、読み取り不能として報告します。
            capture.release()
            # 無効または未対応の動画を明示します。
            raise ValueError(f"Cannot read a valid video stream: {video_path}")
        # メタデータから動画全体の秒数を計算します。
        duration = float(frame_count / fps)
        # 要求範囲の終端を動画の終端以内に制限します。
        final_second = min(duration, duration if end is None else float(end))
        # 終端が開始以前でないことを確認します。
        if final_second <= start and start < duration:
            # 不正な区間を知らせて動画を解放します。
            capture.release()
            # 空区間による無限ループを防ぎます。
            raise ValueError("end must be greater than start")
        # 開始位置までシークし、最初の指定時刻を設定します。
        requested = min(float(start), duration)
        # チャンク内の顔を格納するリストを初期化します。
        faces: list[FaceSample] = []
        # チャンク内の処理済みフレーム数を数えます。
        chunk_frames = 0
        # 最初に読み取ったフレームからポスターを作ります。
        poster: bytes | None = None
        # 次に再開すべき時刻を開始位置に設定します。
        completed = requested
        # 直近チャンクで中断が起きたかを記録します。
        cancelled = False
        # 次の読み取り位置をミリ秒で指定します。
        capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
        # 動画全体の長さを進捗コールバックへ渡せるよう保持します。
        while requested < final_second:
            # チャンク内の最大フレーム数に達したら結果を保存可能にします。
            if chunk_frames >= self._CHUNK_SIZE:
                # 処理位置とデータを呼び出し側へ返します。
                yield AnalysisChunk(duration, poster, faces, completed, False)
                # 次のチャンク用に顔リストを空にします。
                faces = []
                # 次のチャンクのフレーム数をゼロに戻します。
                chunk_frames = 0
                # 先頭チャンク以降はポスターを重複して返しません。
                poster = None
            # 利用者が停止を要求したかを確認します。
            if stop is not None and stop():
                # 中断したことを最終チャンクへ記録します。
                cancelled = True
                # 停止ループに入ります。
                break
            # 指定秒数のフレームへシークします。
            capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
            # フレームを一枚読み込みます。
            success, frame = capture.read()
            # 読めないフレームはファイルの読み取りエラーとして扱います。
            if not success or frame is None:
                # 途中でデコードできない箇所を時刻付きで報告します。
                capture.release()
                # デコード失敗を呼び出し側へ伝えます。
                raise ValueError(f"Could not extract frame at {requested:.2f}s from {video_path}")
            # OpenCV が示すフレーム時刻を可能な範囲で秒に直します。
            actual_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            # 無効な時刻情報の場合は要求時刻を使用します。
            actual_second = actual_ms / 1000.0 if np.isfinite(actual_ms) and actual_ms > 0 else requested
            # 今回処理したフレームを現在のチャンクへ数えます。
            chunk_frames += 1
            # 通常解析では長辺を 1280 ピクセル以下に縮小します。
            if interval > 0.5 and max(frame.shape[:2]) > self._MAX_DIMENSION:
                # 縦横比を維持してフレームを縮小します。
                scale = self._MAX_DIMENSION / max(frame.shape[:2])
                # 高品質縮小補間で画像を変換します。
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # 最初のフレームを動画ポスター JPEG として保持します。
            if poster is None:
                # JPEG 品質 75 でポスターを符号化します。
                poster = self._encode_jpeg(frame, 75)
            # MediaPipe が要求する RGB 色順へ変換します。
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # RGB 配列を MediaPipe Image に変換します。
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            # 公式 Tasks API で顔ランドマークを検出します。
            result = self._landmarker.detect(mp_image)
            # 顔ランドマークの検出結果がある場合はそれを優先します。
            if result.face_landmarks:
                # 各ランドマーク付き顔について切り出しと特徴量を作ります。
                for landmarks in result.face_landmarks:
                    # 全ランドマークから顔の正規化バウンディングボックスを計算します。
                    xs = [point.x for point in landmarks]
                    # 全ランドマークの縦座標を集めます。
                    ys = [point.y for point in landmarks]
                    # 顔枠の左端を画像幅基準で切り捨てます。
                    x0 = max(0, int(min(xs) * frame.shape[1]))
                    # 顔枠の上端を画像高さ基準で切り捨てます。
                    y0 = max(0, int(min(ys) * frame.shape[0]))
                    # 顔枠の右端を画像幅以内に制限します。
                    x1 = min(frame.shape[1], int(max(xs) * frame.shape[1]) + 1)
                    # 顔枠の下端を画像高さ以内に制限します。
                    y1 = min(frame.shape[0], int(max(ys) * frame.shape[0]) + 1)
                    # 顔領域を元のフレームから切り出します。
                    crop = frame[y0:y1, x0:x1]
                    # 小顔や整列失敗でも残す顔サムネイルを作成します。
                    thumbnail = self._encode_jpeg(crop, 80)
                    # 64px以上でランドマークが妥当ならOpenVINO特徴量を計算します。
                    embedding = self._embedding_for_landmarks(frame, landmarks, (x0, y0, x1, y1))
                    # 相対座標、顔画像、可能なら特徴量を保存します。
                    faces.append(FaceSample(actual_second, (x0 / frame.shape[1], y0 / frame.shape[0], (x1 - x0) / frame.shape[1], (y1 - y0) / frame.shape[0]), thumbnail, embedding))
            # MediaPipe が顔を検出しなかった場合だけ Haar の未分類候補を補います。
            else:
                # 補完検出用にフレームをグレースケールへ変換します。
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Haar 検出器で顔候補領域を得ます。
                boxes = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                # 補完候補を特徴量なしの顔として保存します。
                for x, y, width, height in boxes:
                    # 顔切り出しの左端を画像内に収めます。
                    x0 = max(0, int(x))
                    # 顔切り出しの上端を画像内に収めます。
                    y0 = max(0, int(y))
                    # 顔切り出しの右端を画像内に収めます。
                    x1 = min(frame.shape[1], int(x + width))
                    # 顔切り出しの下端を画像内に収めます。
                    y1 = min(frame.shape[0], int(y + height))
                    # 未分類状態で残す顔画像を切り出します。
                    crop = frame[y0:y1, x0:x1]
                    # 未分類顔の JPEG サムネイルを作ります。
                    thumbnail = self._encode_jpeg(crop, 80)
                    # Haar だけの候補には照合特徴量を作らないことを明示します。
                    embedding = None
                    # 相対座標と未分類状態を顔サンプルにして保存します。
                    faces.append(FaceSample(actual_second, (x0 / frame.shape[1], y0 / frame.shape[0], (x1 - x0) / frame.shape[1], (y1 - y0) / frame.shape[0]), thumbnail, embedding))
            # 今回処理した要求時刻の次を再開位置に設定します。
            completed = min(requested + interval, final_second)
            # 進捗コールバックが指定されていれば現状を通知します。
            if progress is not None:
                # 完了秒と動画全体の秒数を報告します。
                progress(completed, duration)
            # 次の要求時刻へ間隔分だけ進みます。
            requested += interval
        # 動画リソースを解放します。
        capture.release()
        # 残りの顔、または中断状態を含む最後のチャンクを返します。
        if faces or cancelled or completed >= final_second:
            # 最後の部分結果を呼び出し側へ返します。
            yield AnalysisChunk(duration, poster, faces, completed, cancelled)

    # OpenCV の画像を指定品質の JPEG バイト列へ変換します。
    @staticmethod
    def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
        # 空画像をエンコードしてしまわないよう確認します。
        if image.size == 0:
            # データ欠落を分かりやすく呼び出し側へ伝えます。
            raise ValueError("Cannot encode an empty image")
        # 指定された品質設定で JPEG を圧縮します。
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # エンコード失敗を成功扱いにしないよう確認します。
        if not success:
            # 画像変換の失敗理由を呼び出し側へ伝えます。
            raise RuntimeError("OpenCV could not encode a JPEG image")
        # NumPy 配列を不変のバイト列として返します。
        return encoded.tobytes()

    # MediaPipe の五点ランドマークから整列画像と特徴ベクトルを作ります。
    def _embedding_for_landmarks(self, frame: np.ndarray, landmarks: list, box: tuple[int, int, int, int]) -> list[float] | None:
        # バウンディングボックスから顔の画素サイズを計算します。
        x0, y0, x1, y1 = box
        # 縦横どちらかが 64px 未満なら小顔として未分類にします。
        if min(x1 - x0, y1 - y0) < 64:
            # サムネイルは呼び出し元が保持し、特徴量だけを欠落にします。
            return None
        # Face Mesh の点を画像ピクセル座標へ変換する関数です。
        def pixel(index: int) -> np.ndarray:
            # 指定ランドマークの正規化座標を画像座標に直します。
            return np.array([landmarks[index].x * frame.shape[1], landmarks[index].y * frame.shape[0]], dtype=np.float64)
        # 左右の目の内外角平均を目の中心として計算します。
        left_eye = (pixel(33) + pixel(133)) / 2.0
        # 右目の内外角平均を目の中心として計算します。
        right_eye = (pixel(362) + pixel(263)) / 2.0
        # 鼻先の標準ランドマークを読み取ります。
        nose_tip = pixel(1)
        # 口の左右端を画像上の左から右へ並べます。
        mouth_points = sorted((pixel(61), pixel(291)), key=lambda point: point[0])
        # 左右目が画像上で逆転していないことを確認します。
        if left_eye[0] >= right_eye[0]:
            # 不自然なランドマーク配置を未分類にします。
            return None
        # 目、鼻、口が上から下の順に並ぶことを確認します。
        if (left_eye[1] + right_eye[1]) / 2.0 >= nose_tip[1] or nose_tip[1] >= (mouth_points[0][1] + mouth_points[1][1]) / 2.0:
            # 顔姿勢や検出の不確かな配置を未分類にします。
            return None
        # 観測された目・鼻・口の五点を並べます。
        observed = np.stack((left_eye, right_eye, nose_tip, mouth_points[0], mouth_points[1]))
        # 公開モデルの参照五点を 128×128 座標に変換します。
        reference = np.array([[0.31556875, 1 - 0.4615741071], [0.6826229167, 1 - 0.4615741071], [0.5002625, 1 - 0.6405053571], [0.349471875, 1 - 0.8246919643], [0.6534364583, 1 - 0.8246919643]], dtype=np.float64) * 128.0
        # 相似変換のため観測五点と参照点を中心化します。
        observed_centered = observed - observed.mean(axis=0)
        # 参照五点の平均位置を求めます。
        reference_centered = reference - reference.mean(axis=0)
        # 変換尺度の計算に使う観測点の二乗和を求めます。
        denominator = float(np.sum(observed_centered ** 2))
        # 重なったランドマークでは変換を計算できないため未分類にします。
        if denominator <= 1e-8:
            # 不正な変換を破棄します。
            return None
        # 二次元相似変換の回転・拡大係数を最小二乗で求めます。
        covariance = observed_centered.T @ reference_centered
        # 反射を含まない回転と尺度を得るため特異値分解します。
        left_vectors, singular_values, right_vectors = np.linalg.svd(covariance)
        # 反射行列を補正し、顔を鏡像反転しない回転にします。
        correction = np.eye(2, dtype=np.float64)
        # 回転行列が反射なら一軸を反転して直します。
        correction[-1, -1] = np.linalg.det(left_vectors @ right_vectors)
        # 観測から参照への相似回転を計算します。
        rotation = left_vectors @ correction @ right_vectors
        # 五点全体から均一な拡大縮小率を計算します。
        scale = float(np.sum(singular_values * np.diag(correction)) / denominator)
        # 線形部分を尺度と回転の積にします。
        linear = scale * rotation
        # 参照中心へ移す平行移動量を計算します。
        translation = reference.mean(axis=0) - observed.mean(axis=0) @ linear
        # OpenCV の affine warp が受け取る 2×3 行列を作ります。
        transform = np.column_stack((linear.T, translation))
        # 五点が参照位置へどの程度一致するかを測ります。
        aligned_points = observed @ linear + translation
        # 五点の平均残差をピクセル単位で算出します。
        residual = float(np.sqrt(np.mean(np.sum((aligned_points - reference) ** 2, axis=1))))
        # 16pxを超える残差やゼロ尺度は信頼しないで未分類にします。
        if not np.isfinite(residual) or residual > 16.0 or scale <= 0:
            # 不確かな整列結果を人物特徴量へ進めません。
            return None
        # 元フレームを白背景の 128×128 BGR 画像へ相似変換します。
        aligned = cv2.warpAffine(frame, transform.astype(np.float32), (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        # HWC 画像をモデル仕様の BGR CHW 配列へ転置します。
        chw = np.transpose(aligned.astype(np.float32), (2, 0, 1))
        # OpenVINO モデルが要求する 1×3×128×128 バッチ次元を加えます。
        tensor = np.expand_dims(chw, axis=0)
        # 準備した画像テンソルでモデルを推論します。
        outputs = self._compiled([tensor])
        # モデルの出力を一次元の 256 要素へ整形します。
        embedding = np.asarray(outputs[self._output], dtype=np.float32).reshape(-1)
        # モデル仕様外の出力次元を成功扱いにしません。
        if embedding.size != 256:
            # 出力形状がモデル期待と違う理由を明示します。
            raise RuntimeError(f"Expected 256 embedding values, got {embedding.size}")
        # Python float のリストとして公開 API へ返します。
        return embedding.tolist()


_SESSION_CACHE: dict[tuple[object, ...], tuple[object, object, object, bool, str]] = {}
_SESSION_CACHE_LOCK = threading.Lock()
_FACE01_PREPROCESSING = "face01-rgb224-nchw-imagenet-align5-v2"
_FACE01_MODEL_NAME = "JAPANESE_FACE_V1.onnx"
_FACE01_MODEL_SHA256 = "e7ca51f4bc85f73ddb830683ac6a09077909fa45a52b2bff41a9c6e8ff267e2f"


class _Face01VideoAnalyzer:
    """Windows FACE01 analyzer with original-resolution landmark alignment."""

    _CHUNK_SIZE = 60

    def __init__(self, options: RecognitionOptions) -> None:
        options.validate()
        self.options = options
        self._closed = False
        model_path = Path(options.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"FACE01 model is missing: {model_path}. Place JAPANESE_FACE_V1.onnx there; "
                "the application does not download model files automatically."
            )
        self.model_path = model_path.resolve()
        digest = hashlib.sha256()
        with self.model_path.open("rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(block)
        self.model_sha256 = digest.hexdigest()
        default_model = Path(__file__).parent / "resources" / _FACE01_MODEL_NAME
        if self.model_path == default_model.resolve() and self.model_sha256 != _FACE01_MODEL_SHA256:
            raise RuntimeError(f"FACE01 model checksum mismatch for {self.model_path}")
        self.embedding_model = f"face01:{self.model_sha256}:{_FACE01_PREPROCESSING}"
        self._dll_directory_handle = None
        if options.tensorrt_dll_dir is not None:
            dll_dir = Path(options.tensorrt_dll_dir).expanduser()
            if not dll_dir.is_dir():
                raise FileNotFoundError(f"TensorRT DLL directory does not exist: {dll_dir}")
            if hasattr(os, "add_dll_directory"):
                self._dll_directory_handle = os.add_dll_directory(str(dll_dir))
        self._session, self._input, self._output, self._dynamic_batch, provider_report = self._load_session()
        self.runtime_description = provider_report
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(str(cascade_path))
        if self._detector.empty():
            raise RuntimeError(f"OpenCV face detector could not be loaded: {cascade_path}")
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError("Python package 'mediapipe' is required for face landmarks") from error
        task_path = Path(__file__).parent / "resources" / "face_landmarker.task"
        if not task_path.is_file():
            raise FileNotFoundError(f"MediaPipe FaceLandmarker task model is missing: {task_path}")
        task_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=20,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(task_options)
        self._mp = mp

    def _load_session(self) -> tuple[object, object, object, bool, str]:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Python package 'onnxruntime-gpu' is required for FACE01") from error
        available = set(ort.get_available_providers())
        preload_note = ""
        if sys.platform == "win32" and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory=str(self.options.tensorrt_dll_dir) if self.options.tensorrt_dll_dir else None)
            except Exception as error:
                preload_note = f"GPU DLL preload unavailable: {error}"
        requested = self.options.acceleration
        if requested == "cpu":
            provider_names = ["CPUExecutionProvider"]
        elif requested == "cuda":
            provider_names = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif requested == "tensorrt":
            provider_names = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            provider_names = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        if sys.platform != "win32" and requested in {"auto", "cuda", "tensorrt"}:
            provider_names = [name for name in provider_names if name == "CPUExecutionProvider"]
        unavailable = [name for name in provider_names if name != "CPUExecutionProvider" and name not in available]
        provider_names = [name for name in provider_names if name in available]
        if not provider_names or provider_names[-1] != "CPUExecutionProvider":
            provider_names.append("CPUExecutionProvider")
        cache_root = self.options.cache_dir or (default_config_path().parent / "InferenceCache")
        cache_root.mkdir(parents=True, exist_ok=True)
        version = getattr(ort, "__version__", "unknown")
        cache_key = (str(self.model_path), self.model_sha256, version, tuple(provider_names), self.options.gpu_device_id, self.options.threads, self.options.precision)
        with _SESSION_CACHE_LOCK:
            cached = _SESSION_CACHE.get(cache_key)
            if cached is not None:
                return cached
            failures: list[str] = []
            candidate_lists: list[list[object]] = []
            if "TensorrtExecutionProvider" in provider_names:
                trt_cache = self._safe_tensorrt_cache(cache_root, version)
                balanced = self.options.precision == "balanced"
                trt_options = {"device_id": str(self.options.gpu_device_id), "trt_engine_cache_enable": "1" if trt_cache else "0", "trt_fp16_enable": "1" if balanced else "0", "trt_use_tf32": "1" if balanced else "0", "trt_max_workspace_size": str(1024 * 1024 * 1024), "trt_engine_cache_prefix": f"face01-{self.model_sha256[:16]}-{self.options.precision}"}
                if trt_cache is not None:
                    trt_options["trt_engine_cache_path"] = str(trt_cache)
                candidate_lists.append([("TensorrtExecutionProvider", trt_options), ("CUDAExecutionProvider", {"device_id": str(self.options.gpu_device_id)}), "CPUExecutionProvider"])
            if "CUDAExecutionProvider" in provider_names:
                candidate_lists.append([("CUDAExecutionProvider", {"device_id": str(self.options.gpu_device_id)}), "CPUExecutionProvider"])
            candidate_lists.append(["CPUExecutionProvider"])
            seen: set[str] = set()
            for providers in candidate_lists:
                label = ",".join(str(item[0] if isinstance(item, tuple) else item) for item in providers)
                if label in seen:
                    continue
                seen.add(label)
                try:
                    session_options = ort.SessionOptions()
                    session_options.intra_op_num_threads = self.options.threads
                    session_options.inter_op_num_threads = 1
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=providers)
                    model_input = session.get_inputs()[0]
                    model_output = session.get_outputs()[0]
                    if len(model_input.shape) != 4 or tuple(model_input.shape[-3:]) != (3, 224, 224):
                        raise RuntimeError(f"FACE01 input must have NCHW shape [N, 3, 224, 224], got {model_input.shape}")
                    if len(model_output.shape) < 2 or model_output.shape[-1] not in (256, "256", None):
                        raise RuntimeError(f"FACE01 output must end in 256 values, got {model_output.shape}")
                    batch_dim = model_input.shape[0] if len(model_input.shape) == 4 else 1
                    dynamic_batch = batch_dim is None or isinstance(batch_dim, str)
                    if not dynamic_batch and batch_dim != 1:
                        raise RuntimeError(f"FACE01 static batch size must be 1 or dynamic, got {batch_dim}")
                    actual = ", ".join(session.get_providers())
                    report = f"FACE01 ONNX Runtime {version}; active providers: {actual}"
                    if unavailable:
                        report += "; unavailable providers: " + ", ".join(unavailable)
                    if preload_note:
                        report += f"; {preload_note}"
                    if failures:
                        report += "; provider fallback: " + " | ".join(failures)
                    result = (session, model_input, model_output, dynamic_batch, report)
                    _SESSION_CACHE[cache_key] = result
                    return result
                except Exception as error:
                    failures.append(f"{label}: {error}")
            raise RuntimeError("Could not create FACE01 inference session: " + " | ".join(failures))

    def _safe_tensorrt_cache(self, cache_root: Path, ort_version: str) -> Path | None:
        """Only persist compiled engines when runtime and adapter identity are known."""
        try:
            import importlib.metadata
            import subprocess

            trt_version = importlib.metadata.version("tensorrt")
            result = subprocess.run(
                ["nvidia-smi", "--id", str(self.options.gpu_device_id), "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            identity = result.stdout.strip().replace(", ", "-").replace(" ", "")
            if not identity or "GPU-" not in identity:
                return None
            key = f"trt-{self.model_sha256[:16]}-ort{ort_version}-trt{trt_version}-gpu{identity}-p{self.options.precision}"
            return cache_root / key
        except Exception:
            return None

    def iter_chunks(self, video_path: Path, interval: float, start: float = 0, end: float | None = None, stop: Callable[[], bool] | None = None, progress: Callable[[float, float], None] | None = None) -> Iterator[AnalysisChunk]:
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be a finite number greater than zero")
        if not np.isfinite(start) or start < 0:
            raise ValueError("start must be a finite number greater than or equal to zero")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot open video stream: {video_path}")
        try:
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0 or not np.isfinite(frame_count) or frame_count <= 0:
                raise ValueError(f"Cannot read a valid video stream: {video_path}")
            duration = float(frame_count / fps)
            final_second = min(duration, duration if end is None else float(end))
            if final_second <= start and start < duration:
                raise ValueError("end must be greater than start")
            requested = min(float(start), duration)
            capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
            faces: list[FaceSample] = []
            chunk_frames = 0
            poster: bytes | None = None
            completed = requested
            cancelled = False
            previous_frame_index = max(0, int(requested * fps)) - 1
            while requested < final_second:
                if chunk_frames >= self._CHUNK_SIZE:
                    yield AnalysisChunk(duration, poster, faces, completed, False)
                    faces, chunk_frames, poster = [], 0, None
                if stop is not None and stop():
                    cancelled = True
                    break
                target_index = max(0, int(round(requested * fps)))
                gap = target_index - previous_frame_index
                if 0 < gap <= max(2, int(fps * 1.25)):
                    if gap == 1:
                        success, original_frame = capture.read()
                    else:
                        for _ in range(gap - 1):
                            if not capture.grab():
                                break
                        success, original_frame = capture.retrieve()
                else:
                    capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
                    success, original_frame = capture.read()
                if not success or original_frame is None:
                    raise ValueError(f"Could not extract frame at {requested:.2f}s from {video_path}")
                previous_frame_index = target_index
                actual_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                actual_second = actual_ms / 1000.0 if np.isfinite(actual_ms) and actual_ms > 0 else requested
                chunk_frames += 1
                frame_h, frame_w = original_frame.shape[:2]
                detect_frame = original_frame
                max_dimension = self.options.detection_max_dimension
                if max(frame_h, frame_w) > max_dimension:
                    scale = max_dimension / max(frame_h, frame_w)
                    detect_frame = cv2.resize(original_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if poster is None:
                    poster = _LegacyVideoAnalyzer._encode_jpeg(detect_frame, 75)
                rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
                result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
                candidates: list[tuple[tuple[int, int, int, int], object | None]] = []
                if result.face_landmarks:
                    for landmarks in result.face_landmarks:
                        xs = [point.x for point in landmarks if np.isfinite(point.x)]
                        ys = [point.y for point in landmarks if np.isfinite(point.y)]
                        if not xs or not ys:
                            continue
                        x0, y0 = max(0, int(min(xs) * frame_w)), max(0, int(min(ys) * frame_h))
                        x1, y1 = min(frame_w, int(max(xs) * frame_w) + 1), min(frame_h, int(max(ys) * frame_h) + 1)
                        candidates.append(((x0, y0, x1, y1), landmarks))
                else:
                    gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
                    boxes = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                    scale_x, scale_y = frame_w / detect_frame.shape[1], frame_h / detect_frame.shape[0]
                    for x, y, width, height in boxes:
                        box = (int(x * scale_x), int(y * scale_y), int((x + width) * scale_x), int((y + height) * scale_y))
                        candidates.append((box, None))
                prepared: list[tuple[tuple[int, int, int, int], bytes, np.ndarray | None, float, str | None]] = []
                for box, landmarks in candidates:
                    x0, y0, x1, y1 = box
                    if x1 <= x0 or y1 <= y0:
                        continue
                    crop = original_frame[y0:y1, x0:x1]
                    thumb = _LegacyVideoAnalyzer._encode_jpeg(crop, 80)
                    if landmarks is None:
                        prepared.append((box, thumb, None, 0.0, "haar_fallback_unclassified"))
                        continue
                    aligned, quality, reason = self._align_and_quality(original_frame, landmarks, box)
                    prepared.append((box, thumb, aligned, quality, reason))
                valid = [item for item in prepared if item[2] is not None]
                embeddings = self._infer_batch([item[2] for item in valid]) if valid else []
                embedding_index = 0
                for box, thumb, aligned, quality, reason in prepared:
                    embedding = None
                    if aligned is not None:
                        embedding = embeddings[embedding_index]
                        embedding_index += 1
                    x0, y0, x1, y1 = box
                    faces.append(FaceSample(actual_second, (x0 / frame_w, y0 / frame_h, (x1 - x0) / frame_w, (y1 - y0) / frame_h), thumb, embedding, self.embedding_model if embedding is not None else None, quality, reason))
                completed = min(requested + interval, final_second)
                if progress is not None:
                    progress(completed, duration)
                requested += interval
            if faces or cancelled or completed >= final_second:
                yield AnalysisChunk(duration, poster, faces, completed, cancelled)
        finally:
            capture.release()

    def _align_and_quality(self, frame: np.ndarray, landmarks: list, box: tuple[int, int, int, int]) -> tuple[np.ndarray | None, float, str | None]:
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        if min(width, height) < self.options.min_face_size:
            return None, 0.0, "face_too_small"
        try:
            source = np.asarray([[landmarks[index].x * frame.shape[1], landmarks[index].y * frame.shape[0]] for index in (263, 362, 33, 133, 2)], dtype=np.float32)
        except (IndexError, AttributeError, TypeError):
            return None, 0.0, "landmarks_missing"
        if source.shape != (5, 2) or not np.isfinite(source).all():
            return None, 0.0, "landmarks_non_finite"
        eye_gap = float(np.linalg.norm(source[0] - source[2]))
        if eye_gap < max(12.0, width * 0.12):
            return None, 0.0, "eye_distance_invalid"
        eye_midpoint = (source[0] + source[2]) * 0.5
        yaw_ratio = abs(float(source[4, 0] - eye_midpoint[0])) / max(eye_gap, 1.0)
        if yaw_ratio > 0.48:
            return None, 0.0, "pose_yaw_out_of_range"
        raw_template = np.asarray([[0.8595674595992, 0.2134981538014], [0.6460604764104, 0.2289674387677], [0.1205750620789, 0.2137274526848], [0.3340850613712, 0.2290642403242], [0.4901123135679, 0.6277975316475]], dtype=np.float32)
        destination = (raw_template + 0.1) / 1.2 * 224.0
        transform, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
        if transform is None or not np.isfinite(transform).all():
            return None, 0.0, "alignment_failed"
        residual_points = cv2.transform(source[None, :, :], transform)[0]
        residual = float(np.sqrt(np.mean(np.sum((residual_points - destination) ** 2, axis=1))))
        if not np.isfinite(residual) or residual > 24.0:
            return None, 0.0, "alignment_residual_high"
        aligned_bgr = cv2.warpAffine(frame, transform, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        saturation = float(np.mean(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[:, :, 1]))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if brightness < 18 or brightness > 242:
            return None, 0.0, "brightness_out_of_range"
        if saturation > 250:
            return None, 0.0, "saturation_out_of_range"
        if blur < 12.0:
            return None, 0.0, "blur_too_high"
        blur_score = min(1.0, blur / 180.0)
        light_score = max(0.0, 1.0 - abs(brightness - 128.0) / 150.0)
        pose_score = max(0.0, 1.0 - yaw_ratio / 0.6)
        align_score = max(0.0, 1.0 - residual / 32.0)
        quality = float(np.clip(0.30 * blur_score + 0.20 * light_score + 0.25 * pose_score + 0.25 * align_score, 0.0, 1.0))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return ((chw - mean) / std).astype(np.float32), quality, None

    def _infer_batch(self, tensors: list[np.ndarray]) -> list[list[float]]:
        if not tensors:
            return []
        input_name = self._input.name
        if self._dynamic_batch:
            rows = []
            for offset in range(0, len(tensors), 8):
                batch = np.stack(tensors[offset : offset + 8]).astype(np.float32)
                outputs = self._session.run([self._output.name], {input_name: batch})[0]
                rows.extend(np.asarray(outputs, dtype=np.float32).reshape(len(batch), -1))
            vectors = np.asarray(rows, dtype=np.float32).reshape(len(tensors), -1)
        else:
            rows = [self._session.run([self._output.name], {input_name: tensor[None, ...].astype(np.float32)})[0] for tensor in tensors]
            vectors = np.asarray(rows, dtype=np.float32).reshape(len(tensors), -1)
        if vectors.shape[1] != 256 or not np.isfinite(vectors).all():
            raise RuntimeError(f"FACE01 produced invalid embeddings with shape {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
            raise RuntimeError("FACE01 produced a zero or non-finite embedding")
        vectors = vectors / norms
        return vectors.astype(np.float32).tolist()

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True


class VideoAnalyzer:
    """Platform-selected analyzer; Darwin keeps the original OpenVINO path."""

    def __init__(self, model_xml: Path | None = None, options: RecognitionOptions | None = None) -> None:
        config = default_config_path()
        self.options = options or RecognitionOptions.load(config if config.is_file() else None)
        self.options.validate()
        if self.options.backend == "openvino":
            legacy_path = Path(model_xml) if model_xml is not None else Path(__file__).resolve().parents[1] / "Sources" / "VideoAtlas" / "Resources" / "Models" / "face-reidentification-retail-0095.xml"
            self._implementation = _LegacyVideoAnalyzer(legacy_path)
            self.embedding_model = "openvino:face-reidentification-retail-0095:v1"
            self.runtime_description = "OpenVINO CPU; legacy 128x128 BGR alignment"
        else:
            self._implementation = _Face01VideoAnalyzer(self.options)
            self.embedding_model = self._implementation.embedding_model
            self.runtime_description = self._implementation.runtime_description

    def iter_chunks(self, video_path: Path, interval: float, start: float = 0, end: float | None = None, stop: Callable[[], bool] | None = None, progress: Callable[[float, float], None] | None = None) -> Iterator[AnalysisChunk]:
        for chunk in self._implementation.iter_chunks(video_path, interval, start, end, stop, progress):
            if self.options.backend == "openvino":
                for face in chunk.faces:
                    if face.embedding is not None:
                        face.embedding_model = self.embedding_model
            yield chunk

    def close(self) -> None:
        close = getattr(self._implementation, "close", None)
        if close is not None:
            close()
