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
class VideoAnalyzer:
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
