from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Callable, Iterator

import cv2
import numpy as np

from .recognition import RecognitionOptions, default_config_path


@dataclass
class FaceSample:
    second: float
    bounding_box: tuple[float, float, float, float]
    thumbnail_jpeg: bytes
    embedding: list[float] | None
    embedding_model: str | None = None
    quality: float = 1.0
    rejection_reason: str | None = None


@dataclass
class AnalysisChunk:
    duration: float
    poster_jpeg: bytes | None
    faces: list[FaceSample]
    completed_second: float
    cancelled: bool


@dataclass
class _SessionState:
    session: object
    model_input: object
    model_output: object
    dynamic_batch: bool
    providers: tuple[str, ...]
    profile: str
    lock: object


_SESSION_CACHE: OrderedDict[tuple[object, ...], _SessionState] = OrderedDict()
_SESSION_CACHE_LOCK = threading.RLock()
_DLL_DIGEST_CACHE: OrderedDict[tuple[object, ...], str] = OrderedDict()
_FACE01_PREPROCESSING = "face01-rgb224-nchw-imagenet-align5-v3"
_FACE01_MODEL_SHA256 = "e7ca51f4bc85f73ddb830683ac6a09077909fa45a52b2bff41a9c6e8ff267e2f"
_LANDMARK_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
_EMBEDDING_SIZE = 512
_CUDA_ARENA_BYTES = 2 * 1024 ** 3
_TRT_WORKSPACE_BYTES = 1024 ** 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dll_identity(path: Path) -> tuple[str, str]:
    path = path.resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _SESSION_CACHE_LOCK:
        digest = _DLL_DIGEST_CACHE.get(key)
        if digest is None:
            digest = _sha256_file(path)
            _DLL_DIGEST_CACHE[key] = digest
            while len(_DLL_DIGEST_CACHE) > 128:
                _DLL_DIGEST_CACHE.popitem(last=False)
        else:
            _DLL_DIGEST_CACHE.move_to_end(key)
    return str(path), digest


def _loaded_gpu_dlls() -> tuple[tuple[str, str], ...]:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    modules = (ctypes.c_void_p * 4096)()
    needed = ctypes.c_ulong()
    psapi.EnumProcessModulesEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    if not psapi.EnumProcessModulesEx(kernel.GetCurrentProcess(), modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
        raise OSError("Cannot enumerate loaded GPU libraries")
    if needed.value > ctypes.sizeof(modules):
        raise OSError("Loaded library list exceeds its allocation")
    paths: set[Path] = set()
    prefixes = ("cud", "cublas", "cufft", "curand", "cusparse", "nvinfer", "nvonnxparser", "nvrtc", "nvcuda", "nvjitlink")
    for module in modules[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        name = ctypes.create_unicode_buffer(32768)
        length = kernel.GetModuleFileNameW(module, name, len(name))
        if not length or length >= len(name):
            raise OSError("Cannot identify a loaded library")
        path = Path(name.value)
        if path.name.lower().startswith(prefixes):
            paths.add(path)
    for directory in {path.parent for path in paths}:
        paths.update(path for path in directory.glob("*.dll") if path.name.lower().startswith(prefixes))
    return tuple(sorted(_dll_identity(path) for path in paths))


def _gpu_identity(device_id: int) -> tuple[object, ...] | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_DEVICE_ORDER"):
        return None
    try:
        driver = ctypes.WinDLL("nvcuda.dll")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGetUuid_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
        driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        device, version, memory = ctypes.c_int(), ctypes.c_int(), ctypes.c_size_t()
        uuid = (ctypes.c_ubyte * 16)()
        results = (
            driver.cuInit(0),
            driver.cuDeviceGet(ctypes.byref(device), device_id),
            driver.cuDeviceGetUuid_v2(uuid, device),
            driver.cuDriverGetVersion(ctypes.byref(version)),
            driver.cuDeviceTotalMem_v2(ctypes.byref(memory), device),
        )
        if any(result != 0 for result in results) or not any(uuid) or version.value <= 0:
            return None
        dlls = _loaded_gpu_dlls()
        names = [Path(path).name.lower() for path, _ in dlls]
        if not any(name.startswith("cudart") for name in names) or not any(name.startswith("cudnn") for name in names):
            return None
        return (bytes(uuid).hex(), version.value, memory.value, dlls)
    except (AttributeError, OSError, ValueError):
        return None


class VideoAnalyzer:
    _CHUNK_SIZE = 60

    def __init__(self, options: RecognitionOptions | None = None) -> None:
        if sys.platform != "win32":
            raise RuntimeError("VideoAtlas supports Windows only")
        config_path = default_config_path()
        self.options = options or RecognitionOptions.load(config_path if config_path.is_file() else None)
        self.options.validate()
        self._closed = False
        self._landmarker = None
        self._session = None
        self._state = None
        self._dll_directory_handle = None
        self._gpu_buffers: OrderedDict[int, tuple[object, object, object]] = OrderedDict()
        self._active = False
        self._lifecycle_lock = threading.RLock()
        self._dll_libraries: list[object] = []
        try:
            self.model_path = self.options.model_path.expanduser().resolve()
            if not self.model_path.is_file():
                raise FileNotFoundError(f"FACE01 model is missing: {self.model_path}. Run Scripts/download_face01.py.")
            self.model_sha256 = _sha256_file(self.model_path)
            default_model = Path(__file__).parent / "resources" / "JAPANESE_FACE_V1.onnx"
            if self.model_path == default_model.resolve() and self.model_sha256 != _FACE01_MODEL_SHA256:
                raise RuntimeError(f"FACE01 model checksum mismatch: {self.model_path}")
            task_path = Path(__file__).parent / "resources" / "face_landmarker.task"
            if not task_path.is_file():
                raise FileNotFoundError(f"MediaPipe FaceLandmarker model is missing: {task_path}")
            if _sha256_file(task_path) != _LANDMARK_MODEL_SHA256:
                raise RuntimeError(f"MediaPipe FaceLandmarker model checksum mismatch: {task_path}")
            if self.options.tensorrt_dll_dir is not None:
                dll_dir = self.options.tensorrt_dll_dir.expanduser().resolve()
                if not dll_dir.is_dir():
                    raise FileNotFoundError(f"TensorRT DLL directory does not exist: {dll_dir}")
                self._dll_directory_handle = os.add_dll_directory(str(dll_dir))
            self._state, report = self._load_session()
            self._session = self._state.session
            self._input = self._state.model_input
            self._output = self._state.model_output
            self._dynamic_batch = self._state.dynamic_batch
            self.embedding_model = f"face01:{self.model_sha256}:{_FACE01_PREPROCESSING}:{self._state.profile}"
            self.runtime_description = report
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(str(cascade_path))
            if self._detector.empty():
                raise RuntimeError(f"OpenCV face detector could not be loaded: {cascade_path}")
            try:
                import mediapipe as mp
            except ImportError as error:
                raise RuntimeError("Python package 'mediapipe' is required for face landmarks") from error
            task_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=20,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(task_options)
            self._mp = mp
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    @staticmethod
    def _validate_model(session: object) -> tuple[object, object, bool]:
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("FACE01 requires exactly one model input and one output")
        model_input, model_output = inputs[0], outputs[0]
        if model_input.type != "tensor(float)" or tuple(model_input.shape[1:]) != (3, 224, 224) or len(model_input.shape) != 4:
            raise ValueError(f"FACE01 input must be float32 [N,3,224,224], got {model_input.type} {model_input.shape}")
        dynamic = model_input.shape[0] is None or isinstance(model_input.shape[0], str)
        if not dynamic and model_input.shape[0] != 1:
            raise ValueError(f"FACE01 static batch must be 1, got {model_input.shape}")
        output_batch = model_output.shape[0] if model_output.shape else None
        if model_output.type != "tensor(float)" or len(model_output.shape) != 2 or model_output.shape[1] != _EMBEDDING_SIZE:
            raise ValueError(f"FACE01 output must be float32 [N,{_EMBEDDING_SIZE}], got {model_output.type} {model_output.shape}")
        if isinstance(output_batch, int) and (output_batch != 1 or dynamic):
            raise ValueError(f"FACE01 output batch is incompatible with its input: {model_output.shape}")
        return model_input, model_output, dynamic

    def _load_session(self) -> tuple[_SessionState, str]:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install the Windows ONNX Runtime requirements for FACE01") from error
        self._ort = ort
        available = set(ort.get_available_providers())
        version = getattr(ort, "__version__", "unknown")
        notes: list[str] = []
        requested = self.options.acceleration
        balanced = self.options.precision == "balanced"
        preferred = ["CPUExecutionProvider"]
        if requested != "cpu":
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if balanced and requested in {"auto", "tensorrt"}:
                preferred.insert(0, "TensorrtExecutionProvider")
            elif requested == "tensorrt":
                notes.append("TensorRT skipped: Accurate requires CUDA FP32 with TF32 disabled")
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception as error:
                    notes.append(f"CUDA/cuDNN preload: {error}")
            if "TensorrtExecutionProvider" in preferred and "TensorrtExecutionProvider" in available:
                for library_name in ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll"):
                    try:
                        self._dll_libraries.append(ctypes.WinDLL(library_name))
                    except OSError as error:
                        notes.append(f"TensorRT DLL load: {error}")
        unavailable = [name for name in preferred if name not in available]
        if unavailable:
            notes.append("unavailable: " + ", ".join(unavailable))
        gpu_identity = _gpu_identity(self.options.gpu_device_id) if requested != "cpu" else None
        runtime_identity = None
        try:
            capi_path = Path(ort.__file__).resolve().parent / "capi"
            files = sorted((*capi_path.glob("*.dll"), *capi_path.glob("*.pyd")))
            if files and version != "unknown":
                runtime_identity = (version, tuple(_dll_identity(path) for path in files), ort.get_build_info())
        except (AttributeError, OSError):
            pass
        search_identity = (os.environ.get("PATH", ""), os.environ.get("CUDA_VISIBLE_DEVICES", ""), os.environ.get("CUDA_PATH", ""), os.environ.get("CUDA_DEVICE_ORDER", ""))
        cache_key = (self.model_path, self.model_sha256, self.options, runtime_identity, gpu_identity, search_identity)
        cacheable = runtime_identity is not None and (requested == "cpu" or gpu_identity is not None)
        with _SESSION_CACHE_LOCK:
            cached = _SESSION_CACHE.get(cache_key) if cacheable else None
            if cached is not None:
                _SESSION_CACHE.move_to_end(cache_key)
                return cached, self._provider_report(cached, version, notes + ["reused session"])
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.options.threads
            session_options.inter_op_num_threads = 1
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            cpu = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
            cpu.disable_fallback()
            model_input, model_output, dynamic = self._validate_model(cpu)
            cuda_options = {
                "device_id": str(self.options.gpu_device_id),
                "use_tf32": "1" if balanced else "0",
                "gpu_mem_limit": str(_CUDA_ARENA_BYTES),
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "cudnn_conv_use_max_workspace": "0",
                "do_copy_in_default_stream": "1",
            }
            candidates: list[list[object]] = []
            if "TensorrtExecutionProvider" in preferred and "TensorrtExecutionProvider" in available:
                trt_options = {
                    "device_id": str(self.options.gpu_device_id),
                    "trt_fp16_enable": "1",
                    "trt_max_workspace_size": str(_TRT_WORKSPACE_BYTES),
                    "trt_engine_cache_enable": "0",
                    "trt_timing_cache_enable": "0",
                    "trt_force_timing_cache": "0",
                    "trt_force_sequential_engine_build": "1",
                    "trt_auxiliary_streams": "0",
                }
                if dynamic:
                    for option, batch_size in (("trt_profile_min_shapes", 1), ("trt_profile_opt_shapes", 4), ("trt_profile_max_shapes", 8)):
                        trt_options[option] = f"{model_input.name}:{batch_size}x3x224x224"
                trt_cache = self._safe_tensorrt_cache(runtime_identity, gpu_identity, trt_options, cuda_options)
                if trt_cache is not None:
                    trt_options.update({
                        "trt_engine_cache_enable": "1",
                        "trt_engine_cache_path": str(trt_cache),
                        "trt_engine_cache_prefix": "face01",
                        "trt_timing_cache_enable": "1",
                        "trt_timing_cache_path": str(trt_cache),
                    })
                else:
                    notes.append("TensorRT persistent cache skipped: runtime identity or writable cache unavailable")
                providers: list[object] = [("TensorrtExecutionProvider", trt_options)]
                if "CUDAExecutionProvider" in available:
                    providers.append(("CUDAExecutionProvider", cuda_options))
                providers.append("CPUExecutionProvider")
                candidates.append(providers)
            if "CUDAExecutionProvider" in preferred and "CUDAExecutionProvider" in available:
                candidates.append([("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"])
            candidates.append(["CPUExecutionProvider"])
            state = None
            for providers in candidates:
                label = tuple(item[0] if isinstance(item, tuple) else item for item in providers)
                try:
                    session = cpu if label == ("CPUExecutionProvider",) else ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=providers)
                    session.disable_fallback()
                    actual = tuple(session.get_providers())
                    if actual != label:
                        notes.append(f"initialization fallback: requested {', '.join(label)}; registered {', '.join(actual)}")
                        if label[0] not in actual:
                            continue
                    registered_options = session.get_provider_options()
                    if "CUDAExecutionProvider" in actual:
                        effective_tf32 = registered_options.get("CUDAExecutionProvider", {}).get("use_tf32")
                        if effective_tf32 != cuda_options["use_tf32"]:
                            raise RuntimeError(f"CUDA use_tf32 was not honored: {effective_tf32}")
                    if "TensorrtExecutionProvider" in actual:
                        if registered_options.get("TensorrtExecutionProvider", {}).get("trt_fp16_enable") != "1":
                            raise RuntimeError("TensorRT FP16 setting was not honored")
                        profile = "trt-fp16-cuda-tf32" if "CUDAExecutionProvider" in actual else "trt-fp16-cpu-fp32"
                    elif "CUDAExecutionProvider" in actual:
                        profile = "cuda-tf32" if balanced else "cuda-fp32"
                    else:
                        profile = "cpu-fp32"
                    numeric_gpu_identity = gpu_identity if profile != "cpu-fp32" else None
                    if runtime_identity is None or (profile != "cpu-fp32" and numeric_gpu_identity is None):
                        identity = os.urandom(10).hex()
                        notes.append("Runtime identity incomplete: embeddings belong to this session; reanalysis is required after restarting analysis")
                    else:
                        identity = hashlib.sha256(repr((runtime_identity, numeric_gpu_identity, actual, self.options.threads, cuda_options["use_tf32"] if "CUDAExecutionProvider" in actual else None)).encode("utf-8")).hexdigest()[:20]
                    state = _SessionState(session, model_input, model_output, dynamic, actual, f"{profile}-ort{version}-{identity}", threading.Lock())
                    if cacheable and actual[0] == preferred[0]:
                        for key, entry in list(_SESSION_CACHE.items()):
                            if entry.providers[0] != "CPUExecutionProvider" and actual[0] != "CPUExecutionProvider":
                                del _SESSION_CACHE[key]
                        _SESSION_CACHE[cache_key] = state
                        while len(_SESSION_CACHE) > 2:
                            _SESSION_CACHE.popitem(last=False)
                    return state, self._provider_report(state, version, notes)
                except Exception as error:
                    notes.append(f"{', '.join(label)} initialization failed: {error}")
            raise RuntimeError("Could not initialize FACE01: " + " | ".join(notes))

    def _safe_tensorrt_cache(self, runtime_identity: object, gpu_identity: object, trt_options: dict, cuda_options: dict) -> Path | None:
        if runtime_identity is None or gpu_identity is None:
            return None
        dll_names = [Path(path).name.lower() for path, _ in gpu_identity[3]]
        if not all(any(name.startswith(prefix) for name in dll_names) for prefix in ("nvinfer_", "nvinfer_plugin_", "nvonnxparser_")):
            return None
        identity = (self.model_sha256, _FACE01_PREPROCESSING, runtime_identity, gpu_identity, tuple(sorted(trt_options.items())), tuple(sorted(cuda_options.items())))
        key = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()
        try:
            root = (self.options.cache_dir or default_config_path().parent / "InferenceCache").expanduser().resolve()
            cache = root / key
            cache.mkdir(parents=True, exist_ok=True)
            if cache.resolve().parent != root:
                return None
            manifest = cache / "identity.json"
            expected = json.dumps({"identity_sha256": key, "model_sha256": self.model_sha256}, sort_keys=True)
            try:
                with manifest.open("x", encoding="utf-8") as stream:
                    stream.write(expected)
            except FileExistsError:
                if manifest.read_text(encoding="utf-8") != expected:
                    return None
            return cache
        except (OSError, ValueError):
            return None

    def _provider_report(self, state: _SessionState, version: str, notes: list[str]) -> str:
        report = f"FACE01 ONNX Runtime {version}; registered providers: {', '.join(state.providers)}; numerical profile: {state.profile.split('-ort')[0]}"
        report += "; individual operations may use CPU fallback"
        if state.providers[0] != "CPUExecutionProvider":
            report += "; CUDA arena cap 2 GiB, TensorRT workspace cap 1 GiB (not total VRAM)"
        if notes:
            report += "; " + " | ".join(notes)
        return report

    @staticmethod
    def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
        if image.size == 0:
            raise ValueError("Cannot encode an empty image")
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("OpenCV could not encode a JPEG image")
        return encoded.tobytes()

    def iter_chunks(self, video_path: Path, interval: float, start: float = 0, end: float | None = None, stop: Callable[[], bool] | None = None, progress: Callable[[float, float], None] | None = None) -> Iterator[AnalysisChunk]:
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be a finite number greater than zero")
        if not np.isfinite(start) or start < 0:
            raise ValueError("start must be a finite number greater than or equal to zero")
        if end is not None and (not np.isfinite(end) or end < 0):
            raise ValueError("end must be a finite number greater than or equal to zero")
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Analyzer is closed")
            if self._active:
                raise RuntimeError("Analyzer is already processing a video")
            self._active = True
        capture = None
        try:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open video stream: {video_path}")
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0 or not np.isfinite(frame_count) or frame_count < 1:
                raise ValueError(f"Cannot read a valid video stream: {video_path}")
            duration = float(frame_count / fps)
            if end is not None and end <= start and start < duration:
                raise ValueError("end must be greater than start")
            final_second = min(duration, duration if end is None else float(end))
            requested = min(float(start), duration)
            faces: list[FaceSample] = []
            chunk_frames = 0
            poster: bytes | None = None
            completed = requested
            next_frame_index = 0
            cancelled = False
            while requested < final_second:
                if self._stopped(stop):
                    cancelled = True
                    break
                if chunk_frames >= self._CHUNK_SIZE:
                    yield AnalysisChunk(duration, poster, faces, completed, False)
                    faces, chunk_frames, poster = [], 0, None
                    if self._stopped(stop):
                        cancelled = True
                        break
                target_index = min(int(frame_count) - 1, max(0, int(np.floor(requested * fps + 1e-7))))
                gap = target_index - next_frame_index
                if 0 <= gap <= max(2, int(fps * 1.25)):
                    skipped = True
                    for _ in range(gap):
                        if self._stopped(stop):
                            cancelled = True
                            break
                        if not capture.grab():
                            skipped = False
                            break
                    if cancelled:
                        break
                    if not skipped:
                        raise ValueError(f"Could not skip to frame at {requested:.2f}s from {video_path}")
                else:
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, target_index):
                        raise ValueError(f"Video decoder cannot seek to frame {target_index}: {video_path}")
                success, original_frame = capture.read()
                if not success or original_frame is None or original_frame.size == 0:
                    raise ValueError(f"Could not extract frame at {requested:.2f}s from {video_path}")
                position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                if not np.isfinite(position) or abs(position - (target_index + 1)) > 0.5:
                    raise ValueError(f"Video decoder reported an unexpected position at {requested:.2f}s: {video_path}")
                next_frame_index = target_index + 1
                frame_faces, frame_poster = self._process_frame(original_frame, requested, stop, poster is None)
                if frame_faces is None or self._stopped(stop):
                    cancelled = True
                    break
                faces.extend(frame_faces)
                if poster is None:
                    poster = frame_poster
                chunk_frames += 1
                next_second = requested + interval
                if not np.isfinite(next_second) or next_second <= requested:
                    raise ValueError("Sampling interval cannot advance the video position")
                completed = min(next_second, final_second)
                if progress is not None:
                    progress(completed, duration)
                requested = completed
            if chunk_frames or cancelled or completed >= final_second:
                yield AnalysisChunk(duration, poster, faces, completed, cancelled)
        finally:
            if capture is not None:
                capture.release()
            with self._lifecycle_lock:
                self._active = False
                if self._closed:
                    self._release_resources()

    def _stopped(self, stop: Callable[[], bool] | None) -> bool:
        return self._closed or (stop is not None and stop())

    def _process_frame(self, original_frame: np.ndarray, second: float, stop: Callable[[], bool] | None, include_poster: bool) -> tuple[list[FaceSample] | None, bytes | None]:
        if self._stopped(stop):
            return None, None
        frame_h, frame_w = original_frame.shape[:2]
        detect_frame = original_frame
        max_dimension = self.options.detection_max_dimension
        if max(frame_h, frame_w) > max_dimension:
            scale = max_dimension / max(frame_h, frame_w)
            detect_frame = cv2.resize(original_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        candidates: list[tuple[tuple[int, int, int, int], object | None]] = []
        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                if not landmarks or any(not np.isfinite(point.x) or not np.isfinite(point.y) for point in landmarks):
                    continue
                x0 = int(np.clip(np.floor(min(point.x for point in landmarks) * frame_w), 0, frame_w))
                y0 = int(np.clip(np.floor(min(point.y for point in landmarks) * frame_h), 0, frame_h))
                x1 = int(np.clip(np.ceil(max(point.x for point in landmarks) * frame_w), 0, frame_w))
                y1 = int(np.clip(np.ceil(max(point.y for point in landmarks) * frame_h), 0, frame_h))
                candidates.append(((x0, y0, x1, y1), landmarks))
        else:
            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            boxes = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            scale_x, scale_y = frame_w / detect_frame.shape[1], frame_h / detect_frame.shape[0]
            for x, y, width, height in boxes:
                box = (max(0, int(x * scale_x)), max(0, int(y * scale_y)), min(frame_w, int((x + width) * scale_x)), min(frame_h, int((y + height) * scale_y)))
                candidates.append((box, None))
        prepared = []
        for box, landmarks in candidates:
            if self._stopped(stop):
                return None, None
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                continue
            thumb = self._encode_jpeg(original_frame[y0:y1, x0:x1], 80)
            if landmarks is None:
                prepared.append((box, thumb, None, 0.0, "haar_fallback_unclassified"))
            else:
                aligned, quality, reason = self._align_and_quality(original_frame, landmarks, box)
                prepared.append((box, thumb, aligned, quality, reason))
        tensors = [item[2] for item in prepared if item[2] is not None]
        try:
            embeddings = self._infer_batch(tensors, stop)
        except Exception as error:
            raise RuntimeError(f"FACE01 inference failed with registered providers {', '.join(self._state.providers)}; automatic runtime provider changes are disabled. Select CPU or another acceleration setting to reanalyze. {error}") from error
        if embeddings is None or self._stopped(stop):
            return None, None
        frame_faces = []
        embedding_index = 0
        for box, thumb, aligned, quality, reason in prepared:
            embedding = None
            if aligned is not None:
                embedding = embeddings[embedding_index]
                embedding_index += 1
            x0, y0, x1, y1 = box
            frame_faces.append(FaceSample(second, (x0 / frame_w, y0 / frame_h, (x1 - x0) / frame_w, (y1 - y0) / frame_h), thumb, embedding, self.embedding_model if embedding is not None else None, quality, reason))
        return frame_faces, self._encode_jpeg(detect_frame, 75) if include_poster else None

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
        if transform is None or not np.isfinite(transform).all() or np.linalg.det(transform[:, :2]) <= 0:
            return None, 0.0, "alignment_failed"
        residual_points = cv2.transform(source[None, :, :], transform)[0]
        residual = float(np.sqrt(np.mean(np.sum((residual_points - destination) ** 2, axis=1))))
        if not np.isfinite(residual) or residual > 24.0:
            return None, 0.0, "alignment_residual_high"
        aligned_bgr = cv2.warpAffine(frame, transform, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        corners = np.asarray([[[0, 0], [223, 0], [223, 223], [0, 223]]], dtype=np.float32)
        frame_corners = cv2.transform(corners, cv2.invertAffineTransform(transform))[0]
        if np.any(frame_corners < 0) or np.any(frame_corners[:, 0] >= frame.shape[1]) or np.any(frame_corners[:, 1] >= frame.shape[0]):
            support = cv2.warpAffine(np.ones(frame.shape[:2], dtype=np.uint8), transform, (224, 224), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if float(np.mean(support)) < 0.85:
                return None, 0.0, "alignment_outside_frame"
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

    def _infer_batch(self, tensors: list[np.ndarray], stop: Callable[[], bool] | None = None) -> list[list[float]] | None:
        rows = []
        batch_limit = 8 if self._dynamic_batch else 1
        for offset in range(0, len(tensors), batch_limit):
            if self._stopped(stop):
                return None
            batch = np.ascontiguousarray(np.stack(tensors[offset:offset + batch_limit]), dtype=np.float32)
            if batch.shape[1:] != (3, 224, 224) or not np.isfinite(batch).all():
                raise ValueError("FACE01 received an invalid input tensor")
            with self._state.lock:
                if self._stopped(stop):
                    return None
                if any(provider in self._state.providers for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")):
                    values = self._gpu_buffers.get(len(batch))
                    if values is None:
                        input_value = self._ort.OrtValue.ortvalue_from_shape_and_type(list(batch.shape), np.float32, "cuda", self.options.gpu_device_id)
                        output_value = self._ort.OrtValue.ortvalue_from_shape_and_type([len(batch), _EMBEDDING_SIZE], np.float32, "cuda", self.options.gpu_device_id)
                        binding = self._session.io_binding()
                        binding.bind_ortvalue_input(self._input.name, input_value)
                        binding.bind_ortvalue_output(self._output.name, output_value)
                        values = (input_value, output_value, binding)
                        self._gpu_buffers[len(batch)] = values
                        while len(self._gpu_buffers) > 2:
                            self._gpu_buffers.popitem(last=False)
                    self._gpu_buffers.move_to_end(len(batch))
                    input_value, output_value, binding = values
                    input_value.update_inplace(batch)
                    binding.synchronize_inputs()
                    self._session.run_with_iobinding(binding)
                    binding.synchronize_outputs()
                    output = output_value.numpy()
                else:
                    output = self._session.run([self._output.name], {self._input.name: batch})[0]
            vectors = np.asarray(output, dtype=np.float32)
            if vectors.shape != (len(batch), _EMBEDDING_SIZE) or not np.isfinite(vectors).all():
                raise RuntimeError(f"FACE01 produced invalid embeddings: {vectors.shape}")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
                raise RuntimeError("FACE01 produced a zero or non-finite embedding")
            rows.extend((vectors / norms).astype(np.float32).tolist())
        return rows

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            if not self._active:
                self._release_resources()

    def _release_resources(self) -> None:
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        finally:
            self._landmarker = None
            self._gpu_buffers.clear()
            self._session = None
            self._state = None
            self._dll_libraries.clear()
            if self._dll_directory_handle is not None:
                self._dll_directory_handle.close()
                self._dll_directory_handle = None
