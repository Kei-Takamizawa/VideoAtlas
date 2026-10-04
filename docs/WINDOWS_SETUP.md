# Windows setup

Use 64-bit Windows, Python 3.12, and PowerShell. Open the project folder and run:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

The setup creates `.venv`, installs the pinned packages, and verifies downloaded JAPANESE FACE V1 and SCRFD model checksums. First-time setup needs internet access; analysis runs locally. Video formats are MP4, MOV, AVI, MKV, M4V, and WebM, subject to available codecs.

## Optional NVIDIA acceleration

```powershell
.\Scripts\setup_windows.cmd -Gpu
```

This installs ONNX Runtime GPU 1.22.0 with its CUDA/cuDNN dependencies instead of the CPU distribution. A compatible NVIDIA driver is required. Accurate mode requests CUDA FP32 with TF32 disabled, then CPU. GPU initialization or execution failure retries on CPU; the changed numerical profile has its own embedding identity.

On Windows, the application discovers installed NVIDIA package DLL directories and adds them to its process-local search paths so cuDNN can load its component libraries. It does not change system-wide PATH settings. The [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/#cuda-and-cudnn) describes the required library search paths.

Balanced mode may use TensorRT FP16, then CUDA TF32, then CPU for the primary model. TensorRT is a separate installation. ONNX Runtime's [CUDA compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) and [TensorRT compatibility table](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) describe the pinned runtime requirements. To supply TensorRT DLLs, configure `tensorrt_dll_dir` in `recognition.json`. SCRFD and AdaFace use CUDA or CPU; they do not use the primary model's TensorRT path.

Provider registration does not establish that every operation executes on the GPU. The primary CUDA arena is limited to 2 GiB and the TensorRT workspace to 1 GiB; these settings do not cap total application VRAM. No processing speed target or speedup is claimed.

## Optional AdaFace

```powershell
.\Scripts\setup_windows.cmd -AdaFace
# Both optional components:
.\Scripts\setup_windows.cmd -Gpu -AdaFace
```

AdaFace provisioning downloads the pinned official R50 MS1MV2 checkpoint (700,286,703 bytes), checks its SHA-256, and exports a local ONNX model. The exporter installs PyTorch and other extra packages; allow additional download time and disk space. The application uses ONNX Runtime after export.

In **Settings → Recognition**, enable AdaFace and choose `videoatlas/resources/adaface_ir50_ms1mv2.onnx`. Choose **Recalculate embeddings** on existing videos with aligned crop caches, or **Reanalyze all** if those caches are absent. Merely downloading AdaFace does not activate it. To reuse an already downloaded official checkpoint:

```powershell
.\.venv\Scripts\python.exe Scripts\setup_adaface.py --checkpoint "C:/Models/adaface_ir50_ms1mv2.ckpt"
```

Official sources: [FACE01](https://github.com/yKesamaru/FACE01_DEV), [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd), [AdaFace](https://github.com/mk-minchul/AdaFace). The source and third-party model terms are separate from this application's license. The provisioning scripts retain terms and provenance locally.

## Using the library

1. Add video folders. Recursive discovery is enabled initially and can be changed in Settings.
2. Start or resume analysis. Only pending/new videos are processed during a normal refresh.
3. Open **People** to see representative faces, videos, tracks, and appearance times.
   Left-click a face to open its videos and optional name field. Unnamed people have no generated aliases. A video can appear under several people.
4. Open **Review queue** and choose Same person (S), Different people (D), or Review later (L).
5. Use **Accuracy evaluation** after collecting reviewed pairs. Reports define both false-positive denominators and show candidate thresholds/model weights.

Automatic merging is initially off. Example thresholds are uncalibrated. When enabled, only HIGH comparisons can merge automatically; missing evidence, model disagreement, overlapping tracks, negative decisions, ambiguous multiple candidates, or unresolved reviewed history block merging. Every cluster member must support an addition.

## Stored data and updates

The default library is `%LOCALAPPDATA%/VideoAtlasPython`, with SQLite, generated thumbnails, aligned crops, settings, and rotating analysis logs. Use `python -m videoatlas --data-dir "C:/MyIndex"` for an explicit directory. Changing the database directory in Settings selects another library on the next launch; it does not move the current index or its images.

Removing a folder from the Settings list disables discovery while retaining indexed results. **Remove folder** and **Clear index** in the sidebar delete the index entries after confirmation. Original video files are never deleted by those actions. Renaming identical content preserves the video ID and results; replacing a file's content creates a new video ID and retains the prior reviewed index as missing.

Settings changes do not silently reanalyze ready videos. **Recalculate embeddings** reuses checksummed crops, **Regroup cached tracks** reapplies identity settings, and **Reanalyze all** reruns frame extraction/detection. Human corrections are retained; unresolved continuity is sent to review. Logs are available in the UI. Failed videos show their error and can be retried.

FACE01 receives RGB 224 × 224 input with ImageNet normalization. Its alignment approximates upstream dlib geometry using SCRFD eye centers; it is explicitly fingerprinted as a different preprocessing method. AdaFace receives its own BGR 112 × 112 crop normalized by mean/std 0.5. Pose and occlusion-quality checks use geometric proxies. Evaluate these choices on representative videos before adopting thresholds.

For masks, select **Settings → Recognition → Upper face (masks, experimental)**, keep automatic merging off, then reanalyze the videos. This mode tries three detector sizes, aligns using only two eye landmarks, and removes nose, mouth, and jaw pixels before embedding inference. Quality checks use the visible region. Full-face and upper-face preprocessing, embeddings, and lossless caches are separate; a full-face cache cannot be reused as an upper-face cache. All videos that should be compared must use the same region. Upper-face automatic merging is unavailable pending threshold calibration. Real-mask accuracy is not established by the supplied unmasked clips or digital mask tests.

The regular sample interval is 0.5 seconds; while faces are detected, the default interval is 0.2 seconds. Short appearances between samples can still be missed. The detector records every detected face and does not choose people using filenames or a dominant-person rule.

See [implementation status](IMPLEMENTATION_STATUS.md) for current verified checks and remaining evaluation limits.
