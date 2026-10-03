# Windows continuation: FACE01 / RTX 4060 Ti

This is an **unfinished implementation snapshot**, saved at the user's request on 2026-10-03 (Japan time). Development was stopped before completion. No tests, compilation, Python imports, app launches, real-video runs, or benchmarks were performed for these changes. Do not describe the snapshot as working or its recognition/speed improvements as measured.

## User's current request

- Improve the distinction between similar **Japanese** faces using FACE01's JAPANESE FACE model.
- Preserve the macOS version's existing OpenVINO analysis behavior.
- Optimize the Windows version for **RTX 4060 Ti**, accommodating both 8 GB and 16 GB variants, with CUDA / TensorRT options.
- Implement only; **do not run tests or launch checks until the user authorizes them**.
- Continue on Windows. The user explicitly asked to stop development here and publish the current work.
- Use cost-conscious subagent orchestration; **do not use Astra as a subagent**.
- Be truthful and quantitative, and consult current primary sources where necessary.
- If editing README, make English the default, describe capabilities for non-engineers, and provide Japanese, Chinese, Hindi, Spanish, Arabic, French, Indonesian, Korean, Russian, and Portuguese choices. Existing README files have not yet been updated for these changes.

## Changes currently written

- `videoatlas/recognition.py`: optional JSON runtime configuration; macOS defaults to OpenVINO and Windows defaults to FACE01.
- `videoatlas/analyzer.py`: original OpenVINO implementation retained behind a wrapper; new FACE01 path, RGB 224×224 / ImageNet normalization, original-resolution crops, quality rejection, conditional batching, and provider fallback/session caching scaffolding.
- `videoatlas/storage.py`: backward-compatible model/quality metadata and Windows app-data storage path.
- `videoatlas/grouping.py`: model-space separation, stricter FACE01 matching, preservation of manual corrections, and disabled FACE01 automatic group reconciliation. The current proposed FACE01 cosine-distance threshold is 0.18, with a runner-up margin of 0.08; these values are **uncalibrated**.
- `videoatlas/controller.py`: runtime status, model migration/reanalysis flow, and persisted acceleration/precision settings.
- `videoatlas/ui.py` and `videoatlas/localization.py`: Windows-only acceleration and precision controls; current menu text and actual provider behavior still need reconciliation.
- `Scripts/download_face01.py`: standard-library downloader for pinned FACE01 and MediaPipe assets, with size/SHA-256 checking and upstream model terms.
- `.gitignore`: downloaded FACE01 weights/terms and generated inference engines excluded from Git.

These are code changes only. Static reading during development found issues; some were corrected before work stopped, but the complete integration was not reviewed or executed.

## Outstanding implementation work

1. Remove the currently written **unsupported** TensorRT provider option `trt_use_tf32`. The final design decision, not yet implemented, is:
   - `accurate`: CUDA with `use_tf32=0`, then CPU, for strict FP32.
   - `balanced`: TensorRT FP16, then CUDA with TF32, then CPU.
   - TensorRT requested with `accurate`: clearly report CUDA/CPU fallback. Avoid process-wide TF32 environment changes.
2. Include precision in the FACE01 embedding identity; the current identity contains model checksum and preprocessing, but **does not yet distinguish FP32/FP16**. Different numerical profiles must not silently share the same embeddings/resume state.
3. Finish provider fallback reporting when a provider is absent or ONNX Runtime silently falls back. Active provider registration does not prove every model operation runs on GPU.
4. Complete the RTX 4060 Ti optimization: bounded session cache, reusable fixed-shape GPU input/output buffers / I/O binding, CUDA memory limit, safe TensorRT engine/timing cache, and explicit model input validation. Current TensorRT workspace is 1 GiB; this is not a bound on all application VRAM.
5. The session cache is currently unbounded. Its key should include all relevant runtime/configuration values and GPU/DLL identity. The TensorRT engine cache path attempts to use model checksum, ORT/TensorRT versions, GPU UUID, driver and precision; confirm exact runtime identity and create its directory safely. Skip persistent engine caching when identity cannot be established.
6. Review frame-sampler seek/grab/read positions and cancellation, generator/resource cleanup, partial-initialization cleanup, face quality/alignment geometry, and preserved manual/excluded records across reanalysis. Existing legacy Mac behavior must remain unchanged.
7. Add Windows dependency/setup files; **they have not been written**. `requirements-python.txt` is still the original macOS-oriented file. A documented compatibility target is ONNX Runtime GPU **1.22.0**, CUDA **12.8**, cuDNN **9**, TensorRT **10.9**. Install only one ONNX Runtime distribution per environment. TensorRT DLLs require a separate installation. Verify official compatibility before implementation; do not silently choose a newer CUDA-major wheel.
8. Update Windows instructions and the public README language selector. Do not equate FACE01's upstream Japanese-face results with this application's measured accuracy. The MediaPipe five-point alignment is an approximation of the upstream dlib chip, not identical preprocessing.

## Assets and licenses

Downloaded models are local ignored files; they are **not included in this Git snapshot**. On a fresh Windows checkout, the supplied downloader can obtain both assets without importing the app:

```powershell
python Scripts/download_face01.py
```

- FACE01 source: `yKesamaru/FACE01_DEV`, revision `afec7ebac709f14224353e7f8b6539711899b1ff`, `face01lib/models/JAPANESE_FACE_V1.onnx`.
- Size: 26,083,027 bytes. SHA-256: `e7ca51f4bc85f73ddb830683ac6a09077909fa45a52b2bff41a9c6e8ff267e2f`.
- MediaPipe task size: 3,758,596 bytes. SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- The downloader saves FACE01_DEV's model terms separately. VideoAtlas's MIT license does not cover these weights. The DEV and trained_models repositories contain different license texts; use the pinned DEV source and review its terms before production/commercial use.

Primary references used during development:

- [FACE01 model preprocessing](https://github.com/yKesamaru/FACE01_DEV/blob/afec7ebac709f14224353e7f8b6539711899b1ff/face01lib/api.py)
- [FACE01_DEV model terms](https://github.com/yKesamaru/FACE01_DEV/blob/afec7ebac709f14224353e7f8b6539711899b1ff/LICENSE/LICENSE)
- [ONNX Runtime CUDA requirements and DLL preload](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime TensorRT requirements and caching](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [NVIDIA TensorRT precision controls](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)

## Windows continuation prompt

Read `WINDOWS_HANDOFF.md` first. Continue the unfinished Windows FACE01 / RTX 4060 Ti implementation, fix the recorded integration issues, preserve macOS's current analysis behavior and all manual corrections, and finish the dependency/setup documentation. Do not test, launch, benchmark, or claim measured speed/accuracy unless I explicitly authorize verification. Use cost-conscious subagents without Astra.
