# Windows setup details

The quick setup uses Python 3.12 and creates an isolated `.venv`. Install a current 64-bit Python 3.12 release and make sure the Python Launcher (`py`) is available. Open PowerShell in the project folder.

## CPU setup

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

The setup downloads the Python packages and pinned face models. Model downloads are placed in `videoatlas\resources`; their size and SHA-256 are checked. The third-party FACE01 terms are saved as `FACE01-LICENSE.txt` in the same folder.

## NVIDIA CUDA setup

```powershell
.\Scripts\setup_windows.cmd -Gpu
.\Scripts\run_windows.cmd
```

This selects the ONNX Runtime GPU package 1.22.0 and its pip CUDA 12 / cuDNN 9 runtime dependencies. The setup uses ONNX Runtime's DLL preloader to locate those packages. A compatible NVIDIA driver is required; you do not need a separate CUDA Toolkit for this setup. Keep only one ONNX Runtime package in `.venv`: this setup installs the GPU distribution in place of the CPU distribution. CUDA acceleration is optional; CPU mode remains available.

ONNX Runtime's compatibility table identifies 1.22 as a CUDA 12.8 / cuDNN 9 build. Its declared `cuda` and `cudnn` extras install NVIDIA CUDA 12 and cuDNN 9 Python packages. If you choose to use a system CUDA installation instead, use CUDA 12.8 or newer with cuDNN 9. The TensorRT execution provider target is TensorRT 10.9 with CUDA 12.0–12.8. See the [CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) and [TensorRT provider requirements](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

## Optional TensorRT

TensorRT is not installed by the setup script. Install NVIDIA TensorRT 10.9 separately, alongside the compatible CUDA and cuDNN libraries, and set the `tensorrt_dll_dir` entry in `%LOCALAPPDATA%\VideoAtlasPython\recognition.json` to the directory containing its DLLs. Use JSON such as:

```json
{
  "acceleration": "tensorrt",
  "precision": "balanced",
  "tensorrt_dll_dir": "C:/NVIDIA/TensorRT-10.9/lib"
}
```

Use a real path from your TensorRT installation. TensorRT is used only with balanced precision; accurate precision selects CUDA with TF32 disabled and then CPU fallback. Provider availability does not mean every model operation runs on that provider.

## Recognition settings

The app supports CPU inference and optional NVIDIA acceleration, with bounded allocations suitable for either RTX 4060 Ti memory variant (8 GB or 16 GB). The CUDA memory arena limit is 2 GiB and the TensorRT workspace limit is 1 GiB; these do not limit total application VRAM. Dynamic models process up to eight faces per batch. Static models process one face at a time.

Accurate mode requests CUDA FP32 with TF32 disabled, then CPU. Balanced mode can use TensorRT FP16, then CUDA TF32, then CPU. Changing the numerical profile restarts analysis while preserving manual assignments and exclusions. No speed comparison between these modes has been measured.

FACE01 uses 224 × 224 RGB input and 512-value embeddings, following its [author's implementation](https://ykesamaru.github.io/FACE01_DEV/_modules/face01lib/api.html). The app's MediaPipe alignment approximates the upstream dlib face chip. Matching uses an uncalibrated cosine-distance threshold of 0.18 and ambiguity margin of 0.08; upstream accuracy figures do not measure this app's results.

## Data and limitations

The index, generated images, and runtime configuration are stored under `%LOCALAPPDATA%\VideoAtlasPython`. Original videos remain where they are. The downloaded weights are third-party files with separate terms. Model inference, playback, and recognition accuracy have not been verified for this Windows port.
