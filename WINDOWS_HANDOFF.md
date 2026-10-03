# Windows development notes

VideoAtlas targets 64-bit Windows. Follow [Windows setup](docs/WINDOWS_SETUP.md) for CPU or NVIDIA installation.

- FACE01 supplies the face embeddings; downloaded weights retain their separate model terms.
- Accurate precision uses CUDA FP32 with TF32 disabled, then CPU. TensorRT requests use this fallback.
- Balanced precision allows TensorRT FP16, CUDA TF32, then CPU.
- Embeddings and resume positions are separated by model, preprocessing, and numerical profile.
- Reanalysis preserves manual assignments and exclusions.
- Provider registration does not establish that every model operation executes on the GPU.
- Matching thresholds are provisional. No recognition accuracy or speed measurements are claimed.

Implementation was reviewed by reading the source. Tests, Python imports, compilation, app launches, and video benchmarks were not performed at the user's request. Future verification requires suitable video samples and explicit authorization.
