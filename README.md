# VideoAtlas

**Windows-only local video library.** VideoAtlas helps you browse videos and find scenes by the people who appear in them. It keeps your videos on your computer and lets you review and correct its face-based suggestions.

## What you can do

- Browse videos inside folders you choose and play them in the app.
- Find candidate people, related videos, and the times their faces appear.
- Pause and resume analysis, then assign, split, merge, or exclude face matches yourself.
- Keep the library index on your PC; the app does not move your original videos.

Automatic groups are suggestions based on visual similarity. They can be wrong, so review and correct them. Face matching has not been measured for accuracy. Processing is local; first-time setup downloads the required software and model files.

## Windows setup

Use 64-bit Windows, Python 3.12, and PowerShell. From the project folder, run:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

This installs the CPU-compatible setup. For an NVIDIA GPU with CUDA acceleration, run `.\Scripts\setup_windows.cmd -Gpu` instead. TensorRT is an optional separate NVIDIA installation; see [Windows setup details](docs/WINDOWS_SETUP.md).

## Language

English | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

## Privacy and models

VideoAtlas stores its index and generated images under your Windows local application data folder. It does not upload video or face data. Setup fetches the pinned FACE01 Japanese-face model and MediaPipe face-landmark model, then verifies their SHA-256 checksums. FACE01's separate model terms are saved beside the downloaded files; review them before use. The project's [MIT License](LICENSE) does not cover third-party models or dependencies.

## Project status

The Windows implementation has not been runtime tested: no tests, launched app, or real-video analysis have been run. The prior macOS app and its setup are not supported by this Windows-only release.
