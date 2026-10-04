# VideoAtlas

**Windows-only local video library.** VideoAtlas helps you browse videos and find scenes by the people who appear in them. It keeps your videos on your computer and lets you review and correct its face-based suggestions.

## What you can do

- Browse MP4, MOV, AVI, MKV, M4V, and WebM videos in the folders you choose.
- Find people, their related videos, and the times they appear using multiple face samples.
- Browse people by their faces; left-click a face to see videos and optionally add your own name. Every detected person in a shared video has a separate record.
- Try upper-face recognition for masks, with the nose, mouth, and jaw excluded. This mode is experimental and needs video reanalysis when selected.
- Review uncertain matches with **Same person**, **Different people**, or **Review later** (S, D, L).
- Pause and resume analysis, name people, merge or split groups, exclude faces, and choose representative images.
- Keep reviewed decisions through reanalysis and evaluate matching suggestions from those decisions.
- Keep the library index on your PC; the app does not move your original videos.

Matches can be wrong. Automatic merging across videos is off by default; use the review queue until you have evaluated the settings on your videos. A small supplied collection has been evaluated, but current thresholds still miss many matches. Filenames are never used to infer identity. Renaming an unchanged video keeps its analysis. Processing is local; first-time setup downloads software and models.

## Windows setup

Use 64-bit Windows, Python 3.12, and PowerShell. From the project folder, run:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

This installs the CPU setup. Add `-Gpu` for NVIDIA acceleration or `-AdaFace` to prepare the optional second recognition model. See [Windows setup details](docs/WINDOWS_SETUP.md), including how to enable AdaFace in Settings.

## Privacy and models

VideoAtlas stores its index and generated face images on your PC. It does not upload video or face data. Setup verifies the downloaded SCRFD and Japanese-face models; AdaFace is downloaded and exported locally when requested. Third-party terms are saved beside the models. The project's [MIT License](LICENSE) does not cover third-party models or dependencies.

## Project status

The [project design](docs/design-specification.md) describes the planned application and its six development phases. It is the specification for future work, not a list of completed features.

The analysis pipeline, reviewed decisions, cache updates, and GUI have been tested with genuine CPU and NVIDIA inference, synthetic clips, and a small supplied video collection. Recognition accuracy remains limited; simulated masks do not establish performance with real masks. See the dated [implementation and validation status](docs/IMPLEMENTATION_STATUS.md) for measured results and remaining limits. English and Japanese interfaces are available.
