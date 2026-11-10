# Development guide

## Starting work with Claude

Open the repository root in Claude Code so that it loads [CLAUDE.md](../CLAUDE.md). Claude handles planning, implementation, self-review, and verification within the same session. Development does not require a second assistant or an alternating review loop.

Read these documents in order:

1. [Claude instructions](../CLAUDE.md): development rules and recognition invariants.
2. [Design specification](design-specification.md): authoritative target behavior and the six development phases in Section 35.
3. [Implementation and validation status](IMPLEMENTATION_STATUS.md): dated observations and remaining evaluation work, not a completion certificate for the specification.
4. [Windows setup](WINDOWS_SETUP.md): runtime setup, model provisioning, and library operations.

For each task, identify the relevant requirements and code, implement the requested scope, run appropriate checks, and review the final diff. Report what changed, checks actually performed, skipped checks and their reasons, and unresolved issues. Update the implementation status only when new observations support it; preserve the dates and limitations of historical measurements.

## Repository map

| Location | Responsibility |
| --- | --- |
| `videoatlas/ui.py`, `localization.py` | PySide6 interface and English/Japanese strings |
| `videoatlas/controller.py` | Library operations and analysis coordination |
| `videoatlas/analyzer.py`, `tracking.py` | Video sampling, face analysis, and within-video tracks |
| `videoatlas/model_adapters.py`, `recognition.py` | Model inference, preprocessing, and recognition options |
| `videoatlas/grouping.py` | Conservative identity comparisons and grouping |
| `videoatlas/storage.py` | SQLite index, caches, and durable reviewed decisions |
| `videoatlas/evaluation.py` | Reviewed-pair evaluation and candidate tuning |
| `Scripts/` | Windows setup, model provisioning, and offline evaluation tools |
| `tests/` | Automated regression and optional genuine-model checks |

## Windows setup and verification

VideoAtlas targets 64-bit Windows and Python 3.12. Run these commands from the repository root in PowerShell:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

See [Windows setup](WINDOWS_SETUP.md) for optional NVIDIA acceleration (`-Gpu`) and AdaFace provisioning (`-AdaFace`). Use a separate test library when exercising index changes:

```powershell
.\.venv\Scripts\python.exe -m videoatlas --data-dir .test-artifacts/manual-library
```

Install the test runner in the development environment, then run the suite:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.test-artifacts/pytest
git diff --check
```

The optional genuine-model integration test needs locally downloaded models and an official sample image at `videoatlas/resources/model-smoke.jpg`; it skips when they are absent. Synthetic fixtures test application behavior. They are not an accuracy benchmark. The application entry point requires Windows; checks on another OS do not establish Windows GUI, playback, or GPU behavior.

## Current implementation context

The engine uses SCRFD landmarks, model-specific FACE01/AdaFace alignment and normalization, and versioned embedding spaces. Numerical GPU/CPU profiles remain separate. Track decisions aggregate multiple quality-filtered samples, and automatic clustering requires HIGH comparisons with every cluster member. Uncalibrated automatic merging is disabled by default; upper-face automatic merging remains unavailable pending calibration.

SQLite migrations preserve the existing JSON-payload index. Reviewed track relations survive person edits and reanalysis. Reanalysis samples prior reviewed timestamps, and unresolved correspondence to reviewed history blocks automatic merging. Regrouping uses cached embeddings; embedding recalculation uses checksummed aligned crops.

The public repository excludes downloaded weights, checkpoints, face caches, test clips, and the user's private Japanese specification. Keep those artifacts local. See the dated [implementation status](IMPLEMENTATION_STATUS.md) for measured results and evaluation limits.
