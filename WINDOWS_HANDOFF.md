# Windows development notes

Follow the authoritative [design specification](docs/design-specification.md). The dated [implementation status](docs/IMPLEMENTATION_STATUS.md) separates implemented features, verified checks, and remaining evaluation work.

VideoAtlas targets 64-bit Windows and Python 3.12. Follow [Windows setup](docs/WINDOWS_SETUP.md) for CPU, optional NVIDIA acceleration, and optional AdaFace provisioning. The public repository excludes downloaded weights, checkpoints, face caches, test clips, and the user's private Japanese specification.

The current engine uses SCRFD landmarks, model-specific FACE01/AdaFace alignment and normalization, and versioned embedding spaces. Numerical GPU/CPU profiles remain separate. Track decisions aggregate multiple quality-filtered samples, and automatic clustering requires HIGH comparisons with every cluster member. Uncalibrated automatic merging is disabled by default.

SQLite migrations preserve the existing JSON-payload index. Reviewed track relations survive person edits and reanalysis. Reanalysis samples prior reviewed timestamps, and unresolved correspondence to reviewed history blocks automatic merging. Regrouping uses cached embeddings; embedding recalculation uses checksummed aligned crops.

Useful verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.test-artifacts/pytest
```

The optional genuine-model integration test needs locally downloaded models and an official sample image at `videoatlas/resources/model-smoke.jpg`; it skips when they are absent. Synthetic fixtures test application behavior. They are not an accuracy benchmark.
