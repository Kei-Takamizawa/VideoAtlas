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
| `videoatlas/embedders.py`, `photos.py` | Replaceable recognition models and still-photo analysis runs |
| `videoatlas/photo_grouping.py`, `photo_evaluation.py`, `photo_cli.py` | Photo grouping policy, accuracy measurement, and `photos` commands |
| `videoatlas/web/` | Browser GUI: loopback-only FastAPI server, access key and sessions, job runner, templates, vendored htmx |
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

## Linux server (headless processing)

The analysis engine also runs on 64-bit Linux without a display. The desktop GUI entry point (`python -m videoatlas`) remains Windows-only. Use Python 3.12, because the pinned NumPy release does not support newer versions. With [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-linux-gpu.txt pytest
.venv/bin/python Scripts/download_face01.py
```

Install the NVIDIA driver on the host; CUDA and cuDNN come from the pip packages. The Qt-based tests also need these Ubuntu packages: `libgl1 libegl1 libopengl0 libfontconfig1 libxkbcommon0 libdbus-1-3 libpulse0 libxrandr2`. Run the suite offscreen:

```bash
mkdir -p .test-artifacts
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q --basetemp=.test-artifacts/pytest
```

Headless commands:

```bash
.venv/bin/python -m videoatlas.cli doctor
.venv/bin/python -m videoatlas.cli images PHOTO_FOLDER --output RUN_FOLDER
```

`doctor` loads the models and reports the ONNX Runtime providers that each model registered. `images` detects, aligns, and embeds every face in a photo folder, then writes `faces.jsonl`, `embeddings.npz`, and `report.json` to a new output folder, which must be outside the photo folder. When the photo folder has one subfolder per person, the folder names serve only as an offline answer key for a single-image pair evaluation after inference.

GPU use is explicit. `--device cuda` is the default and stops with an error when any model cannot run on the GPU. `--device auto` permits a CPU fallback and logs a warning. `--device cpu` uses the CPU only. The same rule applies to the `acceleration` setting everywhere: `cuda` and `tensorrt` now fail instead of falling back to the CPU, and `auto` logs any fallback. TensorRT loading remains Windows-only. The check covers which providers a session registered; ONNX Runtime may still place individual unsupported operators on the CPU inside a CUDA session.

### Photo grouping (experimental)

The `photos` commands group still photos by person and measure the result against a folder-per-person answer key. They are separate from the video pipeline. Measured results and the candidate policy are in [the photo grouping evaluation](photo-grouping-evaluation.md).

```bash
.venv/bin/python Scripts/provision_photo_models.py --models w600k-r50,glintr100,lvface-b
.venv/bin/python -m videoatlas.cli photos analyze PHOTO_FOLDER --output RUN --models face01,glintr100,lvface-b
.venv/bin/python -m videoatlas.cli photos verify RUN --output RUN/verify.json --fuse glintr100,lvface-b
.venv/bin/python -m videoatlas.cli photos calibrate RUN --models glintr100 --cluster-grid none,0.6,0.7 --suggest-threshold 0.3 \
  --output RUN/calibration.json --policy-output RUN/policy.json
.venv/bin/python -m videoatlas.cli photos group RUN --policy RUN/policy.json --output RUN/groups.json
# or use the chosen policy (needs face01,glintr100+flip,lvface-b+flip in analyze):
.venv/bin/python -m videoatlas.cli photos group RUN --policy videoatlas/photo_policies/setting-a.json --output RUN/groups.json
```

- `analyze` detects every face without a size cut-off, records its eye distance and pose, and embeds each usable face with every listed model. It writes `faces.jsonl`, `embeddings/<model>.npz`, and `report.json`. Each model keeps its own embedding key (file hash, preprocessing, CPU/GPU profile).
- `verify` compares models on single-face pairs: AUC, equal error rate, and true-match rate at fixed false-match rates, overall and by face size, with a paired bootstrap over people. `--fuse a,b` also measures the fused score of several models.
- `calibrate` chooses the grouping thresholds. For each person in turn, it chooses thresholds on the other people only, then counts mixing and missing errors that involve the held-out person. It does this for every mixing limit in `--targets`, so the trade-off between mixing and missing is visible, and simulates a user who answers the first N merge suggestions. It writes the policy chosen on all people for `--policy-target`.
- `group` applies a policy file and writes the groups file that the browser GUI reads. `identify` measures the alternative of matching faces to a few named example photos per person.
- `Scripts/evaluate_photo_grouping.sh` runs analyze, verify, and calibrate for several models on the GPU and writes a short `summary.txt`.

The grouping policy is constrained average-linkage clustering. Faces in the same photo never share a group. An optional second step merges two clusters of at least three faces when their mean cross similarity, divided by the geometric mean of their own mean internal similarities, passes a threshold; this joins one person's looks that the first step kept apart. With `cluster_auto_max_size`, two clusters that are both larger than that size never merge automatically; they can only become a merge suggestion, so one wrong decision cannot join two whole people. A lone face joins a group only when the mean of its top-k similarities to the group passes a threshold and leads the next group by a margin; otherwise it is listed for review with suggested groups. Group pairs that may be one person are listed as merge suggestions; only the user's confirmation merges them. Faces that join no group can then form new groups among themselves (`leftover_merge_threshold`, `leftover_min_size`), so a new person with several photos gets a group of their own without joining anyone else's. `photos strangers` measures what happens to people who have no group yet. All thresholds are uncalibrated until `calibrate` has run on representative labelled photos. The candidate recognition models have non-commercial licences.

## Browser GUI

The browser GUI lives in `videoatlas/web/` and runs on the Linux server. See the [browser GUI guide](WEB_GUI.md) for starting it, the SSH tunnel, and its security model. It needs only the hash-pinned packages in `requirements-web.txt`, so its tests run without models, GPU, or Qt:

```bash
uv pip install --python .venv/bin/python --require-hashes -r requirements-web.txt
uv pip install --python .venv/bin/python pytest httpx
.venv/bin/python -m pytest -q tests/test_web_*.py
```

The web package never imports the analysis engine. It starts work as `python -m videoatlas.cli ...` subprocesses, one at a time, through the job runner in `videoatlas/web/jobs.py`. Add a new kind of job to `JOB_KINDS` with a fixed argument list; never build a shell command from user input. When you add or update a web dependency, pin a release that is at least 14 days old and regenerate `requirements-web.txt` with hashes from `requirements-web.in`.

## Current implementation context

The engine uses SCRFD landmarks, model-specific FACE01/AdaFace alignment and normalization, and versioned embedding spaces. Numerical GPU/CPU profiles remain separate. Track decisions aggregate multiple quality-filtered samples, and automatic clustering requires HIGH comparisons with every cluster member. Uncalibrated automatic merging is disabled by default; upper-face automatic merging remains unavailable pending calibration.

SQLite migrations preserve the existing JSON-payload index. Reviewed track relations survive person edits and reanalysis. Reanalysis samples prior reviewed timestamps, and unresolved correspondence to reviewed history blocks automatic merging. Regrouping uses cached embeddings; embedding recalculation uses checksummed aligned crops.

The public repository excludes downloaded weights, checkpoints, face caches, test clips, and the user's private Japanese specification. Keep those artifacts local. See the dated [implementation status](IMPLEMENTATION_STATUS.md) for measured results and evaluation limits.
