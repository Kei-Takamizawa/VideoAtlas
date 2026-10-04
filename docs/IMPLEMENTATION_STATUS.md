# Implementation and validation status

Checked on October 4, 2026 UTC (October 3 locally). The [design specification](design-specification.md) remains authoritative. This page reports implementation and observed checks; it does not certify recognition accuracy.

## Development phases

| Phase | Implemented behavior | Verification and limits |
| --- | --- | --- |
| 1: Analysis engine | Video decoding and adaptive sampling, genuine SCRFD detection, landmarks, model-specific alignment, JAPANESE FACE V1 inference, versioned embeddings, SQLite persistence, cosine similarity | Genuine CPU/CUDA models exercised on official sample imagery, synthetic video, and 20 supplied clips; primary five-point alignment approximates upstream dlib geometry |
| 2: Tracks | Spatial continuity, motion/IoU plus embeddings, multiple quality-filtered samples, configurable sample cap, mean/central/pose representatives | Synthetic association, ambiguity, rejected detections, pose diversity, and duplicate-sample tests |
| 3: Identity decisions | Person creation, multiple-sample statistics, HIGH/MEDIUM/LOW decisions, conservative cluster additions, uncertain-pair queue | All-member checks, negative constraints, overlapping tracks, insufficient evidence, and nontransitive clustering tests; automatic merging initially disabled |
| 4: AdaFace | Official checkpoint provisioning and checked ONNX export; independent crops, normalization, embedding spaces and model scores | Genuine primary-plus-auxiliary CPU inference and cached recalculation exercised; auxiliary model must be enabled in Settings |
| 5: GUI | Face-only unnamed people tiles, associated videos and appearance times; left-click selection and optional names; review shortcuts S/D/L; merge/split, exclusion, representatives, settings, logs, evaluation | Offscreen GUI tests and screen captures; shared videos belong to each detected person; synthetic video decoding, appearance-time seek, play and pause checked |
| 6: Accuracy tools | Durable positive/negative relations and correction history; explicit precision/recall/FP denominators, ROC/AUC; candidate threshold/model-weight grids; cached regrouping; experimental upper-face ablation | Small, manually selected 20-video cohort evaluated with a held-out fourth-video split. Results do not establish population accuracy or cluster-level accuracy |

## Safety and incremental processing

- Model artifact hashes, preprocessing identities, and numerical profiles identify embedding spaces. Comparisons use matching full identities; embeddings from different models or versions are never averaged together.
- Automatic merging requires HIGH comparisons with every relevant cluster track. Model disagreement, insufficient quality/sample evidence, overlapping tracks, negative relations, and unresolved continuity with reviewed history block HIGH. Multiple eligible identities trigger review.
- Reanalysis includes the prior timestamps of reviewed detections. Corrections are reused only at the original timestamp with unambiguous spatial correspondence. Unmapped successor tracks require human review. Confirmed relation endpoints and correction history are retained; inactive pending endpoints are excluded from the actionable queue.
- Renaming unchanged content preserves analysis. Replacing content produces a new video ID, with the old reviewed index retained as missing. Identical copies share analysis at one available location.
- Normal refresh analyzes only pending/new videos. Analysis settings changes are applied through explicit reanalysis, cached embedding recalculation, or cached track regrouping.
- Reanalysis rebuilds unreviewed inferred tracks instead of retaining obsolete fragments. Manual assignments, excluded faces, named identities, and confirmed relation endpoints retain conservative correspondence protections.
- Filenames never enter identity inference. All detected faces are recorded, including companions; a shared video may appear under several people. The default interval is 0.5 seconds and densifies to 0.2 seconds while faces are present. Sampling cannot guarantee detection of every brief appearance.
- Old JSON-payload SQLite indexes migrate without deleting original records. A legacy JSON import is marked once so clearing the index cannot resurrect it. Unsupported future schema versions fail explicitly.
- Logs rotate at 5 MiB with three backups. Failures are recorded per video; model-load failure marks the affected videos failed and retryable. Removing target folders in Settings disables discovery; explicit index removal does not delete original videos.

## Observed verification

The current Windows environment uses Python 3.12.14, NumPy 1.26.4, OpenCV 4.11.0, PySide6 6.11.2, and ONNX Runtime GPU 1.22.0 with CPU fallback.

- Full automated suite: **114 passed in 21.21 seconds**, including the locally provisioned genuine-model integration test. Run with `.venv/Scripts/python.exe -m pytest -q --basetemp=.test-artifacts/pytest`. The optional model test skips when its local models/sample image are absent.
- Initial genuine CPU smoke: 1-second synthetic video, 2 sampled frames, 12 face detections and 10 accepted primary embeddings. Every accepted embedding has 512 dimensions and unit L2 norm.
- Genuine ensemble image smoke: 6 detections, 5 accepted faces; each accepted face produced two independent normalized 512-dimensional embeddings.
- Full controller integration: two distinct 1.5-second synthetic videos, registration, analysis, review-based merge, cached embedding recalculation, and cached regrouping; face IDs and reviewed same-person decisions retained. Changing a review to different-person, then detailed rescan with automatic merging enabled, retained the negative constraint and correction history.
- AdaFace export: Torch/ONNX CPU maximum absolute output difference **8.27 × 10⁻⁷** on the export probe. Cached recalculation maximum embedding difference **0.0** on the smoke samples.
- Qt video playback: a generated 2-second MJPEG clip decoded into valid video frames, sought to 500 ms, played, and paused through the application player.
- The real application entry point showed the main window and exited cleanly in an offscreen startup check. Python compilation, PowerShell setup syntax, dependency consistency (`pip check`), and Git whitespace checks passed.
- Genuine CUDA inference succeeded on an NVIDIA RTX 4060 Ti (8,188 MiB reported by `nvidia-smi`, driver 617.14), using ONNX Runtime GPU 1.22.0 with CUDA 12.9 and cuDNN 9.27. SCRFD, FACE01, and AdaFace retained CUDA/CPU providers after inference; Accurate FACE01 used `use_tf32=0`. Each accepted face produced two independent 512-dimensional embeddings.
- Controlled current five-point full-face comparison: one official sample image, five accepted faces, three warmups and ten trials per CPU/CUDA runtime, excluding initialization. Median detection/alignment/cache/embedding time was **429.63 ms CPU → 106.01 ms CUDA (4.05×)**. Same-crop embedding inference median was **212.18 ms → 63.82 ms (3.32×)**. Maximum normalized CPU/GPU embedding differences were **4.67 × 10⁻⁶** for FACE01 and **2.38 × 10⁻⁷** for AdaFace. Numerical profile keys remain separate. This single-image measurement does not establish a whole-video speedup, increased recognition accuracy, or that every operation ran on GPU.

Synthetic tests exercise software paths. Real-video score evaluations below cover a small selected cohort. Neither automated test counts nor inference success are an accuracy percentage.

## Supplied real-video evaluation

The supplied collection contains 20 videos, 407.02 seconds in total. Filenames provide an offline answer key only after inference. One expected-subject track per video was selected from a visually inspected gallery; companions remain in the application but are excluded from this labelled cohort. There are 190 unordered video pairs: 30 same-person and 160 different-person pairs.

A complete reanalysis with current full-face processing finished with **20 ready videos, 0 failed videos, 2,200 detections, and 1,881 accepted faces**. Every accepted face had two independent 512-dimensional embeddings. It produced 1,016 tracks and 697 active inferred groups with automatic merging off; these group counts are fragmentation, not a count of actual people or proof of successful grouping. The 417.63-second run overlapped another GPU evaluation and includes decoding, inference, SQLite, grouping, and Qt callbacks; it is not a controlled CPU/GPU speed comparison.

At a cosine threshold of 0.80, the original selected-track score evaluation gave Japanese Face V1 **0/30 true matches and 0/160 false matches**, and AdaFace **1/30 true matches and 0/160 false matches**. These are score-only results, separate from the complete multi-sample HIGH policy. They show severe missed matches at the provisional thresholds, rather than calibrated recognition.

A threshold candidate was selected using only pairs among the first three clips per subject: 105 training pairs, including 15 positive and 90 negative pairs. All pairs involving a fourth clip were held out: 85 pairs, including 15 positive and 70 negative pairs. Training selected AdaFace alone at 0.44. On held-out pairs it produced TP=11, FP=1, TN=69, FN=4: precision **91.67%**, recall **73.33%**, false-positive rate **1/70 = 1.43%**, and incorrect-match fraction **1/12 = 8.33%**. The candidate is **not applied**: it produced a false match and does not supply primary-model agreement. Automatic merging remains disabled.

## Experimental upper-face mode

Settings provide `full` and `upper` recognition regions. Upper mode aligns exclusively from the two eyes, removes all pixels below the eye line plus a small lower-eyelid allowance, removes the nasal-root strip below the eye line, and neutralizes the removed area before interpolation and inference. Nose/mouth coordinates and jaw pixels do not determine its embedding input. Quality calculations use visible pixels and exclude the artificial mask edge. Unknown yaw remains unknown instead of being recorded as a frontal measurement.

Upper mode also runs the detector at the configured input size and at half/one-and-a-half sizes, retaining the same confidence threshold and deduplicating faces across sizes. For the pinned 640 input this is 320/640/960. The source model remains unchanged; a checksummed cache copy makes only output proposal-count dimensions symbolic. Its source/derived hashes and transformation version are recorded. Nodes and weights remain identical. Fixed-input adapters retain their supported size.

Full and upper preprocessing identities and lossless caches remain separate for each model. Switching regions requires video reanalysis. Regression tests show that changing hidden pixels or lower-face landmarks leaves the upper input unchanged and that a full-face cache is rejected for upper recalculation. These contracts do not prove real-mask recognition accuracy.

In the first single-size opaque-mask ablation, **56/56 clean targets** and **21/56 masked targets (37.50%)** were detected. Upper-clean processing accepted 52/56 samples; upper-masked processing accepted 21/21 detected samples, which is still only 21/56 end-to-end. Only 2 of 20 masked tracks supplied at least three accepted embeddings. The masked score cohort therefore contained 66 complete pairs rather than all 190; held-out eligible evidence was 30 pairs (5 positive, 25 negative) rather than all 85.

For a fixed 50:50 primary/AdaFace score ensemble, thresholds were selected only on training pairs with zero observed training false matches. Held-out full-clean scores gave TP=6/15, FP=0/70 at 0.39; upper-clean gave TP=4/11, FP=0/51 at 0.71; upper-masked gave TP=3/5, FP=0/25 at 0.69. The last conditional recall is 60%, but only **3/15 = 20%** of all intended held-out same-person pairs received a correct positive result when missing evidence is counted. This is score-only evidence; many pairs do not meet the application's multi-sample HIGH requirements.

Reusing the provisional full-face primary threshold of 0.80 on upper-face inputs was unsafe: the upper-clean primary produced **51/129 false positive pairs**, and the upper-masked primary produced **24/56 false positive pairs**. Configuration therefore rejects upper-face automatic merging pending calibration; human review remains available. The context-added candidate also worsened held-out false matches (masked 1/25 versus embedding-only 0/25), so hair/texture/scale cues are not added to production merge scores.

The actual three-size production detector plus deduplication was then re-evaluated on the same cohort. Masked detection and quality acceptance improved to **36/56 (64.29%)**, with **8/20** tracks providing at least three accepted embeddings. This improvement is detector coverage, not a demonstrated identification improvement. There were 120 complete score pairs (70 excluded for missing evidence), including 54 held-out pairs: 9 positive and 45 negative. At the independently training-selected 50:50 ensemble cutoff of 0.70, held-out results were **TP=2, FP=0, TN=45, FN=7**. Conditional recall was **2/9 = 22.22%**; across all 15 intended held-out positive pairs it was **2/15 = 13.33%**. Another six positive pairs lacked evidence. The primary at the old fixed 0.80 threshold still produced **35/101 false positive pairs** across all complete masked pairs. Upper automatic merging remains unavailable, and no candidate cutoff is applied to production.

The offline ablation separately measures detection after applying opaque digital masks, accepted samples, pair scores, held-out threshold performance, visible upper-area appearance, and eye-gap shooting scale. Hair/ear-area colour, coarse texture, and apparent size are exploratory context cues; they are not stable identity embeddings or automatic-merge authority. Real masks can have different shapes, texture, lighting effects, and detector behavior; [NIST's digital-mask study](https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8311.pdf) also identifies these simulation limits.

## Model provenance

| Artifact | Pinned source | Observed size / SHA-256 |
| --- | --- | --- |
| JAPANESE FACE V1 ONNX | FACE01 revision `afec7ebac709f14224353e7f8b6539711899b1ff` | 26,083,027 bytes; `e7ca51f4bc85f73ddb830683ac6a09077909fa45a52b2bff41a9c6e8ff267e2f` |
| SCRFD `det_10g.onnx` | Official InsightFace `v0.7` buffalo_l release, locally named `scrfd_10g_kps.onnx` | 16,923,827 bytes; `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| AdaFace R50 MS1MV2 checkpoint | Official Google Drive link in AdaFace README, source revision `c60eaa786a42c03444f3df7096dbaf9d57ae010d` | 700,286,703 bytes; `234da6ce931f821e0a8e920063dc2091d8122ea46285c159d2f25d3f762fecb3` |
| Local AdaFace ONNX export | Verified official source/checkpoint, opset 17 | 174,398,318 bytes; `0e2f906e406f75e570ce043f8c2001351bd2a5bd889068785e34c17e52eb33b1` |

Exports from another supported toolchain may have a different file hash; the exporter records its own provenance and checks numerical parity. Downloaded artifacts, their local manifests, sample imagery, face caches, and test videos are excluded from Git.

Primary references: [FACE01 source](https://github.com/yKesamaru/FACE01_DEV/tree/afec7ebac709f14224353e7f8b6539711899b1ff), [SCRFD source](https://github.com/deepinsight/insightface/tree/master/detection/scrfd), [AdaFace source](https://github.com/mk-minchul/AdaFace/tree/c60eaa786a42c03444f3df7096dbaf9d57ae010d), [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Remaining evaluation

- General recognition accuracy, cluster-level false-merge frequency, and suitable production thresholds/model weights remain unestablished. The small supplied cohort is an initial measurement; it is not an independent population test. Real masked clips and more diverse labelled track pairs are needed.
- Tuning reports evaluate the labelled training cohort and are candidate-only. Score-threshold metrics are separate from the complete HIGH eligibility policy and cluster-level merge outcomes. An undefined denominator is reported as null, not zero.
- SCRFD eye-center alignment approximates FACE01's original dlib chip geometry. Pose angles and occlusion quality are geometric proxies, not calibrated 3D pose or occlusion classifiers.
- TensorRT FP16 execution, broad codec compatibility, and large-collection performance have not been benchmarked. No accuracy or speed target is inferred from the specification's examples.
