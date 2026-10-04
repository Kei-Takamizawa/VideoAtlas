# VideoAtlas Design Specification

Specification baseline: October 3, 2026.

This document translates the user's supplied design and preserves all 38 sections. It is the authoritative specification for future development, not a claim that these features have already been implemented or validated. Numbers marked as examples or targets are not calibrated production thresholds. Follow the development phases in Section 35.

## 1. Overview

### 1.1 Application purpose

Develop a personal Windows application that analyzes multiple video files, detects and recognizes faces, and **automatically groups videos by the people who appear in them**.

The intended subjects in the videos are, in principle, Japanese people.

The highest priority is reducing incorrect recognition and grouping, rather than processing speed. Treat merging different people into one identity as a more serious error than splitting one person into multiple identities.

When an automatic decision is ambiguous, do not force a merge. Mark it as pending review so that the user can make the final decision.

## 2. Intended environment

- **Operating system:** Primarily Windows 11.
- **Use:** Personal use only. Third-party distribution, commercial sale, and a cloud service are outside the current scope.
- **Runtime:** Python 3.x.
- **GUI:** PySide6.
- **GPU:** Prefer an NVIDIA GPU when available. Support CPU execution when a GPU is unavailable.

## 3. Core technology

### Main language

Python, because it supports replacing recognition models, using PyTorch models directly, using ONNX Runtime, integrating OpenCV, generating and modifying code with AI assistance, and evaluating accuracy and tuning thresholds.

### Components

| Responsibility | Technology or candidate |
| --- | --- |
| GUI | PySide6 |
| Video processing | FFmpeg and OpenCV |
| Face detection | SCRFD as the first choice; larger models are acceptable when they improve accuracy |
| Primary face recognition | JAPANESE FACE V1 |
| Auxiliary face recognition | AdaFace |
| Future recognition candidates | ArcFace-family models, MagFace, and similar models |
| Database | SQLite |
| Numerical computation | NumPy |
| Initial clustering candidates | HDBSCAN and agglomerative clustering |

Eventually add custom conservative clustering logic.

## 4. Overall system flow

1. Receive video files.
2. Register videos.
3. Extract frames.
4. Detect faces.
5. Detect facial landmarks.
6. Align faces.
7. Track people within each video.
8. Evaluate face quality.
9. Select high-quality faces.
10. Generate face embeddings.
11. Generate representative data for people within each video.
12. Compare people across videos.
13. Generate same-person candidates.
14. Cluster people.
15. Decide between automatic merge, pending review, and separate identities.
16. Create video groups for each person.

## 5. Video management

### 5.1 Register video folders

Allow the user to register one or more folders and optionally search subfolders recursively. Example supported extensions:

- `mp4`
- `mov`
- `avi`
- `mkv`
- `m4v`
- `webm`

### 5.2 Video identification

Identify videos using more than their file paths. Where available, store:

- File size.
- Modification time.
- File hash.
- Duration.
- Resolution.
- FPS.

Renaming a file should not cause the same video to be analyzed again.

### 5.3 Incremental analysis

Normally do not reanalyze previously analyzed videos. Allow analysis of new videos only. When models or settings change, allow optional reanalysis.

## 6. Frame extraction

Face recognition does not need to run on every frame. Sample candidate frames at an interval and combine sampling with tracking.

An example initial sampling rate is **2-5 frames per second**. Allow a higher sampling rate in intervals where faces are detected. Make the rate configurable.

Store:

- Video ID.
- Timestamp.
- Frame number.
- Face detection information.

Allow unnecessary original frame images to be deleted after analysis finishes.

## 7. Face detection

Use SCRFD to detect faces. For each face, obtain:

- Bounding box.
- Detection confidence.
- Landmarks.
- Face size.

Allow very small faces to be excluded from recognition. An example minimum face size is **80 x 80 pixels**. The threshold must be configurable in the settings screen.

## 8. Face alignment

Normalize face position before passing a face to a recognition model. Use landmarks for both eyes, the nose, and the mouth corners as alignment references.

For JAPANESE FACE V1, match FACE01 preprocessing as closely as possible.

Manage the following separately for each model:

- Input resolution.
- RGB/BGR channel order.
- Normalization method.
- Mean.
- Standard deviation.
- Alignment method.
- Embedding postprocessing.

## 9. Tracking within a video

Group faces that appear continuously within a video into a person track. This avoids treating each frame's face as an independent person.

Each track stores:

- Track ID.
- Video ID.
- Start time.
- End time.
- Face detection list.
- Embedding list.
- Quality information.

Combine bounding-box position, intersection over union (IoU), movement, and face-embedding similarity for tracking. Do not track using embeddings alone.

## 10. Face quality evaluation

Do not base identity decisions primarily on low-quality faces. Evaluate:

- Face size.
- Blur.
- Brightness.
- Contrast.
- Frontal orientation.
- Occlusion.
- Detection confidence.
- Alignment quality.

Assign each face a quality score, for example from **0.0 to 1.0**. Low-quality images may be retained as supporting information rather than deleted completely. Prefer high-quality faces when generating representative embeddings.

## 11. Face embedding generation

### 11.1 JAPANESE FACE V1

Use JAPANESE FACE V1 as the primary recognition model. Generate an embedding for each face.

### 11.2 AdaFace

Use AdaFace as an auxiliary recognition model. Its embedding space differs from JAPANESE FACE V1's, so **do not mix the embeddings themselves**. Store them independently, for example as `JapaneseFaceEmbedding` and `AdaFaceEmbedding`.

## 12. Embedding management per person

Do not store only one embedding per person track. Keep multiple high-quality embeddings.

Target: retain up to approximately **20-50 face samples per person track**.

Prefer diversity in:

- Frontal, left-facing, and right-facing poses.
- Smiling and neutral expressions.
- Bright and dark environments.

Do not retain large numbers of nearly identical frames.

## 13. Track representatives

Generate the following for each track:

- High-quality embedding list.
- Mean embedding.
- Central representative embedding candidate.
- Frontal representative embedding.
- Left-facing representative embedding.
- Right-facing representative embedding.
- Representative face image.

Use representative face images in the GUI.

## 14. Comparing people

Do not decide identity from a single image compared with another single image. Compare multiple embeddings from each person.

For example, **20 embeddings from A and 20 from B allow up to 400 pair comparisons**.

Calculate:

- Maximum similarity.
- Mean similarity.
- Median similarity.
- Mean of the top 10% of similarities.
- Mean of the top 20% of similarities.
- Lower-end similarity values.
- Match rate.

Do not decide using maximum similarity alone. Prevent false merges caused by one coincidentally similar image.

## 15. Combining multiple models

Calculate similarity separately for each model, such as Japanese Face Score and AdaFace Score, with an ArcFace Score as a possible future addition.

Do not mix embeddings across models. Combine scores for the final decision.

An initial conceptual formula is:

```text
FinalScore = JapaneseFaceScore * WeightA + AdaFaceScore * WeightB
```

The weights are not predetermined constants. Optimize them using actual video data.

## 16. Three decision levels

| Level | Meaning | Action |
| --- | --- | --- |
| HIGH | High confidence that the subjects are the same person | Eligible for automatic merging |
| MEDIUM | Possibly the same person, but confidence is insufficient | Pending human review |
| LOW | Treat as different people | Keep separate |

**Never merge MEDIUM decisions automatically.** Make conservative decisions because accuracy has priority.

## 17. Rules to prevent false merges

Tighten automatic merge conditions when:

- Few face images are available.
- Only profile views are available.
- Faces are small.
- Image quality is low.
- Models disagree.
- The similarity distribution is unstable.
- Only a few images are unusually similar.

For example, if Japanese Face scores are high and AdaFace scores are low, send the case to review instead of merging automatically.

## 18. Person clustering

Group tracks extracted from all videos into `Person` identities. For example:

```text
Person 001
  Video A / Track 1
  Video B / Track 3
  Video F / Track 2

Person 002
  Video A / Track 2
  Video C / Track 1
```

A video can contain multiple people, and a person can appear in multiple videos. The relationship between videos and people is **many-to-many**.

## 19. Conservative clustering

Do not use simple connectivity to merge identities. Similarity between A and B, and between B and C, is insufficient to automatically merge A, B, and C. Also check whether A and C are sufficiently similar.

When adding a track to a cluster, compare it with multiple tracks in that cluster, as well as the cluster representative. Prevent one incorrect decision from propagating throughout a cluster.

## 20. Pending review

Present uncertain person pairs to the user. A review screen includes:

- Representative faces for A and B.
- JAPANESE FACE Score.
- AdaFace Score.
- Overall decision.
- Videos in which the people appear.

Provide actions:

- **Same person:** Merge the person clusters.
- **Different people:** Store a negative pair to prevent future automatic merging.
- **Review later:** Defer the decision.

## 21. Learning from user corrections

Store the user's same-person and different-person decisions in the database as positive pairs and negative pairs.

Later use this information for threshold tuning, model-weight tuning, and accuracy evaluation.

## 22. Accuracy optimization

Once sufficient human-reviewed data is available, evaluate:

- JAPANESE FACE V1 alone.
- AdaFace alone.
- Multiple models combined.

Evaluate false positives, false negatives, precision, recall, ROC, and AUC.

The most important outcome is minimizing decisions that recognize different people as the same person. The supplied design does not define the denominator for its stated false-positive proportion; report it explicitly using the metric definitions in the implementation clarifications.

## 23. GUI layout

The main screen contains:

- **Left:** Person list.
- **Center:** Selected person's representative faces.
- **Right:** Videos featuring the selected person.
- **Bottom:** Analysis status and progress.

## 24. Person list

Display:

- Person ID.
- Representative face.
- Number of videos featuring the person.
- Track count.
- Registration time.

Initially use names such as `Person 0001` and `Person 0002`. Allow the user to assign a name.

## 25. Person details

Display:

- Representative faces.
- Videos featuring the person.
- Appearance times within each video.
- Similarity information.
- Track information.

Provide actions to:

- Open a video.
- Play from an appearance time.
- Merge with another person.
- Split a person.
- Exclude a face.
- Change the representative face.

## 26. Video list

Display:

- File name.
- Path.
- Duration.
- Analysis status.
- Number of detected people.
- Analysis time.

Selecting a video shows the people who appear in it.

## 27. Review queue screen

Display:

- Person A.
- Person B.
- Representative faces for both people.
- Japanese Face Score.
- AdaFace Score.
- Overall confidence.

Provide same-person, different-people, and defer actions. Support keyboard shortcuts so large review queues can be processed efficiently.

## 28. Database design

### Videos

- `id`
- `path`
- `file_hash`
- `file_size`
- `modified_at`
- `duration`
- `width`
- `height`
- `fps`
- `analyzed_at`
- `analysis_version`

### FaceDetections

- `id`
- `video_id`
- `timestamp`
- `frame_number`
- `x`
- `y`
- `width`
- `height`
- `detection_score`
- `quality_score`
- `pose`
- `track_id`

### FaceEmbeddings

- `id`
- `face_detection_id`
- `model_name`
- `model_version`
- `embedding`

### Tracks

- `id`
- `video_id`
- `start_time`
- `end_time`
- `representative_face_id`
- `person_id`

### Persons

- `id`
- `name`
- `representative_face_id`
- `created_at`
- `updated_at`

### PersonVideos

- `person_id`
- `video_id`

### PersonRelations

- `person_a`
- `person_b`
- `relation`

The relation is one of `confirmed_same`, `confirmed_different`, or `pending`.

### AnalysisSettings

Store the models and settings used for analysis.

## 29. Model version management

Always store `model_name` and `model_version` with embeddings. Do not mix old and new embeddings after a model update. Allow embeddings alone to be recalculated when necessary.

## 30. Caching

Cache recognition results instead of recognizing faces again whenever a video is opened.

Cache:

- Detections.
- Aligned face information.
- Quality scores.
- Embeddings.
- Tracks.
- Person decisions.

Allow processing to restart from the required stage when models are adjusted.

## 31. Settings screen

### Video

- Target folders.
- Target extensions.
- Frame analysis interval.

### Face detection

- Detection model.
- Detection threshold.
- Minimum face size.

### Quality

- Minimum quality score.
- Blur threshold.
- Pose threshold.

### Face recognition

- JAPANESE FACE threshold.
- AdaFace threshold.
- Model weights.

### Automatic merging

- HIGH threshold.
- MEDIUM threshold.

### Other

- GPU use.
- Cache location.
- Database location.
- Log level.

## 32. Logging

Save analysis logs, including:

- Start of video analysis.
- Face detection count.
- Track count.
- Embedding count.
- Person merge count.
- Pending-review count.
- Error details.

Make logs viewable from the UI.

## 33. Error handling

Do not stop the entire application because of:

- A corrupted video.
- An unsupported codec.
- Face detection failure.
- Model loading failure.
- Insufficient GPU memory.
- Analysis failure for a single video.

Record affected videos with the status **Analysis Failed** and allow later reanalysis.

## 34. Performance

Prioritize accuracy. Use embedding caches, video analysis caches, and incremental analysis to avoid repeating work.

When a GPU is available, run face detection and embedding generation on it.

## 35. Initial development priorities

### Phase 1: Working analysis engine

- Video loading.
- Frame extraction.
- SCRFD.
- Face alignment.
- JAPANESE FACE V1.
- Embedding persistence.
- Cosine similarity.
- SQLite.

### Phase 2: Tracking within videos

- Track generation.
- Multiple face samples.
- Quality scores.
- Representative face selection.

### Phase 3: Identity decisions across videos

- Track comparisons.
- Person creation.
- Clustering.

### Phase 4: AdaFace

- Separate embeddings for each model.
- Ensemble decisions.

### Phase 5: GUI

- Person list.
- Video list.
- Person details.
- Pending review.

### Phase 6: Accuracy improvements

- Positive pairs.
- Negative pairs.
- Threshold optimization.
- Weight optimization.
- False-merge analysis.

## 36. Essential design principles

1. Do not decide identity from one frame alone.
2. Keep multiple embeddings per person.
3. Use JAPANESE FACE V1 as the primary model.
4. Use AdaFace as an independent auxiliary decision source.
5. Do not mix embeddings from different models.
6. Prioritize preventing merges of different people.
7. Send uncertain decisions to human review.
8. Accumulate user-confirmed decisions.
9. Optimize thresholds using the user's actual video collection.
10. Make models and preprocessing replaceable.

## 37. Intended completed workflow

1. The user registers video folders.
2. The user selects **Start Analysis**.
3. Only new videos are analyzed automatically.
4. People are grouped automatically.
5. The person list shows identities such as `Person 0001`, `Person 0002`, and `Person 0003`.
6. The user selects `Person 0001`.
7. All videos featuring that person are displayed.
8. Uncertain identity matches appear in the review queue.
9. The user chooses **Same person** or **Different people**.
10. The person database becomes more accurate as reviewed decisions accumulate.

## 38. Final goal

Build **a local person index that organizes large video collections across files by the people who appear in them**.

Combine high-quality face detection, tracking within videos, face quality evaluation, multiple frames, multiple embeddings, Japanese-oriented face recognition, multiple models, conservative clustering, final human review, and threshold optimization using personal data. The goal is to minimize false merges rather than rely solely on a recognition model's standalone accuracy.

---

## Implementation clarifications

These notes identify decisions that the supplied design leaves open; they do not change the 38 sections above.

- **Existing code is not the specification.** Older setup and handoff documents describe the current implementation, including MediaPipe-based alignment and provisional cosine-distance settings. They do not establish compliance with this design or calibrated HIGH/MEDIUM thresholds.
- **Model integration remains to be verified.** Confirm the precise JAPANESE FACE V1 and AdaFace artifacts, versions, terms, and preprocessing against their primary sources when implementing them. A model name alone does not specify those details.
- **Thresholds and score definitions remain open.** Define score ranges, similarity versus distance, match-rate denominators, ensemble calibration, and minimum evidence before choosing automatic merge thresholds. Do not turn example numbers into validated guarantees.
- **Evaluation terminology must be precise.** A false-positive count is FP. The false-positive rate is `FP / (FP + TN)`; the fraction of predicted matches that are wrong is `FP / (TP + FP)`. Report the denominator and evaluation unit, such as track pairs or person merges. The source's priority is preventing different people from being merged; it supplies no numeric acceptance target or minimum reviewed sample count.
- **Persistence details remain open.** Section 28 is a logical entity and field list. Concrete types, keys, indexes, migrations, and retention of confirmed relations through person merges and splits must be specified during implementation.
