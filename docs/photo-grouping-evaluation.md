# Photo grouping evaluation (2026-11-26)

This note records how the photo grouping was measured on labelled photos and what the measurements show. It is evidence for choosing a grouping policy, not a guarantee for other photo collections.

## Data and hardware

- 8 people, one folder per person, 4,615 photos after removing two duplicate pairs that were filed under two people. 4,603 photos count in the answer key.
- Ubuntu server, RTX 4060 Ti 8 GB. Every model ran on CUDA (`--device cuda`, no CPU fallback). Analysis of all photos with five models took 4 min 40 s (16.5 photos/s); peak GPU memory was 2.3 GB.
- Calibration runs on the CPU and took 1 to 15 minutes per setting set.

## How errors are counted

- **Mixing**: faces in a group whose majority is another person, divided by all labelled faces in groups.
- **Missing**: labelled photos that are not in their person's main group (largest group of that person). This includes photos without a usable face, photos left for review, and photos in a second group of the same person.
- **Held-out (leave-one-person-out)**: for each person in turn, settings are chosen on the other seven people under a mixing limit, then all photos are grouped together and only errors that involve the held-out person are counted. A mixing event between two people is counted in both of their folds.
- **After review**: a simulated user answers the first N merge suggestions correctly, using the answer key. It assumes perfect answers.

## Model comparison (single-face pairs, 3,862 faces)

| Model | AUC | TAR at FMR 1e-3 (95% CI over people) |
| --- | --- | --- |
| glintr100 + LVFace-B (fused score) | 0.986 | 79.6% (70.0–90.4) |
| glintr100 | 0.984 | 77.3% (68.7–88.7) |
| w600k_r50 | 0.980 | 71.4% (59.9–84.0) |
| LVFace-B | 0.980 | 70.5% (57.1–84.3) |
| JAPANESE FACE V1 | 0.764 | 3.8% (1.2–12.8) |

Faces with an eye distance under 15 px are much harder for every model (AUC about 0.91 to 0.92 for the ArcFace-family models).

## Grouping results (held-out)

| Round | Setting | Mixing limit on training people | Held-out mixing | Held-out missing | Missing after 100 suggestions |
| --- | --- | --- | --- | --- | --- |
| 1 | JAPANESE FACE V1, first step only | 0.2% | 3.28% | 96% | – |
| 1 | glintr100, first step only | 0.2% | 1.29% (50 faces) | 22.4% | – |
| 2 | fused, cluster merge (clusters of 3+) | any | 0.00% (0 of 3,857) | 26.4% | 10.7% |
| 3 | fused, cluster merge (clusters of 2+) | 0.1% | 0.10% (4) | 10.6% | 7.1% |
| 4 | fused, stricter first step (merge 5–9) | 0.05% | 0.16% (6) | 5.95% | 3.8% |
| 4 | same | 0.1% | **6.13% (244)** | 3.91% | 2.1% |
| 5 | fused, merge 8–12, no size cap | 0.05% | 0.18% (7) | 2.11% | 1.7% |
| 5 | same | 0.5% | **3.36% (134)** | 1.76% | 1.7% |
| 5 | fused, merge 8–12, size cap 30 | 0.05% | 0.34% (13) | 2.02% | 1.7% |
| 5 | same | 0.5% | 0.49% (19) | 1.61% | 1.7% |
| 6 | fused, size cap 30, attach down to 0.5 | 0.05% | 0.39% (15) | 1.96% | 1.6% |

What the rounds show:

- Most missing photos in rounds 1 and 2 were in a second group of the same person (makeup, age, lighting). A strict first step followed by the normalized cluster merge removes most of that split.
- In round 4, a permissive setting merged all 119 faces of one person into the 555-face group of a similar-looking person. The training people could not reveal this risk, because the pair was split across folds. Round 5 shows the same jump without a safeguard (3.36% mixing).
- `cluster_auto_max_size` stops two large clusters from merging automatically; such pairs become merge suggestions. With a cap of 30, mixing stayed at or below 0.70% for every mixing limit tried, and no whole person was merged. At strict limits the cap did not reduce mixing (0.34% with the cap against 0.18% without it at the 0.05% limit, a difference of 6 faces).
- The remaining mixed faces are single faces or pairs, mostly between SeoAh, JooBin, SeoYeon and Hitomi. Some may be filing errors in the answer key; 26 such photos were sent to the user for checking.

## Candidate policy

Chosen on all 8 people at a mixing limit of 0.05% (round 6): fused glintr100 + LVFace-B, first-step merge 10.0 (fused units: spread of different-person scores), cluster merge 0.3 on clusters of 2 or more faces, automatic merge only when the smaller cluster has 30 faces or fewer, attach 0.75 with margin 0.05 over the top 5 faces, merge suggestions from 0.2.

On all 8 people (optimistic, because the same people chose it): 122 groups, the 8 largest match the 8 people; 230 faces left for review in 130 photos; 94 merge suggestions; 1 mixed face (0.03%); missing 2.48% (1.71% without the photos that have no usable face).

## Limits of this evidence

- Only 8 people. Confidence intervals over people are wide, and the chance of meeting a look-alike pair grows with the number of people.
- The settings grids for rounds 2 to 6 were widened after seeing held-out results. The held-out numbers are therefore somewhat optimistic. An unbiased estimate needs people who were not used in any round.
- The review simulation assumes that the user answers every suggestion correctly.
- The candidate models (glintr100, LVFace-B) have non-commercial licences. JAPANESE FACE V1 remains the specified primary model; the user decided to collect more labelled people before changing it.
