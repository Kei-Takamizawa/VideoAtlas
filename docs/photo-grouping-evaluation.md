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
- In round 4, a permissive setting merged all 119 faces of one person into the 555-face group of a similar-looking person. The training people could not reveal this risk: when one of the two look-alike people is held out, the other one has no look-alike among the training people. Round 5 shows the same jump without a safeguard (3.36% mixing).
- `cluster_auto_max_size` stops two large clusters from merging automatically; such pairs become merge suggestions. With a cap of 30, mixing stayed at or below 0.70% for every mixing limit tried, and no whole person was merged. At strict limits the cap did not reduce mixing (0.34% with the cap against 0.18% without it at the 0.05% limit, a difference of 6 faces).
- The remaining mixed faces are single faces or pairs, mostly between SeoAh, JooBin, SeoYeon and Hitomi. Some may be filing errors in the answer key; 26 such photos were sent to the user for checking.

## Candidate policy

Chosen on all 8 people at a mixing limit of 0.05% (round 6): fused glintr100 + LVFace-B, first-step merge 10.0 (fused units: spread of different-person scores), cluster merge 0.3 on clusters of 2 or more faces, automatic merge only when the smaller cluster has 30 faces or fewer, attach 0.75 with margin 0.05 over the top 5 faces, merge suggestions from 0.2.

On all 8 people (optimistic, because the same people chose it): 122 groups, the 8 largest match the 8 people; 230 faces left for review in 130 photos; 94 merge suggestions; 1 mixed face (0.03%); missing 2.48% (1.71% without the photos that have no usable face).

## Test on 20 new people (2026-11-27)

The user added 20 labelled people (15,415 files). None of them was used in rounds 1 to 6. The candidate policy above was applied without any change to all 28 people together (20,028 photos read, 24,398 usable faces, 2,788 photos with two or more usable faces). Only errors that involve the 20 new people are counted.

- Before any further tuning, the 20 new people were split at random (SHA-256 of a fixed seed and the name) into 10 development people and 10 test people. The test people are kept out of every choice that follows.
- Analysis on CUDA with six models (four, plus mirrored-face averaging for two of them): 31 min 12 s (10.7 photos/s), peak GPU memory 5,777 MiB for the whole device. Grouping 24,398 faces: 20 s, 5.9 GiB of RAM.

| People | Labelled faces in groups | Mixed faces | Missing photos | Missing after 200 suggestions |
| --- | --- | --- | --- | --- |
| 20 new people | 13,159 | 9 (0.07%) | 6.48% of 15,024 | 4.57% |
| 10 test people | 4,987 | 3 (0.06%) | 6.95% of 5,684 | 4.52% |
| All 28 people | 17,002 | 10 (0.06%) | 5.54% of 19,533 | 3.97% |

- Mixing: 8 groups hold faces of two people, each with one or two faces of the other person. No person was merged as a whole.
- Missing: 832 of the 974 missing photos of the new people are in a second large group of the same person (for example 481 + 142, 635 + 321, 525 + 256 faces). Such pairs are larger than the automatic-merge cap of 30 faces, so they can only become merge suggestions. The review simulation shows that these suggestions are ranked between 100 and 200, behind many suggestions between small groups.
- The 26 hard photos the user confirmed as correctly filed (look-alike people): 23 are in their own person's main group, 2 are left for review with their own person as the first candidate, and 1 is in another person's group.

Model comparison on the 18 people outside the test set (12,039 single faces, paired person bootstrap, difference in TAR at FMR 1e-3 against glintr100):

| Model | AUC | TAR at FMR 1e-3 | Difference (95% CI) |
| --- | --- | --- | --- |
| glintr100 | 0.978 | 72.9% | reference |
| glintr100 + LVFace-B (fused) | 0.982 | 74.4% | +1.5 (-1.1 to +3.6) |
| both with mirrored-face averaging, fused | 0.982 | 76.7% | +3.8 (+1.7 to +5.5) |
| w600k_r50 + both mirrored, fused | 0.982 | 76.8% | +4.0 (+2.1 to +5.3) |
| both after WCCN, fused | 0.988 | 73.4% | +0.6 (-3.8 to +13.7) |
| JAPANESE FACE V1 (all 28 people) | 0.760 | 4.7% | – |

WCCN raised the AUC but not the true-accept rate at a low false-match rate, which is the range that matters for grouping, so it is not used.

## Limits of this evidence

- 28 people now, 10 of them kept for testing. Confidence intervals over people are still wide, and the chance of meeting a look-alike pair grows with the number of people.
- The settings grids for rounds 2 to 6 were widened after seeing held-out results. The held-out numbers are therefore somewhat optimistic. An unbiased estimate needs people who were not used in any round.
- The review simulation assumes that the user answers every suggestion correctly.
- The candidate models (glintr100, LVFace-B) have non-commercial licences. JAPANESE FACE V1 remains the specified primary model; the user decided to collect more labelled people before changing it.
