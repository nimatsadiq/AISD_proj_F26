---
title: "Response to Phase 1 Repository Feedback"
subtitle: "A Monthly Early Warning System for Food Price Spikes in Nigerian States"
author: "Nimat Ohunene Sadiq | Carnegie Mellon University Africa | nsadiq@andrew.cmu.edu"
date: "04-652 Artificial Intelligence System Design, Fall 2026"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
---

# 0. The report was not reachable

The review opens by noting the Phase 1 report could not be located. That is
correct: `phase1_strategy/Phase1_Strategy_Report.pdf` was absent from the
repository at the time of review. It is now committed at that path and linked
from the README.

This matters for reading the rest of the feedback, because three of the four
recommendations describe decisions the missing document already specifies. They
are listed below with the section that contains them, not to dispute the review
but to record what was already committed before it.

# 1. Recommendations already specified in the Phase 1 report

## 1.1 Group-aware or temporal partitioning

Already committed, Phase 1 sections 4.1 and 4.2, and implemented in
`src/pipeline_builder.temporal_split()`.

The dataset does contain repeated observations of shared entities: each
`(state, commodity)` pair is one series observed monthly for up to 127 months.
Phase 1 states that random splitting is inadmissible for exactly this reason,
and uses a forward chronological split of 2016-01 to 2023-06 for training,
2023-07 to 2024-12 for validation, and 2025-01 onward for test.

One point worth making explicit, which Phase 1 implied rather than stated: a
chronological boundary is simultaneously group-respecting. No `(state,
commodity)` series can appear on both sides of a date cut, so `GroupKFold` is
subsumed rather than omitted. Within the training period, validation uses
`TimeSeriesSplit` with an expanding window.

Stratification is deliberately not applied, since it requires reshuffling and
would destroy the ordering the design exists to respect. Class balance per
split is reported instead, and the measured imbalance is in section 3 below.

## 1.2 Multicollinearity screening

Already committed, Phase 1 section 2.5, as a four-stage protocol: flag pairs at
`|rho| >= 0.85` using Spearman given the skew, compute VIF on survivors and drop
while any exceeds 10, resolve each flagged pair in favour of the feature with
higher mutual information against the target, and apply the reduced set to the
logistic model only while retaining the full set for the trees.

The tie-break hierarchy now reads: higher mutual information first, then lower
missingness ratio, then the shorter window. The missingness criterion was
adopted from this review.

**Measured result.** At `|rho| >= 0.85`, exactly one pair flagged across the
feature set: `mom_3m` with `dev_from_6m` at rho = 0.876 (Figure 3.1). Phase 2
records this as a deviation, because Phase 1 predicted severe collinearity "by
construction" and the data did not support that. Lagged *ratios* at different
horizons prove less redundant than overlapping level windows would be.

## 1.3 Structural and non-random missingness

Already committed, Phase 1 section 2.1, and implemented in
`src.transformers.MissingnessIndicator`.

Phase 1 classifies three distinct mechanisms: panel gap months as **MNAR**,
because markets suspend reporting during insecurity, which is also when prices
spike, so missingness depends on the unobserved value itself; pre-2014 sparsity
as **MCAR**, reflecting the historical extent of WFP's collection programme; and
commodity coverage differences as **MAR**.

The MNAR classification drives three consequences: binary `was_missing` flags
and a `months_since_last_observation` counter enter the feature set;
interpolation is limited to single-month gaps so that long silences are never
bridged; and the interpolation itself is flagged via `price_was_interpolated`.

A finding worth recording: the raw file contains **zero** null cells, because
WFP publishes only observed records. Missingness exists as absent *rows* and
becomes visible only after reshaping onto a continuous monthly grid, where
12.6% of cells are gaps.

Phase 2 records a deviation here. Two of the four planned indicators, `is_gap`
and `months_since_obs`, are structurally constant within the modelling set: a
gap month has no price, therefore no trailing mean, therefore no label, so it is
never a modelled row. The MNAR signal survives through `gaps_last_6m` and
`price_was_interpolated`, which describe gaps in a row's recent history rather
than in the row itself.

# 2. The recommendation that identified a real gap

## 2.1 Probability calibration under weighted loss

**This was not previously addressed and has now been implemented**
(`src/calibration.py`, Figure 6.1).

The criticism is correct. Both models correct the imbalance by reweighting the
loss, `class_weight='balanced'` for logistic regression and `scale_pos_weight`
for the trees. That deliberately shifts the decision boundary toward the
minority class, which is the intended effect on ranking and an unintended effect
on probabilities. Phase 1 listed Brier score and a reliability curve under
metrics, so the distortion would have been *measured*, but no step was specified
to *correct* it.

Why it matters operationally here: the deliverable is a dashboard showing a
ranked warning list with probabilities to a state official. An uncalibrated 0.70
does not mean seventy percent of such months spike, so it is not a number anyone
should act on.

**Implementation.** Platt scaling via `CalibratedClassifierCV`, fitted strictly
on the held-out validation split and never on training data, applied after model
fitting and before threshold selection.

**Measured effect on the test split:**

| Model | Brier (raw) | Brier (calibrated) | Mean score raw | Mean score calibrated |
|---|---|---|---|---|
| Logistic regression | 0.1890 | **0.1287** | 0.414 | 0.300 |
| Gradient boosted trees | 0.1774 | **0.1280** | 0.324 | 0.244 |

Test base rate is 0.127. Brier improves by 32% for the logistic model, and the
mean predicted score falls from 0.414 toward the true base rate.

**What calibration does not change.** Platt and isotonic scaling are monotonic,
so the ranking is identical. Measured on the test split, precision@50 is 0.480
before and after for the logistic model, and PR-AUC is 0.353 before and after.
This is the expected result and worth stating: calibration changes what a
displayed probability means, not which months are ranked highest. The operational
ranking was never wrong; the numbers attached to it were.

**An honest limitation.** Figure 6.1 shows the calibrated curve moving toward
the diagonal without reaching it. The cause is the structural break: the
calibration split carries a 34.9% positive rate against 12.7% in test, because
it spans the post-float inflation period. A calibration map learned at one base
rate transfers imperfectly to another. This is a further argument for
rolling-window retraining, with the calibrator refitted alongside the model
rather than once.

# 3. Measured results for context

| Split | Period | n | Positive rate |
|---|---|---|---|
| Train | 2016-03 to 2023-06 | 4,956 | 16.3% |
| Validation | 2023-07 to 2024-12 | 1,075 | 34.9% |
| Test | 2025-01 to 2026-06 | 909 | 12.7% |

| Model | Threshold | Recall | Precision | PR-AUC | Precision@50 |
|---|---|---|---|---|---|
| Logistic regression | 0.37 | 0.835 | 0.194 | 0.353 | 0.480 |
| Gradient boosted trees | 0.39 | 0.504 | 0.195 | 0.301 | 0.500 |
| Seasonal persistence baseline | n/a | 0.043 | 0.714 | n/a | n/a |

A leakage negative control is reported in the Phase 2 notebook: a feature
constructed from month *t+1* was injected, and test ROC-AUC rose to 1.000
against 0.737 for the real pipeline, confirming the harness detects leakage
when present.

# 4. Adopted for Phase 3

From the review's model and explainability recommendations:

1. **SHAP attribution.** Listed in `requirements.txt` from Phase 1 but not yet
   used. Summary and waterfall plots on test predictions are a Phase 3
   deliverable.
2. **CatBoost and LightGBM** alongside the existing XGBoost, with CatBoost's
   native categorical handling tested against the current one-hot and
   out-of-fold target encoding.
3. **Random Forest** as a variance-robust non-linear benchmark.
4. **ElasticNet** penalty on the logistic baseline, replacing the current L2.
5. **Macro F1** added to the reported metric set alongside PR-AUC and recall.
6. **Calibrator refitted with each rolling window**, per the limitation in
   section 2.1.
