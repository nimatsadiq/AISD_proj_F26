---
title: "Phase 2: Strategy versus Reality Audit"
subtitle: "A Monthly Early Warning System for Food Price Spikes in Nigerian States"
author: "Nimat Ohunene Sadiq | Carnegie Mellon University Africa | nsadiq@andrew.cmu.edu"
date: "04-652 Artificial Intelligence System Design, Fall 2026"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
---

# 1. What was built

The Phase 1 strategy was implemented as a modular scikit-learn pipeline with
all transformations isolated to the training split, eight diagnostic figures
generated at 300 DPI, and a deliberate leakage control to verify the harness.

Everything in this report comes from `pipeline_execution.ipynb`, which runs end
to end from the raw file with no manual steps.

| Stage | Shape |
|---|---|
| Raw WFP file, as distributed | 88,545 x 16 |
| Modelling panel (state x commodity x month) | 9,000 x 32 |
| Labelled rows (three prior months required) | 6,940 |
| Train / Validation / Test design matrices | 4,956 / 1,075 / 909 x **17** |

Feature count fell from 32 panel columns to 17 model inputs with no information
loss: the removed columns are identifiers (`state`, `commodity`, `month`),
intermediate quantities used to build other features (`price_lag1`,
`roll_mean_6`, `trail_mean`), label-construction columns that must never be
seen by the model (`price_next`), and two features that proved constant
(below). No NaNs remain after the pipeline.

# 2. Decision log

| # | Phase 1 plan | Phase 2 reality | Status |
|---|---|---|---|
| 1 | Unit = (state, commodity, month) | Implemented. 107 series, Borno / Yobe / Adamawa | As planned |
| 2 | Median across markets, justified by right-skew | Implemented in `PanelBuilder` | As planned |
| 3 | Label: price in *t+1* exceeds trailing 3-month mean by > 15% | Implemented. 1,300 positives, 18.7% | As planned |
| 4 | Imbalance 4.5 : 1, class weighting not SMOTE | Measured at 4.3 : 1. `class_weight='balanced'` and `scale_pos_weight` | Minor revision |
| 5 | Missingness is MNAR, encoded not imputed | Confirmed, but two of four indicators are unusable | **Deviation 1** |
| 6 | Interpolate 1-month gaps only | Implemented. 356 rows interpolated and flagged | As planned |
| 7 | IQR flags outliers; prices never capped | Implemented. Only ratio features capped, at train-fitted 1st/99th percentiles | As planned |
| 8 | Collinearity screen at \|r\| >= 0.85, VIF > 10 | Screen run. Exactly **one** pair flagged | **Deviation 2** |
| 9 | Target encoding for commodity, fitted out-of-fold | Built but **not used** in the final models | **Deviation 3** |
| 10 | StandardScaler for logistic, none for trees | Implemented as two pipelines sharing one feature stage | As planned |
| 11 | Cyclical month encoding | Implemented (`month_sin`, `month_cos`) | As planned |
| 12 | Forward chronological 70/15/15, no stratification | Implemented | As planned |
| 13 | Rebuild lag features inside every fold | Shown to be unnecessary for backward-only windows | **Deviation 4** |
| 14 | PCA as a diagnostic, not a transformation | Figure 5.1 produced; PCA not adopted | As planned |
| 15 | Recall >= 0.80, precision >= 0.60 | Recall target met; precision target **not** met | **Deviation 5** |

# 3. Deviations and rationale

## Deviation 1: two missingness indicators are structurally constant

Phase 1 planned four MNAR indicators. `is_gap` and `months_since_obs` turn out
to be constant inside the modelling set, and the reason is structural: a gap
month has no price, therefore no trailing mean, therefore no label, so it is
never a modelled row. Every labelled row has `is_gap = 0` by construction.

Both were dropped. The MNAR signal survives through `gaps_last_6m` (four
distinct values in training) and `price_was_interpolated` (63 training rows),
which describe gaps in a row's recent *history* rather than in the row itself.

The underlying classification was not wrong. What was wrong was assuming that a
correct mechanism diagnosis automatically yields usable features.

## Deviation 2: collinearity was far milder than predicted

Phase 1 stated that severe collinearity was "expected by construction" and
designed a four-stage protocol around it. At \|rho\| >= 0.85 exactly one pair
flagged: `mom_3m` with `dev_from_6m` at rho = 0.876 (Figure 3.1), which is
intuitive, since both measure medium-horizon movement.

Lag features are less redundant than anticipated because they are *ratios* at
different horizons rather than overlapping level windows. The protocol was
applied as written and simply had little to do; `mom_3m` was retained on higher
mutual information. The full feature set was kept for the trees, per plan.

Writing a mitigation for a problem that did not materialise is not wasted
effort, but the Phase 1 report overstated the certainty.

## Deviation 3: target encoding built but not used

`OutOfFoldTargetEncoder` was implemented with smoothing and training-fold
fitting as specified. It was then left out of the final pipelines.

The reason is scope. The hybrid scope reduced the commodity count to 37, and
within a single pooled model the `food_type` indicator plus state one-hots
already carry most of the between-commodity variation. Adding a target-encoded
column introduces the pipeline's single largest leakage risk for a marginal
gain. The class is retained in `transformers.py` for the national extension,
where commodity cardinality rises and the trade-off changes.

## Deviation 4: fold-level feature rebuilding is unnecessary here

Phase 1 committed to rebuilding lag features inside every cross-validation
fold, and called this the subtlest available leak. On implementation the
reasoning proved over-cautious.

Every window looks strictly backwards from month *t*, and the label is built
from month *t+1*. A training row at time *t* therefore touches only months at
or before *t*, all of which precede the split boundary. Nothing from the test
period can reach a training feature. The concern is real for *random* splits or
*centred* windows, and neither is used.

The requirement does bind for anything that **learns a parameter**: capping
bounds, imputation values, scaler statistics, target encodings and PCA. All of
those sit inside the `Pipeline` and are fitted on the training split alone.
Conflating the two cases produces either leakage or needless complexity.

## Deviation 5: the recall target is met, the precision target is not

| Model | Threshold | Recall | Precision | PR-AUC | ROC-AUC | Precision@50 | Brier |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.37 | **0.835** | 0.194 | 0.353 | 0.737 | 0.480 | 0.189 |
| Gradient boosted trees | 0.39 | 0.504 | 0.195 | 0.301 | 0.635 | 0.500 | 0.177 |
| Seasonal persistence baseline | n/a | 0.043 | 0.714 | n/a | n/a | n/a | n/a |

Test base rate: 0.127. Thresholds were chosen on the **validation** split as
the point meeting recall 0.80 at the highest available precision; the test
split was scored once at that threshold.

The recall target of 0.80 is met by the logistic model at 0.835. The precision
target of 0.60 is not met, and this is an arithmetic consequence rather than a
tuning failure: at a 12.7% base rate, catching 84% of spikes means issuing many
warnings that do not land. The two targets were not mutually achievable at this
base rate, and Phase 1 set them without checking that.

**Precision@50 is the more honest operational number.** An agency acts on a
ranked list under a fixed budget, not on every alert above a threshold. At
0.48 and 0.50, roughly half of the top 50 warnings are real against a 12.7%
base rate, a lift of around 3.8x.

**On the baseline.** Seasonal persistence is precise (0.714) but fires almost
never, catching 4.3% of spikes. A rule that is right when it speaks and almost
never speaks is useless as a warning system, which is exactly why recall is the
primary metric. Both models beat it on recall by an order of magnitude. The
revised success criterion for Phase 3 should be recall at matched precision,
or precision@k at a fixed k, not two independent thresholds.

**Logistic beats the trees**, which is unusual and worth stating. With 17
features, one of them seasonal, and a strong regime shift between periods, the
simpler model generalises better across the structural break. This supports the
Phase 1 decision to keep an interpretable baseline rather than treating it as a
formality.

# 4. An unplanned finding: the regime break is visible in the labels

| Split | Period | n | Positive rate |
|---|---|---|---|
| Train | 2016-03 to 2023-06 | 4,956 | 16.3% |
| Validation | 2023-07 to 2024-12 | 1,075 | **34.9%** |
| Test | 2025-01 to 2026-06 | 909 | 12.7% |

The validation window carries more than double the positive rate of either
neighbour. It begins in July 2023, the month of the fuel subsidy removal and
the currency float. The structural break that Phase 1 anticipated on economic
grounds is therefore measurable directly in the target variable, not only in
the price series (Figure 1.2).

This has a useful consequence: model selection happened under harsher
conditions than the test period, so the test estimate is conservative. It also
confirms that rolling-window retraining is necessary rather than precautionary,
since a model fitted only on the calm pre-2023 period would face a different
world.

# 5. Leakage verification

A negative control was injected: a feature computed from month *t+1*
(`price_next / trail_mean`, essentially the label itself) was added to the
feature set and the pipeline refitted.

| Configuration | Test ROC-AUC |
|---|---|
| With injected future feature | 1.000 |
| Real pipeline | 0.737 |

The harness detects leakage when it is present. A pipeline that cannot detect
injected leakage cannot be trusted to rule it out, so this control is reported
rather than assumed. The control feature was then removed.

Additional structural guarantees: every fitted transformer sits inside a
`Pipeline` within a `ColumnTransformer`, so `fit` cannot reach held-out data;
the label is built from *t+1* while all features are backward-looking from *t*;
and `price_next` and `trail_mean` are excluded from the model's column list.

# 6. Figures produced

All eight figures are in `figures/` at 300 DPI.

| Figure | File | What it shows |
|---|---|---|
| 1.1 | `fig1_1_missingness_heatmap.png` | Gap months across 107 series; 12.6% of cells |
| 1.2 | `fig1_2_target_distribution.png` | Class balance overall and by split |
| 2.1 | `fig2_1_outlier_boxplots.png` | Ratio features before and after capping |
| 3.1 | `fig3_1_correlation_matrix.png` | Upper-triangle Spearman; one pair boxed |
| 3.2 | `fig3_2_collinear_scatterplots.png` | Most associated pairs with trendlines |
| 3.3 | `fig3_3_target_importance.png` | Point-biserial and mutual information |
| 4.1 | `fig4_1_transformation_grid.png` | Raw, log, min-max, standardised |
| 5.1 | `fig5_1_pca_scree_plot.png` | Scree and cumulative variance, 90% / 95% |

Two figures required reinterpretation for this problem. **Figure 2.1** shows
ratio features rather than prices, because capping prices would delete positive
cases from the training data: an extreme upward movement *is* the target event.
**Figure 5.1** is diagnostic only; PCA is not adopted, since its components
would destroy the interpretability the logistic model exists to provide, and
that model is the one that performs best here.

# 7. What Phase 3 should carry forward

1. Replace the two independent metric targets with recall at matched
   precision, or precision@k at a fixed operational k.
2. Implement rolling-window retraining, now justified by measured evidence
   rather than economic reasoning alone.
3. Acquire the secondary sources (NBS state-level fuel, CBN exchange rates) and
   test whether they improve performance across the 2023 break specifically.
4. Report performance per series, so that a model doing well on a few
   commodities and badly elsewhere is visible rather than averaged away.
