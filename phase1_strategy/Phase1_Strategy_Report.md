---
title: "Phase 1: Preprocessing and Feature Engineering Strategy"
subtitle: "A Monthly Early Warning System for Food Price Spikes in Nigerian States"
author: "Nimat Ohunene Sadiq | Carnegie Mellon University Africa | nsadiq@andrew.cmu.edu"
date: "04-652 Artificial Intelligence System Design, Fall 2026"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
---

# 1. Dataset and Problem Audit

## 1.1 A note on scope: this dataset has two shapes

The WFP file is distributed in long format, one row per market, commodity,
month, unit and price type. That is not the matrix a model trains on. The
modelling unit is one **(state, commodity, month)** cell, because a state
official acts on a state, not on an individual market. Both shapes are reported
throughout, and conflating them would misstate every missingness and imbalance
figure that follows.

| | Rows (N) | Columns (P) |
|---|---|---|
| Raw file as distributed | 88,545 | 16 |
| Modelling panel | 9,593 | 8 before feature engineering |

Raw file retrieved 3 October 2026 from the Humanitarian Data Exchange. A copy
retrieved on 21 September 2026 held 88,556 records, eleven more, all of them
Bread retail observations removed upstream. No price changed on any shared
record. The dataset is revised in place, so the retrieval date is recorded as
part of the audit rather than as a footnote.

## 1.2 Variable inventory

| Column | Type | Distinct | Notes |
|---|---|---|---|
| `date` | Datetime | 295 | Always the 15th; encodes month, not day |
| `admin1` | Nominal | 14 | State |
| `admin2` | Nominal | 49 | LGA; not used at state level |
| `market` | Nominal | 68 | Aggregated away during panel construction |
| `market_id` | Nominal (ID) | 68 | Duplicate of `market` |
| `latitude` | Continuous | 56 | Market location; used for adjacency |
| `longitude` | Continuous | 59 | Market location; used for adjacency |
| `category` | Nominal | 8 | Food group |
| `commodity` | Nominal | 43 | High cardinality relative to N per series |
| `commodity_id` | Nominal (ID) | 43 | Duplicate of `commodity` |
| `unit` | Nominal | 20 | Consistent within commodity |
| `priceflag` | Nominal | 3 | actual / aggregate |
| `pricetype` | Nominal | 2 | Retail / Wholesale |
| `currency` | Nominal | 1 | Constant (NGN); zero variance |
| `price` | Continuous | 25,876 | The signal |
| `usdprice` | Continuous | 8,914 | Derived; see 1.5 |

**Type notes.** `market_id` and `commodity_id` are integers but carry no ordinal
meaning and are dropped as duplicates of their label columns. `currency` has
zero variance and is dropped. `date` is treated as a month index, not a
timestamp. There are no genuinely ordinal variables in the raw file; the only
ordered structure is time itself.

## 1.3 Baseline missingness

**The raw file contains no missing values.** All 88,545 rows are complete across
all 16 columns; 0.000% of cells are null.

This is a reporting artefact, not an absence of missingness. WFP publishes only
observed records, so a market that fails to report produces no row at all.
Treating this as "no missing data, nothing to impute" would be a serious
misreading.

Missingness becomes visible only after reshaping onto a continuous monthly grid:

| Metric | Value |
|---|---|
| Panel cells (113 series x 127 months, unbalanced) | 9,593 |
| Observed cells | 8,005 (83.4%) |
| Gap months | 1,588 (16.6%) |
| Gap rate, Adamawa | 30.6% |
| Gap rate, Borno | 14.3% |
| Gap rate, Yobe | 12.2% |

Adamawa is retained despite near one-third sparsity, because excluding it would
drop the only in-scope state outside the Borno-Yobe conflict core and would
weaken the geographic contrast the model needs. Its higher gap rate is carried
explicitly as a missingness flag rather than smoothed over.

## 1.4 Target variable specification

**Task: binary classification.** `spike_next_month` = 1 if the price in month
*t+1* exceeds its trailing three-month mean by more than 15%.

Two definitions were evaluated empirically before choosing:

| Threshold | Month-on-month (n=7,514) | Versus 3-month trend (n=6,687) |
|---|---|---|
| 5% | 28.7% (2,156) | 37.6% (2,515) |
| 10% | 18.2% (1,365) | 25.5% (1,708) |
| **15%** | 11.7% (881) | **18.0% (1,207)** |
| 20% | 8.3% (621) | 12.9% (861) |
| 25% | 6.2% (463) | 9.0% (605) |

**Justification for 15% against the trailing mean.** The trend-based definition
catches sustained climbs that no single month-on-month change would flag, which
is the failure mode that matters operationally: a household is squeezed by three
months of 6% rises just as surely as by one month of 18%. It costs 827 labelled
rows, because three prior months are required, and that cost is accepted.

The threshold is a declaration of what counts as an emergency, not a statistical
artefact. Below 15% the label starts capturing ordinary inflation in an economy
where food inflation ran above 20% year-on-year in July 2026. Above 20% the
positive count thins across 113 series. At 15% there are 1,207 positives, enough
to learn from.

**Class imbalance: 4.5 : 1** (5,480 negative, 1,207 positive, 18.0% positive).

**Mitigation: class weighting, not synthetic oversampling.** At 4.5 : 1 the
imbalance is mild. SMOTE and its variants interpolate between neighbouring
observations, which for a time series would fabricate price histories that never
occurred and risk leaking across the temporal boundary. `class_weight='balanced'`
for logistic regression and `scale_pos_weight` for gradient boosted trees achieve
the same re-weighting without inventing data.

**Probability calibration.** Weighted loss functions deliberately distort the
decision boundary, so raw predicted scores overstate the probability of a spike
and are not directly interpretable. Scores are therefore calibrated post hoc
(isotonic or Platt scaling, fitted on a held-out split and never on training
data) before any threshold is selected, and calibration quality is reported via
Brier score and a reliability curve. Calibration is monotonic, so it does not
change the ranking used for top-K warnings; it changes what a displayed
probability means.

## 1.5 A derived column worth noting

Dividing `price` by `usdprice` recovers the exchange rate WFP applied: roughly
199 NGN/USD in 2016, 452 in January 2023, 775 after the July 2023 float, a peak
near 1,611 in March 2024, and 1,376 in July 2026. This provides a free
cross-check against the CBN series. `usdprice` is **never** used as a target or
a feature in levels, because it moves with the currency rather than with food
supply, and would import exchange-rate shocks into the label.

---

# 2. Cleaning and Imputation Strategy

## 2.1 Missingness mechanism classification

| Feature group | Mechanism | Evidence and reasoning |
|---|---|---|
| Panel gap months | **MNAR** | Markets suspend reporting during insecurity and disruption, which is also when prices spike. Missingness depends on the unobserved value itself. |
| Early-period sparsity (pre-2014) | **MCAR** | Reflects the historical extent of WFP's collection programme, unrelated to price levels. Handled by restricting to 2016 onward. |
| Commodity coverage differences | **MAR** | Which commodities are tracked depends on the observed state and market, not on the prices themselves. |

The MNAR classification is the consequential one. It has three consequences that
run through the rest of this strategy:

1. **Gaps are informative and must be encoded, not erased.** A binary
   `was_missing` flag and a `months_since_last_observation` counter enter the
   feature set, per the standard treatment of MNAR requiring dedicated
   missingness indicators.
2. **Long gaps must not be bridged.** Interpolating across a six-month silence
   invents a smooth price path through precisely the period most likely to
   contain a spike, which would both destroy the label and leak a reconstructed
   future into the features.
3. **Imputation is bounded.** Only short gaps are filled, and the fill is
   recorded.

## 2.2 Imputation rules

| Case | Rule | Justification |
|---|---|---|
| Gap of 1 month, series otherwise dense | Linear interpolation between neighbours, flagged | A single missing month between two observations is unlikely to conceal a full spike cycle; the cost of dropping the row exceeds the risk. |
| Gap of 2 or more months | **No imputation.** Rows remain unlabelled and are excluded from training | Bridging a long silence fabricates the period of greatest interest. |
| Gap at the start of a series | No backfill | There is no prior information; backfilling would invent history. |
| Missing in a secondary source (fuel, FX, rainfall) | Forward fill up to 1 month, then flag | These are published series with their own lags; carrying the last published value forward is what an operator would actually have. |

**Mean versus median.** Where a central value is required, for example when
collapsing several markets within a state into one state value, the **median** is
used. Market-level prices within a state are right-skewed: a single market in a
besieged area can print a price several times the state norm. The mean would
transmit that excursion into the state series and manufacture a spike that most
of the state never experienced. The median is robust to it. This is the central
tendency decision the assignment asks to be justified on distribution shape, and
skewness is the reason.

**Mode** is not used; there are no categorical features with missing values,
since categorical columns are complete in the raw file by construction.

**KNN imputation is rejected.** It would borrow values from similar
state-commodity-months elsewhere in the dataset, which in a temporal design means
borrowing across the train and test boundary. The leakage risk is not worth the
modest gain over interpolation.

## 2.3 Outlier identification

Distribution shape dictates the rule, and these distributions are not Gaussian.
Price series are right-skewed and non-stationary, with a level shift after the
2023 subsidy removal and currency float. The Z-score rule assumes a stable mean
and standard deviation and would therefore flag an entire post-2023 regime as
outlying. **The IQR rule is used**, with bounds computed per state-commodity
series rather than globally, since a global bound across 43 commodities with
different units and price levels would be meaningless.

Flagging rule: values outside `[Q1 - 1.5 x IQR, Q3 + 1.5 x IQR]`, computed within
each series on the training period only.

## 2.4 Outlier treatment: the decision that defines this project

**Extreme price values are not capped.**

This is a deliberate departure from the default. In most datasets an extreme
value is noise to be winsorized. Here, an extreme upward price movement **is the
target event.** Capping prices at the 1.5 x IQR upper bound would systematically
flatten the largest spikes, and since the label is derived from the price series,
it would delete positive cases from the training data. The model would be trained
to predict a phenomenon that had been removed from its view.

The IQR screen is therefore used **diagnostically, not correctively**. Flagged
values are triaged:

| Finding | Action | Reason |
|---|---|---|
| Physically implausible (non-positive price, unit mismatch, order-of-magnitude error against the series median) | Correct if the cause is identifiable, otherwise treat as a gap | A data-entry error is not a market signal. |
| Extreme but plausible, consistent with neighbouring states or a known shock | **Retain unchanged** | This is the event of interest. |
| Extreme and isolated, with no corroboration in any neighbouring series or adjacent month | Retain, and flag as low-confidence | Single-market excursions are plausible in conflict zones; discarding them would bias the data toward calm periods. |

Row deletion is likewise rejected, for the reasons the assignment sets out:
dropping a full row for one anomalous column reduces N and discards valid signal
in the remaining features. In a panel it is worse than that, because deleting a
row punches a hole in a time series and corrupts every lag feature computed
across it.

**Where capping does apply:** engineered ratio features. A month-on-month ratio
computed across a long gap can produce an absurd multiple. These derived values
are capped at the 1st and 99th percentiles of the training distribution, because
they are artefacts of construction rather than observed market behaviour.

## 2.5 Statistical association and correlation strategy

### Feature-target test matching matrix

The target is binary. Tests are matched to the feature type on each side.

| Feature type | Example | Test against binary target | Why |
|---|---|---|---|
| Continuous | Price momentum, deviation from rolling mean, volatility | **Point-biserial correlation** | The appropriate form of Pearson's r when one variable is dichotomous. |
| Continuous, non-normal | Raw price level, fuel price | **Mann-Whitney U**, plus **Spearman's rho** for monotonic association | Price distributions are skewed; Pearson's assumptions do not hold. |
| Nominal categorical | State, commodity, food type | **Chi-square test of independence**, with **Cramer's V** for effect size | Both variables categorical. Cramer's V is reported because chi-square significance is inflated at n = 6,687. |
| Discrete count | Months since last observation | **Mutual information** | Captures non-monotonic relationships a correlation coefficient would miss. |
| Ordered cyclical | Calendar month | **Chi-square** on month versus label | Tests the seasonality hypothesis directly. |

**Mutual information is additionally computed for all features** as a
model-agnostic ranking, because the tree models are not restricted to linear or
monotonic relationships and a purely linear screen would understate features
that matter to them.

### Multicollinearity protocol

Severe collinearity is **expected by construction here**, not a defect to be
discovered. Momentum over one month and momentum over three months share a term.
Deviation from a six-month mean and from a twelve-month mean overlap heavily.
This is an unavoidable property of lagged feature design.

| Stage | Rule |
|---|---|
| 1. Pairwise screen | Flag pairs with \|r\| >= 0.85 (Spearman, given skew) |
| 2. Variance inflation | Compute VIF on the surviving continuous set; drop while any VIF > 10 |
| 3. Decision rule | Within a flagged pair, retain the feature with the higher mutual information against the target; break ties on lower missingness ratio, then on the shorter window, which is more responsive and less autocorrelated with other features. Never drop both. |
| 4. Model-dependent application | Apply the reduced set to **logistic regression**; retain the full set for **gradient boosted trees** |

Step 4 is the point worth defending. Collinearity destabilises linear
coefficients and makes them uninterpretable, which matters because the logistic
model exists to be interpreted. Tree ensembles are not destabilised by
correlated inputs; they split on whichever is most useful. Applying one
collinearity rule to both models would needlessly discard information from the
model that can use it. Both feature sets are logged, and the difference is
reported in Phase 2.

---

# 3. Encoding, Scaling and Transformation Strategy

## 3.1 Encoding

| Feature | Cardinality | Encoding | Justification |
|---|---|---|---|
| `admin1` (state) | 3 | **One-hot** | Low cardinality, no inherent order. Three columns is a negligible dimensional cost. |
| `commodity` | 43 | **Target encoding with out-of-fold fitting**, plus a coarse one-hot of the 8 `category` values | One-hot on 43 levels against 6,687 rows would add 43 sparse columns and invite overfitting on rare commodities. Target encoding captures that commodities differ sharply in baseline spike rate (tomatoes 32.4%, beef 8.7%) in a single column. |
| `food_type` (perishable / storable) | 2 | **Binary** | Engineered feature. Addresses the mechanism directly: perishables respond to immediate local supply, storables to harvests, reserves and currency. |
| `priceflag` | 3 | **One-hot** | Data-provenance covariate; from 2024 all records are `aggregate`, so this also proxies for period. |
| `calendar_month` | 12 | **Cyclical (sine and cosine)** | Ordinal encoding would assert that December (11) is maximally distant from January (0), when they are adjacent. Sine-cosine encoding preserves the wrap-around, which matters because the planting and harvest cycle is the dominant seasonal driver. |

**Target encoding carries a leakage risk and is handled accordingly.** The
encoding is fitted inside each training fold only, using out-of-fold means with
smoothing toward the global rate for commodities with few observations. Fitting
it on the full dataset would embed test-period spike rates into a training
feature. This is the single most likely place for leakage to enter this pipeline
unnoticed, and it is implemented inside the scikit-learn `Pipeline` so that it
cannot be fitted outside a fold by accident.

**No ordinal encoding is used.** No raw categorical variable has an inherent
rank. Imposing an integer order on state or commodity would assert a false
magnitude relationship.

## 3.2 Scaling: matched to the algorithm, not applied uniformly

| Model | Scaling | Justification |
|---|---|---|
| Regularised logistic regression | **StandardScaler** on all continuous features | L1/L2 penalties shrink coefficients in proportion to feature magnitude. Unscaled inputs would mean a feature measured in naira is penalised differently from one measured as a ratio, purely because of units. Standardisation makes the penalty comparable across features and the coefficients interpretable. |
| Gradient boosted trees | **None** | Trees split on rank order, which is invariant to monotonic rescaling. Scaling would add a fitted transformer to the pipeline with no benefit and one more opportunity for leakage. |

**MinMaxScaler is rejected.** It is bounded by the training minimum and maximum,
and is acutely sensitive to outliers. Given that extreme values are deliberately
retained here (section 2.4), MinMax scaling would compress the entire ordinary
range into a narrow band dominated by a handful of spikes. It is also unsuited to
a non-stationary series: prices in 2026 exceed anything in the 2016 training
range, so test values would fall outside [0, 1] regardless.

## 3.3 Transformation

Price **levels** are right-skewed and non-stationary, and a log transform is
applied where a level is used directly, both to compress the skew and because
differences of logs approximate proportional change, which is the quantity of
interest. The IQR outlier screen in section 2.3 is applied on the log scale for
the same reason.

Most engineered features are already **ratios or percentage changes**, which are
approximately stationary and need no further transformation. This is deliberate:
modelling relative movement rather than absolute naira prices is what allows a
single pooled model to span commodities with different price levels and units,
and it blunts the effect of the currency-driven level shifts of 2023 and 2024.

---

# 4. Split Strategy and Leakage Prevention Architecture

## 4.1 Partitioning

**Forward chronological split, approximately 70 / 15 / 15 by time.**

| Split | Period | Approximate labelled rows |
|---|---|---|
| Train | 2016-01 to 2023-06 | ~4,700 |
| Validation | 2023-07 to 2024-12 | ~1,000 |
| Test | 2025-01 to 2026-07 | ~970 |

Exact boundaries are fixed once and recorded; the figures above are the planned
allocation given 6,687 labelled rows.

**Random splitting is inadmissible.** This dataset contains repeated
observations of the same entity: each (state, commodity) pair is one series
observed monthly for up to 127 months. Price series are autocorrelated, so a
randomly held-out month sits between two training months whose values nearly
determine it. Reported performance would be inflated and would not survive
deployment, where the future is genuinely unseen.

**Stratification is not applied, and this is intentional.** The assignment
rightly recommends stratified splitting for imbalanced classification, but
stratification requires reshuffling, which destroys the temporal ordering that is
the entire point of this design. The class balance of each split is reported
instead, and if a split proves degenerate the boundary is moved rather than the
ordering broken. This is the one place where the generic rubric and a temporal
problem genuinely conflict, and temporal integrity takes precedence.

The validation period deliberately spans the post-float, high-inflation regime,
so that model selection is made under conditions resembling the test period
rather than the calmer earlier years.

## 4.2 Grouping

Each **(state, commodity)** series is a group, and the repeated-measures
structure is what makes group awareness necessary. Group integrity is preserved
implicitly by the temporal split, since every series is cut at the same dates: a
chronological boundary is simultaneously a group-respecting boundary, because no
series can appear on both sides of a date. Where a non-temporal split is ever
required, `GroupKFold` or `StratifiedGroupKFold` keyed on (state, commodity)
would be the correct substitute.

Group awareness also matters for two further decisions: features from
neighbouring states must respect the same time boundary, and performance is
reported per series so that a model doing well on Borno millet and badly
everywhere else is visible rather than averaged away.

## 4.3 Validation scheme

**`TimeSeriesSplit` with expanding window, 5 folds, inside the training period.**
Each fold trains on all data up to a cut point and validates on the following
block, mimicking the real operating pattern in which a model is retrained as new
months arrive.

This also implements the rolling-window retraining committed to in the project
proposal as the mitigation for structural breaks, and lets us measure directly
whether performance degrades around the June 2023 subsidy removal and the 2024
currency moves.

## 4.4 Leakage prevention architecture

Five mechanisms, in order of how easily each is violated.

**1. Every transformer is fitted on `X_train` only.** Imputation values, IQR
bounds, scaler parameters, target encodings, calibration mappings and
collinearity decisions are all derived from training data and applied unchanged
to validation and test. This is enforced structurally by placing every step
inside a scikit-learn `Pipeline` within a `ColumnTransformer`, so that `fit`
cannot touch held-out data.

**2. Features are rebuilt inside each fold.** Rolling means and momentum windows
reach backwards in time. Computing them once over the full panel before splitting
would allow a test-period observation to contribute to a training feature through
a shared window. The feature builder is a pipeline step, not a preprocessing
script run beforehand. This is the subtlest leak available in a time-series
design and the one most often missed.

**3. Publication lag is respected.** Official prices for a month are released
around the middle of the following month. Features for month *t* are therefore
restricted to data that would actually have been published by the prediction
date. A model trained on same-month figures would appear accurate in testing and
fail in operation, because the operator does not have those figures yet.

**4. The label never touches the feature set.** The label is built from month
*t+1*. Every feature is strictly backward-looking from month *t*. Any
accidentally forward-looking feature, such as a centred rolling mean, is excluded
by design.

**5. Target encoding and calibration are fitted out-of-fold.** As set out in
sections 1.4 and 3.1, a commodity's historical spike rate and the probability
calibration mapping are computed from training folds only.

**Verification.** A deliberate negative control is run in Phase 2: a feature
constructed from month *t+1* is injected into a test run. If the pipeline is
sound, performance jumps implausibly, confirming the harness can detect leakage.
It is then removed. A pipeline that cannot detect injected leakage cannot be
trusted to rule it out.

---

# 5. Summary of Commitments for Phase 2

| Area | Decision |
|---|---|
| Scope | Borno, Yobe, Adamawa; retail prices; 2016-01 to 2026-07 |
| Unit | (state, commodity, month) |
| Label | Price in *t+1* exceeds trailing 3-month mean by > 15% |
| Imbalance | 4.5 : 1; class weighting, not SMOTE; calibrate scores afterwards |
| Central tendency | Median across markets, justified by right-skew |
| Missingness | MNAR; indicators retained; interpolation limited to 1-month gaps |
| Outliers | IQR on the log scale, flagged for diagnosis; prices never capped; ratio features capped at 1st/99th percentile |
| Collinearity | \|r\| >= 0.85 and VIF > 10; reduced set for logistic regression, full set for trees |
| Encoding | One-hot (state, priceflag), out-of-fold target encoding (commodity), cyclical (month) |
| Scaling | StandardScaler for logistic regression; none for trees |
| Split | Forward chronological 70/15/15; group-respecting by construction; no stratification; TimeSeriesSplit with 5 expanding folds |
| Metrics | PR-AUC and recall on spikes >= 0.80; precision among top-ranked warnings >= 0.60; Brier score and reliability curve; beat seasonal persistence baseline |

Phase 2 will report each of these against what implementation actually required,
with deviations and their rationale recorded in the decision log.
