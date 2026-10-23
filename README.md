# A Monthly Early Warning System for Food Price Spikes in Nigerian States

**04-652 Artificial Intelligence System Design, Fall 2026**
Nimat Ohunene Sadiq · Carnegie Mellon University Africa · `nsadiq@andrew.cmu.edu`

---

## Problem Statement

Nigerian state governments and relief agencies learn that staple food prices
have spiked only **after** it has happened. Support therefore reaches
households after they have already reduced consumption or sold assets, and
food is procured at peak prices.

Food inflation runs well above general inflation and varies sharply by state.
In July 2026 food inflation was 20.31% against a headline rate of 15.43%, and
in April 2026 Enugu recorded 32.67% against a national figure of 16.06%. Prices
are measured well in Nigeria, but every published source is retrospective.
Nothing forecasts the coming month at state level.

This project predicts, one month ahead, whether the price of a given staple in
a given state is about to rise sharply above its recent trend. The output is a
monthly ranked warning list, giving enough notice to release grain reserves,
begin cash transfers, or procure relief food before prices rise.

**SDG 2 (Zero Hunger)**, Target 2.1 (access to sufficient food year round) and
Target 2.c (timely market information to limit extreme food price volatility).

## Task Type

**Binary classification.** For each `(state, commodity, month)`, predict whether
the price in month *t+1* exceeds its trailing three-month mean by more than 15%.

---

## Dataset Overview

**Source:** WFP Food Prices for Nigeria (WFP Vulnerability Analysis and Mapping
Price Database, via the Humanitarian Data Exchange)
**Link:** https://data.humdata.org/dataset/wfp-food-prices-for-nigeria
**Retrieved:** 3 October 2026 · **Licence:** as stated on the HDX dataset page

The raw file is **not** in version control (see `.gitignore`). The notebook
downloads it automatically if `data/raw/` is empty.

> **This dataset is revised in place.** Three copies taken during this project
> held 88,556 records (21 Sep 2026), 88,545 (3 Oct 2026) and 88,391 (20 Oct
> 2026). All differences were removals; no price changed on any shared record.
> Results below use the **3 October** copy, and re-downloading will shift the
> numbers slightly. This is why the retrieval date is recorded as part of the
> audit rather than as a footnote.

### Two matrices

The file as distributed is not the matrix the model trains on.

| | Rows (N) | Columns (P) |
|---|---|---|
| Raw file, long format | 88,545 | 16 |
| Modelling panel (`state` × `commodity` × `month`) | 9,000 | 32 |
| Labelled rows (three prior months required) | 6,940 | 32 |
| Design matrix after encoding | 4,956 / 1,075 / 909 | **17** |

### What a few rows look like

| date       | admin1 | market          | commodity | unit   | pricetype | priceflag | price  | usdprice |
|:-----------|:-------|:----------------|:----------|:-------|:----------|:----------|-------:|---------:|
| 2026-06-15 | Borno  | Monday          | Millet    | 2.6 KG | Retail    | aggregate | 1262.5 | 0.93 |
| 2026-06-15 | Borno  | Tashan Bama     | Millet    | 2.6 KG | Retail    | aggregate | 1237.5 | 0.91 |
| 2026-06-15 | Borno  | Damassack (CBM) | Millet    | 2.6 KG | Retail    | aggregate | 1200.0 | 0.88 |
| 2026-07-15 | Borno  | Rann            | Millet    | 2.6 KG | Retail    | aggregate | 1200.0 | 0.87 |

One row is **one market, one commodity, one month**. Dates are always the 15th,
so the column encodes a month rather than a day. Several markets in the same
state report the same commodity in the same month, which is why the panel
collapses them to the state **median** (robust to a single besieged market
printing several times the state norm).

### Variable inventory

| Column | Type | Distinct | Note |
|---|---|---|---|
| `date` | Datetime | 295 | Always the 15th; a month index |
| `admin1` | Nominal | 14 | State |
| `admin2` | Nominal | 49 | LGA; unused at state level |
| `market`, `market_id` | Nominal | 68 | Aggregated away |
| `latitude`, `longitude` | Continuous | 56 / 59 | Market location |
| `category` | Nominal | 8 | Food group; source of `food_type` |
| `commodity`, `commodity_id` | Nominal | 43 | |
| `unit` | Nominal | 20 | Consistent *within* a commodity |
| `priceflag` | Nominal | 3 | `actual` / `aggregate` |
| `pricetype` | Nominal | 2 | Retail (76%) / Wholesale |
| `currency` | Nominal | 1 | Constant NGN; zero variance, dropped |
| `price` | Continuous | 25,876 | **The signal** |
| `usdprice` | Continuous | 8,914 | Derived; never used (moves with the currency) |

### Data quality

| Check | Result |
|---|---|
| Missing cells in the raw file | **0** |
| Fully duplicated rows | 0 |
| Duplicates on `(date, market, commodity, unit, pricetype)` | 0 |
| Gap months once reshaped to a monthly grid | 1,132 of 9,000 (12.6%) |
| Single-month gaps interpolated and flagged | 356 |

**The raw file has no nulls, and that is a trap.** WFP publishes only observed
records, so a market that fails to report produces *no row at all*. Missingness
is structural and becomes visible only after reshaping onto a continuous
monthly grid. The duplicate check on the natural key matters more than the
identical-row check: it asks whether one market reported the same commodity
twice in a month at different prices, which would force a choice about which to
keep. It does not happen here.

### Target variable (Y)

`spike_next_month` ∈ {0, 1} — price in *t+1* above its trailing three-month
mean by more than 15%.

| | Value |
|---|---|
| Labelled rows | 6,940 |
| Positive | 1,300 (18.7%) |
| Negative | 5,640 (81.3%) |
| Imbalance | **4.3 : 1** |

Handled with class weighting (`class_weight='balanced'`, `scale_pos_weight`),
not synthetic oversampling: SMOTE interpolates between neighbouring months and
would fabricate price histories that never occurred.

---

## What the data looks like

### Coverage is concentrated, not national

![Records per state](figures/eda_1_rows_per_state.png)

WFP collects prices where it operates. Borno (31,273 records) and Yobe (28,922)
account for about 68% of the file and are the only states running to 2026.
Adamawa continues to November 2025. **Eleven of fourteen states stop in January
2023**, and Enugu, Kwara and most of the south never appear at all. This is why
the project is scoped to the northeast and treats national coverage via NBS
series as planned work rather than a claim.

### The usable history is shorter than the date range suggests

![Records per year](figures/eda_2_records_per_year.png)

The file spans January 2002 to July 2026, but records before 2014 are sparse.
"Back to 2002" is technically true and misleading, so the modelling period
starts in 2016.

### Observed months per series

![Coverage grid](figures/eda_6_coverage_grid.png)

107 state-commodity series across a possible 127 months.

### Spikes cluster strongly by year

![Spikes by year](figures/eda_3_spikes_by_year.png)

| Year | Spike rate | Year | Spike rate |
|---|---|---|---|
| 2016 | 22.0% | 2022 | 10.1% |
| 2017 | 22.6% | 2023 | 32.0% |
| 2018 | 10.2% | **2024** | **37.1%** |
| 2019 | 5.8% | 2025 | 7.4% |
| 2020 | 26.0% | 2026 | 24.3% |
| 2021 | 17.7% | | |

2024 is the worst year in the series at 37.1%, and 2019 the calmest at 5.8%.
The jump from 2022 to 2023-24 coincides with fuel subsidy removal and the
currency float.

### And by calendar month

![Spikes by month](figures/eda_4_spikes_by_month.png)

Spike rate peaks in **June at 31.0%** and troughs in **October at 10.7%**, a
three-fold swing. June sits at the end of the lean season before the main
harvest; October follows it. Perishables and storables diverge after the
harvest: storables fall to 8% by October while perishables stay near 16% and
climb again into November and December. This is the seasonal structure the
model's cyclical month encoding is designed to capture, and the evidence that
splitting by `food_type` is justified.

### Prices themselves

![Price series](figures/eda_5_price_series.png)

Four staples in Borno, with spike months marked and July 2023 dashed. Local
rice roughly triples after mid-2023. Tomatoes swing violently throughout,
consistent with their 32% spike rate against 6% for imported rice. This level
shift is the structural break that makes rolling-window retraining necessary.

---

## Repository Structure

```
.
├── README.md
├── .gitignore                       <- excludes data/
├── requirements.txt
├── data/
│   ├── raw/                         <- WFP file lands here (untracked)
│   └── processed/                   <- panel + results (untracked)
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf   <- audit, imputation, leakage plan
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb <- runs end to end
│   ├── Phase2_Final_Report.pdf      <- decision log, strategy vs reality
│   └── Feedback_Response.pdf        <- response to Phase 1 review
├── src/
│   ├── transformers.py              <- custom scikit-learn transformers
│   ├── pipeline_builder.py          <- ColumnTransformer + Pipelines
│   ├── calibration.py               <- post-hoc probability calibration
│   ├── figures.py                   <- the 8 required diagnostics
│   └── eda_figures.py               <- the descriptive figures above
└── figures/                         <- all figures, 300 DPI
```

## Reproduction

```bash
git clone https://github.com/nimatsadiq/AISD_proj_F26.git
cd AISD_proj_F26
pip install -r requirements.txt
jupyter notebook phase2_implementation/notebooks/pipeline_execution.ipynb
```

On Colab, uncomment the clone and install lines in the first cell. The data
cell fetches the raw file from HDX if `data/raw/` is empty, so no manual
download is needed. No GPU is required: the data is tabular and fits in memory.

## Pipeline Summary

1. **Panel construction** — filter to retail prices in the three in-scope
   states; collapse markets to the state median; reindex each series onto a
   continuous monthly grid so gaps become explicit.
2. **Features** — backward-looking momentum, deviation from rolling means,
   volatility, cyclical calendar month, same month in prior years,
   neighbouring-state changes, fuel-price momentum, missingness flags.
3. **Label** — month *t+1* against the trailing three-month mean at 15%.
4. **Split** — forward chronological 70/15/15; no stratification, because it
   would require reshuffling and destroy temporal order.
5. **Models** — regularised logistic regression (scaled) and gradient boosted
   trees (unscaled), against a seasonal persistence baseline.

## Reports

| Document | What it covers |
|---|---|
| [Phase 1 Strategy Report](phase1_strategy/Phase1_Strategy_Report.pdf) | Data audit, missingness mechanisms, imputation and outlier rules, encoding and scaling, split design and leakage architecture |
| [Phase 2 Final Report](phase2_implementation/Phase2_Final_Report.pdf) | Decision log, five documented deviations, leakage verification |
| [Feedback Response](phase2_implementation/Feedback_Response.pdf) | Response to Phase 1 review, including probability calibration |

## Results

| Model | Threshold | Recall | Precision | PR-AUC | ROC-AUC | Precision@50 |
|---|---|---|---|---|---|---|
| Logistic regression | 0.37 | **0.835** | 0.194 | 0.353 | 0.737 | 0.480 |
| Gradient boosted trees | 0.39 | 0.504 | 0.195 | 0.301 | 0.635 | 0.500 |
| Seasonal persistence baseline | n/a | 0.043 | 0.714 | n/a | n/a | n/a |

Test base rate 0.127. Thresholds chosen on the validation split; test scored
once. Recall target of 0.80 met; the 0.60 precision target is not jointly
achievable at this base rate, so precision@50 is the honest operational figure.

Scores are **calibrated** post hoc (Platt scaling, fitted on validation only)
before any threshold is chosen, because weighted loss inflates raw
probabilities. Brier improves from 0.189 to 0.129 for the logistic model, while
precision@50 and PR-AUC are unchanged, since calibration is monotonic.

![Calibration](figures/fig6_1_calibration.png)

Full discussion in `Phase2_Final_Report.pdf` and `Feedback_Response.pdf`.

## Status

Phase 1 (strategy) and Phase 2 (implementation and diagnostics) complete.
