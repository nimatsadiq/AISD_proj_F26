# A Monthly Early Warning System for Food Price Spikes in Nigerian States

04-652 Artificial Intelligence System Design, Fall 2026
Nimat Ohunene Sadiq, Carnegie Mellon University Africa
`nsadiq@andrew.cmu.edu`

---

## Problem Statement

Nigerian state governments and relief agencies learn that staple food prices
have spiked only after it has happened. Support therefore reaches households
after they have already reduced consumption or sold assets, and food is
procured at peak prices.

Food inflation runs well above general inflation and varies sharply by state.
In July 2026 food inflation was 20.31% against a headline rate of 15.43%, and
in April 2026 Enugu recorded 32.67% against a national figure of 16.06%. Prices
are measured well in Nigeria, but every published source is retrospective. None
forecasts the coming month at state level.

This project predicts, one month ahead, whether the price of a given staple food
in a given state is about to rise sharply above its recent trend. The output is a
monthly ranked warning list for state governments and relief agencies, giving
enough notice to release grain reserves, begin cash transfers, or procure relief
food before prices rise.

The project addresses **SDG 2 (Zero Hunger)**, specifically Target 2.1 (access to
sufficient food year round) and Target 2.c (timely access to market information
to help limit extreme food price volatility).

## Task Type

**Binary classification.**

For each (state, commodity, month), predict whether the price in month *t+1*
exceeds its trailing three-month mean by more than 15%.

## Dataset Overview

**Primary source:** WFP Food Prices for Nigeria, WFP Vulnerability Analysis and
Mapping Price Database, distributed via the Humanitarian Data Exchange.

- Link: <https://data.humdata.org/dataset/wfp-food-prices-for-nigeria>
- Retrieved: **3 October 2026**
- Licence: as stated on the HDX dataset page

The raw file is excluded from version control (see `.gitignore`). Download it
from the link above into `data/raw/` to reproduce this work. Note that WFP
revises this dataset: a copy retrieved on 21 September 2026 contained 88,556
records, eleven more than the version used here, so the retrieval date above is
load-bearing.

### Two matrices

This project reports two shapes, because the file as distributed is not the
matrix the model is trained on.

| | Rows (N) | Columns (P) |
|---|---|---|
| Raw file as distributed | 88,545 | 16 |
| Modelling panel (state x commodity x month) | 9,593 | 8 before feature engineering |

The raw file is in long format: one row per market, commodity, month, unit and
price type. The modelling panel aggregates markets to state level and reindexes
each series onto a continuous monthly grid.

### Raw file coverage

- Date range: January 2002 to July 2026 (295 distinct months)
- 14 states (`admin1`), 49 LGAs, 68 markets
- 43 commodities across 8 categories
- Price types: Retail (76%) and Wholesale
- **No missing cells.** All 88,545 rows are complete, because WFP publishes only
  observed records. Missingness appears as absent rows, not empty cells, and
  becomes visible only after reshaping to the panel.

Coverage is uneven. Dense monthly coverage continuing to 2026 exists only for
Borno, Yobe and Adamawa; most other states end in January 2023, and records
before 2014 are sparse. The project is therefore built and validated on those
three states, with extension to the remaining states via NBS state-level series
stated as planned work.

### Modelling panel

- Scope: Borno, Yobe, Adamawa; retail prices; January 2016 to July 2026
- 113 state-commodity series, 9,593 (state, commodity, month) cells
- 8,005 cells observed (83.4%), 1,588 gap months (16.6%)
- Gap rate by state: Adamawa 30.6%, Borno 14.3%, Yobe 12.2%

### Target variable (Y)

`spike_next_month` ∈ {0, 1}, where 1 indicates the price in month *t+1* exceeds
its trailing three-month mean by more than 15%.

- Labelled rows: 6,687 (rows need three prior months to form the trend)
- Positive: 1,207 (18.0%); negative: 5,480
- Class imbalance ratio: **4.5 : 1**

Mitigation is class weighting rather than synthetic oversampling; see the Phase 1
strategy report for the justification.

### Secondary sources (planned)

Named in the project proposal and to be acquired for Phase 2:

- NBS Consumer Price Index, Selected Food Price Watch, and state-level petrol and
  diesel price watches: <https://nigerianstat.gov.ng/>
- CBN exchange rate series: <https://www.cbn.gov.ng/rates/>
- Rainfall indicators for Nigeria via HDX: <https://data.humdata.org/group/nga>

## Repository Structure

```
.
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                      <- download the WFP file here (untracked)
│   └── processed/                <- cleaned feature matrix (untracked)
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf
├── phase2_implementation/
│   └── notebooks/
├── src/
│   ├── __init__.py
│   ├── transformers.py
│   └── pipeline_builder.py
└── figures/
```

## Reproduction Steps

1. Clone the repository.

   ```bash
   git clone https://github.com/nimatsadiq/AISD_proj_F26.git
   cd AISD_proj_F26
   ```

2. Create the environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   On Google Colab, clone the repository and run
   `pip install -r requirements.txt` in the first cell.

3. Download the data. Fetch the Nigeria food prices CSV from
   <https://data.humdata.org/dataset/wfp-food-prices-for-nigeria> and save it to
   `data/raw/wfp_food_prices_nga.csv`.

4. Run the Phase 2 notebook in `phase2_implementation/notebooks/`.

## Pipeline Summary

1. **Panel construction.** Filter to retail prices in the three in-scope states;
   collapse markets within a state to the median; reindex each state-commodity
   series onto a continuous monthly grid so gaps become explicit.
2. **Feature construction.** Backward-looking price momentum, deviation from
   rolling means, volatility, calendar month, same month in prior years,
   neighbouring-state changes, and missingness flags.
3. **Label construction.** Compare month *t+1* against the trailing three-month
   mean at a 15% threshold.
4. **Split.** Forward chronological split, with features rebuilt inside each fold.
5. **Models.** Regularised logistic regression as an interpretable baseline, then
   gradient boosted trees, measured against a seasonal persistence baseline.

Primary metric is recall on spikes (target ≥ 0.80), with precision among
top-ranked warnings (target ≥ 0.60).

## Status

Phase 1 (strategy and design) complete. Phase 2 (implementation and diagnostics)
in progress.
