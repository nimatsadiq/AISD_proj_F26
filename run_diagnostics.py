"""Compute every statistic cited in the technical report and save as JSON.

Run from the repository root:

    python -m src.run_diagnostics

Writes `data/processed/diagnostics.json`. The report quotes these numbers, so
regenerating this file is how a reader verifies any claim in it.

All statistics are computed on the training split alone. A diagnostic computed
on the full dataset is a decision informed by the test period, which is leakage
by a slower route than fitting a scaler on it.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from . import diagnostics as D
from .pipeline_builder import (
    NUMERIC_FEATURES,
    RATIO_FEATURES,
    TARGET,
    build_panel,
    temporal_split,
)

RAW = "data/raw/wfp_food_prices_nga.csv"
OUT = "data/processed/diagnostics.json"


def main(raw_path=RAW, out_path=OUT, threshold=0.15):
    raw = pd.read_csv(raw_path)
    panel = build_panel(raw, threshold=threshold)
    train, val, test = temporal_split(panel)

    feats = [c for c in NUMERIC_FEATURES if train[c].notna().sum() > 50]
    ratios = [c for c in RATIO_FEATURES if c in train.columns]

    r = {}

    # ---- shapes -----------------------------------------------------------
    r["dimensions"] = {
        "raw": list(raw.shape),
        "panel": list(panel.shape),
        "labelled": int(panel[TARGET].notna().sum()),
        "train": list(train.shape),
        "validation": list(val.shape),
        "test": list(test.shape),
        "series": int(panel.groupby(["state", "commodity"]).ngroups),
    }

    # ---- target -----------------------------------------------------------
    lab = panel.dropna(subset=[TARGET])
    pos = int(lab[TARGET].sum())
    r["target"] = {
        "labelled_rows": int(len(lab)),
        "positive": pos,
        "negative": int(len(lab) - pos),
        "positive_rate": round(float(lab[TARGET].mean()), 4),
        "imbalance_ratio": round(float((len(lab) - pos) / pos), 2),
        "rate_by_split": {
            "train": round(float(train[TARGET].mean()), 4),
            "validation": round(float(val[TARGET].mean()), 4),
            "test": round(float(test[TARGET].mean()), 4),
        },
    }

    # ---- missingness ------------------------------------------------------
    r["missingness"] = {
        "raw_null_cells": int(raw.isna().sum().sum()),
        "panel_gap_cells": int(panel["is_gap"].sum()),
        "panel_gap_rate": round(float(panel["is_gap"].mean()), 4),
        "interpolated_rows": int(panel["price_was_interpolated"].sum()),
        "gap_rate_by_state": {
            k: round(float(v), 4)
            for k, v in panel.groupby("state")["is_gap"].mean().items()
        },
        "littles_mcar": D.littles_mcar_test(train, feats),
        "cramers_v": D.missingness_association(panel),
    }

    # ---- distribution and outliers ---------------------------------------
    fence = D.iqr_fence(train["price"])
    above = train["price"] > fence["upper_fence"]
    r["outliers"] = {
        "price_shape": D.distribution_shape(train["price"], "price"),
        "price_fence": fence,
        "flagged_price_spike_rate": round(float(train.loc[above, TARGET].mean()), 4),
        "base_spike_rate": round(float(train[TARGET].mean()), 4),
        "flagged_preceding_spike": int(train.loc[above, TARGET].sum()),
        "ratio_features": {
            c: {**D.iqr_fence(train[c]),
                "skewness": D.distribution_shape(train[c], c)["skewness"],
                "kurtosis": D.distribution_shape(train[c], c)["kurtosis"]}
            for c in ratios
        },
    }

    # ---- transformation ---------------------------------------------------
    r["transformation"] = D.transformation_effect(train["price"])

    # ---- collinearity -----------------------------------------------------
    vif_full = D.variance_inflation(train, feats)
    over = vif_full[vif_full["VIF"] > 10]["feature"].tolist()
    reduced = [c for c in feats if c not in over]
    vif_red = D.variance_inflation(train, reduced) if over else vif_full
    r["collinearity"] = {
        "pairs_over_threshold": D.collinear_pairs(train, feats, threshold=0.85),
        "vif_full": vif_full.to_dict("records"),
        "vif_over_10": over,
        "vif_max_before": round(float(vif_full["VIF"].max()), 2),
        "vif_max_after": round(float(vif_red["VIF"].max()), 2),
        "features_after_reduction": reduced,
    }

    # ---- target association ----------------------------------------------
    r["target_association"] = D.target_association(train, feats, TARGET).to_dict("records")

    # ---- dimensionality ---------------------------------------------------
    r["pca"] = D.pca_variance(train, feats)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2, default=float)
    print(f"wrote {out_path}")
    return r


if __name__ == "__main__":
    main()
