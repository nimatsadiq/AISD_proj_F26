"""Generate the warning feed the dashboard reads.

This is the operating loop, run for real. For each month in the display
window the model is refitted on history only, recalibrated on a held-back
slice, and used to score every state-and-food pair for the month ahead. The
highest-ranked pairs are written out with the reasons behind them and, where
the outcome is already known, whether the price did in fact rise.

Nothing here sees its own future. The model scoring month *t* is fitted on
rows strictly before *t*, which is the same discipline `src/rolling.py`
uses to measure performance. The difference is only in what is kept: that
module keeps metrics, this one keeps the warnings themselves.

The file it writes, `data/processed/dashboard_warnings.json`, carries a
`meta` block alongside the warnings. The base rate in that block is counted
over *every* pair scored in the window, not over the shortlist, because the
shortlist is a biased sample by construction: it is the rows the model was
most worried about. Dividing shortlist accuracy by shortlist base rate would
compare the model against itself and report a lift near one no matter how
well it worked.

    python -m src.make_warnings
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import calibrate
from .explain import compute_shap, warning_reasons
from .pipeline_builder import TARGET, build_logistic_pipeline, build_panel

RAW = "data/raw/wfp_food_prices_nga.csv"
OUT = "data/processed/dashboard_warnings.json"

WINDOW = 48      # months of training history, the policy selected in rolling.py
TOP_N = 8        # warnings kept per month; the dashboard slider shows 3 to 8
MIN_TRAIN = 400


def _fit_for(df, month, window=WINDOW, min_train=MIN_TRAIN):
    """Fit and calibrate on history strictly before `month`, or return None."""
    history = df[df["month"] < month]
    cutoff = month - pd.DateOffset(months=window)
    history = history[history["month"] >= cutoff]
    if len(history) < min_train or history[TARGET].sum() < 20:
        return None, None

    cal_cut = month - pd.DateOffset(months=6)
    fit_part = history[history["month"] < cal_cut]
    cal_part = history[history["month"] >= cal_cut]
    if len(fit_part) < min_train or fit_part[TARGET].sum() < 20:
        fit_part, cal_part = history, None

    model = build_logistic_pipeline().fit(fit_part, fit_part[TARGET].astype(int))
    uncalibrated = model

    if cal_part is not None and cal_part[TARGET].nunique() > 1:
        try:
            model = calibrate(model, cal_part, cal_part[TARGET].astype(int),
                              method="sigmoid")
        except Exception:
            pass
    return model, (uncalibrated, fit_part)


def build_warnings(panel, months_shown=12, top_n=TOP_N):
    """Walk forward over the display window, keeping the top-ranked warnings."""
    df = panel.dropna(subset=[TARGET]).copy()
    df["month"] = pd.to_datetime(df["month"])
    all_months = sorted(df["month"].unique())
    window_months = all_months[-months_shown:]

    warnings, scored_rows, scored_spikes = [], 0, 0

    for m in window_months:
        model, parts = _fit_for(df, m)
        if model is None:
            continue
        uncalibrated, fit_part = parts

        current = df[df["month"] == m]
        if current.empty:
            continue

        p = model.predict_proba(current)[:, 1]
        scored_rows += len(current)
        scored_spikes += int(current[TARGET].sum())

        order = np.argsort(-p)[:min(top_n, len(current))]
        rows = current.iloc[order]

        # Reasons come from the uncalibrated model: calibration rescales the
        # probability but does not change which features drove the ranking,
        # and the linear explainer needs the linear model underneath.
        try:
            values, _, names, _ = compute_shap(uncalibrated, fit_part, rows)
            detail = warning_reasons(values, names, rows, p[order])
        except Exception:
            detail = [{"state": r["state"], "commodity": r["commodity"],
                       "month": pd.Timestamp(r["month"]).strftime("%Y-%m"),
                       "score": round(float(s), 3),
                       "reasons": ["reasons unavailable"]}
                      for (_, r), s in zip(rows.iterrows(), p[order])]

        for rank, (d, (_, r)) in enumerate(zip(detail, rows.iterrows()), start=1):
            d["actual"] = int(r[TARGET])
            d["rank"] = rank
            d["pairs_this_month"] = int(len(current))
            d["spikes_this_month"] = int(current[TARGET].sum())
            warnings.append(d)

    meta = {
        "months": len(set(w["month"] for w in warnings)),
        "pairs_scored": scored_rows,
        "spikes": scored_spikes,
        # Share of ALL pairs scored that went on to rise. This is the honest
        # comparator for the shortlist: the chance of being right by picking
        # a food at random.
        "base_rate": round(scored_spikes / scored_rows, 4) if scored_rows else None,
        "train_window_months": WINDOW,
        "retrained": "monthly, on history only",
    }
    # Precision of the shortlist at each length the dashboard offers, so the
    # slider reports measured numbers rather than recomputing from a slice.
    by_month = {}
    for w in warnings:
        by_month.setdefault(w["month"], []).append(w)
    meta["precision_at_k"] = {}
    for k in range(3, top_n + 1):
        shown = hit = 0
        for rows_ in by_month.values():
            top = sorted(rows_, key=lambda x: -x["score"])[:k]
            shown += len(top)
            hit += sum(x["actual"] for x in top)
        meta["precision_at_k"][str(k)] = round(hit / shown, 4) if shown else None

    return warnings, meta


def main(raw_path=RAW, out_path=OUT, months_shown=12):
    raw = pd.read_csv(raw_path)
    panel = build_panel(raw)
    warnings, meta = build_warnings(panel, months_shown=months_shown)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"meta": meta, "warnings": warnings}, indent=1))

    print(f"wrote {len(warnings)} warnings over {meta['months']} months -> {out_path}")
    print(f"base rate {meta['base_rate']:.1%} over {meta['pairs_scored']} pairs scored")
    for k, v in meta["precision_at_k"].items():
        lift = v / meta["base_rate"] if meta["base_rate"] else float("nan")
        print(f"  top {k}: {v:.1%} correct  ({lift:.1f}x base rate)")
    return meta


if __name__ == "__main__":
    main()
