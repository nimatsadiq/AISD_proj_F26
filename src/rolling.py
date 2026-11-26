"""Rolling-window retraining: the structural-break mitigation, implemented.

The design document committed to retraining on a moving window rather than
fitting once on all history, on the grounds that the June 2023 fuel subsidy
removal and currency float broke the relationship between past and future
prices. That commitment was never implemented, so this module does it and
measures whether it helps.

The operating pattern being simulated is the real one. Each month an analyst
has the data published up to that point, refits on a recent window, and issues
warnings for the month ahead. Nothing here sees the future: the model scoring
month *t+1* is fitted only on months at or before *t*.

Two window policies are compared:

* **expanding** - train on everything available so far, the conventional
  default, which assumes the past stays relevant.
* **rolling** - train on the most recent `window` months only, deliberately
  discarding older data on the argument that it describes a different economy.

If the regime-break reasoning is right, rolling should hold up better after
mid-2023. If it is wrong, more data should win, and that is worth knowing too.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .calibration import calibrate
from .pipeline_builder import TARGET, build_logistic_pipeline, temporal_split


def _precision_at_recall(y, p, target=0.80):
    if y.sum() == 0 or len(np.unique(y)) < 2:
        return np.nan
    pr, rc, _ = precision_recall_curve(y, p)
    feasible = pr[rc >= target]
    return float(feasible.max()) if len(feasible) else np.nan


def rolling_evaluation(panel, start="2023-01", window=36, policy="rolling",
                       calibrate_each=True, min_train=400, keep_predictions=False,
                       purge=1, pipeline_factory=None):
    """Walk forward month by month, refitting as an operator would.

    For each evaluation month *m*: fit on rows strictly before the purge
    boundary, optionally recalibrate on the most recent held-back slice, then
    score month *m*.

    **The purge window.** The label for month *t* is built from the price in
    month *t+1*. So the row at month *m-1* carries a label computed from the
    price in month *m*, which is the month being scored. Training on it would
    let the outcome of the test month reach the model through the label, even
    though no feature ever touches a future price. `purge=1` drops that row:
    history ends at *m-2*, and the gap is exactly as wide as the label's
    forward reach. Set `purge=0` to reproduce the un-purged numbers.

    `policy='rolling'` keeps only the last `window` months before *m*;
    `policy='expanding'` keeps everything. Returns one row per month plus the
    pooled predictions, so performance can be read either way.
    """
    df = panel.dropna(subset=[TARGET]).copy()
    df["month"] = pd.to_datetime(df["month"])
    months = sorted(df.loc[df["month"] >= start, "month"].unique())

    rows, pooled_y, pooled_p, kept = [], [], [], []

    for m in months:
        # Purge the boundary: the label at m-1 was computed from month m's
        # price, so training on it leaks the test month's outcome.
        purge_cut = m - pd.DateOffset(months=purge) if purge else m
        history = df[df["month"] < purge_cut]
        if policy == "rolling":
            cutoff = m - pd.DateOffset(months=window)
            history = history[history["month"] >= cutoff]
        if len(history) < min_train or history[TARGET].sum() < 20:
            continue

        # Hold back the most recent 6 months of history to calibrate on, so the
        # calibration mapping is never fitted on the rows the model trained on.
        cal_cut = m - pd.DateOffset(months=6)
        fit_part = history[history["month"] < cal_cut]
        cal_part = history[history["month"] >= cal_cut]
        if len(fit_part) < min_train or fit_part[TARGET].sum() < 20:
            fit_part, cal_part = history, None

        y_fit = fit_part[TARGET].astype(int)
        # `pipeline_factory` exists so an ablation can remove a feature group
        # and still run through this exact loop. Reimplementing the walk-forward
        # inside the ablation script would let the two drift apart, and then the
        # ablation would be measuring a different system from the one reported.
        build = pipeline_factory or build_logistic_pipeline
        model = build().fit(fit_part, y_fit)

        if calibrate_each and cal_part is not None and cal_part[TARGET].nunique() > 1:
            try:
                model = calibrate(model, cal_part, cal_part[TARGET].astype(int),
                                  method="sigmoid")
            except Exception:
                pass

        current = df[df["month"] == m]
        if current.empty:
            continue
        y_true = current[TARGET].astype(int).values
        p = model.predict_proba(current)[:, 1]

        pooled_y.append(y_true)
        pooled_p.append(p)
        if keep_predictions:
            kept.append(pd.DataFrame({
                "month": pd.Timestamp(m).strftime("%Y-%m"),
                "state": current["state"].values,
                "commodity": current["commodity"].values,
                "y": y_true, "p": p}))

        k = min(5, len(current))
        top = np.argsort(-p)[:k]
        rows.append({
            "month": pd.Timestamp(m).strftime("%Y-%m"),
            "train_rows": int(len(fit_part)),
            "train_from": fit_part["month"].min().strftime("%Y-%m"),
            "n": int(len(current)),
            "spikes": int(y_true.sum()),
            "top5_hits": int(y_true[top].sum()),
            "pr_auc": round(float(average_precision_score(y_true, p)), 4)
            if y_true.sum() else np.nan,
        })

    y_all = np.concatenate(pooled_y) if pooled_y else np.array([])
    p_all = np.concatenate(pooled_p) if pooled_p else np.array([])

    summary = {
        "policy": policy,
        "window_months": window if policy == "rolling" else None,
        "months_evaluated": len(rows),
        "total_rows": int(len(y_all)),
        "total_spikes": int(y_all.sum()) if len(y_all) else 0,
        "pr_auc_pooled": round(float(average_precision_score(y_all, p_all)), 4)
        if len(y_all) and y_all.sum() else None,
        "precision_at_recall_80": round(_precision_at_recall(y_all, p_all), 4)
        if len(y_all) and y_all.sum() else None,
        "top5_precision": round(
            sum(r["top5_hits"] for r in rows) / max(sum(min(5, r["n"]) for r in rows), 1), 4),
        "top5_recall": round(
            sum(r["top5_hits"] for r in rows) / max(sum(r["spikes"] for r in rows), 1), 4),
    }
    if keep_predictions:
        preds = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
        return summary, pd.DataFrame(rows), preds
    return summary, pd.DataFrame(rows)


def compare_policies(panel, start="2023-01", windows=(24, 36, 48)):
    """Rolling windows against the expanding baseline, same evaluation months."""
    results = []
    s, _ = rolling_evaluation(panel, start=start, policy="expanding")
    results.append(s)
    for w in windows:
        s, _ = rolling_evaluation(panel, start=start, policy="rolling", window=w)
        results.append(s)
    return pd.DataFrame(results)
