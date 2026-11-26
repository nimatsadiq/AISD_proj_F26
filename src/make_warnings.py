"""Generate the warning feed the dashboard reads.

This is the operating loop, run for real. For each month in the display
window the model is refitted on history only, recalibrated on a held-back
slice, and used to score every state-and-food pair for the month ahead. The
highest-ranked pairs are written out with the reasons behind them and, where
the outcome is already known, whether the price did in fact rise.

The newest month in the file is a special case and the reason this system
can call itself an early warning. Its outcome is not known: the label for
month *t* is computed from the price at *t+1*, so the last month in any
download never has one. An earlier version of this script dropped that row,
which meant every warning it published was for a month whose answer was
already sitting in the same file. It now scores that row too and publishes a
warning for the month after it, marked `pending` with a null outcome. The
scored months stay in the feed as the evidence that the pending one is worth
reading.

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
well it worked. The pending month is excluded from every accuracy figure,
for the obvious reason.

    python -m src.make_warnings
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import calibrate
from .explain import compute_shap, warning_reasons
from .pipeline_builder import TARGET, build_logistic_pipeline, build_panel
from .transformers import FUEL_COMMODITIES, IN_SCOPE_STATES

RAW = "data/raw/wfp_food_prices_nga.csv"
OUT = "data/processed/dashboard_warnings.json"

WINDOW = 48      # months of training history, the policy selected in rolling.py
PURGE = 1        # months dropped at the boundary; equals the label's forward reach
TOP_N = 8        # warnings kept per month; the dashboard slider shows 3 to 8
MIN_TRAIN = 400


def _next_month(m):
    """The month a warning is about, given the month whose prices it read."""
    return (pd.Timestamp(m) + pd.DateOffset(months=1)).strftime("%Y-%m")


def _fit_for(df, month, window=WINDOW, min_train=MIN_TRAIN, purge=PURGE):
    """Fit and calibrate on history before the purge boundary, or return None.

    The purge matches the one in `rolling.py`, and for the same reason: the
    label at month *t* is computed from the price at *t+1*, so the row at
    `month - 1` carries the outcome of the month being scored. Production has
    to use the same boundary the evaluation measured, or the deployed system
    is not the system the numbers describe.
    """
    purge_cut = month - pd.DateOffset(months=purge) if purge else month
    history = df[df["month"] < purge_cut]
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


def _rank_and_explain(model, uncalibrated, fit_part, current, top_n):
    """Score one month's rows and return the top-ranked ones with reasons."""
    p = model.predict_proba(current)[:, 1]
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
    return detail, rows, p


def _coverage(panel, raw, month, lookback=6):
    """How complete the scoring month looks beside the months before it.

    WFP collects through the month, so the newest month in a download is
    usually partial: markets that have not reported yet are simply absent.
    That absence is invisible by the time it reaches the model. `PanelBuilder`
    collapses the markets in a state to their median, so a state that normally
    reports from six markets and this month reported from three still produces
    one perfectly ordinary-looking number per commodity, and nothing in that
    number records which markets stood behind it.

    So this counts two different things. Pairs come from the panel and catch a
    commodity dropping out entirely. Markets come from the raw records, which
    is the only place the information still exists, and catch the case that
    actually happens: the same commodities, priced in half as many places.
    """
    months = sorted(panel["month"].unique())
    if month not in months:
        return None
    idx = months.index(month)
    prior = months[max(0, idx - lookback):idx]
    if not prior:
        return None

    pairs_now = int((panel["month"] == month).sum())
    pairs_norm = float(np.median([int((panel["month"] == m).sum()) for m in prior]))
    out = {
        "pairs": pairs_now,
        "recent_median_pairs": pairs_norm,
        # Under four fifths of the recent norm counts as partial. The warning
        # still publishes: a thin month is a reason to read the list with
        # care, not a reason to leave a buyer with nothing.
        "partial": bool(pairs_norm and pairs_now < 0.8 * pairs_norm),
    }
    out["share_of_normal"] = (round(pairs_now / pairs_norm, 3)
                              if pairs_norm else None)

    if raw is None:
        return out

    # The same filters PanelBuilder applies, so the count describes the
    # records that actually reach the model and not the whole country.
    r = raw.copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r[(r["pricetype"] == "Retail")
          & (r["admin1"].isin(IN_SCOPE_STATES))
          & (~r["commodity"].isin(FUEL_COMMODITIES))]
    per_month = r.groupby(pd.Grouper(key="date", freq="MS"))["market"].nunique()

    now_ts = pd.Timestamp(month)
    if now_ts not in per_month.index:
        return out
    prior_counts = [int(per_month[pd.Timestamp(m)]) for m in prior
                    if pd.Timestamp(m) in per_month.index]
    if not prior_counts:
        return out

    markets_now = int(per_month[now_ts])
    markets_norm = float(np.median(prior_counts))
    out["markets"] = markets_now
    out["recent_median_markets"] = markets_norm
    out["market_share_of_normal"] = (round(markets_now / markets_norm, 3)
                                     if markets_norm else None)
    if markets_norm and markets_now < 0.8 * markets_norm:
        out["partial"] = True
        out["share_of_normal"] = out["market_share_of_normal"]
    return out


def _live_month(panel):
    """The newest month that has features but no known outcome, or None.

    This is the row the system exists to score. If it comes back None the
    download ends on a month that can already be marked right or wrong, and
    there is nothing forward-looking to publish.
    """
    last = panel["month"].max()
    if pd.isna(last):
        return None
    if panel.loc[panel["month"] == last, TARGET].notna().any():
        return None
    return last


def _score_live(labelled, panel, month, top_n=TOP_N):
    """Warnings for a month that has not happened yet.

    Fitting runs through the same `_fit_for` the backtest uses, purge and
    all, so the model issuing the live warning is the model the measured
    numbers describe. The only difference is in what comes back: the outcome
    has not happened, so `actual` is null and `status` is "pending".
    """
    model, parts = _fit_for(labelled, month)
    if model is None:
        return []
    uncalibrated, fit_part = parts

    current = panel[panel["month"] == month]
    if current.empty:
        return []

    detail, rows, _ = _rank_and_explain(model, uncalibrated, fit_part,
                                        current, top_n)
    predicts = _next_month(month)
    out = []
    for rank, d in enumerate(detail, start=1):
        d["actual"] = None
        d["status"] = "pending"
        d["predicts"] = predicts
        d["rank"] = rank
        d["pairs_this_month"] = int(len(current))
        d["spikes_this_month"] = None
        out.append(d)
    return out


def build_warnings(panel, months_shown=12, top_n=TOP_N, raw=None):
    """Walk forward over the display window, then score the pending month.

    `raw` is optional and used only to measure how many markets reported in
    the pending month, which the panel can no longer tell you.
    """
    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"])
    df = panel.dropna(subset=[TARGET]).copy()
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

        scored_rows += len(current)
        scored_spikes += int(current[TARGET].sum())

        detail, rows, _ = _rank_and_explain(model, uncalibrated, fit_part,
                                            current, top_n)
        predicts = _next_month(m)

        for rank, (d, (_, r)) in enumerate(zip(detail, rows.iterrows()), start=1):
            d["actual"] = int(r[TARGET])
            d["status"] = "scored"
            d["predicts"] = predicts
            d["rank"] = rank
            d["pairs_this_month"] = int(len(current))
            d["spikes_this_month"] = int(current[TARGET].sum())
            warnings.append(d)

    # The pending month, which is the whole point of the system.
    live = _live_month(panel)
    live_meta = None
    if live is not None:
        live_rows = _score_live(df, panel, live, top_n)
        if live_rows:
            warnings.extend(live_rows)
            live_meta = {
                "scored_from": pd.Timestamp(live).strftime("%Y-%m"),
                "predicts": _next_month(live),
                "warnings": len(live_rows),
                "pairs_scored": live_rows[0]["pairs_this_month"],
                "coverage": _coverage(panel, raw, live),
            }

    # Accuracy is measured over the scored months only. A pending month has
    # no outcome, so including it would either crash the sum or, worse,
    # silently count an unknown as a miss.
    by_month = {}
    for w in warnings:
        if w.get("actual") is None:
            continue
        by_month.setdefault(w["month"], []).append(w)

    meta = {
        "months": len(by_month),
        "pairs_scored": scored_rows,
        "spikes": scored_spikes,
        # Share of ALL pairs scored that went on to rise. This is the honest
        # comparator for the shortlist: the chance of being right by picking
        # a food at random.
        "base_rate": round(scored_spikes / scored_rows, 4) if scored_rows else None,
        "train_window_months": WINDOW,
        "purge_months": PURGE,
        "retrained": "monthly, on history only",
        # Null when the download ends on a month whose outcome is already
        # known, which means there is nothing to warn about yet.
        "live": live_meta,
    }
    # Precision of the shortlist at each length the dashboard offers, so the
    # slider reports measured numbers rather than recomputing from a slice.
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
    warnings, meta = build_warnings(panel, months_shown=months_shown, raw=raw)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"meta": meta, "warnings": warnings}, indent=1))

    print(f"wrote {len(warnings)} warnings over {meta['months']} scored months "
          f"-> {out_path}")
    print(f"base rate {meta['base_rate']:.1%} over {meta['pairs_scored']} pairs scored")
    for k, v in meta["precision_at_k"].items():
        lift = v / meta["base_rate"] if meta["base_rate"] else float("nan")
        print(f"  top {k}: {v:.1%} correct  ({lift:.1f}x base rate)")

    live = meta.get("live")
    if live:
        print(f"pending: {live['warnings']} warnings for {live['predicts']}, "
              f"scored from {live['scored_from']} prices "
              f"({live['pairs_scored']} pairs)")
        cov = live.get("coverage") or {}
        if cov.get("markets") is not None:
            print(f"  {cov['markets']} markets reported in "
                  f"{live['scored_from']}, against a recent median of "
                  f"{cov['recent_median_markets']:.0f}")
        if cov.get("partial"):
            print(f"  NOTE: {live['scored_from']} is only "
                  f"{cov['share_of_normal']:.0%} of its usual size, so some "
                  f"markets have not reported yet. The warning stands, but it "
                  f"rests on a thinner month than the ones it was measured on.")
    else:
        print("pending: none. The file ends on a month that already has an "
              "outcome, so there is nothing forward-looking to publish.")
    return meta


if __name__ == "__main__":
    main()
