"""Month-clustered uncertainty, and where the performance actually comes from.

The reviewer's sharpest statistical point: the 909 test rows are not 909
independent trials. They are 107 state-and-commodity series watched over the
same 18 months, and a fuel or currency shock moves all of them together. The
effective sample size is closer to the number of *months* than the number of
rows, so an ordinary bootstrap over rows would report a confidence interval
several times too narrow.

So this module resamples **whole months, with replacement**, and recomputes
the metric on each resampled set of months. A month is the unit that is
plausibly independent; everything inside it is not.

It also breaks performance down by commodity and by state, because a pooled
number can hide a model that works on storable grain and fails on tomatoes.

    python -m src.uncertainty
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RNG = np.random.default_rng(20260315)
OUT = "data/processed/uncertainty.json"


# ---------------------------------------------------------------- metrics --

def pr_auc(df):
    if df["y"].nunique() < 2:
        return np.nan
    return float(average_precision_score(df["y"], df["p"]))


def top_k_precision(df, k=5):
    """Share of each month's k highest-scored pairs that did spike."""
    hits = shown = 0
    for _, g in df.groupby("month"):
        take = g.nlargest(min(k, len(g)), "p")
        hits += int(take["y"].sum())
        shown += len(take)
    return float(hits / shown) if shown else np.nan


def base_rate(df):
    return float(df["y"].mean()) if len(df) else np.nan


def lift(df, k=5):
    b = base_rate(df)
    return float(top_k_precision(df, k) / b) if b else np.nan


METRICS = {
    "pr_auc": pr_auc,
    "top3_precision": lambda d: top_k_precision(d, 3),
    "top5_precision": lambda d: top_k_precision(d, 5),
    "top8_precision": lambda d: top_k_precision(d, 8),
    "base_rate": base_rate,
    "lift_top5": lambda d: lift(d, 5),
}


# ------------------------------------------------------------- bootstrap ---

def month_block_bootstrap(preds, metric, n_boot=2000, alpha=0.05):
    """Resample MONTHS with replacement; return the point estimate and CI.

    Resampling months rather than rows is the whole point: it keeps each
    month's rows together, so the interval reflects how much the answer
    would move if we had watched a different 18 months, not how much it
    would move if we had watched a few more commodities in the same ones.
    """
    months = preds["month"].unique()
    by_month = {m: g for m, g in preds.groupby("month")}
    point = metric(preds)

    draws = []
    for _ in range(n_boot):
        pick = RNG.choice(months, size=len(months), replace=True)
        sample = pd.concat([by_month[m] for m in pick], ignore_index=True)
        v = metric(sample)
        if not np.isnan(v):
            draws.append(v)

    if not draws:
        return {"point": point, "lo": None, "hi": None, "n_months": len(months)}
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": round(float(point), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "n_months": int(len(months)), "n_boot": len(draws)}


def row_bootstrap(preds, metric, n_boot=2000, alpha=0.05):
    """The naive version, resampling rows, kept only to show the contrast."""
    draws = []
    n = len(preds)
    for _ in range(n_boot):
        s = preds.iloc[RNG.integers(0, n, n)]
        v = metric(s)
        if not np.isnan(v):
            draws.append(v)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4)}


# ---------------------------------------------------------- where it works --

def by_group(preds, key, k=5, min_rows=40):
    """Per-commodity or per-state performance, so pooled numbers cannot hide.

    Top-k precision is a *monthly shortlist* measure and does not survive
    being sliced by commodity, so within a group this reports PR-AUC against
    that group's own base rate, which is the fair comparison.
    """
    rows = []
    for name, g in preds.groupby(key):
        if len(g) < min_rows or g["y"].nunique() < 2:
            continue
        rows.append({key: name, "n": int(len(g)),
                     "base_rate": round(base_rate(g), 4),
                     "pr_auc": round(pr_auc(g), 4),
                     "lift": round(pr_auc(g) / base_rate(g), 2)
                     if base_rate(g) else None})
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def main(raw_path="data/raw/wfp_food_prices_nga.csv", out_path=OUT):
    from .pipeline_builder import build_panel
    from .rolling import rolling_evaluation

    raw = pd.read_csv(raw_path, low_memory=False)
    panel = build_panel(raw)
    _, _, preds = rolling_evaluation(panel, start="2023-01", window=48,
                                     policy="rolling", keep_predictions=True)

    report = {"n_rows": int(len(preds)),
              "n_months": int(preds["month"].nunique()),
              "n_series": int(preds.groupby(["state", "commodity"]).ngroups)}

    print(f"{report['n_rows']} rows, {report['n_months']} months, "
          f"{report['n_series']} series")
    print("\nMonth-clustered 95% intervals (months resampled, not rows):")
    report["metrics"] = {}
    for name, fn in METRICS.items():
        ci = month_block_bootstrap(preds, fn)
        report["metrics"][name] = ci
        print(f"  {name:16s} {ci['point']:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]")

    naive = row_bootstrap(preds, METRICS["top5_precision"])
    clustered = report["metrics"]["top5_precision"]
    report["naive_row_bootstrap_top5"] = naive
    wn = naive["hi"] - naive["lo"]
    wc = clustered["hi"] - clustered["lo"]
    report["ci_width_ratio"] = round(wc / wn, 2) if wn else None
    print(f"\n  Resampling rows instead would give [{naive['lo']:.3f}, {naive['hi']:.3f}],"
          f" an interval {wc / wn:.1f}x narrower than honest.")

    for key in ("commodity", "state"):
        tab = by_group(preds, key)
        report[f"by_{key}"] = tab.to_dict("records")
        print(f"\nBy {key} (PR-AUC against that group's own base rate):")
        print(tab.to_string(index=False))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_path}")
    return report


if __name__ == "__main__":
    main()
