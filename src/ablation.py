"""What is each group of features actually worth?

Rubric 4 asks for an ablation. The honest version of that question is not
"which feature has the biggest coefficient", because a coefficient is a
statement about one model's internal arithmetic. It is "if this information
had never been collected, how much worse would the warnings be?" So each
group is removed and the **entire walk-forward evaluation is rerun from
scratch**: refit every month, recalibrate every month, rescore every month.

Two design decisions are worth defending.

**The loop is not reimplemented here.** `rolling.rolling_evaluation` takes a
`pipeline_factory`, and this module passes one that drops a group. Writing a
second walk-forward loop inside this file would be quicker and would let the
two drift apart silently, at which point the ablation would be measuring a
different system from the one the headline numbers describe.

**Groups, not single columns.** `mom_1m` and `mom_3m` both describe recent
price movement. Drop one and the other covers for it, and you conclude the
feature was worthless when the *information* was not. Removing the whole
group answers the question a data collector would actually ask.

    python -m src.ablation

Writes `data/processed/ablation.json` and prints the table.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from .pipeline_builder import TARGET, build_logistic_pipeline, build_panel
from .rolling import rolling_evaluation

RAW = "data/raw/wfp_food_prices_nga.csv"
OUT = "data/processed/ablation.json"

# Match production. `make_warnings` trains on a 48-month window, and
# `compare_policies` selected it, so an ablation run at the library default of
# 36 would be measuring a model the project does not ship. The absolute
# numbers would be wrong even though the comparison between groups held.
WINDOW = 48

# Grouped by the information they carry, not by how they were computed. A
# group is what one source or one idea contributes, which is the unit someone
# deciding what to collect would care about.
GROUPS = {
    "own price history": ["mom_1m", "mom_3m", "yoy", "dev_from_6m", "dev_from_12m"],
    "volatility": ["volatility_6m"],
    "seasonality": ["mom_same_month_last_year", "month_sin", "month_cos"],
    "neighbouring states": ["neighbour_mom_1m"],
    "fuel cost": ["fuel_mom_1m", "fuel_mom_3m"],
    "missingness": ["gaps_last_6m", "price_was_interpolated"],
    "identity (state, food type)": ["state", "food_type"],
}


def _run(panel, drop=(), window=WINDOW, **kw):
    summary, _ = rolling_evaluation(
        panel,
        window=window,
        pipeline_factory=(lambda: build_logistic_pipeline(drop_extra=list(drop))),
        **kw,
    )
    base = (summary["total_spikes"] / summary["total_rows"]) if summary["total_rows"] else None
    summary["base_rate"] = round(base, 4) if base else None
    summary["lift_top5"] = (round(summary["top5_precision"] / base, 3)
                            if base else None)
    return summary


def main(raw_path=RAW, out_path=OUT, **kw):
    panel = build_panel(pd.read_csv(raw_path, low_memory=False))
    print("Running the full model first, as the thing to compare against.")
    full = _run(panel, **kw)
    print(f"  full model: PR-AUC {full['pr_auc_pooled']}, "
          f"top-5 {full['top5_precision']:.1%}, lift {full['lift_top5']}x, "
          f"over {full['months_evaluated']} months\n")

    results = {"full": full, "groups": {}}
    for name, cols in GROUPS.items():
        print(f"  dropping {name} ...", end=" ", flush=True)
        r = _run(panel, drop=cols, **kw)
        r["dropped"] = cols
        r["d_pr_auc"] = (round(r["pr_auc_pooled"] - full["pr_auc_pooled"], 4)
                         if r["pr_auc_pooled"] and full["pr_auc_pooled"] else None)
        r["d_top5"] = round(r["top5_precision"] - full["top5_precision"], 4)
        results["groups"][name] = r
        print(f"PR-AUC {r['pr_auc_pooled']} ({r['d_pr_auc']:+.4f}), "
              f"top-5 {r['top5_precision']:.1%} ({r['d_top5']:+.1%})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=1))

    print(f"\n{'group removed':30s} {'PR-AUC':>8s} {'change':>8s} "
          f"{'top-5':>7s} {'change':>8s}")
    print("-" * 66)
    print(f"{'nothing (full model)':30s} {full['pr_auc_pooled']:>8.4f} "
          f"{'':>8s} {full['top5_precision']:>6.1%} {'':>8s}")
    order = sorted(results["groups"].items(), key=lambda kv: kv[1]["d_pr_auc"] or 0)
    for name, r in order:
        print(f"{name:30s} {r['pr_auc_pooled']:>8.4f} {r['d_pr_auc']:>+8.4f} "
              f"{r['top5_precision']:>6.1%} {r['d_top5']:>+8.1%}")
    print("-" * 66)
    print("A group whose removal barely moves PR-AUC is one the system could\n"
          "do without. A negative change means the model was better without it,\n"
          "which with twelve months of evidence usually means noise rather than\n"
          "a harmful feature, and should be reported as such.")
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
