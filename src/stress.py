"""How does the system fail when its input is wrong?

Rubric 4 asks for stress tests. An accuracy number says how the system does
when everything works. This says what it does when something breaks, which is
the question that matters once it is deployed, because in production the
input is wrong eventually and nobody is watching when it happens.

Each test corrupts the RAW file and rebuilds everything from there, rather
than poking the finished panel. That is where the damage actually arrives:
WFP publishes a thin month, the naira moves, a market stops reporting. By
the time the panel is built the damage is already baked in and invisible.

What is measured is not accuracy. The corrupted months have no known outcome
to score against, so the question is *stability*: how much does the shortlist
move, and does the system notice anything is wrong? A system that silently
returns a confident, completely different answer is worse than one that
returns the same answer, and far worse than one that says it is unsure.

    python -m src.stress

Writes `data/processed/stress.json` and prints the table.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .make_warnings import _live_month, _score_live, _coverage, TOP_N
from .pipeline_builder import TARGET, build_panel

RAW = "data/raw/wfp_food_prices_nga.csv"
OUT = "data/processed/stress.json"


def _shortlist(raw, top_n=TOP_N):
    """Score the newest month of a raw frame and return its shortlist."""
    panel = build_panel(raw)
    panel["month"] = pd.to_datetime(panel["month"])
    labelled = panel.dropna(subset=[TARGET])
    live = _live_month(panel)
    if live is None:
        return None, None, None
    rows = _score_live(labelled, panel, live, top_n)
    cov = _coverage(panel, raw, live)
    return rows, cov, pd.Timestamp(live).strftime("%Y-%m")


def _overlap(a, b, k=5):
    """How many of the baseline's top k survive in the corrupted top k."""
    if not a or not b:
        return None
    A = {(r["state"], r["commodity"]) for r in sorted(a, key=lambda r: r["rank"])[:k]}
    B = {(r["state"], r["commodity"]) for r in sorted(b, key=lambda r: r["rank"])[:k]}
    return len(A & B)


# ---------------------------------------------------------------- the faults

def drop_half_the_markets(raw, month, rng):
    """Half the markets go quiet. This happens every month, to a degree."""
    out = raw.copy()
    m = out["date"].dt.to_period("M") == month
    mk = sorted(out.loc[m, "market"].unique())
    if len(mk) < 2:
        return out
    keep = set(rng.choice(mk, size=max(1, len(mk) // 2), replace=False))
    return out[~m | out["market"].isin(keep)].copy()


def shift_price_level(raw, month, rng, factor=1.20):
    """Every price 20% higher at once: a devaluation, not a food story.

    This is the test the system should arguably fail. A uniform shift moves
    every price above its own trailing mean, so a model built on relative
    movement has no way to tell it from genuine, simultaneous scarcity.
    """
    out = raw.copy()
    m = out["date"].dt.to_period("M") == month
    out.loc[m, "price"] = out.loc[m, "price"] * factor
    return out


def drop_a_commodity(raw, month, rng):
    """One staple stops being collected."""
    out = raw.copy()
    m = out["date"].dt.to_period("M") == month
    present = sorted(out.loc[m, "commodity"].unique())
    if not present:
        return out
    victim = rng.choice(present)
    return out[~(m & (out["commodity"] == victim))].copy()


def freeze_the_month(raw, month, rng):
    """Nothing moved at all: the newest month repeats the one before it.

    A stuck upstream feed looks exactly like this, and it is the failure most
    likely to go unnoticed, because the file still arrives and still parses.
    """
    out = raw.copy()
    prev = month - 1
    cur = out["date"].dt.to_period("M") == month
    src = out[out["date"].dt.to_period("M") == prev].copy()
    if src.empty:
        return out
    src["date"] = src["date"] + pd.offsets.MonthBegin(1)
    src["date"] = src["date"] + pd.Timedelta(days=14)
    return pd.concat([out[~cur], src], ignore_index=True)


def add_a_wild_outlier(raw, month, rng, factor=12.0):
    """One price is entered twelve times too high. A typo, or a unit change."""
    out = raw.copy()
    m = out["date"].dt.to_period("M") == month
    idx = out.index[m]
    if not len(idx):
        return out
    out.loc[rng.choice(idx), "price"] *= factor
    return out


FAULTS = {
    "half the markets stop reporting": drop_half_the_markets,
    "all prices 20% higher (devaluation)": shift_price_level,
    "one commodity disappears": drop_a_commodity,
    "the feed freezes (month repeats)": freeze_the_month,
    "one price entered 12x too high": add_a_wild_outlier,
}


def main(raw_path=RAW, out_path=OUT, seed=0):
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(raw_path, low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"])

    base_rows, base_cov, month = _shortlist(raw)
    if base_rows is None:
        raise SystemExit("No month to score: the file ends on a month that "
                         "already has an outcome.")
    period = pd.Period(month, freq="M")
    print(f"Baseline: scoring {month}, top pick "
          f"{base_rows[0]['state']} {base_rows[0]['commodity']} "
          f"at {base_rows[0]['score']:.3f}")
    print(f"  coverage check says partial={bool((base_cov or {}).get('partial'))}\n")

    results = {"month_scored": month,
               "baseline": {"top5": [[r["state"], r["commodity"], r["score"]]
                                     for r in base_rows[:5]],
                            "coverage": base_cov},
               "faults": {}}

    for name, fault in FAULTS.items():
        print(f"  {name} ...", end=" ", flush=True)
        try:
            rows, cov, _ = _shortlist(fault(raw, period, rng))
            if rows is None:
                raise RuntimeError("produced no shortlist at all")
            kept = _overlap(base_rows, rows)
            top = rows[0]
            results["faults"][name] = {
                "survived": True,
                "kept_of_top5": kept,
                "new_top_pick": f"{top['state']} {top['commodity']}",
                "new_top_score": top["score"],
                "score_change": round(top["score"] - base_rows[0]["score"], 3),
                "flagged_partial": bool((cov or {}).get("partial")),
                "coverage": cov,
            }
            print(f"kept {kept}/5, top now {top['state']} {top['commodity']} "
                  f"({top['score']:.3f}), flagged={results['faults'][name]['flagged_partial']}")
        except Exception as exc:
            results["faults"][name] = {"survived": False, "error": repr(exc)}
            print(f"CRASHED: {exc!r}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=1, default=str))

    print(f"\n{'fault injected':38s} {'ran':>4s} {'kept':>5s} {'top moved':>10s} {'flagged':>8s}")
    print("-" * 70)
    for name, r in results["faults"].items():
        if not r["survived"]:
            print(f"{name:38s} {'no':>4s} {'-':>5s} {'-':>10s} {'-':>8s}")
            continue
        print(f"{name:38s} {'yes':>4s} {str(r['kept_of_top5'])+'/5':>5s} "
              f"{r['score_change']:>+10.3f} {str(r['flagged_partial']):>8s}")
    print("-" * 70)
    print("'kept' is how many of the five original picks survive. 'flagged' is\n"
          "whether the system noticed its input was degraded. A fault that is\n"
          "survived but not flagged is the dangerous kind: the answer changes\n"
          "and nothing says so.")
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
