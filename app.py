"""Monthly food price spike warning dashboard.

Run locally:
    python -m src.make_warnings      # builds data/processed/dashboard_warnings.json
    streamlit run app.py

Deploy free on Streamlit Community Cloud by pointing it at this repository.
A static single-file version of the same dashboard is in `dashboard.html`.

The dashboard presents a *ranked shortlist*, not a threshold alarm. That is a
deliberate design decision, and it is the central lesson of the evaluation: a
threshold tuned to catch 85% of spikes flags 58% of all state-commodity pairs
every month, which is not a warning anyone can act on. A list of the five
highest-ranked risks is something a procurement officer can actually work
through.

Each warning carries the factors that produced it, because a ranked list with
no reasons is a black box an official has no cause to trust.

One number here is easy to get wrong, so it is computed in
`src/make_warnings.py` and carried in the file rather than recomputed from the
warnings: the **background rate**. It is the share of *every* state-and-food
pair scored that went on to spike. Computing it from the warnings themselves
would measure the shortlist against the shortlist, because the warnings are by
construction the rows the model was most worried about, and the resulting
"lift" would sit near 1.0 no matter how well the system worked.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path("data/processed/dashboard_warnings.json")

# Ask the deployed service first, fall back to the local file. Streamlit runs
# server side, so unlike the browser dashboard it is not subject to the rule
# against an HTTPS page calling an HTTP service, and it can read the live
# endpoint directly. Point it somewhere else with SPIKE_API.
API = os.environ.get("SPIKE_API", "http://56.228.75.61").rstrip("/")
API_TIMEOUT = 4          # seconds; a slow answer is worse than the local file

st.set_page_config(page_title="Food Price Spike Warnings",
                   page_icon="chart", layout="wide")


def _frame(rows, meta):
    df = pd.DataFrame(rows)
    if df.empty:
        return df, meta
    df["month"] = pd.to_datetime(df["month"])
    if "actual" not in df.columns:
        df["actual"] = pd.NA
    return df.sort_values(["month", "score"], ascending=[False, False]), meta


@st.cache_data(ttl=300)
def load():
    """Return (warnings frame, meta dict, where it came from).

    The live service is the source of truth, because the local file is only
    as fresh as the last time somebody ran the batch job on this machine. If
    the service cannot be reached the file still works, which is what matters
    on a laptop with no network. Either way the page says which it used: a
    dashboard that falls back silently is worse than one that fails loudly.
    """
    try:
        with urllib.request.urlopen(API + "/warnings?k=20", timeout=API_TIMEOUT) as r:
            if r.status != 200:
                raise urllib.error.URLError(f"service answered {r.status}")
        # /warnings gives one month; the store behind it gives every month,
        # which is what the history chart needs.
        with urllib.request.urlopen(API + "/feed", timeout=API_TIMEOUT) as r:
            blob = json.loads(r.read().decode("utf-8"))
        rows = [{"state": w["s"], "commodity": w["c"], "month": w["m"],
                 "score": w["p"], "reasons": w["r"], "actual": w["a"],
                 "status": w.get("status"), "predicts": w.get("predicts"),
                 "pairs_this_month": w.get("pairs_this_month")}
                for w in blob["warnings"]]
        df, meta = _frame(rows, blob.get("meta", {}))
        return df, meta, f"live from {API}"
    except Exception as exc:                      # network, timeout, bad JSON
        if not DATA.exists():
            return pd.DataFrame(), {}, f"no data: {API} unreachable ({exc}) and no local file"
        blob = json.loads(DATA.read_text())
        if isinstance(blob, list):                # older export: warnings only
            rows, meta = blob, {}
        else:
            rows, meta = blob["warnings"], blob.get("meta", {})
        df, meta = _frame(rows, meta)
        return df, meta, f"local file, {API} unreachable ({exc})"


df, meta, source = load()

st.title("Food price spike early warning")

live = source.startswith("live")
pend = (meta.get("live") or {}).get("predicts")
if live:
    st.success(f"Live from the prediction service"
               + (f". Currently warning for {pend}." if pend else "."))
else:
    st.warning(f"Showing data from the {source}.")
st.caption(
    "Each month, for every state and staple, the system estimates the chance "
    "that the price will rise more than 15% above its recent trend next month. "
    "It reports a ranked shortlist rather than an alarm."
)

if df.empty:
    st.warning(
        "No warning data found. Build it with `python -m src.make_warnings`, "
        "which writes `data/processed/dashboard_warnings.json`."
    )
    st.stop()

months = sorted(df["month"].unique(), reverse=True)
col_a, col_b = st.columns([2, 1])
with col_a:
    chosen = st.selectbox(
        "Month", months,
        format_func=lambda m: pd.Timestamp(m).strftime("%B %Y"),
    )
with col_b:
    k = st.slider("Warnings to show", 3, 8, 5,
                  help="A shorter list is more often right. A longer list "
                       "catches more spikes. That is the whole trade-off.")

current = df[df["month"] == chosen].nlargest(k, "score")

# ---- headline numbers, measured rather than recomputed from the shortlist ---
base = meta.get("base_rate")
prec = (meta.get("precision_at_k") or {}).get(str(k))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Months tracked", meta.get("months", df["month"].nunique()),
          help="Retrained every month on history only.")
m2.metric(f"Correct in top {k}", f"{prec:.0%}" if prec is not None else "—",
          help="Share of the shortlist that was followed by a real rise.")
m3.metric("Background rate", f"{base:.0%}" if base else "—",
          help="Share of all pairs scored that rose. This is what you would "
               "get by picking a food at random.")
m4.metric("Lift", f"{prec / base:.1f}x" if (prec and base) else "—",
          help="How much better than picking at random.")

st.subheader(f"Top {k} risks for {pd.Timestamp(chosen).strftime('%B %Y')}")

for _, row in current.iterrows():
    left, right = st.columns([1, 4])
    with left:
        st.metric(f"{row['commodity']}", f"{row['score']:.0%}", row["state"])
    with right:
        st.write("**Why:** " + "; ".join(row["reasons"]))
        if "actual" in row and pd.notna(row["actual"]):
            if row["actual"] == 1:
                st.success("A sharp rise did follow.")
            else:
                st.info("No sharp rise followed.")
    st.divider()

# ---- historical accuracy, stated plainly ------------------------------------
# Only months whose outcome is known belong here. The month being warned about
# has no answer yet, and `sum()` over missing values returns 0, so including it
# would draw the live prediction as a month the system got completely wrong.
scored = df[df["actual"].notna()] if "actual" in df.columns else df.iloc[0:0]
if not scored.empty:
    st.subheader("How the shortlist has performed")
    st.caption(
        "Months whose outcome is known. The month currently being warned "
        "about is not here: nobody knows yet whether it was right."
    )
    hist = (
        scored.groupby("month")
        .apply(lambda g: pd.Series({
            "shown": min(k, len(g)),
            "correct": int(g.nlargest(min(k, len(g)), "score")["actual"].sum()),
        }), include_groups=False)
        .reset_index()
    )
    hist["hit_rate"] = hist["correct"] / hist["shown"]

    chart = hist.set_index(hist["month"].dt.strftime("%Y-%m"))[["hit_rate"]]
    st.bar_chart(chart, height=220)

    if base:
        st.caption(
            f"The dashed comparison is the background rate of {base:.0%}: a month "
            "above that is a month the shortlist beat guessing. The system is "
            "designed to narrow attention, not to be certain."
        )

with st.expander("How to read these numbers"):
    st.markdown(
        f"""
A **spike** means the price next month exceeds its trailing three-month mean
by more than 15%.

The score is a calibrated probability: a score of 0.60 means that, among all
the times the system has said 0.60, roughly 60% did go on to spike.

**Why a shortlist and not an alarm.** Tuning a threshold to catch 85% of
spikes causes the system to flag 58% of every state-and-commodity pair each
month. That is not guidance, it is the whole market. Ranking and showing the
top few is less complete and far more usable.

**What the lift means.** Of all the pairs scored, {base:.0%} rose. Of the ones
this shortlist named, {prec:.0%} rose. Looking where it points is worth roughly
{prec / base:.1f} times a guess.

**What this cannot do.** It does not forecast how large the increase will be,
and it cannot anticipate a policy shock such as a subsidy removal or a
currency float, because those causes are not present in price history.
        """ if (base and prec) else """
A **spike** means the price next month exceeds its trailing three-month mean
by more than 15%. The score is a calibrated probability.
        """
    )
