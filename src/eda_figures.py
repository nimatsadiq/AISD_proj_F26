"""Descriptive figures for the README: what the data actually contains.

These answer "what is in this dataset?" rather than "is the pipeline sound?".
The eight numbered diagnostic figures in `figures.py` serve the second
question; these serve the first, so that someone opening the repository
understands the data without running anything.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
INK = "#1f2328"
MUTED = "#6b7280"
PRIMARY = "#b85042"
SECONDARY = "#4a7b9d"
ACCENT = "#7d9b76"

plt.rcParams.update({
    "savefig.dpi": DPI, "savefig.bbox": "tight", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelcolor": INK,
    "axes.edgecolor": "#d4d4d8", "axes.spines.top": False,
    "axes.spines.right": False, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#e5e7eb", "grid.linewidth": 0.6,
})


def _out(figdir, name):
    Path(figdir).mkdir(parents=True, exist_ok=True)
    return str(Path(figdir) / name)


def eda_rows_per_state(raw, figdir="figures"):
    """How many records each state contributes, and when its coverage ends."""
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"])
    g = (
        d.groupby("admin1")
        .agg(rows=("price", "size"), last=("date", "max"), markets=("market", "nunique"))
        .sort_values("rows")
    )
    live = g["last"] >= "2025-01-01"
    colours = [PRIMARY if v else MUTED for v in live]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(g.index, g["rows"], color=colours, height=0.68)
    for b, (state, row) in zip(bars, g.iterrows()):
        ax.text(b.get_width() + 400, b.get_y() + b.get_height() / 2,
                f"{int(row['rows']):,}  ·  ends {row['last'].strftime('%b %Y')}"
                f"  ·  {int(row['markets'])} mkts",
                va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, g["rows"].max() * 1.52)
    ax.set_xlabel("Price records")
    ax.set_title("Records per state, and where coverage stops")
    ax.grid(axis="x", alpha=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PRIMARY),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["Continues into 2025 or later (in scope)",
                        "Coverage ends earlier (out of scope)"],
              frameon=False, fontsize=8, loc="lower right")
    p = _out(figdir, "eda_1_rows_per_state.png")
    fig.savefig(p); plt.close(fig)
    return p


def eda_records_per_year(raw, figdir="figures"):
    """Records per year: the usable history is much shorter than 2002 to 2026."""
    d = raw.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    counts = d.groupby("year").size()
    colours = [MUTED if y < 2016 else SECONDARY for y in counts.index]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(counts.index, counts.values, color=colours, width=0.72)
    ax.axvline(2015.5, color=PRIMARY, lw=1.6, ls="--")
    ax.text(2015.75, counts.max() * 0.9, "modelling period\nstarts 2016",
            fontsize=8.5, color=PRIMARY)
    ax.set_xlabel("Year"); ax.set_ylabel("Price records")
    ax.set_title("Records per year: the file spans 2002 to 2026, the dense history does not")
    ax.grid(axis="y", alpha=0.5)
    p = _out(figdir, "eda_2_records_per_year.png")
    fig.savefig(p); plt.close(fig)
    return p


def eda_spikes_by_year(panel, target="spike_next_month", figdir="figures"):
    """Spike rate by year, with the count behind each rate."""
    d = panel.dropna(subset=[target]).copy()
    d["year"] = pd.to_datetime(d["month"]).dt.year
    g = d.groupby("year")[target].agg(["mean", "sum", "size"])

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(g.index, g["mean"] * 100, color=SECONDARY, width=0.68)
    peak = g["mean"].idxmax()
    for b, (yr, row) in zip(bars, g.iterrows()):
        if yr == peak:
            b.set_color(PRIMARY)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{int(row['sum'])}/{int(row['size'])}", ha="center",
                fontsize=7.5, color=MUTED)
    ax.axhline(d[target].mean() * 100, color=INK, lw=1.2, ls="--",
               label=f"overall {d[target].mean()*100:.1f}%")
    ax.set_ylabel("Spike rate (%)"); ax.set_xlabel("Year")
    ax.set_ylim(0, g["mean"].max() * 100 * 1.25)
    ax.set_title("Spike rate by year (labels show spikes / labelled months)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.5)
    p = _out(figdir, "eda_3_spikes_by_year.png")
    fig.savefig(p); plt.close(fig)
    return p


def eda_spikes_by_month(panel, target="spike_next_month", figdir="figures"):
    """Seasonality: spike rate by calendar month, overall and by food type."""
    d = panel.dropna(subset=[target]).copy()
    d["m"] = pd.to_datetime(d["month"]).dt.month
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    overall = d.groupby("m")[target].mean() * 100
    by_type = d.groupby(["m", "food_type"])[target].mean().unstack() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    bars = ax1.bar(range(1, 13), overall.reindex(range(1, 13)).values,
                   color=SECONDARY, width=0.68)
    bars[int(overall.idxmax()) - 1].set_color(PRIMARY)
    ax1.axhline(d[target].mean() * 100, color=INK, lw=1.2, ls="--")
    ax1.set_xticks(range(1, 13)); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("Spike rate (%)")
    ax1.set_title("All commodities")
    ax1.grid(axis="y", alpha=0.5)

    for col, colour in zip(by_type.columns, [PRIMARY, ACCENT]):
        ax2.plot(by_type.index, by_type[col], marker="o", markersize=5,
                 lw=2, color=colour, label=col)
    ax2.set_xticks(range(1, 13)); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Spike rate (%)")
    ax2.set_title("Perishable vs storable")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.5)

    fig.suptitle("Spike rate by calendar month: the planting and harvest cycle",
                 y=1.03, fontsize=11, fontweight="bold")
    p = _out(figdir, "eda_4_spikes_by_month.png")
    fig.savefig(p); plt.close(fig)
    return p


def eda_price_series(panel, commodities=None, state="Borno", figdir="figures"):
    """Actual price paths for a few staples, with the 2023 break marked."""
    commodities = commodities or ["Rice (local)", "Millet", "Tomatoes", "Oil (palm)"]
    d = panel[(panel["state"] == state) & (panel["commodity"].isin(commodities))].copy()
    d["month"] = pd.to_datetime(d["month"])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.2), sharex=True)
    for ax, c in zip(axes.ravel(), commodities):
        s = d[d["commodity"] == c].sort_values("month")
        ax.plot(s["month"], s["price"], lw=1.6, color=SECONDARY)
        spikes = s[s["spike_next_month"] == 1]
        ax.scatter(spikes["month"], spikes["price"], s=16, color=PRIMARY,
                   zorder=3, label="spike next month")
        ax.axvline(pd.Timestamp("2023-07-01"), color=INK, lw=1.1, ls="--")
        ax.set_title(c, fontsize=10)
        ax.grid(alpha=0.45)
        ax.set_ylabel("NGN")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].text(pd.Timestamp("2023-08-01"),
                    axes[0, 0].get_ylim()[1] * 0.55,
                    "subsidy removal\n& currency float", fontsize=7.5, color=INK)
    fig.suptitle(f"Price paths in {state}: level shift after mid-2023",
                 y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = _out(figdir, "eda_5_price_series.png")
    fig.savefig(p); plt.close(fig)
    return p


def eda_coverage_grid(panel, figdir="figures"):
    """Observed months per state-commodity series, as a compact grid."""
    m = panel.pivot_table(index="commodity", columns="state",
                          values="price", aggfunc="count").fillna(0)
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6.2, 9))
    im = ax.imshow(m.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=7.5)
    for i in range(len(m)):
        for j in range(len(m.columns)):
            v = int(m.values[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=6.5,
                        color="white" if v > m.values.max() * 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.5, label="Observed months")
    ax.set_title("Observed months per series\n(2016-01 to 2026-07, max 127)", fontsize=10)
    p = _out(figdir, "eda_6_coverage_grid.png")
    fig.savefig(p); plt.close(fig)
    return p
