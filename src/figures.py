"""Generate the eight Phase 2 diagnostic figures at 300 DPI.

Each function returns the path it wrote. Colour choices follow one rule per
job: a single hue light-to-dark for magnitude (missingness, PCA), a two-hue
diverging scale with a neutral midpoint for correlation (polarity), and a
fixed two-colour assignment for before/after comparisons. No rainbow scales,
and identity is never carried by colour alone.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .pipeline_builder import NUMERIC_FEATURES, RATIO_FEATURES, TARGET

DPI = 300
INK = "#1f2328"
MUTED = "#6b7280"
PRIMARY = "#b85042"      # before / series 1
SECONDARY = "#4a7b9d"    # after / series 2
SEQ_CMAP = "Blues"       # magnitude
DIV_CMAP = "RdBu_r"      # polarity, neutral midpoint

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK,
    "axes.edgecolor": "#d4d4d8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.6,
})


def _out(figdir, name):
    Path(figdir).mkdir(parents=True, exist_ok=True)
    return str(Path(figdir) / name)


# ---------------------------------------------------------------- 1.1
def fig1_1_missingness_heatmap(panel, figdir="figures"):
    """Where the gap months are, by series and month."""
    m = panel.pivot_table(
        index=["state", "commodity"], columns="month", values="is_gap", aggfunc="max"
    )
    m = m.loc[m.index.sortlevel(0)[0]]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(m.values, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1, interpolation="nearest")

    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([f"{s} · {c}" for s, c in m.index], fontsize=5.5)
    cols = pd.to_datetime(m.columns)
    ticks = [i for i, d in enumerate(cols) if d.month == 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([cols[i].year for i in ticks])
    ax.set_xlabel("Month")
    ax.set_title("Figure 1.1  Missingness pattern: gap months by state-commodity series")

    gap_rate = panel["is_gap"].mean()
    ax.text(
        0.995, 1.012,
        f"shaded = no observation   ·   {gap_rate:.1%} of cells are gaps",
        transform=ax.transAxes, ha="right", fontsize=8, color=MUTED,
    )
    p = _out(figdir, "fig1_1_missingness_heatmap.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 1.2
def fig1_2_target_distribution(panel, splits, figdir="figures"):
    """Class balance overall and per temporal split."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    y = panel[TARGET].dropna()
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    bars = ax1.bar(["No spike", "Spike"], counts, color=[MUTED, PRIMARY], width=0.55)
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c/sum(counts):.1%})",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Labelled rows")
    ax1.set_ylim(0, max(counts) * 1.22)
    ax1.set_title("Overall class balance")
    ax1.text(0.5, -0.17, f"imbalance {counts[0]/counts[1]:.1f} : 1",
             transform=ax1.transAxes, ha="center", fontsize=9, color=MUTED)

    names = list(splits.keys())
    rates = [splits[k][TARGET].mean() * 100 for k in names]
    ns = [len(splits[k]) for k in names]
    bars = ax2.bar(names, rates, color=SECONDARY, width=0.55)
    for b, r, n in zip(bars, rates, ns):
        ax2.text(b.get_x() + b.get_width() / 2, r, f"{r:.1f}%\nn={n:,}",
                 ha="center", va="bottom", fontsize=9)
    ax2.axhline(y.mean() * 100, color=PRIMARY, lw=1.6, ls="--",
                label=f"overall {y.mean()*100:.1f}%")
    ax2.set_ylabel("Positive rate (%)")
    ax2.set_ylim(0, max(rates) * 1.3)
    ax2.set_title("Positive rate by temporal split")
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 1.2  Target distribution: spike in month t+1", y=1.02,
                 fontsize=11, fontweight="bold")
    p = _out(figdir, "fig1_2_target_distribution.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 2.1
def fig2_1_outlier_boxplots(train, capper, figdir="figures"):
    """Ratio features before and after percentile capping.

    Price *levels* are deliberately absent from this figure. An extreme
    upward price movement is the target event, so capping prices would delete
    positive cases from the training data. Only engineered ratios are capped,
    because an absurd ratio is an artefact of construction.
    """
    cols = [c for c in RATIO_FEATURES if c in train.columns]
    before = train[cols]
    after = capper.transform(train[cols])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, data, title, colour in [
        (axes[0], before, "Before capping", PRIMARY),
        (axes[1], after, "After capping (1st-99th pct, train-fitted)", SECONDARY),
    ]:
        bp = ax.boxplot(
            [data[c].dropna() for c in cols], tick_labels=cols, vert=True,
            patch_artist=True, widths=0.55,
            flierprops=dict(marker="o", markersize=2, alpha=0.35,
                            markerfacecolor=colour, markeredgecolor="none"),
            medianprops=dict(color=INK, lw=1.2),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=1.2)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=70, labelsize=7.5)
        ax.axhline(0, color=MUTED, lw=0.7, ls=":")
        ax.grid(axis="y", alpha=0.5)
    axes[0].set_ylabel("Ratio value")

    fig.suptitle(
        "Figure 2.1  Outlier treatment: engineered ratio features only "
        "(price levels are never capped)",
        y=1.03, fontsize=11, fontweight="bold")
    p = _out(figdir, "fig2_1_outlier_boxplots.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 3.1
def fig3_1_correlation_heatmap(train, figdir="figures", thresh=0.85):
    """Upper-triangle Spearman correlation, collinear pairs annotated."""
    cols = [c for c in NUMERIC_FEATURES if train[c].notna().sum() > 50]
    corr = train[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    shown = np.where(mask, corr.values, np.nan)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(shown, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=70, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7.5)

    flagged = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.values[i, j]
            if abs(r) >= thresh:
                flagged.append((cols[i], cols[j], r))
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor=INK, lw=2))
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=6.5,
                        fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.72, label="Spearman rho")
    ax.set_title(f"Figure 3.1  Feature correlation (upper triangle); "
                 f"{len(flagged)} pair(s) at |rho| >= {thresh} boxed")
    p = _out(figdir, "fig3_1_correlation_matrix.png")
    fig.savefig(p); plt.close(fig)
    return p, flagged


# ---------------------------------------------------------------- 3.2
def fig3_2_collinear_scatter(train, flagged, figdir="figures", corr_fallback=4):
    """Scatter with fitted trendline for the most collinear pairs."""
    pairs = flagged
    if not pairs:
        cols = [c for c in NUMERIC_FEATURES if train[c].notna().sum() > 50]
        c = train[cols].corr(method="spearman").abs()
        cand = [(cols[i], cols[j], c.values[i, j])
                for i in range(len(cols)) for j in range(i + 1, len(cols))]
        pairs = sorted(cand, key=lambda t: -t[2])[:corr_fallback]

    pairs = pairs[:4]
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    axes = np.atleast_1d(axes)
    for ax, (a, b, r) in zip(axes, pairs):
        d = train[[a, b]].dropna()
        ax.scatter(d[a], d[b], s=7, alpha=0.3, color=SECONDARY, edgecolors="none")
        if len(d) > 2:
            sl, ic = np.polyfit(d[a], d[b], 1)
            xs = np.linspace(d[a].min(), d[a].max(), 50)
            ax.plot(xs, sl * xs + ic, color=PRIMARY, lw=2)
        ax.set_xlabel(a, fontsize=8); ax.set_ylabel(b, fontsize=8)
        ax.set_title(f"rho = {r:.2f}", fontsize=9)
        ax.grid(alpha=0.4)
    fig.suptitle("Figure 3.2  Most strongly associated feature pairs, with fitted trendline",
                 y=1.06, fontsize=11, fontweight="bold")
    p = _out(figdir, "fig3_2_collinear_scatterplots.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 3.3
def fig3_3_feature_importance(train, figdir="figures"):
    """Point-biserial correlation and mutual information against the target."""
    cols = [c for c in NUMERIC_FEATURES if train[c].notna().sum() > 50]
    y = train[TARGET].astype(int)

    pb = {}
    for c in cols:
        d = train[[c]].join(y).dropna()
        if d[c].nunique() > 1:
            pb[c] = abs(stats.pointbiserialr(d[TARGET], d[c])[0])
    pb = pd.Series(pb).sort_values()

    Xi = SimpleImputer(strategy="median").fit_transform(train[cols])
    mi = pd.Series(mutual_info_classif(Xi, y, random_state=42), index=cols)
    mi = mi.reindex(pb.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    ax1.barh(pb.index, pb.values, color=PRIMARY, height=0.65)
    ax1.set_xlabel("|point-biserial r|"); ax1.set_title("Linear association")
    ax1.grid(axis="x", alpha=0.5)
    ax2.barh(mi.index, mi.values, color=SECONDARY, height=0.65)
    ax2.set_xlabel("Mutual information"); ax2.set_title("Any association (model-agnostic)")
    ax2.grid(axis="x", alpha=0.5)
    ax1.tick_params(labelsize=8)

    fig.suptitle("Figure 3.3  Feature-to-target association, training split only",
                 y=1.01, fontsize=11, fontweight="bold")
    p = _out(figdir, "fig3_3_target_importance.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 4.1
def fig4_1_transformation_grid(train, col="price", figdir="figures"):
    """Raw, log, min-max and standardised distributions of one feature."""
    x = train[col].dropna()
    x = x[x > 0]
    v = x.values.reshape(-1, 1)

    panels = [
        ("Raw", x.values, PRIMARY),
        ("Log", np.log1p(x.values), SECONDARY),
        ("Min-Max [0,1]", MinMaxScaler().fit_transform(v).ravel(), SECONDARY),
        ("Standardised (mu=0, sigma=1)", StandardScaler().fit_transform(v).ravel(), SECONDARY),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (title, data, colour) in zip(axes.ravel(), panels):
        ax.hist(data, bins=45, color=colour, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{title}   (skew {stats.skew(data):.2f})", fontsize=10)
        ax.grid(axis="y", alpha=0.45)
    fig.suptitle(f"Figure 4.1  Transformation stages for '{col}'",
                 y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = _out(figdir, "fig4_1_transformation_grid.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 5.1
def fig5_1_pca_scree(train, figdir="figures"):
    """Scree and cumulative variance, with 90% and 95% cutoffs."""
    cols = [c for c in NUMERIC_FEATURES if train[c].notna().sum() > 50]
    X = SimpleImputer(strategy="median").fit_transform(train[cols])
    X = StandardScaler().fit_transform(X)
    pca = PCA().fit(X)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)

    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(1, len(ev) + 1)
    ax.bar(idx, ev * 100, color=SECONDARY, alpha=0.8, width=0.65, label="Individual")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Variance explained (%)")
    ax.grid(axis="y", alpha=0.45)

    ax2 = ax.twinx()
    ax2.plot(idx, cum * 100, color=PRIMARY, lw=2, marker="o", markersize=4,
             label="Cumulative")
    ax2.set_ylabel("Cumulative (%)"); ax2.set_ylim(0, 105); ax2.grid(False)
    for lvl, ls in [(90, "--"), (95, ":")]:
        ax2.axhline(lvl, color=MUTED, lw=1.1, ls=ls)
        k = int(np.searchsorted(cum * 100, lvl) + 1)
        ax2.text(len(ev) * 0.02, lvl + 1.5, f"{lvl}% at {k} components",
                 fontsize=8, color=MUTED)

    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    ax.set_title("Figure 5.1  PCA scree and cumulative variance (diagnostic only)")
    p = _out(figdir, "fig5_1_pca_scree_plot.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 6.1
def fig6_1_calibration(y_true, p_raw, p_cal, figdir="figures", n_bins=10):
    """Reliability curve before and after post-hoc calibration.

    Weighted loss inflates predicted scores, so an uncalibrated curve sits
    below the diagonal: the model says 0.6 where the observed rate is 0.3.
    Calibration is monotonic, so ranking and precision@k are unchanged; what
    changes is whether a displayed probability means anything.
    """
    from sklearn.calibration import calibration_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--", label="perfect")
    for p, label, colour in [(p_raw, "Raw (weighted loss)", PRIMARY),
                             (p_cal, "Calibrated", SECONDARY)]:
        obs, pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax1.plot(pred, obs, marker="o", markersize=5, lw=2, color=colour, label=label)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed spike rate")
    ax1.set_title("Reliability curve")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.45)

    ax2.hist(p_raw, bins=28, alpha=0.6, color=PRIMARY, label="Raw", edgecolor="white", linewidth=0.4)
    ax2.hist(p_cal, bins=28, alpha=0.6, color=SECONDARY, label="Calibrated", edgecolor="white", linewidth=0.4)
    ax2.axvline(np.mean(y_true), color=INK, lw=1.6, ls="--",
                label=f"base rate {np.mean(y_true):.3f}")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Score distribution")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(axis="y", alpha=0.45)

    fig.suptitle("Figure 6.1  Probability calibration under weighted loss",
                 y=1.02, fontsize=11, fontweight="bold")
    p = _out(figdir, "fig6_1_calibration.png")
    fig.savefig(p); plt.close(fig)
    return p
