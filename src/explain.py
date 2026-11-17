"""SHAP attribution: why the model issued a given warning.

A ranked warning list is only actionable if the person receiving it can see
what drove each entry. "Rice in Borno, probability 0.71" invites the question
"because of what?", and without an answer the list is a black box an official
has no reason to trust.

SHAP values answer that per prediction. Each feature gets a signed
contribution to that specific score, and the contributions sum to the
difference between the score and the dataset average. For the linear model
these are exact, because a linear model's contribution per feature is simply
coefficient times standardised value.

Explainers are fitted on training data only, in keeping with every other
fitted object in this pipeline.
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
UP = "#b85042"      # pushes the score up, toward a warning
DOWN = "#4a7b9d"    # pushes the score down


def _out(figdir, name):
    Path(figdir).mkdir(parents=True, exist_ok=True)
    return str(Path(figdir) / name)


def compute_shap(pipeline, train, explain_rows):
    """SHAP values for `explain_rows` under a fitted pipeline.

    The pipeline's preprocessing stage is applied first, then the explainer
    runs on the transformed matrix, so attributions are expressed in terms of
    the model's actual input features rather than raw panel columns.
    """
    import shap

    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_train = prep.transform(train)
    X_rows = prep.transform(explain_rows)
    names = list(prep.get_feature_names_out())

    if hasattr(model, "coef_"):
        # Exact for a linear model; the background is the training mean.
        explainer = shap.LinearExplainer(model, X_train)
        values = explainer.shap_values(X_rows)
        base = float(explainer.expected_value)
    else:
        background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        values = explainer.shap_values(X_rows, silent=True)
        base = float(explainer.expected_value)

    return np.asarray(values), np.asarray(X_rows), names, base


def fig8_1_shap_summary(values, X, names, figdir="figures", top_n=12):
    """Which features drive warnings overall, and in which direction.

    Mean absolute SHAP ranks features by how much they move scores. The
    signed panel beside it shows whether a high value of that feature pushes
    toward a warning or away from one, which mean-absolute hides.
    """
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[-top_n:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))

    ax1.barh([names[i] for i in order], mean_abs[order], color=MUTED, height=0.66)
    ax1.set_xlabel("Mean |SHAP value|")
    ax1.set_title("How much each feature moves the score")
    ax1.grid(axis="x", alpha=0.45)
    ax1.tick_params(labelsize=8.5)

    for row, i in enumerate(order):
        v = values[:, i]
        x = X[:, i]
        # Colour by whether the feature value is above or below its median.
        high = x > np.median(x)
        ax2.scatter(v[high], np.full(high.sum(), row) + np.random.uniform(-0.16, 0.16, high.sum()),
                    s=5, alpha=0.35, color=UP, edgecolors="none")
        ax2.scatter(v[~high], np.full((~high).sum(), row) + np.random.uniform(-0.16, 0.16, (~high).sum()),
                    s=5, alpha=0.35, color=DOWN, edgecolors="none")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([names[i] for i in order], fontsize=8.5)
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.set_xlabel("SHAP value (right = pushes toward a warning)")
    ax2.set_title("Direction of effect")
    ax2.grid(axis="x", alpha=0.45)

    handles = [plt.Line2D([], [], marker="o", ls="", color=UP, label="feature value high"),
               plt.Line2D([], [], marker="o", ls="", color=DOWN, label="feature value low")]
    ax2.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Figure 8.1  What drives a warning, across the test period",
                 y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = _out(figdir, "fig8_1_shap_summary.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def fig8_2_shap_waterfall(values, X, names, base, rows_meta, scores,
                          figdir="figures", n=3, top_n=8):
    """Per-warning explanation for the highest-ranked alerts.

    This is the figure that turns a ranked list into something an official can
    act on: for each warning, the handful of factors that produced it, with
    the direction and size of each.
    """
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6))
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, range(n)):
        v = values[idx]
        order = np.argsort(np.abs(v))[-top_n:]
        labels = [names[i] for i in order]
        vals = v[order]
        colours = [UP if x > 0 else DOWN for x in vals]

        ax.barh(range(len(order)), vals, color=colours, height=0.64)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color=INK, lw=1.0)
        meta = rows_meta.iloc[idx]
        ax.set_title(f"{meta['commodity']}, {meta['state']}\n"
                     f"{pd.to_datetime(meta['month']).strftime('%b %Y')}  "
                     f"(score {scores[idx]:.2f})", fontsize=9.5)
        ax.set_xlabel("Contribution to the score", fontsize=8.5)
        ax.grid(axis="x", alpha=0.45)

    fig.suptitle("Figure 8.2  Why these specific warnings were issued  "
                 "(red raises the score, blue lowers it)",
                 y=1.04, fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = _out(figdir, "fig8_2_shap_waterfall.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def warning_reasons(values, names, rows_meta, scores, top_n=3):
    """Plain-language reason strings for a ranked warning list.

    Turns the largest positive contributions into a short phrase, so the
    dashboard can print "rice, Borno: prices already climbing, unusually
    volatile" rather than a vector of numbers.
    """
    phrases = {
        "mom_1m": "price rose last month",
        "mom_3m": "price climbing over three months",
        "yoy": "well above this time last year",
        "dev_from_6m": "above its six-month trend",
        "dev_from_12m": "above its twelve-month trend",
        "volatility_6m": "unusually volatile recently",
        "mom_same_month_last_year": "this month was bad last year",
        "neighbour_mom_1m": "neighbouring states rising",
        "fuel_mom_1m": "fuel cost rose last month",
        "fuel_mom_3m": "fuel cost climbing",
        "gaps_last_6m": "patchy recent reporting",
        "price_was_interpolated": "a recent month was estimated",
        "month_sin": "seasonal timing",
        "month_cos": "seasonal timing",
    }
    # Identity one-hots (state, food type) are legitimate model inputs but are
    # not reasons an official can act on: "because it is Borno" explains
    # nothing they did not already know. They stay in the figures and out of
    # the written reason strings.
    identity = ("state_", "food_type_", "priceflag_")

    out = []
    for i in range(len(values)):
        v = values[i]
        order = np.argsort(v)[::-1]
        reasons = []
        for j in order:
            if v[j] <= 0 or len(reasons) >= top_n:
                continue
            name = names[j]
            key = name.split("__")[-1]
            if key.startswith(identity):
                continue
            reasons.append(phrases.get(key, key.replace("_", " ")))
        meta = rows_meta.iloc[i]
        out.append({
            "state": meta["state"],
            "commodity": meta["commodity"],
            "month": pd.to_datetime(meta["month"]).strftime("%Y-%m"),
            "score": round(float(scores[i]), 3),
            "reasons": reasons or ["no single dominant factor"],
        })
    return out
